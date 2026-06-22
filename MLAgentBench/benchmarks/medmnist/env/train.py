"""Train a classifier on PneumoniaMNIST, evaluate on ChestMNIST 3-class OOD.

Scientific AI task: train on 2 classes (normal, pneumonia), then detect
whether a test sample is normal, pneumonia, or an unseen OOD class (consolidation).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os, sys

sys.path.insert(0, os.path.dirname(__file__))
from pathlib import Path
from loader import get_datasets, CLASS_NAMES, OOD_CLASS, N_CLASSES


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_classes = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        smooth_labels = one_hot * (1 - self.smoothing) + self.smoothing / n_classes
        log_probs = F.log_softmax(pred, dim=1)
        return -(smooth_labels * log_probs).sum(dim=1).mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, target):
        log_probs = F.log_softmax(pred, dim=1)
        probs = log_probs.exp()
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        pt = (one_hot * probs).sum(dim=1)
        focal_weight = (1 - pt) ** self.gamma
        if self.alpha is not None:
            alpha_t = self.alpha[target]
            focal_weight = focal_weight * alpha_t
        return -(focal_weight * log_probs.gather(1, target.unsqueeze(1)).squeeze()).mean()


try:
    from torchvision.models import densenet121, DenseNet121_Weights
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False


def create_pretrained_densenet(num_classes=3):
    """DenseNet-121 with ImageNet weights, adapted for 28x28 grayscale input."""
    weights = DenseNet121_Weights.IMAGENET1K_V1
    model = densenet121(weights=weights)
    old_conv = model.features.conv0
    model.features.conv0 = nn.Conv2d(
        1, old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=1,
        padding=old_conv.padding,
        bias=False,
    )
    with torch.no_grad():
        model.features.conv0.weight.data = old_conv.weight.data.sum(dim=1, keepdim=True)
    model.features.pool0 = nn.Identity()
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes),
    )
    return model


class SimpleCNN(nn.Module):
    """3-layer CNN for 28x28 chest X-rays. Outputs 3 logits for better OOD detection."""

    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.dropout3 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 3)
        self.num_classes = num_classes

    def forward(self, x):
        x = self.pool(F.leaky_relu(self.bn1(self.conv1(x)), 0.1))
        x = self.pool(F.leaky_relu(self.bn2(self.conv2(x)), 0.1))
        x = self.dropout3(self.pool(F.leaky_relu(self.bn3(self.conv3(x)), 0.1)))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc1(x), 0.1)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for X, y in tqdm(loader, desc="Train", leave=False):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (pred.argmax(1) == y).sum().item()
        total += y.size(0)
    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        pred = model(X).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    return correct / total, np.array(all_preds), np.array(all_labels)


def ood_metrics(all_labels, all_preds, ood_cls=2):
    """Compute OOD detection metrics.

    OOD positive = consolidation class (2). In-distribution = normal(0) + pneumonia(1).
    """
    ood_gt = (all_labels == ood_cls).astype(int)
    ood_pred = (all_preds == ood_cls).astype(int)
    tp = ((ood_pred == 1) & (ood_gt == 1)).sum()
    fp = ((ood_pred == 1) & (ood_gt == 0)).sum()
    fn = ((ood_pred == 0) & (ood_gt == 1)).sum()
    tn = ((ood_pred == 0) & (ood_gt == 0)).sum()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def per_class_accuracy(all_labels, all_preds, num_classes=3):
    accs = {}
    for c in range(num_classes):
        mask = all_labels == c
        if mask.sum() > 0:
            accs[CLASS_NAMES[c]] = (all_preds[mask] == c).sum().item() / mask.sum()
    return accs


def in_distribution_accuracy(all_labels, all_preds, id_classes=(0, 1)):
    """Accuracy on in-distribution classes (normal, pneumonia) only."""
    mask = np.zeros(len(all_labels), dtype=bool)
    for c in id_classes:
        mask |= (all_labels == c)
    if mask.sum() == 0:
        return 0.0
    return (all_preds[mask] == all_labels[mask]).sum().item() / mask.sum()


def save_viz_data(model, loader, device, val_loader=None, out_dir=None):
    """Save PCA embeddings, sample images, and confusion matrix for dashboard."""
    import json, base64, io
    features_list, labels_list, images_list, source_list = [], [], [], []
    hook_handle = model.fc1.register_forward_hook(lambda m, i, o: features_list.append(i[0].detach().cpu()))

    model.eval()

    def extract(loader, source_name):
        with torch.no_grad():
            for X, y in loader:
                X_img = X.cpu()
                X = X.to(device)
                model(X)
                labels_list.extend(y.numpy())
                images_list.append(X_img)
                source_list.extend([source_name] * len(y))

    extract(loader, "ChestMNIST (test)")
    if val_loader is not None:
        extract(val_loader, "PneumoniaMNIST (val)")

    hook_handle.remove()
    all_features = torch.cat(features_list, dim=0).numpy()
    all_labels = np.array(labels_list)
    all_images = torch.cat(images_list, dim=0)
    all_sources = np.array(source_list)

    # PCA on combined features so both datasets share the same space
    mean = all_features.mean(axis=0)
    centered = all_features - mean
    cov = centered.T @ centered / (all_features.shape[0] - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1][:2]
    embedding_2d = centered @ eigvecs[:, idx]

    # Sample images from test set only (first 3 per class)
    sample_images = {}
    for c in range(3):
        mask = (all_labels == c) & (all_sources == "ChestMNIST (test)")
        idxs = np.where(mask)[0][:3]
        sample_images[CLASS_NAMES[c]] = []
        for sidx in idxs:
            img = all_images[sidx].squeeze().numpy()
            buf = io.BytesIO()
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            plt.figure(figsize=(2, 2))
            plt.imshow(img, cmap='gray', vmin=0, vmax=1)
            plt.axis('off')
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            plt.close()
            sample_images[CLASS_NAMES[c]].append(base64.b64encode(buf.getvalue()).decode())

    # OOD predictions on test set only
    ood_preds = []
    with torch.no_grad():
        for X, _ in loader:
            X = X.to(device)
            probs = F.softmax(model(X), dim=1)
            max_probs = probs.max(1).values
            ood = (max_probs < 0.7).cpu().numpy().astype(int) * 2
            ood_preds.extend(ood)
    ood_preds = np.array(ood_preds)

    test_mask = all_sources == "ChestMNIST (test)"
    ood_idx = 0
    embeddings = []
    for i in range(len(all_labels)):
        if test_mask[i]:
            this_ood = int(ood_preds[ood_idx])
            ood_idx += 1
        else:
            this_ood = 0
        label_name = CLASS_NAMES[int(all_labels[i])] if int(all_labels[i]) < 3 else "unknown"
        embeddings.append({
            "x": float(embedding_2d[i, 0]), "y": float(embedding_2d[i, 1]),
            "label": int(all_labels[i]), "label_name": label_name,
            "dataset": str(all_sources[i]),
            "pred_ood": this_ood, "is_ood": bool(this_ood == 2),
        })

    # Per-class accuracy (on test set using OOD predictions)
    test_labels = all_labels[test_mask]
    per_class_acc = {}
    for c in range(3):
        mask = test_labels == c
        if mask.sum() > 0:
            correct = int((ood_preds[mask] == c).sum())
            per_class_acc[CLASS_NAMES[c]] = {
                "total": int(mask.sum()),
                "correct": correct,
                "accuracy": round(correct / int(mask.sum()), 4),
            }

    data = {
        "per_class_accuracy": per_class_acc,
        "class_names": CLASS_NAMES,
        "embeddings": embeddings,
        "sample_images": sample_images,
        "pca_explained_variance": [float(eigvals[-1] / eigvals.sum()), float(eigvals[-2] / eigvals.sum())],
        "total_samples": len(all_labels),
    }
    if out_dir is None:
        # Auto-detect: save to experiments/loop-*/viz/ or ./viz/
        project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
        exp_dir = project_root / "experiments"
        loop_dirs = sorted(exp_dir.glob("loop-*/results.tsv"), reverse=True)
        out_dir = str(loop_dirs[0].parent) if loop_dirs else str(Path.cwd())
    out_path = Path(out_dir) / "viz" / "data.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Viz data saved to {out_path}")


def create_model(model_name="SimpleCNN", num_classes=2, pretrained=False):
    if model_name == "DenseNet121" and HAS_TORCHVISION and pretrained:
        model = create_pretrained_densenet(num_classes=3)
        model.num_classes = num_classes
        return model
    return SimpleCNN(num_classes=num_classes)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds, val_ds, test_ds = get_datasets()
    print(f"Train: {len(train_ds)} (PneumoniaMNIST, 2 classes)")
    print(f"Val:   {len(val_ds)}")
    print(f"Test:  {len(test_ds)} (ChestMNIST 3-class: normal={300}, pneumonia={59}, consolidation={241})")

    batch_size = 64
    epochs = 20
    lr = 1e-3

    from torchvision import transforms
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    ])

    class AugmentedDataset(torch.utils.data.Dataset):
        def __init__(self, base_ds, transform):
            self.base_ds = base_ds
            self.transform = transform
        def __len__(self):
            return len(self.base_ds)
        def __getitem__(self, idx):
            img, label = self.base_ds[idx]
            img_np = img.numpy().squeeze()
            from PIL import Image
            img_pil = Image.fromarray((img_np * 255).astype(np.uint8), mode='L')
            img_pil = self.transform(img_pil)
            img = torch.from_numpy(np.array(img_pil).astype(np.float32) / 255.0).unsqueeze(0)
            return img, label

    train_aug = AugmentedDataset(train_ds, train_transform)
    train_loader = DataLoader(train_aug, batch_size, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_ds, batch_size, shuffle=False, num_workers=1)
    test_loader = DataLoader(test_ds, batch_size, shuffle=False, num_workers=1)

    model = create_model(model_name="SimpleCNN", num_classes=2).to(device)
    criterion = FocalLoss(gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_acc = 0
    for epoch in range(epochs):
        loss, acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_acc, _, _ = evaluate(model, val_loader, device)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "medmnist_model.pth")
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}: train_acc={acc:.4f}, val_acc={val_acc:.4f}")

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load("medmnist_model.pth"))
    test_acc, preds, labels = evaluate(model, test_loader, device)

    # Predictions: model outputs 2 classes. Map OOD detection using ODIN:
    # Temperature scaling + small input perturbation for better ID/OOD separation.
    model.eval()
    temperature = 2.0
    epsilon = 0.001
    ood_preds = []
    for X, _ in test_loader:
        X = X.to(device).requires_grad_(True)
        logits = model(X)
        max_logit = logits.max(1).values
        grad = torch.autograd.grad(max_logit.sum(), X, create_graph=False)[0]
        X_adv = X - epsilon * grad.sign()
        with torch.no_grad():
            logits_adv = model(X_adv)
            probs = F.softmax(logits_adv / temperature, dim=1)
            max_probs = probs.max(1).values
        threshold = 0.3
        ood = (max_probs < threshold).cpu().numpy().astype(int) * 2
        ood_preds.extend(ood)
        X.detach_()

    ood_preds = np.array(ood_preds)
    metrics = ood_metrics(labels, ood_preds)
    per_class = per_class_accuracy(labels, preds)

    print(f"\n{'='*50}")
    print(f"  Test Results (ChestMNIST 3-class)")
    print(f"{'='*50}")
    print(f"  Standard accuracy:       {test_acc:.4f}")
    print(f"  OOD Detection F1:        {metrics['f1']:.4f}")
    print(f"  OOD Precision:           {metrics['precision']:.4f}")
    print(f"  OOD Recall:              {metrics['recall']:.4f}")
    print(f"  Confusion (TP/FP/FN/TN): {metrics['tp']}/{metrics['fp']}/{metrics['fn']}/{metrics['tn']}")
    print()
    for name, acc in per_class.items():
        print(f"  {name:15s} accuracy: {acc:.4f}")
    print(f"{'='*50}")

    save_viz_data(model, test_loader, device)
