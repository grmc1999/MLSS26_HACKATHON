"""Generate visualization data for the dashboard.

Extracts feature embeddings, sample images, and accuracy data
from the trained model and saves them for the dashboard.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json, base64, io, sys, os
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'MLAgentBench', 'benchmarks', 'medmnist', 'env'))
from loader import get_datasets, CLASS_NAMES
from train import SimpleCNN


def extract_features(model, loader, device):
    """Extract penultimate layer features for all samples."""
    labels_list = []
    preds_list = []
    images_list = []
    features_captured = []

    def hook_fn(module, input, output):
        features_captured.append(input[0].detach().cpu())

    hook = model.fc1.register_forward_hook(hook_fn)

    model.eval()
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            logits = model(X)
            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(1)
            labels_list.extend(y.numpy())
            preds_list.extend(preds.cpu().numpy())
            images_list.append(X.cpu())

    hook.remove()
    all_features = torch.cat(features_captured, dim=0).numpy()
    all_labels = np.array(labels_list)
    all_preds = np.array(preds_list)
    all_images = torch.cat(images_list, dim=0)

    return all_features, all_labels, all_preds, all_images


def compute_pca(features, n_components=2):
    """Compute PCA manually (no sklearn dependency)."""
    mean = features.mean(axis=0)
    centered = features - mean
    cov = centered.T @ centered / (features.shape[0] - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1][:n_components]
    components = eigvecs[:, idx]
    return centered @ components, eigvals[idx]


def image_to_base64(img_tensor):
    """Convert a normalized tensor image to base64 PNG."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    img = img_tensor.squeeze().numpy()
    buf = io.BytesIO()
    plt.figure(figsize=(2, 2))
    plt.imshow(img, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds, val_ds, test_ds = get_datasets()
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=1)

    model = SimpleCNN(num_classes=2).to(device)
    state_dict = torch.load("MLAgentBench/benchmarks/medmnist/env/medmnist_model.pth", map_location=device)
    model.load_state_dict(state_dict)

    features, labels, preds, images = extract_features(model, test_loader, device)
    print(f"Extracted {len(features)} features, dim={features.shape[1]}")

    embedding_2d, eigvals = compute_pca(features)
    print(f"PCA done, explained variance: {eigvals[0]/eigvals.sum():.3f}, {eigvals[1]/eigvals.sum():.3f}")

    # Build per-class samples
    class_samples = {}
    for c in range(3):
        mask = labels == c
        idxs = np.where(mask)[0]
        if len(idxs) > 0:
            sample_idxs = idxs[:4]
            class_samples[CLASS_NAMES[c]] = [
                {"embedding": embedding_2d[idx].tolist(),
                 "label": int(labels[idx]),
                 "pred": int(preds[idx]),
                 "correct": bool(labels[idx] == preds[idx])}
                for idx in idxs[:20]
            ]

    # Per-class accuracy
    per_class_acc = {}
    for c in range(3):
        mask = labels == c
        if mask.sum() > 0:
            per_class_acc[CLASS_NAMES[c]] = {
                "total": int(mask.sum()),
                "correct": int((preds[mask] == c).sum()),
                "accuracy": float((preds[mask] == c).sum() / mask.sum()),
            }

    # Confusion matrix
    conf_matrix = np.zeros((3, 3), dtype=int)
    for t, p in zip(labels, preds):
        conf_matrix[t, p] += 1

    # Compute OOD predictions via softmax threshold (matches run_medmnist.py logic)
    model.eval()
    ood_preds = []
    with torch.no_grad():
        for X, _ in test_loader:
            X = X.to(device)
            logits = model(X)
            probs = F.softmax(logits, dim=1)
            max_probs, _ = probs.max(1)
            threshold = 0.7
            ood = (max_probs < threshold).cpu().numpy().astype(int) * 2
            ood_preds.extend(ood)
    ood_preds = np.array(ood_preds)

    # OOD confusion matrix
    ood_conf_matrix = np.zeros((3, 3), dtype=int)
    for t, p in zip(labels, ood_preds):
        ood_conf_matrix[t, p] += 1

    # OOD metrics
    ood_gt = (labels == 2).astype(int)
    tp = ((ood_preds == 2) & (ood_gt == 1)).sum()
    fp = ((ood_preds == 2) & (ood_gt == 0)).sum()
    fn = ((ood_preds == 0) & (ood_gt == 1)).sum()
    ood_precision = float(tp / (tp + fp + 1e-8))
    ood_recall = float(tp / (tp + fn + 1e-8))
    ood_f1 = float(2 * ood_precision * ood_recall / (ood_precision + ood_recall + 1e-8))

    # Sample images (first 3 from each class)
    sample_images = {}
    for c in range(3):
        mask = labels == c
        idxs = np.where(mask)[0][:3]
        sample_images[CLASS_NAMES[c]] = []
        for idx in idxs:
            sample_images[CLASS_NAMES[c]].append(image_to_base64(images[idx]))

    # Embeddings for scatter plot (with OOD labels)
    embeddings_data = []
    for i in range(len(labels)):
        embeddings_data.append({
            "x": float(embedding_2d[i, 0]),
            "y": float(embedding_2d[i, 1]),
            "label": int(labels[i]),
            "label_name": CLASS_NAMES[int(labels[i])],
            "pred_raw": int(preds[i]),
            "pred_ood": int(ood_preds[i]),
            "is_ood": bool(int(ood_preds[i]) == 2),
        })

    viz_data = {
        "per_class_accuracy": per_class_acc,
        "confusion_matrix": conf_matrix.tolist(),
        "ood_confusion_matrix": ood_conf_matrix.tolist(),
        "class_names": CLASS_NAMES,
        "embeddings": embeddings_data,
        "sample_images": sample_images,
        "pca_explained_variance": [float(eigvals[0] / eigvals.sum()), float(eigvals[1] / eigvals.sum())],
        "total_samples": len(labels),
        "test_acc": float((preds == labels).sum() / len(labels)),
        "ood_f1": ood_f1,
        "ood_precision": ood_precision,
        "ood_recall": ood_recall,
        "raw_test_acc": float((preds == labels).sum() / len(labels)),
    }

    loop_dirs = sorted(Path("experiments").glob("loop-*"), reverse=True)
    if loop_dirs:
        out_dir = loop_dirs[0] / "viz"
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / "data.json"
        with open(out_path, "w") as f:
            json.dump(viz_data, f, indent=2)
        print(f"Saved viz data to {out_path}")
    else:
        out_path = "experiments/viz_data.json"
        with open(out_path, "w") as f:
            json.dump(viz_data, f, indent=2)
        print(f"Saved viz data to {out_path}")

    print(f"Embeddings: {len(embeddings_data)} points")
    print(f"Per-class: {per_class_acc}")
    print(f"Confusion matrix:\n{conf_matrix}")


if __name__ == "__main__":
    main()
