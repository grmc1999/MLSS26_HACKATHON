#!/usr/bin/env python3
"""CLI to train and evaluate MedMNIST chest X-ray OOD task.

Trains on PneumoniaMNIST (2 classes), evaluates on ChestMNIST 3-class subset.

Usage:
    python scripts/run_medmnist.py                     # train with defaults
    python scripts/run_medmnist.py --epochs 30 --lr 1e-3
    python scripts/run_medmnist.py --list              # list past runs
"""
import subprocess, sys, os, json, time, argparse
from pathlib import Path

ENV_DIR = Path(__file__).resolve().parent.parent / "MLAgentBench" / "benchmarks" / "medmnist" / "env"
LOGS_DIR = Path(__file__).resolve().parent.parent / "experiments"
os.chdir(ENV_DIR)
sys.path.insert(0, str(ENV_DIR))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
LOG_FILE = LOGS_DIR / "runs.jsonl"


def train_model(args):
    from loader import get_datasets, CLASS_NAMES, OOD_CLASS
    from train import create_model, FocalLoss, train_epoch, evaluate, ood_metrics, per_class_accuracy, save_viz_data, in_distribution_accuracy
    import torch, torch.nn as nn, numpy as np
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from torchvision import transforms
    from PIL import Image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, val_ds, test_ds = get_datasets()
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
            img_pil = Image.fromarray((img_np * 255).astype(np.uint8), mode='L')
            img_pil = self.transform(img_pil)
            img = torch.from_numpy(np.array(img_pil).astype(np.float32) / 255.0).unsqueeze(0)
            return img, label

    train_loader = DataLoader(AugmentedDataset(train_ds, train_transform), args.batch, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_ds, args.batch, shuffle=False, num_workers=1)
    test_loader = DataLoader(test_ds, args.batch, shuffle=False, num_workers=1)

    model = create_model(model_name=args.model, num_classes=2, pretrained=args.pretrained).to(device)
    criterion = FocalLoss(gamma=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    start = time.time()
    ema_model = None
    ema_decay = 0.999
    best_val_acc, best_epoch = 0, 0
    for epoch in range(args.epochs):
        loss, acc = train_epoch(model, train_loader, optimizer, criterion, device)
        if ema_model is None:
            ema_model = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    ema_model[k] = ema_decay * ema_model[k] + (1 - ema_decay) * v
        val_acc, _, _ = evaluate(model, val_loader, device)
        if val_acc > best_val_acc:
            best_val_acc, best_epoch = val_acc, epoch
            torch.save(model.state_dict(), "medmnist_model.pth")
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{args.epochs}: acc={acc:.4f}, val={val_acc:.4f}")

    elapsed = time.time() - start
    best_sd = torch.load("medmnist_model.pth")
    if ema_model is not None:
        for k in best_sd:
            if k in ema_model:
                best_sd[k] = ema_model[k]
    model.load_state_dict(best_sd)

    # Test evaluation
    import torch.nn.functional as F
    test_acc, preds, labels = evaluate(model, test_loader, device)
    model.eval()
    ood_preds = []
    with torch.no_grad():
        for X, _ in test_loader:
            X = X.to(device)
            logits = model(X)
            probs = F.softmax(logits, dim=1)
            max_probs, _ = probs.max(1)
            ood = (max_probs < args.ood_threshold).cpu().numpy().astype(int) * 2
            ood_preds.extend(ood)
    ood_preds = np.array(ood_preds)
    metrics = ood_metrics(labels, ood_preds)
    per_class = per_class_accuracy(labels, ood_preds, num_classes=3)
    test_acc_id = in_distribution_accuracy(labels, ood_preds, id_classes=(0, 1))

    result = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model + ("_pretrained" if args.pretrained else ""),
        "epochs": args.epochs,
        "lr": args.lr,
        "batch": args.batch,
        "params": sum(p.numel() for p in model.parameters()),
        "elapsed_s": round(elapsed, 1),
        "test_acc": round(test_acc, 4),
        "test_acc_id": round(test_acc_id, 4),
        "val_acc": round(best_val_acc, 4),
        "ood_f1": round(metrics["f1"], 4),
        "ood_precision": round(metrics["precision"], 4),
        "ood_recall": round(metrics["recall"], 4),
    }

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(result) + "\n")

    print(f"\n{'='*50}")
    print(f"  MedMNIST Chest X-ray OOD Results")
    print(f"{'='*50}")
    print(f"  Val Accuracy (PneumoniaMNIST):  {best_val_acc:.4f}")
    print(f"  Test ID Acc (Normal+Pneumonia): {test_acc_id:.4f}")
    print(f"  Test 3-class Accuracy:          {test_acc:.4f}")
    print(f"  OOD F1 Score:                   {metrics['f1']:.4f}")
    print(f"  OOD Precision:                  {metrics['precision']:.4f}")
    print(f"  OOD Recall:                     {metrics['recall']:.4f}")
    for name, acc in per_class.items():
        print(f"  {name:15s} accuracy: {acc:.4f}")
    print(f"  Params:                         {result['params']:,}")
    print(f"  Time:                           {elapsed:.1f}s")
    print(f"{'='*50}")

    try:
        save_viz_data(model, test_loader, device, val_loader=val_loader)
    except Exception as e:
        print(f"Viz data save skipped: {e}")

    return result


def list_runs():
    if not LOG_FILE.exists():
        print("No runs yet.")
        return
    with open(LOG_FILE) as f:
        runs = [json.loads(line) for line in f]
    print(f"{'Run #':<6} {'Test Acc':<10} {'OOD F1':<10} {'Params':<10} {'Time':<8}")
    print("-"*50)
    for i, r in enumerate(runs):
        print(f"{i:<6} {r['test_acc']:<10.4f} {r['ood_f1']:<10.4f} {r['params']:<10,} {r['elapsed_s']:<8.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedMNIST chest X-ray OOD experiment CLI")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch", type=int, default=64, help="Batch size")
    parser.add_argument("--model", type=str, default="SimpleCNN", help="Model: SimpleCNN or DenseNet121")
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet pretrained weights")
    parser.add_argument("--ood-threshold", type=float, default=0.7, help="Softmax threshold for OOD detection")
    parser.add_argument("--list", action="store_true", help="List past runs")
    args = parser.parse_args()

    if args.list:
        list_runs()
    else:
        train_model(args)
