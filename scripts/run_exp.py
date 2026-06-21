#!/usr/bin/env python3
"""
CLI to iteratively train and evaluate contrail detection models.
Usage:
    python scripts/run_exp.py                      # train with defaults
    python scripts/run_exp.py --epochs 100 --lr 1e-3 --batch 4
    python scripts/run_exp.py --model unet --base-ch 64
    python scripts/run_exp.py --list                # list past runs
"""
import subprocess, sys, os, json, time, argparse
from pathlib import Path

ENV_DIR = Path(__file__).resolve().parent.parent / "MLAgentBench" / "benchmarks" / "identify-contrails" / "env"
LOGS_DIR = Path(__file__).resolve().parent.parent / "experiments"
os.chdir(ENV_DIR)
sys.path.insert(0, str(ENV_DIR))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

LOG_FILE = LOGS_DIR / "runs.jsonl"

def train_model(args):
    from train import UNet, ICRGWDataset, dice_score, ce_loss
    import torch, numpy as np, pandas as pd
    from torch.utils.data import DataLoader
    from sklearn.model_selection import train_test_split
    from encode import rle_encode, list_to_string
    from tqdm import tqdm

    data_path = "./train"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_ids = os.listdir(data_path)
    ids_train, ids_valid = train_test_split(image_ids, test_size=0.1, random_state=42)

    train_ds = ICRGWDataset(data_path, ids_train, 2, augment=True)
    valid_ds = ICRGWDataset(data_path, ids_valid, 2)
    train_loader = DataLoader(train_ds, args.batch, shuffle=True, num_workers=1)
    valid_loader = DataLoader(valid_ds, 1, shuffle=False, num_workers=1)

    model = UNet(n_channels=3, n_classes=2, base_ch=args.base_ch).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    best_dice, best_epoch = 0, 0
    start = time.time()
    for epoch in range(args.epochs):
        model.train()
        for X, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = ce_loss(model(X), y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        val_score = 0
        with torch.no_grad():
            for X, y in valid_loader:
                X, y = X.to(device), y.to(device)
                val_score += dice_score(model(X), y).item()
        val_score /= len(valid_loader)
        if val_score > best_dice:
            best_dice, best_epoch = val_score, epoch
            torch.save(model.state_dict(), "u-net.pth")

    elapsed = time.time() - start

    model.load_state_dict(torch.load("u-net.pth"))
    model.eval()
    submission = pd.read_csv("sample_submission.csv", index_col="record_id")
    test_ds = ICRGWDataset("test/", os.listdir("test"), 2)
    with torch.no_grad():
        for idx, (X, _) in enumerate(test_ds):
            X = X.to(device)
            pred = model(X.unsqueeze(0))[:, :, 2:-2, 2:-2]
            pred = torch.argmax(pred, dim=1)[0].detach().cpu().numpy()
            submission.loc[int(test_ds.ids[idx]), "encoded_pixels"] = list_to_string(rle_encode(pred))
    submission.to_csv("submission.csv")

    # Run eval
    trail_path = Path(__file__).resolve().parent.parent / "MLAgentBench" / "benchmarks" / "identify-contrails" / "scripts"
    sys.path.insert(0, str(trail_path))
    from eval import get_score
    test_dice = get_score()

    result = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": "unet",
        "base_ch": args.base_ch,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch": args.batch,
        "params": sum(p.numel() for p in model.parameters()),
        "elapsed_s": round(elapsed, 1),
        "best_val_dice": round(best_dice, 4),
        "best_epoch": best_epoch + 1,
        "test_dice": round(test_dice, 4),
    }
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(result) + "\n")

    print(f"\n{'='*50}")
    print(f"  Val Dice:  {result['best_val_dice']:.4f} (epoch {result['best_epoch']})")
    print(f"  Test Dice: {result['test_dice']:.4f}")
    print(f"  Params:    {result['params']:,}")
    print(f"  Time:      {result['elapsed_s']:.1f}s")
    print(f"{'='*50}")
    return result


def list_runs():
    if not LOG_FILE.exists():
        print("No runs yet.")
        return
    with open(LOG_FILE) as f:
        runs = [json.loads(line) for line in f]
    print(f"{'Run #':<6} {'Val Dice':<10} {'Test Dice':<10} {'Params':<10} {'Epochs':<8} {'LR':<10} {'Time':<8} {'Model'}")
    print("-"*75)
    for i, r in enumerate(runs):
        print(f"{i:<6} {r['best_val_dice']:<10.4f} {r['test_dice']:<10.4f} {r['params']:<10,} {r['epochs']:<8} {r['lr']:<10} {r['elapsed_s']:<8.1f} {r['model']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Contrail detection experiment CLI")
    parser.add_argument("--model", default="unet", choices=["unet"], help="Model architecture")
    parser.add_argument("--base-ch", type=int, default=32, help="Base channels for U-Net")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch", type=int, default=2, help="Batch size")
    parser.add_argument("--list", action="store_true", help="List past experiment runs")
    args = parser.parse_args()

    if args.list:
        list_runs()
    else:
        train_model(args)
