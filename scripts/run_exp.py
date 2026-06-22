#!/usr/bin/env python3
"""
CLI to iteratively train and evaluate flu forecasting models.
Usage:
    python scripts/run_exp.py                      # train with defaults
    python scripts/run_exp.py --epochs 200 --lr 1e-3 --hidden-dim 128
    python scripts/run_exp.py --model lstm --hidden-dim 256
    python scripts/run_exp.py --list                # list past runs
"""
import subprocess, sys, os, json, time, argparse
from pathlib import Path

ENV_DIR = Path(__file__).resolve().parent.parent / "env"
LOGS_DIR = Path(__file__).resolve().parent.parent / "experiments"
os.chdir(ENV_DIR)
sys.path.insert(0, str(ENV_DIR))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

LOG_FILE = LOGS_DIR / "runs.jsonl"

def train_model(args):
    from train import create_model, prepare_data, train_epoch, evaluate
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = prepare_data(
        batch_size=args.batch,
        input_steps=5,
        forecast_steps=10,
    )

    model = create_model(
        model_type=args.model,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        forecast_steps=10,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_mae, best_epoch = float("inf"), 0
    start = time.time()
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_mae = evaluate(model, val_loader, device)
        scheduler.step(val_mae)

        if val_mae < best_mae:
            best_mae, best_epoch = val_mae, epoch
            torch.save(model.state_dict(), "best_model.pth")

    elapsed = time.time() - start

    model.load_state_dict(torch.load("best_model.pth"))
    test_mae = evaluate(model, test_loader, device)

    result = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch": args.batch,
        "params": sum(p.numel() for p in model.parameters()),
        "elapsed_s": round(elapsed, 1),
        "best_val_mae": round(best_mae, 6),
        "best_epoch": best_epoch + 1,
        "test_mae": round(test_mae, 6),
    }
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(result) + "\n")

    print(f"\n{'='*50}")
    print(f"  Val MAE:   {result['best_val_mae']:.6f} (epoch {result['best_epoch']})")
    print(f"  Test MAE:  {result['test_mae']:.6f}")
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
    header = f"{'Run #':<6} {'Val MAE':<10} {'Test MAE':<10} {'Params':<10} {'Epochs':<8} {'LR':<10} {'Time':<8} {'Model'}"
    print(header)
    print("-" * 75)
    for i, r in enumerate(runs):
        print(f"{i:<6} {r['best_val_mae']:<10.6f} {r['test_mae']:<10.6f} {r['params']:<10,} {r['epochs']:<8} {r['lr']:<10} {r['elapsed_s']:<8.1f} {r['model']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flu forecasting experiment CLI")
    parser.add_argument("--model", default="lstm", choices=["lstm", "gru", "tcn", "transformer"], help="Model architecture")
    parser.add_argument("--input-dim", type=int, default=1, help="Input feature dimension")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of recurrent/transformer layers")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch", type=int, default=64, help="Batch size")
    parser.add_argument("--list", action="store_true", help="List past experiment runs")
    args = parser.parse_args()

    if args.list:
        list_runs()
    else:
        train_model(args)
