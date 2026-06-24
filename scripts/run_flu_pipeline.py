#!/usr/bin/env python3
"""
Comparison pipeline: adjoint-matched diffusion fine-tuning vs. baselines, across
multiple target countries and fine-tuning-data regimes (the cross-country
distribution-shift proof of the adjoint-matching method).

Usage:
    python scripts/run_flu_pipeline.py
    python scripts/run_flu_pipeline.py --countries FRA,AUS --regimes 1,5 \
        --pretrain-epochs 5 --finetune-epochs 5   # quick smoke test
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = ROOT_DIR / "experiments"
sys.path.insert(0, str(ROOT_DIR))

from env.data import TARGET_COUNTRIES, fit_sir_params, get_finetune_series, get_pretrain_series, load_finetune_data, load_pretrain_data
from env.eval import evaluate, evaluate_full, evaluate_seasonal_naive
from env.train import FINETUNE_VARIANTS, build_run_record, create_model, diffusion_loss_step, get_finetune_loss_fn, prepare_finetune_variant, train_epoch

BASELINE_MODELS = ["lstm", "gru", "tcn", "transformer"]
RUNS_LOG = EXPERIMENTS_DIR / "runs.jsonl"


def log_run(record):
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RUNS_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")


def train_baseline(model_type, loaders, device, epochs, lr, hidden_dim=128, num_layers=2, batch=64):
    train_loader, val_loader, test_loader = loaders
    model = create_model(model_type, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_mae, best_epoch = float("inf"), 0
    start = time.time()
    for epoch in range(epochs):
        train_epoch(model, train_loader, optimizer, device)
        val_mae = evaluate(model, val_loader, device)
        if val_mae < best_mae:
            best_mae, best_epoch = val_mae, epoch + 1
    elapsed = time.time() - start
    metrics = evaluate_full(model, test_loader, device)
    log_run(build_run_record(model_type, hidden_dim, num_layers, epochs, lr, batch, model, elapsed, best_mae, best_epoch, metrics["mae"]))
    return metrics, elapsed


def pretrain_diffusion(loaders, device, epochs, lr, hidden_dim=64, batch=64):
    train_loader, val_loader, test_loader = loaders
    model = create_model("diffusion", hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_mae, best_epoch = float("inf"), 0
    start = time.time()
    for epoch in range(epochs):
        train_epoch(model, train_loader, optimizer, device, loss_fn=diffusion_loss_step)
        val_mae = evaluate(model, val_loader, device)
        if val_mae < best_mae:
            best_mae, best_epoch = val_mae, epoch + 1
    elapsed = time.time() - start
    metrics = evaluate_full(model, test_loader, device)
    log_run(build_run_record("diffusion_pretrain_us", hidden_dim, 1, epochs, lr, batch, model, elapsed, best_mae, best_epoch, metrics["mae"]))
    return model, metrics, elapsed


def run_finetune_variant(pretrained_model, variant, country, n_seasons, device, epochs, lr, beta, gamma, hidden_dim=64):
    train_loader, val_loader, test_loader, stats = load_finetune_data(country, n_seasons, batch_size=16)
    model = prepare_finetune_variant(pretrained_model, variant, hidden_dim=hidden_dim).to(device)
    loss_fn = get_finetune_loss_fn(variant, beta, gamma)
    start = time.time()
    if loss_fn is not None:
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
        for _ in range(epochs):
            train_epoch(model, train_loader, optimizer, device, loss_fn=loss_fn)
    elapsed = time.time() - start
    metrics = evaluate_full(model, test_loader, device, beta=beta, gamma=gamma, stats=stats)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return metrics, elapsed, n_trainable


def run_finetune_baseline_from_scratch(model_type, country, n_seasons, device, epochs, lr, hidden_dim=64, num_layers=2):
    train_loader, val_loader, test_loader, stats = load_finetune_data(country, n_seasons, batch_size=16)
    model = create_model(model_type, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    start = time.time()
    for _ in range(epochs):
        train_epoch(model, train_loader, optimizer, device)
    elapsed = time.time() - start
    metrics = evaluate_full(model, test_loader, device, stats=stats)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return metrics, elapsed, n_trainable


def plot_comparison(df, countries, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(countries), figsize=(5 * len(countries), 4), squeeze=False)
    axes = axes[0]
    for ax, country in zip(axes, countries):
        sub = df[(df["country"] == country) & (df["variant"].isin(FINETUNE_VARIANTS))]
        for variant in FINETUNE_VARIANTS:
            v = sub[sub["variant"] == variant].sort_values("n_seasons")
            if not v.empty:
                ax.plot(v["n_seasons"], v["mae"], marker="o", label=variant)
        ax.set_title(country)
        ax.set_xlabel("fine-tuning seasons")
        ax.set_ylabel("Test MAE")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)


def main():
    parser = argparse.ArgumentParser(description="Adjoint-matched diffusion vs. baselines: cross-country fine-tuning comparison")
    parser.add_argument("--countries", default=",".join(TARGET_COUNTRIES))
    parser.add_argument("--regimes", default="1,2,3,5,10")
    parser.add_argument("--pretrain-epochs", type=int, default=100)
    parser.add_argument("--finetune-epochs", type=int, default=30)
    parser.add_argument("--baseline-epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    args = parser.parse_args()

    countries = [c for c in args.countries.split(",") if c]
    regimes = [int(r) for r in args.regimes.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []

    print("=== Step 1: baselines on CDC (US) pretrain data ===")
    pretrain_loaders = load_pretrain_data(batch_size=64)
    for model_type in BASELINE_MODELS:
        metrics, elapsed = train_baseline(model_type, pretrain_loaders, device, args.baseline_epochs, args.lr)
        print(f"  {model_type}: test_mae={metrics['mae']:.4f}")
        rows.append({"country": "US", "n_seasons": None, "variant": "pretrain_baseline", "model_type": model_type, "elapsed_s": round(elapsed, 1), **metrics})
    naive_metrics = evaluate_seasonal_naive(pretrain_loaders[2])
    print(f"  seasonal_naive: test_mae={naive_metrics['mae']:.4f}")
    rows.append({"country": "US", "n_seasons": None, "variant": "pretrain_baseline", "model_type": "seasonal_naive", "elapsed_s": 0.0, **naive_metrics})

    print("=== Step 2: pretrain diffusion backbone on CDC (US) data ===")
    diffusion_model, diff_metrics, diff_elapsed = pretrain_diffusion(pretrain_loaders, device, args.pretrain_epochs, args.lr, hidden_dim=args.hidden_dim)
    print(f"  diffusion pretrain: test_mae={diff_metrics['mae']:.4f}")
    checkpoint_dir = EXPERIMENTS_DIR / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(diffusion_model.state_dict(), checkpoint_dir / "diffusion_pretrained_us.pt")
    rows.append({"country": "US", "n_seasons": None, "variant": "pretrain_diffusion", "model_type": "diffusion", "elapsed_s": round(diff_elapsed, 1), **diff_metrics})

    us_df, us_col = get_pretrain_series()
    us_beta, us_gamma = fit_sir_params(us_df[us_col], us_df["S"])
    print(f"  US SIR fit: beta={us_beta:.4f} gamma={us_gamma:.4f}")

    print("=== Step 3: per-country, per-regime fine-tuning comparison ===")
    for country in countries:
        target_df, target_col = get_finetune_series(country)
        beta, gamma = fit_sir_params(target_df[target_col], target_df["S"])
        print(f"  {country} SIR fit: beta={beta:.4f} gamma={gamma:.4f}")
        for n_seasons in regimes:
            for variant in FINETUNE_VARIANTS:
                metrics, elapsed, n_trainable = run_finetune_variant(
                    diffusion_model, variant, country, n_seasons, device, args.finetune_epochs, args.lr, beta, gamma, hidden_dim=args.hidden_dim
                )
                print(f"  {country} n_seasons={n_seasons} {variant}: test_mae={metrics['mae']:.4f} (trainable={n_trainable})")
                rows.append({"country": country, "n_seasons": n_seasons, "variant": variant, "model_type": "diffusion", "elapsed_s": round(elapsed, 1), "trainable_params": n_trainable, **metrics})
            for model_type in BASELINE_MODELS:
                metrics, elapsed, n_trainable = run_finetune_baseline_from_scratch(model_type, country, n_seasons, device, args.finetune_epochs, args.lr, hidden_dim=args.hidden_dim)
                variant_name = f"from_scratch_{model_type}"
                print(f"  {country} n_seasons={n_seasons} {variant_name}: test_mae={metrics['mae']:.4f}")
                rows.append({"country": country, "n_seasons": n_seasons, "variant": variant_name, "model_type": model_type, "elapsed_s": round(elapsed, 1), "trainable_params": n_trainable, **metrics})

    print("=== Step 4: writing comparison CSV ===")
    df = pd.DataFrame(rows)
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = EXPERIMENTS_DIR / "flu_comparison.csv"
    df.to_csv(out_csv, index=False)
    print(f"  wrote {out_csv}")

    print("=== Step 5: plotting comparison ===")
    out_plot = EXPERIMENTS_DIR / "flu_comparison_plot.png"
    plot_comparison(df, countries, out_plot)
    print(f"  wrote {out_plot}")

    proposed = df[df["variant"] == "lora_adjoint_matched"]
    if not proposed.empty:
        print(f"Test MAE: {proposed['mae'].mean():.4f}")


if __name__ == "__main__":
    main()
