"""Append a result row to the human-readable results.tsv file.

Usage:
    python scripts/append_results_tsv.py --task flu \
        --results experiments/loop-flu-YYMMDD-HHMM/results.tsv \
        --iteration 4 --commit abc1234 --test-mae 0.5823 \
        --val-mae 0.6011 --params 298881 --memory-gb 1.2 \
        --status keep --description "Added dropout 0.2"
"""

import argparse
import csv
import os
import sys
from pathlib import Path

from pipeline_utils import ensure_parent_dir


FLU_HEADER = ["iteration", "commit", "test_mae", "val_mae", "params", "memory_gb", "status", "description"]
MEDMNIST_HEADER = ["iteration", "commit", "test_acc", "ood_f1", "val_acc", "test_acc_id", "memory_gb", "status", "description"]


def get_header(task: str) -> list[str]:
    if task == "flu":
        return FLU_HEADER
    return MEDMNIST_HEADER


def make_row(header: list[str], args: argparse.Namespace) -> list[str]:
    mapping = {
        "iteration": str(args.iteration),
        "commit": args.commit or "",
        "test_mae": _fmt(args.test_mae),
        "val_mae": _fmt(args.val_mae),
        "test_acc": _fmt(args.test_acc),
        "ood_f1": _fmt(args.ood_f1),
        "val_acc": _fmt(args.val_acc),
        "test_acc_id": _fmt(args.test_acc_id),
        "params": str(args.params) if args.params is not None else "",
        "memory_gb": _fmt(args.memory_gb),
        "status": args.status or "",
        "description": args.description or "",
    }
    return [mapping.get(col, "") for col in header]


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def main():
    parser = argparse.ArgumentParser(description="Append result row to results.tsv")
    parser.add_argument("--task", required=True, choices=["medmnist", "flu"])
    parser.add_argument("--results", required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--commit", default=None)
    parser.add_argument("--test-mae", type=float, default=None)
    parser.add_argument("--val-mae", type=float, default=None)
    parser.add_argument("--test-acc", type=float, default=None)
    parser.add_argument("--ood-f1", type=float, default=None)
    parser.add_argument("--val-acc", type=float, default=None)
    parser.add_argument("--test-acc-id", type=float, default=None)
    parser.add_argument("--params", type=int, default=None)
    parser.add_argument("--memory-gb", type=float, default=None)
    parser.add_argument("--status", default=None)
    parser.add_argument("--description", default=None)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing row for the same iteration")
    args = parser.parse_args()

    header = get_header(args.task)
    row = make_row(header, args)
    results_path = Path(args.results)
    ensure_parent_dir(str(results_path))

    rows = []
    file_exists = results_path.exists()

    if file_exists:
        with open(results_path, "r", newline="") as f:
            reader = csv.reader(f, delimiter="\t")
            for r in reader:
                rows.append(r)

    if file_exists and len(rows) > 0:
        existing_header = rows[0]
        if existing_header != header:
            print(f"WARNING: existing header {existing_header} != expected {header}", file=sys.stderr)
    else:
        rows = [header]

    if not args.overwrite:
        it_col = header.index("iteration")
        for r in rows[1:]:
            if len(r) > it_col and r[it_col] == str(args.iteration):
                print(f"WARNING: iteration {args.iteration} already exists in {args.results}. "
                      f"Pass --overwrite to replace.", file=sys.stderr)
                sys.exit(1)

    if args.overwrite:
        it_col = header.index("iteration")
        rows = [rows[0]] + [r for r in rows[1:] if not (len(r) > it_col and r[it_col] == str(args.iteration))]

    rows.append(row)

    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(rows)

    print(f"Appended iteration {args.iteration} to {args.results}")


if __name__ == "__main__":
    main()
