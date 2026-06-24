"""Generate and validate the mandatory per-iteration JSON log.

Usage:
    python scripts/make_iter_log.py \
        --task flu --iteration 4 --status keep \
        --metric-name "Test MAE" --metric-direction lower_is_better \
        --metric-value 0.5823 --baseline 0.5897 \
        --change-type regularization --change-file env/train.py \
        --diff-summary "Added dropout 0.2 between GRU encoder layers" \
        --hypothesis "..." --mechanism "..." --expected-delta "..." \
        --risk "low" --baseline-compared-to "env.baseline/" \
        --elapsed-s 42.3 --memory-gb 1.2 --params 298881 \
        --commit-hash abc1234 \
        --commit-message "pipeline: time_series_expert -- Add dropout" \
        --out experiments/loop-flu-YYMMDD-HHMM/iterations/iter-4.json
"""

import argparse
import sys

from pipeline_utils import (
    ensure_parent_dir,
    get_git_commit_hash,
    reject_empty_required,
    utc_now_iso,
    write_json,
)

VALID_STATUSES = {"keep", "discard", "crash"}
VALID_DIRECTIONS = {"higher_is_better", "lower_is_better"}
VALID_TASKS = {"flu"}


def validate_fields(args: argparse.Namespace) -> dict:
    errors = []

    def req(v, name):
        try:
            return reject_empty_required(v, name)
        except ValueError as e:
            errors.append(str(e))
            return ""

    task = req(args.task, "task")
    status = req(args.status, "status")
    direction = req(args.metric_direction, "metric-direction")
    change_type = req(args.change_type, "change-type")

    if task and task not in VALID_TASKS:
        errors.append(f"task must be one of {VALID_TASKS}, got {task}")
    if status and status not in VALID_STATUSES:
        errors.append(f"status must be one of {VALID_STATUSES}, got {status}")
    if direction and direction not in VALID_DIRECTIONS:
        errors.append(f"direction must be one of {VALID_DIRECTIONS}, got {direction}")

    if errors:
        print("Validation errors:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        sys.exit(1)

    commit_hash = args.commit_hash or get_git_commit_hash() or "unknown"
    req(commit_hash, "commit-hash")

    return {
        "iteration": args.iteration,
        "task": task,
        "commit": commit_hash,
        "timestamp": utc_now_iso(),
        "status": status,
        "metric": {
            "name": req(args.metric_name, "metric-name"),
            "direction": direction,
            "value": args.metric_value,
            "baseline": args.baseline,
        },
        "change": {
            "type": change_type,
            "file": req(args.change_file, "change-file"),
            "diff_summary": req(args.diff_summary, "diff-summary"),
        },
        "jury_reasoning": {
            "hypothesis": req(args.hypothesis, "hypothesis"),
            "mechanism": req(args.mechanism, "mechanism"),
            "expected_delta": req(args.expected_delta, "expected-delta"),
            "risk": req(args.risk, "risk"),
            "baseline_compared_to": req(args.baseline_compared_to, "baseline-compared-to"),
        },
        "resources": {
            "elapsed_s": args.elapsed_s,
            "memory_gb": args.memory_gb,
            "params": args.params,
        },
        "git": {
            "commit_hash": commit_hash,
            "commit_message": req(args.commit_message, "commit-message"),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Generate per-iteration JSON log")
    parser.add_argument("--task", required=True, choices=list(VALID_TASKS))
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--status", required=True, choices=list(VALID_STATUSES))
    parser.add_argument("--metric-name", required=True)
    parser.add_argument("--metric-direction", required=True, choices=list(VALID_DIRECTIONS))
    parser.add_argument("--metric-value", type=float, required=True)
    parser.add_argument("--baseline", type=float, required=True)
    parser.add_argument("--change-type", required=True)
    parser.add_argument("--change-file", required=True)
    parser.add_argument("--diff-summary", required=True)
    parser.add_argument("--hypothesis", required=True)
    parser.add_argument("--mechanism", required=True)
    parser.add_argument("--expected-delta", required=True)
    parser.add_argument("--risk", required=True)
    parser.add_argument("--baseline-compared-to", required=True)
    parser.add_argument("--elapsed-s", type=float, required=True)
    parser.add_argument("--memory-gb", type=float, required=True)
    parser.add_argument("--params", type=int, required=True)
    parser.add_argument("--commit-hash", default=None)
    parser.add_argument("--commit-message", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    record = validate_fields(args)
    write_json(args.out, record)
    print(f"Wrote iteration log: {args.out}")
    print(f"  iteration={args.iteration}, status={args.status}, metric_value={args.metric_value}")


if __name__ == "__main__":
    main()
