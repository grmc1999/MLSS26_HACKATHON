---
name: autoresearch
description: "Simple flu iteration loop: modify train.py → verify → keep/discard"
argument-hint: "[Goal: <text>] [Iterations: N] [RAG: yes|no]"
---

EXECUTE IMMEDIATELY.

## Parse Arguments

Extract from $ARGUMENTS:
- `Goal:` — what to improve
- `Iterations:` or `--iterations` — default 5. "unlimited" for unbounded.
- `RAG:` — "yes" or "no" (default: yes)

## Task Configuration

| Setting | Value |
|---------|-------|
| `TRAIN_PY` | `env/train.py` |
| `VERIFY_CMD` | `python scripts/run_flu_pipeline.py --pretrain-epochs 30 --finetune-epochs 10` |
| `METRIC` | Test MAE (lower is better) |
| `METRIC_CMD` | `grep "Test MAE" run.log \| awk '{print $NF}'` |

## Precondition Checks

1. Verify `env/train.py` exists
2. Verify `scripts/run_flu_pipeline.py` exists
3. Check clean git working tree

## Establish Baseline (Iteration 0)

1. Run baseline: `{VERIFY_CMD} > run.log 2>&1`
2. Extract metric, record as iteration 0

## Iteration Loop

For each iteration:

1. **Research**: Run RAG + web search to find relevant methods. Every change must be grounded in literature.
2. **Plan**: Define ONE focused change to `env/train.py`
3. **Implement**: Make the change
4. **Jury**: `python scripts/code_jury.py --task flu --env-dir env --train-py env/train.py --input-shape "(4, 5, 1)" --expected-output-shape "(4, 10, 1)" --out jury.json`
5. **Commit**: `git add -f env/train.py && git commit -m "experiment: {description}"`
6. **Run**: `{VERIFY_CMD} > run.log 2>&1`
7. **Decide**: Keep if metric improved, revert if worse

## Time Constraints
- ~5 min per experiment. Kill at 10 min.
