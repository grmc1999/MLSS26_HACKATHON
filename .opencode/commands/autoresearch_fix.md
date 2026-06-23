---
name: autoresearch_fix
description: "Crush errors in the OOD pipeline one-by-one until zero remain"
argument-hint: "[Target: train|eval|data] [Iterations: N] [--from-debug]"
---

EXECUTE IMMEDIATELY.

## Parse Arguments

Extract from $ARGUMENTS:
- `Target:` or `--target` — what to fix: train, eval, data
- `Iterations:` or `--iterations` — default 20. "unlimited" for unbounded.
- `--from-debug` — read findings from previous debug session

## Setup (if Target missing)

1. Run `python scripts/run_medmnist.py 2>&1 | tail -50` to detect current errors
2. Present results:
   Q1 (Fix What): "Found errors. Fix what?" — crashes, accuracy, OOD F1, all
   Q2 (Launch): "Ready?" — fix until zero or fix with limit

## Establish Baseline (Iteration 0)

1. Run the experiment: capture error count or metric
2. Record baseline

## Iteration Loop

### Phase 1: Prioritize
Order: crashes → wrong results → poor accuracy → OOD detection → warnings

### Phase 2: Fix ONE Thing
- Make ONE focused fix in train.py
- Common fixes: model architecture, loss function, hyperparameters, OOD threshold, data preprocessing

### Phase 2b: Code Jury
- Run `pytest tests/test_train_unit.py -q` from the project root.
- Fails → the fix didn't work or broke something else; iterate before spending a full run on it.

### Phase 3: Verify
- `python scripts/run_medmnist.py > run.log 2>&1`
- Check: did error count decrease? Did metric improve?

### Phase 4: Decide
- **keep** — error count decreased or metric improved
- **discard** — same or worse → `git revert HEAD --no-edit`
- **crash** — command failed → revert

## Summary
Print: total errors fixed, remaining errors, fix success rate.
