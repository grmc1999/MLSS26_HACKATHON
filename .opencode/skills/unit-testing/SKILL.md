---
name: unit-testing
description: "Fast pytest checks (no training run) that every train.py change must pass before commit"
version: 1.0.0
---

# Unit Testing — Code Jury, formalized

Replaces the manual "Code Jury" (5 inline `python -c` commands) from
`.opencode/commands/autoresearch_pipeline.md` Phase 4b with a real, reusable
pytest file: `tests/test_train_unit.py`.

## Run it

```bash
pytest tests/test_train_unit.py -q
```

Takes seconds. No GPU required, no dataset download, no API key.

## What it checks

- **Syntax/import** — `train.py` imports cleanly (catches typos before a 20-epoch run starts)
- **Forward pass** — `create_model()` output shape is `(batch, 3)` for batch sizes 1/4/16
- **Loss** — finite, never NaN/inf
- **Backward pass** — every parameter gets a gradient, all finite
- **One-batch overfit** — the model can reduce loss on a single tiny batch (classic sanity check)
- **`ood_metrics` / `per_class_accuracy` / `in_distribution_accuracy`** — correctness against
  hand-computed confusion matrices, not just "doesn't crash". The old Code Jury never checked this.

## When to use

- Automatically: Phase 4b of `/autoresearch_pipeline`, Phase 2b of `/autoresearch` and `/autoresearch_fix`
- Manually: after any edit to `MLAgentBench/benchmarks/medmnist/env/train.py`, before committing

## Adding a test

If a new architecture, loss, or metric function is added to `train.py`, add a matching
`test_*` function to `tests/test_train_unit.py` — mirror the function under test, parametrize
over edge cases (empty class, perfect predictor, etc.) instead of writing one test per value.

## Related

See [[regression-testing]] for contract tests (e.g. orchestrator stdout parsing) and
[[e2e-testing]] for real-data pipeline runs.
