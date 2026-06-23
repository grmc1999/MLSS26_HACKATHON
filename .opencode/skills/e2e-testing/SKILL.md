---
name: e2e-testing
description: "Short real-data pipeline run (GPU if available) before committing to a full 20-epoch experiment"
version: 1.0.0
---

# E2E Testing — Real Data, Short Run

[[unit-testing]] only feeds `train.py` random tensors, so it can't catch bugs in data loading,
label encoding, or anything that only shows up with the real distribution. This skill runs the
actual `loader.get_datasets()` → `train_epoch()` → `evaluate()` → `ood_metrics()` chain for one
epoch on real PneumoniaMNIST/ChestMNIST data.

## Run it

```bash
pytest tests/test_pipeline_e2e.py -q -m e2e
```

Uses `torch.device("cuda" if torch.cuda.is_available() else "cpu")` — same device selection as
`train.py` itself. Takes a few seconds on GPU.

## What it checks

- `get_datasets()` returns real, correctly-shaped data (no synthetic/invented data)
- One real training epoch produces a finite loss and an accuracy in `[0, 1]`
- `evaluate()` predictions align 1:1 with labels
- `ood_metrics()` confusion counts sum to the full test set size

## Data dependency

First run downloads PneumoniaMNIST/ChestMNIST via the `medmnist` package (cached under
`~/.medmnist`) and builds `data/medmnist_subset/chestmnist_3class.npz` once (cached — not
rebuilt on subsequent runs). See `MLAgentBench/benchmarks/medmnist/env/loader.py`.

## When to use

- Before a long experiment loop, to catch real-data issues a unit test with fake tensors can't see
- After any change to `loader.py`

## Related

See [[unit-testing]] for fast no-training checks and [[regression-testing]] for contract checks.
