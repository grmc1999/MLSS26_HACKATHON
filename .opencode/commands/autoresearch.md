---
name: autoresearch
description: "Autonomous iteration loop: modify, verify, keep/discard against Test Accuracy or OOD F1"
argument-hint: "[Goal: <text>] [Metric: Test Accuracy|OOD F1] [Iterations: N] [Pretrained: yes|no] [--evals]"
---

EXECUTE IMMEDIATELY.

## Parse Arguments

Extract from $ARGUMENTS:
- `Goal:` — what to improve
- `Metric:` — `Test Accuracy` (default) or `OOD F1`
- `Iterations:` or `--iterations` — default 25. "unlimited" for unbounded.
- `Pretrained:` — "yes" to search and finetune pretrained models, "no" to train from scratch (default: no)
- `--evals` — enable mid-loop checkpoints
- `--evals-interval N` — checkpoint frequency override

## Setup (if required context missing)

If Goal or Metric missing → use question (single batched call):
  Q1 (Goal): "What do you want to improve?" — Test Accuracy, OOD F1, or both
  Q2 (Iterations): "Iterations?" — default 25
  Q3 (Pretrained): "Start from scratch or finetune a pretrained model?" — Scratch (default) or Pretrained

## Precondition Checks

1. Verify git repo exists
2. Check clean working tree — warn if dirty
3. Verify `MLAgentBench/benchmarks/medmnist/env/train.py` exists

## Pretrained Model Search (Phase 0) — only if `Pretrained: yes`

If Pretrained=yes, search for suitable models before running the baseline:

1. **HuggingFace Hub**: Search `huggingface.co/models` with keywords (`pneumonia`, `chest-xray`, `medical`, `torchvision`) using WebFetch
2. **GitHub**: Look for open-source pretrained weights (CheXNet, COVID-Net, DenseNet variants)
3. **torchvision**: Available models (`densenet121`, `resnet18`, `efficientnet-b0`) adapted for 28×28 grayscale input by modifying the first conv layer
4. **Integration**: Load via `torch.hub.load()` or `torchvision.models`, adapt input layer (1-channel 28×28), replace classifier head for 3-output OOD logits
5. **Baseline run**: run finetuned model, record as iteration 0
6. **Fallback**: If finetuning doesn't improve over scratch, revert to SimpleCNN baseline

If Pretrained=no, skip directly to Establish Baseline (train SimpleCNN from scratch).

## Establish Baseline (Iteration 0)

1. Run: `python scripts/run_medmnist.py > run.log 2>&1`
2. Extract: `grep "Test 3-class Accuracy" run.log | awk '{print $NF}'` and `grep "OOD F1 Score" run.log | awk '{print $NF}'`
3. Record as iteration 0 in results.tsv: commit, test_acc, ood_f1, memory_gb, status, description
4. Base metric from chosen Metric: Test Accuracy or OOD F1

## Iteration Loop

For each iteration (1 to max_iterations):

### Phase 1: Review
- Read last 10-20 lines of results.tsv
- Run `git log --oneline -10` — see what worked/failed
- Run `git diff HEAD~1` if last was keep to see what improved metric
- Identify: what worked, what failed, what's untried

### Phase 2: Modify
- Make ONE focused change to `MLAgentBench/benchmarks/medmnist/env/train.py`
- Allowed: model architecture, optimizer, hyperparams, loss, augmentation, calibration, OOD strategy
- NOT allowed: modify eval code, install packages, modify loader.py

### Phase 2b: Code Jury
- Run `pytest tests/test_train_unit.py -q` from the project root.
- Fails → fix the change before committing (catches crashes in seconds instead of after a 5min run).

### Phase 3: Commit
- `git add -f MLAgentBench/benchmarks/medmnist/env/train.py && git commit -m "experiment: {description}"`

### Phase 4: Run
- `python scripts/run_medmnist.py > run.log 2>&1`
- Extract: `grep "Test 3-class Accuracy" run.log | awk '{print $NF}'` and `grep "OOD F1 Score" run.log | awk '{print $NF}'`
- Record peak memory from `nvidia-smi`

### Phase 5: Decide
- **keep** — metric improved → commit stays
- **discard** — metric worsened → `git revert HEAD --no-edit`; restore train.py from HEAD
- **crash** — run failed/crashed → revert; read tail -50 run.log for stack trace; attempt fix or skip

### Phase 6: Log
Append to results.tsv (tab-separated): commit, test_acc, ood_f1, memory_gb, status, description
DO NOT commit results.tsv.

### Eval Checkpoint
If --evals: check if current_iteration % interval == 0 → run checkpoint analysis.

### Bounded Check
If bounded: current_iteration >= max_iterations → exit loop, print summary.

## Time Constraints
- Each experiment: ~5 min. Kill at 10 min (`kill %1`). Treat timeout as crash.
- Never stop looping — continue indefinitely until max_iterations or manual interrupt.

## Summary
Print: total iterations, kept/discarded counts, starting → final metric, improvement %.
