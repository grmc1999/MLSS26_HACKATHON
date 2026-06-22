---
name: autoresearch_ship
description: "Lock in the best chest X-ray OOD model: final eval, export, register"
argument-hint: "[--dry-run] [--checklist-only]"
---

EXECUTE IMMEDIATELY.

## Parse Arguments

Extract from $ARGUMENTS:
- `--dry-run` — validate everything but don't ship
- `--checklist-only` — just generate checklist

## Setup

question (single batch):
  Q1 (Action): "What to do?" — ship best model, generate submission, create report, all

## Phase 1: Identify Best Model
- Scan results.tsv for best metrics
- Identify the commit with highest Test Accuracy and/or OOD F1
- Check if checkpoint exists: `ls MLAgentBench/benchmarks/medmnist/env/medmnist_model.pth`

## Phase 2: Checklist
- [ ] Best model identified from results
- [ ] Checkpoint file exists
- [ ] Final eval reproduces best metric
- [ ] Test Accuracy and OOD F1 recorded
- [ ] Model params and architecture documented
- [ ] Results logged to experiments/runs.jsonl

## Phase 3: Final Eval
- `git checkout {best_commit}`
- `python scripts/run_medmnist.py > run.log 2>&1`
- Extract and confirm metrics

## Phase 4: Log
Create `experiments/best-{YYMMDD}-{HHMM}/` with:
- final_metrics.txt — Test Accuracy, OOD F1, per-class accuracies
- model_info.txt — architecture, params, training config
- checkpoint copy
