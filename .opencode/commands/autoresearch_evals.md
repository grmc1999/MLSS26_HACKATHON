---
name: autoresearch_evals
description: "Analyze experiment results: trends, plateaus, regressions across all runs"
argument-hint: "[path/to/results.tsv] [--format text|json|md]"
---

EXECUTE IMMEDIATELY.

## Parse Arguments

Extract from $ARGUMENTS:
- Positional path to a specific TSV or JSONL file
- `--format` — output format: text (default), json, md

## Input Discovery

1. If path provided → use that file
2. If no path → scan:
   - `experiments/results.tsv` (canonical format)
   - `experiments/runs.jsonl` (JSONL format)
   - `experiments/loop-*/results.tsv` (loop output)
3. If multiple found → question: "Which results to analyze?"

## Parse Results

### TSV Format (from autorearch loop)
Columns: commit, test_acc, ood_f1, memory_gb, status, description

### JSONL Format (from run_medmnist.py)
Fields: timestamp, model, epochs, lr, test_acc, ood_f1, params, elapsed_s

## Report

### Key Metrics
- Total experiments: N | Kept: X | Discarded: Y | Crash: Z
- Best Test Accuracy: value (commit)
- Best OOD F1: value (commit)
- Starting → final: improvement/degradation

### Trend Analysis
- Test Accuracy progression over runs
- OOD F1 progression over runs
- Correlation between params and accuracy
- Correlation between memory and performance

### Recommendations
Based on trend analysis:
- Continue current strategy
- Switch focus (accuracy vs OOD)
- Try specific method from research
- Early stop if plateaued
