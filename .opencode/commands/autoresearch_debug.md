---
name: autoresearch_debug
description: "Hunt bugs in the chest X-ray OOD pipeline: hypothesize, test, falsify, repeat"
argument-hint: "[Symptom: <text>] [Iterations: N] [--fix] [RAG: yes|no]"
---

EXECUTE IMMEDIATELY.

## Parse Arguments

Extract from $ARGUMENTS:
- `Symptom:` or `--symptom` — error message or unexpected behavior
- `Iterations:` or `--iterations` — default 15. "unlimited" for unbounded.
- `--fix` — shorthand for `--chain fix`

## Setup (if Symptom missing)

1. Run `python scripts/run_medmnist.py 2>&1 | tail -50` to check current state
2. question (single batch):
   Q1 (Issue): "What's wrong?" — training crashes, poor accuracy, OOD not detected, memory error
   Q2 (Scope): "Which file?" — train.py (default), loader.py
   Q3 (Depth): "How deep?" — quick (5), standard (15), deep (30)
   Q4 (RAG): "Use RAG literature search to guide experiments?" — Yes or No

## Investigation Techniques

| Technique | When to Use |
|---|---|
| Binary search | Training was working, now it's not |
| Minimal reproduction | Simplify to smallest failing config |
| Trace | Follow execution through the training loop |
| Pattern search | Check for common bugs: wrong device, mismatched dimensions, NaN gradients |

## Focus Areas for This Project

Common issues to check:
- Model output dimension (2 classes for PneumoniaMNIST vs 3 classes for test eval)
- OOD threshold too high/low (default 0.7)
- Device mismatch (CPU vs CUDA)
- Data shape (28×28 grayscale, batch dimension)
- Softmax vs logits in OOD scoring
- Memory growth (check nvidia-smi between runs)
- NaN in loss (check for division by zero in ood_metrics)

## Establish Baseline

1. Run experiment: `python scripts/run_medmnist.py > run.log 2>&1`
2. Check for errors, unexpected output, low metrics
3. Create debug dir: `experiments/debug-{YYMMDD}-{HHMM}/`

## Iteration Loop

### Phase 1: Hypothesize
- Form ONE specific, falsifiable hypothesis
- Format: "I hypothesize that {X} because {evidence}. Test by {Y}."

### Phase 2: Investigate
- Read relevant code, inspect train.py logic
- Look at the data flow: loader → model → loss → OOD scoring
- Check shapes, types, device placement

### Phase 3: Classify
- **confirmed** — bug found with evidence (file:line)
- **disproven** — hypothesis wrong
- **inconclusive** — needs different approach

### Phase 4: Log
Record findings. If --fix, chain to fix.

## Summary
Print: total hypotheses tested, confirmed bugs with file:line.
