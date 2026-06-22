---
name: autoresearch_plan
description: "Convert a chest X-ray OOD goal into validated Scope, Metric, Verify config"
argument-hint: "[Goal: <text>] [RAG: yes|no]"
---

EXECUTE IMMEDIATELY.

## Parse Arguments

Extract from $ARGUMENTS:
- `Goal:` — text after keyword, or full $ARGUMENTS if no keyword
Remaining text = goal description.

## Setup (if Goal missing)

question (single batch):
  Q1 (Goal): "What do you want to achieve?" — improve accuracy, OOD detection, calibration, etc.
   Q2 (Focus): "Which metric?" — Test Accuracy, OOD F1, both, or something else
   Q3 (RAG): "Use RAG literature search to guide experiments?" — Yes or No

## Phase 1: Analyze Goal

Parse the goal to determine:
- Is it measurable? (metric-driven vs subjective)
- What's the natural scope? (just train.py, or also scripts)
- What subcommand fits best? (autoresearch, fix, debug, scientific)

## Phase 2: Derive Scope

This project has a fixed scope:
- Modify: `MLAgentBench/benchmarks/medmnist/env/train.py`
- Read-only: eval code, loader.py, scripts
- Allowed changes: architecture, optimizer, hyperparams, loss, augmentation, calibration, OOD strategy

## Phase 3: Derive Metric + Direction

Available metrics:
- `Test Accuracy` — higher_is_better (default)
- `OOD F1` — higher_is_better
- Both — composite score

## Phase 4: Derive Verify Command

Standard verify:
```
python scripts/run_medmnist.py > run.log 2>&1 && grep "Test Accuracy" run.log | awk '{print $NF}'
```
For OOD F1:
```
python scripts/run_medmnist.py > run.log 2>&1 && grep "OOD F1" run.log | awk '{print $NF}'
```

## Phase 5: Suggest Iterations

Based on goal complexity:
- Simple tuning → 10-15
- Architecture change → 20-25
- Full OOD pipeline → 25-30

## Phase 6: Present Config

Output a ready-to-run autoresearch config block:
```
/autoresearch
Goal: {derived goal}
Metric: {derived metric}
Iterations: {suggested count}
```
