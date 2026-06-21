---
name: autoresearch_plan
description: "Convert a goal into validated Scope, Metric, Direction, Verify config"
argument-hint: "[Goal: <text>] [--chain <targets>]"
---

EXECUTE IMMEDIATELY.

## Parse Arguments

Extract from $ARGUMENTS:
- `Goal:` — what to achieve
- `--chain <targets>` — downstream commands

## Setup (if Goal missing)

Ask once:
- Goal: "What do you want to achieve?"
- Type: "Improve metric, fix errors, audit security, explore, document, ship"

## Phase 1: Analyze Goal

Parse the goal:
- Is it measurable? (Dice Score, loss, params, speed)
- What's the natural scope? (train.py, model architecture, data pipeline)
- What subcommand fits best? (core loop, debug, fix, security)

## Phase 2: Derive Scope

Scan project structure. Relevant files:
- `MLAgentBench/benchmarks/identify-contrails/env/train.py` — main training script
- `MLAgentBench/benchmarks/identify-contrails/env/encode.py` — RLE encoding
- `scripts/run_exp.py` — experiment CLI
- `MLAgentBench/benchmarks/identify-contrails/scripts/eval.py` — evaluation

## Phase 3: Derive Metric + Direction

For contrails task:
- Dice Score (higher_is_better) — primary metric
- Validation Loss (lower_is_better) — secondary
- Training speed (higher_is_better) — efficiency

## Phase 4: Derive Verify Command

Propose: `python scripts/run_exp.py --epochs 50 2>&1 | tail -5`

For Dice: extract with `| grep "Test Dice" | awk '{print $NF}'`

Safety screen the command for dangerous operations.

## Phase 5: Suggest Iterations

- Architecture changes → 15-20
- Hyperparameter tuning → 20-30
- Data pipeline changes → 10-15

## Phase 6: Present Config

Output ready config:

```
/autoresearch
Goal: {goal}
Metric: {metric}
Direction: {direction}
Verify: {command}
Iterations: {count}
```

Ask: "Run this now, or adjust?"
