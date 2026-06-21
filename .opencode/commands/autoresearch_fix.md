---
name: autoresearch_fix
description: "Crush errors one-by-one until zero remain"
argument-hint: "[--target <cmd>] [--guard <cmd>] [Iterations: N]"
---

EXECUTE IMMEDIATELY.

## Parse Arguments
- `--target <cmd>` — verify command (e.g., `python train.py`)
- `--guard <cmd>` — safety command that must pass
- `Iterations:` — default 20

## How It Works

1. Auto-detect what's broken (runtime errors, NaN, shape mismatches)
2. Prioritize: blockers first
3. Fix ONE thing
4. Commit with `experiment: fix {description}`
5. Verify error count decreased
6. Guard check (if configured)
7. Keep/Revert
8. Repeat

## Stop Condition
Stops automatically when error count hits zero.

## Chain from debug
`/autoresearch_debug --chain fix`
