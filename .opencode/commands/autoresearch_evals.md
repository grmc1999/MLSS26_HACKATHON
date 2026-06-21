---
name: autoresearch_evals
description: "Analyze experiment results: trends, plateaus, convergence"
argument-hint: "[--file <tsv>]"
---

EXECUTE IMMEDIATELY.

## Parse Arguments
- `--file <tsv>` — results file to analyze (default: latest in experiments/)

## How It Works

1. Read the results TSV
2. Compute per-checkpoint deltas (floor(max_iterations/3) intervals)
3. Detect: trends, plateaus, convergence signals
4. Report: best iteration, improvement %, stalled detection
5. Recommendation: continue / stop / change strategy

## Output
```
Metric: {start} → {end} ({delta})
Kept: {n}/{total}
Trend: {up/flat/down}
Recommendation: {continue|stop|change strategy}
```
