---
name: autoresearch_learn
description: "Extract cross-iteration lessons: what worked, what failed, what patterns emerged"
argument-hint: "[--mode init|update] [Iterations: N]"
---

EXECUTE IMMEDIATELY.

## Parse Arguments

Extract from $ARGUMENTS:
- `--mode <mode>` — init (create from scratch), update (refresh existing), check (read-only)
- `Iterations:` or `--iterations` — default 10.

## Setup

question (single batch):
  Q1 (Mode): "What to do?" — init (generate lessons from results), update (append newer results), check (read-only)

## Process

### Review Results
- Read results.tsv from all experiments/ directories
- Read runs.jsonl for historical data
- Summarize findings by category

### Categories to Analyze

| Category | Questions |
|---|---|
| Architecture | Which model designs worked best? Conv layers, widths, depths? |
| Training | Best learning rates, batch sizes, optimizers, schedulers? |
| Loss | Cross-entropy vs focal vs label smoothing impact? |
| OOD Detection | Threshold tuning, Mahalanobis, energy score effectiveness? |
| Data | Augmentation impact? Normalization strategy? |
| Calibration | Temperature scaling impact on OOD F1? |

### Output
Create `experiments/learn-{YYMMDD}-{HHMM}/` with:
- `lessons.md` — structured lessons organized by category
- `recommendations.md` — actionable recommendations for next experiments
- `patterns.tsv` — pattern matrix (category, what, outcome, confidence)
