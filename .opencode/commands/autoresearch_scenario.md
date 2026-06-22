---
name: autoresearch_scenario
description: "Explore what-if scenarios: different data splits, class distributions, noise levels"
argument-hint: "[Scenario: <text>] [Iterations: N] [RAG: yes|no]"
---

EXECUTE IMMEDIATELY.

## Parse Arguments

Extract from $ARGUMENTS:
- `Scenario:` or `--scenario` — what scenario to explore
- `Iterations:` or `--iterations` — default 15. "unlimited" for unbounded.

## Setup

question (single batch):
  Q1 (Scenario): "What scenario to explore?" — class imbalance, domain shift, noise, data size
   Q2 (Depth): "Depth?" — standard (15), deep (30)
   Q3 (RAG): "Use RAG literature search to guide experiments?" — Yes or No

## Scenario Dimensions (Chest X-ray OOD)

| Dimension | Example |
|---|---|
| Class imbalance | What if pneumonia is 90% vs 10%? |
| Domain shift | What if test X-rays come from different hospital? |
| Noise | What if inputs have Gaussian noise? |
| Data size | What if only 10% of training data available? |
| Threshold sensitivity | How does OOD F1 change with threshold 0.1-0.9? |
| Model capacity | What if model is smaller/larger? |

## Process
1. For each scenario: modify train.py or run_medmnist.py flags
2. Run experiment and record metrics
3. Compare to baseline
4. Log results to `experiments/scenario-{YYMMDD}-{HHMM}/`
