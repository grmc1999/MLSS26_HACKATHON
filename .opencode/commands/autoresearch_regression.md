---
name: autoresearch_regression
description: "Regression stability gate: verify new changes don't break existing results"
argument-hint: "[--candidate <commit>] [--baseline <commit>] [RAG: yes|no]"
---

EXECUTE IMMEDIATELY.

## Parse Arguments

Extract from $ARGUMENTS:
- `--candidate <commit>` — commit to test (default: HEAD)
- `--baseline <commit>` — commit to compare against (default: previous kept)

## Setup

question (single batch):
  Q1 (Candidate): "Which commit to test?" — HEAD (default) or specific commit
   Q2 (Baseline): "Compare against?" — best known commit or specific
   Q3 (RAG): "Use RAG literature search to guide experiments?" — Yes or No

## Process

### 1. Establish Baseline
- `git checkout {baseline}`
- `python scripts/run_medmnist.py > baseline.log 2>&1`
- Record: Test Accuracy, OOD F1, params, memory

### 2. Evaluate Candidate
- `git checkout {candidate}`
- `python scripts/run_medmnist.py > candidate.log 2>&1`
- Record: Test Accuracy, OOD F1, params, memory

### 3. Compare (8 dimensions)

| Dimension | Check | Pass/Fail |
|---|---|---|
| Test Accuracy | candidate >= baseline - 0.01 | |
| OOD F1 | candidate >= baseline - 0.01 | |
| Params | candidate <= baseline * 2 | |
| Memory | candidate <= baseline * 1.5 | |
| Runtime | candidate <= baseline * 2 | |
| No crashes | run completed without error | |
| Reproducible | run 2x gives same metrics | |
| OOD detection | OOD F1 > 0 (non-trivial) | |

### 4. Verdict
- **STABLE** — all dimensions pass
- **UNSTABLE** — any critical dimension fails
- **DEGRADED** — non-critical dimensions fail

## Output
Create `experiments/regression-{YYMMDD}-{HHMM}/` with comparison report.
