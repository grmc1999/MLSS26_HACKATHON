---
name: autoresearch_pipeline
description: "Multi-expert pipeline: research → discuss → code → review, with RAG"
argument-hint: "[Goal: <text>] [Iterations: N] [RAG: yes|no]"
---

EXECUTE IMMEDIATELY.

## Parse Arguments

Extract from $ARGUMENTS:
- `Goal:` — what to improve
- `Iterations:` or `--iterations` — default 5
- `RAG:` — "yes" or "no" (default: yes)

## Setup (if missing)

question (single batch):
  Q1 (Goal): "What do you want to improve?" — ID Test Acc, OOD F1, both
  Q2 (Iterations): "Iterations?" — default 5
  Q3 (RAG): "Use RAG literature search?" — Yes or No

## Precondition Checks

1. Verify git repo, clean working tree
2. Verify train.py exists

## Establish Baseline (Iteration 0)

1. `python scripts/run_medmnist.py > run.log 2>&1`
2. Extract Test Accuracy and OOD F1
3. Record in results.tsv

## Iteration Loop (multi-expert pipeline)

For each iteration (1 to max_iterations):

### Phase 1: Research (research_literature + medical_expert)
- **If RAG enabled**: query FAISS index for relevant papers
- **research_literature**: what does the literature say about this approach?
- **medical_expert**: is this clinically meaningful for chest X-ray OOD?
- Output: research brief with paper citations

### Phase 2: Plan (autoresearch + llm_expert)
- **llm_expert**: synthesize research into an experiment plan
- **autoresearch**: define hypothesis, expected outcome, fallback
- Output: experiment plan with success criteria

### Phase 3: Implement (cv_expert OR dl_expert)
- Route to **cv_expert** if change involves architecture/augmentation
- Route to **dl_expert** if change involves loss/optimizer/calibration
- Output: ONE focused change to train.py

### Phase 4: Review (robustness_expert + continual_learning)
- **robustness_expert**: will this hurt OOD detection? Check calibration impact
- **continual_learning**: is this safe to keep? Assess forgetting risk
- Output: go/no-go recommendation

### Phase 5: Execute
- Commit: `git add -f MLAgentBench/benchmarks/medmnist/env/train.py && git commit -m "pipeline: {description}"`
- Run: `python scripts/run_medmnist.py > run.log 2>&1`
- Extract Test Accuracy and OOD F1

### Phase 6: Decide
- **keep** if metric improved → commit stays
- **discard** if worsened → `git revert HEAD --no-edit`
- Log to results.tsv: commit, test_acc, ood_f1, memory_gb, status, description

### Bounded Check
If iteration >= max_iterations → exit loop.

## Summary
Print: total iterations, kept/discarded, best metrics, most consulted agents.
