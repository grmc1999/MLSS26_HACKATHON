---
name: autoresearch_pipeline
description: "Multi-expert pipeline: 8 agents collaborate per iteration — research → plan → code → review"
argument-hint: "[Goal: <text>] [Metric: ID Test Acc|OOD F1] [Iterations: N] [RAG: yes|no] [--evals]"
---

EXECUTE IMMEDIATELY.

## Parse Arguments

Extract from $ARGUMENTS:
- `Goal:` — what to improve
- `Metric:` — `ID Test Acc` (default) or `OOD F1`
- `Iterations:` or `--iterations` — default 5. "unlimited" for unbounded.
- `RAG:` — "yes" or "no" (default: yes)
- `--evals` — enable mid-loop checkpoints
- `--evals-interval N` — checkpoint frequency override

## Setup (if required context missing)

If Goal or Metric missing → use question (single batched call):
  Q1 (Goal): "What do you want to improve?" — ID Test Acc, OOD F1, or both
  Q2 (Iterations): "Iterations?" — default 5
  Q3 (RAG): "Use RAG literature search to guide experiments?" — Yes or No

## Precondition Checks

1. Verify git repo exists
2. Check clean working tree — warn if dirty
3. Verify `MLAgentBench/benchmarks/medmnist/env/train.py` exists

## Establish Baseline (Iteration 0)

1. Run: `python scripts/run_medmnist.py > run.log 2>&1`
2. Extract: `grep "Val Accuracy" run.log | awk '{print $NF}'` and `grep "Test ID Acc" run.log | awk '{print $NF}'` and `grep "OOD F1" run.log | awk '{print $NF}'`
3. Record as iteration 0 in results.tsv (9 cols): iteration, commit, test_acc, ood_f1, val_acc, test_acc_id, memory_gb, status, description
4. Base metric from chosen Metric: ID Test Acc or OOD F1

## Iteration Loop (Multi-Expert Pipeline)

For each iteration (1 to max_iterations):

### Phase 1: Research (research_literature + medical_expert)

**Goal**: Understand the problem, find relevant methods from literature.

- **If RAG is enabled**: run `search_medical_literature(query, k=5)` using keywords from previous iteration results
- **research_literature**: what does the literature say about improving this metric? Cite specific papers from the index
- **medical_expert**: is the proposed direction clinically meaningful? Pneumonia vs consolidation distinction, domain shift implications
- Output: research brief (2-4 sentences) with paper citations

### Phase 2: Plan (autoresearch + llm_expert)

**Goal**: Convert research into a concrete experiment plan.

- **llm_expert**: synthesize the research brief into a clear hypothesis
- **autoresearch**: define the experiment: what ONE change to train.py, expected outcome, fallback plan
- Check against previous iterations — is this truly untried?
- Output: experiment plan with hypothesis + expected delta

### Phase 3: Implement (cv_expert OR dl_expert)

**Goal**: Write the code change.

- Route to **cv_expert** if change involves: model architecture, data augmentation, preprocessing, OOD scoring method, attention, pooling
- Route to **dl_expert** if change involves: loss function, optimizer, learning rate, scheduler, regularization, dropout, batch size, calibration, temperature
- Make ONE focused change to `MLAgentBench/benchmarks/medmnist/env/train.py`
- Allowed: model architecture, optimizer, hyperparams, loss, augmentation, calibration, OOD strategy
- NOT allowed: modify eval code, install packages, modify loader.py
- Output: modified train.py

### Phase 4: Review (robustness_expert + continual_learning)

**Goal**: Validate the change before running.

- **robustness_expert**: assess impact on OOD detection. Will this hurt calibration? Does it affect the confidence threshold? Check for known failure modes
- **continual_learning**: is this safe to keep? Will it cause forgetting? What's the rollback plan?
- Output: go/no-go recommendation + risk assessment

### Phase 5: Commit

- `git add -f MLAgentBench/benchmarks/medmnist/env/train.py && git commit -m "pipeline: {agent} — {description}"`

### Phase 6: Run

- `python scripts/run_medmnist.py > run.log 2>&1`
- Extract: `grep "Val Accuracy" run.log | awk '{print $NF}'`, `grep "Test ID Acc" run.log | awk '{print $NF}'`, `grep "OOD F1" run.log | awk '{print $NF}'`
- Record peak memory from `nvidia-smi`

### Phase 7: Decide

- **keep** — metric improved → commit stays
- **discard** — metric worsened → `git revert HEAD --no-edit`; restore train.py from HEAD
- **crash** — run failed/crashed → revert; read tail -50 run.log for stack trace; attempt fix or skip

### Phase 8: Log

Append to results.tsv (tab-separated, 9 cols):
  iteration, commit, test_acc, ood_f1, val_acc, test_acc_id, memory_gb, status, description
DO NOT commit results.tsv.

### Eval Checkpoint
If --evals: check if current_iteration % interval == 0 → run checkpoint analysis.

### Bounded Check
If bounded: current_iteration >= max_iterations → exit loop, print summary.

## Time Constraints
- Each experiment: ~5 min. Kill at 10 min (`kill %1`). Treat timeout as crash.
- Never stop looping — continue indefinitely until max_iterations or manual interrupt.

## Summary
Print: total iterations, kept/discarded counts, starting → final metric, improvement %, most consulted agents.
