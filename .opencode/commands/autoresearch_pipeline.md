---
name: autoresearch_pipeline
description: "Multi-expert pipeline: 8 agents + code jury per iteration — research → plan → code → jury → review → commit → run → decide → log"
argument-hint: "[Goal: <text>] [Task: medmnist|flu] [Metric: ...] [Iterations: N] [RAG: yes|no] [Pretrained: yes|no] [--evals]"
---

EXECUTE IMMEDIATELY.

## Parse Arguments

Extract from $ARGUMENTS:
- `Goal:` — what to improve
- `Task:` — `medmnist` (default) or `flu`
- `Metric:` — task-dependent (see below)
- `Iterations:` or `--iterations` — default 5. "unlimited" for unbounded.
- `RAG:` — "yes" or "no" (default: yes)
- `Pretrained:` — "yes" to search and finetune pretrained models, "no" to train from scratch (default: no)
- `--evals` — enable mid-loop checkpoints
- `--evals-interval N` — checkpoint frequency override

### Task Configurations

| Setting | medmnist | flu |
|---------|----------|-----|
| `ENV_DIR` | `MLAgentBench/benchmarks/medmnist/env/` | `env/` |
| `TRAIN_PY` | `{ENV_DIR}/train.py` | `{ENV_DIR}/train.py` |
| `RUNNER` | `scripts/run_medmnist.py` | `scripts/run_exp.py` |
| `METRIC` | ID Test Acc (default) or OOD F1 | Test MAE (default) |
| `METRIC_CMD` | `grep "Test ID Acc" run.log \| awk '{print $NF}'` | `grep "Test MAE" run.log \| awk '{print $NF}'` |
| `DIRECTION` | higher_is_better | lower_is_better |
| `VERIFY_CMD` | `python scripts/run_medmnist.py` | `python scripts/run_exp.py --epochs 50` |
| `EXPERT` | medical_expert (chest X-ray) | satellite_expert (flu/ILI) |
| `INPUT_SHAPE` | `(4, 1, 28, 28)` → `(4, 3)` | `(4, 5, 1)` → `(4, 10)` |
| `LOG_COLS` | test_acc, ood_f1, val_acc, test_acc_id | test_mae, val_mae, params |

## Setup (if required context missing)

If Goal or Metric missing → use question (single batched call):
  Q1 (Task): "Which task?" — medmnist (chest X-ray OOD) or flu (ILI forecasting)
  Q2 (Goal): "What do you want to improve?" — depends on task
  Q3 (Iterations): "Iterations?" — default 5
  Q4 (RAG): "Use RAG literature search to guide experiments?" — Yes or No
  Q5 (Pretrained): "Start from scratch or finetune a pretrained model?" — Scratch (default) or Pretrained

## Precondition Checks

1. Verify git repo exists
2. Check clean working tree — warn if dirty
3. Verify `{ENV_DIR}/train.py` exists
4. Verify `{RUNNER}` exists

## Pretrained Model Search (Phase 0) — only if `Pretrained: yes`

For **medmnist**: search HuggingFace / torchvision for DenseNet, ResNet, EfficientNet adapted for 28×28 grayscale.
For **flu**: no standard pretrained models — skip to baseline.

If Pretrained=no, skip directly to Establish Baseline.

## Establish Baseline (Iteration 0)

1. Run: `{VERIFY_CMD} > run.log 2>&1`
2. Extract: `{METRIC_CMD}`
3. Record as iteration 0 in `experiments/results.tsv`
4. Base metric from chosen Metric

## Iteration Loop (Multi-Expert Pipeline)

For each iteration (1 to max_iterations):

### Phase 1: Research (research_literature + task_expert)

**Goal**: Understand the problem, find relevant methods from literature.

- **If RAG is enabled**: run `search_medical_literature(query, k=5)` using task-specific keywords
- **research_literature**: what does the literature say about improving this metric?
- **task_expert**: consult via `orchestrator.consult_agent("medical_expert", question)` (loads BioMistral-7B on GPU 1). For flu, use `orchestrator.consult_agent("time_series_expert", question)` (loads Qwen2.5-Math-7B on GPU 1).
- Output: research brief (2-4 sentences) with paper citations

### Phase 2: Plan (autoresearch + llm_expert)

**Goal**: Convert research into a concrete experiment plan.

- **llm_expert**: synthesize the research brief into a clear hypothesis
- **autoresearch**: define the experiment: what ONE change to {TRAIN_PY}, expected outcome, fallback plan
- Check against previous iterations — is this truly untried?
- Output: experiment plan with hypothesis + expected delta

### Phase 3: Implement (cv_expert OR dl_expert)

**Goal**: Write the code change.

- Route to **cv_expert** if change involves: model architecture, data augmentation, preprocessing, attention, pooling
- Route to **dl_expert** if change involves: loss function, optimizer, learning rate, scheduler, regularization, dropout, batch size, calibration
- Consult `orchestrator.consult_agent("code_expert", question)` for implementation advice (loads Qwen2.5-Coder-7B on GPU 1)
- Or consult `orchestrator.consult_agent("time_series_expert", question)` for mathematical/statistical reasoning
- Make ONE focused change to `{TRAIN_PY}`
- Allowed: model architecture, optimizer, hyperparams, loss, augmentation, calibration
- NOT allowed: modify eval code, data loading code, install packages
- Output: modified train.py

### Phase 4: Review (robustness_expert + continual_learning)

**Goal**: Validate the change before running.

- **robustness_expert**: assess impact on the target metric via `orchestrator.consult_agent("time_series_expert", question)`. Check for known failure modes, overfitting risk, calibration impact
- **continual_learning**: is this safe to keep? Will it cause forgetting? What's the rollback plan? Consult `orchestrator.consult_agent("code_expert", question)` for versioning advice
- Output: go/no-go recommendation + risk assessment

### Phase 4b: Code Jury (syntax + shape + gradient check)

**Goal**: Verify code correctness before committing — catch bugs in seconds instead of waiting 5min for a crash.

Run these checks in `{ENV_DIR}`:

1. **Syntax check**: `python -c "import py_compile; py_compile.compile('train.py', doraise=True)"`
2. **Model instantiation**: import and create the model
3. **Forward pass**: feed a random tensor of shape `{INPUT_SHAPE}` and verify output shape
4. **Loss**: compute loss, verify finite
5. **Backward pass**: call `.backward()`, verify gradients non-None

If any check fails → **STOP**, diagnose the error, fix the code, re-run jury.
If all pass → **PASS**, safe to commit.

### Phase 5: Commit

- `git add -f {TRAIN_PY} && git commit -m "pipeline: {agent} — {description}"`

### Phase 6: Run

- `{VERIFY_CMD} > run.log 2>&1`
- Extract: `{METRIC_CMD}`
- Record peak memory from `nvidia-smi`

### Phase 7: Decide

- **keep** — metric improved → commit stays
- **discard** — metric worsened → `git revert HEAD --no-edit`; restore {TRAIN_PY} from HEAD
- **crash** — run failed/crashed → revert; read tail -50 run.log for stack trace; attempt fix or skip

### Phase 8: Log

Append to results.tsv (tab-separated):
  iteration, commit, test_metric, secondary_metric, val_metric, extra, memory_gb, status, description
DO NOT commit results.tsv.

### Eval Checkpoint
If --evals: check if current_iteration % interval == 0 → run checkpoint analysis.

### Bounded Check
If bounded: current_iteration >= max_iterations → exit loop, print summary.

## Time Constraints
- medmnist: ~5 min per experiment (25 epochs). Kill at 10 min.
- flu: ~2 min per experiment (50 epochs). Kill at 5 min.
- Treat timeout as crash.

## Summary
Print: total iterations, kept/discarded counts, starting → final metric, improvement %, most consulted agents.
