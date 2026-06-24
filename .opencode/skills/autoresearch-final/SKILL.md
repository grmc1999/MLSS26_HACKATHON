---
name: autoresearch_final
description: "Multi-expert pipeline via deterministic scripts: research → plan → code → jury → commit → run → decide → log"
argument-hint: "[Goal: <text>] [Task: medmnist|flu] [Metric: ...] [Iterations: N] [RAG: yes|no] [Pretrained: yes|no] [--evals]"
---

EXECUTE IMMEDIATELY.

## Parse Arguments

Extract from $ARGUMENTS:
- `Goal:` — what to improve
- `Task:` — `medmnist` (default) or `flu`
- `Metric:` — task-dependent
- `Iterations:` or `--iterations` — default 5. "unlimited" for unbounded.
- `RAG:` — "yes" or "no" (default: yes)
- `Pretrained:` — "yes" to search and finetune pretrained models (default: no)

### Task Configuration

| Setting | medmnist | flu |
|---------|----------|-----|
| `ENV_DIR` | `MLAgentBench/benchmarks/medmnist/env/` | `env/` |
| `RUNNER` | `scripts/run_medmnist.py` | `scripts/run_flu_pipeline.py` |
| `VERIFY_CMD` | `python scripts/run_medmnist.py` | `python scripts/run_flu_pipeline.py --pretrain-epochs 30 --finetune-epochs 10` |
| `METRIC_CMD` | `grep "Test ID Acc" run.log \| awk '{print $NF}'` | `grep "Test MAE" run.log \| awk '{print $NF}'` |
| `DIRECTION` | higher_is_better | lower_is_better |
| `EXPERT` | medical_expert | time_series_expert |
| `INPUT_SHAPE` | `(4, 1, 28, 28)` | `(4, 5, 1)` |
| `RAG_SCRIPT` | `python scripts/run_rag_search.py --task medmnist --k 5` | `python scripts/run_rag_search.py --task flu --k 5` |
| `JURY_CMD` | `python scripts/code_jury.py --task medmnist --env-dir MLAgentBench/benchmarks/medmnist/env --train-py MLAgentBench/benchmarks/medmnist/env/train.py --input-shape "(4, 1, 28, 28)" --expected-output-shape "(4, 3)"` | `python scripts/code_jury.py --task flu --env-dir env --train-py env/train.py --input-shape "(4, 5, 1)" --expected-output-shape "(4, 10, 1)"` |
| `LOG_DIR` | `experiments/loop-medmnist-{YYMMDD}-{HHMM}` | `experiments/loop-flu-{YYMMDD}-{HHMM}` |

## Setup

If Goal or Metric missing → use question tool:
  Q1: Task? medmnist or flu
  Q2: Goal? What metric to improve and why
  Q3: Iterations? default 5
  Q4: RAG? yes (default) or no

Precondition: verify git repo, clean tree, `{ENV_DIR}/train.py` and `{RUNNER}` exist.

## Establish Baseline (Iteration 0)

1. Backup env: `cp -r {ENV_DIR} experiments/loop-{task}-{YYMMDD}-{HHMM}/env.baseline/`
2. `git add -f experiments/loop-{task}-{YYMMDD}-{HHMM}/env.baseline/ && git commit -m "baseline: {task} env backup (iteration 0)"`
3. Run: `{VERIFY_CMD} > run.log 2>&1 && {METRIC_CMD}`
4. Record in `experiments/loop-{task}-{YYMMDD}-{HHMM}/results.tsv` as iteration 0

## Iteration Loop

For each iteration (1 to max_iterations):

### Phase 1: Research
```bash
RAG_SCRIPT --iteration N --query "<hypothesis-specific query>" --out {LOG_DIR}/iterations/iter-N-rag.json
```
If no RAG: still research via web/arxiv. Output: 2-4 sentence research brief.

### Phase 2: Plan
State ONE hypothesis, expected delta, mechanism, risk. Reference the research brief.

### Phase 3: Implement
Make ONE change to `{ENV_DIR}/train.py`. No eval/data changes. No package installs.

### Phase 4b: Code Jury
```bash
JURY_CMD --model-factory <create_model|DiffusionForecaster> --out {LOG_DIR}/iterations/iter-N-jury.json
```
If fail → fix code, re-run. On pass, construct commit message with JURY REASONING block.

### Phase 5: Commit
```bash
git add -f {ENV_DIR}/train.py && git commit -m "pipeline: {EXPERT} — {description}

JURY REASONING:
- Hypothesis: ...
- Mechanism: ...
- Expected delta: ...
- Risk assessment: ...
- Baseline comparison: ...
"
```

### Phase 6: Run
```bash
{VERIFY_CMD} > run.log 2>&1
METRIC=$( {METRIC_CMD} )
```

### Phase 7: Decide
- **keep**: metric improved → commit stays
- **discard**: metric worsened → `git revert HEAD --no-edit`
- **crash**: revert, diagnose tail -50 run.log, fix or skip

### Phase 8: Log
```bash
python scripts/make_iter_log.py \
  --task {task} --iteration N --status {keep|discard|crash} \
  --metric-name "<name>" --metric-direction {DIRECTION} \
  --metric-value $METRIC --baseline <prev_best> \
  --change-type "<type>" --change-file "{ENV_DIR}/train.py" \
  --diff-summary "<one-line>" \
  --hypothesis "..." --mechanism "..." --expected-delta "..." \
  --risk "low|medium|high" --baseline-compared-to "env.baseline/" \
  --elapsed-s <seconds> --memory-gb <nvidia-smi> --params <count> \
  --commit-hash $(git rev-parse HEAD) \
  --commit-message "<full message>" \
  --out {LOG_DIR}/iterations/iter-N.json

python scripts/append_results_tsv.py \
  --task {task} --results {LOG_DIR}/results.tsv \
  --iteration N --commit $(git rev-parse HEAD) \
  --test-mae $METRIC --params <count> --memory-gb <nvidia-smi> \
  --status {keep|discard|crash} --description "<one-line>"
```
Do NOT commit results.tsv.

### Phase 9: Adaptive RAG Refresh (when iteration % 10 == 0)
Score papers by kept/discarded ratio, prune bottom 30%, discover new papers via Tavily + arxiv, rebuild index.

### Phase 10: Research Reset (when iteration % 20 == 0 and metric plateaued)
Diagnose failures via evo-memory, force paradigm shift via research-ideation Critic persona, reset exploration strategy.

### Bounded Check
If current_iteration >= max_iterations → exit loop, print summary.

## Time Constraints
- Kill experiment at 10 min if not finished (treat as crash).

## Summary
Print: total iterations, kept/discarded, starting→final metric, improvement %, most consulted agents.
