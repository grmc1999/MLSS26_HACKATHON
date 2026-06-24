---
name: autoresearch_final
description: "Multi-expert pipeline: deterministic scripts for RAG / code jury / logging — research → plan → code → jury → commit → run → decide → log"
argument-hint: "[Goal: <text>] [Iterations: N] [RAG: yes|no] [--evals]"
---

EXECUTE IMMEDIATELY.

## Parse Arguments

Extract from $ARGUMENTS:
- `Goal:` — what to improve
- `Iterations:` or `--iterations` — default 5. "unlimited" for unbounded.
- `RAG:` — "yes" or "no" (default: yes)

### Task Configuration

| Setting | Value |
|---------|-------|
| `ENV_DIR` | `env/` |
| `TRAIN_PY` | `env/train.py` |
| `RUNNER` | `scripts/run_flu_pipeline.py` |
| `METRIC` | Test MAE |
| `METRIC_CMD` | `grep "Test MAE" run.log \| awk '{print $NF}'` |
| `DIRECTION` | lower_is_better |
| `VERIFY_CMD` | `python scripts/run_flu_pipeline.py --pretrain-epochs 30 --finetune-epochs 10` |
| `EXPERT` | time_series_expert (flu/ILI) |
| `INPUT_SHAPE` | `(4, 5, 1)` → `(4, 10, 1)` |
| `LOG_COLS` | test_mae, val_mae, params |
| `RAG_INDEX` | `index_output_flu/` (22 papers, MiniLM, from `literature_flu_md/` markdown files) |

## Setup (if required context missing)

If Goal or Metric missing → use question tool:
  Q1 (Goal): "What do you want to improve?" — depends on task
  Q2 (Iterations): "Iterations?" — default 5
  Q3 (RAG): "Use RAG literature search to guide experiments?" — Yes or No

## Precondition Checks

1. Verify git repo exists
2. Check clean working tree — warn if dirty
3. Verify `env/train.py` exists
4. Verify `scripts/run_flu_pipeline.py` exists

## Establish Baseline (Iteration 0)

1. **Backup the entire env directory** before any modification:
   ```bash
   BASELINE_DIR="experiments/loop-flu-{YYMMDD}-{HHMM}/env.baseline"
   mkdir -p "$BASELINE_DIR"
   cp -r env/* "$BASELINE_DIR/"
   git add -f "$BASELINE_DIR" && git commit -m "baseline: flu env backup (iteration 0)"
   ```
   The `env.baseline/` directory is **read-only for the entire pipeline** — it is never modified, and all experiment results are compared against it.

2. Run: `{VERIFY_CMD} > run.log 2>&1`
3. Extract: `{METRIC_CMD}`
4. Record as iteration 0 in `experiments/loop-flu-{YYMMDD}-{HHMM}/results.tsv`
5. Base metric from chosen Metric

## Iteration Loop

For each iteration (1 to max_iterations):

### HARD RULES (every iteration, no exceptions)

1. **Phase 1 Research is MANDATORY every iteration** — you must run RAG and/or arxiv search to ground the change. No blind guesses.
2. **Phase 8 JSON log must have ALL fields filled** — `jury_reasoning` with hypothesis, mechanism, expected_delta, risk. Never write `N/A`. If you don't know, research more.
3. **Phase 4b Code Jury is MANDATORY** — syntax + forward + loss + backward. No skips.
4. **One change per iteration** — never batch multiple changes.

### Phase 1: Research (MANDATORY every iteration — RAG + arxiv)

**Goal**: Understand the problem, find relevant methods from literature. **Do not skip this phase on any iteration. Every change must be grounded in literature.**

**Step 1 — Problem Analysis**: Before searching literature, identify what makes this problem hard:
- **flu**: domain shift between US CDC ILINet (training) and WHO FluID (test on France, Mexico, Australia, South Africa). Different countries have different flu seasons, reporting standards, and healthcare systems. Australia is in the southern hemisphere — opposite flu season. A model that memorizes US patterns will fail globally.
- Use this analysis to guide what kind of solution is needed (regularization? domain adaptation? seasonal features? calibration?).

**Search strategy** (run ALL of these every iteration, in order):

1. **Local RAG** (if enabled): ALWAYS run first.
   ```bash
   python scripts/run_rag_search.py \
     --task flu --iteration N \
     --query "<hypothesis-specific query>" \
     --k 5 \
     --out experiments/loop-flu-{YYMMDD}-{HHMM}/iterations/iter-N-rag.json
   ```

2. **Tavily web search** (tavily-search skill): ALWAYS run. Use query specific to the current iteration's hypothesis (not a generic repeat from iter 1). `tvly search <query> --depth=basic --max-results=5 --include-answer`. Use task-specific keywords: ILI forecasting, cross-country generalization, time series domain adaptation.

3. **paper-navigator** (EvoSkill): ALWAYS run for at least one source. Use `python3 skills/paper-navigator/scripts/scholar_search.py <technique>` to find papers with rubric-based relevance scores. If S2_API_KEY is available, also use `citation_traverse.py` for citation chains.

4. **research-ideation** (EvoSkill): run if literature search alone didn't yield a clear direction. Loads prior ideation memory (evo-memory) to avoid repeating dead ends. Run the multi-track ideation pipeline: generate ideas across 3 personas (Innovator/Pragmatist/Critic), refine with 5 evolution strategies, rank via ELO tournament.

**Expert synthesis**:
- Load `experiment-craft` skill for diagnosing what technique families have been tried and what's left unexplored
- Load `evo-memory` to check prior iterations' kept/discarded patterns before proposing new changes
- Synthesize findings into a research brief (2-4 sentences) identifying the problem type + citations for relevant methods

### Phase 2: Plan (research-ideation + experiment-pipeline)

**Goal**: Convert research into a concrete experiment plan.

- **research-ideation** (EvoSkill): synthesize research brief into a clear hypothesis using the idea refinement pipeline. Generate counter-arguments via the Critic persona to stress-test the hypothesis.
- **experiment-pipeline** (EvoSkill): define the experiment — what ONE change to `env/train.py`, which stage it belongs to (1: baseline reproduction, 2: hyperparameter tuning, 3: proposed method, 4: ablation), attempt budget, expected outcome, fallback plan.
- Check `evo-memory` history — is this truly untried? Load prior ideation memory (M_I) and experimentation memory (M_E) to verify.
- If internet research needed: use `tvly search` (tavily-search) to check for recent work on the proposed approach.
- Output: experiment plan with hypothesis + expected delta + stage assignment

### Phase 3: Implement (paper-navigator + experiment-craft)

**Goal**: Write the code change.

- **Route by change type**:
  - Architecture/preprocessing/attention → load `paper-navigator` to search for relevant implementations. Use `python3 scholar_search.py <technique>` to find papers, then `python3 fetch_paper.py` to get details.
  - Loss/optimizer/scheduler/regularization → use `tvly search` with `--include-answer` to find best practices and code patterns.
  - Statistical/mathematical reasoning → consult training code and paper implementations via `paper-navigator`.
- **experiment-craft** (EvoSkill): follow the 5-step diagnostic for implementation. If unsure, test the change in isolation (Step 2: Find a Working Version) before integrating.
- Make ONE focused change to `env/train.py`
- Allowed: model architecture, optimizer, hyperparams, loss, augmentation, calibration
- NOT allowed: modify eval code, data loading code, install packages
- Output: modified train.py

### Phase 4: Review (experiment-craft + evo-memory)

**Goal**: Validate the change before running.

- **experiment-craft** (EvoSkill): assess impact on the target metric using the 5-step diagnostic. Check for known failure modes — overfitting risk, calibration impact, hyperparameter sensitivity.
- **evo-memory** (EvoSkill): is this safe to keep? Will it cause forgetting? Check experimentation memory (M_E) for similar approaches that failed before. What's the rollback plan?
- If further validation needed: use `tvly search` (tavily-search) to check if this approach has known failure cases reported online.
- Output: go/no-go recommendation + risk assessment

### Phase 4b: Code Jury (syntax + shape + gradient + reasoning)

**Goal**: Verify code correctness before committing and explicitly document the scientific reasoning behind the change.

```bash
python scripts/code_jury.py \
  --task flu \
  --env-dir env \
  --train-py env/train.py \
  --input-shape "(4, 5, 1)" \
  --expected-output-shape "(4, 10, 1)" \
  --out experiments/loop-flu-{YYMMDD}-{HHMM}/iterations/iter-N-jury.json
```

If any check fails → **STOP**, diagnose the error, fix the code, re-run jury.
If all pass → **PASS**, safe to commit.

**Before passing, the Jury must log the following reasoning explicitly in the commit message**:

```
JURY REASONING:
- Hypothesis: [What did you expect this change to do, and why? Reference the research or theory that motivated it.]
- Mechanism: [How does the change affect the model's behavior? E.g., "adds a regularization term that penalizes extreme weights, reducing overfitting to US-specific patterns."]
- Expected delta: [Quantitative prediction: "expect Test MAE to decrease by 0.01"]
- Risk assessment: [low / medium / high] — what could go wrong and why?
- Baseline comparison: [Which baseline config from env.baseline/ is this compared against?]
```

The commit message format becomes:
```
pipeline: {agent} — {short description}

JURY REASONING:
- Hypothesis: ...
- Mechanism: ...
- Expected delta: ...
- Risk assessment: ...
- Baseline comparison: ...
```

### Phase 5: Commit

- `git add -f env/train.py && git commit -m "pipeline: {agent} — {description}"`

### Phase 6: Run

- `{VERIFY_CMD} > run.log 2>&1`
- Extract: `{METRIC_CMD}`
- Record peak memory from `nvidia-smi`

### Phase 7: Decide

- **keep** — metric improved → commit stays
- **discard** — metric worsened → `git revert HEAD --no-edit`; restore env/train.py from HEAD
- **crash** — run failed/crashed → revert; read tail -50 run.log for stack trace; attempt fix or skip

### Phase 8: Log

Use the deterministic logging scripts:

```bash
python scripts/make_iter_log.py \
  --task flu --iteration N --status {keep|discard|crash} \
  --metric-name "Test MAE" --metric-direction lower_is_better \
  --metric-value $METRIC --baseline <prev_best> \
  --change-type "<architecture|loss|optimizer|hyperparameter|augmentation>" \
  --change-file "env/train.py" \
  --diff-summary "<one-line description of what changed>" \
  --hypothesis "..." --mechanism "..." --expected-delta "..." \
  --risk "low|medium|high" --baseline-compared-to "env.baseline/" \
  --elapsed-s <seconds> --memory-gb <nvidia-smi> --params <count> \
  --commit-hash $(git rev-parse HEAD) \
  --commit-message "<full commit message>" \
  --out experiments/loop-flu-{YYMMDD}-{HHMM}/iterations/iter-N.json

python scripts/append_results_tsv.py \
  --task flu \
  --results experiments/loop-flu-{YYMMDD}-{HHMM}/results.tsv \
  --iteration N --commit $(git rev-parse HEAD) \
  --test-mae $METRIC --params <count> --memory-gb <nvidia-smi> \
  --status {keep|discard|crash} --description "<one-line>"
```

Do NOT commit results.tsv.

The JSON log is machine-readable and can be processed by evo-memory, aggregated across runs, or fed into the dashboard's visualization pipeline. The TSV stays as the human-readable summary.

The logging scripts validate that ALL fields are filled — never write `N/A` or empty strings. If you lack the information to fill a field, you did not do enough research in Phase 1 — go back and research.

### Phase 9: Adaptive RAG Refresh (every 10 iterations)
When iteration % 10 == 0:

1. **Score existing papers**: for each paper in the literature corpus, count times consulted (from RAG logs), times a suggested change was KEPT vs DISCARDED. Score = kept_count - discarded_count.
2. **Prune**: remove bottom 30% lowest-scoring papers from corpus and index.
3. **Discover new papers** (use Tavily + paper-navigator):
   - Run `tvly research <query>` (tavily-research) with keywords from successful iterations to get a multi-source cited report on recent work
   - Run `tvly search <query> --depth=advanced --include-domains=arxiv.org,github.com,paperswithcode.com --max-results=10` (tavily-search) for specific papers and code repos
   - Use `python3 paper-navigator/scripts/arxiv_monitor.py` and `paper-navigator/scripts/trending.py` for academic paper discovery by subcategory
   - Download new PDFs to `literature_flu/`
4. **Ingest**: convert new PDFs to markdown (`literature_flu_md/`), rebuild FAISS index (`index_output_flu/`).
5. Replace old index with refreshed one.

### Phase 10: Research Reset (every 20 iterations)
When iteration % 20 == 0 AND metric hasn't improved in last 10 iterations:

1. **Diagnose plateau**: load `evo-memory` (IVE: Idea Validation Evolution) to classify each discarded experiment as implementation failure vs fundamental direction failure. Identify what technique family has been exhausted (loss, architecture, data, OOD scoring).
2. **Force paradigm shift**: use `research-ideation` with the Critic persona to stress-test the current paradigm and generate cross-domain ideas. Use `tvly search` (tavily-search) for SOTA in adjacent fields.
3. **Run IDE** (Idea Direction Evolution): update ideation memory (M_I) with the ranked directions, marking exhausted paths as unsuccessful.
4. **Reset short-term memory**: treat as new research phase. Commit history and best metric preserved, exploration strategy resets.
5. Prevents local optima — like epsilon-greedy in RL, when exploitation plateaus.

### Eval Checkpoint
If --evals: check if current_iteration % interval == 0 → run checkpoint analysis.

### Bounded Check
If bounded: current_iteration >= max_iterations → exit loop, print summary.

## Time Constraints
- ~5 min per experiment (30 pretrain + 10 finetune epochs). Kill at 10 min.
- Treat timeout as crash.

## Summary
Print: total iterations, kept/discarded counts, starting → final metric, improvement %, most consulted agents.
