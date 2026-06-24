---
name: autoresearch_pipeline
description: "Multi-expert pipeline: 8 agents + code jury per iteration — research → plan → code → jury → review → commit → run → decide → log"
argument-hint: "[Goal: <text>] [Task: flu] [Metric: ...] [Iterations: N] [RAG: yes|no] [--evals]"
---

EXECUTE IMMEDIATELY.

## Parse Arguments

Extract from $ARGUMENTS:
- `Goal:` — what to improve
- `Task:` — `flu`
- `Metric:` — Test MAE
- `Iterations:` or `--iterations` — default 5. "unlimited" for unbounded.
- `RAG:` — "yes" or "no" (default: yes)
- `--evals` — enable mid-loop checkpoints
- `--evals-interval N` — checkpoint frequency override

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
3. Verify `{ENV_DIR}/train.py` exists
4. Verify `{RUNNER}` exists

## Establish Baseline (Iteration 0)

## Establish Baseline (Iteration 0)

1. **Backup the entire env directory** before any modification:
   ```bash
   BASELINE_DIR="experiments/loop-{task}-{YYMMDD}-{HHMM}/env.baseline"
   mkdir -p "$BASELINE_DIR"
   cp -r {ENV_DIR}/* "$BASELINE_DIR/"
   git add -f "$BASELINE_DIR" && git commit -m "baseline: {task} env backup (iteration 0)"
   ```
   The `env.baseline/` directory is **read-only for the entire pipeline** — it is never modified, and all experiment results are compared against it. It serves as the ground truth reference.

2. Run: `{VERIFY_CMD} > run.log 2>&1`
3. Extract: `{METRIC_CMD}`
4. Record as iteration 0 in `experiments/loop-{task}-{YYMMDD}-{HHMM}/results.tsv`
5. Base metric from chosen Metric

## Iteration Loop (Multi-Expert Pipeline)

For each iteration (1 to max_iterations):

### HARD RULES (every iteration, no exceptions)

1. **Phase 1 Research is MANDATORY every iteration** — you must run RAG and/or arxiv search to ground the change. No blind guesses.
2. **Phase 8 JSON log must have ALL fields filled** — `jury_reasoning` with hypothesis, mechanism, expected_delta, risk. Never write `N/A`. If you don't know, research more.
3. **Phase 4b Code Jury is MANDATORY** — syntax + forward + loss + backward. No skips.
4. **One change per iteration** — never batch multiple changes.

### Phase 1: Research (MANDATORY every iteration — RAG + arxiv)

**Goal**: Understand the problem, find relevant methods from literature. **Do not skip this phase on any iteration. Every change must be grounded in literature.**

**Step 1 — Problem Analysis**: Identify what makes this problem hard:
- **flu**: domain shift between US CDC ILINet (training) and WHO FluID (test on France, Mexico, Australia, South Africa). Different countries have different flu seasons, reporting standards, and healthcare systems. Australia is in the southern hemisphere — opposite flu season. A model that memorizes US patterns will fail globally.
- Use this analysis to guide what kind of solution is needed (regularization? domain adaptation? seasonal features? calibration?).

**Search strategy** (run ALL of these every iteration, in order):

1. **Local RAG** (if enabled): ALWAYS run first.
   - **flu**: run `search_flu_context_rag(query, k=5)`. Uses FAISS vector search (`index_output_flu/`) + FalkorDB knowledge graph for relational queries.

2. **Tavily web search** (tavily-search skill): ALWAYS run. Use query specific to the current iteration's hypothesis (not a generic repeat from iter 1). `tvly search <query> --depth=basic --max-results=5 --include-answer`. Use keywords: ILI forecasting, cross-country generalization, time series domain adaptation.

3. **paper-navigator** (EvoSkill): ALWAYS run for at least one source. Use `python3 skills/paper-navigator/scripts/scholar_search.py <technique>` to find papers with rubric-based relevance scores. If S2_API_KEY is available, also use `citation_traverse.py` for citation chains.

4. **research-ideation** (EvoSkill): run if literature search alone didn't yield a clear direction. Loads prior ideation memory (evo-memory) to avoid repeating dead ends. Run the multi-track ideation pipeline: generate ideas across 3 personas (Innovator/Pragmatist/Critic), refine with 5 evolution strategies, rank via ELO tournament.

**Expert synthesis**:
- Load `experiment-craft` skill for diagnosing what technique families have been tried and what's left unexplored
- Load `evo-memory` to check prior iterations' kept/discarded patterns before proposing new changes
- Synthesize findings into a research brief (2-4 sentences) identifying the problem type + citations for relevant methods

### Phase 2: Plan (research-ideation + experiment-pipeline)

**Goal**: Convert research into a concrete experiment plan.

- **research-ideation** (EvoSkill): synthesize research brief into a clear hypothesis using the idea refinement pipeline. Generate counter-arguments via the Critic persona to stress-test the hypothesis.
- **experiment-pipeline** (EvoSkill): define the experiment — what ONE change to {TRAIN_PY}, which stage it belongs to (1: baseline reproduction, 2: hyperparameter tuning, 3: proposed method, 4: ablation), attempt budget, expected outcome, fallback plan.
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
- Make ONE focused change to `{TRAIN_PY}`
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

Run these checks in `{ENV_DIR}`:

1. **Syntax check**: `python -c "import py_compile; py_compile.compile('train.py', doraise=True)"`
2. **Model instantiation**: import and create the model
3. **Forward pass**: feed a random tensor of shape `{INPUT_SHAPE}` and verify output shape
4. **Loss**: compute loss, verify finite
5. **Backward pass**: call `.backward()`, verify gradients non-None

If any check fails → **STOP**, diagnose the error, fix the code, re-run jury.
If all pass → **PASS**, safe to commit.

**Before passing, the Jury must log the following reasoning explicitly in the commit message** (this is a scientific record, not just a code change):

```
JURY REASONING:
- Hypothesis: [What did you expect this change to do, and why? Reference the research or theory that motivated it.]
- Mechanism: [How does the change affect the model's behavior? E.g., "adds a regularization term that penalizes extreme weights, reducing overfitting to US-specific patterns."]
- Expected delta: [Quantitative prediction: "expect OOD F1 to improve by 0.02–0.05" or "expect Test MAE to decrease by 0.01"]
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

This ensures every experiment has a documented scientific rationale that can be reviewed, challenged, and learned from in future iterations.

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
  iteration, commit, test_acc/mae, ood_f1/val_mae, val_acc, test_acc_id, memory_gb, status, description
DO NOT commit results.tsv.

**MANDATORY: Generate a per-experiment JSON log** in `experiments/loop-{task}-{YYMMDD}-{HHMM}/iterations/` named `iter-{N}.json` with the full experiment record. ALL fields must be filled — never write `N/A` or empty strings in `jury_reasoning`, `change`, or `metric`. If you lack the information to fill a field, you did not do enough research in Phase 1 — go back and research.

```json
{
  "iteration": 1,
  "task": "flu",
  "commit": "abc1234",
  "timestamp": "2026-06-23T21:30:00Z",
  "status": "keep",
  "metric": {
    "name": "Test MAE",
    "direction": "lower_is_better",
    "value": 0.5823,
    "baseline": 0.5897
  },
  "change": {
    "type": "architecture / loss / optimizer / hyperparameter / augmentation",
    "file": "env/train.py",
    "diff_summary": "Added dropout 0.2 between GRU encoder layers"
  },
  "jury_reasoning": {
    "hypothesis": "Dropout will reduce overfitting to US-specific patterns and improve cross-country generalization.",
    "mechanism": "nn.Dropout(0.2) applied to encoder hidden states before decoding, forcing the model to learn redundant representations.",
    "expected_delta": "-0.01 to -0.03 Test MAE",
    "risk": "low — dropout only active during training",
    "baseline_compared_to": "env.baseline/"
  },
  "resources": {
    "elapsed_s": 42.3,
    "memory_gb": 1.2,
    "params": 298881
  },
  "git": {
    "commit_hash": "abc1234",
    "commit_message": "pipeline: time_series_expert — Add dropout 0.2 to GRUSeq2Seq"
  }
}
```

This JSON is machine-readable and can be processed by evo-memory, aggregated across runs, or fed into the dashboard's visualization pipeline. The TSV stays as the human-readable summary.


### Phase 9: Adaptive RAG Refresh (every 10 iterations)
When iteration % 10 == 0:

1. **Score existing papers**: for each paper in the literature corpus, count times consulted (from RAG logs), times a suggested change was KEPT vs DISCARDED. Score = kept_count - discarded_count.
2. **Prune**: remove bottom 30% lowest-scoring papers from corpus and index.
3. **Discover new papers** (use Tavily + paper-navigator):
   - Run `tvly research <query>` (tavily-research) with keywords from successful iterations to get a multi-source cited report on recent work
   - Run `tvly search <query> --depth=advanced --include-domains=arxiv.org,github.com,paperswithcode.com --max-results=10` (tavily-search) for specific papers and code repos
   - Use `python3 paper-navigator/scripts/arxiv_monitor.py` and `paper-navigator/scripts/trending.py` for academic paper discovery by subcategory
   - Download new PDFs to `literature/` or `literature_flu/`
4. **Ingest**: convert new PDFs to markdown (`literature_md/` or `literature_flu_md/`), rebuild FAISS index (`index_output/` or `index_output_flu/`).
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
