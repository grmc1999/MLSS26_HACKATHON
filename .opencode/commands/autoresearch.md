---
name: autoresearch
description: "Autonomous iteration loop: modify, verify, keep/discard against any metric"
argument-hint: "[Goal: <text>] [Metric: <text>] [Verify: <cmd>] [Iterations: N] [--evals]"
---

EXECUTE IMMEDIATELY — do not deliberate before reading this protocol.

## Parse Arguments

Extract from $ARGUMENTS:
- `Goal:` — what to improve
- `Metric:` — what to measure (e.g., Dice Score)
- `Direction:` — higher_is_better (default) or lower_is_better
- `Verify:` — shell command that outputs a number (e.g., `python scripts/run_exp.py --epochs 20 | grep "Test Dice" | awk '{print $NF}'`)
- `Iterations:` — integer N (default: 25)
- `--evals` — enable mid-loop checkpoints

## Setup (if context missing)

If Goal, Metric, or Verify missing → ask once:
- Goal: "What do you want to improve?"
- Metric: "How to measure?" (Dice Score, loss, parameters, speed)
- Verify: "Shell command that outputs a number for the metric"
- Iterations: "How many iterations?" (default 25)

## Precondition Checks

1. Verify git repo: `git rev-parse --git-dir`
2. Check clean working tree: `git status --porcelain` — warn if dirty
3. Activate venv: `source .venv/bin/activate`
4. Verify GPU: `python -c "import torch; print(torch.cuda.get_device_name(0))"`

## Establish Baseline (Iteration 0)

1. Run `python scripts/run_exp.py --list` to see prior results
2. Run Verify command → extract numeric metric
3. Record in TSV: `experiments/loop-{YYMMDD}-{HHMM}/results.tsv`
4. Header: `# metric_direction: {direction}\niteration\ttimestamp\tcommit\tmetric\tdelta\tstatus\tdescription`

## Iteration Loop

For each iteration (1 to max_iterations):

### Phase 1: Review
- Read last results from TSV
- Run `git log --oneline -10` — what worked/failed last
- If last iteration was "keep" → `git diff HEAD~1` to see what improved

### Phase 2: Modify
- Make ONE focused change to the training code:
  - `MLAgentBench/benchmarks/identify-contrails/env/train.py` — model architecture, loss, aug
  - `scripts/run_exp.py` — experiment pipeline
- Change must be atomic

### Phase 3: Commit
- `git add` changed files
- `git commit -m "experiment: {description}"`
- Record commit SHA

### Phase 4: Verify
- Run Verify command → extract new metric
- Calculate delta from previous iteration

### Phase 5: Decide
- **keep** — metric improved → commit stays
- **discard** — metric worsened → `git revert HEAD --no-edit`
- **crash** — verify failed → `git revert HEAD --no-edit`
- **no-op** — no change made

### Phase 6: Log
Append row to TSV: iteration, timestamp, commit/-, metric, delta, status, description

### Bounded Check
If bounded: current_iteration >= max_iterations → exit loop, print summary.

## Summary (after loop ends)

Print: total iterations, kept/discarded, starting → final metric, improvement %, top changes.

## Chain Handoff

Write `experiments/loop-{YYMMDD}-{HHMM}/handoff.json` with config, results, status.
