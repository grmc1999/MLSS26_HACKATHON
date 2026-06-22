---
name: autoresearch_scientific
description: "Scientific AI AutoResearch: merge autoresearch loop with 8 specialized domain agents for ML experimentation"
version: 2.2.0
---

# Autoresearch Scientific — Autonomous Scientific AI Experimentation

Merges **Karpathy's autoresearch loop** with **8 specialized scientific AI agents**. At each iteration, the orchestrator can consult domain experts (CV, DL, Satellite, Physics, etc.) before deciding what to modify and whether to keep or discard.

## Safety Invariants
- Bounded by default. Override with `Iterations: unlimited`.
- All results logged to `experiments/loop-*/results.tsv` and `experiments/runs.jsonl`.
- Chain handoff via `handoff.json`.
- Reverts on crash or metric regression. Never breaks the codebase.

## Scientific AI Agents

| Agent | Role | When to Consult |
|-------|------|----------------|
| **Research Literature** | Paper search, SOTA methods | Before trying new architecture |
| **CV Expert** | CNN/transformer architecture, augmentation | When modifying model or data pipeline |
| **DL Expert** | Loss functions, optimizers, schedulers | When tuning training hyperparameters |
| **Satellite Expert** | RESPNET/ILINET, seasonal patterns, ERA5 | When working with epidemiological data |
| **Physics Expert** | PINNs, advection, CSI metrics | When adding physical constraints |
| **Continual Learning** | EWC, replay, checkpoints | When managing model versions across iterations |
| **LLM Expert** | Multi-agent coordination | When synthesizing advice from multiple agents |
| **AutoResearch** | Experiment planning, strategy | Default — orchestrates the loop itself |

## Subcommands

| Command | Purpose | Default Iterations |
|---------|---------|-------------------|
| `/autoresearch_scientific` | Full autonomy loop with agent consultation | 25 |
| `/autoresearch_scientific_plan` | Consult agents → generate next hypothesis | N/A |
| `/autoresearch_scientific_run` | Run one iteration: modify → commit → run → eval → keep/discard | 1 |
| `/autoresearch_scientific_fix` | Diagnose crash with agent help, repair code | 15 |
| `/autoresearch_scientific_analyze` | Deep analysis with statistical rigor | N/A |
| `/autoresearch_scientific_ship` | Lock best model: final eval, export, submission | N/A |
| `/autoresearch_scientific_evals` | Full metric suite: MAE, RMSE, quantile loss | N/A |
| `/autoresearch_scientific_probe` | Model internals: activations, gradients, attention | N/A |
| `/autoresearch_scientific_improve` | Targeted improvement on weakest cases | 10 |
| `/autoresearch_scientific_debug` | Interactive debugging with agent support | 10 |
| `/autoresearch_scientific_learn` | Extract cross-iteration lessons | N/A |
| `/autoresearch_scientific_reason` | Chain-of-thought trajectory analysis | N/A |
| `/autoresearch_scientific_scenario` | What-if across weather, time, geography | 10 |
| `/autoresearch_scientific_regression` | Verify no regression vs best checkpoint | N/A |

## Context (MLSS26_HACKATHON)

### Hardware
- 2× NVIDIA RTX PRO 6000 Blackwell (98GB VRAM each)
- GPU 0 = training

### Experiment CLI
- `python scripts/run_exp.py --epochs N --lr X --hidden-dim N --batch N`
- `python scripts/run_exp.py --list` — past runs
- Metric: **Test MAE** (lower is better)

### Data
- CDC ILINET surveillance network (weekly %ILI rates)
- RESPNET respiratory pathogen data
- 5 past epiweeks → 10 future epiweeks forecast
- Multi-region (HHS regions) spatiotemporal data

### Current Best
- Test MAE: **TBD** (baseline: seasonal naive)

## Orchestrator Loop

```
LOOP FOREVER (bounded):
  1. Consult routing: analyze state → pick best agent for next step
  2. Agent proposes hypothesis + code change
  3. Modify train.py with one focused change
  4. git commit
  5. Run: python scripts/run_exp.py --epochs 100
  6. Extract metric: grep "Test MAE" from output
  7. If improved → KEEP (advance branch, update best)
  8. If worse/crash → DISCARD (git revert, restore worktree)
  9. Log iteration to TSV + JSONL
  10. Check convergence: plateaud >5? Ceiling >50? Predicate met?
  11. If converged → handoff; else → repeat
```

### Decision Logic
```
improvement > 0         → KEEP   (advance branch)
improvement <= 0        → DISCARD (git revert)
crash (OOM/bug/timeout) → DISCARD (git revert, read trace)
5+ flat iterations      → CONVERGED (handoff)
50+ total iterations    → CEILING (handoff)
```

## Universal Flags

| Flag | Purpose |
|------|---------|
| `Iterations: N` | Override iteration count |
| `--evals` | Mid-loop checkpoints + final summary |
| `--agent <role>` | Force specific primary agent |
| `--chain <target>` | Sequential handoff after completion |
| `--dry-run` | Print config, no execution |

## Handoff

On completion, writes `experiments/loop-{YYMMDD}-{HHMM}/handoff.json`:

```json
{
  "config": { "goal", "metric", "direction", "verify", "iterations" },
  "results": { "total", "kept", "discarded", "start_metric", "final_metric", "best_metric" },
  "iterations": [ { "iteration", "metric", "delta", "status", "description" } ],
  "agent_log": [ { "agent", "action", "timestamp" } ]
}
```
