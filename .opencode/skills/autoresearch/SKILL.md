---
name: autoresearch
description: "Autonomous iteration loop: modify, verify, keep/discard against any metric"
version: 2.2.0
---

# Autoresearch — Autonomous Goal-directed Iteration

## Safety Invariants (all subcommands)
- Bounded by default. Override with `Iterations: unlimited`.
- All results logged to `experiments/` directory.
- Chain handoff via `handoff.json`. Evals reads `experiments/runs.jsonl`.

## Task-Specific Context (MLSS26_HACKATHON)

This project benchmarks contrail detection on GOES-16 satellite imagery.

### Hardware
- 2× NVIDIA RTX PRO 6000 Blackwell (98GB VRAM each)
- GPU 0 = training, GPU 1 = available for LLM serving

### Data
- Synthetic: 5 train + 5 test samples (256×256, 9 bands, 8 time steps)
- Real Kaggle dataset (60GB) available for download

### Experiment CLI
- `python scripts/run_exp.py --epochs N --lr X --base-ch N` trains and evaluates
- `python scripts/run_exp.py --list` shows past runs
- Default model: U-Net (depth 4, base_ch 32, ~835K params)
- Metric: Dice Score (higher is better)

### Metrics
- Dice Score: range [0, 1], higher is better
- Loss: Cross-Entropy with class weights [0.57, 4.17]
- Training speed: ~25 it/s on Blackwell (real data: ~10 it/s)

## Subcommands

| Command | Does | Default Iterations |
|---|---|---|
| `/autoresearch` | Iterate against a metric: modify → verify → keep/discard | 25 |
| `/autoresearch_plan` | Convert a goal into validated Scope, Metric, Verify config | N/A |
| `/autoresearch_debug` | Hunt bugs: hypothesize → test → falsify → repeat | 15 |
| `/autoresearch_fix` | Crush errors one-by-one until zero remain | 20 |
| `/autoresearch_security` | STRIDE + OWASP audit with red-team personas | 15 |
| `/autoresearch_ship` | Ship through 8 phases: checklist → dry-run → deploy → verify | N/A |
| `/autoresearch_scenario` | Generate edge cases across 12 dimensions | 20 |
| `/autoresearch_predict` | 5 expert personas debate before implementation | N/A |
| `/autoresearch_learn` | Scout codebase → generate docs or wiki → validate → fix loop | 10 |
| `/autoresearch_reason` | Adversarial debate with blind judges until convergence | 8 |
| `/autoresearch_probe` | 8 personas interrogate requirements until saturation | 15 |
| `/autoresearch_improve` | Research ICP challenges, discover improvements, generate PRDs | 15 |
| `/autoresearch_evals` | Analyze iteration results: trends, plateaus, regressions | N/A |
| `/autoresearch_regression` | Regression stability gate: baseline vs candidate | N/A |
| `/autoresearch_scientific` | Scientific AI mode: merge autoresearch loop with 8 specialized agents | 25 |

## Universal Flags

| Flag | Applies To | Purpose |
|---|---|---|
| `Iterations: N` | All looping | Set iteration count |
| `--evals` | All looping | Mid-loop checkpoints + final summary |
| `--chain <targets>` | All | Sequential handoff after completion |
| `--dry-run` | Orchestrator | Print derived config, no execution |

## Scientific AI Mode (subcommand: `/autoresearch_scientific`)

When invoked, this subcommand activates the full Scientific AI pipeline:
1. **Route** the problem to the best specialized agent (CV, DL, Satellite, Physics, etc.)
2. **Consult** the agent for a hypothesis + code change proposal
3. **Execute** the experiment (modify `train.py` → commit → run → eval)
4. **Decide** keep (metric improved) or discard (metric worsened/crash)
5. **Log** results to TSV + dashboard
6. **Repeat** up to N iterations

### Agent Routing
Keywords map problems to agents automatically:
- `image`, `unet`, `augment`, `architecture` → **cv_expert**
- `loss`, `optimizer`, `train`, `learning rate` → **dl_expert**
- `satellite`, `band`, `goes`, `contrail`, `era5` → **satellite_expert**
- `physics`, `advection`, `csi`, `pde` → **physics_expert**
- `paper`, `sota`, `literature`, `arxiv` → **research_literature**
- `forget`, `checkpoint`, `ewc`, `version` → **continual_learning**
- `prompt`, `multi-agent`, `coordinate` → **llm_expert**
- default → **autoresearch**

### CLI
```bash
bash scripts/run_autoresearch_scientific.sh [agent] [iterations]
python -m MLAgentBench.agents.orchestrator --agent cv_expert --iterations 25
```

---

## Orchestrator Loop

```
1. Analyze goal → classify into archetype
2. Derive success predicate (shell command + expected output)
3. Confirm with user
4. Round-0 dry-run predicate
5. Loop until predicate satisfied:
   a. Assess state
   b. Pick next subcommand
   c. Run it
   d. Record outcome
   e. Check convergence
6. Ship gate (if applicable) or CONVERGED
```

Stop conditions: predicate met, plateau (>5 flat units), ceiling (>50 cycles), blocked.
