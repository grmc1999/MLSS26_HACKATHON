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

This project benchmarks OOD detection on MedMNIST chest X-ray data.

### Hardware
- 2× NVIDIA RTX PRO 6000 Blackwell (98GB VRAM each)
- GPU 0 = training, GPU 1 = available for LLM serving

### Data
- Train: PneumoniaMNIST (5,856 samples, 28×28 grayscale, binary: normal vs pneumonia)
- Test: ChestMNIST 3-class (normal, pneumonia, consolidation — consolidation is OOD)

### Experiment CLI
- `python scripts/run_medmnist.py --epochs N --lr X` trains and evaluates
- `python scripts/run_medmnist.py --list` shows past runs
- Default model: Simple CNN (3 conv layers, ~100K params)
- Metric: Test Accuracy + OOD F1

### Metrics
- Test Accuracy: range [0, 1], higher is better
- OOD F1: macro F1 across 3 classes (normal, pneumonia, consolidation)
- Baseline: ~22% test acc, ~0.15 OOD F1

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
1. **Route** the problem to the best specialized agent (CV, DL, Medical, Robustness, etc.)
2. **Consult** the agent for a hypothesis + code change proposal
3. **Execute** the experiment (modify `train.py` → commit → run → eval)
4. **Decide** keep (metric improved) or discard (metric worsened/crash)
5. **Log** results to TSV + dashboard
6. **Repeat** up to N iterations

### Agent Routing
Keywords map problems to agents automatically:
- `image`, `cnn`, `architecture`, `augment`, `classification` → **cv_expert**
- `loss`, `optimizer`, `train`, `learning rate`, `calibration` → **dl_expert**
- `pneumonia`, `chest`, `xray`, `medical`, `medmnist` → **medical_expert**
- `ood`, `robustness`, `detection`, `threshold`, `confidence` → **robustness_expert**
- `paper`, `sota`, `literature`, `arxiv` → **research_literature**
- `forget`, `checkpoint`, `ewc`, `version` → **continual_learning**
- `prompt`, `multi-agent`, `coordinate` → **llm_expert**
- default → **autoresearch**

### CLI
```bash
bash scripts/run_autoresearch_scientific.sh [agent] [iterations]
python -m MLAgentBench.agents.orchestrator --agent medical_expert --iterations 25
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
