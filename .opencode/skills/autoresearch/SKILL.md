---
name: autoresearch
description: "Autonomous iteration loop for chest X-ray OOD detection: modify, verify, keep/discard"
version: 2.2.0
---

# Autoresearch — Chest X-ray OOD Detection

15 real opencode slash commands defined in `.opencode/commands/autoresearch_*.md`.

## Commands

| Command | Purpose |
|---------|---------|
| `/autoresearch` | Iterate against metric: modify train.py → verify → keep/discard |
| `/autoresearch_plan` | Convert a goal into validated Scope, Metric, Verify config |
| `/autoresearch_debug` | Hunt bugs: hypothesize → test → falsify → repeat |
| `/autoresearch_fix` | Crush errors one-by-one until zero remain |
| `/autoresearch_security` | STRIDE + OWASP audit with red-team personas |
| `/autoresearch_ship` | Lock best model: final eval, export checkpoint |
| `/autoresearch_scenario` | Explore edge cases: imbalance, noise, domain shift |
| `/autoresearch_predict` | 5 expert personas debate before implementation |
| `/autoresearch_learn` | Extract lessons from past iterations |
| `/autoresearch_reason` | Adversarial debate with blind judge convergence |
| `/autoresearch_probe` | 8 personas interrogate requirements |
| `/autoresearch_improve` | Research SOTA methods, discover improvements |
| `/autoresearch_evals` | Analyze iteration results: trends, plateaus |
| `/autoresearch_regression` | Stability gate: baseline vs candidate |
| `/autoresearch_scientific` | Full loop + 8 specialized agents for chest X-ray OOD |

## Context

- Modify only: `MLAgentBench/benchmarks/medmnist/env/train.py`
- Run: `python scripts/run_medmnist.py > run.log 2>&1`
- Metrics: Test Accuracy (stdout) + OOD F1 (stdout)
- Log to: `experiments/results.tsv` (tab-separated)

## Pretrained Model Finetuning

Configurable via `Pretrained: yes/no` argument (default: no).

- **Sources**: HuggingFace Hub, PyTorch Hub, GitHub (CheXNet, COVID-Net)
- **Models**: torchvision ResNet/DenseNet/EfficientNet, biomedical vision transformers
- **Adaptation**: modify first conv (3→1 channel), resize 28→224, replace classifier head
- **Decision**: compare finetuned vs scratch; keep whichever performs better

## Subcommands Reference

Subcommands are self-contained in `.opencode/commands/` with full instructions.
