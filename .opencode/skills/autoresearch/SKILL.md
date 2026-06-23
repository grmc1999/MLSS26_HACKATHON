---
name: autoresearch
description: "Autonomous iteration loop for chest X-ray OOD detection: modify, verify, keep/discard"
version: 2.2.0
---

# Autoresearch — Chest X-ray OOD Detection

16 real opencode slash commands defined in `.opencode/commands/autoresearch_*.md`.

## Commands

| Command | Purpose |
|---------|---------|
| `/autoresearch` | Iterate against metric: modify train.py → verify → keep/discard |
| `/autoresearch_pipeline` | Multi-expert pipeline: 8 agents in sequence per iteration — research → plan → code → review |

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
