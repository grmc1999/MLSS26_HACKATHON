---
name: autoresearch_scientific
description: "Scientific AI mode: autoresearch loop with 8 specialized agents for chest X-ray OOD detection"
argument-hint: "[Goal: <text>] [Agent: <role>] [Iterations: N] [Pretrained: yes|no] [--evals]"
---

EXECUTE IMMEDIATELY.

## Parse Arguments

Extract from $ARGUMENTS:
- `Goal:` — what to improve (Test Accuracy, OOD F1, calibration, architecture)
- `Agent:` or `--agent` — primary agent role (default: autoresearch)
- `Iterations:` or `--iterations` — default 25. "unlimited" for unbounded.
- `Pretrained:` — "yes" to search and finetune pretrained models, "no" to train from scratch (default: no)
- `--evals` — enable mid-loop checkpoints
- `--evals-interval N` — checkpoint frequency override

## Agent Roles

| Role | Strengths | Use When |
|------|-----------|----------|
| `autoresearch` | Experiment planning, iteration strategy | Default, general improvement |
| `cv_expert` | CNN architecture, data augmentation, OOD scoring | Changing model, adding preprocessing |
| `dl_expert` | Loss functions, optimizers, calibration, training loop | Tuning hyperparams, adding calibration |
| `medical_expert` | Chest X-ray knowledge, MedMNIST, pneumonia patterns | Data understanding, domain knowledge |
| `robustness_expert` | OOD theory, uncertainty, confidence calibration | Improving OOD detection specifically |
| `research_literature` | Paper search, SOTA methods, literature review | Exploring new methods from papers |
| `continual_learning` | Anti-forgetting, checkpoint versioning, replay | Managing model versions across iterations |
| `llm_expert` | Multi-agent coordination, prompt design | Complex multi-objective optimization |

## Setup (if Goal missing)

question (single batch):
  Q1 (Goal): "What do you want to improve?" — Test Accuracy, OOD F1, both, architecture
  Q2 (Agent): "Primary agent?" — autoresearch (default), cv_expert, dl_expert, medical_expert, robustness_expert
  Q3 (Iterations): "Iterations?" — default 25
  Q4 (Pretrained): "Start from scratch or finetune a pretrained model?" — Scratch (default) or Pretrained

## Routing Logic

If Agent not specified, route based on Goal keywords:
- `cnn`, `architecture`, `augment`, `image` → **cv_expert**
- `loss`, `optimizer`, `train`, `learning rate`, `calibration` → **dl_expert**
- `pneumonia`, `chest`, `xray`, `medical`, `medmnist` → **medical_expert**
- `ood`, `robustness`, `detection`, `threshold`, `confidence` → **robustness_expert**
- `paper`, `sota`, `literature`, `arxiv` → **research_literature**
- `forget`, `checkpoint`, `ewc`, `version` → **continual_learning**
- `prompt`, `multi-agent`, `coordinate` → **llm_expert**
- default → **autoresearch**

## Precondition Checks

1. Verify git repo, clean working tree
2. Verify train.py exists and is runnable
3. Verify .env has OPENROUTER_API_KEY (for agent consultation)

## Pretrained Model Search (Phase 0) — only if `Pretrained: yes`

If Pretrained=yes, search for suitable models before running the baseline:

1. **Sources to check**: HuggingFace Hub (`huggingface.co/models`), PyTorch Hub (`pytorch.org/hub`), GitHub (CheXNet, COVID-Net, OpenMedical)
2. **Keywords**: `pneumonia`, `chest-xray`, `medical-imaging`, `torchvision`, `resnet`, `densenet`, `efficientnet`
3. **Adaptation**: Pretrained ImageNet models (ResNet, DenseNet) adapted by:
   - Changing first conv layer from 3→1 channel (sum/average across RGB or replicate grayscale)
   - Resizing 28×28 inputs to the model's expected input size (e.g., 224×224)
   - Replacing classifier head with 3-output logits for OOD detection
4. **Baseline run**: Run finetuned model, record as iteration 0
5. **Re-search**: If iteration results plateau, re-check for new models
6. **Fallback**: If finetuning doesn't improve over scratch, revert to SimpleCNN baseline

If Pretrained=no, skip to Establish Baseline (train SimpleCNN from scratch).

## Establish Baseline (Iteration 0)

1. `python scripts/run_medmnist.py > run.log 2>&1`
2. Extract Test Accuracy and OOD F1
3. Record in results.tsv

## Iteration Loop

For each iteration (1 to max_iterations):

### Phase 1: Consult Agent
- Route the current problem to the best agent
- Agent provides: hypothesis + specific code change proposal
- Agent response includes scientific reasoning

### Phase 2: Modify
- Apply the agent's proposed change to train.py
- Make ONE focused change per iteration

### Phase 3: Commit
- `git add -f MLAgentBench/benchmarks/medmnist/env/train.py && git commit -m "experiment: {agent} — {description}"`

### Phase 4: Run
- `python scripts/run_medmnist.py > run.log 2>&1`
- Extract Test Accuracy and OOD F1

### Phase 5: Decide
- **keep** — metric improved → commit stays, advance branch
- **discard** — metric worsened → `git revert HEAD --no-edit`; restore train.py
- **crash** — revert; attempt fix or skip

### Phase 6: Log
Append to results.tsv: commit, test_acc, ood_f1, memory_gb, status, description

### Eval Checkpoint
If --evals: check if current_iteration % interval == 0 → run checkpoint analysis.

### Bounded Check
If bounded: current_iteration >= max_iterations → exit loop.

## Summary
Print: total iterations, kept/discarded, best metrics, most effective agent.

## Chain Handoff
Write handoff.json for downstream commands.
