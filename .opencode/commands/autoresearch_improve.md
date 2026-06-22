---
name: autoresearch_improve
description: "Research OOD detection improvements, discover SOTA methods, generate experiment proposals"
argument-hint: "[Goal: <text>] [Iterations: N] [Pretrained: yes|no]"
---

EXECUTE IMMEDIATELY.

## Parse Arguments

Extract from $ARGUMENTS:
- `Goal:` or `--goal` — what to improve (Test Accuracy, OOD F1, calibration)
- `Iterations:` or `--iterations` — default 15.
- `Pretrained:` — "yes" to search and finetune pretrained models, "no" to train from scratch (default: no)

## Setup

question (single batch):
  Q1 (Goal): "What to improve?" — Test Accuracy, OOD F1, calibration, model efficiency
  Q2 (Depth): "Research depth?" — shallow (5), standard (15), deep (30)
  Q3 (Pretrained): "Start from scratch or finetune a pretrained model?" — Scratch (default) or Pretrained

## Research Categories

### Pretrained Models & Finetuning
- HuggingFace hub: medical imaging models (`microsoft/ biomedclip`, `google/vit-base-pneumonia`)
- torchvision models: ResNet, DenseNet, EfficientNet adapted for 28×28 chest X-rays
- GitHub: CheXNet (DenseNet-121), COVID-Net, OpenMED
- Adaptation strategies: modify first conv for 1-channel input, resize 28→224, classifier head replacement
- Transfer learning vs scratch: which performs better under domain shift?

### Architecture Improvements
- ResNet, DenseNet, EfficientNet for 28×28
- Attention mechanisms for small images
- Wide vs deep trade-offs
- Depthwise separable convolutions

### OOD Detection Methods
- Mahalanobis distance (Lee et al., 2018)
- Energy-based OOD (Liu et al., 2020)
- ODIN (Liang et al., 2017)
- ReAct (Sun et al., 2021)
- KNN-based OOD detection
- Ensemble uncertainty estimation

### Training Improvements
- Focal loss for class imbalance
- Label smoothing for calibration
- Mixup/CutMix augmentation
- Contrastive pretraining
- Temperature scaling

### Calibration Methods
- Post-hoc temperature scaling
- Platt scaling
- Isotonic regression
- Bayesian approximation methods

## Process

1. For each research direction: read relevant code/context from project
2. Propose specific change to train.py
3. Run experiment: `python scripts/run_medmnist.py > run.log 2>&1`
4. Record metrics
5. Keep if improved, discard if worse

## Output
Create `experiments/improve-{YYMMDD}-{HHMM}/` with findings and proposals.
