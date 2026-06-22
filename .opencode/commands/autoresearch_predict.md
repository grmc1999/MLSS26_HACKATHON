---
name: autoresearch_predict
description: "Multi-expert prediction: estimate expected improvement before running an experiment"
argument-hint: "[Proposal: <text>]"
---

EXECUTE IMMEDIATELY.

## Parse Arguments

Extract from $ARGUMENTS:
- `Proposal:` or `--proposal` — description of the proposed change

## Setup

question (single batch):
  Q1 (Proposal): "What change do you want to predict the outcome of?"

## Expert Personas

Each persona independently analyzes the proposal:

1. **Architecture Expert** — evaluates impact on model structure
   - How will this change affect forward pass?
   - Parameter count impact?
   - Training speed impact?

2. **Training Dynamics Expert** — evaluates training behavior
   - Will this converge faster/slower?
   - Gradient flow implications?
   - Overfitting risk?

3. **OOD Detection Expert** — evaluates OOD-specific impact
   - How will this affect confidence scores?
   - Impact on softmax threshold separation?
   - Calibration implications?

4. **Data Expert** — evaluates data interaction
   - How does this interact with 28×28 grayscale data?
   - Pneumonia vs consolidation separability impact?
   - Domain shift considerations?

5. **Devil's Advocate** — identifies pitfalls
   - What could go wrong?
   - Hidden assumptions?
   - Failure modes?

## Synthesis

After all 5 respond:
1. Summarize expected metric impact: Test Accuracy and OOD F1 deltas
2. Identify top 3 risks
3. Recommend: proceed, modify approach, or skip
