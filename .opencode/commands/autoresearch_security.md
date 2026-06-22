---
name: autoresearch_security
description: "Audit the OOD pipeline for security issues: leak paths, data integrity, adversarial robustness"
argument-hint: "[Iterations: N]"
---

EXECUTE IMMEDIATELY.

## Parse Arguments

Extract from $ARGUMENTS:
- `Iterations:` or `--iterations` — default 10. "unlimited" for unbounded.

## Setup

question (single batch):
  Q1 (Focus): "Security audit focus?" — data leakage, model poisoning, adversarial robustness, API key exposure, all

## Audit Areas

### Data Integrity
- Check data loading paths: does loader.py handle corrupted files?
- Check data splits: any data leakage between train/test?
- Verify OOD class (consolidation) is truly unseen during training

### API Key Security
- Check that OPENROUTER_API_KEY is never logged or committed
- Check .env is in .gitignore
- Check run.log doesn't capture env vars

### Model Security
- Check for hardcoded thresholds that could be exploited
- Check model save/load paths for injection
- Verify checkpoint files don't contain training data

### Adversarial Robustness
- Test model sensitivity to small perturbations
- Check OOD detection under adversarial examples
- Evaluate robustness of softmax threshold

## Output
Create `experiments/security-{YYMMDD}-{HHMM}/` with findings report.
