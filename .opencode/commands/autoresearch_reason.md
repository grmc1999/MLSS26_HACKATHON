---
name: autoresearch_reason
description: "Adversarial debate about the experiment trajectory with blind judge convergence"
argument-hint: "[Topic: <text>] [Iterations: N] [RAG: yes|no]"
---

EXECUTE IMMEDIATELY.

## Parse Arguments

Extract from $ARGUMENTS:
- `Topic:` or `--topic` — what to reason about
- `Iterations:` or `--iterations` — default 8.

## Setup

question (single batch):
   Q1 (Topic): "What decision needs reasoned debate?" — next architecture move, threshold strategy, whether to switch approach
   Q2 (RAG): "Use RAG literature search to guide experiments?" — Yes or No

## Process

1. **Generate-A** — Propose a position on the topic
2. **Critic** — Attack position A's weaknesses
3. **Generate-B** — Respond with improved position
4. **Synthesizer** — Merge A and B into synthesized position
5. **Blind Judge** — Randomly labeled judge selects the stronger position
6. **Repeat** — Winner becomes new A for next round
7. **Convergence** — Stop after 3 consecutive wins by same position

## Topics for This Project

- "Should we focus on Test Accuracy or OOD F1?"
- "Is a deeper CNN worth the parameter cost?"
- "Should we use Mahalanobis distance or energy score for OOD?"
- "Is temperature scaling post-hoc or in-training better?"
- "Should we ensemble or use a single strong model?"

## Output
Create `experiments/reason-{YYMMDD}-{HHMM}/` with debate lineage and final recommendation.
