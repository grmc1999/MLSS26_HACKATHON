---
name: autoresearch_probe
description: "Interrogate the project and requirements: surface constraints, assumptions, blind spots"
argument-hint: "[Topic: <text>] [--chain <targets>]"
---

EXECUTE IMMEDIATELY.

## Parse Arguments

Extract from $ARGUMENTS:
- `Topic:` or `--topic` — what to probe
- `--chain <targets>` — chain to downstream commands after probing

## Setup

question (single batch):
  Q1 (Topic): "What area needs probing?" — data quality, model assumptions, metric validity, experiment design

## 8 Personas

Each persona interrogates the project from their angle:

1. **Skeptic** — "What assumptions are we making?"
2. **Edge-Case Hunter** — "What extreme inputs would break this?"
3. **Scope Sentinel** — "Are we solving the right problem?"
4. **Ambiguity Detective** — "What's underspecified?"
5. **Contradiction Finder** — "What contradicts our current approach?"
6. **Prior-Art Investigator** — "What established methods are we ignoring?"
7. **Success-Criteria Auditor** — "Are our metrics measuring what we think?"
8. **Constraint Excavator** — "What hidden constraints exist?"

## Probe Questions for This Project

- Is the OOD class truly unseen? Check data splits.
- Does the softmax threshold generalize across data distributions?
- Is 28×28 resolution sufficient for pneumonia detection?
- Are metrics computed correctly? Check ood_metrics function.
- What's the domain shift between PneumoniaMNIST and ChestMNIST?
- Is the baseline reproducible?

## Output
Create `experiments/probe-{YYMMDD}-{HHMM}/` with:
- `findings.md` — all surfaced constraints and insights
- `handoff.json` — ready to chain into plan, autoresearch, or other commands
