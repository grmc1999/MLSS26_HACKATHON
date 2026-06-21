---
name: autoresearch_debug
description: "Hunt bugs in the contrails training pipeline via hypothesis testing"
argument-hint: "[Symptom: <text>] [Scope: <glob>] [Iterations: N]"
---

EXECUTE IMMEDIATELY.

## Parse Arguments
- `Symptom:` — observed failure
- `Scope:` — files to investigate
- `Iterations:` — default 15

## How It Works

1. Gather symptoms (error messages, wrong metrics, NaN losses)
2. Reconnaissance — inspect relevant code
3. Hypothesize root cause (specific, testable)
4. Test ONE hypothesis per iteration
5. Classify: confirmed / disproven / inconclusive
6. Log every finding with code evidence (file:line)
7. Repeat

## Loop
Each iteration: Hypothesis → Test → Classify → Log → Next

## Chain with fix
`/autoresearch:debug --fix` — after hunting, auto-fix findings
