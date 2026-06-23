---
name: regression-testing
description: "Contract tests that catch silent breakage between train.py's output and orchestrator.py's parsing"
version: 1.0.0
---

# Regression Testing — Contract Checks

Different from `/autoresearch_regression` (which compares baseline vs candidate *metric values*,
e.g. ID Test Acc, OOD F1). This skill checks that the *code* talking to other code hasn't drifted —
specifically, that what `train.py` prints is what `orchestrator.py` (or any grep-based command)
actually parses.

## Run it

```bash
pytest tests/test_metric_contract_regression.py tests/test_orchestrator_routing.py -q -m regression
```

## What it checks

`tests/test_metric_contract_regression.py` feeds `ExperimentManager.run_experiment()` the exact
stdout both `train.py` (`"Standard accuracy:"`, `"OOD Detection F1:"`) and `run_medmnist.py`
(`"Test 3-class Accuracy:"`, `"OOD F1 Score:"`) print, and asserts the parsed metric matches
either way. This used to fail — `orchestrator.py:274-276` searched only for `"Test Accuracy:"` /
`"OOD F1:"`, which never appeared in either script's real output, so `run_experiment()` silently
returned `metric=None` for every run. Fixed in `orchestrator.py` by matching both label sets
(`Standard accuracy|Test 3-class Accuracy` and `OOD Detection F1|OOD F1 Score`).

(The markdown-driven `/autoresearch` commands grep for their own patterns and were fixed
separately to match `run_medmnist.py`'s real labels — see their command files.)

`tests/test_orchestrator_routing.py::test_every_routable_agent_has_a_system_prompt` checks the
other direction: every agent name in `orchestrator.ROUTING_KEYWORDS` must have a non-empty entry
in `agent_specialized.AGENT_PROMPTS` — catches a routed-to agent silently having no prompt.

## When to use

- Whenever `train.py`'s print statements change, or `orchestrator.py`'s parsing regex changes
- Whenever `ROUTING_KEYWORDS` or `AGENT_PROMPTS` gain/lose an agent

## Related

See [[unit-testing]] for fast no-training checks, [[e2e-testing]] for real-data runs.
