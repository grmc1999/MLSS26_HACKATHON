"""Regression test: ExperimentManager.run_experiment() must be able to read the
metric that the actual training scripts print to stdout.

The keep/discard loop depends on this contract. train.py prints "Standard accuracy:" /
"OOD Detection F1:" (see train.py __main__); run_medmnist.py prints "Test 3-class
Accuracy:" / "OOD F1 Score:". orchestrator.py's run_experiment() regex (orchestrator.py
:274-277) is matched against both label sets so neither script's output is silently
dropped to metric=None.
"""
import subprocess

import pytest

from MLAgentBench.agents.orchestrator import ExperimentManager

# Copied verbatim from train.py's __main__ print block (values substituted).
TRAIN_PY_STDOUT = (
    "==================================================\n"
    "  Test Results (ChestMNIST 3-class)\n"
    "==================================================\n"
    "  Standard accuracy:       0.8123\n"
    "  OOD Detection F1:        0.4567\n"
)

# Copied verbatim from run_medmnist.py's train_model() print block (values substituted).
RUN_MEDMNIST_STDOUT = (
    "==================================================\n"
    "  MedMNIST Chest X-ray OOD Results\n"
    "==================================================\n"
    "  Val Accuracy (PneumoniaMNIST):  0.9000\n"
    "  Test ID Acc (Normal+Pneumonia): 0.8500\n"
    "  Test 3-class Accuracy:          0.8123\n"
    "  OOD F1 Score:                   0.4567\n"
)


@pytest.mark.regression
@pytest.mark.parametrize("stdout", [TRAIN_PY_STDOUT, RUN_MEDMNIST_STDOUT])
def test_run_experiment_parses_real_accuracy_line(tmp_path, monkeypatch, stdout):
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(args=a, returncode=0, stdout=stdout, stderr=""),
    )
    manager = ExperimentManager(log_dir=tmp_path)
    result = manager.run_experiment(cmd="echo stub")
    assert result["metric"] == pytest.approx(0.8123)
