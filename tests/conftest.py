"""Shared fixtures for the medmnist OOD pipeline tests."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ENV_DIR = Path(__file__).resolve().parent.parent / "MLAgentBench" / "benchmarks" / "medmnist" / "env"
sys.path.insert(0, str(ENV_DIR))


@pytest.fixture(autouse=True)
def fixed_seed():
    """Every test is deterministic regardless of run order."""
    torch.manual_seed(0)
    np.random.seed(0)


@pytest.fixture
def device():
    """GPU if available, else CPU — matches train.py's own device selection."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
