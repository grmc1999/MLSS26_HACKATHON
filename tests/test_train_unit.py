"""Unit tests for MLAgentBench/benchmarks/medmnist/env/train.py.

Formalizes the manual "Code Jury" from .opencode/commands/autoresearch_pipeline.md
(Phase 4b) as a real, reusable pytest file: syntax, model instantiation, forward
pass shape, loss finiteness, and gradient flow. Also covers what the Jury never
checked — that the OOD metric math itself is correct.
"""
import numpy as np
import pytest
import torch
import torch.nn as nn

import train

pytestmark = pytest.mark.unit

INPUT_SHAPE = (4, 1, 28, 28)


def test_module_imports_without_error():
    """Catches syntax/import errors before a 5-20min training run starts."""
    assert hasattr(train, "create_model")


@pytest.mark.parametrize("batch_size", [1, 4, 16])
def test_forward_pass_output_shape(device, batch_size):
    model = train.create_model("SimpleCNN", num_classes=2).to(device)
    x = torch.randn(batch_size, 1, 28, 28, device=device)
    out = model(x)
    assert out.shape == (batch_size, 3)


def test_loss_is_finite():
    model = train.create_model("SimpleCNN", num_classes=2)
    x = torch.randn(*INPUT_SHAPE)
    y = torch.randint(0, 3, (INPUT_SHAPE[0],))
    loss = train.FocalLoss(gamma=2.0, logit_norm=True)(model(x), y)
    assert torch.isfinite(loss)


def test_backward_pass_produces_gradients():
    model = train.create_model("SimpleCNN", num_classes=2)
    x = torch.randn(*INPUT_SHAPE)
    y = torch.randint(0, 3, (INPUT_SHAPE[0],))
    loss = train.FocalLoss(gamma=2.0, logit_norm=True)(model(x), y)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)
    assert all(torch.isfinite(g).all() for g in grads)


def test_one_optimizer_step_reduces_loss_on_a_single_batch():
    """Classic sanity check: the model must be able to overfit one tiny batch."""
    torch.manual_seed(0)
    model = train.create_model("SimpleCNN", num_classes=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()
    x = torch.randn(8, 1, 28, 28)
    y = torch.randint(0, 3, (8,))

    losses = []
    for _ in range(20):
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0]


@pytest.mark.parametrize(
    "labels,preds,expected",
    [
        # perfect detector
        (np.array([0, 1, 2, 2]), np.array([0, 1, 2, 2]), {"tp": 2, "fp": 0, "fn": 0, "tn": 2}),
        # misses both OOD samples
        (np.array([0, 1, 2, 2]), np.array([0, 1, 0, 1]), {"tp": 0, "fp": 0, "fn": 2, "tn": 2}),
        # false-alarms on both ID samples
        (np.array([0, 1, 2, 2]), np.array([2, 2, 2, 2]), {"tp": 2, "fp": 2, "fn": 0, "tn": 0}),
    ],
)
def test_ood_metrics_confusion_counts(labels, preds, expected):
    metrics = train.ood_metrics(labels, preds, ood_cls=2)
    assert metrics["tp"] == expected["tp"]
    assert metrics["fp"] == expected["fp"]
    assert metrics["fn"] == expected["fn"]
    assert metrics["tn"] == expected["tn"]


def test_ood_metrics_f1_is_one_for_a_perfect_detector():
    labels = np.array([0, 1, 2, 2])
    metrics = train.ood_metrics(labels, labels, ood_cls=2)
    assert metrics["f1"] == pytest.approx(1.0, abs=1e-4)


def test_per_class_accuracy_matches_manual_count():
    labels = np.array([0, 0, 1, 1, 2, 2, 2])
    preds = np.array([0, 1, 1, 1, 2, 2, 0])
    accs = train.per_class_accuracy(labels, preds, num_classes=3)
    assert accs["normal"] == pytest.approx(0.5)
    assert accs["pneumonia"] == pytest.approx(1.0)
    assert accs["consolidation"] == pytest.approx(2 / 3)


def test_in_distribution_accuracy_ignores_ood_class():
    labels = np.array([0, 0, 1, 2, 2])
    preds = np.array([0, 1, 1, 0, 1])  # OOD predictions on class 2 are irrelevant
    acc = train.in_distribution_accuracy(labels, preds, id_classes=(0, 1))
    assert acc == pytest.approx(2 / 3)


def test_in_distribution_accuracy_empty_mask_returns_zero():
    labels = np.array([2, 2, 2])
    preds = np.array([2, 2, 2])
    assert train.in_distribution_accuracy(labels, preds, id_classes=(0, 1)) == 0.0
