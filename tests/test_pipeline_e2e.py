"""End-to-end smoke test: real data -> train -> evaluate -> OOD metrics.

Unlike test_train_unit.py (random tensors, no I/O), this exercises the actual
loader.get_datasets() contract and runs a couple of real batches through the
full train/eval/OOD-scoring pipeline on the device train.py itself would pick
(GPU if available, else CPU). It does not assert on accuracy values -- only
that the pipeline runs end-to-end and produces well-formed metrics.
"""
import pytest
import torch
from torch.utils.data import DataLoader

import train
from loader import get_datasets

EPOCHS = 1
BATCH_SIZE = 16


@pytest.mark.e2e
def test_real_data_pipeline_runs_end_to_end(device):
    train_ds, val_ds, test_ds = get_datasets()

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = train.create_model("SimpleCNN", num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = train.FocalLoss(gamma=2.0, logit_norm=True)

    loss, train_acc = train.train_epoch(model, train_loader, optimizer, criterion, device)
    assert torch.isfinite(torch.tensor(loss))
    assert 0.0 <= train_acc <= 1.0

    test_acc, preds, labels = train.evaluate(model, test_loader, device)
    assert 0.0 <= test_acc <= 1.0
    assert preds.shape == labels.shape

    metrics = train.ood_metrics(labels, preds, ood_cls=2)
    assert 0.0 <= metrics["f1"] <= 1.0
    assert metrics["tp"] + metrics["fp"] + metrics["fn"] + metrics["tn"] == len(labels)
