"""Real data loader: PneumoniaMNIST (train/val) + ChestMNIST 3-class OOD subset (test).

The test subset is built once from the official ChestMNIST test split, keeping only
single-label samples for normal (no finding), pneumonia (label 6), and consolidation
(label 8) -- the only single-label disease classes available in ChestMNIST. The normal
pool is downsampled to N_NORMAL with a fixed seed; pneumonia/consolidation are kept in
full. Result is cached to SUBSET_NPZ so it isn't rebuilt on every run.
"""
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from medmnist import PneumoniaMNIST, ChestMNIST

CLASS_NAMES = ["normal", "pneumonia", "consolidation"]
OOD_CLASS = 2
N_CLASSES = 3

DATA_DIR = Path(__file__).resolve().parents[4] / "data" / "medmnist_subset"
SUBSET_NPZ = DATA_DIR / "chestmnist_3class.npz"
N_NORMAL = 300
SEED = 0


class ArrayDataset(Dataset):
    """(images, labels) numpy arrays as a torch Dataset of (1, 28, 28) float tensors."""

    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.images[idx]).float().unsqueeze(0) / 255.0
        return img, int(self.labels[idx])


def _build_chestmnist_3class_subset():
    ds = ChestMNIST(split="test", download=True, size=28)
    multihot = ds.labels
    no_finding = multihot.sum(axis=1) == 0
    pure_pneumonia = (multihot[:, 6] == 1) & (multihot.sum(axis=1) == 1)
    pure_consolidation = (multihot[:, 8] == 1) & (multihot.sum(axis=1) == 1)

    rng = np.random.default_rng(SEED)
    normal_idx = rng.choice(np.flatnonzero(no_finding), size=N_NORMAL, replace=False)
    pneumonia_idx = np.flatnonzero(pure_pneumonia)
    consolidation_idx = np.flatnonzero(pure_consolidation)

    images = np.concatenate([ds.imgs[normal_idx], ds.imgs[pneumonia_idx], ds.imgs[consolidation_idx]])
    labels = np.concatenate([
        np.zeros(len(normal_idx), dtype=np.int64),
        np.ones(len(pneumonia_idx), dtype=np.int64),
        np.full(len(consolidation_idx), 2, dtype=np.int64),
    ])

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(SUBSET_NPZ, images=images, labels=labels)
    return images, labels


def _load_chestmnist_3class_subset():
    if SUBSET_NPZ.exists():
        data = np.load(SUBSET_NPZ)
        return data["images"], data["labels"]
    return _build_chestmnist_3class_subset()


def get_datasets():
    """Train/val on PneumoniaMNIST (2 classes); test on the ChestMNIST 3-class OOD subset."""
    train_src = PneumoniaMNIST(split="train", download=True, size=28)
    val_src = PneumoniaMNIST(split="val", download=True, size=28)
    test_images, test_labels = _load_chestmnist_3class_subset()

    train_ds = ArrayDataset(train_src.imgs, train_src.labels.squeeze(-1))
    val_ds = ArrayDataset(val_src.imgs, val_src.labels.squeeze(-1))
    test_ds = ArrayDataset(test_images, test_labels)
    return train_ds, val_ds, test_ds
