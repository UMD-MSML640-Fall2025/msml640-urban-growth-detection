"""
PyTorch Dataset and helper functions for loading patch .npz files.

Each .npz file must contain:
    - 'image': (C, H, W) float32, C = 6 channels
    - 'mask':  (1, H, W) uint8 (0 or 1)

Directory structure:
    data/processed/patches/<AOI>/<AOI>_patch_00001.npz

Usage:
    from models.dataset import PatchDataset, get_train_val_loaders
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PATCH_DIR = PROJECT_ROOT / "data" / "processed" / "patches"


class PatchDataset(Dataset):
    def __init__(self, patch_paths: List[Path]):
        self.patch_paths = patch_paths

    def __len__(self):
        return len(self.patch_paths)

    def __getitem__(self, idx):
        path = self.patch_paths[idx]
        data = np.load(path)

        image = data["image"]  # (C, H, W), float32
        mask = data["mask"]    # (1, H, W), uint8

        # Convert to torch tensors
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float()  # will use BCEWithLogitsLoss

        return image, mask


def collect_patch_paths(min_positive_only: bool = False) -> List[Path]:
    """
    Collect all .npz patch files.

    If min_positive_only=True, filters to keep patches that have
    at least 1 positive pixel (mask == 1).
    """
    all_paths: List[Path] = []

    for aoi_dir in PATCH_DIR.glob("*"):
        if not aoi_dir.is_dir():
            continue
        for npz_file in aoi_dir.glob("*.npz"):
            all_paths.append(npz_file)

    if not min_positive_only:
        return sorted(all_paths)

    # Filter patches to keep those with any positive mask
    filtered_paths: List[Path] = []
    for p in all_paths:
        data = np.load(p)
        mask = data["mask"]  # (1, H, W)
        if (mask > 0).any():
            filtered_paths.append(p)

    return sorted(filtered_paths)


def get_train_val_loaders(
    batch_size: int = 16,
    val_fraction: float = 0.2,
    min_positive_only: bool = False,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train/val DataLoaders.

    - min_positive_only=True -> only patches with any change (for a focused model)
    """
    all_paths = collect_patch_paths(min_positive_only=min_positive_only)
    if not all_paths:
        raise RuntimeError(f"No patch .npz files found in {PATCH_DIR}")

    dataset = PatchDataset(all_paths)

    # Train/val split
    val_size = int(len(dataset) * val_fraction)
    train_size = len(dataset) - val_size

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader