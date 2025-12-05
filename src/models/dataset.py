"""
Improved Dataset with data augmentation for urban growth detection.

Adds:
- Random flips and rotations
- Mild color jitter on RGB channels
- Optional balanced sampling of positive/negative patches

Each patch file is an .npz with:
    image: (C, H, W), float32   (e.g., 6 channels: RGB before/after or NDVI/NDBI/etc.)
    mask:  (1, H, W), uint8     (urban change mask)
"""

from pathlib import Path
from typing import List, Tuple
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


# Repo root: <project>/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PATCH_DIR = PROJECT_ROOT / "data" / "processed" / "patches"


class PatchDataset(Dataset):
    """Basic patch dataset with optional on-the-fly augmentation."""

    def __init__(self, patch_paths: List[Path], augment: bool = False):
        self.patch_paths = patch_paths
        self.augment = augment

    def __len__(self) -> int:
        return len(self.patch_paths)

    def apply_augmentation(
        self, image: np.ndarray, mask: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply random augmentations to image and mask.

        image: (C, H, W)
        mask:  (1, H, W)

        Returns tensors (image, mask) with same shapes.
        """
        # Convert to tensors
        image_t = torch.from_numpy(image).float()
        mask_t = torch.from_numpy(mask).float()

        # --- Geometric transforms ------------------------------------------------
        # Random horizontal flip
        if random.random() > 0.5:
            image_t = TF.hflip(image_t)
            mask_t = TF.hflip(mask_t)

        # Random vertical flip
        if random.random() > 0.5:
            image_t = TF.vflip(image_t)
            mask_t = TF.vflip(mask_t)

        # Random rotation (90/180/270). Use NEAREST for mask to keep 0/1.
        if random.random() > 0.5:
            angle = random.choice([90, 180, 270])
            image_t = TF.rotate(
                image_t,
                angle,
                interpolation=InterpolationMode.BILINEAR,
            )
            mask_t = TF.rotate(
                mask_t,
                angle,
                interpolation=InterpolationMode.NEAREST,
            )

        # --- Photometric transforms (RGB channels only) --------------------------
        # Slight brightness/contrast jitter
        if random.random() > 0.5:
            # Assume first 3 channels ~ RGB
            rgb = image_t[:3, :, :]
            brightness = random.uniform(0.9, 1.1)
            contrast = random.uniform(0.9, 1.1)
            rgb = TF.adjust_brightness(rgb, brightness)
            rgb = TF.adjust_contrast(rgb, contrast)
            image_t[:3, :, :] = rgb

        # Small Gaussian noise sometimes
        if random.random() > 0.7:
            noise = torch.randn_like(image_t) * 0.02
            image_t = torch.clamp(image_t + noise, 0.0, 1.0)

        return image_t, mask_t

    def __getitem__(self, idx: int):
        path = self.patch_paths[idx]
        data = np.load(path)

        image = data["image"]  # (C, H, W)
        mask = data["mask"]    # (1, H, W)

        if self.augment:
            image_t, mask_t = self.apply_augmentation(image, mask)
        else:
            image_t = torch.from_numpy(image).float()
            mask_t = torch.from_numpy(mask).float()

        return image_t, mask_t


class BalancedPatchDataset(Dataset):
    """
    Dataset that over-samples positive patches to reduce class imbalance.

    positive_ratio: approx fraction of samples that will contain any change.
    """

    def __init__(
        self,
        patch_paths: List[Path],
        augment: bool = False,
        positive_ratio: float = 0.5,
    ):
        self.augment = augment
        self.positive_ratio = positive_ratio

        self.positive_paths: List[Path] = []
        self.negative_paths: List[Path] = []

        # Split into positive / negative by mask contents
        for p in patch_paths:
            data = np.load(p)
            mask = data["mask"]
            if (mask > 0).any():
                self.positive_paths.append(p)
            else:
                self.negative_paths.append(p)

        print(
            f"[INFO] BalancedPatchDataset: "
            f"{len(self.positive_paths)} positive, "
            f"{len(self.negative_paths)} negative patches"
        )

        if self.positive_paths and self.negative_paths:
            # Roughly balance both by oversampling the smaller group
            self.length = max(len(self.positive_paths), len(self.negative_paths)) * 2
        else:
            self.length = len(patch_paths)

    def __len__(self) -> int:
        return self.length

    def _load_patch(self, path: Path):
        data = np.load(path)
        image = data["image"]
        mask = data["mask"]
        return image, mask

    def __getitem__(self, idx: int):
        # Decide positive vs negative for this sample
        use_positive = (
            random.random() < self.positive_ratio and len(self.positive_paths) > 0
        )

        if use_positive:
            path = random.choice(self.positive_paths)
        else:
            # Fallback if one of the lists is empty
            if self.negative_paths:
                path = random.choice(self.negative_paths)
            else:
                path = random.choice(self.positive_paths)

        image, mask = self._load_patch(path)

        if self.augment:
            # Reuse augmentation from PatchDataset
            image_t, mask_t = PatchDataset([path]).apply_augmentation(image, mask)
        else:
            image_t = torch.from_numpy(image).float()
            mask_t = torch.from_numpy(mask).float()

        return image_t, mask_t


def collect_patch_paths(min_positive_only: bool = False) -> List[Path]:
    """
    Collect all .npz patch files across AOI subdirectories.

    If min_positive_only is True, keep only patches where mask has any positive pixel.
    """
    all_paths: List[Path] = []

    for aoi_dir in PATCH_DIR.glob("*"):
        if not aoi_dir.is_dir():
            continue
        for npz_file in aoi_dir.glob("*.npz"):
            all_paths.append(npz_file)

    if not all_paths:
        return []

    if not min_positive_only:
        return sorted(all_paths)

    filtered_paths: List[Path] = []
    for p in all_paths:
        data = np.load(p)
        mask = data["mask"]
        if (mask > 0).any():
            filtered_paths.append(p)

    return sorted(filtered_paths)


def get_train_val_loaders(
    batch_size: int = 16,
    val_fraction: float = 0.2,
    min_positive_only: bool = False,
    use_augmentation: bool = True,
    use_balanced_sampling: bool = True,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train/val DataLoaders with optional augmentation and balanced sampling.
    """
    all_paths = collect_patch_paths(min_positive_only=min_positive_only)
    if not all_paths:
        raise RuntimeError(f"No patch .npz files found in {PATCH_DIR}")

    # Shuffle indices with a fixed seed for reproducibility
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(all_paths), generator=generator).tolist()

    val_size = int(len(all_paths) * val_fraction)
    train_size = len(all_paths) - val_size

    train_paths = [all_paths[i] for i in indices[:train_size]]
    val_paths = [all_paths[i] for i in indices[train_size:]]

    # Build datasets
    if use_balanced_sampling:
        train_ds: Dataset = BalancedPatchDataset(
            train_paths, augment=use_augmentation, positive_ratio=0.5
        )
    else:
        train_ds = PatchDataset(train_paths, augment=use_augmentation)

    val_ds = PatchDataset(val_paths, augment=False)  # never augment validation

    print(f"[INFO] Training samples:   {len(train_ds)}")
    print(f"[INFO] Validation samples: {len(val_ds)}")
    print(
        f"[INFO] Augmentation: {use_augmentation}, "
        f"Balanced sampling: {use_balanced_sampling}"
    )

    # num_workers=0 is safest on macOS; you can bump to 2/4 if you like
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )

    return train_loader, val_loader