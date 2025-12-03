"""
Extract multi-channel image patches + corresponding masks for U-Net training.

Inputs (per AOI):

    data/gee_exports/processed/
        <AOI>_ndvi_before.tif
        <AOI>_ndvi_after.tif
        <AOI>_ndvi_change.tif
        <AOI>_ndbi_before.tif
        <AOI>_ndbi_after.tif
        <AOI>_ndbi_change.tif

    data/processed/masks/
        <AOI>_urban_change_mask.tif

Outputs:

    data/processed/patches/<AOI>/
        <AOI>_patch_00001.npz
        <AOI>_patch_00002.npz
        ...

Each .npz file contains:
    - 'image': (C, H, W) float32  (C = 6 channels)
    - 'mask':  (1, H, W) uint8    (0 or 1)

Run from project root:

    python src/preprocessing/patch_extractor.py
    # or customized:
    python src/preprocessing/patch_extractor.py --patch-size 128 --stride 64
"""

from pathlib import Path
import argparse

import numpy as np
import rasterio


# ---------- Paths ----------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GEE_PROC = PROJECT_ROOT / "data" / "gee_exports" / "processed"
MASK_DIR = PROJECT_ROOT / "data" / "processed" / "masks"
PATCH_DIR = PROJECT_ROOT / "data" / "processed" / "patches"
PATCH_DIR.mkdir(exist_ok=True, parents=True)


# ---------- Helpers ----------

def discover_aois():
    """Infer AOI names from *_urban_change_mask.tif in MASK_DIR."""
    names = []
    for f in MASK_DIR.glob("*_urban_change_mask.tif"):
        aoi_name = f.stem.replace("_urban_change_mask", "")
        names.append(aoi_name)
    return sorted(names)


def load_single_band(path: Path) -> tuple[np.ndarray, dict]:
    """Load a single-band GeoTIFF and return (array, profile)."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        profile = src.profile
        nodata = src.nodata

    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr)

    return arr, profile


def load_mask(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read(1).astype("uint8")
    return arr


def extract_patches_for_aoi(
    aoi: str,
    patch_size: int = 128,
    stride: int = 128,
    min_valid_fraction: float = 0.9,
):
    """
    Extract patches for one AOI.

    - patch_size: window size (H = W = patch_size)
    - stride: step size between windows
    - min_valid_fraction: fraction of non-NaN pixels required
                          to keep a patch
    """
    print(f"[INFO] Extracting patches for AOI: {aoi}")

    # ---- Load the 6 index channels ----
    band_paths = {
        "ndvi_before": GEE_PROC / f"{aoi}_ndvi_before.tif",
        "ndvi_after":  GEE_PROC / f"{aoi}_ndvi_after.tif",
        "ndvi_change": GEE_PROC / f"{aoi}_ndvi_change.tif",
        "ndbi_before": GEE_PROC / f"{aoi}_ndbi_before.tif",
        "ndbi_after":  GEE_PROC / f"{aoi}_ndbi_after.tif",
        "ndbi_change": GEE_PROC / f"{aoi}_ndbi_change.tif",
    }

    missing = [k for k, p in band_paths.items() if not p.exists()]
    if missing:
        print(f"[WARN] Missing bands for {aoi}: {missing}. Skipping.")
        return

    arrays = []
    profile = None

    for name, path in band_paths.items():
        arr, profile = load_single_band(path)
        arrays.append(arr)

    # Stack into (C, H, W)
    image_stack = np.stack(arrays, axis=0)  # (6, H, W)

    # ---- Load mask ----
    mask_path = MASK_DIR / f"{aoi}_urban_change_mask.tif"
    if not mask_path.exists():
        print(f"[WARN] Missing mask for {aoi}: {mask_path}. Skipping.")
        return

    mask = load_mask(mask_path)  # (H, W)

    # ---- Basic sanity check ----
    _, H, W = image_stack.shape
    if mask.shape != (H, W):
        print(
            f"[ERROR] Shape mismatch for {aoi}: image {image_stack.shape}, "
            f"mask {mask.shape}"
        )
        return

    # ---- Patch extraction ----
    aoi_patch_dir = PATCH_DIR / aoi
    aoi_patch_dir.mkdir(exist_ok=True, parents=True)

    patch_id = 0
    pos_patches = 0
    total_patches = 0

    for row in range(0, H - patch_size + 1, stride):
        for col in range(0, W - patch_size + 1, stride):
            img_patch = image_stack[:, row:row + patch_size, col:col + patch_size]
            mask_patch = mask[row:row + patch_size, col:col + patch_size]

            # Skip patches with too many NaNs
            valid_mask = ~np.isnan(img_patch)
            valid_fraction = valid_mask.mean()
            if valid_fraction < min_valid_fraction:
                continue

            # Replace NaNs with 0 (safe after masking)
            img_patch = np.nan_to_num(img_patch, nan=0.0).astype("float32")

            # Count positive pixels
            has_positive = (mask_patch > 0).any()

            patch_id += 1
            total_patches += 1
            if has_positive:
                pos_patches += 1

            out_path = aoi_patch_dir / f"{aoi}_patch_{patch_id:05d}.npz"
            np.savez_compressed(
                out_path,
                image=img_patch,
                mask=mask_patch[np.newaxis, ...].astype("uint8"),  # (1, H, W)
            )

    if total_patches == 0:
        print(f"[WARN] No patches extracted for {aoi}. Check sizes / thresholds.")
    else:
        print(
            f"[OK] {aoi}: saved {total_patches} patches "
            f"({pos_patches} with any urban change)"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Extract multi-channel image + mask patches for U-Net training."
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=128,
        help="Height/width of square patches (default: 128)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=128,
        help="Stride between patches (default: 128; use <patch_size for overlap).",
    )
    parser.add_argument(
        "--min-valid-fraction",
        type=float,
        default=0.9,
        help="Minimum fraction of non-NaN pixels required to keep a patch.",
    )

    args = parser.parse_args()

    aois = discover_aois()
    if not aois:
        print(f"[ERROR] No *_urban_change_mask.tif found in {MASK_DIR}")
        return

    print(f"[INFO] Found AOIs: {aois}")
    print(
        f"[INFO] Using patch_size={args.patch_size}, stride={args.stride}, "
        f"min_valid_fraction={args.min_valid_fraction}"
    )

    for aoi in aois:
        extract_patches_for_aoi(
            aoi,
            patch_size=args.patch_size,
            stride=args.stride,
            min_valid_fraction=args.min_valid_fraction,
        )


if __name__ == "__main__":
    main()