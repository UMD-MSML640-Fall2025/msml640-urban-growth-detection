"""
Create full-AOI comparison grids:

  [ RGB BEFORE | RGB AFTER | TRUE CHANGE MASK | PREDICTED CHANGE MASK ]

Outputs (per AOI):
  data/processed/samples/preview_plots/<AOI>_full_aoi_comparison.png

Run from project root:

    python -m src.visualization.full_aoi_maps
"""

from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt

from src.preprocessing.compute_indices import AOI_NAMES

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_BEFORE_DIR = PROJECT_ROOT / "data" / "gee_exports" / "raw_before"
RAW_AFTER_DIR  = PROJECT_ROOT / "data" / "gee_exports" / "raw_after"

# UPDATED: true masks now live in masks/true/
MASKS_ROOT     = PROJECT_ROOT / "data" / "processed" / "masks"
TRUE_MASK_DIR  = MASKS_ROOT / "true"
PRED_MASK_DIR  = MASKS_ROOT / "predicted"

OUT_DIR = PROJECT_ROOT / "data" / "processed" / "samples" / "preview_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_rgb(path: Path) -> np.ndarray:
    """
    Load Sentinel-2 RGB (B4,B3,B2) as float32 in [0,1].
    """
    with rasterio.open(path) as src:
        # Sentinel-2: B2=2, B3=3, B4=4
        r = src.read(4).astype("float32")
        g = src.read(3).astype("float32")
        b = src.read(2).astype("float32")
        nodata = src.nodata

    if nodata is not None:
        mask = (r == nodata) | (g == nodata) | (b == nodata)
        r[mask] = 0
        g[mask] = 0
        b[mask] = 0

    rgb = np.stack([r, g, b], axis=-1)
    rgb = rgb / 10000.0
    rgb = np.clip(rgb, 0, 1)
    return rgb


def load_mask(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        nodata = src.nodata
    if nodata is not None:
        arr[arr == nodata] = 0
    return arr


def plot_aoi(aoi: str):
    before_path    = RAW_BEFORE_DIR / f"{aoi}_before.tif"
    after_path     = RAW_AFTER_DIR  / f"{aoi}_after.tif"
    true_mask_path = TRUE_MASK_DIR  / f"{aoi}_urban_change_mask.tif"
    pred_mask_path = PRED_MASK_DIR  / f"{aoi}_urban_change_pred_mask.tif"

    if not before_path.exists() or not after_path.exists():
        raise FileNotFoundError(f"Missing RGB for {aoi}")
    if not true_mask_path.exists():
        raise FileNotFoundError(f"Missing TRUE mask for {aoi}: {true_mask_path}")
    if not pred_mask_path.exists():
        raise FileNotFoundError(f"Missing PRED mask for {aoi}: {pred_mask_path}")

    rgb_before = load_rgb(before_path)
    rgb_after  = load_rgb(after_path)
    true_mask  = load_mask(true_mask_path)
    pred_mask  = load_mask(pred_mask_path)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(rgb_before)
    axes[0].set_title(f"{aoi}\nRGB Before")
    axes[0].axis("off")

    axes[1].imshow(rgb_after)
    axes[1].set_title("RGB After")
    axes[1].axis("off")

    im2 = axes[2].imshow(true_mask, cmap="Reds")
    axes[2].set_title("True urban change")
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    im3 = axes[3].imshow(pred_mask, cmap="Blues")
    axes[3].set_title("Predicted urban change")
    axes[3].axis("off")
    fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    fig.suptitle(f"Urban Growth Detection – {aoi}", fontsize=14)

    out_path = OUT_DIR / f"{aoi}_full_aoi_comparison.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[OK] Saved {out_path}")


def main():
    for aoi in AOI_NAMES:
        plot_aoi(aoi)


if __name__ == "__main__":
    main()