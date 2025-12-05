"""
Overlay predicted urban change on top of RGB AFTER image.

For each AOI, produce:

    [ RGB After | RGB After + predicted-change overlay ]

Outputs:
    output/processed/samples/rgb_overlay/<AOI>_rgb_overlay.png

Run from project root:

    python -m src.visualization.overlay_rgb_predictions
"""

from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt

from src.preprocessing.compute_indices import AOI_NAMES

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_AFTER_DIR = PROJECT_ROOT / "data" / "gee_exports" / "raw_after"
TRUE_MASK_DIR = PROJECT_ROOT / "data" / "processed" / "masks"
PRED_MASK_DIR = TRUE_MASK_DIR / "predicted"

OUT_DIR = PROJECT_ROOT / "output" / "processed" / "samples" / "rgb_overlay"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_rgb(path: Path) -> np.ndarray:
    """Load Sentinel-2 RGB (B4,B3,B2) scaled to [0,1]."""
    with rasterio.open(path) as src:
        r = src.read(4).astype("float32")
        g = src.read(3).astype("float32")
        b = src.read(2).astype("float32")
        nodata = src.nodata

    if nodata is not None:
        mask = (r == nodata) | (g == nodata) | (b == nodata)
        r[mask] = 0
        g[mask] = 0
        b[mask] = 0

    rgb = np.stack([r, g, b], axis=-1) / 10000.0
    return np.clip(rgb, 0, 1)


def load_mask(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        nodata = src.nodata
    if nodata is not None:
        arr[arr == nodata] = 0
    return (arr > 0).astype("float32")


def overlay(rgb: np.ndarray, mask: np.ndarray, color=(0, 0, 1), alpha=0.5):
    """
    Simple alpha-blend overlay of mask on RGB.
    color: (R,G,B) in [0,1] for the overlay color
    """
    overlay_img = rgb.copy()
    mask_expanded = mask[..., None]  # (H, W, 1)

    color_arr = np.array(color, dtype="float32").reshape(1, 1, 3)
    overlay_img = (1 - alpha * mask_expanded) * overlay_img + alpha * mask_expanded * color_arr
    overlay_img = np.clip(overlay_img, 0, 1)
    return overlay_img


def plot_aoi(aoi: str):
    after_path = RAW_AFTER_DIR / f"{aoi}_after.tif"
    pred_mask_path = PRED_MASK_DIR / f"{aoi}_urban_change_pred_mask.tif"

    if not after_path.exists():
        print(f"[WARN] Missing RGB AFTER for {aoi}: {after_path}")
        return
    if not pred_mask_path.exists():
        print(f"[WARN] Missing predicted mask for {aoi}: {pred_mask_path}")
        return

    rgb_after = load_rgb(after_path)
    pred_mask = load_mask(pred_mask_path)

    if rgb_after.shape[:2] != pred_mask.shape:
        raise RuntimeError(
            f"Shape mismatch for {aoi}: RGB {rgb_after.shape} vs mask {pred_mask.shape}"
        )

    rgb_overlay = overlay(rgb_after, pred_mask, color=(0, 0, 1), alpha=0.6)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(rgb_after)
    axes[0].set_title(f"{aoi}\nRGB After")
    axes[0].axis("off")

    axes[1].imshow(rgb_overlay)
    axes[1].set_title("RGB After + predicted change")
    axes[1].axis("off")

    fig.suptitle(f"Urban Growth – Prediction Overlay – {aoi}", fontsize=14)
    plt.tight_layout()

    out_path = OUT_DIR / f"{aoi}_rgb_overlay.png"
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[OK] Saved overlay → {out_path}")


def main():
    for aoi in AOI_NAMES:
        plot_aoi(aoi)


if __name__ == "__main__":
    main()