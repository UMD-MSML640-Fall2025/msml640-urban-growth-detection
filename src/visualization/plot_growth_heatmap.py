"""
Plot heatmaps of predicted urban-growth probability for each AOI.

Outputs:
  data/processed/samples/preview_plots/heatmaps/<AOI>_pred_prob_heatmap.png

Run:

    python -m src.visualization.plot_growth_heatmap
"""

from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt

from src.preprocessing.compute_indices import AOI_NAMES

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRED_MASK_DIR = PROJECT_ROOT / "data" / "processed" / "masks" / "predicted"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "samples" / "preview_plots" / "heatmaps"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_prob(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        nodata = src.nodata
    if nodata is not None:
        arr[arr == nodata] = np.nan
    return arr


def main():
    for aoi in AOI_NAMES:
        prob_path = PRED_MASK_DIR / f"{aoi}_urban_change_pred_prob.tif"
        if not prob_path.exists():
            print(f"[WARN] Missing prob map for {aoi}")
            continue

        prob = load_prob(prob_path)

        plt.figure(figsize=(5, 5))
        im = plt.imshow(prob, cmap="viridis", vmin=0, vmax=1)
        plt.title(f"{aoi} – Predicted urban growth probability")
        plt.axis("off")
        plt.colorbar(im, fraction=0.046, pad=0.04, label="P(growth)")

        out_path = OUT_DIR / f"{aoi}_pred_prob_heatmap.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"[OK] Saved {out_path}")


if __name__ == "__main__":
    main()