"""
Compute per-AOI area statistics and detection metrics.

Outputs:
  data/processed/samples/urban_growth_area_stats.csv

Run:

    python -m src.change_detection.area_statistics
"""

from pathlib import Path
import csv

import numpy as np
import rasterio

from src.preprocessing.compute_indices import AOI_NAMES

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRUE_MASK_DIR = PROJECT_ROOT / "data" / "processed" / "masks" / "true"
PRED_MASK_DIR = PROJECT_ROOT / "data" / "processed" / "masks" / "predicted"
OUT_CSV = PROJECT_ROOT / "data" / "processed" / "samples" / "urban_growth_area_stats.csv"

PIXEL_SIZE_M = 10.0           # Sentinel-2 10m resolution
PIXEL_AREA_M2 = PIXEL_SIZE_M * PIXEL_SIZE_M   # 100 m^2
M2_PER_HECTARE = 10000.0
M2_PER_KM2 = 1_000_000.0


def load_mask(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read(1)
        nodata = src.nodata
    if nodata is not None:
        arr = np.where(arr == nodata, 0, arr)
    return (arr > 0).astype(np.uint8)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    assert y_true.shape == y_pred.shape
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)

    tp = np.logical_and(y_true, y_pred).sum()
    fp = np.logical_and(~y_true, y_pred).sum()
    fn = np.logical_and(y_true, ~y_pred).sum()

    intersection = tp
    union = tp + fp + fn + 1e-6
    iou = intersection / union

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)

    return iou, precision, recall, tp, fp, fn


def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for aoi in AOI_NAMES:
        true_path = TRUE_MASK_DIR / f"{aoi}_urban_change_mask.tif"
        pred_path = PRED_MASK_DIR / f"{aoi}_urban_change_pred_mask.tif"

        if not true_path.exists():
            print(f"[WARN] Missing true mask for {aoi}, skipping")
            continue
        if not pred_path.exists():
            print(f"[WARN] Missing predicted mask for {aoi}, skipping")
            continue

        true_mask = load_mask(true_path)
        pred_mask = load_mask(pred_path)

        true_pixels = true_mask.sum()
        pred_pixels = pred_mask.sum()

        true_area_m2 = true_pixels * PIXEL_AREA_M2
        pred_area_m2 = pred_pixels * PIXEL_AREA_M2

        true_area_ha = true_area_m2 / M2_PER_HECTARE
        pred_area_ha = pred_area_m2 / M2_PER_HECTARE

        true_area_km2 = true_area_m2 / M2_PER_KM2
        pred_area_km2 = pred_area_m2 / M2_PER_KM2

        iou, precision, recall, tp, fp, fn = compute_metrics(true_mask, pred_mask)

        rows.append({
            "aoi": aoi,
            "true_pixels": int(true_pixels),
            "pred_pixels": int(pred_pixels),
            "true_area_ha": true_area_ha,
            "pred_area_ha": pred_area_ha,
            "true_area_km2": true_area_km2,
            "pred_area_km2": pred_area_km2,
            "IoU": iou,
            "precision": precision,
            "recall": recall,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
        })

    fieldnames = list(rows[0].keys()) if rows else []
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Saved stats to {OUT_CSV}")


if __name__ == "__main__":
    main()