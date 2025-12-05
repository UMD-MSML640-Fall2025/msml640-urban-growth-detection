"""
Compute full-AOI segmentation metrics between TRUE and PREDICTED masks.

Inputs (per AOI):
    data/processed/masks/
        <AOI>_urban_change_mask.tif          (ground truth, 0/1)
    data/processed/masks/predicted/
        <AOI>_urban_change_pred_mask.tif     (model prediction, 0/1)

Output:
    output/processed/masks/predicted/full_aoi_metrics.csv

Run from project root:

    python -m src.change_detection.eval_full_aoi
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import rasterio

from src.preprocessing.compute_indices import AOI_NAMES

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRUE_MASK_DIR = PROJECT_ROOT / "data" / "processed" / "masks" / "true"
PRED_MASK_DIR = PROJECT_ROOT / "data" / "processed" / "masks" / "predicted"
OUT_CSV = (
    PROJECT_ROOT
    / "output"
    / "processed"
    / "masks"
    / "predicted"
    / "full_aoi_metrics.csv"
)


def discover_aois() -> List[str]:
    """
    Use AOI_NAMES, but only keep those that actually have BOTH true+pred masks.
    This prevents the 'ALL=0' situation when files are missing.
    """
    valid = []
    for aoi in AOI_NAMES:
        true_path = TRUE_MASK_DIR / f"{aoi}_urban_change_mask.tif"
        pred_path = PRED_MASK_DIR / f"{aoi}_urban_change_pred_mask.tif"
        if true_path.exists() and pred_path.exists():
            valid.append(aoi)
        else:
            if not true_path.exists():
                print(f"[WARN] Missing TRUE mask for {aoi}: {true_path}")
            if not pred_path.exists():
                print(f"[WARN] Missing PRED mask for {aoi}: {pred_path}")
    if not valid:
        raise RuntimeError(
            f"No AOIs with both true+pred masks under "
            f"{TRUE_MASK_DIR} and {PRED_MASK_DIR}"
        )
    print(f"[INFO] Evaluating AOIs: {valid}")
    return valid


def load_mask(path: Path) -> np.ndarray:
    """Load mask as float, keep nodata as NaN so we can ignore it later."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        nodata = src.nodata

    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr)

    # we keep it as float; threshold happens in compute_confusion
    return arr


def compute_confusion(true: np.ndarray, pred: np.ndarray) -> Dict[str, int]:
    """
    Compute TP/FP/FN/TN, ignoring NaN pixels and cropping to common extent.
    """
    H = min(true.shape[0], pred.shape[0])
    W = min(true.shape[1], pred.shape[1])
    true = true[:H, :W]
    pred = pred[:H, :W]

    valid = ~np.isnan(true)
    if np.isnan(pred).any():
        valid &= ~np.isnan(pred)

    t = (true > 0.5) & valid
    p = (pred > 0.5) & valid

    tp = int(np.logical_and(t, p).sum())
    fp = int(np.logical_and(~t, p).sum())
    fn = int(np.logical_and(t, ~p).sum())
    tn = int(np.logical_and(~t, ~p).sum())

    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def metrics_from_counts(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    eps = 1e-8

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)

    return {
        "iou": float(iou),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
    }


def main():
    aois = discover_aois()

    rows: List[Dict] = []
    total = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}

    for aoi in aois:
        true_path = TRUE_MASK_DIR / f"{aoi}_urban_change_mask.tif"
        pred_path = PRED_MASK_DIR / f"{aoi}_urban_change_pred_mask.tif"

        true = load_mask(true_path)
        pred = load_mask(pred_path)

        counts = compute_confusion(true, pred)
        for k in total:
            total[k] += counts[k]

        m = metrics_from_counts(**counts)
        row = {"AOI": aoi}
        row.update(counts)
        row.update(m)
        rows.append(row)

        print(
            f"[{aoi}] IoU={m['iou']:.3f}, F1={m['f1']:.3f}, "
            f"P={m['precision']:.3f}, R={m['recall']:.3f}, "
            f"TP={counts['tp']}, FP={counts['fp']}, FN={counts['fn']}, TN={counts['tn']}"
        )

    df = pd.DataFrame(rows)

    # Global micro-averaged metrics
    global_metrics = metrics_from_counts(**total)
    global_row = {"AOI": "ALL"}
    global_row.update(total)
    global_row.update(global_metrics)
    df = pd.concat([df, pd.DataFrame([global_row])], ignore_index=True)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n[OK] Saved full-AOI metrics → {OUT_CSV}")
    print(df)


if __name__ == "__main__":
    main()