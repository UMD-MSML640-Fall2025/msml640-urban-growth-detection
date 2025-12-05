"""
Create a summary 'Model Performance' figure from full-AOI metrics.

Input:
    output/processed/masks/predicted/full_aoi_metrics.csv
        (created by src.change_detection.eval_full_aoi)

Output:
    output/processed/samples/preview_plots/model_performance.png

Run from project root:

    python -m src.visualization.model_performance
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRED_MASK_DIR = PROJECT_ROOT / "output" / "processed" / "masks" / "predicted"
METRICS_CSV = PRED_MASK_DIR / "full_aoi_metrics.csv"

OUT_DIR = PROJECT_ROOT / "output" / "processed" / "samples" / "preview_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "model_performance.png"


def main():
    if not METRICS_CSV.exists():
        raise FileNotFoundError(
            f"Metrics CSV not found: {METRICS_CSV}\n"
            "Run `python -m src.change_detection.eval_full_aoi` first."
        )

    df = pd.read_csv(METRICS_CSV)

    # exclude global "ALL" row from per-AOI bars
    per_aoi = df[df["AOI"] != "ALL"].copy()
    per_aoi = per_aoi.sort_values("AOI")

    fig, ax = plt.subplots(figsize=(10, 5))

    x = range(len(per_aoi))
    width = 0.35

    ax.bar([i - width / 2 for i in x], per_aoi["iou"], width=width, label="IoU")
    ax.bar([i + width / 2 for i in x], per_aoi["f1"], width=width, label="F1-score")

    ax.set_xticks(list(x))
    ax.set_xticklabels(per_aoi["AOI"], rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.set_title("Urban Growth Detection – Full-AOI Performance")
    ax.legend()

    # Optionally show global average as horizontal lines
    global_row = df[df["AOI"] == "ALL"].iloc[0]
    ax.axhline(global_row["iou"], color="gray", linestyle="--", alpha=0.5)
    ax.text(
        len(per_aoi) - 0.5,
        global_row["iou"] + 0.01,
        f"Global IoU={global_row['iou']:.2f}",
        ha="right",
        va="bottom",
        fontsize=8,
        color="gray",
    )

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=200)
    plt.close(fig)
    print(f"[OK] Saved model performance figure → {OUT_PATH}")


if __name__ == "__main__":
    main()