"""
Generate pixel-wise urban change masks from NDVI / NDBI change rasters.

Reads (per AOI) from:
    data/gee_exports/processed/
        <AOI>_ndvi_change.tif
        <AOI>_ndbi_change.tif

Writes (per AOI) to:
    data/processed/masks/
        <AOI>_urban_change_mask.tif

Usage (run from project root):

    python src/change_detection/diff_masks.py
    # or with custom thresholds:
    python src/change_detection/diff_masks.py --ndbi-threshold 0.25 --ndvi-threshold 0.0
"""

from pathlib import Path
import argparse

import numpy as np
import rasterio


# ---------- Paths ----------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GEE_PROC = PROJECT_ROOT / "data" / "gee_exports" / "processed"
MASK_DIR = PROJECT_ROOT / "data" / "processed" / "masks"
MASK_DIR.mkdir(exist_ok=True, parents=True)


# ---------- Helpers ----------

def discover_aois():
    """Infer AOI names from *_ndvi_change.tif files."""
    names = []
    for f in GEE_PROC.glob("*_ndvi_change.tif"):
        aoi_name = f.stem.replace("_ndvi_change", "")
        names.append(aoi_name)
    return sorted(names)


def load_band(path: Path) -> tuple[np.ndarray, dict]:
    """Load a single-band GeoTIFF and return (array, profile)."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        profile = src.profile
        nodata = src.nodata

    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr)

    return arr, profile


def save_mask(path: Path, mask: np.ndarray, profile: dict):
    """Save a boolean/0-1 mask as uint8 GeoTIFF."""
    profile = profile.copy()
    profile.update(
        count=1,
        dtype="uint8",
        nodata=0,
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(mask.astype("uint8"), 1)


def build_change_mask(
    aoi: str,
    ndbi_threshold: float = 0.2,
    ndvi_threshold: float = 0.0,
):
    """
    Build an urban change mask for a single AOI.

    A pixel is considered "new built-up" if:
        NDBI_change > ndbi_threshold  AND  NDVI_change < ndvi_threshold
    """
    ndvi_path = GEE_PROC / f"{aoi}_ndvi_change.tif"
    ndbi_path = GEE_PROC / f"{aoi}_ndbi_change.tif"

    if not ndvi_path.exists() or not ndbi_path.exists():
        print(f"[WARN] Missing NDVI/NDBI change rasters for {aoi}, skipping.")
        return

    ndvi_change, profile = load_band(ndvi_path)
    ndbi_change, _ = load_band(ndbi_path)

    # Build conditions
    valid = ~np.isnan(ndvi_change) & ~np.isnan(ndbi_change)

    new_urban = (
        (ndbi_change > ndbi_threshold) &
        (ndvi_change < ndvi_threshold) &
        valid
    )

    # Some summary stats for sanity check
    total_pixels = valid.sum()
    new_pixels = new_urban.sum()
    if total_pixels > 0:
        pct = new_pixels / total_pixels * 100.0
    else:
        pct = 0.0

    print(
        f"[INFO] {aoi}: {new_pixels} / {total_pixels} pixels "
        f"({pct:.2f}%) flagged as new urban"
    )

    # Save mask
    out_path = MASK_DIR / f"{aoi}_urban_change_mask.tif"
    save_mask(out_path, new_urban, profile)
    print(f"[OK] Saved mask: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate urban change masks from NDVI/NDBI change rasters."
    )
    parser.add_argument(
        "--ndbi-threshold",
        type=float,
        default=0.2,
        help="Threshold on NDBI_change for new built-up (default: 0.2)",
    )
    parser.add_argument(
        "--ndvi-threshold",
        type=float,
        default=0.0,
        help="Threshold on NDVI_change for vegetation loss (default: 0.0)",
    )

    args = parser.parse_args()

    aois = discover_aois()
    if not aois:
        print(f"[ERROR] No *_ndvi_change.tif found in {GEE_PROC}")
        return

    print(f"[INFO] Found AOIs: {aois}")
    print(
        f"[INFO] Using thresholds: NDBI_change > {args.ndbi_threshold}, "
        f"NDVI_change < {args.ndvi_threshold}"
    )

    for aoi in aois:
        build_change_mask(
            aoi,
            ndbi_threshold=args.ndbi_threshold,
            ndvi_threshold=args.ndvi_threshold,
        )


if __name__ == "__main__":
    main()