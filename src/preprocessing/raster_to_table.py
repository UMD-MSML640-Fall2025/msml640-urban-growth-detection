"""
Convert NDVI / NDBI rasters into per-AOI tabular features.

Reads (per AOI) from:
    data/gee_exports/processed/
        <AOI>_ndvi_before.tif
        <AOI>_ndvi_after.tif
        <AOI>_ndvi_change.tif
        <AOI>_ndbi_before.tif
        <AOI>_ndbi_after.tif
        <AOI>_ndbi_change.tif

Writes:
    data/processed/samples/urban_growth_features_per_aoi.csv

Run from project root:

    python src/preprocessing/raster_to_table.py
"""

from pathlib import Path
import numpy as np
import rasterio
import pandas as pd


# ---------- Paths ----------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GEE_PROC = PROJECT_ROOT / "data" / "gee_exports" / "processed"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "samples"
OUT_DIR.mkdir(exist_ok=True, parents=True)

OUT_CSV = OUT_DIR / "urban_growth_features_per_aoi.csv"


# ---------- Helpers ----------

def load_single_band(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
    # mask out nodata if present
    if hasattr(src, "nodata") and src.nodata is not None:
        arr = np.where(arr == src.nodata, np.nan, arr)
    return arr


def discover_aois():
    """Infer AOI names from *_ndvi_before.tif files."""
    names = []
    for f in GEE_PROC.glob("*_ndvi_before.tif"):
        aoi_name = f.stem.replace("_ndvi_before", "")
        names.append(aoi_name)
    return sorted(names)


def compute_features_for_aoi(aoi: str) -> dict:
    """Compute NDVI/NDBI summary statistics for a single AOI."""
    ndvi_before = load_single_band(GEE_PROC / f"{aoi}_ndvi_before.tif")
    ndvi_after  = load_single_band(GEE_PROC / f"{aoi}_ndvi_after.tif")
    ndvi_change = load_single_band(GEE_PROC / f"{aoi}_ndvi_change.tif")

    ndbi_before = load_single_band(GEE_PROC / f"{aoi}_ndbi_before.tif")
    ndbi_after  = load_single_band(GEE_PROC / f"{aoi}_ndbi_after.tif")
    ndbi_change = load_single_band(GEE_PROC / f"{aoi}_ndbi_change.tif")

    # flatten and drop NaNs
    def clean(x):
        return x.reshape(-1)

    ndvi_b_flat = clean(ndvi_before)
    ndvi_a_flat = clean(ndvi_after)
    ndvi_c_flat = clean(ndvi_change)

    ndbi_b_flat = clean(ndbi_before)
    ndbi_a_flat = clean(ndbi_after)
    ndbi_c_flat = clean(ndbi_change)

    # mask out NaNs (if any)
    valid_mask = ~np.isnan(ndvi_b_flat) & ~np.isnan(ndvi_a_flat) & ~np.isnan(ndvi_c_flat) \
                 & ~np.isnan(ndbi_b_flat) & ~np.isnan(ndbi_a_flat) & ~np.isnan(ndbi_c_flat)

    ndvi_b_flat = ndvi_b_flat[valid_mask]
    ndvi_a_flat = ndvi_a_flat[valid_mask]
    ndvi_c_flat = ndvi_c_flat[valid_mask]
    ndbi_b_flat = ndbi_b_flat[valid_mask]
    ndbi_a_flat = ndbi_a_flat[valid_mask]
    ndbi_c_flat = ndbi_c_flat[valid_mask]

    # ---- Simple statistics ----
    features = {
        "AOI": aoi,
        "NDVI_before_mean": float(np.nanmean(ndvi_b_flat)),
        "NDVI_after_mean": float(np.nanmean(ndvi_a_flat)),
        "NDVI_change_mean": float(np.nanmean(ndvi_c_flat)),
        "NDVI_before_std": float(np.nanstd(ndvi_b_flat)),
        "NDVI_after_std": float(np.nanstd(ndvi_a_flat)),
        "NDBI_before_mean": float(np.nanmean(ndbi_b_flat)),
        "NDBI_after_mean": float(np.nanmean(ndbi_a_flat)),
        "NDBI_change_mean": float(np.nanmean(ndbi_c_flat)),
        "NDBI_before_std": float(np.nanstd(ndbi_b_flat)),
        "NDBI_after_std": float(np.nanstd(ndbi_a_flat)),
    }

    # ---- Urbanization proxy ----
    # Mark pixels that look like "new built-up":
    # NDBI increased a lot AND NDVI decreased (vegetation loss)
    new_urban_mask = (ndbi_c_flat > 0.2) & (ndvi_c_flat < 0.0)

    if new_urban_mask.size > 0:
        urbanization_percent = float(new_urban_mask.mean() * 100.0)
    else:
        urbanization_percent = 0.0

    features["Urbanization_percent"] = urbanization_percent

    # Also record fraction of strong vegetation loss
    veg_loss_mask = ndvi_c_flat < -0.2
    if veg_loss_mask.size > 0:
        features["Vegetation_loss_percent"] = float(veg_loss_mask.mean() * 100.0)
    else:
        features["Vegetation_loss_percent"] = 0.0

    return features


def main():
    aois = discover_aois()
    if not aois:
        print(f"[ERROR] No *_ndvi_before.tif files found in {GEE_PROC}")
        return

    print(f"[INFO] Found AOIs: {aois}")

    rows = []
    for aoi in aois:
        print(f"[INFO] Computing features for {aoi}")
        feats = compute_features_for_aoi(aoi)
        rows.append(feats)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"[OK] Wrote features table to {OUT_CSV}")
    print(df)


if __name__ == "__main__":
    main()