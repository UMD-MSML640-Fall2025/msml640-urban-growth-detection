"""
Compute NDVI / NDBI indices for all AOIs.

Reads:
    data/gee_exports/raw_before/<AOI>_before.tif
    data/gee_exports/raw_after/<AOI>_after.tif

Writes (per AOI) to data/gee_exports/processed/:
    <AOI>_ndvi_before.tif
    <AOI>_ndvi_after.tif
    <AOI>_ndvi_change.tif
    <AOI>_ndbi_before.tif
    <AOI>_ndbi_after.tif
    <AOI>_ndbi_change.tif

Run this script from the project root:
    python src/preprocessing/compute_indices.py
"""

from pathlib import Path
import numpy as np
import rasterio

# List of all AOIs used in the project
AOI_NAMES = [
    "Apex_HaddonHall",
    "Austin_Pflugerville",
    "Charlotte_SteeleCreek",
    "Dallas_Frisco",
    "Denver_Stonegate",
    "Houston_Spring",
    "LasVegas_Inspirada",
    "Phoenix_Gilbert",
    "Raleigh_BrierCreek",
    "Raleigh_NorthHills"
]

# ---------- Paths ----------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GEE_ROOT = PROJECT_ROOT / "data" / "gee_exports"

RAW_BEFORE = GEE_ROOT / "raw_before"
RAW_AFTER = GEE_ROOT / "raw_after"
PROC_DIR = GEE_ROOT / "processed"

PROC_DIR.mkdir(exist_ok=True, parents=True)


# ---------- Band helpers (0-based indices in the raster arrays) ----------

# Sentinel-2 band order in your exports:
# [B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B11, B12]
B4_IDX = 3   # Red
B8_IDX = 7   # NIR
B11_IDX = 10 # SWIR


def safe_index(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Avoid division by zero for vegetation / built-up indices."""
    eps = 1e-6
    return (numerator - denominator) / (numerator + denominator + eps)


def compute_ndvi(stack: np.ndarray) -> np.ndarray:
    """Compute NDVI from a (bands, H, W) Sentinel-2 stack."""
    red = stack[B4_IDX]
    nir = stack[B8_IDX]
    return safe_index(nir, red)


def compute_ndbi(stack: np.ndarray) -> np.ndarray:
    """Compute NDBI from a (bands, H, W) Sentinel-2 stack."""
    swir = stack[B11_IDX]
    nir = stack[B8_IDX]
    return safe_index(swir, nir)


def load_tif(path: Path):
    """Load a GeoTIFF as (array, profile)."""
    with rasterio.open(path) as src:
        arr = src.read().astype("float32")
        profile = src.profile
    return arr, profile


def save_single_band(out_path: Path, arr: np.ndarray, profile):
    """Save a single-band array as GeoTIFF with updated profile."""
    profile = profile.copy()
    profile.update(count=1, dtype="float32")
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr.astype("float32"), 1)


def process_aoi(aoi_name: str):
    """Compute NDVI/NDBI indices for one AOI and write outputs."""
    before_path = RAW_BEFORE / f"{aoi_name}_before.tif"
    after_path = RAW_AFTER / f"{aoi_name}_after.tif"

    if not before_path.exists():
        print(f"[WARN] Missing BEFORE image for {aoi_name}: {before_path}")
        return
    if not after_path.exists():
        print(f"[WARN] Missing AFTER image for {aoi_name}: {after_path}")
        return

    print(f"[INFO] Processing AOI: {aoi_name}")

    before_stack, profile = load_tif(before_path)
    after_stack, _ = load_tif(after_path)

    # --- Compute NDVI ---
    ndvi_before = compute_ndvi(before_stack)
    ndvi_after = compute_ndvi(after_stack)
    ndvi_change = ndvi_after - ndvi_before

    # --- Compute NDBI ---
    ndbi_before = compute_ndbi(before_stack)
    ndbi_after = compute_ndbi(after_stack)
    ndbi_change = ndbi_after - ndbi_before

    # --- Save outputs ---
    save_single_band(PROC_DIR / f"{aoi_name}_ndvi_before.tif", ndvi_before, profile)
    save_single_band(PROC_DIR / f"{aoi_name}_ndvi_after.tif", ndvi_after, profile)
    save_single_band(PROC_DIR / f"{aoi_name}_ndvi_change.tif", ndvi_change, profile)

    save_single_band(PROC_DIR / f"{aoi_name}_ndbi_before.tif", ndbi_before, profile)
    save_single_band(PROC_DIR / f"{aoi_name}_ndbi_after.tif", ndbi_after, profile)
    save_single_band(PROC_DIR / f"{aoi_name}_ndbi_change.tif", ndbi_change, profile)

    print(f"[OK] Saved indices for {aoi_name} in {PROC_DIR}")


def discover_aois():
    """Infer AOI names from *_before.tif files in RAW_BEFORE."""
    names = []
    for before_file in RAW_BEFORE.glob("*_before.tif"):
        stem = before_file.stem  # e.g., 'Austin_Pflugerville_before'
        aoi_name = stem.replace("_before", "")
        names.append(aoi_name)
    return sorted(names)


def main():
    aois = discover_aois()
    if not aois:
        print(f"[ERROR] No *_before.tif found in {RAW_BEFORE}")
        return

    print(f"[INFO] Found AOIs: {aois}")
    for name in aois:
        process_aoi(name)


if __name__ == "__main__":
    main()