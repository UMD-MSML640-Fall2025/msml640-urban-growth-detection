"""
Generate full-AOI predicted urban-growth masks using the trained U-Net.

Inputs (per AOI):
    data/gee_exports/processed/
        <AOI>_ndvi_before.tif
        <AOI>_ndvi_after.tif
        <AOI>_ndvi_change.tif
        <AOI>_ndbi_before.tif
        <AOI>_ndbi_after.tif
        <AOI>_ndbi_change.tif

Outputs:
    data/processed/masks/predicted/
        <AOI>_urban_change_pred_prob.tif  (float32 probs)
        <AOI>_urban_change_pred_mask.tif  (uint8 0/1)

Run from project root:

    python -m src.change_detection.predict_full_aoi
"""

from pathlib import Path

import numpy as np
import rasterio
import torch
from tqdm import tqdm

from src.models.unet import UNet  # Attention U-Net alias
from src.preprocessing.compute_indices import AOI_NAMES  # list of AOI names


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GEE_PROC = PROJECT_ROOT / "data" / "gee_exports" / "processed"
TRUE_MASK_DIR = PROJECT_ROOT / "data" / "processed" / "masks"
PRED_MASK_DIR = TRUE_MASK_DIR / "predicted"
PRED_MASK_DIR.mkdir(exist_ok=True, parents=True)

CKPT_PATH = PROJECT_ROOT / "models" / "checkpoints" / "unet_best.pt"

PATCH_SIZE = 64
STRIDE = 32


def discover_aois():
    """Fallback if AOI_NAMES is not available."""
    names = []
    for f in GEE_PROC.glob("*_ndvi_before.tif"):
        aoi_name = f.stem.replace("_ndvi_before", "")
        names.append(aoi_name)
    return sorted(names)


def load_band(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        profile = src.profile
        nodata = src.nodata
    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr)
    return arr, profile


def build_image_stack(aoi: str):
    band_paths = {
        "ndvi_before": GEE_PROC / f"{aoi}_ndvi_before.tif",
        "ndvi_after":  GEE_PROC / f"{aoi}_ndvi_after.tif",
        "ndvi_change": GEE_PROC / f"{aoi}_ndvi_change.tif",
        "ndbi_before": GEE_PROC / f"{aoi}_ndbi_before.tif",
        "ndbi_after":  GEE_PROC / f"{aoi}_ndbi_after.tif",
        "ndbi_change": GEE_PROC / f"{aoi}_ndbi_change.tif",
    }

    arrays = []
    profile = None
    for name, path in band_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing band {name} for {aoi}: {path}")
        arr, profile = load_band(path)
        arrays.append(arr)

    stack = np.stack(arrays, axis=0)  # (6, H, W)
    stack = np.nan_to_num(stack, nan=0.0).astype("float32")
    return stack, profile


def predict_full_aoi(model, device, aoi: str):
    print(f"[INFO] Predicting AOI: {aoi}")
    image_stack, profile = build_image_stack(aoi)   # (6, H, W)
    _, H, W = image_stack.shape

    prob_map = np.zeros((H, W), dtype="float32")
    count_map = np.zeros((H, W), dtype="float32")

    # sliding-window inference
    for row in tqdm(range(0, H - PATCH_SIZE + 1, STRIDE), desc=f"{aoi} rows"):
        for col in range(0, W - PATCH_SIZE + 1, STRIDE):
            patch = image_stack[:, row:row + PATCH_SIZE, col:col + PATCH_SIZE]
            patch_t = torch.from_numpy(patch).unsqueeze(0).to(device)  # (1, 6, 64, 64)

            with torch.no_grad():
                logits = model(patch_t)
                probs = torch.sigmoid(logits).cpu().numpy()[0, 0, :, :]  # (64, 64)

            prob_map[row:row + PATCH_SIZE, col:col + PATCH_SIZE] += probs
            count_map[row:row + PATCH_SIZE, col:col + PATCH_SIZE] += 1.0

    # Avoid division by zero
    count_map[count_map == 0] = 1.0
    avg_prob = prob_map / count_map
    pred_mask = (avg_prob > 0.5).astype("uint8")

    # Save probability map
    prob_profile = profile.copy()
    prob_profile.update(dtype=rasterio.float32, count=1, nodata=None)

    prob_path = PRED_MASK_DIR / f"{aoi}_urban_change_pred_prob.tif"
    with rasterio.open(prob_path, "w", **prob_profile) as dst:
        dst.write(avg_prob.astype("float32"), 1)

    # Save binary mask
    mask_profile = profile.copy()
    mask_profile.update(dtype=rasterio.uint8, count=1, nodata=0)

    mask_path = PRED_MASK_DIR / f"{aoi}_urban_change_pred_mask.tif"
    with rasterio.open(mask_path, "w", **mask_profile) as dst:
        dst.write(pred_mask, 1)

    print(f"[OK] Saved {aoi} prob map -> {prob_path}")
    print(f"[OK] Saved {aoi} pred mask -> {mask_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Build model
    model = UNet(n_channels=6, n_classes=1).to(device)

    # 🔑 Correct checkpoint loading
    ckpt = torch.load(CKPT_PATH, map_location=device)
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt  # in case you ever save raw state_dict

    model.load_state_dict(state_dict)
    model.eval()

    # AOI list
    try:
        aois = AOI_NAMES
    except Exception:
        aois = discover_aois()

    print(f"[INFO] AOIs: {aois}")

    for aoi in aois:
        predict_full_aoi(model, device, aoi)


if __name__ == "__main__":
    main()