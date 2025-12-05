from pathlib import Path
import numpy as np
import rasterio

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRUE = PROJECT_ROOT / "data" / "processed" / "masks" / "true"
PRED = PROJECT_ROOT / "data" / "processed" / "masks" / "predicted"

aoi = "Phoenix_Gilbert"

true_path = TRUE / f"{aoi}_urban_change_mask.tif"
pred_path = PRED / f"{aoi}_urban_change_pred_mask.tif"

print("TRUE path:", true_path)
print("PRED path:", pred_path)
print("TRUE exists?", true_path.exists())
print("PRED exists?", pred_path.exists())

with rasterio.open(true_path) as src:
    t = src.read(1)

with rasterio.open(pred_path) as src:
    p = src.read(1)

diff = (t != p).sum()
print("Num differing pixels:", diff)