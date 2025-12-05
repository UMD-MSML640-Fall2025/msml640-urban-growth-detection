# Urban Growth Detection from Multi-Temporal Sentinel-2 Imagery

**MSML 640 – Computer Vision • Fall 2025**  
**Author:** Pragati Rao  

This project detects **urban growth between 2017 and 2023–2024** using multi-temporal Sentinel-2 imagery over 10 Areas of Interest (AOIs) in the United States.  

We combine:

- Classical **remote-sensing indices** (NDVI, NDBI, change maps)  
- A **patch-based Attention U-Net** segmentation model  
- **Sliding-window full-AOI inference** + visualizations and metrics  

The goal is to automatically highlight *where* new built-up structures appear and show how much urban area has expanded in each AOI.

---

## 1. Research Question

**Can we reliably detect medium-scale urban growth between 2017 and 2023–2024 using multi-temporal Sentinel-2 imagery, NDVI/NDBI change features, and a patch-based U-Net–style model?**

In particular, we ask:

- Where has urban land cover increased within each AOI?  
- Can an ML model generalize urban-change patterns across different US suburbs with limited labeled data?  
- How well do classical spectral indices plus a U-Net perform compared to simple index thresholding?

---

## 2. Data

### 2.1 AOIs

We use 10 AOIs inspired by SpaceNet-7 style regions (suburban growth hotspots):

- Apex_HaddonHall  
- Austin_Pflugerville  
- Charlotte_SteeleCreek  
- Dallas_Frisco  
- Denver_Stonegate  
- Houston_Spring  
- LasVegas_Inspirada  
- Phoenix_Gilbert  
- Raleigh_BrierCreek  
- Raleigh_NorthHills  

### 2.2 Source & Time Windows

- **Imagery:** Sentinel-2 Level-2A surface reflectance  
- **Before period:** 2017-01-01 to 2017-12-31  
- **After period:** 2023-01-01 to 2024-12-31  
- **Resolution:** ~10 m per pixel  

Cloud-masked median composites for each AOI and time window were exported from **Google Earth Engine** as GeoTIFFs.

Directory layout:

```text
data/
  gee_exports/
    raw_before/     # <AOI>_before.tif      (S2 composite ~2017)
    raw_after/      # <AOI>_after.tif       (S2 composite ~2023–24)
    processed/      # NDVI/NDBI per AOI (see below)
```

### 2.3 Derived Bands (Indices)

From the exported Sentinel-2 composites, we compute:

**NDVI (Normalized Difference Vegetation Index)**

- High NDVI → vegetation (parks, fields, forests)
- Low NDVI → built-up, bare soil, roads, rooftops

**NDBI (Normalized Difference Built-up Index)**

- Higher NDBI → built-up surfaces and impervious areas

For each AOI we create:

- `<AOI>_ndvi_before.tif`
- `<AOI>_ndvi_after.tif`
- `<AOI>_ndvi_change.tif` (= after – before)
- `<AOI>_ndbi_before.tif`
- `<AOI>_ndbi_after.tif`
- `<AOI>_ndbi_change.tif` (= after – before)

These live in:

```text
data/gee_exports/processed/
  <AOI>_ndvi_before.tif
  <AOI>_ndvi_after.tif
  <AOI>_ndvi_change.tif
  <AOI>_ndbi_before.tif
  <AOI>_ndbi_after.tif
  <AOI>_ndbi_change.tif
```

---

## 3. Ground-Truth (Pseudo-Labels)

Because hand-labeling urban change polygons is expensive, we use a heuristic pseudo-labeling approach:

- Index thresholds on NDVI and NDBI change (e.g., "vegetation decreased" + "built-up increased").
- Morphological cleaning to remove tiny speckles and enforce minimum object size.
- Export binary masks indicating "urban growth" pixels.

These pseudo-ground-truth masks are saved as:

```text
data/processed/masks/true/
  <AOI>_urban_change_mask.tif   # 0/1, heuristic "true" change
```

**Important caveat:**  
These masks are not manually labeled; they are derived from the same NDVI/NDBI indices used in the model input. That means evaluation primarily measures how well the model learns our thresholding heuristic rather than its performance against real human annotation.

---

## 4. Methods

### 4.1 Classical Indices → Change Features

**Script:** `src/preprocessing/compute_indices.py`

- Reads Sentinel-2 before/after composites.
- Computes NDVI and NDBI for both time periods.
- Computes change layers (after – before).
- Saves per-AOI GeoTIFFs in `data/gee_exports/processed/`.

These 6 channels form the input feature stack for the model:

```
[NDVI_before, NDVI_after, NDVI_change, NDBI_before, NDBI_after, NDBI_change]
```

### 4.2 Patch Extraction

**Script:** `src/preprocessing/patch_extractor.py`

Reads:

- The 6-channel NDVI/NDBI stacks
- The pseudo-ground-truth `*_urban_change_mask.tif`

Extracts 64×64 patches with stride 32.

Discards patches with too many NoData pixels.

For each patch, saves:

```text
data/processed/patches/<AOI>/
  patch_XXXX.npz   # contains:
                   #   image: (6, H, W) float32
                   #   mask:  (1, H, W) uint8  (0/1)
```

We later classify patches as:

- **Positive patches:** contain ≥ 1 urban-change pixel (mask > 0 anywhere).
- **Negative patches:** all zeros (no change).

For this project, we end up with roughly:

- ~290 positive patches
- ~635 negative patches
- → 1270 training samples, 231 validation samples after splitting.

### 4.3 Dataset & Data Augmentation

**Module:** `src/models/dataset.py`

Key features:

**PatchDataset:**

- Loads `.npz` patch files.
- Optional data augmentation:
  - Random horizontal / vertical flips
  - Random rotations (90°, 180°, 270°)
  - Brightness & contrast jitter for RGB channels
  - Small Gaussian noise

**BalancedPatchDataset:**

- Separates positive vs negative patches.
- Samples a controlled mix to mitigate class imbalance.
- Gives roughly 50/50 positive/negative samples in each batch.

**get_train_val_loaders(...):**

- Shuffles patch paths.
- Splits into train / validation sets.
- Instantiates the chosen dataset (balanced vs standard) with augmentation turned on for training and off for validation.

### 4.4 Model Architecture – Attention U-Net

**Module:** `src/models/unet.py`

We use an Attention U-Net with residual blocks:

**Encoder:**

- Stacked ResidualConv blocks with down-sampling (MaxPool).
- Channel progression: 6 → 64 → 128 → 256 → 512 → 512.

**Decoder:**

- Transposed convolution up-sampling.
- Attention gates on skip connections to focus on salient encoder features.
- Residual blocks after concatenation.

**Output:**

- 1-channel logits (same spatial size as input patch).
- Activation during inference: sigmoid → change probability per pixel.

Alternative architecture `DeepLabV3Plus` is also defined but not used as the main model in this version.

### 4.5 Loss Function & Training

**Module:** `src/models/train_unet.py`

**Loss:**

- Combined loss = 0.5 × Focal Loss + 0.5 × Dice Loss
- Focal loss handles class imbalance (few positive pixels).
- Dice loss focuses on overlap quality between masks.

**Optimizer & training tricks:**

- Optimizer: Adam(lr=1e-3, weight_decay=1e-5)
- LR scheduler: ReduceLROnPlateau on validation F1 (reduces LR when progress stalls)
- Gradient clipping: max_norm=1.0 to stabilize training
- Early stopping: stops if F1 doesn't improve for several epochs
- Training device: CPU (no GPU), so patch size and model depth are chosen to be tractable.

### 4.6 Full-AOI Inference

**Module:** `src/change_detection/predict_full_aoi.py`

For each AOI:

1. Builds a 6-channel (6, H, W) stack from the NDVI/NDBI GeoTIFFs.
2. Runs a sliding window over the full AOI:
   - Patch size = 64
   - Stride = 32
3. For each patch:
   - Run the model → predicted logits → sigmoid probabilities.
4. Aggregate overlapping predictions by averaging.
5. Threshold the final probability map at 0.5 → binary "urban change" prediction.

**Outputs:**

```text
data/processed/masks/predicted/
  <AOI>_urban_change_pred_prob.tif  # float32 [0,1]
  <AOI>_urban_change_pred_mask.tif  # uint8 0/1
```

---

## 5. Repository Structure

High-level layout:

```text
msml640-urban-growth-detection/
  data/
    gee_exports/
      raw_before/
      raw_after/
      processed/           # NDVI/NDBI before/after/change
    processed/
      masks/
        true/              # <AOI>_urban_change_mask.tif (pseudo GT)
        predicted/         # <AOI>_urban_change_pred_*.tif
      patches/             # .npz training patches
      samples/
        preview_plots/     # figures for report

  models/
    checkpoints/
      unet_best.pt         # best Attention U-Net weights

  output/
    processed/
      masks/
        predicted/
          full_aoi_metrics.csv

  src/
    preprocessing/
      compute_indices.py
      patch_extractor.py
    models/
      dataset.py
      unet.py              # Attention U-Net + DeepLabV3+ (optional)
      train_unet.py
    change_detection/
      predict_full_aoi.py
      eval_full_aoi.py
    visualization/
      full_aoi_maps.py     # 4-panel RGB+mask comparisons
      rgb_overlay.py       # overlay on RGB "after"
      inspect_predictions.py

  test/
    sanity_check.py        # small debug script

  requirements.txt
  README.md
```

---

## 6. How to Run

### 6.1 Environment Setup

```bash
# From project root
python -m venv .venv
source .venv/bin/activate        # Mac/Linux
# or .venv\Scripts\activate.bat   # Windows

pip install -r requirements.txt
```

Key dependencies:

- torch, torchvision
- numpy, pandas
- rasterio
- tqdm
- matplotlib

### 6.2 End-to-End Pipeline

**1. Compute indices (if not precomputed)**

```bash
python -m src.preprocessing.compute_indices
```

**2. Extract patches for training**

```bash
python -m src.preprocessing.patch_extractor \
    --patch-size 64 --stride 32
```

**3. Train Attention U-Net**

```bash
python -m src.models.train_unet
```

This writes the best model checkpoint to:

```text
models/checkpoints/unet_best.pt
```

**4. Run full-AOI inference**

```bash
python -m src.change_detection.predict_full_aoi
```

**5. Evaluate full-AOI masks (vs pseudo-labels)**

```bash
python -m src.change_detection.eval_full_aoi
```

Metrics are saved to:

```text
output/processed/masks/predicted/full_aoi_metrics.csv
```

**6. Generate visualizations**

Full AOI 4-panel comparisons:

```bash
python -m src.visualization.full_aoi_maps
```

Patch-level true vs predicted probabilities:

```bash
python -m src.visualization.inspect_predictions
```

RGB overlay with predicted change:

```bash
python -m src.visualization.rgb_overlay
```

All figures are saved under:

```text
data/processed/samples/preview_plots/
```

---

## 7. Results

### 7.1 Patch-Level Validation (Held-Out Patches)

Using balanced sampling, focal+dice loss, and data augmentation, the best model achieved approximately:

| Metric    | Value (validation, patch level) |
|-----------|----------------------------------|
| F1 score  | ~0.92                           |
| IoU       | ~0.86                           |
| Precision | high (model rarely predicts change where none exists) |
| Recall    | high (most change pixels are captured) |

**Note:** Pixel accuracy is near 0.99+ but is not very informative due to heavy class imbalance (vast majority of pixels are "no change").

### 7.2 Qualitative AOI-Level Observations

From the full-AOI comparison grids (`*_full_aoi_comparison.png`) and RGB overlays:

**Works well on:**

- **Charlotte_SteeleCreek:** clean detection of multiple new residential clusters, spatial alignment is strong.
- **Phoenix_Gilbert & Denver_Stonegate** (after improvements): captures main growth patches reasonably well.
- **Raleigh_NorthHills:** picks up scattered suburban infill with multiple small but correct change regions.

**Still challenging on:**

- AOIs with very subtle or tiny developments (e.g., a few new houses or narrow roads).
- Places where spectral signatures of new buildings are weakly different from existing rooftops or bare soil.
- Very sparse or linear patterns (e.g., small road extensions) where patch-level context is limited.

---

## 8. Evaluation Caveat (Pseudo-Labels)

The script `src/change_detection.eval_full_aoi.py` compares:

- `data/processed/masks/true/<AOI>_urban_change_mask.tif`
- `data/processed/masks/predicted/<AOI>_urban_change_pred_mask.tif`

Since the "true" masks are pseudo-labels derived from NDVI/NDBI thresholds (not human annotations), the model can effectively learn the thresholding rule itself.

As a result:

- Full-AOI metrics like IoU and F1 can appear very close to 1.0, showing that predictions match the pseudo-labels extremely well.
- This is a sanity check that the pipeline is internally consistent, not a realistic measure of performance against real-world ground truth.

In the report and README, we explicitly acknowledge that:

"The full-AOI metrics reflect consistency with our NDVI/NDBI pseudo-labeling heuristic, not true human-labeled accuracy. Future work should incorporate independent, manually validated ground truth."

---

## 9. Limitations & Future Work

### 9.1 Limitations

**Pseudo-ground truth:**

- Urban change masks are generated using NDVI/NDBI thresholds, not manual labels.
- Any biases/noise in the heuristic labels are learned by the model.

**Limited geographic diversity:**

- Only 10 AOIs, all US suburban regions.
- The model may not generalize to dense urban cores, rural areas, or other countries.

**Class imbalance:**

- Urban change pixels are a tiny fraction of the image.
- Balanced sampling + focal/dice loss mitigate this, but rare patterns may still be underrepresented.

**Spectral information:**

- Model uses only NDVI/NDBI derivatives (6 channels).
- Raw Sentinel-2 bands (e.g., SWIR, NIR, Red Edge) and/or SAR data could improve discrimination of built-up vs soil.

**Temporal simplification:**

- Only two composite time points ("before" vs "after").
- No explicit seasonal normalization or multi-year trajectories.

### 9.2 Future Work

- Incorporate manually annotated ground truth for at least a subset of AOIs.
- Experiment with full spectral stacks and additional indices (e.g., Urban Index, MNDWI).
- Try alternative architectures (e.g., DeepLabV3+, HRNet, Swin U-Net) and compare.
- Add multi-scale context (larger patches or pyramid features) to improve detection of small, linear structures.
- Evaluate on fully held-out cities to measure cross-region generalization.

---

## 10. Acknowledgements

- MSML 640 – Computer Vision, University of Maryland.
- Google Earth Engine for Sentinel-2 data access and cloud-masked composites.
- Inspiration from SpaceNet-7 multi-temporal urban change detection challenges.