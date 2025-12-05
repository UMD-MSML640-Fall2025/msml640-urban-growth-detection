"""
Visualize *positive* patch-level predictions:

For a few random patches that contain urban change, show:

    [ TRUE MASK | PREDICTED PROBABILITIES ]

Output:
    data/processed/samples/preview_plots/patch_predictions.png

Run from project root:

    python -m src.visualization.inspect_predictions
"""

from pathlib import Path
import random

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.models.unet import AttentionUNet
from src.models.dataset import PATCH_DIR  # same PATCH_DIR as dataset.py


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CKPT_PATH = PROJECT_ROOT / "models" / "checkpoints" / "unet_best.pt"

OUT_DIR = PROJECT_ROOT / "data" / "processed" / "samples" / "preview_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "patch_predictions.png"


def load_model(device: torch.device) -> AttentionUNet:
    """Load the trained Attention U-Net from checkpoint."""
    print(f"[INFO] Loading model from {CKPT_PATH}")
    model = AttentionUNet(n_channels=6, n_classes=1).to(device)

    ckpt = torch.load(CKPT_PATH, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict)
    model.eval()
    return model


def collect_positive_patches(min_pos_pixels: int = 10):
    """
    Collect .npz patches that actually contain urban change.

    min_pos_pixels: minimum number of positive pixels required in the mask
    """
    pos_paths = []
    for aoi_dir in PATCH_DIR.glob("*"):
        if not aoi_dir.is_dir():
            continue
        for npz_file in aoi_dir.glob("*.npz"):
            data = np.load(npz_file)
            mask = data["mask"]  # (1, H, W)
            if (mask > 0).sum() >= min_pos_pixels:
                pos_paths.append(npz_file)

    if not pos_paths:
        raise RuntimeError(
            f"No positive patches (>= {min_pos_pixels} pixels) found under {PATCH_DIR}"
        )

    print(f"[INFO] Found {len(pos_paths)} positive patches (>= {min_pos_pixels} pixels).")
    return sorted(pos_paths)


def visualize_random_patches(num_examples: int = 6, min_pos_pixels: int = 10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    model = load_model(device)
    patch_paths = collect_positive_patches(min_pos_pixels=min_pos_pixels)

    if num_examples > len(patch_paths):
        num_examples = len(patch_paths)

    random.seed(42)
    sample_paths = random.sample(patch_paths, num_examples)

    n_rows = num_examples
    n_cols = 2  # true vs predicted probs

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(6 * n_cols, 3 * n_rows), squeeze=False
    )

    for row_idx, path in enumerate(sample_paths):
        data = np.load(path)
        image = data["image"]      # (C, H, W)
        mask_true = data["mask"]   # (1, H, W)

        # Model prediction
        img_tensor = torch.from_numpy(image).unsqueeze(0).to(device)  # (1, 6, H, W)
        with torch.no_grad():
            logits = model(img_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0, 0]  # (H, W)
        # hard mask if you want it
        # mask_pred = (probs > 0.5).astype("float32")

        # TRUE mask
        ax_true = axes[row_idx, 0]
        ax_true.imshow(mask_true[0], cmap="Reds", vmin=0, vmax=1)
        ax_true.set_title(f"True mask\n{path.parent.name}")
        ax_true.axis("off")

        # PRED probability map
        ax_pred = axes[row_idx, 1]
        im = ax_pred.imshow(probs, cmap="Blues", vmin=0, vmax=1)
        ax_pred.set_title("Predicted probabilities")
        ax_pred.axis("off")

    fig.suptitle("Random *Positive* Patch Predictions – True vs Predicted", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(OUT_PATH, dpi=200)
    plt.close(fig)
    print(f"[OK] Saved visualization to {OUT_PATH}")


if __name__ == "__main__":
    visualize_random_patches(num_examples=6, min_pos_pixels=10)