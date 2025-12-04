"""
Train a U-Net model on the extracted patches.

Usage (from project root):

    python src/models/train_unet.py

This will:
    - load patches from data/processed/patches/
    - build train/val DataLoaders
    - train U-Net for N epochs
    - compute IoU and pixel accuracy on val
    - save best model to models/checkpoints/unet_best.pt
"""

from pathlib import Path
import time

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter  # optional, but useful

from .dataset import get_train_val_loaders
from .unet import UNet


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CKPT_DIR = PROJECT_ROOT / "models" / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True, parents=True)


def iou_score(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Compute mean IoU over batch.
    logits: (B, 1, H, W), raw scores
    targets: (B, 1, H, W), 0/1
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection + 1e-6

    iou = (intersection + 1e-6) / union
    return iou.mean().item()


def pixel_accuracy(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    correct = (preds == targets).float().mean()
    return correct.item()


def train(
    num_epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-3,
    val_fraction: float = 0.2,
    min_positive_only: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    train_loader, val_loader = get_train_val_loaders(
        batch_size=batch_size,
        val_fraction=val_fraction,
        min_positive_only=min_positive_only,
    )

    model = UNet(n_channels=6, n_classes=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    writer = SummaryWriter(log_dir=str(PROJECT_ROOT / "runs" / "unet_training"))

    best_val_iou = 0.0

    global_step = 0
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0

        for images, masks in train_loader:
            images = images.to(device)  # (B, 6, 64, 64)
            masks = masks.to(device)    # (B, 1, 64, 64)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1

        avg_train_loss = epoch_loss / len(train_loader)

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        val_acc = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                logits = model(images)
                loss = criterion(logits, masks)

                val_loss += loss.item()
                val_iou += iou_score(logits, masks)
                val_acc += pixel_accuracy(logits, masks)

        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)
        avg_val_acc = val_acc / len(val_loader)

        print(
            f"[Epoch {epoch:03d}] "
            f"TrainLoss={avg_train_loss:.4f} "
            f"ValLoss={avg_val_loss:.4f} "
            f"IoU={avg_val_iou:.4f} "
            f"PixelAcc={avg_val_acc:.4f}"
        )

        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        writer.add_scalar("IoU/val", avg_val_iou, epoch)
        writer.add_scalar("PixelAcc/val", avg_val_acc, epoch)

        # Save best model by IoU
        if avg_val_iou > best_val_iou:
            best_val_iou = avg_val_iou
            ckpt_path = CKPT_DIR / "unet_best.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"[OK] New best IoU={best_val_iou:.4f}, saved to {ckpt_path}")

    writer.close()
    print(f"[DONE] Training complete. Best val IoU={best_val_iou:.4f}")


if __name__ == "__main__":
    start = time.time()
    train(
        num_epochs=20,
        batch_size=16,
        lr=1e-3,
        val_fraction=0.2,
        min_positive_only=False,  # you can set True to focus on positive patches only
    )
    print(f"[INFO] Finished in {(time.time() - start)/60:.1f} minutes")