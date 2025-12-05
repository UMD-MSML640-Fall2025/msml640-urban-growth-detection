"""
Improved U-Net training for urban growth detection.

Features:
- Attention U-Net backbone (via models.unet.UNet alias)
- Combined Focal + Dice loss for severe class imbalance
- Data augmentation + balanced patch sampling (see dataset.py)
- Rich metrics: IoU, Precision, Recall, F1, Accuracy
- Learning-rate scheduling (ReduceLROnPlateau)
- Early stopping
- TensorBoard logging

Run:
    python -m src.models.train_unet
"""

from pathlib import Path
import time

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from .dataset import get_train_val_loaders
from .unet import UNet


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CKPT_DIR = PROJECT_ROOT / "models" / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------- Losses -----------------------------------------


class FocalLoss(nn.Module):
    """Binary Focal Loss for class imbalance."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)
            focal_weight = focal_weight * alpha_weight

        loss = focal_weight * bce
        return loss.mean()


class DiceLoss(nn.Module):
    """Soft Dice loss (1 - Dice coefficient)."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        probs = torch.sigmoid(logits)

        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        dice = (2 * intersection + self.smooth) / (
            probs_flat.sum() + targets_flat.sum() + self.smooth
        )
        return 1 - dice


class CombinedLoss(nn.Module):
    """Weighted sum of Focal and Dice losses."""

    def __init__(self, focal_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.focal = FocalLoss(alpha=0.25, gamma=2.0)
        self.dice = DiceLoss(smooth=1.0)
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        focal_loss = self.focal(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.focal_weight * focal_loss + self.dice_weight * dice_loss


# ----------------------------- Metrics ----------------------------------------


def compute_metrics(
    logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.35
):
    """
    Compute IoU, Precision, Recall, F1, Accuracy for a batch.
    """
    targets = targets.float()
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()
    tn = ((1 - preds) * (1 - targets)).sum()

    eps = 1e-7
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    accuracy = (tp + tn + eps) / (tp + tn + fp + fn + eps)

    return {
        "iou": iou.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item(),
        "accuracy": accuracy.item(),
    }


class EarlyStopping:
    """Stop training when validation metric stops improving."""

    def __init__(self, patience: int = 7, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_metric: float):
        if self.best_score is None:
            self.best_score = val_metric
            return

        if val_metric < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_metric
            self.counter = 0


# ----------------------------- Training loop ----------------------------------


def train(
    num_epochs: int = 50,
    batch_size: int = 16,
    lr: float = 1e-3,
    val_fraction: float = 0.2,
    min_positive_only: bool = False,
    use_focal_loss: bool = True,
    patience: int = 10,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    train_loader, val_loader = get_train_val_loaders(
        batch_size=batch_size,
        val_fraction=val_fraction,
        min_positive_only=min_positive_only,
        use_augmentation=True,
        use_balanced_sampling=True,
    )

    model = UNet(n_channels=6, n_classes=1).to(device)

    if use_focal_loss:
        criterion = CombinedLoss(focal_weight=0.5, dice_weight=0.5)
        print("[INFO] Loss: Combined Focal + Dice")
    else:
        pos_weight = torch.tensor([10.0], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"[INFO] Loss: Weighted BCE (pos_weight={pos_weight.item():.1f})")

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    early_stopper = EarlyStopping(patience=patience)

    writer = SummaryWriter(log_dir=str(PROJECT_ROOT / "runs" / "unet_improved"))

    best_val_f1 = 0.0
    best_val_iou = 0.0

    for epoch in range(1, num_epochs + 1):
        # ----------------- Train -----------------
        model.train()
        train_loss = 0.0
        train_metrics_sum = {"iou": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()

            # Gradient clipping for stability
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            batch_metrics = compute_metrics(logits, masks)
            for k in train_metrics_sum.keys():
                train_metrics_sum[k] += batch_metrics[k]

        avg_train_loss = train_loss / len(train_loader)
        train_metrics = {
            k: v / len(train_loader) for k, v in train_metrics_sum.items()
        }

        # ----------------- Validation -----------------
        model.eval()
        val_loss = 0.0
        val_metrics_sum = {
            "iou": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "accuracy": 0.0,
        }

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                logits = model(images)
                loss = criterion(logits, masks)
                val_loss += loss.item()

                batch_metrics = compute_metrics(logits, masks)
                for k in val_metrics_sum.keys():
                    val_metrics_sum[k] += batch_metrics[k]

        avg_val_loss = val_loss / len(val_loader)
        val_metrics = {
            k: v / len(val_loader) for k, v in val_metrics_sum.items()
        }

        # Logging
        print(
            f"[Epoch {epoch:03d}] "
            f"Loss {avg_train_loss:.4f}/{avg_val_loss:.4f} | "
            f"IoU {train_metrics['iou']:.3f}/{val_metrics['iou']:.3f} | "
            f"F1 {train_metrics['f1']:.3f}/{val_metrics['f1']:.3f} | "
            f"Rec {val_metrics['recall']:.3f} Prec {val_metrics['precision']:.3f}"
        )

        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        for k, v in val_metrics.items():
            writer.add_scalar(f"Metrics/val_{k}", v, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        # LR schedule + checkpointing on F1
        scheduler.step(val_metrics["f1"])

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_val_iou = val_metrics["iou"]

            ckpt_path = CKPT_DIR / "unet_best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_f1": best_val_f1,
                    "val_iou": best_val_iou,
                    "val_metrics": val_metrics,
                },
                ckpt_path,
            )
            print(
                f"[✓] New best model: F1={best_val_f1:.4f}, "
                f"IoU={best_val_iou:.4f} → {ckpt_path}"
            )

        # Early stopping
        early_stopper(val_metrics["f1"])
        if early_stopper.early_stop:
            print(f"[INFO] Early stopping at epoch {epoch}")
            break

    writer.close()
    print("\n[DONE] Training complete.")
    print(f"Best Validation F1:  {best_val_f1:.4f}")
    print(f"Best Validation IoU: {best_val_iou:.4f}")


if __name__ == "__main__":
    start = time.time()
    train(
        num_epochs=50,
        batch_size=8,      # safer on CPU with bigger model
        lr=1e-3,
        val_fraction=0.2,
        min_positive_only=False,
        use_focal_loss=True,
        patience=10,
    )
    print(f"[INFO] Finished in {(time.time() - start) / 60:.1f} minutes")