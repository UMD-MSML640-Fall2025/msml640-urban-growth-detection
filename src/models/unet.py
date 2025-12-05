"""
Attention U-Net for urban growth detection.

Features:
- Residual encoder/decoder blocks
- Attention gates on skip connections
- Works with 6-channel input (e.g., NDVI_before, NDVI_after, NDBI_before, NDBI_after, etc.)

Input:  (B, 6, H, W)
Output: (B, 1, H, W)  -- logits (use sigmoid for probabilities)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGate(nn.Module):
    """Attention gate to highlight relevant encoder features."""

    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        g: gating signal from decoder (coarser features)
        x: encoder feature map (skip connection)
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class ResidualConv(nn.Module):
    """Residual block with two 3x3 convs + BN + ReLU."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        # 1x1 conv to match channel dimensions for residual
        self.residual = nn.Identity()
        if in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + residual
        out = self.relu(out)
        return out


class DownBlock(nn.Module):
    """Downsampling block: maxpool + residual conv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ResidualConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.conv(x)
        return x


class AttentionUpBlock(nn.Module):
    """
    Upsampling block with attention gate on the skip connection.

    in_channels:  channels of the decoder feature BEFORE upsampling (bottleneck side)
    out_channels: channels of the decoder feature AFTER upsampling
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Upsample: decoder goes from in_channels -> out_channels
        self.up = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )

        # Encoder feature map has in_channels channels (x_encoder),
        # decoder gating signal has out_channels channels (after upconv).
        self.attention = AttentionGate(
            F_g=out_channels,   # gating (decoder) channels
            F_l=in_channels,    # encoder skip channels
            F_int=out_channels // 2,
        )

        # After attention, we concatenate encoder (in_channels) + decoder (out_channels)
        self.conv = ResidualConv(in_channels + out_channels, out_channels)

    def forward(self, x_decoder: torch.Tensor, x_encoder: torch.Tensor) -> torch.Tensor:
        # Upsample decoder feature
        x_decoder = self.up(x_decoder)

        # Apply attention on encoder feature map using decoder as gate
        x_encoder = self.attention(g=x_decoder, x=x_encoder)

        # Handle odd spatial sizes
        diff_y = x_encoder.size(2) - x_decoder.size(2)
        diff_x = x_encoder.size(3) - x_decoder.size(3)
        x_decoder = F.pad(
            x_decoder,
            [diff_x // 2, diff_x - diff_x // 2,
             diff_y // 2, diff_y - diff_y // 2],
        )

        # Concatenate along channels and refine
        x = torch.cat([x_encoder, x_decoder], dim=1)
        x = self.conv(x)
        return x

class AttentionUNet(nn.Module):
    """
    Attention U-Net with residual encoder and attention-gated skip connections.
    """

    def __init__(self, n_channels: int = 6, n_classes: int = 1):
        super().__init__()

        # Encoder
        self.inc = ResidualConv(n_channels, 64)
        self.down1 = DownBlock(64, 128)
        self.down2 = DownBlock(128, 256)
        self.down3 = DownBlock(256, 512)
        self.down4 = DownBlock(512, 512)  # bottleneck

        # Decoder with attention
        self.up1 = AttentionUpBlock(512, 256)
        self.up2 = AttentionUpBlock(256, 128)
        self.up3 = AttentionUpBlock(128, 64)
        self.up4 = AttentionUpBlock(64, 64)

        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.inc(x)     # 64
        x2 = self.down1(x1)  # 128
        x3 = self.down2(x2)  # 256
        x4 = self.down3(x3)  # 512
        x5 = self.down4(x4)  # 512

        # Decoder
        x = self.up1(x5, x4)  # 256
        x = self.up2(x, x3)   # 128
        x = self.up3(x, x2)   # 64
        x = self.up4(x, x1)   # 64

        logits = self.outc(x)
        return logits


# Backwards-compatible alias: your training script can keep using "UNet"
class UNet(AttentionUNet):
    """Alias so existing code importing UNet still works."""
    def __init__(self, n_channels: int = 6, n_classes: int = 1):
        super().__init__(n_channels=n_channels, n_classes=n_classes)


if __name__ == "__main__":
    # Quick sanity check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionUNet(n_channels=6, n_classes=1).to(device)
    x = torch.randn(2, 6, 64, 64).to(device)
    y = model(x)
    print("Output shape:", y.shape)
    print("Params:", sum(p.numel() for p in model.parameters()))