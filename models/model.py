from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two Conv(+BN)+ReLU blocks with padding=1 to preserve H,W."""

    def __init__(self, in_channels: int, out_channels: int, use_batch_norm: bool = False) -> None:
        super().__init__()
        layers = []
        for i in range(2):
            conv_in = in_channels if i == 0 else out_channels
            layers.append(nn.Conv2d(conv_in, out_channels, kernel_size=3, padding=1, bias=not use_batch_norm))
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        up_mode: Literal["bilinear", "transposed"] = "bilinear",
        use_skip_connections: bool = True,
        use_batch_norm: bool = False,
    ) -> None:
        super().__init__()
        self.use_skip_connections = use_skip_connections

        if up_mode == "bilinear":
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            )
        elif up_mode == "transposed":
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        else:
            raise ValueError("up_mode must be either 'bilinear' or 'transposed'")

        conv_in = out_channels + skip_channels if use_skip_connections else out_channels
        self.conv = DoubleConv(conv_in, out_channels, use_batch_norm=use_batch_norm)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        x = self.up(x)

        if self.use_skip_connections:
            if skip is None:
                raise ValueError("skip tensor is required when skip connections are enabled.")

            # With 256x256 input and padding=1, cropping is not needed.
            # This fallback only handles odd-sized custom inputs.
            if skip.shape[-2:] != x.shape[-2:]:
                skip = F.interpolate(skip, size=x.shape[-2:], mode="bilinear", align_corners=True)

            x = torch.cat([x, skip], dim=1)

        return self.conv(x)


class UNet(nn.Module):
    """Configurable U-Net for 9-class football semantic segmentation.

    Expected input:  [B, 3, 256, 256]
    Output logits:   [B, num_classes, 256, 256]
    Prediction:      logits.argmax(dim=1) -> [B, 256, 256]
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 9,
        base_channels: int = 64,
        up_mode: Literal["bilinear", "transposed"] = "bilinear",
        use_skip_connections: bool = True,
        use_batch_norm: bool = False,
    ) -> None:
        super().__init__()

        c1, c2, c3, c4, c5 = [base_channels * (2 ** i) for i in range(5)]

        self.enc1 = DoubleConv(in_channels, c1, use_batch_norm)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(c1, c2, use_batch_norm)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(c2, c3, use_batch_norm)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = DoubleConv(c3, c4, use_batch_norm)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(c4, c5, use_batch_norm)

        self.up4 = UpBlock(c5, c4, c4, up_mode, use_skip_connections, use_batch_norm)
        self.up3 = UpBlock(c4, c3, c3, up_mode, use_skip_connections, use_batch_norm)
        self.up2 = UpBlock(c3, c2, c2, up_mode, use_skip_connections, use_batch_norm)
        self.up1 = UpBlock(c2, c1, c1, up_mode, use_skip_connections, use_batch_norm)

        self.classifier = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc1 = self.enc1(x)          # [B, 64, 256, 256]
        x = self.pool1(enc1)         # [B, 64, 128, 128]

        enc2 = self.enc2(x)          # [B, 128, 128, 128]
        x = self.pool2(enc2)         # [B, 128, 64, 64]

        enc3 = self.enc3(x)          # [B, 256, 64, 64]
        x = self.pool3(enc3)         # [B, 256, 32, 32]

        enc4 = self.enc4(x)          # [B, 512, 32, 32]
        x = self.pool4(enc4)         # [B, 512, 16, 16]

        x = self.bottleneck(x)       # [B, 1024, 16, 16]

        x = self.up4(x, enc4)        # [B, 512, 32, 32]
        x = self.up3(x, enc3)        # [B, 256, 64, 64]
        x = self.up2(x, enc2)        # [B, 128, 128, 128]
        x = self.up1(x, enc1)        # [B, 64, 256, 256]

        return self.classifier(x)    # [B, num_classes, 256, 256]


def build_model(cfg: dict) -> UNet:
    model_cfg = cfg["model"]
    return UNet(
        in_channels=model_cfg.get("in_channels", 3),
        num_classes=model_cfg.get("num_classes", 9),
        base_channels=model_cfg.get("base_channels", 64),
        up_mode=model_cfg.get("up_mode", "bilinear"),
        use_skip_connections=model_cfg.get("use_skip_connections", True),
        use_batch_norm=model_cfg.get("use_batch_norm", False),
    )


if __name__ == "__main__":
    model = UNet(num_classes=9, up_mode="bilinear", use_skip_connections=True)
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    print(y.shape)
    assert y.shape == (2, 9, 256, 256)
