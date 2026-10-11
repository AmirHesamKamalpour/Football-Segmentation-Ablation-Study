from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Manual multi-class focal loss for semantic segmentation.

    logits:  [B, C, H, W]
    targets: [B, H, W] with integer class IDs
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | list[float] | None = None,
        ignore_index: int = 255,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        if alpha is None:
            self.register_buffer("alpha", None)
        else:
            self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits,
            targets,
            reduction="none",
            ignore_index=self.ignore_index,
        )

        valid = targets != self.ignore_index
        pt = torch.exp(-ce)
        focal = (1.0 - pt).pow(self.gamma) * ce

        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            safe_targets = targets.clamp(min=0, max=alpha.numel() - 1)
            alpha_t = alpha[safe_targets]
            focal = alpha_t * focal

        focal = focal[valid]

        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        if self.reduction == "none":
            return focal
        raise ValueError(f"Unsupported reduction: {self.reduction}")
