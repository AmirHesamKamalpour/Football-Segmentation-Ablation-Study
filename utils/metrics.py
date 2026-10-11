from __future__ import annotations

import torch


def confusion_matrix_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int | None = None,
) -> torch.Tensor:
    preds = torch.argmax(logits, dim=1)
    return confusion_matrix_from_predictions(preds, targets, num_classes, ignore_index)


def confusion_matrix_from_predictions(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int | None = None,
) -> torch.Tensor:
    preds = preds.detach().flatten().long()
    targets = targets.detach().flatten().long()

    valid = (targets >= 0) & (targets < num_classes)
    if ignore_index is not None:
        valid &= targets != ignore_index

    preds = preds[valid]
    targets = targets[valid]

    indices = targets * num_classes + preds
    cm = torch.bincount(indices, minlength=num_classes * num_classes)
    return cm.reshape(num_classes, num_classes)


def iou_from_confusion_matrix(cm: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    tp = torch.diag(cm).float()
    fp = cm.sum(dim=0).float() - tp
    fn = cm.sum(dim=1).float() - tp
    return tp / (tp + fp + fn + eps)


def mean_iou_from_confusion_matrix(cm: torch.Tensor, eps: float = 1e-7) -> float:
    iou = iou_from_confusion_matrix(cm, eps)
    valid_classes = cm.sum(dim=1) > 0
    if valid_classes.any():
        return iou[valid_classes].mean().item()
    return 0.0


def dice_from_confusion_matrix(cm: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    tp = torch.diag(cm).float()
    fp = cm.sum(dim=0).float() - tp
    fn = cm.sum(dim=1).float() - tp
    return (2 * tp) / (2 * tp + fp + fn + eps)


def pixel_accuracy_from_confusion_matrix(cm: torch.Tensor, eps: float = 1e-7) -> float:
    return (torch.diag(cm).sum().float() / (cm.sum().float() + eps)).item()
