"""
Boundary-Sensitive Loss for YOLOv8
Adds a boundary distance penalty to the standard detection loss.

The boundary loss penalizes misalignment between predicted and ground-truth
box edges (left, right, top, bottom), encouraging tighter box fit — critical
for medical imaging where lesion boundary precision matters.

Formula:
    L_boundary = mean(|pred_l - gt_l| + |pred_r - gt_r| + |pred_t - gt_t| + |pred_b - gt_b|)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def compute_boundary_loss(
    pred_boxes: torch.Tensor,  # (N, 4) in xyxy format (normalized)
    gt_boxes: torch.Tensor,    # (N, 4) in xyxy format (normalized)
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute boundary-sensitive loss between predicted and GT boxes.

    Computes L1 distance for each of the 4 box edges independently,
    normalized by the GT box dimensions to give equal weight to small and large boxes.

    Args:
        pred_boxes: Predicted bounding boxes in xyxy format (N, 4)
        gt_boxes: Ground truth bounding boxes in xyxy format (N, 4)
        reduction: "mean" | "sum" | "none"

    Returns:
        Boundary loss tensor.
    """
    # Split into edges: left, top, right, bottom
    pred_l, pred_t, pred_r, pred_b = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3]
    gt_l, gt_t, gt_r, gt_b = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2], gt_boxes[:, 3]

    # Box dimensions for normalization
    gt_w = (gt_r - gt_l).clamp(min=1e-7)
    gt_h = (gt_b - gt_t).clamp(min=1e-7)

    # L1 distance for each edge, normalized by corresponding dimension
    loss_l = torch.abs(pred_l - gt_l) / gt_w
    loss_r = torch.abs(pred_r - gt_r) / gt_w
    loss_t = torch.abs(pred_t - gt_t) / gt_h
    loss_b = torch.abs(pred_b - gt_b) / gt_h

    # Sum of edge losses
    loss = loss_l + loss_r + loss_t + loss_b

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


class BoundaryLoss(nn.Module):
    """
    Standalone boundary loss module.

    Args:
        reduction: "mean" | "sum" | "none"
        weight:    Weight multiplier for this loss term
    """

    def __init__(self, reduction: str = "mean", weight: float = 0.5):
        super().__init__()
        self.reduction = reduction
        self.weight = weight

    def forward(self, pred_boxes: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
        return self.weight * compute_boundary_loss(pred_boxes, gt_boxes, self.reduction)


def ciou_loss(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Complete IoU loss (reference implementation).
    Used when we want to compute it independently.

    Args:
        pred: Predicted boxes in xyxy (N, 4)
        gt:   Ground truth boxes in xyxy (N, 4)

    Returns:
        CIoU loss (1 - CIoU)
    """
    # Convert xyxy to xywh
    pred_xywh = torch.zeros_like(pred)
    pred_xywh[:, 0] = (pred[:, 0] + pred[:, 2]) / 2  # cx
    pred_xywh[:, 1] = (pred[:, 1] + pred[:, 3]) / 2  # cy
    pred_xywh[:, 2] = pred[:, 2] - pred[:, 0]         # w
    pred_xywh[:, 3] = pred[:, 3] - pred[:, 1]         # h

    gt_xywh = torch.zeros_like(gt)
    gt_xywh[:, 0] = (gt[:, 0] + gt[:, 2]) / 2
    gt_xywh[:, 1] = (gt[:, 1] + gt[:, 3]) / 2
    gt_xywh[:, 2] = gt[:, 2] - gt[:, 0]
    gt_xywh[:, 3] = gt[:, 3] - gt[:, 1]

    # IoU
    inter_x1 = torch.max(pred[:, 0], gt[:, 0])
    inter_y1 = torch.max(pred[:, 1], gt[:, 1])
    inter_x2 = torch.min(pred[:, 2], gt[:, 2])
    inter_y2 = torch.min(pred[:, 3], gt[:, 3])
    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    pred_area = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    gt_area = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
    union = pred_area + gt_area - inter_area + eps
    iou = inter_area / union

    # Center distance
    c2 = (pred_xywh[:, 0] - gt_xywh[:, 0]) ** 2 + (pred_xywh[:, 1] - gt_xywh[:, 1]) ** 2
    # Enclosing box
    enclose_x1 = torch.min(pred[:, 0], gt[:, 0])
    enclose_y1 = torch.min(pred[:, 1], gt[:, 1])
    enclose_x2 = torch.max(pred[:, 2], gt[:, 2])
    enclose_y2 = torch.max(pred[:, 3], gt[:, 3])
    enclose_diag = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2 + eps

    # Aspect ratio
    v = (4 / (math.pi ** 2)) * ((torch.atan(gt_xywh[:, 2] / (gt_xywh[:, 3] + eps)) -
                                  torch.atan(pred_xywh[:, 2] / (pred_xywh[:, 3] + eps))) ** 2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    ciou = iou - (c2 / enclose_diag + alpha * v)
    return 1 - ciou


def combined_loss(
    pred_boxes: torch.Tensor,
    gt_boxes: torch.Tensor,
    boundary_weight: float = 0.5,
    iou_weight: float = 1.0,
) -> dict:
    """
    Compute combined CIoU + Boundary loss.

    Returns dict with individual components for logging.
    """
    loss_iou = ciou_loss(pred_boxes, gt_boxes).mean()
    loss_boundary = compute_boundary_loss(pred_boxes, gt_boxes)

    total = iou_weight * loss_iou + boundary_weight * loss_boundary

    return {
        "loss_iou": loss_iou.item() if isinstance(loss_iou, torch.Tensor) else loss_iou,
        "loss_boundary": loss_boundary.item(),
        "loss_total": total.item(),
    }
