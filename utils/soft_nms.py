"""
Soft-NMS: Soft Non-Maximum Suppression
Reference: "Soft-NMS — Improving Object Detection With One Line of Code" (ICCV 2017)

Uses Gaussian penalty to decay scores of overlapping boxes instead of
hard-suppressing them, preventing missed detections in dense lesion scenarios.
"""

import torch


def soft_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.5,
    sigma: float = 0.5,
    score_threshold: float = 0.001,
    method: str = "gaussian",
) -> tuple:
    """
    Soft-NMS for bounding box suppression.

    Args:
        boxes:          Tensor of shape (N, 4) in xyxy format
        scores:         Tensor of shape (N,) with confidence scores
        iou_threshold:  IoU threshold for overlap suppression
        sigma:          Gaussian penalty sigma
        score_threshold: Minimum score to keep a box
        method:         "gaussian" | "linear"

    Returns:
        keep:       Indices of kept boxes
        new_scores: Updated scores for kept boxes
    """
    if boxes.numel() == 0:
        return torch.tensor([], dtype=torch.long, device=boxes.device), scores

    # Sort by score descending
    _, order = scores.sort(descending=True)
    boxes = boxes[order]
    scores = scores[order]

    keep = []
    while order.numel() > 0:
        if order.numel() == 1:
            keep.append(order[0])
            break

        # Keep the highest-score box
        keep.append(order[0])

        # Compute IoU of the kept box (first) vs the rest
        ious = _box_iou(boxes[0:1], boxes[1:])

        if method == "gaussian":
            penalty = torch.exp(-(ious * ious) / sigma)
        elif method == "linear":
            mask = ious < iou_threshold
            penalty = torch.where(mask, torch.ones_like(ious), 1 - ious)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Decay scores
        scores[1:] = scores[1:] * penalty.squeeze(0)

        # Remove boxes that have been fully decayed
        remaining_mask = scores[1:] > score_threshold

        # Prepare for next iteration
        boxes = boxes[1:][remaining_mask]
        scores = scores[1:][remaining_mask]
        order = order[1:][remaining_mask]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device), scores


def _box_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """Compute IoU between box1 and each box in box2 (broadcast)."""
    # box1: (1, 4), box2: (M, 4)

    # Intersection
    inter_x1 = torch.max(box1[:, 0], box2[:, 0])
    inter_y1 = torch.max(box1[:, 1], box2[:, 1])
    inter_x2 = torch.min(box1[:, 2], box2[:, 2])
    inter_y2 = torch.min(box1[:, 3], box2[:, 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    # Union
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union_area = area1 + area2 - inter_area

    return inter_area / (union_area + 1e-7)
