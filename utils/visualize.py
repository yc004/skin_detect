"""
Visualization utilities for skin lesion detection results.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple


# Class names and colors (BGR format for OpenCV)
CLASS_CONFIG = {
    0: {"name": "skin_cancer", "color": (0, 0, 255), "risk": "HIGH"},           # Red
    1: {"name": "nevus", "color": (255, 200, 0), "risk": "LOW"},                # Cyan
    2: {"name": "actinic_keratosis", "color": (0, 200, 255), "risk": "MEDIUM"}, # Orange
    3: {"name": "seborrheic_keratosis", "color": (255, 255, 0), "risk": "LOW"}, # Yellow
    4: {"name": "benign_tumor", "color": (0, 255, 0), "risk": "LOW"},           # Green
    5: {"name": "vascular_lesion", "color": (255, 0, 255), "risk": "LOW"},      # Magenta
    6: {"name": "wart", "color": (128, 255, 128), "risk": "LOW"},               # Light Green
    7: {"name": "infestation_bite", "color": (255, 128, 128), "risk": "LOW"},   # Light Blue
}

FRIENDLY_NAMES = {
    "skin_cancer": "Skin Cancer",
    "nevus": "Nevus (Mole)",
    "actinic_keratosis": "Actinic Keratosis",
    "seborrheic_keratosis": "Seborrheic Keratosis",
    "benign_tumor": "Benign Tumor",
    "vascular_lesion": "Vascular Lesion",
    "wart": "Wart (Verruca)",
    "infestation_bite": "Infestation / Bite",
}

RISK_COLORS = {
    "HIGH": (0, 0, 255),
    "MEDIUM": (0, 165, 255),
    "LOW": (0, 255, 0),
}


def draw_detections(
    image: np.ndarray,
    boxes: np.ndarray,       # xyxy format, normalized or absolute
    classes: np.ndarray,     # class IDs
    confidences: np.ndarray, # confidence scores
    normalized: bool = True,
    thickness: int = 2,
    font_scale: float = 0.5,
    show_confidence: bool = True,
) -> np.ndarray:
    """
    Draw detection boxes on an image.

    Args:
        image: BGR image (H, W, 3)
        boxes: Nx4 array of bounding boxes
        classes: N array of class IDs
        confidences: N array of confidence scores
        normalized: Whether boxes are normalized (0-1)
        thickness: Box line thickness
        font_scale: Font scale for labels
        show_confidence: Whether to show confidence values

    Returns:
        Annotated BGR image.
    """
    img = image.copy()
    h, w = img.shape[:2]

    for box, cls_id, conf in zip(boxes, classes, confidences):
        cls_id = int(cls_id)
        if cls_id not in CLASS_CONFIG:
            continue

        # Convert boxes to pixel coordinates
        if normalized:
            x1, y1, x2, y2 = box
            x1, y1 = int(x1 * w), int(y1 * h)
            x2, y2 = int(x2 * w), int(y2 * h)
        else:
            x1, y1, x2, y2 = map(int, box)

        color = CLASS_CONFIG[cls_id]["color"]
        name = CLASS_CONFIG[cls_id]["name"]
        risk = CLASS_CONFIG[cls_id]["risk"]

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # Draw label with confidence
        if show_confidence:
            label = f"{FRIENDLY_NAMES.get(name, name)} {conf:.2f}"
        else:
            label = FRIENDLY_NAMES.get(name, name)

        # Background rectangle for text
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(img, (x1, y1 - text_h - 8), (x1 + text_w + 4, y1), color, -1)

        # Text
        cv2.putText(
            img, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA
        )

        # Risk indicator (small colored circle)
        if risk == "HIGH":
            cv2.circle(img, (x1 + 8, y1 + 8), 6, (0, 0, 255), -1)
            cv2.putText(img, "!", (x1 + 5, y1 + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return img


def draw_legend(image: np.ndarray, x: int = 10, y: int = 10) -> np.ndarray:
    """Draw a class-color legend on the image."""
    overlay = image.copy()
    line_h = 20
    padding = 5
    box_w = 15

    for cls_id, config in CLASS_CONFIG.items():
        color = config["color"]
        name = FRIENDLY_NAMES.get(config["name"], config["name"])
        risk = config["risk"]
        risk_color = RISK_COLORS[risk]

        # Color box
        cv2.rectangle(overlay, (x, y), (x + box_w, y + box_w), color, -1)
        cv2.rectangle(overlay, (x, y), (x + box_w, y + box_w), (50, 50, 50), 1)

        # Text
        cv2.putText(overlay, f"{name} [{risk}]", (x + box_w + padding, y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
        y += line_h

    # Semi-transparent overlay
    alpha = 0.85
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)


def create_comparison(
    images: List[np.ndarray],
    titles: List[str],
    output_path: str = None,
) -> np.ndarray:
    """
    Create a side-by-side comparison of detection results from different models.

    Args:
        images: List of annotated images
        titles: List of titles for each image
        output_path: Optional path to save the comparison

    Returns:
        Combined comparison image.
    """
    n = len(images)
    if n == 0:
        return None

    # Find max height, uniform width
    max_h = max(img.shape[0] for img in images)
    widths = [int(img.shape[1] * max_h / img.shape[0]) for img in images]
    total_w = sum(widths) + (n - 1) * 4  # 4px gap

    # Canvas
    canvas = np.zeros((max_h + 40, total_w, 3), dtype=np.uint8)
    canvas[:, :] = (40, 40, 40)

    x_offset = 0
    for i, (img, title) in enumerate(zip(images, titles)):
        # Resize to uniform height
        h, w = img.shape[:2]
        new_w = int(w * max_h / h)
        resized = cv2.resize(img, (new_w, max_h))

        canvas[:max_h, x_offset:x_offset + new_w] = resized

        # Title
        cv2.putText(canvas, title, (x_offset + 5, max_h + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)

        x_offset += new_w + 4

    if output_path:
        cv2.imwrite(str(output_path), canvas)
        print(f"Comparison saved to: {output_path}")

    return canvas


def generate_risk_report(detections: list) -> dict:
    """
    Generate a structured risk report from detections.

    Args:
        detections: List of (class_id, confidence, bbox) tuples

    Returns:
        Dict with lesion list, risk summary, and alert status.
    """
    report = {
        "total_lesions": len(detections),
        "high_risk_count": 0,
        "medium_risk_count": 0,
        "low_risk_count": 0,
        "alert": False,
        "alert_message": "",
        "lesions": [],
    }

    for cls_id, conf, bbox in detections:
        cls_id = int(cls_id)
        risk = CLASS_CONFIG.get(cls_id, {}).get("risk", "LOW")
        name = CLASS_CONFIG.get(cls_id, {}).get("name", "unknown")
        friendly = FRIENDLY_NAMES.get(name, name)

        if risk == "HIGH":
            report["high_risk_count"] += 1
        elif risk == "MEDIUM":
            report["medium_risk_count"] += 1
        else:
            report["low_risk_count"] += 1

        report["lesions"].append({
            "class": friendly,
            "confidence": float(conf),
            "risk": risk,
            "bbox": [float(b) for b in bbox],
        })

    if report["high_risk_count"] > 0:
        report["alert"] = True
        report["alert_message"] = (
            f"ALERT: {report['high_risk_count']} lesion(s) classified as skin cancer. "
            "Immediate dermatologist consultation is recommended."
        )
    elif report["medium_risk_count"] > 0:
        report["alert"] = True
        report["alert_message"] = (
            f"Note: {report['medium_risk_count']} lesion(s) with medium risk detected. "
            "Clinical follow-up is advised."
        )

    return report
