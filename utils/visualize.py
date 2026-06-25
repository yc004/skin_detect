"""
Visualization utilities for skin disease classification.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F


# ============================================================
# Inference Visualization
# ============================================================

RISK_COLORS_BGR = {
    "HIGH":   (0, 0, 255),    # Red
    "MEDIUM": (0, 165, 255),  # Orange
    "LOW":    (0, 255, 0),    # Green
}


def draw_classification_result(
    image: np.ndarray,
    predictions: List[Dict],
    class_risk: Dict[str, str] = None,
) -> np.ndarray:
    """
    Draw top-K classification results on the image.

    Args:
        image: BGR image
        predictions: List of {class, confidence, risk} dicts
        class_risk: Optional risk mapping (not used here, risk comes from predictions)

    Returns:
        Annotated BGR image.
    """
    img = image.copy()
    h, w = img.shape[:2]

    # Semi-transparent overlay panel on the right
    panel_w = min(320, w // 2)
    overlay = np.zeros((h, panel_w, 3), dtype=np.uint8)
    overlay[:, :] = (30, 30, 35)

    # Blend panel
    img[:, -panel_w:] = cv2.addWeighted(img[:, -panel_w:], 0.3, overlay, 0.7, 0)

    # Title
    cv2.putText(img, "Classification Result", (w - panel_w + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.line(img, (w - panel_w + 10, 38), (w - 10, 38), (80, 80, 80), 1)

    y = 65
    for i, pred in enumerate(predictions):
        name = pred["class"]
        conf = pred["confidence"]
        risk = pred.get("risk", "LOW")
        color = RISK_COLORS_BGR.get(risk, (180, 180, 180))

        # Confidence bar
        bar_w = int((panel_w - 30) * conf)
        cv2.rectangle(img, (w - panel_w + 12, y), (w - panel_w + 12 + bar_w, y + 18), color, -1)
        cv2.rectangle(img, (w - panel_w + 12, y), (w - panel_w + panel_w - 14, y + 18), (60, 60, 60), 1)

        # Text
        risk_icon = {"HIGH": "!!", "MEDIUM": "! ", "LOW": "  "}.get(risk, "  ")
        label = f"{risk_icon} {name}"
        cv2.putText(img, label, (w - panel_w + 16, y + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

        # Percentage
        pct = f"{conf:.1%}"
        cv2.putText(img, pct, (w - 45, y + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)

        y += 28

    # Top prediction banner
    top = predictions[0]
    top_risk = top.get("risk", "LOW")
    banner_color = RISK_COLORS_BGR.get(top_risk, (100, 100, 100))

    cv2.rectangle(img, (0, 0), (w, 28), banner_color, -1)
    banner_text = f"Prediction: {top['class']} ({top['confidence']:.1%})"
    if top_risk == "HIGH":
        banner_text = f"!! HIGH RISK: {top['class']} ({top['confidence']:.1%}) — Seek Clinical Advice"
    cv2.putText(img, banner_text, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return img


# ============================================================
# Grad-CAM
# ============================================================

def draw_gradcam(
    model,
    image_bgr: np.ndarray,
    class_names: List[str],
    transform,
    device: str = "mps",
    target_layer_name: str = "stages.3",
) -> np.ndarray:
    """
    Generate Grad-CAM heatmap overlay for ConvNeXt.

    Args:
        model: ConvNeXt model
        image_bgr: Input BGR image
        class_names: List of class names
        transform: Preprocessing transform
        device: Device
        target_layer_name: Target layer for Grad-CAM (ConvNeXt stage 3 = last spatial)

    Returns:
        BGR image with Grad-CAM heatmap overlay.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_bgr.shape[:2]

    tensor = transform(image_rgb).unsqueeze(0).to(device)

    # Find target layer
    target_layer = None
    for name, module in model.named_modules():
        if name == target_layer_name:
            target_layer = module
            break

    if target_layer is None:
        # Fallback: use last Conv2d in stage 3
        for name, module in model.named_modules():
            if "stages.3" in name and isinstance(module, torch.nn.Conv2d):
                target_layer = module

    if target_layer is None:
        print("[WARN] Could not find target layer for Grad-CAM, skipping")
        return image_bgr

    activations = {}
    gradients = {}

    def forward_hook(module, input, output):
        activations["value"] = output

    def backward_hook(module, grad_in, grad_out):
        gradients["value"] = grad_out[0]

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    # Forward pass
    model.zero_grad()
    logits = model(tensor)
    pred_idx = logits.argmax(dim=1).item()

    # Backward
    logits[0, pred_idx].backward()

    fh.remove()
    bh.remove()

    # Generate heatmap
    act = activations["value"].detach()    # (1, C, H', W')
    grad = gradients["value"].detach()      # (1, C, H', W')

    # Global average pooling of gradients
    weights = grad.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
    cam = (weights * act).sum(dim=1).squeeze(0)     # (H', W')

    cam = F.relu(cam)
    if cam.max() > 0:
        cam = cam / cam.max()

    # Resize to original image size
    cam = cam.cpu().numpy()
    cam = cv2.resize(cam, (w, h))
    cam = np.uint8(255 * cam)

    # Apply colormap and overlay
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    result = cv2.addWeighted(image_bgr, 0.5, heatmap, 0.5, 0)

    # Add label
    class_name = class_names[pred_idx]
    cv2.putText(result, f"Grad-CAM: {class_name}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    return result


# ============================================================
# Training Visualization
# ============================================================

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: str,
    figsize: tuple = (18, 16),
):
    """
    Plot and save a confusion matrix.
    """
    from sklearn.metrics import confusion_matrix as cm_func

    cm = cm_func(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

    # Labels
    n = len(class_names)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix ({n} classes)")

    # Annotate cells with count + percentage
    for i in range(n):
        for j in range(n):
            if cm[i, j] > 0:
                text = f"{cm[i, j]}\n({cm_norm[i, j]:.1%})"
                color = "white" if cm_norm[i, j] > 0.6 else "black"
                ax.text(j, i, text, ha="center", va="center",
                        fontsize=6, color=color)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix → {save_path}")


def plot_training_curves(history: dict, save_path: str):
    """
    Plot training/validation loss and accuracy curves.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss", linewidth=1.5)
    ax1.plot(epochs, history["val_loss"], "r-", label="Val Loss", linewidth=1.5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, history["train_acc"], "b-", label="Train Acc", linewidth=1.5)
    ax2.plot(epochs, history["val_acc"], "r-", label="Val Acc", linewidth=1.5)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Mark best epoch
    best_idx = history["val_acc"].index(max(history["val_acc"]))
    ax2.annotate(f"Best: {history['val_acc'][best_idx]:.4f}",
                 xy=(best_idx + 1, history["val_acc"][best_idx]),
                 xytext=(best_idx + 1 + 2, history["val_acc"][best_idx] - 0.05),
                 arrowprops=dict(arrowstyle="->", color="green"),
                 fontsize=10, color="green")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Training curves → {save_path}")
