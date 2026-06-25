"""
Visualization utilities for skin disease classification.

Includes:
  - Inference: result overlay, Grad-CAM heatmap
  - Training: loss/acc curves, confusion matrix
  - Analysis: ROC curves, per-class metrics, misclassified samples
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import torch
import torch.nn.functional as F


# ============================================================
# Inference Visualization
# ============================================================

RISK_COLORS_BGR = {
    "HIGH":   (0, 0, 255),
    "MEDIUM": (0, 165, 255),
    "LOW":    (0, 255, 0),
}


def draw_classification_result(
    image: np.ndarray,
    predictions: List[Dict],
    class_risk: Dict[str, str] = None,
) -> np.ndarray:
    """Draw top-K predictions on the image with confidence bars."""
    img = image.copy()
    h, w = img.shape[:2]

    panel_w = min(320, w // 2)
    overlay = np.zeros((h, panel_w, 3), dtype=np.uint8)
    overlay[:, :] = (30, 30, 35)
    img[:, -panel_w:] = cv2.addWeighted(img[:, -panel_w:], 0.3, overlay, 0.7, 0)

    cv2.putText(img, "Classification Result", (w - panel_w + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.line(img, (w - panel_w + 10, 38), (w - 10, 38), (80, 80, 80), 1)

    y = 65
    for i, pred in enumerate(predictions):
        name = pred.get("class_zh", pred.get("class", "unknown"))
        conf = pred["confidence"]
        risk = pred.get("risk", "LOW")
        color = RISK_COLORS_BGR.get(risk, (180, 180, 180))

        bar_w = int((panel_w - 30) * conf)
        cv2.rectangle(img, (w - panel_w + 12, y), (w - panel_w + 12 + bar_w, y + 18), color, -1)
        cv2.rectangle(img, (w - panel_w + 12, y), (w - panel_w + panel_w - 14, y + 18), (60, 60, 60), 1)

        risk_icon = {"HIGH": "!!", "MEDIUM": "! ", "LOW": "  "}.get(risk, "  ")
        cv2.putText(img, f"{risk_icon} {name}", (w - panel_w + 16, y + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, f"{conf:.1%}", (w - 45, y + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)
        y += 28

    top = predictions[0]
    top_name = top.get("class_zh", top.get("class", "unknown"))
    top_risk = top.get("risk", "LOW")
    banner_color = RISK_COLORS_BGR.get(top_risk, (100, 100, 100))
    cv2.rectangle(img, (0, 0), (w, 28), banner_color, -1)
    banner_text = f"Prediction: {top_name} ({top['confidence']:.1%})"
    if top_risk == "HIGH":
        banner_text = f"!! HIGH RISK: {top_name} ({top['confidence']:.1%}) — Seek Clinical Advice"
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
    """Generate Grad-CAM heatmap overlay."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_bgr.shape[:2]

    tensor = transform(image_rgb).unsqueeze(0).to(device)

    target_layer = None
    for name, module in model.named_modules():
        if name == target_layer_name:
            target_layer = module
            break

    if target_layer is None:
        for name, module in model.named_modules():
            if "stages.3" in name and isinstance(module, torch.nn.Conv2d):
                target_layer = module
                break

    if target_layer is None:
        return image_bgr

    activations = {}
    gradients = {}

    def forward_hook(module, input, output):
        activations["value"] = output

    def backward_hook(module, grad_in, grad_out):
        gradients["value"] = grad_out[0]

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    model.zero_grad()
    logits = model(tensor)
    pred_idx = logits.argmax(dim=1).item()
    logits[0, pred_idx].backward()

    fh.remove()
    bh.remove()

    act = activations["value"].detach()
    grad = gradients["value"].detach()

    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = (weights * act).sum(dim=1).squeeze(0)
    cam = F.relu(cam)
    if cam.max() > 0:
        cam = cam / cam.max()

    cam = cam.cpu().numpy()
    cam = cv2.resize(cam, (w, h))
    cam = np.uint8(255 * cam)

    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    result = cv2.addWeighted(image_bgr, 0.5, heatmap, 0.5, 0)

    cv2.putText(result, f"Grad-CAM: {class_names[pred_idx]}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    return result


# ============================================================
# Training Curves
# ============================================================

def plot_training_curves(history: dict, save_path: str):
    """Plot loss and accuracy curves with best-epoch annotation."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss", linewidth=1.5)
    ax1.plot(epochs, history["val_loss"], "r-", label="Val Loss", linewidth=1.5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax2.plot(epochs, history["train_acc"], "b-", label="Train Acc", linewidth=1.5)
    ax2.plot(epochs, history["val_acc"], "r-", label="Val Acc", linewidth=1.5)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Mark best
    if history["val_acc"]:
        best_idx = history["val_acc"].index(max(history["val_acc"]))
        ax2.annotate(f"Best: {history['val_acc'][best_idx]:.4f}",
                     xy=(best_idx + 1, history['val_acc'][best_idx]),
                     xytext=(best_idx + 1 + 2, history['val_acc'][best_idx] - 0.05),
                     arrowprops=dict(arrowstyle="->", color="green"),
                     fontsize=10, color="green")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Training curves → {save_path}")


# ============================================================
# Confusion Matrix
# ============================================================

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: str,
    normalize: bool = True,
    figsize: tuple = (20, 18),
    title: str = "Confusion Matrix",
):
    """Plot and save a confusion matrix (normalized + raw counts)."""
    from sklearn.metrics import confusion_matrix as cm_func

    cm = cm_func(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

    n = len(class_names)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    # Annotate
    for i in range(n):
        for j in range(n):
            if cm[i, j] > 0:
                text = f"{cm[i, j]}\n({cm_norm[i, j]:.1%})"
                color = "white" if cm_norm[i, j] > 0.6 else "black"
                ax.text(j, i, text, ha="center", va="center", fontsize=5.5, color=color)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix → {save_path}")

    return cm


# ============================================================
# ROC Curves (One-vs-Rest)
# ============================================================

def plot_roc_curves(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    class_names: List[str],
    save_path: str,
    figsize: tuple = (20, 16),
):
    """
    Plot one-vs-rest ROC curves for each class with AUC scores.

    Args:
        y_true: (N,) integer labels
        y_probs: (N, C) predicted probabilities
        class_names: List of class names
        save_path: Output path
    """
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize

    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=range(n_classes))

    # Compute ROC for each class
    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Micro-average
    fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), y_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot
    n_cols = 4
    n_rows = (n_classes + n_cols - 1) // n_cols + 1  # +1 for micro-average summary

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    # Micro-average on first subplot
    ax = axes[0]
    ax.plot(fpr["micro"], tpr["micro"], color="darkred", lw=2,
            label=f"Micro-avg (AUC={roc_auc['micro']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Micro-Average ROC")
    ax.legend(loc="lower right", fontsize=7)
    ax.grid(True, alpha=0.2)

    # Per-class
    colors = plt.cm.tab20(np.linspace(0, 1, n_classes))
    for i in range(n_classes):
        ax = axes[i + 1]
        ax.plot(fpr[i], tpr[i], color=colors[i], lw=1.5,
                label=f"AUC={roc_auc[i]:.3f}")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_title(f"{class_names[i]}", fontsize=8)
        ax.legend(loc="lower right", fontsize=6)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=6)

    # Hide unused subplots
    for i in range(n_classes + 1, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle("ROC Curves — One-vs-Rest", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ROC curves → {save_path}")

    # Return AUC summary
    auc_summary = {class_names[i]: float(roc_auc[i]) for i in range(n_classes)}
    auc_summary["micro_avg"] = float(roc_auc["micro"])
    return auc_summary


# ============================================================
# Per-Class Metrics Bar Chart
# ============================================================

def plot_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: str,
    figsize: tuple = (22, 10),
):
    """
    Plot per-class precision, recall, F1 as grouped bar chart.

    Returns: dict of per-class metrics
    """
    from sklearn.metrics import precision_recall_fscore_support

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, zero_division=0
    )

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=figsize)

    bars1 = ax.bar(x - width, precision, width, label="Precision", color="#4fc3f7", edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x, recall, width, label="Recall", color="#81c784", edgecolor="white", linewidth=0.5)
    bars3 = ax.bar(x + width, f1, width, label="F1 Score", color="#ce93d8", edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Class")
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Metrics: Precision, Recall, F1")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=7)
    ax.set_ylim([0, 1.05])
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.2, axis="y")

    # Annotate F1 values
    for i, v in enumerate(f1):
        ax.text(i + width, v + 0.02, f"{v:.2f}", ha="center", fontsize=6, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Per-class metrics → {save_path}")

    # Build metrics dict
    metrics = {}
    for i, name in enumerate(class_names):
        metrics[name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
    return metrics


# ============================================================
# Misclassified Samples Grid
# ============================================================

def plot_misclassified(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    images: List[np.ndarray],
    class_names: List[str],
    save_path: str,
    top_n: int = 16,
):
    """
    Show a grid of the most confidently wrong predictions.

    Args:
        y_true, y_pred: labels
        y_probs: (N, C) probabilities
        images: List of BGR images (must align with y_true indices)
        class_names: Class name list
        save_path: Output path
        top_n: Number of errors to show
    """
    # Find misclassified indices
    errors = np.where(y_true != y_pred)[0]
    if len(errors) == 0:
        print("  No misclassified samples to show.")
        return

    # Sort by confidence (most confident errors first)
    confidences = y_probs[errors, y_pred[errors]]
    sorted_idx = np.argsort(confidences)[::-1]  # descending
    errors = errors[sorted_idx][:top_n]

    n = len(errors)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    if rows * cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, idx in enumerate(errors):
        ax = axes[i]
        img_rgb = cv2.cvtColor(images[idx], cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
        ax.axis("off")

        true_name = class_names[y_true[idx]]
        pred_name = class_names[y_pred[idx]]
        conf = y_probs[idx, y_pred[idx]]
        ax.set_title(f"True: {true_name}\nPred: {pred_name} ({conf:.1%})",
                     fontsize=8, color="red", fontweight="bold")

    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle(f"Top {n} Misclassified Samples (Most Confident Errors)", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    if n > 0:
        print(f"  Misclassified samples → {save_path}")
