#!/usr/bin/env python3
"""
Skin Disease Classification — Gradio Interactive Demo.

Features:
  - Upload skin images for disease classification
  - Show top-3 predictions with confidence bars
  - Risk level alerts for high-risk conditions
  - Optional Grad-CAM visualization
"""

import sys
import os
import json
from pathlib import Path

import cv2
import numpy as np
import gradio as gr

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import timm

from utils.visualize import draw_classification_result, draw_gradcam


# ============================================================
# Model Loading
# ============================================================

MODEL = None
CLASS_NAMES = []
CLASS_RISK = {}
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_PATH = "runs/convnext/best.pt"
TRANSFORM = None


def load_model_once():
    """Load model on first request (lazy)."""
    global MODEL, CLASS_NAMES, CLASS_RISK, TRANSFORM

    if MODEL is not None:
        return

    if not os.path.exists(MODEL_PATH):
        return

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    CLASS_NAMES = checkpoint.get("class_names", [])
    CLASS_RISK = checkpoint.get("class_risk", {})

    if not CLASS_NAMES:
        config_path = Path(MODEL_PATH).parent / "class_config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            CLASS_NAMES = config["class_names"]
            CLASS_RISK = config.get("class_risk", {})

    num_classes = len(CLASS_NAMES)
    MODEL = timm.create_model(
        "convnext_tiny.fb_in22k_ft_in1k",
        pretrained=False,
        num_classes=num_classes,
    )
    MODEL.load_state_dict(checkpoint["model_state_dict"])
    MODEL = MODEL.to(DEVICE)
    MODEL.eval()

    TRANSFORM = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(int(224 * 1.14)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    print(f"Model loaded: {MODEL_PATH} ({num_classes} classes) on {DEVICE}")


# ============================================================
# Inference
# ============================================================

@torch.no_grad()
def classify(image: np.ndarray, show_cam: bool = False):
    """Run classification and return annotated image + report HTML."""
    load_model_once()

    if MODEL is None:
        return image, "<p style='color:red'>Model not found. Train first: python train.py</p>"

    if image is None:
        return None, "<p style='color:gray'>No image provided.</p>"

    h, w = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Inference
    tensor = TRANSFORM(image_rgb).unsqueeze(0).to(DEVICE)
    logits = MODEL(tensor)
    probs = F.softmax(logits, dim=1)

    topk_probs, topk_indices = torch.topk(probs, min(3, len(CLASS_NAMES)))
    topk_probs = topk_probs.cpu().numpy()[0]
    topk_indices = topk_indices.cpu().numpy()[0]

    predictions = []
    for prob, idx in zip(topk_probs, topk_indices):
        name = CLASS_NAMES[idx]
        risk = CLASS_RISK.get(name, "LOW")
        predictions.append({"class": name, "confidence": float(prob), "risk": risk})

    # Draw result
    annotated = draw_classification_result(image, predictions, CLASS_RISK)

    # Build report
    report_html = _build_report(predictions)

    # Grad-CAM
    if show_cam:
        cam_image = draw_gradcam(MODEL, image, CLASS_NAMES, TRANSFORM, DEVICE)
        return annotated, report_html, cam_image

    return annotated, report_html, None


# ============================================================
# Report HTML
# ============================================================

def _build_report(predictions: list) -> str:
    """Build HTML report."""
    top = predictions[0]
    risk = top["risk"]
    risk_color = {"HIGH": "#ef5350", "MEDIUM": "#ff9800", "LOW": "#81c784"}[risk]
    risk_icon = {"HIGH": "🚨", "MEDIUM": "⚠️", "LOW": "✅"}[risk]

    rows = ""
    for i, pred in enumerate(predictions):
        r_color = {"HIGH": "#ef5350", "MEDIUM": "#ff9800", "LOW": "#81c784"}[pred["risk"]]
        r_icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}[pred["risk"]]
        bold = "font-weight:bold;" if i == 0 else ""
        rows += f"""
        <tr style="{bold}">
            <td style="padding:8px; border-bottom:1px solid #333;">{r_icon}</td>
            <td style="padding:8px; border-bottom:1px solid #333;">{pred['class']}</td>
            <td style="padding:8px; border-bottom:1px solid #333; color:{r_color};">{pred['risk']}</td>
            <td style="padding:8px; border-bottom:1px solid #333;">{pred['confidence']:.1%}</td>
        </tr>"""

    alert_html = ""
    if risk == "HIGH":
        alert_html = f"""
        <div style="padding:15px; background:#3a1a1a; border-radius:8px; margin:10px 0; border-left:4px solid {risk_color};">
            <span style="font-size:20px;">{risk_icon}</span>
            <b style="color:{risk_color};">HIGH RISK: {top['class']}</b>
            <p style="color:#ccc; margin:5px 0 0 0;">Immediate dermatologist consultation is recommended.</p>
        </div>"""
    elif risk == "MEDIUM":
        alert_html = f"""
        <div style="padding:15px; background:#2a2a1a; border-radius:8px; margin:10px 0; border-left:4px solid {risk_color};">
            <span style="font-size:20px;">{risk_icon}</span>
            <b style="color:{risk_color};">MEDIUM RISK: {top['class']}</b>
            <p style="color:#ccc; margin:5px 0 0 0;">Clinical follow-up is advised.</p>
        </div>"""
    else:
        alert_html = f"""
        <div style="padding:15px; background:#1a2a1a; border-radius:8px; margin:10px 0; border-left:4px solid {risk_color};">
            <span style="font-size:20px;">{risk_icon}</span>
            <b style="color:{risk_color};">Low Risk: {top['class']}</b>
            <p style="color:#ccc; margin:5px 0 0 0;">Likely benign. Routine monitoring suggested.</p>
        </div>"""

    return f"""
    <div style="padding:20px; background:#1a1a2e; border-radius:12px; color:#e0e0e0; font-family:sans-serif;">
        <h3 style="margin-top:0; color:#4fc3f7;">🔬 Classification Results</h3>
        <p style="color:#aaa;">Model: <b>ConvNeXt-Tiny</b> | 22 classes</p>

        {alert_html}

        <table style="width:100%; border-collapse:collapse; margin-top:15px;">
        <thead>
            <tr style="background:#2a2a3e;">
                <th style="padding:8px; text-align:left;"></th>
                <th style="padding:8px; text-align:left;">Condition</th>
                <th style="padding:8px; text-align:left;">Risk</th>
                <th style="padding:8px; text-align:left;">Confidence</th>
            </tr>
        </thead>
        <tbody>{rows}</tbody>
        </table>

        <div style="margin-top:15px; padding:10px; background:#1a1a1a; border-radius:6px; font-size:11px; color:#888;">
            <p>⚠️ <i>This is an AI research tool. All results require clinical verification.</i></p>
        </div>
    </div>
    """


# ============================================================
# Gradio UI
# ============================================================

def build_ui():
    theme = gr.themes.Soft(primary_hue="blue", secondary_hue="slate")

    with gr.Blocks(
        theme=theme,
        title="Skin Disease Classification — ConvNeXt",
        css="""
        footer { visibility: hidden; }
        """
    ) as demo:
        gr.Markdown("""
        # 🩺 Skin Disease Classification
        ### AI-Assisted 22-Class Skin Disease Diagnosis with ConvNeXt-Tiny
        ---
        """)

        with gr.Tabs():
            # Tab 1: Classification
            with gr.TabItem("📸 Classify"):
                with gr.Row():
                    with gr.Column(scale=3):
                        input_image = gr.Image(label="Upload Skin Image", type="numpy", height=480)
                        with gr.Row():
                            cam_checkbox = gr.Checkbox(label="Show Grad-CAM Heatmap", value=False)
                        classify_btn = gr.Button("🔬 Analyze", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        output_image = gr.Image(label="Result", type="numpy", height=480)
                        report_html = gr.HTML(label="Report")

                with gr.Row():
                    cam_output = gr.Image(label="Grad-CAM (Activation Map)", type="numpy", visible=True, height=300)

                classify_btn.click(
                    fn=classify,
                    inputs=[input_image, cam_checkbox],
                    outputs=[output_image, report_html, cam_output],
                )

            # Tab 2: About
            with gr.TabItem("ℹ️ About"):
                gr.Markdown("""
                ## About This System

                ### 🧠 Model
                - **Architecture**: ConvNeXt-Tiny (pretrained on ImageNet-22k)
                - **Task**: 22-class skin disease classification
                - **Input**: 224×224 RGB image

                ### 🎯 Classes (22)
                | 🔴 HIGH Risk | 🟡 MEDIUM Risk | 🟢 LOW Risk |
                |-------------|---------------|------------|
                | Skin Cancer | Actinic Keratosis | Acne, Benign Tumors, Bullous |
                | | Lupus, Vasculitis | Candidiasis, Drug Eruption, Eczema |
                | | | Infestations/Bites, Lichen, Moles |
                | | | Psoriasis, Rosacea, Seborrheic Keratoses |
                | | | Sun/Sunlight Damage, Tinea, Unknown/Normal |
                | | | Vascular Tumors, Vitiligo, Warts |

                ### ⚠️ Disclaimer
                **This is a research tool and NOT a medical device.** All results require
                verification by a qualified dermatologist.
                """)

        gr.Markdown("""
        ---
        <div style="text-align:center; color:#666; font-size:12px;">
        Skin Disease Classification | ConvNeXt-Tiny | Research Prototype | Not for clinical use
        </div>
        """)

    return demo


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Launch Skin Disease Classification Demo")
    parser.add_argument("--port", type=int, default=7860, help="Gradio server port")
    parser.add_argument("--share", action="store_true", help="Create public link")
    args = parser.parse_args()

    # Check model
    if not os.path.exists(MODEL_PATH):
        print("=" * 60)
        print("⚠️  MODEL NOT FOUND")
        print(f"   Expected: {MODEL_PATH}")
        print("   Run: python train.py")
        print("=" * 60)

    demo = build_ui()
    demo.queue(max_size=10)
    demo.launch(server_port=args.port, share=args.share, show_error=True)


if __name__ == "__main__":
    main()
