#!/usr/bin/env python3
"""
Skin Lesion Detection — Gradio Interactive Demo

Features:
  - Upload skin images for lesion detection
  - Select model variant (baseline/boundary/improved)
  - Compare two models side-by-side
  - Show risk report with high-risk alerts
  - Live webcam capture mode
"""

import sys
import os
from pathlib import Path

import cv2
import numpy as np
import gradio as gr

# Register custom modules before anything else
import ultralytics.nn.modules as ul_nn_modules
import ultralytics.nn.tasks as ul_tasks
from models.attention import CoordAtt
from models.asff import ASFFHead

ul_nn_modules.CoordAtt = CoordAtt
ul_nn_modules.ASFFHead = ASFFHead
ul_tasks.CoordAtt = CoordAtt
ul_tasks.ASFFHead = ASFFHead

from ultralytics import YOLO

from utils.visualize import (
    draw_detections, draw_legend, create_comparison,
    generate_risk_report, FRIENDLY_NAMES, CLASS_CONFIG,
)
from utils.soft_nms import soft_nms

import torch


# ============================================================
# Model Cache
# ============================================================

MODEL_CACHE = {}
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

MODEL_PATHS = {
    "baseline": "runs/baseline/weights/best.pt",
    "boundary": "runs/boundary/weights/best.pt",
    "improved": "runs/improved/weights/best.pt",
}

MODEL_DESCRIPTIONS = {
    "baseline": "YOLOv8s Standard (CIoU + DFL)",
    "boundary": "YOLOv8s + Boundary-Sensitive Loss",
    "improved": "YOLOv8s + ASFF + CA + Boundary Loss",
}


def get_model(variant: str):
    """Load or retrieve cached model."""
    if variant not in MODEL_CACHE:
        path = MODEL_PATHS.get(variant)
        if not path or not os.path.exists(path):
            return None, f"Model '{variant}' not found at {path}\nPlease train this variant first."
        try:
            model = YOLO(path)
            if DEVICE != "cpu":
                model.to(DEVICE)
            MODEL_CACHE[variant] = model
        except Exception as e:
            return None, f"Failed to load model: {e}"
    return MODEL_CACHE[variant], None


# ============================================================
# Detection Logic
# ============================================================

def detect(
    image: np.ndarray,
    model_variant: str = "improved",
    conf_threshold: float = 0.25,
    use_soft_nms: bool = False,
) -> tuple:
    """
    Run detection on an image.

    Returns: (annotated_image, report_html)
    """
    if image is None:
        return None, "<p style='color:gray'>No image provided.</p>"

    model, error = get_model(model_variant)
    if error:
        return image, f"<p style='color:red'>{error}</p>"

    # Run inference
    results = model.predict(
        image, imgsz=640, conf=conf_threshold, iou=0.45, verbose=False,
    )

    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        annotated = image.copy()
        report_html = _build_report_html(None, model_variant)
        return annotated, report_html

    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()

    # Soft-NMS
    if use_soft_nms and len(boxes) > 1:
        boxes_t, scores_t = torch.from_numpy(boxes).float(), torch.from_numpy(confidences).float()
        keep, updated = soft_nms(boxes_t, scores_t, sigma=0.5)
        boxes = boxes[keep.numpy()]
        classes = classes[keep.numpy()]
        confidences = updated[keep.numpy()].numpy()

    # Visualize
    annotated = draw_detections(image, boxes, classes, confidences, normalized=False)

    # Build report
    det_list = [(c, conf, b) for c, conf, b in zip(classes, confidences, boxes)]
    report = generate_risk_report(det_list)
    report_html = _build_report_html(report, model_variant)

    return annotated, report_html


def compare_detect(
    image: np.ndarray,
    model_a: str = "baseline",
    model_b: str = "improved",
    conf_threshold: float = 0.25,
) -> np.ndarray:
    """Compare two model variants side-by-side."""
    if image is None:
        return None

    images = []
    titles = []

    for variant in [model_a, model_b]:
        model, error = get_model(variant)
        if error:
            err_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
            cv2.putText(err_img, f"Model not available", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 255), 2)
            images.append(err_img)
            titles.append(f"{variant}: ERROR")
            continue

        results = model.predict(image, imgsz=640, conf=conf_threshold, iou=0.45, verbose=False)
        result = results[0]

        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            annotated = draw_detections(image.copy(), boxes, classes, confidences, normalized=False)
        else:
            annotated = image.copy()

        annotated = draw_legend(annotated)
        images.append(annotated)
        titles.append(f"{MODEL_DESCRIPTIONS.get(variant, variant)}")

    return create_comparison(images, titles)


# ============================================================
# Report Builder
# ============================================================

def _build_report_html(report: dict, model_variant: str) -> str:
    """Build an HTML report string for display in Gradio."""
    model_label = MODEL_DESCRIPTIONS.get(model_variant, model_variant)

    if report is None or report["total_lesions"] == 0:
        return f"""
        <div style="padding:20px; background:#1a1a2e; border-radius:12px; color:#e0e0e0; font-family:sans-serif;">
            <h3 style="margin-top:0; color:#4fc3f7;">🔬 Detection Results</h3>
            <p style="color:#aaa;">Model: <b>{model_label}</b></p>
            <div style="padding:15px; background:#2a2a3e; border-radius:8px; margin:10px 0;">
                <p style="color:#81c784; font-size:16px;">✅ No lesions detected</p>
            </div>
            <p style="color:#888; font-size:12px;">This image appears clean. If clinical concern exists, consult a dermatologist.</p>
        </div>
        """

    # Risk summary
    if report["alert"]:
        alert_color = "#ef5350" if report["high_risk_count"] > 0 else "#ff9800"
        alert_icon = "🚨" if report["high_risk_count"] > 0 else "⚠️"
    else:
        alert_color = "#81c784"
        alert_icon = "✅"

    lesion_rows = ""
    for i, lesion in enumerate(report["lesions"]):
        risk_color = {"HIGH": "#ef5350", "MEDIUM": "#ff9800", "LOW": "#81c784"}[lesion["risk"]]
        lesion_rows += f"""
        <tr>
            <td style="padding:8px; border-bottom:1px solid #333;">{i+1}</td>
            <td style="padding:8px; border-bottom:1px solid #333;">{lesion['class']}</td>
            <td style="padding:8px; border-bottom:1px solid #333; color:{risk_color};">{lesion['risk']}</td>
            <td style="padding:8px; border-bottom:1px solid #333;">{lesion['confidence']:.1%}</td>
        </tr>"""

    return f"""
    <div style="padding:20px; background:#1a1a2e; border-radius:12px; color:#e0e0e0; font-family:sans-serif; max-height:600px; overflow-y:auto;">
        <h3 style="margin-top:0; color:#4fc3f7;">🔬 Detection Results</h3>
        <p style="color:#aaa;">Model: <b>{model_label}</b></p>
        <p style="color:#aaa;">Lesions found: <b>{report['total_lesions']}</b></p>

        <div style="padding:15px; background:#2a2a3e; border-radius:8px; margin:10px 0; border-left:4px solid {alert_color};">
            <span style="font-size:20px;">{alert_icon}</span>
            <b style="color:{alert_color};">{report['alert_message'] if report['alert'] else 'No high-risk findings'}</b>
        </div>

        <table style="width:100%; border-collapse:collapse; margin-top:15px;">
        <thead>
            <tr style="background:#2a2a3e;">
                <th style="padding:8px; text-align:left;">#</th>
                <th style="padding:8px; text-align:left;">Lesion Type</th>
                <th style="padding:8px; text-align:left;">Risk</th>
                <th style="padding:8px; text-align:left;">Confidence</th>
            </tr>
        </thead>
        <tbody>
            {lesion_rows}
        </tbody>
        </table>

        <div style="margin-top:15px; padding:10px; background:#1a1a1a; border-radius:6px; font-size:11px; color:#888;">
            <p><b>Risk Key:</b>
            <span style="color:#ef5350;">🔴 HIGH — Suspected malignancy, immediate referral</span> |
            <span style="color:#ff9800;">🟡 MEDIUM — Clinical follow-up advised</span> |
            <span style="color:#81c784;">🟢 LOW — Benign appearance</span>
            </p>
            <p>⚠️ <i>This is an AI research tool. All results require clinical verification.</i></p>
        </div>
    </div>
    """


# ============================================================
# Gradio UI
# ============================================================

def build_ui():
    """Build the Gradio interface."""
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
    )

    with gr.Blocks(
        theme=theme,
        title="Skin Lesion Detection System",
        css="""
        .risk-high { color: #ef5350 !important; font-weight: bold; }
        .risk-medium { color: #ff9800 !important; }
        .risk-low { color: #81c784 !important; }
        footer { visibility: hidden; }
        """
    ) as demo:
        gr.Markdown("""
        # 🩺 Skin Lesion Detection System
        ### AI-Assisted Multi-Class Skin Lesion Detection with Improved YOLOv8
        ---
        """)

        with gr.Tabs():
            # Tab 1: Single image detection
            with gr.TabItem("📸 Image Detection"):
                with gr.Row():
                    with gr.Column(scale=3):
                        with gr.Row():
                            input_image = gr.Image(
                                label="Upload Skin Image",
                                type="numpy",
                                height=480,
                            )
                        with gr.Row():
                            model_select = gr.Dropdown(
                                choices=list(MODEL_PATHS.keys()),
                                value="improved",
                                label="Model Variant",
                                info="Choose which model to use for detection",
                            )
                        with gr.Row():
                            conf_slider = gr.Slider(
                                minimum=0.1, maximum=0.9, value=0.25, step=0.05,
                                label="Confidence Threshold",
                                info="Lower = more detections, higher = fewer false positives",
                            )
                            soft_nms_check = gr.Checkbox(
                                label="Use Soft-NMS",
                                value=False,
                                info="Prevents suppression in dense lesions",
                            )
                        detect_btn = gr.Button("🔍 Detect Lesions", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        output_image = gr.Image(
                            label="Detection Results",
                            type="numpy",
                            height=480,
                        )
                        report_html = gr.HTML(label="Detection Report")

                detect_btn.click(
                    fn=detect,
                    inputs=[input_image, model_select, conf_slider, soft_nms_check],
                    outputs=[output_image, report_html],
                )

            # Tab 2: Model Comparison
            with gr.TabItem("⚖️ Model Comparison"):
                with gr.Row():
                    with gr.Column(scale=1):
                        compare_image = gr.Image(
                            label="Upload Skin Image",
                            type="numpy",
                            height=400,
                        )
                        with gr.Row():
                            model_a_select = gr.Dropdown(
                                choices=list(MODEL_PATHS.keys()),
                                value="baseline",
                                label="Model A (Left)",
                            )
                            model_b_select = gr.Dropdown(
                                choices=list(MODEL_PATHS.keys()),
                                value="improved",
                                label="Model B (Right)",
                            )
                        compare_conf = gr.Slider(
                            minimum=0.1, maximum=0.9, value=0.25, step=0.05,
                            label="Confidence Threshold",
                        )
                        compare_btn = gr.Button("🔍 Compare Models", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        comparison_output = gr.Image(
                            label="Side-by-Side Comparison",
                            type="numpy",
                            height=500,
                        )

                compare_btn.click(
                    fn=compare_detect,
                    inputs=[compare_image, model_a_select, model_b_select, compare_conf],
                    outputs=[comparison_output],
                )

            # Tab 3: Webcam / Live Capture
            with gr.TabItem("📷 Live Capture"):
                gr.Markdown("""
                ### 📷 Real-Time Lesion Detection

                Use your webcam or upload a screen capture for live detection.
                Works best with dermoscopic images displayed on screen.
                """)

                with gr.Row():
                    webcam_input = gr.Image(
                        label="Webcam / Screen Capture",
                        type="numpy",
                        source="webcam",
                        streaming=True,
                        height=400,
                    )
                    webcam_output = gr.Image(
                        label="Live Detection",
                        type="numpy",
                        height=400,
                    )

                with gr.Row():
                    webcam_model = gr.Dropdown(
                        choices=list(MODEL_PATHS.keys()),
                        value="improved",
                        label="Model",
                        scale=1,
                    )
                    webcam_conf = gr.Slider(
                        0.1, 0.9, 0.25, step=0.05,
                        label="Confidence",
                        scale=1,
                    )
                    webcam_btn = gr.Button("▶️ Start Detection", variant="primary", scale=1)

                webcam_btn.click(
                    fn=detect,
                    inputs=[webcam_input, webcam_model, webcam_conf, gr.Checkbox(value=False, visible=False)],
                    outputs=[webcam_output, gr.HTML(visible=False)],
                )

            # Tab 4: About
            with gr.TabItem("ℹ️ About"):
                gr.Markdown("""
                ## About This System

                This is a research prototype for **AI-assisted skin lesion detection** based on an improved YOLOv8 architecture.

                ### 🧠 Model Variants
                | Variant | Architecture | Loss |
                |---------|-------------|------|
                | **Baseline** | Standard YOLOv8s | CIoU + DFL |
                | **Boundary** | Standard YOLOv8s | CIoU + DFL + Boundary-Sensitive |
                | **Improved** | YOLOv8s + CA + ASFF | CIoU + DFL + Boundary-Sensitive |

                ### 🔧 Improvements
                - **Coordinate Attention (CA)**: Enhances spatial localization for irregular lesion shapes
                - **Adaptive Spatial Feature Fusion (ASFF)**: Multi-scale fusion for varying lesion sizes
                - **Boundary-Sensitive Loss**: Tighter bounding boxes around lesion edges
                - **Soft-NMS**: Prevents missed detections in dense multi-lesion cases

                ### 🎯 Detection Classes
                - 🔴 **Skin Cancer** (HIGH RISK)
                - 🟡 **Actinic Keratosis** (MEDIUM RISK)
                - 🟢 **Nevus (Mole)** (LOW RISK)
                - 🟢 **Seborrheic Keratosis** (LOW RISK)
                - 🟢 **Benign Tumor** (LOW RISK)
                - 🟢 **Vascular Lesion** (LOW RISK)
                - 🟢 **Wart** (LOW RISK)
                - 🟢 **Infestation / Bite** (LOW RISK)

                ### ⚠️ Disclaimer
                **This is a research tool and NOT a medical device.** All results require
                verification by a qualified dermatologist. Do not use for clinical diagnosis.
                """)

        gr.Markdown("""
        ---
        <div style="text-align:center; color:#666; font-size:12px;">
        Skin Lesion Detection System | Improved YOLOv8 | Research Prototype | Not for clinical use
        </div>
        """)

    return demo


def main():
    parser = argparse.ArgumentParser(description="Launch Skin Lesion Detection Demo")
    parser.add_argument("--port", type=int, default=7860, help="Gradio server port")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    # Check if any model checkpoints exist
    available = []
    for variant, path in MODEL_PATHS.items():
        if os.path.exists(path):
            available.append(variant)

    if not available:
        print("=" * 60)
        print("⚠️  NO TRAINED MODELS FOUND")
        print("=" * 60)
        print("The demo will start but detection won't work until you train models.")
        print("Run: python train.py --model all")
        print()
        print(f"Looking for checkpoints at:")
        for v, p in MODEL_PATHS.items():
            print(f"  {v}: {p}  {'(NOT FOUND)' if not os.path.exists(p) else '(OK)'}")
        print("=" * 60)

    demo = build_ui()
    demo.queue(max_size=10)
    demo.launch(
        server_port=args.port,
        share=args.share,
        debug=args.debug,
        show_error=True,
    )


if __name__ == "__main__":
    import argparse
    main()
