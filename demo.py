#!/usr/bin/env python3
"""
Skin Disease Classification — Gradio Interactive Demo.

Auto-detects trained models from runs/, supports model switching,
Grad-CAM, batch upload, and clinical risk reporting.

Usage:
    python demo.py                          # auto-detect model
    python demo.py --model runs/E4_v2_gem_ema/best.pt
    python demo.py --port 8080 --share
"""

import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import gradio as gr

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import timm

from models.modules import ConvNeXtWithFeatures
from utils.visualize import draw_classification_result, draw_gradcam


# ============================================================
# Model Discovery
# ============================================================

def find_models(runs_dir: str = "runs") -> list:
    """Find all trained model checkpoints in runs/ directory."""
    runs = Path(runs_dir)
    models = []
    if not runs.exists():
        return models

    for pt_file in sorted(runs.rglob("best.pt"), key=lambda p: p.stat().st_mtime, reverse=True):
        # Try to load summary for display name
        summary_file = pt_file.parent / "experiment_summary.json"
        info = {}
        if summary_file.exists():
            try:
                with open(summary_file) as f:
                    info = json.load(f)
            except Exception:
                pass

        display_name = info.get("variant", pt_file.parent.name)
        val_acc = info.get("val_accuracy") or info.get("best_val_acc")
        acc_str = f"Val Acc: {val_acc:.2%}" if val_acc else ""

        models.append({
            "path": str(pt_file),
            "name": display_name,
            "dir": str(pt_file.parent.name),
            "val_acc": val_acc,
            "acc_str": acc_str,
            "num_params": info.get("num_params", "?"),
            "config": info.get("config", {}),
        })

    return models


# ============================================================
# Model Manager
# ============================================================

class ModelManager:
    """Handles model loading, caching, and switching."""

    def __init__(self):
        self.cache = {}  # path -> (model, class_names, class_risk, img_size, transform, info)
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"

    def load(self, model_path: str):
        """Load a model, using cache if available."""
        if model_path in self.cache:
            return self.cache[model_path]

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        print(f"Loading: {model_path} ...")

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        class_names = checkpoint.get("class_names", [])
        class_risk = checkpoint.get("class_risk", {})

        if not class_names:
            config_path = Path(model_path).parent / "experiment_summary.json"
            if config_path.exists():
                with open(config_path) as f:
                    c = json.load(f)
                class_names = c.get("class_names", [])
                class_risk = c.get("class_risk", {})

        num_classes = len(class_names)

        model_cfg = checkpoint.get("config", {})
        model_name = model_cfg.get("model_name", "convnext_tiny")
        pooling = model_cfg.get("pooling", "avg")
        use_multi_scale = model_cfg.get("multi_scale", False)
        img_size = model_cfg.get("img_size", 224)

        MODEL_MAP = {
            "convnext_tiny": "convnext_tiny.fb_in22k_ft_in1k",
            "convnextv2_tiny": "convnextv2_tiny.fcmae_ft_in22k_in1k",
            "convnext_small": "convnext_small.fb_in22k_ft_in1k",
        }
        timm_name = MODEL_MAP.get(model_name, "convnext_tiny.fb_in22k_ft_in1k")

        backbone = timm.create_model(timm_name, pretrained=False, num_classes=num_classes)
        model = ConvNeXtWithFeatures(
            backbone=backbone, num_classes=num_classes,
            dropout=0.3, use_multi_scale=use_multi_scale, pooling=pooling,
        )

        # Handle EMA state if present
        if "ema_state" in checkpoint:
            from models.modules import ModelEMA
            ema = ModelEMA(model, decay=0.999)
            ema.load_state_dict(checkpoint["ema_state"])
            ema.apply_shadow(model)

        # Remap state_dict for backward compatibility
        state_dict = checkpoint["model_state_dict"]

        # Fix 1: Old code wrapped backbone as `self.backbone = backbone`
        #         → keys have "backbone." prefix. New code uses self.stem/self.stages.
        if any(k.startswith("backbone.") for k in state_dict.keys()):
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}

        # Fix 2: Old head order was norm→pool→flatten→dropout→linear (head.0=norm).
        #         New head order is  pool→flatten→norm→dropout→linear (head.2=norm).
        #         Remap head.0 (norm) → head.2.
        if "head.0.weight" in state_dict and "head.2.weight" not in state_dict:
            remap = {}
            for k in list(state_dict.keys()):
                if k.startswith("head."):
                    if k.startswith("head.0."):
                        remap[k] = k.replace("head.0.", "head.2.")  # norm moved
                    elif k.startswith("head.4."):
                        remap[k] = k  # linear unchanged
            for old_k, new_k in remap.items():
                state_dict[new_k] = state_dict.pop(old_k)

        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(int(img_size * 1.14)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
        ])

        info = {
            "model_name": model_name,
            "pooling": pooling,
            "multi_scale": use_multi_scale,
            "ema": model_cfg.get("ema", False) or "ema_state" in checkpoint,
            "img_size": img_size,
            "num_classes": num_classes,
            "val_acc": checkpoint.get("val_acc"),
            "epoch": checkpoint.get("epoch"),
        }

        self.cache[model_path] = (model, class_names, class_risk, img_size, transform, info)
        print(f"  ✓ Loaded: {model_name} | pooling={pooling} | ms={use_multi_scale} | img={img_size}")
        return self.cache[model_path]

    def clear_cache(self):
        self.cache.clear()
        torch.cuda.empty_cache() if self.device == "cuda" else None


# Global singleton
manager = ModelManager()


# ============================================================
# Inference
# ============================================================

@torch.no_grad()
def classify(image, model_path, show_cam, top_k):
    """Classify a single image."""
    if image is None:
        return None, _empty_report(), None, _model_info_html(model_path)

    try:
        model, class_names, class_risk, img_size, transform, info = manager.load(model_path)
    except Exception as e:
        return image, f"<p style='color:red'>Failed to load model: {e}</p>", None, ""

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = transform(image_rgb).unsqueeze(0).to(manager.device)

    logits = model(tensor)
    probs = F.softmax(logits, dim=1)

    k = min(top_k, len(class_names))
    topk_probs, topk_indices = torch.topk(probs, k)
    topk_probs = topk_probs.cpu().numpy()[0]
    topk_indices = topk_indices.cpu().numpy()[0]

    predictions = []
    for prob, idx in zip(topk_probs, topk_indices):
        name = class_names[idx]
        risk = class_risk.get(name, "LOW")
        predictions.append({"class": name, "confidence": float(prob), "risk": risk})

    # Draw
    annotated = draw_classification_result(image, predictions, class_risk)
    report_html = _build_report(predictions, info)
    info_html = _model_info_html(model_path)

    # Grad-CAM
    cam_image = None
    if show_cam:
        cam_image = draw_gradcam(model, image, class_names, transform, manager.device)

    return annotated, report_html, cam_image, info_html


def classify_batch(files, model_path, show_cam):
    """Classify multiple images, return gallery + summary."""
    if not files:
        return [], "<p style='color:gray'>No images selected.</p>"

    try:
        model, class_names, class_risk, img_size, transform, info = manager.load(model_path)
    except Exception as e:
        return [], f"<p style='color:red'>Failed to load: {e}</p>"

    results = []
    summary_parts = []
    risk_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}

    for file_path in files:
        img = cv2.imread(file_path)
        if img is None:
            continue

        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = transform(image_rgb).unsqueeze(0).to(manager.device)

        logits = model(tensor)
        probs = F.softmax(logits, dim=1)
        top_prob, top_idx = probs.max(dim=1)
        name = class_names[top_idx.item()]
        risk = class_risk.get(name, "LOW")
        risk_counts[risk] += 1

        annotated = draw_classification_result(img, [
            {"class": name, "confidence": float(top_prob), "risk": risk}
        ], class_risk)

        results.append((annotated, f"{Path(file_path).name}\n{name} ({top_prob:.1%})"))
        summary_parts.append(f"{Path(file_path).name}: {name} ({top_prob:.1%}) [{risk}]")

    summary = f"""
    <div style="padding:15px; background:#1a1a2e; border-radius:8px; color:#e0e0e0; font-family:sans-serif;">
        <h4>Batch Summary ({len(results)} images)</h4>
        <p>🔴 HIGH: {risk_counts['HIGH']} | 🟡 MEDIUM: {risk_counts['MEDIUM']} | 🟢 LOW: {risk_counts['LOW']}</p>
        <div style="max-height:300px; overflow-y:auto; font-size:12px;">
            {'<br>'.join(summary_parts)}
        </div>
    </div>
    """

    return results, summary


# ============================================================
# HTML Reports
# ============================================================

def _build_report(predictions: list, info: dict) -> str:
    top = predictions[0]
    risk = top["risk"]
    risk_color = {"HIGH": "#ef5350", "MEDIUM": "#ff9800", "LOW": "#66bb6a"}[risk]

    # Confidence bar chart using CSS
    bar_rows = ""
    colors = ["#4fc3f7", "#81c784", "#ce93d8", "#ffb74d", "#e57373"]
    for i, pred in enumerate(predictions):
        color = colors[min(i, len(colors)-1)]
        bar_rows += f"""
        <div style="margin: 8px 0;">
            <div style="display:flex; justify-content:space-between; font-size:13px; margin-bottom:2px;">
                <span>{'→' if i == 0 else ' '} {pred['class']}</span>
                <span style="color:#aaa;">{pred['confidence']:.1%}</span>
            </div>
            <div style="background:#2a2a3e; border-radius:4px; height:8px;">
                <div style="background:{color}; border-radius:4px; height:8px; width:{pred['confidence']*100:.0f}%; transition:width 0.3s;"></div>
            </div>
        </div>"""

    risk_messages = {
        "HIGH":   ("🚨 高风险警告", "建议立即就医，由皮肤科医生进行专业诊断。", "#3a1a1a", "#ef5350"),
        "MEDIUM": ("⚠️ 中等风险", "建议安排临床随访，进一步评估。", "#2a2a1a", "#ff9800"),
        "LOW":    ("✅ 低风险", "倾向良性表现，建议常规观察。", "#1a2a1a", "#66bb6a"),
    }
    risk_title, risk_msg, bg, border = risk_messages[risk]

    # Model badge
    variant = info.get("model_name", "ConvNeXt").replace("_", " ").title()
    extras = []
    if info.get("pooling") == "gem":
        extras.append("GeM")
    if info.get("multi_scale"):
        extras.append("MS")
    if info.get("ema"):
        extras.append("EMA")
    variant_str = f"{variant}" + (f" + {'+'.join(extras)}" if extras else "")

    return f"""
    <div style="padding:20px; background:#1a1a2e; border-radius:12px; color:#e0e0e0; font-family:system-ui,sans-serif; height:100%;">
        <h3 style="margin:0 0 5px 0; color:#4fc3f7; font-size:18px;">🔬 分类结果</h3>
        <p style="color:#888; font-size:11px; margin:0 0 12px 0;">{variant_str}</p>

        <div style="padding:12px; background:{bg}; border-radius:8px; margin-bottom:15px; border-left:4px solid {border};">
            <div style="font-size:15px; font-weight:bold; color:{border};">{risk_title}: {top['class']}</div>
            <p style="color:#bbb; margin:4px 0 0 0; font-size:12px;">{risk_msg}</p>
        </div>

        <p style="color:#aaa; font-size:12px; margin:0 0 8px 0;">Top-{len(predictions)} Predictions</p>
        {bar_rows}

        <div style="margin-top:14px; padding:8px 10px; background:#111; border-radius:6px; font-size:10px; color:#666; text-align:center;">
            ⚠️ 本系统为研究原型，不可用于临床诊断
        </div>
    </div>
    """


def _empty_report() -> str:
    return """
    <div style="padding:20px; background:#1a1a2e; border-radius:12px; color:#e0e0e0; font-family:system-ui,sans-serif; height:100%;">
        <h3 style="margin:0; color:#4fc3f7;">🔬 分类结果</h3>
        <p style="color:#888; margin-top:20px; text-align:center;">请上传一张皮肤图像以开始分析</p>
    </div>
    """


def _model_info_html(model_path: str) -> str:
    """Render model info sidebar."""
    if not model_path or not os.path.exists(model_path):
        return "<div style='color:#666;font-size:12px;'>No model loaded</div>"

    cached = manager.cache.get(model_path)
    if cached is None:
        return "<div style='color:#666;font-size:12px;'>Loading...</div>"

    _, _, _, _, _, info = cached

    variant = info.get("model_name", "?").replace("_", " ").title()
    pooling = info.get("pooling", "?")
    ms = "✓" if info.get("multi_scale") else "✗"
    ema = "✓" if info.get("ema") else "✗"
    img_size = info.get("img_size", "?")
    val_acc = info.get("val_acc")
    acc_str = f"{val_acc:.2%}" if val_acc else "?"

    return f"""
    <div style="padding:12px; background:#1a1a2e; border-radius:8px; color:#ccc; font-size:12px; font-family:system-ui,sans-serif;">
        <b style="color:#4fc3f7;">Model Info</b>
        <table style="width:100%; margin-top:6px; font-size:11px;">
            <tr><td style="color:#888;">Architecture</td><td>{variant}</td></tr>
            <tr><td style="color:#888;">Pooling</td><td>{pooling.upper()}</td></tr>
            <tr><td style="color:#888;">Multi-Scale</td><td>{ms}</td></tr>
            <tr><td style="color:#888;">EMA</td><td>{ema}</td></tr>
            <tr><td style="color:#888;">Image Size</td><td>{img_size}px</td></tr>
            <tr><td style="color:#888;">Val Accuracy</td><td style="color:#81c784;">{acc_str}</td></tr>
        </table>
    </div>
    """


# ============================================================
# Gradio UI
# ============================================================

def build_ui(models: list):
    """Build the full Gradio interface."""
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
    )

    # Model choices
    if models:
        model_choices = [m["path"] for m in models]
        default_model = model_choices[0]
        model_labels = [f"{m['name']} ({m['acc_str']})" if m['acc_str'] else m['name'] for m in models]
    else:
        model_choices = ["runs/convnext_v2_tiny_GeM_EMA/best.pt"]
        model_labels = ["No trained models found"]
        default_model = model_choices[0]

    with gr.Blocks(
        theme=theme,
        title="Skin Disease Classification — ConvNeXt",
        css="""
        footer { visibility: hidden; }
        .gradio-container { max-width: 1200px !important; }
        """,
    ) as demo:
        # Header
        gr.Markdown("""
        # 🩺 皮肤疾病智能辅助分类系统
        ### 基于改进 ConvNeXt 的 22 类皮肤疾病诊断 | AI Research Prototype
        """)

        with gr.Row():
            # ── Left sidebar ──
            with gr.Column(scale=1, min_width=200):
                gr.Markdown("### ⚙️ 设置")

                model_dropdown = gr.Dropdown(
                    choices=model_choices,
                    value=default_model,
                    label="模型选择",
                    info="从 runs/ 自动检测已训练模型",
                    interactive=True,
                )

                top_k_slider = gr.Slider(
                    minimum=1, maximum=5, value=3, step=1,
                    label="显示 Top-K",
                    info="显示前 K 个最可能的类别",
                )

                cam_checkbox = gr.Checkbox(
                    label="Grad-CAM 热力图",
                    value=False,
                    info="显示模型关注的图像区域",
                )

                model_info = gr.HTML(value=_model_info_html(default_model))

                gr.Markdown("""
                ---
                ### 📋 风险等级
                | 图标 | 等级 | 建议 |
                |------|------|------|
                | 🔴 | HIGH | 立即就医 |
                | 🟡 | MEDIUM | 临床随访 |
                | 🟢 | LOW | 常规观察 |
                """)

            # ── Main area ──
            with gr.Column(scale=2):
                with gr.Tabs():
                    # Tab 1: Single Image
                    with gr.TabItem("📸 单图分析"):
                        with gr.Row():
                            with gr.Column(scale=3):
                                input_image = gr.Image(
                                    label="上传皮肤图像",
                                    type="numpy",
                                    height=420,
                                    sources=["upload", "clipboard", "webcam"],
                                )
                                with gr.Row():
                                    classify_btn = gr.Button(
                                        "🔬 开始分析", variant="primary", size="lg"
                                    )
                                    clear_btn = gr.Button("🗑️ 清除", size="lg")

                            with gr.Column(scale=2):
                                output_image = gr.Image(
                                    label="分析结果", type="numpy", height=300
                                )
                                report_html = gr.HTML(
                                    value=_empty_report(),
                                    elem_id="report",
                                )

                        # Grad-CAM row (shown when checkbox is on)
                        with gr.Row():
                            cam_output = gr.Image(
                                label="Grad-CAM 激活热力图",
                                type="numpy",
                                height=250,
                                visible=True,
                            )

                    # Tab 2: Batch
                    with gr.TabItem("📚 批量分析"):
                        gr.Markdown("批量上传多张图像，快速筛查。")
                        batch_files = gr.File(
                            label="选择图像文件",
                            file_count="multiple",
                            file_types=["image"],
                        )
                        batch_btn = gr.Button("🔬 批量分析", variant="primary")
                        batch_gallery = gr.Gallery(
                            label="分类结果",
                            columns=4,
                            height=400,
                        )
                        batch_summary = gr.HTML()

                    # Tab 3: About
                    with gr.TabItem("ℹ️ 关于"):
                        gr.Markdown("""
                        ## 关于本系统

        这是基于 ConvNeXt 架构的皮肤疾病分类研究原型。

        ### 模型架构
        | 组件 | 说明 |
        |------|------|
        | 骨干网络 | ConvNeXtV2-Tiny (FCMAE + ImageNet-22k pretrained) |
        | 池化 | GeM Pooling (Generalized Mean, p=3.0) |
        | 特征融合 | Multi-Scale (Stage 2/3/4) |
        | 正则化 | EMA (decay=0.999) + Label Smoothing (0.1) |
        | 预训练 | FCMAE → ImageNet-22k → Skin Disease Fine-tune |

        ### 数据集
        - **22 类**皮肤疾病
        - **15,444 张**图像（训练 13,898 + 测试 1,546）
        - 来源: Skin Disease Dataset

        ### 技术栈
        - PyTorch + timm (ConvNeXt)
        - Grad-CAM 可解释性
        - Gradio Web 界面

        ### ⚠️ 免责声明
        **本系统仅为学术研究原型，未获医疗器械认证。**
        所有预测结果仅供研究参考，不可用于临床诊断。
        任何医疗决策应由合格的皮肤科医生做出。
        """)

        # Footer
        gr.Markdown("""
        ---
        <div style="text-align:center; color:#555; font-size:11px;">
        Skin Disease Classification System | ConvNeXtV2 + GeM + EMA | Research Prototype | &copy; 2025
        </div>
        """)

        # ── Events ──
        classify_btn.click(
            fn=classify,
            inputs=[input_image, model_dropdown, cam_checkbox, top_k_slider],
            outputs=[output_image, report_html, cam_output, model_info],
        )

        clear_btn.click(
            fn=lambda: (None, _empty_report(), None, _model_info_html(default_model)),
            outputs=[input_image, report_html, cam_output, model_info],
        )

        batch_btn.click(
            fn=classify_batch,
            inputs=[batch_files, model_dropdown, cam_checkbox],
            outputs=[batch_gallery, batch_summary],
        )

        # Update model info when model changes
        model_dropdown.change(
            fn=lambda p: _model_info_html(p),
            inputs=[model_dropdown],
            outputs=[model_info],
        )

    return demo


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Skin Disease Classification Demo")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to specific model checkpoint (auto-detect if not set)")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # Discover models
    models = find_models("runs")
    if args.model:
        # User specified a model — add it to the list if not found
        if not any(m["path"] == args.model for m in models):
            models.insert(0, {"path": args.model, "name": Path(args.model).parent.name, "acc_str": ""})

    if not models:
        print("=" * 60)
        print("⚠️  NO TRAINED MODELS FOUND in runs/")
        print("   Train a model first:")
        print("   python train.py --model convnextv2_tiny --pooling gem --ema")
        print("=" * 60)
    else:
        print(f"Found {len(models)} trained model(s):")
        for m in models[:5]:
            print(f"  • {m['name']}  {m['acc_str']}")

    demo = build_ui(models)
    demo.queue(max_size=20)
    demo.launch(
        server_port=args.port,
        share=args.share,
        debug=args.debug,
        show_error=True,
    )


if __name__ == "__main__":
    main()
