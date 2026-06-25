#!/usr/bin/env python3
"""
Flask web app for skin disease classification.
Replaces Gradio with vanilla HTML/CSS/JS + SSE streaming.

Usage:
    python web_app/app.py
    python web_app/app.py --port 8080
"""

import sys
import os
import json
import argparse
import io
import base64
from pathlib import Path

# Add parent dir for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import timm
from flask import Flask, request, jsonify, render_template, Response
from PIL import Image, ImageDraw, ImageFont

from models.modules import ConvNeXtWithFeatures

# ============================================================
# Config
# ============================================================

CLASS_NAMES_ZH = {
    "Acne": "痤疮", "Actinic Keratosis": "光化性角化病",
    "Benign Tumors": "良性肿瘤", "Bullous": "大疱性皮肤病",
    "Candidiasis": "念珠菌病", "Drug Eruption": "药疹",
    "Eczema": "湿疹", "Infestations/Bites": "寄生虫/虫咬",
    "Lichen": "苔藓", "Lupus": "红斑狼疮", "Moles": "痣",
    "Psoriasis": "银屑病", "Rosacea": "玫瑰痤疮",
    "Seborrheic Keratoses": "脂溢性角化病", "Skin Cancer": "皮肤癌",
    "Sun/Sunlight Damage": "日光性损伤", "Tinea": "癣",
    "Unknown/Normal": "正常/未知", "Vascular Tumors": "血管性肿瘤",
    "Vasculitis": "血管炎", "Vitiligo": "白癜风", "Warts": "疣",
}

DISEASE_KB = {
    "Skin Cancer": {"overview":"皮肤癌包括基底细胞癌、鳞状细胞癌和黑色素瘤等，是全球最常见的恶性肿瘤之一。早期发现治愈率很高。","symptoms":"不愈合的溃疡、进行性增大的结节、色素性皮损ABCDE改变。","treatment":"需尽快皮肤科就诊。手术完整切除（Mohs手术为首选）。根据病理类型决定后续治疗（放疗、靶向治疗、免疫治疗）。","precautions":"立即预约皮肤科医生。严格防晒。终身定期皮肤检查（每3-6个月）。"},
    "Acne": {"overview":"痤疮是一种常见的毛囊皮脂腺慢性炎症性皮肤病，好发于面部、胸背部。","symptoms":"粉刺（黑头/白头）、炎性丘疹、脓疱、结节、囊肿。","treatment":"轻度：外用维A酸/过氧化苯甲酰。中度：外用+口服抗生素。重度：口服异维A酸、光动力疗法。","precautions":"避免挤压皮损。使用非致痘性护肤品。减少高糖/高脂饮食。规律作息。"},
}

# Default KB entry for other classes
_DEFAULT_KB = {"overview":"请参考AI助手提供的详细信息。","symptoms":"因个体差异而异，需结合临床表现综合判断。","treatment":"建议咨询皮肤科医生获取个性化治疗方案。","precautions":"定期观察皮损变化。如有异常及时就医。"}

app = Flask(__name__)

# ============================================================
# Model loading
# ============================================================

MODEL = None
CLASS_NAMES = []
CLASS_RISK = {}
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
TRANSFORM = None
IMG_SIZE = 224
MODEL_INFO = {}

# LLM client
LLM_AVAILABLE = False
LLM_CONFIG = {}


def _load_config():
    """Load config.json or config.example.json."""
    for name in ["config.json", "config.example.json"]:
        p = Path(__file__).parent.parent / name
        if p.exists():
            try:
                cfg = json.loads(p.read_text())
                api_key = cfg.get("llm", {}).get("api_key", "")
                if "your-api-key" in api_key or "sk-your" in api_key:
                    continue
                return cfg
            except Exception:
                continue
    return {}


def load_model(model_path: str = None):
    """Load the best available model."""
    global MODEL, CLASS_NAMES, CLASS_RISK, TRANSFORM, IMG_SIZE, MODEL_INFO

    if MODEL is not None:
        return

    # Find best checkpoint
    if model_path is None:
        runs = sorted(Path(__file__).parent.parent.rglob("runs/*/best.pt"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
        if not runs:
            print("[ERROR] No trained model found.")
            return
        model_path = str(runs[0])

    print(f"Loading model: {model_path}")
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    CLASS_NAMES = ckpt.get("class_names", [])
    class_names_zh = ckpt.get("class_names_zh") or CLASS_NAMES_ZH
    CLASS_RISK = ckpt.get("class_risk", {})

    if not CLASS_NAMES:
        config_path = Path(model_path).parent / "experiment_summary.json"
        if config_path.exists():
            c = json.loads(config_path.read_text())
            CLASS_NAMES = c.get("class_names", [])
            class_names_zh = c.get("class_names_zh") or CLASS_NAMES_ZH
            CLASS_RISK = c.get("class_risk", {})

    num_classes = len(CLASS_NAMES)
    cfg = ckpt.get("config", {})
    model_name = cfg.get("model_name", "convnext_tiny")
    pooling = cfg.get("pooling", "avg")
    use_multi_scale = cfg.get("multi_scale", False)
    IMG_SIZE = cfg.get("img_size", 224)

    MODEL_MAP = {
        "convnext_tiny": "convnext_tiny.fb_in22k_ft_in1k",
        "convnextv2_tiny": "convnextv2_tiny.fcmae_ft_in22k_in1k",
        "convnext_small": "convnext_small.fb_in22k_ft_in1k",
    }
    timm_name = MODEL_MAP.get(model_name, "convnext_tiny.fb_in22k_ft_in1k")

    backbone = timm.create_model(timm_name, pretrained=False, num_classes=num_classes)
    MODEL = ConvNeXtWithFeatures(
        backbone=backbone, num_classes=num_classes,
        dropout=0.3, use_multi_scale=use_multi_scale, pooling=pooling,
    )

    # EMA
    if "ema_state" in ckpt:
        from models.modules import ModelEMA
        ema = ModelEMA(MODEL, decay=0.999)
        ema.load_state_dict(ckpt["ema_state"])
        ema.apply_shadow(MODEL)

    # State dict remapping
    sd = ckpt["model_state_dict"]
    if any(k.startswith("backbone.") for k in sd):
        sd = {k.replace("backbone.", ""): v for k, v in sd.items()}
    if "head.0.weight" in sd and "head.2.weight" not in sd:
        remap = {}
        for k in list(sd.keys()):
            if k.startswith("head.0."): remap[k] = k.replace("head.0.", "head.2.")
            elif k.startswith("head.4."): remap[k] = k
        for ok, nk in remap.items(): sd[nk] = sd.pop(ok)
    MODEL.load_state_dict(sd)
    MODEL = MODEL.to(DEVICE)
    MODEL.eval()

    TRANSFORM = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(int(IMG_SIZE * 1.14)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    MODEL_INFO = {
        "model_name": model_name, "pooling": pooling,
        "multi_scale": use_multi_scale,
        "ema": cfg.get("ema", False) or "ema_state" in ckpt,
        "val_acc": ckpt.get("val_acc"),
        "num_classes": num_classes,
    }
    # Store zh mapping for templates
    app.config["CLASS_NAMES_ZH"] = class_names_zh
    print(f"  ✓ Model loaded: {model_name} | Val Acc: {MODEL_INFO.get('val_acc', 0):.2%}")


# ============================================================
# LLM helpers
# ============================================================

def _init_llm():
    global LLM_AVAILABLE, LLM_CONFIG
    cfg = _load_config().get("llm", {})
    api_key = cfg.get("api_key")
    if api_key:
        LLM_CONFIG = cfg
        LLM_AVAILABLE = True
        print(f"🤖 LLM: {cfg.get('api_model', '?')} @ {cfg.get('api_base', '?')}")


def _call_llm(messages: list, stream: bool = True):
    """Call LLM API. If stream=True, returns a generator yielding tokens."""
    if not LLM_AVAILABLE:
        if stream:
            yield "[AI服务未配置]"
        else:
            return "[AI服务未配置]"
        return

    import urllib.request
    data = json.dumps({
        "model": LLM_CONFIG["api_model"],
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 600,
        "stream": stream,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{LLM_CONFIG['api_base']}/chat/completions",
        data=data,
        headers={"Authorization": f"Bearer {LLM_CONFIG['api_key']}", "Content-Type": "application/json"},
    )

    try:
        resp = urllib.request.urlopen(req, timeout=60)
        if stream:
            for line in resp:
                line = line.decode("utf-8").strip()
                if line.startswith("data: ") and line != "data: [DONE]":
                    try:
                        chunk = json.loads(line[6:])
                        choice = chunk.get("choices", [{}])[0]
                        if choice.get("finish_reason") is not None:
                            continue
                        content = choice.get("delta", {}).get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue
        else:
            result = json.loads(resp.read())
            return result["choices"][0]["message"]["content"]
    except Exception as e:
        msg = f"[AI调用失败: {e}]"
        if stream:
            yield msg
        else:
            return msg


# ============================================================
# Routes — Page
# ============================================================

@app.route("/")
def index():
    return render_template("index.html",
                           llm_available=LLM_AVAILABLE,
                           model_info=MODEL_INFO)


# ============================================================
# Routes — Classify
# ============================================================

@app.route("/api/classify", methods=["POST"])
def api_classify():
    """Classify uploaded image, return predictions + KB."""
    if MODEL is None:
        return jsonify({"error": "模型未加载"}), 500

    file = request.files.get("image")
    if not file:
        return jsonify({"error": "未上传图片"}), 400

    img_bytes = file.read()
    img_np = np.frombuffer(img_bytes, np.uint8)
    img_bgr = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return jsonify({"error": "无法解析图片"}), 400

    # Classify
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    with torch.no_grad():
        tensor = TRANSFORM(img_rgb).unsqueeze(0).to(DEVICE)
        logits = MODEL(tensor)
        probs = F.softmax(logits, dim=1)

    topk = torch.topk(probs, min(5, len(CLASS_NAMES)))
    topk_probs = topk.values.cpu().numpy()[0]
    topk_indices = topk.indices.cpu().numpy()[0]

    predictions = []
    for prob, idx in zip(topk_probs, topk_indices):
        name_en = CLASS_NAMES[idx]
        name_zh = app.config["CLASS_NAMES_ZH"].get(name_en, name_en)
        risk = CLASS_RISK.get(name_en, "LOW")
        kb = DISEASE_KB.get(name_en, _DEFAULT_KB)
        predictions.append({
            "class_en": name_en,
            "class_zh": name_zh,
            "confidence": float(prob),
            "risk": risk,
            "kb": kb,
        })

    return jsonify({"predictions": predictions})


# ============================================================
# Routes — LLM report (SSE streaming)
# ============================================================

@app.route("/api/report", methods=["POST"])
def api_report():
    """Stream LLM report via Server-Sent Events."""
    data = request.get_json()
    top = data.get("top", {})
    disease_en = top.get("class_en", "")
    disease_zh = top.get("class_zh", "")
    confidence = top.get("confidence", 0)
    risk = top.get("risk", "LOW")
    kb = top.get("kb", {})

    prompt = f"""你是皮肤科AI助手。分类结果: {disease_en}（{disease_zh}），置信度{confidence:.1%}，风险{risk}。
已知信息: {kb.get('overview','')} 症状: {kb.get('symptoms','')} 治疗: {kb.get('treatment','')}
请用中文输出简洁医学报告（每段不超过3行）:
【疾病概述】→ 【可能症状】→ 【建议措施】→ 【就医指引】→ ⚠️ AI生成仅供参考"""

    messages = [
        {"role": "system", "content": "你是皮肤科AI助手。"},
        {"role": "user", "content": prompt},
    ]

    def generate():
        for token in _call_llm(messages, stream=True):
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield "data: [DONE]\n\n"

    return Response(generate(), mimetype="text/event-stream")


# ============================================================
# Routes — Chat (SSE streaming)
# ============================================================

@app.route("/api/chat", methods=["POST"])
def api_chat():
    """Stream chat response via SSE."""
    data = request.get_json()
    message = data.get("message", "")
    history = data.get("history", [])  # [{role, content}, ...]
    ctx = data.get("context", {})  # classification context

    top_en = ctx.get("class_en", "")
    top_zh = ctx.get("class_zh", "")
    top_conf = ctx.get("confidence", 0)
    top_risk = ctx.get("risk", "LOW")
    kb = ctx.get("kb", {})

    system_msg = f"""你是皮肤科AI助手。分类结果: {top_en}（{top_zh}），置信度{top_conf:.1%}，风险{top_risk}。
已知: {kb.get('overview','')} 症状: {kb.get('symptoms','')} 治疗: {kb.get('treatment','')}
基于以上回答用户问题。用中文，简洁专业。"""

    messages = [{"role": "system", "content": system_msg}]
    for h in history[-6:]:
        role = h.get("role", "")
        content = h.get("content", "")
        if role in ("user", "assistant"):
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": message})

    def generate():
        for token in _call_llm(messages, stream=True):
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield "data: [DONE]\n\n"

    return Response(generate(), mimetype="text/event-stream")


# ============================================================
# Routes — Grad-CAM
# ============================================================

@app.route("/api/gradcam", methods=["POST"])
def api_gradcam():
    """Generate Grad-CAM heatmap for the uploaded image."""
    if MODEL is None:
        return jsonify({"error": "模型未加载"}), 500

    file = request.files.get("image")
    if not file:
        return jsonify({"error": "未上传图片"}), 400

    img_bytes = file.read()
    img_np = np.frombuffer(img_bytes, np.uint8)
    img_bgr = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return jsonify({"error": "无法解析图片"}), 400

    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = TRANSFORM(img_rgb).unsqueeze(0).to(DEVICE)

    # Find last conv layer in stages.3
    target_layer = None
    for name, module in MODEL.named_modules():
        if "stages.3" in name and isinstance(module, torch.nn.Conv2d):
            target_layer = module

    if target_layer is None:
        return jsonify({"error": "无法定位目标层"}), 500

    activations = {}
    gradients = {}

    def forward_hook(m, inp, out):
        activations["v"] = out

    def backward_hook(m, g_in, g_out):
        gradients["v"] = g_out[0]

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    MODEL.zero_grad()
    logits = MODEL(tensor)
    pred_idx = logits.argmax(dim=1).item()
    logits[0, pred_idx].backward()

    fh.remove()
    bh.remove()

    act = activations["v"].detach()
    grad = gradients["v"].detach()

    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = (weights * act).sum(dim=1).squeeze(0)
    cam = F.relu(cam)
    if cam.max() > 0:
        cam = cam / cam.max()

    cam = cam.cpu().numpy()
    cam = cv2.resize(cam, (w, h))
    cam = np.uint8(255 * cam)

    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr, 0.4, heatmap, 0.6, 0)

    # Add label with PIL (supports Chinese)
    class_en = CLASS_NAMES[pred_idx]
    class_zh = app.config["CLASS_NAMES_ZH"].get(class_en, class_en)
    label = f"Grad-CAM: {class_zh}"
    # Render with PIL
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(overlay_rgb)
    draw = ImageDraw.Draw(pil_img)
    # Find CJK font
    font = None
    for fp in ["/System/Library/Fonts/PingFang.ttc", "/System/Library/Fonts/STHeiti Light.ttc",
               "C:/Windows/Fonts/msyh.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
        if Path(fp).exists():
            font = ImageFont.truetype(fp, 18)
            break
    if font is None:
        font = ImageFont.load_default()
    draw.text((10, 6), label, font=font, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
    overlay = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    _, buf = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 92])
    img_b64 = base64.b64encode(buf).decode("utf-8")
    return jsonify({"heatmap": f"data:image/jpeg;base64,{img_b64}", "class": class_zh, "class_en": class_en})


# ============================================================
# Routes — Models list
# ============================================================

@app.route("/api/models")
def api_models():
    models = []
    for ckpt in sorted(Path(__file__).parent.parent.rglob("runs/*/best.pt"),
                       key=lambda p: p.stat().st_mtime, reverse=True):
        summary = ckpt.parent / "experiment_summary.json"
        info = {}
        if summary.exists():
            try:
                info = json.loads(summary.read_text())
            except Exception:
                pass
        models.append({
            "name": info.get("variant", ckpt.parent.name),
            "val_acc": info.get("best_val_acc") or info.get("val_accuracy"),
        })
    return jsonify(models)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    _init_llm()
    load_model(args.model)

    app.run(host="0.0.0.0", port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
