#!/usr/bin/env python3
"""
Skin Disease Classification — Inference Script.

Usage:
    python detect.py --image test.jpg
    python detect.py --dir data/Test/
    python detect.py --image test.jpg --cam  # Show Grad-CAM heatmap
"""

import argparse
import sys
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import cv2

import torchvision.transforms as transforms
import timm

from utils.visualize import draw_classification_result, draw_gradcam


# ============================================================
# Model Loading
# ============================================================

def load_model(checkpoint_path: str, device: str = "mps"):
    """Load trained ConvNeXt model with class config."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    class_names = checkpoint.get("class_names")
    class_risk = checkpoint.get("class_risk", {})

    if class_names is None:
        # Try loading from companion config file
        config_path = Path(checkpoint_path).parent / "class_config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            class_names = config["class_names"]
            class_risk = config.get("class_risk", {})
        else:
            raise ValueError("Class names not found in checkpoint or config file")

    num_classes = len(class_names)

    model = timm.create_model(
        "convnext_tiny.fb_in22k_ft_in1k",
        pretrained=False,
        num_classes=num_classes,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Loaded model (epoch {checkpoint.get('epoch', '?')}, val_acc={checkpoint.get('val_acc', 0):.4f})")
    print(f"Classes: {num_classes}")

    return model, class_names, class_risk


# ============================================================
# Transforms
# ============================================================

def get_transform(img_size: int = 224):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ============================================================
# Inference
# ============================================================

@torch.no_grad()
def predict(model, image_bgr: np.ndarray, class_names: list, class_risk: dict,
            transform, device: str = "mps", top_k: int = 3, use_cam: bool = False):
    """
    Run classification on a single image.

    Returns: dict with top_k predictions and risk info.
    """
    h, w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Preprocess
    tensor = transform(image_rgb).unsqueeze(0).to(device)

    # Inference
    logits = model(tensor)
    probs = F.softmax(logits, dim=1)

    topk_probs, topk_indices = torch.topk(probs, top_k)
    topk_probs = topk_probs.cpu().numpy()[0]
    topk_indices = topk_indices.cpu().numpy()[0]

    predictions = []
    for prob, idx in zip(topk_probs, topk_indices):
        name = class_names[idx]
        risk = class_risk.get(name, "LOW")
        predictions.append({
            "class": name,
            "confidence": float(prob),
            "risk": risk,
        })

    result = {
        "predictions": predictions,
        "top_class": predictions[0]["class"],
        "top_confidence": predictions[0]["confidence"],
        "top_risk": predictions[0]["risk"],
    }

    # Grad-CAM
    cam_image = None
    if use_cam:
        cam_image = draw_gradcam(model, image_bgr, class_names, transform, device)

    return result, cam_image


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Skin Disease Classification Inference"
    )
    parser.add_argument("--model", type=str, default="runs/convnext/best.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to single image")
    parser.add_argument("--dir", type=str, default=None,
                        help="Path to image directory")
    parser.add_argument("--output", type=str, default="results",
                        help="Output directory for results")
    parser.add_argument("--cam", action="store_true",
                        help="Generate Grad-CAM heatmap")
    parser.add_argument("--device", type=str, default="mps",
                        help="Device (mps/cpu/cuda)")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Show top-K predictions")

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model, class_names, class_risk = load_model(args.model, args.device)
    transform = get_transform()

    # Collect images
    images_to_process = []
    if args.image:
        images_to_process.append(Path(args.image))
    elif args.dir:
        img_dir = Path(args.dir)
        images_to_process.extend(sorted(
            p for p in img_dir.rglob("*")
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
        ))
    else:
        print("[ERROR] Specify --image or --dir")
        sys.exit(1)

    print(f"\nProcessing {len(images_to_process)} image(s)...\n")

    for img_path in images_to_process:
        print(f"{'─' * 50}")
        print(f"Image: {img_path.name}")

        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  [SKIP] Cannot read image")
            continue

        result, cam_image = predict(
            model, image, class_names, class_risk, transform,
            device=args.device, top_k=args.top_k, use_cam=args.cam,
        )

        # Print results
        for i, pred in enumerate(result["predictions"]):
            risk_icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(pred["risk"], "⚪")
            marker = " →" if i == 0 else "  "
            print(f"{marker} {risk_icon} {pred['class']:<30s} {pred['confidence']:.2%}  [{pred['risk']}]")

        # Alert
        if result["top_risk"] == "HIGH":
            print(f"  ⚠️  HIGH RISK: {result['top_class']} — clinical consultation advised")

        # Save result
        annotated = draw_classification_result(image, result["predictions"], class_risk)
        out_path = output_dir / f"result_{img_path.stem}.jpg"
        cv2.imwrite(str(out_path), annotated)

        if cam_image is not None:
            cam_path = output_dir / f"cam_{img_path.stem}.jpg"
            cv2.imwrite(str(cam_path), cam_image)

    print(f"\nResults saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
