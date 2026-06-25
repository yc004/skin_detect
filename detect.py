#!/usr/bin/env python3
"""
Skin Lesion Detection — Inference & Comparison Script

Usage:
    python detect.py --model runs/baseline/weights/best.pt --image test_image.jpg
    python detect.py --compare baseline,boundary,improved --image test_image.jpg
    python detect.py --model runs/improved/weights/best.pt --dir data/images/test/
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List, Dict

import cv2
import numpy as np
import torch

# Register custom modules
import ultralytics.nn.modules as ul_nn_modules
import ultralytics.nn.tasks as ul_tasks
from models.attention import CoordAtt
from models.asff import ASFFHead

ul_nn_modules.CoordAtt = CoordAtt
ul_nn_modules.ASFFHead = ASFFHead
ul_tasks.CoordAtt = CoordAtt
ul_tasks.ASFFHead = ASFFHead

from ultralytics import YOLO

from utils.visualize import draw_detections, draw_legend, create_comparison, generate_risk_report
from utils.soft_nms import soft_nms


def load_model(weights_path: str, device: str = "mps") -> YOLO:
    """Load a trained YOLO model checkpoint."""
    model = YOLO(weights_path)
    if device != "cpu":
        try:
            model.to(device)
        except Exception:
            print(f"[WARN] Could not move model to {device}, using CPU")
    return model


def run_detection(
    model: YOLO,
    image: np.ndarray,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    use_soft_nms: bool = False,
    soft_nms_sigma: float = 0.5,
    img_size: int = 640,
) -> tuple:
    """
    Run detection on a single image.

    Returns:
        (boxes, classes, confidences) — all as numpy arrays in xyxy pixel format
    """
    h, w = image.shape[:2]

    # Run inference
    results = model.predict(
        image,
        imgsz=img_size,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False,
    )

    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        return np.array([]), np.array([]), np.array([])

    boxes = result.boxes.xyxy.cpu().numpy()  # xyxy pixel format
    classes = result.boxes.cls.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()

    # Apply Soft-NMS if requested
    if use_soft_nms and len(boxes) > 1:
        boxes_tensor = torch.from_numpy(boxes).float()
        scores_tensor = torch.from_numpy(confidences).float()

        keep_indices, updated_scores = soft_nms(
            boxes_tensor, scores_tensor,
            iou_threshold=iou_threshold,
            sigma=soft_nms_sigma,
            method="gaussian",
        )

        boxes = boxes[keep_indices.numpy()]
        classes = classes[keep_indices.numpy()]
        confidences = updated_scores[keep_indices.numpy()].numpy()

    return boxes, classes, confidences


def detect_image(
    model: YOLO,
    image_path: str,
    output_path: str = None,
    conf_threshold: float = 0.25,
    use_soft_nms: bool = False,
):
    """Run detection on an image file and save/save the result."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Cannot read image: {image_path}")
        return None, None

    boxes, classes, confidences = run_detection(
        model, image, conf_threshold=conf_threshold, use_soft_nms=use_soft_nms,
    )

    if len(boxes) == 0:
        print(f"No lesions detected in: {image_path}")
        annotated = image.copy()
    else:
        annotated = draw_detections(image, boxes, classes, confidences, normalized=False)
        annotated = draw_legend(annotated)

    if output_path:
        cv2.imwrite(output_path, annotated)
        print(f"Result saved to: {output_path}")

    # Generate report
    det_list = [(c, conf, b) for c, conf, b in zip(classes, confidences, boxes)]
    report = generate_risk_report(det_list)
    if report["alert"]:
        print(f"⚠️  {report['alert_message']}")

    return annotated, report


def detect_comparison(
    model_paths: Dict[str, str],
    image_path: str,
    output_path: str = None,
    conf_threshold: float = 0.25,
    device: str = "mps",
):
    """
    Run detection with multiple models and create a side-by-side comparison.

    Args:
        model_paths: Dict of {label: checkpoint_path} for each model variant
        image_path:  Path to input image
        output_path: Where to save comparison image
        conf_threshold: Confidence threshold
        device: Device for inference
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Cannot read image: {image_path}")
        return None

    images = []
    titles = []
    reports = {}

    for label, path in model_paths.items():
        if not os.path.exists(path):
            print(f"[SKIP] Model not found: {path}")
            continue

        print(f"Running detection with: {label}")
        model = load_model(path, device)
        boxes, classes, confidences = run_detection(
            model, image, conf_threshold=conf_threshold,
        )

        annotated = draw_detections(image.copy(), boxes, classes, confidences, normalized=False)
        annotated = draw_legend(annotated)

        # Add model label
        cv2.putText(annotated, label.upper(), (10, annotated.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        images.append(annotated)
        titles.append(label)
        reports[label] = generate_risk_report(
            [(c, conf, b) for c, conf, b in zip(classes, confidences, boxes)]
        )

    if len(images) == 0:
        print("[ERROR] No models could be loaded.")
        return None

    # Create comparison
    comparison = create_comparison(images, titles, output_path)

    # Print comparison
    print("\n" + "=" * 60)
    print("DETECTION COMPARISON")
    print("=" * 60)
    for label, report in reports.items():
        print(f"\n{label}:")
        print(f"  Lesions detected: {report['total_lesions']}")
        print(f"  High risk: {report['high_risk_count']}")
        print(f"  Medium risk: {report['medium_risk_count']}")
        for lesion in report["lesions"]:
            print(f"    - {lesion['class']} ({lesion['risk']}): {lesion['confidence']:.3f}")

    return comparison


def main():
    parser = argparse.ArgumentParser(
        description="Skin lesion detection with YOLOv8"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to trained model weights (.pt)"
    )
    parser.add_argument(
        "--compare", type=str, default=None,
        help="Compare models: 'baseline,boundary,improved'"
    )
    parser.add_argument(
        "--image", type=str, default=None,
        help="Path to single input image"
    )
    parser.add_argument(
        "--dir", type=str, default=None,
        help="Path to directory of images"
    )
    parser.add_argument(
        "--output", type=str, default="results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Confidence threshold"
    )
    parser.add_argument(
        "--soft-nms", action="store_true",
        help="Use Soft-NMS post-processing"
    )
    parser.add_argument(
        "--device", type=str, default="mps",
        help="Device (mps/cpu/cuda)"
    )

    args = parser.parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve model paths for comparison mode
    model_paths = {}
    if args.compare:
        variant_names = args.compare.split(",")
        for v in variant_names:
            v = v.strip()
            ckpt = Path(f"runs/{v}/weights/best.pt")
            if ckpt.exists():
                model_paths[v] = str(ckpt)
            else:
                print(f"[WARN] Checkpoint for '{v}' not found: {ckpt}")
    elif args.model:
        model_paths["model"] = args.model
    else:
        # Default: try all variants
        base = Path("runs")
        for v in ["baseline", "boundary", "improved"]:
            ckpt = base / v / "weights" / "best.pt"
            if ckpt.exists():
                model_paths[v] = str(ckpt)
        if not model_paths:
            print("[ERROR] No model checkpoints found. Train models first.")
            sys.exit(1)

    # Determine images to process
    images_to_process = []
    if args.image:
        images_to_process.append(args.image)
    elif args.dir:
        img_dir = Path(args.dir)
        images_to_process.extend(sorted(
            str(p) for p in img_dir.glob("*")
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
        ))
    else:
        # Default: sample from test set
        test_dir = Path("data/images/test")
        if test_dir.exists():
            images_to_process = sorted(str(p) for p in test_dir.glob("*.jpg"))[:10]
        else:
            print("[ERROR] No images specified and no default test dir found.")
            sys.exit(1)

    print(f"Processing {len(images_to_process)} image(s) with {len(model_paths)} model(s)...")

    # Run detection
    if len(model_paths) > 1 and len(images_to_process) <= 10:
        # Comparison mode
        for img_path in images_to_process:
            img_name = Path(img_path).stem
            out_path = str(output_dir / f"compare_{img_name}.jpg")
            detect_comparison(
                model_paths, img_path, out_path,
                conf_threshold=args.conf, device=args.device,
            )
    else:
        # Single model mode
        model = load_model(list(model_paths.values())[0], args.device)
        for img_path in images_to_process:
            img_name = Path(img_path).stem
            out_path = str(output_dir / f"detect_{img_name}.jpg")
            detect_image(
                model, img_path, out_path,
                conf_threshold=args.conf, use_soft_nms=args.soft_nms,
            )

    print(f"\nAll results saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
