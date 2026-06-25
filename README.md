# 🩺 Skin Lesion Detection System

**AI-Assisted Multi-Class Skin Lesion Detection with Improved YOLOv8**

---

## Overview

This project builds a deep learning system to **detect and classify skin lesions** in dermoscopic images. It uses an improved YOLOv8 architecture enhanced with:

- **Coordinate Attention (CA)** — Better spatial localization for irregular lesion shapes
- **Adaptive Spatial Feature Fusion (ASFF)** — Multi-scale fusion for varying lesion sizes
- **Boundary-Sensitive Loss** — Tighter bounding boxes for medical imaging

## Quick Start

```bash
# 1. Activate environment
conda activate cv

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate pseudo-labels from classification dataset
python utils/pseudo_label.py \
    --raw dataset/SkinDisease/SkinDisease \
    --out data \
    --img-size 640

# 4. Train models (all 3 variants)
python train.py --model all --epochs 100

# 5. Run detection on a test image
python detect.py --compare baseline,boundary,improved --image data/images/test/example.jpg

# 6. Launch interactive demo
python demo.py
```

## Project Structure

```
skin_detect/
├── data/
│   ├── images/                  # YOLO-format images (train/val/test)
│   ├── labels/                  # YOLO-format labels
│   └── dataset.yaml             # Class definitions
├── models/
│   ├── yolov8s.yaml             # Baseline config
│   ├── yolov8s_asff_ca.yaml     # Improved config
│   ├── attention.py             # Coordinate Attention module
│   └── asff.py                  # ASFF module
├── utils/
│   ├── pseudo_label.py          # Pseudo-label generation
│   ├── loss.py                  # Boundary-sensitive loss
│   ├── soft_nms.py              # Soft-NMS implementation
│   └── visualize.py             # Visualization utilities
├── runs/                        # Training outputs
│   ├── baseline/
│   ├── boundary/
│   └── improved/
├── results/                     # Detection outputs
├── train.py                     # Training script
├── detect.py                    # Detection & comparison
├── demo.py                      # Gradio web demo
├── requirements.txt
└── README.md
```

## Model Variants

| Model | Architecture | Loss | Purpose |
|-------|-------------|------|---------|
| `baseline` | YOLOv8s | CIoU + DFL | Control baseline |
| `boundary` | YOLOv8s | CIoU + DFL + Boundary | Isolate loss improvement |
| `improved` | YOLOv8s + CA + ASFF | CIoU + DFL + Boundary | Full improvements |

## Detection Classes

| Class | Risk Level | Color |
|-------|-----------|-------|
| Skin Cancer | 🔴 HIGH | Red |
| Actinic Keratosis | 🟡 MEDIUM | Orange |
| Nevus (Mole) | 🟢 LOW | Cyan |
| Seborrheic Keratosis | 🟢 LOW | Yellow |
| Benign Tumor | 🟢 LOW | Green |
| Vascular Lesion | 🟢 LOW | Magenta |

## Training Configuration

- **Input Size**: 640×640
- **Optimizer**: AdamW, lr=1e-3, cosine annealing
- **Batch Size**: 16
- **Epochs**: 100
- **Pretrained**: COCO weights (transfer learning)
- **Device**: MPS (Apple Silicon) / CUDA / CPU

## Key Innovations

1. **Pseudo-Label Generation**: CV-based segmentation (LAB color space, Otsu thresholding, morphological cleanup) converts classification images to detection format.

2. **Coordinate Attention**: 1D horizontal + vertical pooling captures long-range spatial dependencies while preserving precise positional information — critical for irregular lesion boundaries.

3. **ASFF Fusion**: Three FPN levels are adaptively fused with learned per-pixel weights, enabling detection heads to leverage multi-scale context.

4. **Boundary-Sensitive Loss**: L1 edge-distance penalty on the four box boundaries encourages tight lesion fit, improving clinical relevance.

5. **Soft-NMS**: Gaussian score decay prevents suppression of neighboring lesions in dense cases.

## ⚠️ Disclaimer

**This is a research prototype and NOT a medical device.** All results require verification by a qualified dermatologist. Do not use for clinical diagnosis.
