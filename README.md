# 🩺 Skin Disease Classification

**AI-Assisted 22-Class Skin Disease Diagnosis with ConvNeXt-Tiny**

---

## Overview

This project builds a deep learning system to **classify skin diseases** from clinical/dermoscopic images. It uses ConvNeXt-Tiny pretrained on ImageNet-22k, fine-tuned on a 22-class dermatology dataset.

## Quick Start

```bash
# 1. Activate environment
conda activate cv

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model
python train.py --data dataset/SkinDisease/SkinDisease --epochs 50

# 4. Run inference on a test image
python detect.py --image test.jpg --cam

# 5. Launch interactive demo
python demo.py
```

## Project Structure

```
skin_detect/
├── dataset/                     # Raw datasets
│   └── SkinDisease/             # 22-class skin disease dataset
│       └── SkinDisease/
│           ├── Train/           # Training images by class
│           └── Test/            # Test images by class
├── models/
│   └── __init__.py
├── utils/
│   ├── visualize.py             # CAM, confusion matrix, training curves
│   └── __init__.py
├── runs/convnext/               # Training outputs
│   ├── best.pt                  # Best model checkpoint
│   ├── class_config.json        # Class names & risk levels
│   ├── confusion_matrix.png
│   └── training_curves.png
├── results/                     # Inference outputs
├── train.py                     # Training script
├── detect.py                    # Inference script
├── demo.py                      # Gradio web demo
├── requirements.txt
└── README.md
```

## Classification Classes (22)

| 🔴 HIGH Risk | 🟡 MEDIUM Risk | 🟢 LOW Risk |
|-------------|---------------|------------|
| Skin Cancer | Actinic Keratosis | Acne, Benign Tumors, Bullous |
| | Lupus, Vasculitis | Candidiasis, Drug Eruption, Eczema |
| | | Infestations/Bites, Lichen, Moles |
| | | Psoriasis, Rosacea, Seborrheic Keratoses |
| | | Sun/Sunlight Damage, Tinea, Unknown/Normal |
| | | Vascular Tumors, Vitiligo, Warts |

## Training Configuration

- **Architecture**: ConvNeXt-Tiny (28M params, pretrained on ImageNet-22k)
- **Input Size**: 224×224
- **Optimizer**: AdamW, lr=1e-4, cosine warm restarts
- **Loss**: CrossEntropy with label smoothing (0.1)
- **Batch Size**: 32
- **Epochs**: 50 (with early stopping, patience=10)
- **Augmentation**: RandCrop, Flip, Rotation, ColorJitter
- **Device**: MPS (Apple Silicon) / CUDA / CPU

## Key Features

- **Grad-CAM**: Visualize which regions the model focuses on for its prediction
- **Risk Stratification**: Automatic HIGH/MEDIUM/LOW risk classification for clinical triage
- **Label Smoothing**: Improves calibration and robustness
- **Early Stopping**: Prevents overfitting

## ⚠️ Disclaimer

**This is a research prototype and NOT a medical device.** All results require verification by a qualified dermatologist. Do not use for clinical diagnosis.
