#!/usr/bin/env python3
"""
Skin Disease Classification — ConvNeXt Training Pipeline.

Supports ablation study across multiple dimensions:
  - Backbone:   convnext_tiny (V1) | convnextv2_tiny (V2)
  - Pooling:    avg (AdaptiveAvgPool) | gem (GeM Pooling)
  - Multi-scale: single | multi (stage 2/3/4 fusion)
  - EMA:        on | off

Usage:
  # Baseline V1
  python train.py --model convnext_tiny --pooling avg

  # V2 + GeM + EMA
  python train.py --model convnextv2_tiny --pooling gem --ema

  # Multi-scale fusion variant
  python train.py --model convnext_tiny --multi-scale
"""

import argparse
import sys
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

import timm
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report

from models.modules import ConvNeXtWithFeatures, ModelEMA, GeMPool
from utils.visualize import plot_confusion_matrix, plot_training_curves

# ============================================================
# Config
# ============================================================

CLASS_NAMES = [
    "Acne", "Actinic Keratosis", "Benign Tumors", "Bullous",
    "Candidiasis", "Drug Eruption", "Eczema", "Infestations/Bites",
    "Lichen", "Lupus", "Moles", "Psoriasis",
    "Rosacea", "Seborrheic Keratoses", "Skin Cancer", "Sun/Sunlight Damage",
    "Tinea", "Unknown/Normal", "Vascular Tumors", "Vasculitis",
    "Vitiligo", "Warts",
]

NUM_CLASSES = len(CLASS_NAMES)

CLASS_RISK = {
    "Skin Cancer": "HIGH", "Actinic Keratosis": "MEDIUM",
    "Lupus": "MEDIUM", "Vasculitis": "MEDIUM", "Bullous": "MEDIUM",
    "Benign Tumors": "LOW", "Moles": "LOW", "Seborrheic Keratoses": "LOW",
    "Vascular Tumors": "LOW", "Warts": "LOW", "Acne": "LOW",
    "Candidiasis": "LOW", "Drug Eruption": "LOW", "Eczema": "LOW",
    "Infestations/Bites": "LOW", "Lichen": "LOW", "Psoriasis": "LOW",
    "Rosacea": "LOW", "Sun/Sunlight Damage": "LOW", "Tinea": "LOW",
    "Unknown/Normal": "LOW", "Vitiligo": "LOW",
}


# ============================================================
# Model Registry
# ============================================================

MODEL_CONFIGS = {
    "convnext_tiny": {
        "timm_name": "convnext_tiny.fb_in22k_ft_in1k",
        "desc": "ConvNeXt-Tiny (V1)",
        "default_img_size": 224,
        "channels": [192, 384, 768],
    },
    "convnextv2_tiny": {
        "timm_name": "convnextv2_tiny.fcmae_ft_in22k_in1k",
        "desc": "ConvNeXtV2-Tiny (GRN + FCMAE pretrained)",
        "default_img_size": 224,
        "channels": [192, 384, 768],
    },
    "convnext_small": {
        "timm_name": "convnext_small.fb_in22k_ft_in1k",
        "desc": "ConvNeXt-Small (V1)",
        "default_img_size": 224,
        "channels": [192, 384, 768],
    },
}


# ============================================================
# Data
# ============================================================

def build_transforms(img_size: int = 224, is_train: bool = True):
    """Build data augmentation transforms."""
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(img_size * 1.14)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
        ])


def build_dataloaders(data_root: str, img_size: int = 224, batch_size: int = 32,
                       num_workers: int = 4, val_split: float = 0.15):
    """Build train/val/test dataloaders."""
    train_dir = Path(data_root) / "Train"
    test_dir = Path(data_root) / "Test"

    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")

    # Full datasets
    train_ds_full = ImageFolder(str(train_dir), transform=build_transforms(img_size, is_train=True))
    val_ds_full = ImageFolder(str(train_dir), transform=build_transforms(img_size, is_train=False))

    # Split
    n_val = int(len(train_ds_full) * val_split)
    n_train = len(train_ds_full) - n_val
    train_ds, val_ds = random_split(
        train_ds_full, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    # Re-assign val with eval transforms
    val_indices = [train_ds_full.samples[i] for i in val_ds.indices]
    val_ds_full.samples = val_indices
    val_ds_full.targets = [s[1] for s in val_indices]
    # Simpler approach: recreate
    train_subset = torch.utils.data.Subset(
        ImageFolder(str(train_dir), transform=build_transforms(img_size, is_train=True)),
        train_ds.indices
    )
    val_subset = torch.utils.data.Subset(
        ImageFolder(str(train_dir), transform=build_transforms(img_size, is_train=False)),
        val_ds.indices
    )

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    test_loader = None
    if test_dir.exists():
        test_ds = ImageFolder(str(test_dir), transform=build_transforms(img_size, is_train=False))
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers, pin_memory=True)
        print(f"Test:  {len(test_ds)} images")

    print(f"Train: {len(train_subset)} images")
    print(f"Val:   {len(val_subset)} images")

    return train_loader, val_loader, test_loader


# ============================================================
# Model Building
# ============================================================

def build_model(
    model_name: str = "convnext_tiny",
    num_classes: int = 22,
    pooling: str = "avg",
    dropout: float = 0.3,
    use_multi_scale: bool = False,
):
    """
    Build a ConvNeXt classifier with specified variant.

    Args:
        model_name: One of MODEL_CONFIGS keys
        num_classes: Number of output classes
        pooling: "avg" for AdaptiveAvgPool2d, "gem" for GeM Pooling
        dropout: Dropout rate before classifier
        use_multi_scale: If True, use MultiScaleHead (stage 2/3/4 fusion)

    Returns:
        model, extra_info dict
    """
    cfg = MODEL_CONFIGS[model_name]

    # Load pretrained backbone
    backbone = timm.create_model(cfg["timm_name"], pretrained=True, num_classes=num_classes)

    # Wrap with feature extraction + custom head
    model = ConvNeXtWithFeatures(
        backbone=backbone,
        num_classes=num_classes,
        dropout=dropout,
        use_multi_scale=use_multi_scale,
        pooling=pooling,
    )

    extra_info = {
        "backbone": model_name,
        "pooling": pooling,
        "multi_scale": use_multi_scale,
        "num_params": sum(p.numel() for p in model.parameters()),
        "num_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }

    return model, extra_info


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion, device: str):
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        with autocast():
            logits = model(images)
            loss = criterion(logits, labels)

        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        total_loss += loss.item() * images.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = (np.array(all_preds) == np.array(all_labels)).mean()

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "preds": np.array(all_preds),
        "labels": np.array(all_labels),
        "probs": np.array(all_probs),
    }


# ============================================================
# Training
# ============================================================

def train_epoch(model, loader, optimizer, criterion, scaler, device, ema=None):
    """Train one epoch."""
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        with autocast():
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update EMA after each step
        if ema is not None:
            ema.update(model)

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        batch_acc = (preds == labels).float().mean()
        pbar.set_postfix({"loss": f"{loss.item():.3f}", "acc": f"{batch_acc:.3f}"})

    avg_loss = total_loss / len(loader.dataset)
    acc = (np.array(all_preds) == np.array(all_labels)).mean()
    return avg_loss, acc


def train_model(
    data_root: str,
    output_dir: str = "runs/convnext",
    model_name: str = "convnext_tiny",
    img_size: int = 224,
    batch_size: int = 32,
    epochs: int = 50,
    lr: float = 1e-4,
    weight_decay: float = 0.05,
    dropout: float = 0.3,
    pooling: str = "avg",
    use_multi_scale: bool = False,
    use_ema: bool = False,
    ema_decay: float = 0.999,
    device: str = "mps",
    num_workers: int = 4,
    patience: int = 10,
    label_smoothing: float = 0.1,
):
    """Full training pipeline with ablation support."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build experiment name from variant
    variant_parts = [model_name]
    if pooling == "gem":
        variant_parts.append("GeM")
    if use_multi_scale:
        variant_parts.append("MS")
    if use_ema:
        variant_parts.append("EMA")
    variant_name = "_".join(variant_parts)

    print("\n" + "=" * 60)
    print(f"ConvNeXt Classification — {variant_name}")
    print(f"Data:    {data_root}")
    print(f"Device:  {device}")
    print(f"Epochs:  {epochs} | Batch: {batch_size} | LR: {lr}")
    print("=" * 60)

    # Data
    train_loader, val_loader, test_loader = build_dataloaders(
        data_root, img_size, batch_size, num_workers
    )

    # Model
    model, extra_info = build_model(
        model_name=model_name,
        num_classes=NUM_CLASSES,
        pooling=pooling,
        dropout=dropout,
        use_multi_scale=use_multi_scale,
    )
    model = model.to(device)

    print(f"\nModel: {MODEL_CONFIGS[model_name]['desc']}")
    print(f"Pooling: {pooling} | Multi-scale: {use_multi_scale} | EMA: {use_ema}")
    print(f"Params: {extra_info['num_trainable']:,} trainable / {extra_info['num_params']:,} total")

    # EMA
    ema = ModelEMA(model, decay=ema_decay) if use_ema else None

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=epochs // 3, T_mult=2, eta_min=lr * 0.01
    )

    scaler = GradScaler()

    # Training loop
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        print("-" * 30)

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, scaler, device, ema
        )

        # Validate (with EMA weights if enabled)
        if ema is not None:
            ema.apply_shadow(model)
        val_result = evaluate(model, val_loader, criterion, device)
        if ema is not None:
            ema.restore(model)

        val_loss, val_acc = val_result["loss"], val_result["accuracy"]

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f} | LR: {current_lr:.2e}")

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0

            # Save with EMA state if enabled
            ema_state = ema.state_dict() if ema is not None else None

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "class_names": CLASS_NAMES,
                "class_risk": CLASS_RISK,
                "config": {
                    "model_name": model_name,
                    "pooling": pooling,
                    "multi_scale": use_multi_scale,
                    "ema": use_ema,
                    "img_size": img_size,
                },
            }
            if ema_state:
                checkpoint["ema_state"] = ema_state

            torch.save(checkpoint, output_dir / "best.pt")
            print(f"  ✓ Best model saved (acc={val_acc:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{patience})")

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    print(f"Best epoch: {best_epoch} | Best val acc: {best_val_acc:.4f}")

    # Load best weights (use "cpu" to avoid device serialization issues)
    checkpoint = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    # If we used EMA, apply EMA weights for final eval
    if ema is not None and "ema_state" in checkpoint:
        ema.load_state_dict(checkpoint["ema_state"])
        ema.apply_shadow(model)
        val_result = evaluate(model, val_loader, criterion, device)
        ema.restore(model)
        print(f"EMA Val Acc: {val_result['accuracy']:.4f}")
    else:
        val_result = evaluate(model, val_loader, criterion, device)

    print(f"\nValidation Set:")
    print(f"  Accuracy: {val_result['accuracy']:.4f}")
    print(f"  Loss:     {val_result['loss']:.4f}")

    print("\nClassification Report:")
    report = classification_report(
        val_result["labels"], val_result["preds"],
        target_names=CLASS_NAMES, zero_division=0
    )
    print(report)

    # Confusion matrix
    cm_path = output_dir / f"confusion_matrix_{variant_name}.png"
    plot_confusion_matrix(val_result["labels"], val_result["preds"],
                           CLASS_NAMES, str(cm_path))

    # Training curves
    curves_path = output_dir / f"training_curves_{variant_name}.png"
    plot_training_curves(history, str(curves_path))

    # Test set
    if test_loader is not None:
        test_result = evaluate(model, test_loader, criterion, device)
        print(f"\nTest Set:")
        print(f"  Accuracy: {test_result['accuracy']:.4f}")
        print(f"  Loss:     {test_result['loss']:.4f}")

        cm_path = output_dir / f"confusion_matrix_test_{variant_name}.png"
        plot_confusion_matrix(test_result["labels"], test_result["preds"],
                               CLASS_NAMES, str(cm_path))

    # Save config
    config_data = {
        "variant": variant_name,
        "class_names": CLASS_NAMES,
        "class_risk": CLASS_RISK,
        "num_classes": NUM_CLASSES,
        "img_size": img_size,
        "config": checkpoint["config"],
        "num_params": extra_info["num_trainable"],
    }
    with open(output_dir / f"config_{variant_name}.json", "w") as f:
        json.dump(config_data, f, indent=2)

    print(f"\nOutputs saved to: {output_dir.absolute()}")
    print(f"  - best.pt")
    print(f"  - config_{variant_name}.json")
    print(f"  - confusion_matrix_{variant_name}.png")
    print(f"  - training_curves_{variant_name}.png")

    return model


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train ConvNeXt for Skin Disease Classification (ablation-ready)"
    )
    # Data
    parser.add_argument("--data", type=str, default="dataset/SkinDisease/SkinDisease",
                        help="Dataset root path")
    parser.add_argument("--output", type=str, default="runs/convnext",
                        help="Output directory")

    # Model ablation
    parser.add_argument("--model", type=str, default="convnext_tiny",
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Backbone variant (V1 or V2)")
    parser.add_argument("--pooling", type=str, default="avg",
                        choices=["avg", "gem"],
                        help="Pooling method")
    parser.add_argument("--multi-scale", action="store_true",
                        help="Enable multi-scale feature fusion (stages 2/3/4)")
    parser.add_argument("--ema", action="store_true",
                        help="Enable EMA weight tracking")

    # Training
    parser.add_argument("--img-size", type=int, default=224,
                        help="Input image size (224 or 384)")
    parser.add_argument("--batch", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Max epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.05,
                        help="Weight decay")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout rate")
    parser.add_argument("--label-smoothing", type=float, default=0.1,
                        help="Label smoothing")
    parser.add_argument("--ema-decay", type=float, default=0.999,
                        help="EMA decay rate")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience")

    # System
    parser.add_argument("--device", type=str, default="mps",
                        help="Device (mps/cpu/cuda)")
    parser.add_argument("--workers", type=int, default=4,
                        help="DataLoader workers")

    # Quick test
    parser.add_argument("--quick", action="store_true",
                        help="Quick test: 5 epochs")

    args = parser.parse_args()

    if not Path(args.data).exists():
        print(f"[ERROR] Dataset not found: {args.data}")
        sys.exit(1)

    epochs = 5 if args.quick else args.epochs

    train_model(
        data_root=args.data,
        output_dir=args.output,
        model_name=args.model,
        img_size=args.img_size,
        batch_size=args.batch,
        epochs=epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        pooling=args.pooling,
        use_multi_scale=args.multi_scale,
        use_ema=args.ema,
        ema_decay=args.ema_decay,
        device=args.device,
        num_workers=args.workers,
        patience=args.patience,
        label_smoothing=args.label_smoothing,
    )


if __name__ == "__main__":
    main()
