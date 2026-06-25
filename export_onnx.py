#!/usr/bin/env python3
"""
Export trained ConvNeXt models to ONNX format.

Usage:
    python export_onnx.py                          # export all models in runs/
    python export_onnx.py --model runs/convnext_tiny/best.pt
"""

import argparse
import json
from pathlib import Path

import torch
import timm

from models.modules import ConvNeXtWithFeatures

CLASS_NAMES_ZH_FALLBACK = {
    "Acne": "痤疮", "Actinic Keratosis": "光化性角化病",
    "Benign Tumors": "良性肿瘤", "Bullous": "大疱性皮肤病",
    "Candidiasis": "念珠菌病", "Drug Eruption": "药疹",
    "Eczema": "湿疹", "Infestations/Bites": "寄生虫/虫咬",
    "Lichen": "苔藓", "Lupus": "红斑狼疮", "Moles": "痣",
    "Psoriasis": "银屑病", "Rosacea": "玫瑰痤疮",
    "Seborrheic Keratoses": "脂溢性角化病", "Skin Cancer": "皮肤癌",
    "Sun/Sunlight Damage": "日光性损伤", "Tinea": "癣",
    "Unknown/Normal": "未知/正常", "Vascular Tumors": "血管性肿瘤",
    "Vasculitis": "血管炎", "Vitiligo": "白癜风", "Warts": "疣",
}


def load_checkpoint(checkpoint_path: str, device: str = "cpu"):
    """Load model from checkpoint with all backward compat remapping."""
    print(f"Loading: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    class_names = checkpoint.get("class_names", [])
    class_names_zh = checkpoint.get("class_names_zh") or CLASS_NAMES_ZH_FALLBACK
    num_classes = len(class_names)
    model_cfg = checkpoint.get("config", {})
    model_name = model_cfg.get("model_name", "convnext_tiny")
    pooling = model_cfg.get("pooling", "avg")
    use_multi_scale = model_cfg.get("multi_scale", False)

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
    # EM状态加载（若存在）
    if "ema_state" in checkpoint:
        ema_state = checkpoint["ema_state"]
        for name in ema_state["shadow"]:
            # 映射EMA阴影到模型参数（去掉可能的backbone.前缀）
            model_key = name
            if name not in dict(model.named_parameters()):
                model_key = name.replace("shadow.", "").replace("backbone.", "")
                if model_key not in dict(model.named_parameters()):
                    continue
            # 直接用shadow覆盖参数
            param = dict(model.named_parameters()).get(model_key)
            if param is not None:
                param.data.copy_(ema_state["shadow"][name])
    else:
        # 普通状态字典加载
        state_dict = checkpoint["model_state_dict"]
        if any(k.startswith("backbone.") for k in state_dict):
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        if "head.0.weight" in state_dict and "head.2.weight" not in state_dict:
            remap = {}
            for k in list(state_dict.keys()):
                if k.startswith("head.0."):
                    remap[k] = k.replace("head.0.", "head.2.")
                elif k.startswith("head.4."):
                    remap[k] = k
            for old_k, new_k in remap.items():
                state_dict[new_k] = state_dict.pop(old_k)
        model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()
    return model, class_names, class_names_zh, model_cfg


def export_onnx(model, output_path: str, img_size: int = 224, device: str = "cpu"):
    """Export model to ONNX with dynamic batch size."""
    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size, device=device)

    print(f"  Exporting ONNX to: {output_path}")

    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        opset_version=14,
        do_constant_folding=True,
    )

    # Verify
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"  ✓ ONNX model valid: {output_path}")
    print(f"    Input:  {onnx_model.graph.input[0].name} {[d.dim_value for d in onnx_model.graph.input[0].type.tensor_type.shape.dim]}")
    print(f"    Output: {onnx_model.graph.output[0].name}")


def main():
    parser = argparse.ArgumentParser(description="Export ConvNeXt models to ONNX")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to specific checkpoint (default: all in runs/)")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if args.model:
        checkpoints = [Path(args.model)]
    else:
        checkpoints = sorted(Path("runs").rglob("best.pt"),
                             key=lambda p: p.stat().st_mtime, reverse=True)

    if not checkpoints:
        print("No checkpoints found.")
        return

    for ckpt_path in checkpoints:
        if not ckpt_path.exists():
            print(f"  [SKIP] Not found: {ckpt_path}")
            continue

        model, class_names, class_names_zh, model_cfg = load_checkpoint(str(ckpt_path), args.device)

        # Save ONNX alongside the checkpoint
        onnx_path = ckpt_path.parent / f"{ckpt_path.parent.name}.onnx"
        export_onnx(model, str(onnx_path), args.img_size, args.device)

        # Also save a lightweight config for inference
        config_path = ckpt_path.parent / "onnx_config.json"
        config_data = {
            "class_names": class_names,
            "class_names_zh": class_names_zh,
            "img_size": args.img_size,
            "model_config": model_cfg,
        }
        config_path.write_text(json.dumps(config_data, indent=2, ensure_ascii=False))
        print(f"  ✓ Config saved: {config_path}")

        del model

    print(f"\n{'=' * 50}")
    print(f"Exported {len(checkpoints)} model(s) to ONNX.")
    print(f"Files are in runs/<experiment_name>/")


if __name__ == "__main__":
    main()
