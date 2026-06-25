"""
Adaptive Spatial Feature Fusion (ASFF)
Reference: "Learning Spatial Fusion for Single-Shot Object Detection" (2019)

Fuses features from multiple FPN levels with learned per-pixel weights.
Three ASFF modules: ASFF_3, ASFF_4, ASFF_5 — one per detection head level.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ASFF(nn.Module):
    """
    Adaptive Spatial Feature Fusion for a single target level.

    Takes N feature maps at different resolutions,
    resizes them to the target resolution,
    learns spatial weights, and fuses.

    Args:
        level:      Target level index (0=P3/80x80, 1=P4/40x40, 2=P5/20x20)
        num_levels: How many FPN levels to fuse (default: 3)
        channels:   List of channel dims for each level [c3, c4, c5]
    """

    def __init__(self, level: int, num_levels: int = 3, channels: list = None):
        super().__init__()
        self.level = level
        self.num_levels = num_levels
        if channels is None:
            channels = [128, 256, 512]  # Default for YOLOv8s

        # Ensure all input levels share the same channels after 1x1 conv
        target_ch = channels[level]
        self.compress = nn.ModuleList()
        for i in range(num_levels):
            self.compress.append(
                nn.Conv2d(channels[i], target_ch, kernel_size=1, bias=False)
                if channels[i] != target_ch
                else nn.Identity()
            )

        # Weight prediction: concat all levels → 3-channel spatial attention map → softmax
        self.weight_conv = nn.Sequential(
            nn.Conv2d(target_ch * num_levels, target_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(target_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(target_ch, num_levels, kernel_size=1, bias=True),
        )

    def _resize_to(self, x: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        """Resize feature map to target spatial size."""
        if x.shape[2] == target_h and x.shape[3] == target_w:
            return x
        return F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)

    def forward(self, features: list) -> torch.Tensor:
        """
        Args:
            features: List of feature maps [P3, P4, P5]

        Returns:
            Fused feature map at the target level resolution.
        """
        target_h, target_w = features[self.level].shape[2:]

        # Compress channels and resize each level to target spatial size
        resized = []
        for i, feat in enumerate(features):
            compressed = self.compress[i](feat)
            resized.append(self._resize_to(compressed, target_h, target_w))

        # Concat all levels: (B, 3C, H, W)
        concat = torch.cat(resized, dim=1)

        # Predict spatial weights: (B, 3, H, W)
        weights = self.weight_conv(concat)
        weights = F.softmax(weights, dim=1)  # Softmax across levels

        # Weighted sum fusion
        fused = torch.zeros_like(resized[0])
        for i, feat in enumerate(resized):
            fused += weights[:, i:i + 1] * feat

        return fused


class ASFFHead(nn.Module):
    """
    ASFF-augmented detection head wrapper.
    Replaces the standard 3-level detection with ASFF-fused features.

    Takes the 3 FPN outputs (P3/P4/P5), runs each through its ASFF module,
    then passes to the detection head.
    """

    def __init__(self, channels: list = None):
        super().__init__()
        if channels is None:
            channels = [128, 256, 512]

        self.asff_3 = ASFF(level=0, channels=channels)
        self.asff_4 = ASFF(level=1, channels=channels)
        self.asff_5 = ASFF(level=2, channels=channels)

    def forward(self, features: list) -> list:
        """
        Args:
            features: List of 3 FPN outputs [P3, P4, P5]
        Returns:
            ASFF-fused features at each level [F3, F4, F5]
        """
        return [
            self.asff_3(features),
            self.asff_4(features),
            self.asff_5(features),
        ]
