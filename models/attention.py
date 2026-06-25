"""
Coordinate Attention (CA) Module
Reference: "Coordinate Attention for Efficient Mobile Network Design" (CVPR 2021)

Encodes both channel relationships and long-range spatial dependencies
using 1D horizontal + vertical pooling and a shared transformation.
"""

import torch
import torch.nn as nn


class CoordAtt(nn.Module):
    """
    Coordinate Attention Layer.

    Args:
        inp:    Input channels
        oup:    Output channels (default: same as input)
        reduction: Reduction ratio for the shared 1x1 conv (default: 32)
    """

    def __init__(self, inp: int, oup: int = None, reduction: int = 32):
        super().__init__()
        oup = oup or inp
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # (H, 1)
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # (1, W)

        mip = max(8, inp // reduction)

        # Shared 1x1 conv for both directions
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)  # Using ReLU as in the paper; SiLU also works

        # Separate transform for H and W
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        n, c, h, w = x.size()

        # Horizontal pool: (B, C, H, 1)
        x_h = self.pool_h(x)
        # Vertical pool: (B, C, 1, W)
        x_w = self.pool_w(x)

        # Permute x_w to (B, C, 1, W) → (B, C, W, 1) → cat with x_h
        x_w = x_w.permute(0, 1, 3, 2)  # (B, C, W, 1)

        # Concatenate along spatial dim: (B, C, H+W, 1)
        y = torch.cat([x_h, x_w], dim=2)

        # Shared transform
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # Split back
        x_h, x_w = torch.split(y, [h, w], dim=2)

        # Separate transforms
        x_w = x_w.permute(0, 1, 3, 2)  # (B, C, W, 1) → (B, C, 1, W)
        a_h = self.conv_h(x_h).sigmoid()  # (B, C, H, 1)
        a_w = self.conv_w(x_w).sigmoid()  # (B, C, 1, W)

        # Apply attention
        out = identity * a_h * a_w

        return out


class CoordAttWrapper(nn.Module):
    """
    Wrapper to insert Coordinate Attention after a C2f block in ultralytics.
    Simply applies CA as a residual attention gate.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.ca = CoordAtt(channels)

    def forward(self, x):
        return self.ca(x)
