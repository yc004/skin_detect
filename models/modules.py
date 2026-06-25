"""
Custom modules for ConvNeXt classification — ablation-ready.

Contains:
  - GeMPool: Generalized Mean Pooling (replaces AdaptiveAvgPool2d)
  - MultiScaleHead: Fuses features from multiple backbone stages
  - ModelEMA: Exponential Moving Average of model weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


# ============================================================
# GeM Pooling
# ============================================================

class GeMPool(nn.Module):
    """
    Generalized Mean Pooling.

    GeM(p) = (Avg_pool(feature^p)) ^ (1/p)

    - p=1   → average pooling
    - p→∞   → max pooling
    - p=3   → common default for fine-grained retrieval

    p is a learnable parameter, initialized to `p_init`.

    Reference: Radenovic et al., "Fine-tuning CNN Image Retrieval
               with No Human Annotation", PAMI 2018.
    """

    def __init__(self, p_init: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p_init)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) feature map

        Returns:
            (B, C) pooled descriptor
        """
        # Clamp p to avoid numerical issues
        p = self.p.clamp(min=1.0, max=10.0)

        # GeM: (mean(x^p))^(1/p)
        x = x.clamp(min=self.eps)
        x_p = x.pow(p)
        x_pooled = x_p.mean(dim=(2, 3))  # (B, C)
        x_out = x_pooled.pow(1.0 / p)

        return x_out

    def extra_repr(self) -> str:
        return f"p={self.p.item():.2f}"


# ============================================================
# Multi-Scale Feature Fusion Head
# ============================================================

class MultiScaleHead(nn.Module):
    """
    Extract features from multiple ConvNeXt stages, project each to
    a common dimension, concat, and classify.

    ConvNeXt stage output channels (Tiny):
      Stage 1: 96
      Stage 2: 192
      Stage 3: 384
      Stage 4: 768

    We take Stage 2, 3, 4, resize to fixed spatial size, project each
    to `hidden_dim`, pool (avg or gem), concat, then classify.
    """

    def __init__(
        self,
        in_channels: tuple = (192, 384, 768),
        hidden_dim: int = 256,
        num_classes: int = 22,
        dropout: float = 0.3,
        pooling: str = "avg",   # "avg" | "gem"
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.pooling_type = pooling

        # Projection layers: stage_i → hidden_dim
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(inplace=True),
            )
            for ch in in_channels
        ])

        # Pooling
        if pooling == "gem":
            self.pool = GeMPool()
        else:
            self.pool = nn.AdaptiveAvgPool2d(1)

        # Total channels after concat
        total_dim = hidden_dim * len(in_channels)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(total_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim * 2, num_classes),
        )

    def forward(self, features: dict) -> torch.Tensor:
        """
        Args:
            features: Dict[str, Tensor] with keys "stage2", "stage3", "stage4"
                      Each tensor is (B, C, H, W)

        Returns:
            (B, num_classes) logits
        """
        pooled = []
        for i, proj in enumerate(self.projections):
            f = features[f"stage{i+2}"]        # stage2, stage3, stage4
            f = proj(f)                          # (B, hidden_dim, H, W)
            f = self.pool(f)                     # (B, hidden_dim, 1, 1)
            pooled.append(f.flatten(1))          # (B, hidden_dim)

        fused = torch.cat(pooled, dim=1)         # (B, hidden_dim * 3)
        return self.classifier(fused)


# ============================================================
# ConvNeXt with Feature Extraction Hooks
# ============================================================

class ConvNeXtWithFeatures(nn.Module):
    """
    Wraps a timm ConvNeXt model and exposes intermediate stage features.

    Stages in ConvNeXt-Tiny:
      - stem          → (B, 96, H/4, W/4)
      - stages.0      → (B, 96, H/4, W/4)   ← Stage 1
      - stages.1      → (B, 192, H/8, W/8)  ← Stage 2
      - stages.2      → (B, 384, H/16, W/16) ← Stage 3
      - stages.3      → (B, 768, H/32, W/32) ← Stage 4
      - head.norm + fc for classification

    We capture stage 1-4 outputs, then apply either:
      - Default head (avgpool + classifier) — single-scale
      - MultiScaleHead                         — multi-scale fusion
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int = 22,
        dropout: float = 0.3,
        use_multi_scale: bool = False,
        pooling: str = "avg",   # "avg" | "gem"
    ):
        super().__init__()
        self.backbone = backbone
        self.use_multi_scale = use_multi_scale
        self.stage_channels = []

        # Detect stage channels
        self._probe_channels(backbone)

        if use_multi_scale:
            # Multi-scale fusion head
            stage_chs = tuple(self.stage_channels[-3:]) if len(self.stage_channels) >= 3 else (192, 384, 768)
            self.head = MultiScaleHead(
                in_channels=stage_chs,
                hidden_dim=256,
                num_classes=num_classes,
                dropout=dropout,
                pooling=pooling,
            )
            # Replace the backbone's original head with identity
            if hasattr(backbone, 'head'):
                backbone.head = nn.Identity()
            if hasattr(backbone, 'fc'):
                backbone.fc = nn.Identity()
        else:
            # Single-scale: replace classifier head
            in_features = self._get_backbone_feat_dim(backbone)
            if pooling == "gem":
                pool_layer = GeMPool()
            else:
                pool_layer = None

            # Replace backbone head
            if hasattr(backbone, 'head'):
                backbone.head.fc = nn.Linear(
                    backbone.head.norm.normalized_shape[0] if hasattr(backbone.head, 'norm') else 768,
                    num_classes
                )
                # Override whole head to add dropout
                backbone.head = nn.Sequential(
                    backbone.head.norm if hasattr(backbone.head, 'norm') else nn.Identity(),
                    pool_layer if pool_layer else nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(1),
                    nn.Dropout(dropout),
                    nn.Linear(
                        backbone.head.norm.normalized_shape[0] if hasattr(backbone.head, 'norm') else 768,
                        num_classes
                    )
                )

        self._stage_features = {}

    def _probe_channels(self, backbone):
        """Probe each stage's output channels with a dummy forward."""
        try:
            dummy = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                _ = self._forward_stages(dummy)
        except Exception:
            self.stage_channels = [96, 192, 384, 768]  # ConvNeXt-Tiny defaults

    def _forward_stages(self, x):
        """Forward through backbone, capturing stage outputs."""
        self._stage_features = {}

        # Stem
        x = self.backbone.stem(x)
        self._stage_features["stage0"] = x  # Before stages

        # Stages
        for i, stage in enumerate(self.backbone.stages):
            x = stage(x)
            self._stage_features[f"stage{i+1}"] = x

        return x

    def _get_backbone_feat_dim(self, backbone):
        """Try to determine the backbone's feature dimension."""
        # Common paths in timm ConvNeXt
        if hasattr(backbone, 'head'):
            head = backbone.head
            if hasattr(head, 'fc') and hasattr(head.fc, 'in_features'):
                return head.fc.in_features
            if hasattr(head, 'norm'):
                if isinstance(head.norm, nn.LayerNorm):
                    return head.norm.normalized_shape[0]
                if hasattr(head.norm, 'normalized_shape'):
                    return head.norm.normalized_shape[0]
        if hasattr(backbone, 'num_features'):
            return backbone.num_features
        return 768  # ConvNeXt-Tiny default

    def forward(self, x):
        if self.use_multi_scale:
            _ = self._forward_stages(x)
            # Use stage2, stage3, stage4 for multi-scale
            features = {
                "stage2": self._stage_features.get("stage2"),
                "stage3": self._stage_features.get("stage3"),
                "stage4": self._stage_features.get("stage4"),
            }
            if features["stage4"] is None:
                # Fallback: forward again through head path
                raise RuntimeError("Multi-scale features not captured")
            return self.head(features)
        else:
            return self.backbone(x)


# ============================================================
# Model EMA
# ============================================================

class ModelEMA:
    """
    Exponential Moving Average of model weights.

    Keeps a shadow copy of model parameters updated as:
        shadow = decay * shadow + (1 - decay) * param

    The EMA model is used for validation/inference, not training.

    Usage:
        ema = ModelEMA(model, decay=0.999)
        # ... after each optimizer.step() ...
        ema.update(model)
        # ... at eval time ...
        ema.apply_shadow(model)   # replace model weights with EMA
        # ... evaluate ...
        ema.restore(model)        # restore training weights
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self._register(model)

    def _register(self, model: nn.Module):
        """Initialize shadow from current model parameters."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module):
        """Update shadow parameters after a training step."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply_shadow(self, model: nn.Module):
        """Replace model parameters with EMA shadow for evaluation."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        """Restore original training parameters."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup.clear()

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state_dict: dict):
        self.decay = state_dict["decay"]
        self.shadow = state_dict["shadow"]
