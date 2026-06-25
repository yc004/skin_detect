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

    def __init__(self, p_init: float = 1.0, eps: float = 1e-6):
        super().__init__()
        # Start from 1.0 (avg pooling) — too high p_init (e.g. 3.0) causes
        # severe early-training slowdown because GeM amplifies random feature
        # activations when backbone weights are still unadapted.
        self.p = nn.Parameter(torch.ones(1) * p_init)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) feature map

        Returns:
            (B, C) pooled descriptor
        """
        # Clamp p: min=1 (avg pool), max=5 (prevent numerical overflow)
        p = self.p.clamp(min=1.0, max=5.0)

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
    Wraps a timm ConvNeXt model, discards its original head, and attaches
    a clean custom classifier.

    Two modes:
      - Single-scale: stem → stages → pool → classifier
      - Multi-scale:  stem → stages → [S2,S3,S4] → MultiScaleHead

    This avoids fragile surgery on `backbone.head` and gives full control
    over pooling (avg / GeM) and dropout.
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int = 22,
        dropout: float = 0.3,
        use_multi_scale: bool = False,
        pooling: str = "avg",
    ):
        super().__init__()
        self.use_multi_scale = use_multi_scale

        # Pull out stem + stages, discard original head
        self.stem = backbone.stem
        self.stages = backbone.stages

        # Determine feature dim at stage 4 output
        feat_dim = 768  # ConvNeXt-Tiny default
        if hasattr(backbone, 'head'):
            head = backbone.head
            if hasattr(head, 'norm') and isinstance(head.norm, nn.LayerNorm):
                feat_dim = head.norm.normalized_shape[0]

        if use_multi_scale:
            # Stage channels: [S1, S2, S3, S4] — use last 3 for fusion
            stage_chs = [96, 192, 384, feat_dim][-3:]  # (192, 384, 768)
            self.head = MultiScaleHead(
                in_channels=tuple(stage_chs),
                hidden_dim=256,
                num_classes=num_classes,
                dropout=dropout,
                pooling=pooling,
            )
        else:
            # Clean single-scale head
            norm_layer = nn.LayerNorm(feat_dim, eps=1e-6)

            if pooling == "gem":
                pool_layer = GeMPool(p_init=1.0)
            else:
                pool_layer = nn.AdaptiveAvgPool2d(1)

            self.head = nn.Sequential(
                norm_layer,
                pool_layer,
                nn.Flatten(1),
                nn.Dropout(dropout),
                nn.Linear(feat_dim, num_classes),
            )

    def _extract_stage_features(self, x):
        """Run stem + stages, return dict of stage outputs."""
        features = {}
        x = self.stem(x)
        features["stage0"] = x
        for i, stage in enumerate(self.stages):
            x = stage(x)
            features[f"stage{i+1}"] = x
        return features, x  # dict, final (stage4) output

    def forward(self, x):
        if self.use_multi_scale:
            feats, _ = self._extract_stage_features(x)
            return self.head({
                "stage2": feats.get("stage2"),
                "stage3": feats.get("stage3"),
                "stage4": feats.get("stage4"),
            })
        else:
            _, stage4 = self._extract_stage_features(x)
            return self.head(stage4)


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
