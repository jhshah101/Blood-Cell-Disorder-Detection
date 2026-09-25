"""Model factory."""
from __future__ import annotations

import torch.nn as nn

from ..config import CNN_BASELINES, Config
from .baselines import CNNBaseline
from .color_features import ColorFeatureBranch
from .eca import ECABlock1d, ECABlock2d, adaptive_kernel_size
from .hybrid import HybridViTECACF, InputNormalization

__all__ = [
    "build_model",
    "HybridViTECACF",
    "CNNBaseline",
    "ColorFeatureBranch",
    "ECABlock1d",
    "ECABlock2d",
    "InputNormalization",
    "adaptive_kernel_size",
]


def build_model(cfg: Config, num_classes: int) -> nn.Module:
    m = cfg.model
    if m.backbone in CNN_BASELINES:
        return CNNBaseline(m.backbone, num_classes, pretrained=m.pretrained, normalization=cfg.data.normalization)
    return HybridViTECACF(num_classes, m, normalization=cfg.data.normalization, img_size=cfg.data.img_size)
