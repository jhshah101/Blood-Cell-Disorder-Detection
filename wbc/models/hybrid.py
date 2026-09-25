"""The hybrid ViT-ECA-CF classifier - one unambiguous computational graph.

Input normalisation is part of the model (``InputNormalization``) so that the
training script, the evaluation scripts and the inference backend cannot drift
apart, and so that the colour branch can read the *raw* [0, 1] RGB tensor.

Two backbones are supported and selected by ``ModelConfig.backbone``:

``cnn_hybrid``
    raw RGB -> normalise -> ResNet-18 (conv1 ... layer4) -> (B, 512, g, g)
            -> ECA (k = 5 adaptive) -> g*g structural tokens of dim 512
    raw RGB -> colour descriptors on the same g x g tiling -> linear -> d_c (3)
    concat  -> 512 + d_c (= 515) -> linear projection to ``embed_dim``
            -> [CLS] + learnable positional embedding -> Transformer encoder
               (``depth`` x ``num_heads``, pre-LN, GELU MLP) -> LN -> linear head
    The 515-dimensional fused token cannot feed a 12-head attention layer
    directly (515 = 5 x 103 is not divisible by 12), so the projection to
    ``embed_dim`` is a structural necessity, not a stylistic choice.

``vit_b16``
    raw RGB -> normalise -> ViT-B/16 patch embedding (16 x 16 conv, 196 tokens)
            -> ECA over the 768 channels -> (+ colour descriptors, d_c) ->
            linear fusion back to 768 (initialised to the identity on the first
            768 inputs so pretrained behaviour is preserved at step 0) ->
            pretrained ViT-B encoder (12 layers, 12 heads, MLP 3072) -> head.

Both paths share the colour branch, the ECA rule, the head and the input
normalisation, so ablations differ only in the component under test.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tvm

from ..config import Config, ModelConfig
from .color_features import ColorFeatureBranch
from .eca import ECABlock1d, ECABlock2d
from .transformer import TransformerEncoder

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class InputNormalization(nn.Module):
    """``imagenet`` (fixed channel statistics), ``per_image`` (Eq. 1) or ``none``."""

    def __init__(self, mode: str = "imagenet"):
        super().__init__()
        if mode not in ("imagenet", "per_image", "none"):
            raise ValueError(f"Unknown normalisation mode {mode!r}")
        self.mode = mode
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "imagenet":
            return (x - self.mean) / self.std
        if self.mode == "per_image":
            mu = x.mean(dim=(1, 2, 3), keepdim=True)
            sd = x.std(dim=(1, 2, 3), keepdim=True, unbiased=False)
            return (x - mu) / (sd + 1e-6)
        return x

    def extra_repr(self) -> str:
        return f"mode={self.mode}"


class ResNet18Features(nn.Module):
    """ResNet-18 up to ``layer4``: ``(B, 3, H, W) -> (B, 512, H/32, W/32)``."""

    out_channels = 512

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        m = tvm.resnet18(weights=weights)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = m.layer1, m.layer2, m.layer3, m.layer4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)


def resize_pos_embed(pos: torch.Tensor, new_tokens: int) -> torch.Tensor:
    """Bilinearly interpolate the patch part of a ``(1, 1+N0, D)`` positional embedding."""
    if pos.shape[1] == new_tokens:
        return pos
    cls_pos, patch_pos = pos[:, :1], pos[:, 1:]
    n0, d = patch_pos.shape[1], patch_pos.shape[2]
    g0 = int(math.sqrt(n0))
    g1 = int(math.sqrt(new_tokens - 1))
    if g0 * g0 != n0 or g1 * g1 != new_tokens - 1:
        raise ValueError("Positional embedding interpolation assumes square token grids")
    patch_pos = patch_pos.reshape(1, g0, g0, d).permute(0, 3, 1, 2)
    patch_pos = F.interpolate(patch_pos, size=(g1, g1), mode="bilinear", align_corners=False)
    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, g1 * g1, d)
    return torch.cat([cls_pos, patch_pos], dim=1)


class HybridViTECACF(nn.Module):
    """ViT-ECA-CF classifier (see module docstring)."""

    def __init__(self, num_classes: int, mcfg: ModelConfig, normalization: str = "imagenet", img_size: int = 224):
        super().__init__()
        if mcfg.backbone not in ("cnn_hybrid", "vit_b16"):
            raise ValueError("HybridViTECACF supports backbone 'cnn_hybrid' or 'vit_b16'")
        self.backbone_name = mcfg.backbone
        self.num_classes = num_classes
        self.input_norm = InputNormalization(normalization)
        self.use_eca = bool(mcfg.use_eca)
        self.use_cf = bool(mcfg.use_color_features)
        color_dim = int(mcfg.color_dim) if self.use_cf else 0

        if self.use_cf:
            self.color = ColorFeatureBranch(
                out_dim=color_dim,
                bins=mcfg.color_bins,
                color_spaces=mcfg.color_spaces,
                granularity=mcfg.color_granularity,
            )
        else:
            self.color = None

        if mcfg.backbone == "cnn_hybrid":
            self.cnn = ResNet18Features(pretrained=mcfg.pretrained)
            channels = self.cnn.out_channels
            self.eca = ECABlock2d(channels, mcfg.eca_k, mcfg.eca_gamma, mcfg.eca_b) if self.use_eca else nn.Identity()
            self.embed_dim = int(mcfg.embed_dim)
            self.fused_dim = channels + color_dim  # 512 + 3 = 515 with the defaults
            self.proj = nn.Linear(self.fused_dim, self.embed_dim)
            grid = img_size // 32
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.pos_embed = nn.Parameter(torch.zeros(1, 1 + grid * grid, self.embed_dim))
            self.pos_drop = nn.Dropout(mcfg.dropout)
            self.encoder = TransformerEncoder(
                self.embed_dim, mcfg.depth, mcfg.num_heads, mcfg.mlp_ratio, mcfg.dropout, mcfg.drop_path
            )
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:  # vit_b16
            weights = tvm.ViT_B_16_Weights.IMAGENET1K_V1 if mcfg.pretrained else None
            vit = tvm.vit_b_16(weights=weights)
            self.patch_size = vit.patch_size
            self.conv_proj = vit.conv_proj  # (B, 768, 14, 14)
            self.embed_dim = vit.hidden_dim  # 768
            self.eca = ECABlock1d(self.embed_dim, mcfg.eca_k, mcfg.eca_gamma, mcfg.eca_b) if self.use_eca else nn.Identity()
            self.fused_dim = self.embed_dim + color_dim
            self.fuse = nn.Linear(self.fused_dim, self.embed_dim)
            with torch.no_grad():  # identity on the structural part, zero on the colour part
                self.fuse.weight.zero_()
                self.fuse.weight[:, : self.embed_dim] = torch.eye(self.embed_dim)
                self.fuse.bias.zero_()
            self.cls_token = vit.class_token
            self.pos_embed = vit.encoder.pos_embedding
            self.pos_drop = vit.encoder.dropout
            self.vit_layers = vit.encoder.layers
            self.vit_ln = vit.encoder.ln

        self.head = nn.Linear(self.embed_dim, num_classes)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    # ------------------------------------------------------------------ #
    def structural_tokens(self, x_norm: torch.Tensor):
        """Return ``(tokens (B, N, C), grid)`` for the selected backbone."""
        if self.backbone_name == "cnn_hybrid":
            f = self.cnn(x_norm)  # (B, 512, g, g)
            f = self.eca(f)
            g = f.shape[-1]
            return f.flatten(2).transpose(1, 2), g
        f = self.conv_proj(x_norm)  # (B, 768, 14, 14)
        g = f.shape[-1]
        tokens = f.flatten(2).transpose(1, 2)
        return self.eca(tokens), g

    def forward_features(self, raw: torch.Tensor) -> torch.Tensor:
        """``raw`` is ``(B, 3, H, W)`` in [0, 1]; returns the [CLS] representation."""
        x = self.input_norm(raw)
        tokens, g = self.structural_tokens(x)
        if self.use_cf:
            tokens = torch.cat([tokens, self.color(raw, g)], dim=-1)  # (B, N, 515)
        if self.backbone_name == "cnn_hybrid":
            tokens = self.proj(tokens)
        else:
            tokens = self.fuse(tokens)
        b = tokens.shape[0]
        cls = self.cls_token.expand(b, -1, -1)
        seq = torch.cat([cls, tokens], dim=1)
        seq = seq + resize_pos_embed(self.pos_embed, seq.shape[1])
        seq = self.pos_drop(seq)
        if self.backbone_name == "cnn_hybrid":
            seq = self.encoder(seq)
        else:
            seq = self.vit_ln(self.vit_layers(seq))
        return seq[:, 0]

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(raw))
