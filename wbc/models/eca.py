"""Efficient Channel Attention (Wang et al., CVPR 2020).

Two flavours are provided that share the same mechanism - global average
pooling, a 1-D convolution across neighbouring channels, a sigmoid gate and a
channel-wise rescaling:

* :class:`ECABlock2d` for convolutional feature maps ``(B, C, H, W)``
* :class:`ECABlock1d` for token sequences ``(B, N, C)`` (ViT patch tokens)

The kernel size follows the adaptive rule of the ECA-Net paper,

    k = | log2(C) / gamma + b / gamma |_odd ,   gamma = 2, b = 1,

implemented exactly as in the reference code (``t = int(abs((log2 C + b) /
gamma))``, rounded *up* to the next odd integer).  For C = 512 (ResNet-18
``layer4``) and C = 768 (ViT-B) this yields k = 5, which is the value the
manuscript reports; the hard-coded k = 3 of the earlier public script did not
follow the rule and has been removed.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


def adaptive_kernel_size(channels: int, gamma: int = 2, b: int = 1) -> int:
    """Kernel size of the ECA 1-D convolution for ``channels`` channels."""
    if channels < 1:
        raise ValueError("channels must be positive")
    t = int(abs((math.log2(channels) + b) / gamma))
    k = t if t % 2 else t + 1
    return max(k, 1)


class _ECACore(nn.Module):
    def __init__(self, channels: int, k_size: Optional[int] = None, gamma: int = 2, b: int = 1):
        super().__init__()
        k = adaptive_kernel_size(channels, gamma, b) if k_size is None else int(k_size)
        if k % 2 == 0:
            raise ValueError("ECA kernel size must be odd")
        self.channels = channels
        self.k_size = k
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)

    def gate(self, pooled: torch.Tensor) -> torch.Tensor:
        """``pooled`` is ``(B, C)``; returns the sigmoid attention ``(B, C)``."""
        y = self.conv(pooled.unsqueeze(1))  # (B, 1, C) -> local cross-channel interaction
        return torch.sigmoid(y.squeeze(1))

    def extra_repr(self) -> str:
        return f"channels={self.channels}, k_size={self.k_size}"


class ECABlock2d(_ECACore):
    """ECA over a ``(B, C, H, W)`` feature map."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(dim=(2, 3))  # global average pooling, Eq. (4)
        a = self.gate(pooled)  # Eq. (5)
        return x * a[:, :, None, None]


class ECABlock1d(_ECACore):
    """ECA over a ``(B, N, C)`` token sequence (channels last)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(dim=1)  # average over tokens
        a = self.gate(pooled)
        return x * a[:, None, :]
