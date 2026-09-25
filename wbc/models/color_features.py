"""Colour-feature (CF) branch.

For every colour channel of the requested colour spaces (RGB, HSV, CIELAB) the
branch computes, on each token region of the image, the four moment
descriptors of manuscript Eqs. (7)-(10) and a fixed-bin histogram, Eq. (11):

* mean                mu      = E[x]
* standard deviation  sigma   = sqrt(E[x^2] - mu^2)
* skewness            gamma_1 = E[(x - mu)^3] / sigma^3          (asymmetry)
* kurtosis            kappa   = E[(x - mu)^4] / sigma^4          (tailedness, 3 for a Gaussian)
* histogram           h_k     = mean( 1[x in bin k] ),  k = 1..B

Every channel is first mapped to [0, 1] so that histogram bins are comparable
across colour spaces.  The raw descriptor vector has
``n_channels * (4 + bins)`` entries (9 * 12 = 108 with the defaults) and is
projected by one linear layer to ``out_dim`` values (3 in the manuscript).  The
projection is the only learnable part of the branch, exactly as described in
the Methods section; the descriptors themselves are deterministic functions of
the pixels and need no gradient.

All colour-space conversions are written in pure PyTorch so the branch runs on
the GPU inside the model, needs no OpenCV / scikit-image dependency and sees
the *un-normalised* [0, 1] RGB tensor (the model applies its own input
normalisation afterwards, see :mod:`wbc.models.hybrid`).
"""
from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

_EPS = 1e-6


# --------------------------------------------------------------------------- #
# Colour-space conversions (inputs are (B, 3, H, W) in [0, 1])
# --------------------------------------------------------------------------- #
def rgb_to_hsv(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """RGB -> HSV with H, S, V all in [0, 1] (same convention as torchvision)."""
    r, g, b = x.unbind(dim=1)
    maxc, _ = x.max(dim=1)
    minc, _ = x.min(dim=1)
    v = maxc
    delta = maxc - minc
    s = delta / (maxc + eps)
    rc = (maxc - r) / (delta + eps)
    gc = (maxc - g) / (delta + eps)
    bc = (maxc - b) / (delta + eps)
    h = torch.where(maxc == r, bc - gc, torch.where(maxc == g, 2.0 + rc - bc, 4.0 + gc - rc))
    h = (h / 6.0) % 1.0
    h = torch.where(delta > eps, h, torch.zeros_like(h))
    return torch.stack([h, s, v], dim=1)


def rgb_to_lab(x: torch.Tensor) -> torch.Tensor:
    """sRGB (D65) -> CIELAB.  Returns L in [0, 100], a and b roughly in [-128, 127]."""
    lin = torch.where(x > 0.04045, ((x + 0.055) / 1.055).clamp_min(0) ** 2.4, x / 12.92)
    r, g, b = lin.unbind(dim=1)
    X = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    Y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    Z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
    X = X / 0.95047
    Z = Z / 1.08883

    def f(t: torch.Tensor) -> torch.Tensor:
        return torch.where(t > 0.008856, t.clamp_min(1e-8) ** (1.0 / 3.0), 7.787 * t + 16.0 / 116.0)

    fx, fy, fz = f(X), f(Y), f(Z)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    bb = 200.0 * (fy - fz)
    return torch.stack([L, a, bb], dim=1)


def lab_to_unit(lab: torch.Tensor) -> torch.Tensor:
    """Map CIELAB to [0, 1] per channel so that histogram bins are shared."""
    L = (lab[:, 0:1] / 100.0).clamp(0, 1)
    a = ((lab[:, 1:2] + 128.0) / 255.0).clamp(0, 1)
    b = ((lab[:, 2:3] + 128.0) / 255.0).clamp(0, 1)
    return torch.cat([L, a, b], dim=1)


def stack_color_spaces(rgb: torch.Tensor, spaces: Sequence[str]) -> torch.Tensor:
    """Concatenate the requested colour spaces along the channel axis, all in [0, 1]."""
    chans: List[torch.Tensor] = []
    for space in spaces:
        if space == "rgb":
            chans.append(rgb.clamp(0, 1))
        elif space == "hsv":
            chans.append(rgb_to_hsv(rgb.clamp(0, 1)))
        elif space == "lab":
            chans.append(lab_to_unit(rgb_to_lab(rgb.clamp(0, 1))))
        else:
            raise ValueError(f"Unknown colour space {space!r}")
    return torch.cat(chans, dim=1)


# --------------------------------------------------------------------------- #
# Region descriptors
# --------------------------------------------------------------------------- #
def region_moments(x: torch.Tensor, grid: int) -> torch.Tensor:
    """Mean, std, skewness and kurtosis of every channel on a ``grid x grid`` tiling.

    ``x`` is ``(B, K, H, W)``; the result is ``(B, 4K, grid, grid)`` ordered as
    [mean_1..K, std_1..K, skew_1..K, kurt_1..K].  Moments are obtained from
    pooled powers, which is exact and fully vectorised.
    """
    m1 = F.adaptive_avg_pool2d(x, grid)
    m2 = F.adaptive_avg_pool2d(x * x, grid)
    m3 = F.adaptive_avg_pool2d(x * x * x, grid)
    m4 = F.adaptive_avg_pool2d(x * x * x * x, grid)
    var = (m2 - m1 * m1).clamp_min(0.0)
    sd = torch.sqrt(var + _EPS)
    central3 = m3 - 3.0 * m1 * var - m1 * m1 * m1  # E[(x - mu)^3]
    central4 = m4 - 4.0 * m1 * m3 + 6.0 * m1 * m1 * m2 - 3.0 * m1**4  # E[(x - mu)^4]
    skew = central3 / (sd**3 + _EPS)
    kurt = central4 / (var * var + _EPS)
    return torch.cat([m1, sd, skew, kurt], dim=1)


def region_histograms(x: torch.Tensor, grid: int, bins: int) -> torch.Tensor:
    """Normalised ``bins``-bin histogram of every channel on a ``grid x grid`` tiling.

    ``x`` is ``(B, K, H, W)`` in [0, 1]; the result is ``(B, K * bins, grid, grid)``
    with bin fractions summing to one over the ``bins`` axis of each channel.
    """
    edges = torch.linspace(0.0, 1.0, bins + 1, device=x.device, dtype=x.dtype)
    feats: List[torch.Tensor] = []
    for k in range(bins):
        lo, hi = edges[k], edges[k + 1]
        if k == bins - 1:
            ind = (x >= lo) & (x <= hi)
        else:
            ind = (x >= lo) & (x < hi)
        feats.append(F.adaptive_avg_pool2d(ind.to(x.dtype), grid))
    return torch.cat(feats, dim=1)


class ColorFeatureBranch(nn.Module):
    """Hand-crafted colour descriptors -> one linear projection to ``out_dim``."""

    def __init__(
        self,
        out_dim: int = 3,
        bins: int = 8,
        color_spaces: Sequence[str] = ("rgb", "hsv", "lab"),
        granularity: str = "patch",
    ):
        super().__init__()
        if granularity not in ("patch", "image"):
            raise ValueError("granularity must be 'patch' or 'image'")
        self.color_spaces = tuple(color_spaces)
        self.bins = int(bins)
        self.granularity = granularity
        self.n_channels = 3 * len(self.color_spaces)
        self.raw_dim = self.n_channels * (4 + self.bins)
        self.out_dim = int(out_dim)
        # LayerNorm puts moments (unbounded) and histogram fractions (in [0, 1])
        # on a common scale before the single fully connected layer.
        self.proj = nn.Sequential(nn.LayerNorm(self.raw_dim), nn.Linear(self.raw_dim, self.out_dim))

    @torch.no_grad()
    def descriptors(self, rgb: torch.Tensor, grid: int) -> torch.Tensor:
        """Deterministic descriptor tensor ``(B, grid*grid, raw_dim)``."""
        g = grid if self.granularity == "patch" else 1
        chans = stack_color_spaces(rgb.float(), self.color_spaces)
        moments = region_moments(chans, g)
        hists = region_histograms(chans, g, self.bins)
        feats = torch.cat([moments, hists], dim=1)  # (B, raw_dim, g, g)
        feats = feats.flatten(2).transpose(1, 2)  # (B, g*g, raw_dim)
        if self.granularity == "image":
            feats = feats.expand(-1, grid * grid, -1)
        return feats.contiguous()

    def forward(self, rgb: torch.Tensor, grid: int) -> torch.Tensor:
        feats = self.descriptors(rgb, grid).to(self.proj[1].weight.dtype)
        return self.proj(feats)  # (B, grid*grid, out_dim)

    def extra_repr(self) -> str:
        return (
            f"spaces={self.color_spaces}, bins={self.bins}, raw_dim={self.raw_dim}, "
            f"out_dim={self.out_dim}, granularity={self.granularity}"
        )
