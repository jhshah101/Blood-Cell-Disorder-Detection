import colorsys

import numpy as np
import torch
from scipy import stats as sps

from wbc.models.color_features import (
    ColorFeatureBranch,
    region_histograms,
    region_moments,
    rgb_to_hsv,
    rgb_to_lab,
    stack_color_spaces,
)


def test_hsv_matches_colorsys():
    rng = np.random.default_rng(0)
    px = rng.random((1, 3, 1, 5)).astype(np.float32)
    ours = rgb_to_hsv(torch.from_numpy(px))[0, :, 0]
    for j in range(5):
        h, s, v = colorsys.rgb_to_hsv(*px[0, :, 0, j])
        assert abs(ours[0, j].item() - h) < 1e-4
        assert abs(ours[1, j].item() - s) < 1e-4
        assert abs(ours[2, j].item() - v) < 1e-4


def test_lab_reference_colours():
    white = torch.ones(1, 3, 1, 1)
    black = torch.zeros(1, 3, 1, 1)
    lw = rgb_to_lab(white)[0, :, 0, 0]
    lb = rgb_to_lab(black)[0, :, 0, 0]
    assert abs(lw[0].item() - 100.0) < 0.1 and abs(lw[1].item()) < 0.5 and abs(lw[2].item()) < 0.5
    assert abs(lb[0].item()) < 1e-3
    red = torch.tensor([1.0, 0.0, 0.0]).view(1, 3, 1, 1)
    lr = rgb_to_lab(red)[0, :, 0, 0]
    assert abs(lr[0].item() - 53.24) < 0.5 and abs(lr[1].item() - 80.09) < 1.0 and abs(lr[2].item() - 67.20) < 1.0


def test_region_moments_match_scipy_definitions():
    torch.manual_seed(0)
    x = torch.rand(1, 2, 32, 32)  # one region when grid = 1
    m = region_moments(x, 1)[0, :, 0, 0]  # [mean_1, mean_2, sd_1, sd_2, skew_1, skew_2, kurt_1, kurt_2]
    for c in range(2):
        v = x[0, c].numpy().ravel()
        assert abs(m[c].item() - v.mean()) < 1e-5
        assert abs(m[2 + c].item() - v.std()) < 1e-4
        assert abs(m[4 + c].item() - sps.skew(v)) < 1e-3
        assert abs(m[6 + c].item() - sps.kurtosis(v, fisher=False)) < 1e-3  # non-excess kurtosis


def test_histograms_sum_to_one_per_channel():
    torch.manual_seed(0)
    x = torch.rand(2, 3, 64, 64)
    h = region_histograms(x, 2, 8)  # (2, 3*8, 2, 2), ordered bin-major
    h = h.view(2, 8, 3, 2, 2).sum(dim=1)
    assert torch.allclose(h, torch.ones_like(h), atol=1e-6)


def test_branch_output_shape_and_manuscript_dimensions():
    branch = ColorFeatureBranch(out_dim=3, bins=8, color_spaces=("rgb", "hsv", "lab"), granularity="patch")
    assert branch.raw_dim == 9 * (4 + 8)  # 108 raw descriptors
    x = torch.rand(2, 3, 224, 224)
    out = branch(x, 7)
    assert out.shape == (2, 49, 3)
    out_img = ColorFeatureBranch(out_dim=3, granularity="image")(x, 7)
    assert out_img.shape == (2, 49, 3)
    assert torch.allclose(out_img[:, 0], out_img[:, 1])  # broadcast per image


def test_stack_color_spaces_range():
    x = torch.rand(1, 3, 16, 16)
    s = stack_color_spaces(x, ("rgb", "hsv", "lab"))
    assert s.shape[1] == 9
    assert s.min() >= 0 and s.max() <= 1
