import torch

from wbc.models.eca import ECABlock1d, ECABlock2d, adaptive_kernel_size


def test_adaptive_kernel_matches_manuscript_values():
    # gamma = 2, b = 1 as in ECA-Net and in the manuscript
    assert adaptive_kernel_size(512) == 5   # ResNet-18 layer4
    assert adaptive_kernel_size(768) == 5   # ViT-B hidden size
    assert adaptive_kernel_size(256) == 5
    assert adaptive_kernel_size(128) == 5
    assert adaptive_kernel_size(64) == 3
    assert adaptive_kernel_size(2048) == 7
    for c in (16, 32, 64, 128, 256, 512, 768, 1024, 2048):
        assert adaptive_kernel_size(c) % 2 == 1


def test_eca2d_shapes_and_gating():
    blk = ECABlock2d(512, None)
    assert blk.k_size == 5
    x = torch.randn(2, 512, 7, 7)
    y = blk(x)
    assert y.shape == x.shape
    # output is the input rescaled by a per-channel factor in (0, 1)
    ratio = (y / x).mean(dim=(2, 3))
    assert torch.all(ratio > 0) and torch.all(ratio < 1)


def test_eca1d_matches_eca2d_on_flattened_tokens():
    torch.manual_seed(0)
    b2 = ECABlock2d(32, 3)
    b1 = ECABlock1d(32, 3)
    b1.conv.weight.data.copy_(b2.conv.weight.data)
    x = torch.randn(2, 32, 4, 4)
    y2 = b2(x).flatten(2).transpose(1, 2)
    y1 = b1(x.flatten(2).transpose(1, 2))
    assert torch.allclose(y1, y2, atol=1e-6)


def test_fixed_kernel_override():
    assert ECABlock2d(512, 3).k_size == 3
