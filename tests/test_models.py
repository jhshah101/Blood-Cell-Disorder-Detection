import pytest
import torch

from wbc.benchmark import count_flops, count_parameters
from wbc.config import ModelConfig, load_config
from wbc.models import build_model
from wbc.models.hybrid import HybridViTECACF, InputNormalization, resize_pos_embed


def _small_cfg(**over):
    base = dict(backbone="cnn_hybrid", pretrained=False, embed_dim=64, depth=1, num_heads=4, dropout=0.0, drop_path=0.0)
    base.update(over)
    return ModelConfig(**base)


def test_input_normalisation_modes():
    x = torch.rand(2, 3, 8, 8)
    assert torch.equal(InputNormalization("none")(x), x)
    y = InputNormalization("per_image")(x)
    assert torch.allclose(y.mean(dim=(1, 2, 3)), torch.zeros(2), atol=1e-5)
    assert torch.allclose(y.std(dim=(1, 2, 3), unbiased=False), torch.ones(2), atol=1e-3)
    z = InputNormalization("imagenet")(x)
    assert z.shape == x.shape


def test_cnn_hybrid_forward_and_token_dimensions():
    m = HybridViTECACF(5, _small_cfg(), normalization="imagenet", img_size=64)
    assert m.fused_dim == 512 + 3  # the manuscript's 515-d fused token
    assert m.eca.k_size == 5
    out = m(torch.rand(2, 3, 64, 64))
    assert out.shape == (2, 5)
    out.sum().backward()
    assert m.color.proj[1].weight.grad is not None  # colour projection is trained


def test_cnn_hybrid_without_eca_and_cf():
    m = HybridViTECACF(5, _small_cfg(use_eca=False, use_color_features=False), img_size=64)
    assert isinstance(m.eca, torch.nn.Identity) and m.color is None and m.fused_dim == 512
    assert m(torch.rand(1, 3, 64, 64)).shape == (1, 5)


def test_pos_embed_interpolation_handles_other_image_sizes():
    m = HybridViTECACF(5, _small_cfg(), img_size=64)  # pos embed built for 2x2 grid
    assert m(torch.rand(1, 3, 96, 96)).shape == (1, 5)  # 3x3 grid -> interpolated
    pos = torch.randn(1, 1 + 4, 8)
    assert resize_pos_embed(pos, 1 + 9).shape == (1, 10, 8)
    with pytest.raises(ValueError):
        resize_pos_embed(pos, 1 + 5)


def test_vit_b16_backbone_builds_and_runs():
    cfg = ModelConfig(backbone="vit_b16", pretrained=False)
    m = HybridViTECACF(5, cfg, img_size=64)
    assert m.eca.k_size == 5 and m.fused_dim == 768 + 3
    # fusion layer starts as the identity on the structural part
    assert torch.allclose(m.fuse.weight[:, :768], torch.eye(768))
    assert m(torch.rand(1, 3, 64, 64)).shape == (1, 5)


def test_cnn_baselines_share_input_pipeline():
    cfg = load_config(overrides=["model.backbone=resnet18", "model.pretrained=false", "data.normalization=per_image"])
    m = build_model(cfg, 5)
    assert m.input_norm.mode == "per_image"
    assert m(torch.rand(1, 3, 64, 64)).shape == (1, 5)


def test_benchmark_helpers():
    m = HybridViTECACF(5, _small_cfg(), img_size=64)
    p = count_parameters(m)
    assert p["total"] == p["trainable"] > 11_000_000  # ResNet-18 alone is ~11.2 M
    f = count_flops(m, (1, 3, 64, 64))
    assert f["gflops"] > 0 and abs(f["gflops"] - 2 * f["gmacs"]) < 1e-9


def test_flop_counter_sees_attention():
    """Regression: the fused MHA fast path under no_grad reports zero FLOPs."""
    mha = torch.nn.MultiheadAttention(64, 4, batch_first=True).eval()

    class Wrap(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mha = mha

        def forward(self, x):
            x = x.flatten(2).transpose(1, 2)  # (B, N, 64)
            return self.mha(x, x, x, need_weights=False)[0]

    f = count_flops(Wrap(), (1, 64, 4, 4))
    n, d = 16, 64
    expected_macs = 4 * n * d * d + 2 * n * n * d  # QKV + out projection + QK^T + AV
    assert f["gmacs"] * 1e9 >= 0.95 * expected_macs
