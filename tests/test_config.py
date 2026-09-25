import pytest

from wbc.config import Config, apply_override, config_from_dict, load_config, to_dict


def test_defaults_are_valid():
    cfg = load_config()
    assert cfg.model.backbone == "cnn_hybrid"
    assert cfg.loss.name == "alr"
    assert cfg.train.selection_metric == "macro_f1"


def test_override_parses_yaml_scalars():
    cfg = load_config()
    apply_override(cfg, "train.lr=5e-5")
    apply_override(cfg, "model.use_eca=false")
    apply_override(cfg, "model.eca_k=null")
    apply_override(cfg, "data.minority_classes=[Basophil]")
    assert cfg.train.lr == 5e-5
    assert cfg.model.use_eca is False
    assert cfg.model.eca_k is None
    assert cfg.data.minority_classes == ["Basophil"]


def test_unknown_key_is_rejected():
    with pytest.raises(KeyError):
        config_from_dict({"model": {"backbone": "cnn_hybrid", "bogus": 1}})
    cfg = load_config()
    with pytest.raises(KeyError):
        apply_override(cfg, "train.learning_rate=1")


def test_validation_catches_inconsistent_geometry():
    with pytest.raises(ValueError):
        config_from_dict({"model": {"embed_dim": 515, "num_heads": 12}})
    with pytest.raises(ValueError):
        config_from_dict({"model": {"eca_k": 4}})
    with pytest.raises(ValueError):
        config_from_dict({"loss": {"name": "dice"}})


def test_roundtrip():
    cfg = load_config(overrides=["experiment=x", "train.epochs=3"])
    again = config_from_dict(to_dict(cfg))
    assert again == cfg
