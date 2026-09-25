"""Typed experiment configuration.

Every hyper-parameter that the manuscript reports (optimiser, learning rate,
epochs, batch size, ECA kernel rule, transformer geometry, loss family and the
ALR hyper-parameters alpha / beta / gamma) is a named field here.  The resolved
configuration is serialised next to every checkpoint and every results file and
can be overridden from the command line with ``--set section.key=value``.  There
is therefore one source of truth for *what was run*, which is the property the
reviewers found missing between the manuscript, the README and the scripts.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

CNN_BASELINES = (
    "resnet18",
    "resnet50",
    "densenet121",
    "mobilenet_v2",
    "googlenet",
    "efficientnet_b1",
    "squeezenet1_0",
)
HYBRID_BACKBONES = ("cnn_hybrid", "vit_b16")


@dataclass
class DataConfig:
    """Where the data lives and how it is partitioned and pre-processed."""

    train_dir: str = "data/Raabin-WBC/Train"
    test_dir: str = "data/Raabin-WBC/TestA"
    # The validation split is carved out of the *predefined training set* once,
    # with ``split_seed``, and re-used unchanged by every experiment and every
    # training seed.  The test directory is never read before the final
    # evaluation.
    split_file: str = "splits/raabin_train_val.json"
    split_seed: int = 42
    val_fraction: float = 0.15
    # Optional CSV with columns ``path,group`` (slide / patient / acquisition
    # identifier).  When present the split is group-aware so that images from
    # the same group never straddle the train / validation boundary.
    group_file: Optional[str] = None
    img_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    # ``imagenet``  : fixed channel mean / std (what the public code did)
    # ``per_image`` : image-wise standardisation, manuscript Eq. (1)
    # ``none``      : leave the [0, 1] tensor untouched
    normalization: str = "imagenet"
    # ``none`` is the augmentation-free protocol of the paper.  ``basic`` is the
    # fully specified flip / rotate pipeline used for the Table 10 comparison and
    # is applied to the training split only.
    augment: str = "none"
    # ``shuffle`` or ``weighted`` (WeightedRandomSampler, inverse frequency).
    sampler: str = "shuffle"
    minority_classes: List[str] = field(
        default_factory=lambda: ["Basophil", "Eosinophil", "Monocyte"]
    )


@dataclass
class ModelConfig:
    """Architecture.  ``backbone`` selects one unambiguous computational graph."""

    # cnn_hybrid : ResNet-18 -> ECA -> (+ colour features) -> Transformer encoder
    # vit_b16    : ViT-B/16 patch embedding -> ECA -> (+ colour) -> ViT-B encoder
    # <cnn name> : plain torchvision CNN baseline (see CNN_BASELINES)
    backbone: str = "cnn_hybrid"
    pretrained: bool = True
    # Transformer geometry (used by ``cnn_hybrid``; ``vit_b16`` is fixed to the
    # ViT-B geometry 768 / 12 / 12 / 3072 by construction).
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    drop_path: float = 0.1
    # Efficient Channel Attention.  ``eca_k=None`` applies the adaptive rule of
    # ECA-Net, k = |log2(C)/gamma + b/gamma|_odd, which gives k = 5 for both
    # C = 512 (ResNet-18) and C = 768 (ViT-B).
    use_eca: bool = True
    eca_gamma: int = 2
    eca_b: int = 1
    eca_k: Optional[int] = None
    # Colour-feature (CF) branch: per-channel mean / std / skewness / kurtosis
    # and a ``color_bins``-bin histogram over the RGB, HSV and Lab channels,
    # projected by one linear layer to ``color_dim`` values that are
    # concatenated to every structural token (512 + 3 = 515 in the manuscript).
    use_color_features: bool = True
    color_dim: int = 3
    color_bins: int = 8
    color_spaces: List[str] = field(default_factory=lambda: ["rgb", "hsv", "lab"])
    # ``patch`` computes the descriptors on the image region that corresponds to
    # each token; ``image`` computes them once per image and broadcasts.
    color_granularity: str = "patch"


@dataclass
class LossConfig:
    """Loss family and its hyper-parameters (all reported in results files)."""

    # ce | wce | focal | alr
    name: str = "alr"
    # Eq. (13): w_c = (N_max / N_c) ** alpha, then rescaled to mean one.
    alpha: float = 1.0
    focal_gamma: float = 2.0
    # Eq. (14) multiplicative factors.
    alr_beta: float = 0.5
    alr_gamma: float = 0.5
    # Corrections that make Eq. (14) a *relative* reweighting: after the
    # multiplicative step the weights are rescaled to mean one
    # (``alr_normalize="mean_one"``) and clipped to [alr_w_min, alr_w_max].
    # ``alr_normalize="none"`` reproduces the monotone behaviour the reviewer
    # described and exists only for illustration.
    alr_normalize: str = "mean_one"
    alr_w_min: float = 0.2
    alr_w_max: float = 5.0
    # Exponential smoothing of successive weight vectors (0 disables).
    alr_momentum: float = 0.0
    label_smoothing: float = 0.0


@dataclass
class TrainConfig:
    epochs: int = 20
    optimizer: str = "adam"  # adam | adamw | sgd
    lr: float = 1e-4
    weight_decay: float = 0.0
    scheduler: str = "cosine"  # none | cosine
    warmup_epochs: int = 1
    grad_clip: float = 1.0
    amp: bool = True
    seed: int = 0
    deterministic: bool = True
    # Model selection uses the VALIDATION split only.  ``macro_f1`` (higher is
    # better) or ``val_loss`` (lower is better).
    selection_metric: str = "macro_f1"
    early_stopping_patience: int = 0  # 0 disables
    device: str = "auto"
    log_interval: int = 50


@dataclass
class Config:
    experiment: str = "vit_eca_cf_alr"
    output_dir: str = "runs"
    notes: str = ""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


# --------------------------------------------------------------------------- #
# (De)serialisation helpers
# --------------------------------------------------------------------------- #
_SECTION_TYPES = {"data": DataConfig, "model": ModelConfig, "loss": LossConfig, "train": TrainConfig}


def _coerce(value: Any, reference: Any, name: str) -> Any:
    """Coerce ``value`` to the type of ``reference`` (the field default).

    PyYAML follows YAML 1.1 and reads ``5e-5`` as a *string*; without this
    step a command-line override such as ``train.lr=5e-5`` would silently
    set a text value.  ``None`` references (Optional fields) and containers
    are left untouched.
    """
    if reference is None or value is None:
        return value
    if isinstance(reference, bool):
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in ("true", "false", "yes", "no", "on", "off"):
            return value.strip().lower() in ("true", "yes", "on")
        raise TypeError(f"{name} expects a boolean, got {value!r}")
    if isinstance(reference, int):
        if isinstance(value, bool):
            raise TypeError(f"{name} expects an integer, got a boolean")
        if isinstance(value, (int, float, str)):
            f = float(value)
            if f != int(f):
                raise TypeError(f"{name} expects an integer, got {value!r}")
            return int(f)
    if isinstance(reference, float):
        if isinstance(value, (int, float, str)) and not isinstance(value, bool):
            return float(value)
        raise TypeError(f"{name} expects a number, got {value!r}")
    if isinstance(reference, str):
        return str(value)
    return value


def _from_dict(cls, data: Optional[Dict[str, Any]]):
    if data is None:
        return cls()
    if not isinstance(data, dict):
        raise TypeError(f"Expected a mapping for {cls.__name__}, got {type(data).__name__}")
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise KeyError(f"Unknown key(s) for {cls.__name__}: {sorted(unknown)}")
    defaults = cls()
    kwargs: Dict[str, Any] = {}
    for name in known:
        if name not in data:
            continue
        value = data[name]
        if cls is Config and name in _SECTION_TYPES:
            kwargs[name] = _from_dict(_SECTION_TYPES[name], value)
        else:
            kwargs[name] = _coerce(value, getattr(defaults, name), f"{cls.__name__}.{name}")
    return cls(**kwargs)


def to_dict(cfg: Config) -> Dict[str, Any]:
    return dataclasses.asdict(cfg)


def _parse_scalar(text: str) -> Any:
    """Parse a CLI override value with YAML semantics (1e-4, true, null, [a,b])."""
    return yaml.safe_load(text)


def apply_override(cfg: Config, assignment: str) -> None:
    """Apply ``section.key=value`` (or ``key=value`` for top-level fields)."""
    if "=" not in assignment:
        raise ValueError(f"Override must look like section.key=value, got {assignment!r}")
    dotted, raw = assignment.split("=", 1)
    parts = dotted.strip().split(".")
    target: Any = cfg
    for part in parts[:-1]:
        if not hasattr(target, part):
            raise KeyError(f"Unknown config section {part!r} in {assignment!r}")
        target = getattr(target, part)
    leaf = parts[-1]
    if not hasattr(target, leaf):
        raise KeyError(f"Unknown config key {leaf!r} in {assignment!r}")
    reference = getattr(type(target)(), leaf) if is_dataclass(target) else getattr(target, leaf)
    setattr(target, leaf, _coerce(_parse_scalar(raw), reference, dotted.strip()))


def load_config(path: Optional[str] = None, overrides: Optional[List[str]] = None) -> Config:
    data: Dict[str, Any] = {}
    if path is not None:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    cfg = _from_dict(Config, data)
    for item in overrides or []:
        apply_override(cfg, item)
    validate(cfg)
    return cfg


def config_from_dict(data: Dict[str, Any]) -> Config:
    cfg = _from_dict(Config, data)
    validate(cfg)
    return cfg


def save_config(cfg: Config, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(to_dict(cfg), fh, sort_keys=False)


def is_cnn_baseline(cfg: Config) -> bool:
    return cfg.model.backbone in CNN_BASELINES


def validate(cfg: Config) -> None:
    m, d, l, t = cfg.model, cfg.data, cfg.loss, cfg.train
    if m.backbone not in HYBRID_BACKBONES + CNN_BASELINES:
        raise ValueError(
            f"model.backbone must be one of {HYBRID_BACKBONES + CNN_BASELINES}, got {m.backbone!r}"
        )
    if m.backbone == "cnn_hybrid" and m.embed_dim % m.num_heads != 0:
        raise ValueError("model.embed_dim must be divisible by model.num_heads")
    if m.eca_k is not None and (m.eca_k < 1 or m.eca_k % 2 == 0):
        raise ValueError("model.eca_k must be a positive odd integer or null (adaptive)")
    if d.normalization not in ("imagenet", "per_image", "none"):
        raise ValueError("data.normalization must be imagenet | per_image | none")
    if d.augment not in ("none", "basic"):
        raise ValueError("data.augment must be none | basic")
    if d.sampler not in ("shuffle", "weighted"):
        raise ValueError("data.sampler must be shuffle | weighted")
    if not 0.0 < d.val_fraction < 0.5:
        raise ValueError("data.val_fraction must lie in (0, 0.5)")
    if l.name not in ("ce", "wce", "focal", "alr"):
        raise ValueError("loss.name must be ce | wce | focal | alr")
    if l.alr_normalize not in ("mean_one", "none"):
        raise ValueError("loss.alr_normalize must be mean_one | none")
    if l.alr_w_min <= 0 or l.alr_w_max < l.alr_w_min:
        raise ValueError("loss.alr_w_min must be > 0 and <= loss.alr_w_max")
    if t.optimizer not in ("adam", "adamw", "sgd"):
        raise ValueError("train.optimizer must be adam | adamw | sgd")
    if t.scheduler not in ("none", "cosine"):
        raise ValueError("train.scheduler must be none | cosine")
    if t.selection_metric not in ("macro_f1", "val_loss", "balanced_accuracy", "accuracy"):
        raise ValueError(
            "train.selection_metric must be macro_f1 | val_loss | balanced_accuracy | accuracy"
        )
    if m.color_granularity not in ("patch", "image"):
        raise ValueError("model.color_granularity must be patch | image")
    for space in m.color_spaces:
        if space not in ("rgb", "hsv", "lab"):
            raise ValueError(f"Unknown colour space {space!r}")
