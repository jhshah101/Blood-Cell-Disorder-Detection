"""Checkpoint I/O: a checkpoint carries its own configuration and class list,
so any consumer (evaluation scripts, the inference server) rebuilds exactly
the network that was trained, with the same input normalisation."""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn

from .config import Config, config_from_dict
from .models import build_model


def load_checkpoint(path: str | Path, device: torch.device) -> Tuple[nn.Module, Config, List[str], dict]:
    payload = torch.load(str(path), map_location=device, weights_only=False)
    if "config" not in payload or "classes" not in payload:
        raise ValueError(
            f"{path} is not a checkpoint written by this repository (missing 'config' / 'classes'); "
            "legacy state_dict-only files cannot be rebuilt unambiguously."
        )
    cfg = config_from_dict(payload["config"])
    classes = list(payload["classes"])
    model = build_model(cfg, len(classes))
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    return model, cfg, classes, payload
