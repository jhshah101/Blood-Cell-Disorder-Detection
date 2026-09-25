"""Grad-CAM with an explicit, reportable protocol.

Grad-CAM (Selvaraju et al.) is a *localisation visualisation*; it does not by
itself establish clinical relevance or faithfulness.  What a manuscript can
legitimately report is the protocol under which the maps were produced, and
this module fixes that protocol:

* target layer   : the last convolutional feature map before tokenisation
                   (``model.eca`` output for ``cnn_hybrid``; the patch
                   embedding for ``vit_b16``; ``layer4`` / ``features`` for
                   the CNN baselines);
* target class   : the *ground-truth* class unless stated otherwise;
* example choice : ``select_examples`` draws a fixed number of images per class
                   from the test predictions with a seed, and records for each
                   whether the model classified it correctly;
* normalisation  : every map is min-max normalised to [0, 1] on its own
                   (``per_map=True``) - the alternative, a shared scale, is
                   also available and the choice is written into the output.
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


def default_target_module(model: nn.Module) -> nn.Module:
    name = getattr(model, "backbone_name", "")
    if name == "cnn_hybrid":
        return model.eca if not isinstance(model.eca, nn.Identity) else model.cnn.layer4
    if name == "vit_b16":
        return model.conv_proj
    net = getattr(model, "net", model)
    if hasattr(net, "layer4"):
        return net.layer4
    if hasattr(net, "features"):
        return net.features
    raise ValueError("Could not infer a Grad-CAM target layer; pass one explicitly")


class GradCAM:
    def __init__(self, model: nn.Module, target: Optional[nn.Module] = None):
        self.model = model
        self.target = target or default_target_module(model)
        self._act: Optional[torch.Tensor] = None
        self._grad: Optional[torch.Tensor] = None
        self._handles = [
            self.target.register_forward_hook(self._save_act),
            self.target.register_full_backward_hook(self._save_grad),
        ]

    def _save_act(self, _m, _i, out):
        self._act = out if isinstance(out, torch.Tensor) else out[0]

    def _save_grad(self, _m, _gi, go):
        self._grad = go[0]

    def remove(self) -> None:
        for h in self._handles:
            h.remove()

    def __call__(self, x: torch.Tensor, class_idx: Optional[Sequence[int]] = None, per_map: bool = True) -> np.ndarray:
        """``x`` is ``(B, 3, H, W)`` in [0, 1]; returns ``(B, H, W)`` maps in [0, 1]."""
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        x = x.requires_grad_(True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(1).tolist()
        score = logits.gather(1, torch.as_tensor(class_idx, device=logits.device)[:, None]).sum()
        score.backward()
        act, grad = self._act, self._grad
        if act.ndim == 3:  # (B, N, C) tokens -> (B, C, g, g)
            b, n, c = act.shape
            g = int(round(n**0.5))
            act = act.transpose(1, 2).reshape(b, c, g, g)
            grad = grad.transpose(1, 2).reshape(b, c, g, g)
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * act).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        cam = cam.detach().cpu().numpy()
        if per_map:
            for i in range(cam.shape[0]):
                lo, hi = cam[i].min(), cam[i].max()
                cam[i] = (cam[i] - lo) / (hi - lo + 1e-8)
        else:
            lo, hi = cam.min(), cam.max()
            cam = (cam - lo) / (hi - lo + 1e-8)
        return cam


def select_examples(predictions_csv: str | Path, per_class: int, seed: int, class_names: Sequence[str], only_correct: Optional[bool] = None) -> List[Dict]:
    """Seeded, documented choice of test images for the Grad-CAM panel."""
    rows: List[Dict] = []
    with open(predictions_csv, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            r["path"] = json.loads(r["path"]) if r["path"].startswith('"') else r["path"]
            r["y_true"], r["y_pred"], r["correct"] = int(r["y_true"]), int(r["y_pred"]), bool(int(r["correct"]))
            rows.append(r)
    rng = random.Random(seed)
    chosen: List[Dict] = []
    for c, name in enumerate(class_names):
        pool = [r for r in rows if r["y_true"] == c and (only_correct is None or r["correct"] == only_correct)]
        rng.shuffle(pool)
        for r in pool[:per_class]:
            chosen.append({"class": name, "path": r["path"], "correct": r["correct"], "predicted": class_names[r["y_pred"]]})
    return chosen


def overlay(image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Blend a [0, 1] heat map (JET-like ramp) over an RGB image."""
    cam = np.clip(cam, 0, 1)
    r = np.clip(1.5 - abs(4 * cam - 3), 0, 1)
    g = np.clip(1.5 - abs(4 * cam - 2), 0, 1)
    b = np.clip(1.5 - abs(4 * cam - 1), 0, 1)
    heat = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)
    heat_img = Image.fromarray(heat).resize(image.size, Image.BILINEAR)
    return Image.blend(image.convert("RGB"), heat_img, alpha)
