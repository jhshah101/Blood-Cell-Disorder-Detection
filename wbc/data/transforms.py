"""Input pipelines.

The transforms produce a ``[0, 1]`` float tensor and nothing else: input
normalisation (ImageNet statistics or image-wise standardisation) is part of
the model, so it is impossible for training, evaluation and the inference
server to use different statistics.

``augment="none"`` is the augmentation-free protocol of the manuscript.
``augment="basic"`` is the fully specified pipeline used for the augmented /
non-augmented comparison (Table 10).  It is applied to the training split only,
in the order listed in :data:`AUGMENTATION_SPEC`, and every parameter is
recorded in the results file so the experiment is reproducible.
"""
from __future__ import annotations

from typing import Dict, List

from torchvision import transforms

AUGMENTATION_SPEC: Dict[str, List[Dict]] = {
    "none": [],
    "basic": [
        {"op": "RandomHorizontalFlip", "p": 0.5},
        {"op": "RandomVerticalFlip", "p": 0.5},
        {"op": "RandomRotation", "degrees": 15, "interpolation": "bilinear", "fill": 0},
    ],
}


def build_transforms(img_size: int, augment: str = "none", train: bool = False) -> transforms.Compose:
    ops: List = [transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR)]
    if train and augment == "basic":
        ops += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(15, interpolation=transforms.InterpolationMode.BILINEAR, fill=0),
        ]
    elif augment not in ("none", "basic"):
        raise ValueError(f"Unknown augmentation policy {augment!r}")
    ops.append(transforms.ToTensor())  # -> float32 in [0, 1], colour space untouched
    return transforms.Compose(ops)
