"""Datasets and loaders implementing the leakage-safe protocol."""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import datasets

from ..config import Config
from ..seed import make_generator, seed_worker
from .splits import load_split, make_split, read_group_file, save_split, split_indices
from .transforms import build_transforms

log = logging.getLogger("wbc")


class IndexedDataset(Dataset):
    """Wrap a dataset so each item is ``(image, label, index)``; the index lets
    the trainer write per-image predictions with their file paths."""

    def __init__(self, base: Dataset, paths: Sequence[str]):
        self.base = base
        self.paths = list(paths)
        if len(self.paths) != len(base):
            raise ValueError("paths must have one entry per dataset item")

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int):
        x, y = self.base[i]
        return x, y, i


@dataclass
class DataBundle:
    train: IndexedDataset
    val: IndexedDataset
    test: Optional[IndexedDataset]
    class_names: List[str]
    train_counts: List[int]
    val_counts: List[int]
    test_counts: Optional[List[int]]
    split_info: Dict


def _counts(targets: Sequence[int], n_classes: int) -> List[int]:
    c = Counter(int(t) for t in targets)
    return [int(c.get(i, 0)) for i in range(n_classes)]


def _check_dir(path: str, what: str) -> Path:
    p = Path(path)
    if not p.is_dir():
        raise FileNotFoundError(
            f"{what} directory {p} does not exist. Point data.{what.lower()}_dir at an ImageFolder "
            f"tree (one sub-folder per class)."
        )
    return p


def build_datasets(cfg: Config, include_test: bool = True) -> DataBundle:
    d = cfg.data
    train_root = _check_dir(d.train_dir, "Train")

    train_tf = build_transforms(d.img_size, d.augment, train=True)
    eval_tf = build_transforms(d.img_size, "none", train=False)

    # Two ImageFolder views of the same tree: augmentation for the training
    # subset, deterministic pre-processing for the validation subset.
    folder_train_view = datasets.ImageFolder(str(train_root), transform=train_tf)
    folder_eval_view = datasets.ImageFolder(str(train_root), transform=eval_tf)
    class_names = list(folder_train_view.classes)

    split_path = Path(d.split_file)
    if split_path.exists():
        split = load_split(split_path)
        if split["class_names"] != class_names:
            raise RuntimeError(
                f"Split file {split_path} was built for classes {split['class_names']} but the folder has {class_names}"
            )
        log.info("Loaded validation split %s (%s, seed %s)", split_path, split["method"], split["seed"])
    else:
        groups = read_group_file(d.group_file) if d.group_file else None
        split = make_split(folder_train_view.samples, class_names, d.val_fraction, d.split_seed, root=train_root, groups=groups)
        save_split(split, split_path)
        log.warning(
            "No split file found; created %s with %s (seed %s, val_fraction %.2f). Commit this file so every "
            "experiment shares the identical partition.",
            split_path, split["method"], d.split_seed, d.val_fraction,
        )
    train_idx, val_idx = split_indices(split, folder_train_view.samples, root=train_root)

    train_subset = Subset(folder_train_view, train_idx)
    val_subset = Subset(folder_eval_view, val_idx)
    train_paths = [folder_train_view.samples[i][0] for i in train_idx]
    val_paths = [folder_eval_view.samples[i][0] for i in val_idx]
    train_targets = [folder_train_view.samples[i][1] for i in train_idx]
    val_targets = [folder_eval_view.samples[i][1] for i in val_idx]

    test_ds = None
    test_counts = None
    if include_test:
        test_root = _check_dir(d.test_dir, "Test")
        folder_test = datasets.ImageFolder(str(test_root), transform=eval_tf)
        if list(folder_test.classes) != class_names:
            raise RuntimeError(
                f"Test classes {folder_test.classes} differ from training classes {class_names}; "
                "use scripts/evaluate_external.py with an explicit class mapping for external sets."
            )
        test_ds = IndexedDataset(folder_test, [p for p, _ in folder_test.samples])
        test_counts = _counts(folder_test.targets, len(class_names))

    return DataBundle(
        train=IndexedDataset(train_subset, train_paths),
        val=IndexedDataset(val_subset, val_paths),
        test=test_ds,
        class_names=class_names,
        train_counts=_counts(train_targets, len(class_names)),
        val_counts=_counts(val_targets, len(class_names)),
        test_counts=test_counts,
        split_info={k: v for k, v in split.items() if k not in ("train", "val")},
    )


def build_loaders(cfg: Config, bundle: DataBundle, seed: int, device: torch.device) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    d = cfg.data
    g = make_generator(seed)
    common = dict(
        num_workers=d.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=d.num_workers > 0,
        worker_init_fn=seed_worker,
    )
    if d.sampler == "weighted":
        targets = torch.tensor([bundle.train.base.dataset.samples[i][1] for i in bundle.train.base.indices])
        counts = torch.tensor(bundle.train_counts, dtype=torch.float)
        per_sample = (1.0 / counts.clamp_min(1))[targets]
        sampler = WeightedRandomSampler(per_sample.double(), num_samples=len(targets), replacement=True, generator=g)
        train_loader = DataLoader(bundle.train, batch_size=d.batch_size, sampler=sampler, drop_last=False, **common)
    else:
        train_loader = DataLoader(bundle.train, batch_size=d.batch_size, shuffle=True, generator=g, drop_last=False, **common)
    val_loader = DataLoader(bundle.val, batch_size=d.batch_size, shuffle=False, **common)
    test_loader = None
    if bundle.test is not None:
        test_loader = DataLoader(bundle.test, batch_size=d.batch_size, shuffle=False, **common)
    return train_loader, val_loader, test_loader
