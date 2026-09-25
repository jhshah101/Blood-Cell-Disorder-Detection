"""Auditable train / validation partition of the predefined training set.

The split is created **once** with a fixed ``split_seed``, written to a JSON
file that lists every image path with its class, and re-used unchanged by
every experiment and every training seed.  Model selection and early stopping
read the validation split only; the test directory is opened exactly once,
after training, by the final evaluation.

Two strategies are available:

* stratified (default)  - ``StratifiedShuffleSplit`` preserves the class
                          proportions of the imbalanced training set;
* group-aware           - when a ``path,group`` CSV (slide / patient /
                          acquisition id) is supplied, ``StratifiedGroupKFold``
                          keeps all images of a group on one side of the
                          boundary.  Raabin-WBC does not publish such
                          identifiers; the hook exists so that the protocol
                          can be upgraded the moment they are available.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit


def read_group_file(path: str | Path) -> Dict[str, str]:
    groups: Dict[str, str] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if "path" not in reader.fieldnames or "group" not in reader.fieldnames:
            raise ValueError("group file must have 'path' and 'group' columns")
        for row in reader:
            groups[Path(row["path"]).as_posix()] = str(row["group"])
    return groups


def make_split(
    samples: Sequence[Tuple[str, int]],
    class_names: Sequence[str],
    val_fraction: float,
    seed: int,
    root: Optional[str | Path] = None,
    groups: Optional[Dict[str, str]] = None,
) -> Dict:
    """Return a JSON-serialisable description of the train / validation split."""
    if not 0 < val_fraction < 0.5:
        raise ValueError("val_fraction must lie in (0, 0.5)")
    paths = [Path(p) for p, _ in samples]
    labels = np.asarray([y for _, y in samples], dtype=int)
    if root is not None:
        root = Path(root)
        rel = [p.relative_to(root).as_posix() if p.is_absolute() or root in p.parents else p.as_posix() for p in paths]
    else:
        rel = [p.as_posix() for p in paths]

    if groups:
        g = np.asarray([groups.get(r, groups.get(str(p), r)) for r, p in zip(rel, paths)])
        n_splits = max(2, int(round(1.0 / val_fraction)))
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        train_idx, val_idx = next(splitter.split(np.zeros(len(labels)), labels, g))
        method = f"StratifiedGroupKFold(n_splits={n_splits}, first fold as validation)"
        overlap = set(g[train_idx]) & set(g[val_idx])
        if overlap:
            raise RuntimeError(f"Group leakage detected for groups {sorted(overlap)[:5]}")
    else:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
        train_idx, val_idx = next(splitter.split(np.zeros(len(labels)), labels))
        method = "StratifiedShuffleSplit"

    train_idx = sorted(int(i) for i in train_idx)
    val_idx = sorted(int(i) for i in val_idx)

    def _counts(idx: List[int]) -> Dict[str, int]:
        c = Counter(int(labels[i]) for i in idx)
        return {class_names[k]: int(c.get(k, 0)) for k in range(len(class_names))}

    return {
        "method": method,
        "seed": int(seed),
        "val_fraction": float(val_fraction),
        "group_aware": bool(groups),
        "class_names": list(class_names),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "train_counts": _counts(train_idx),
        "val_counts": _counts(val_idx),
        "train": [{"path": rel[i], "label": int(labels[i])} for i in train_idx],
        "val": [{"path": rel[i], "label": int(labels[i])} for i in val_idx],
    }


def save_split(split: Dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(split, fh, indent=1)


def load_split(path: str | Path) -> Dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def split_indices(split: Dict, samples: Sequence[Tuple[str, int]], root: Optional[str | Path] = None) -> Tuple[List[int], List[int]]:
    """Map the stored relative paths back onto the indices of ``samples``."""
    root = Path(root) if root is not None else None
    lookup: Dict[str, int] = {}
    for i, (p, _) in enumerate(samples):
        p = Path(p)
        key = p.relative_to(root).as_posix() if root is not None and root in p.parents else p.as_posix()
        lookup[key] = i
    missing: List[str] = []

    def _resolve(entries):
        out = []
        for e in entries:
            i = lookup.get(e["path"])
            if i is None:
                missing.append(e["path"])
            else:
                if samples[i][1] != e["label"]:
                    raise RuntimeError(f"Label mismatch for {e['path']}: split says {e['label']}, folder says {samples[i][1]}")
                out.append(i)
        return out

    train_idx = _resolve(split["train"])
    val_idx = _resolve(split["val"])
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} image(s) listed in the split file are not present in the training folder, "
            f"e.g. {missing[:3]}. Re-create the split with scripts/make_splits.py or fix the data path."
        )
    if set(train_idx) & set(val_idx):
        raise RuntimeError("Train and validation index sets overlap")
    return train_idx, val_idx
