#!/usr/bin/env python
"""Create the auditable train / validation split of the predefined training set.

    python scripts/make_splits.py --config configs/default.yaml
    python scripts/make_splits.py --train-dir data/Raabin-WBC/Train --out splits/raabin_train_val.json \
        --val-fraction 0.15 --seed 42 [--group-file groups.csv]

The resulting JSON lists every image with its class and is meant to be
committed, so that every experiment and every training seed uses the same
partition and the test set is never consulted for model selection.
"""
import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from torchvision import datasets

from wbc.config import load_config
from wbc.data.splits import make_split, read_group_file, save_split


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="YAML config; its data section provides the defaults")
    ap.add_argument("--train-dir")
    ap.add_argument("--out")
    ap.add_argument("--val-fraction", type=float)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--group-file", help="CSV with path,group columns for group-aware splitting")
    ap.add_argument("--force", action="store_true", help="overwrite an existing split file")
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    train_dir = args.train_dir or cfg.data.train_dir
    out = Path(args.out or cfg.data.split_file)
    val_fraction = args.val_fraction if args.val_fraction is not None else cfg.data.val_fraction
    seed = args.seed if args.seed is not None else cfg.data.split_seed
    group_file = args.group_file or cfg.data.group_file

    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists; pass --force to overwrite (this changes the partition for every experiment)")
    folder = datasets.ImageFolder(train_dir)
    groups = read_group_file(group_file) if group_file else None
    split = make_split(folder.samples, folder.classes, val_fraction, seed, root=train_dir, groups=groups)
    save_split(split, out)
    print(f"Wrote {out}")
    print(f"  method        : {split['method']}")
    print(f"  seed          : {split['seed']}  val_fraction: {split['val_fraction']}")
    print(f"  train counts  : {split['train_counts']}")
    print(f"  val counts    : {split['val_counts']}")


if __name__ == "__main__":
    main()
