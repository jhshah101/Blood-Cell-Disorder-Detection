#!/usr/bin/env python
"""Train one model for one seed.

    python scripts/train.py --config configs/default.yaml --seed 0
    python scripts/train.py --config configs/ablation/vit_eca_cf.yaml --set train.lr=5e-5 --set data.num_workers=0

Outputs (under <output_dir>/<experiment>/seed<seed>/):
    config.yaml            resolved configuration
    history.csv            per-epoch losses, lr, validation metrics, ALR weights
    best.pt / last.pt      checkpoints (model + classes + config)
    val_predictions.csv    per-image validation predictions
    test_predictions.csv   per-image test predictions (test evaluated once)
    results.json           everything, including bootstrap CIs
"""
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import _bootstrap  # noqa: F401

from wbc.config import Config, load_config
from wbc.data import build_datasets, build_loaders
from wbc.engine import Trainer
from wbc.losses import build_criterion
from wbc.models import build_model
from wbc.seed import set_seed
from wbc.utils import get_device, setup_logging


def run(cfg: Config, output_dir: Optional[Path] = None) -> Dict:
    set_seed(cfg.train.seed, cfg.train.deterministic)
    device = get_device(cfg.train.device)
    out = Path(output_dir) if output_dir else Path(cfg.output_dir) / cfg.experiment / f"seed{cfg.train.seed}"
    logger = setup_logging(out)

    bundle = build_datasets(cfg)
    logger.info("classes %s | train counts %s | val counts %s | test counts %s",
                bundle.class_names, bundle.train_counts, bundle.val_counts, bundle.test_counts)
    train_loader, val_loader, test_loader = build_loaders(cfg, bundle, cfg.train.seed, device)

    model = build_model(cfg, len(bundle.class_names))
    criterion = build_criterion(cfg.loss, bundle.train_counts, device)
    logger.info("model %s | loss %s | initial class weights %s", cfg.model.backbone, cfg.loss.name,
                getattr(criterion, "weights", getattr(criterion, "weight", None)))

    trainer = Trainer(
        cfg, model, criterion, train_loader, val_loader, test_loader, bundle.class_names, device, out,
        counts={"train": bundle.train_counts, "val": bundle.val_counts, "test": bundle.test_counts},
        split_info=bundle.split_info,
    )
    return trainer.fit()


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="section.key=value")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--output-dir")
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    if args.seed is not None:
        cfg.train.seed = args.seed
    run(cfg, Path(args.output_dir) if args.output_dir else None)


if __name__ == "__main__":
    main()
