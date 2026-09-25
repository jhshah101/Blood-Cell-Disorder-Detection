#!/usr/bin/env python
"""Evaluate a checkpoint on an ImageFolder whose classes match the training classes.

    python scripts/evaluate.py --checkpoint runs/vit_eca_cf_alr/seed0/best.pt --data-dir data/Raabin-WBC/TestA
"""
import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from wbc.checkpoint import load_checkpoint
from wbc.data import IndexedDataset, build_transforms
from wbc.metrics import accuracy_statistic, bootstrap_ci, compute_metrics, macro_f1_statistic, verify_aggregation_identities
from wbc.utils import get_device, save_json


@torch.no_grad()
def predict(model, loader, device):
    ys, ps = [], []
    for x, y, _ in loader:
        logits = model(x.to(device)).float()
        ys.append(y)
        ps.append(logits.softmax(1).cpu())
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--output", help="JSON path (default: next to the checkpoint)")
    args = ap.parse_args()

    device = get_device()
    model, cfg, classes, _ = load_checkpoint(args.checkpoint, device)
    folder = datasets.ImageFolder(args.data_dir, transform=build_transforms(cfg.data.img_size))
    if list(folder.classes) != classes:
        raise SystemExit(f"Folder classes {folder.classes} differ from training classes {classes}; use evaluate_external.py")
    loader = DataLoader(IndexedDataset(folder, [p for p, _ in folder.samples]), batch_size=args.batch_size, num_workers=args.num_workers)
    y_true, probs = predict(model, loader, device)
    y_pred = probs.argmax(1)
    metrics = compute_metrics(y_true, y_pred, classes, cfg.data.minority_classes, probs)
    problems = verify_aggregation_identities(metrics)
    out = {
        "checkpoint": args.checkpoint,
        "data_dir": args.data_dir,
        "metrics": metrics,
        "ci": {
            "macro_f1": bootstrap_ci(y_true, y_pred, macro_f1_statistic(len(classes))),
            "accuracy": bootstrap_ci(y_true, y_pred, accuracy_statistic),
        },
        "aggregation_identities_ok": not problems,
    }
    path = Path(args.output) if args.output else Path(args.checkpoint).with_name("eval_" + Path(args.data_dir).name + ".json")
    save_json(out, path)
    print(f"accuracy {metrics['accuracy']:.4f} | macro-F1 {metrics['macro_f1']:.4f} | balanced acc {metrics['balanced_accuracy']:.4f}")
    for c, m in metrics["per_class"].items():
        print(f"  {c:12s} P {m['precision']:.4f} R {m['recall']:.4f} F1 {m['f1']:.4f} n={m['support']}")
    if "minority_f1" in metrics:
        print(f"majority F1 {metrics['majority_f1']:.4f} | minority F1 {metrics['minority_f1']:.4f} | ratio {metrics['f1_balance_ratio']:.3f}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
