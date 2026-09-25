#!/usr/bin/env python
"""Grad-CAM panel with a recorded selection protocol.

    python scripts/gradcam.py --checkpoint runs/vit_eca_cf_alr/seed0/best.pt \
        --predictions runs/vit_eca_cf_alr/seed0/test_predictions.csv --per-class 2 --seed 0 --only-correct

Writes <output-dir>/<class>_<k>.png overlays and manifest.json listing, for
every panel image, its path, true class, predicted class, correctness and the
normalisation used - i.e. everything a figure caption must state.
"""
import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import torch
from PIL import Image

from wbc.checkpoint import load_checkpoint
from wbc.data import build_transforms
from wbc.explain import GradCAM, overlay, select_examples
from wbc.utils import get_device, save_json


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--predictions", required=True, help="test_predictions.csv written by train.py")
    ap.add_argument("--per-class", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only-correct", action="store_true")
    ap.add_argument("--shared-scale", action="store_true", help="normalise all maps with one global min/max")
    ap.add_argument("--output-dir", default="gradcam")
    args = ap.parse_args()

    device = get_device()
    model, cfg, classes, _ = load_checkpoint(args.checkpoint, device)
    tf = build_transforms(cfg.data.img_size)
    chosen = select_examples(args.predictions, args.per_class, args.seed, classes, True if args.only_correct else None)
    cam = GradCAM(model)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"protocol": {"target_class": "ground truth", "normalisation": "shared" if args.shared_scale else "per map",
                             "selection": f"{args.per_class} per class, seed {args.seed}, only_correct={args.only_correct}"}, "images": []}
    counters = {c: 0 for c in classes}
    for item in chosen:
        img = Image.open(item["path"]).convert("RGB")
        x = tf(img).unsqueeze(0).to(device)
        heat = cam(x, [classes.index(item["class"])], per_map=not args.shared_scale)[0]
        counters[item["class"]] += 1
        name = f"{item['class']}_{counters[item['class']]}.png"
        overlay(img.resize((cfg.data.img_size, cfg.data.img_size)), heat).save(out / name)
        manifest["images"].append({**item, "file": name})
    cam.remove()
    save_json(manifest, out / "manifest.json")
    print(f"wrote {len(manifest['images'])} overlays to {out}")


if __name__ == "__main__":
    main()
