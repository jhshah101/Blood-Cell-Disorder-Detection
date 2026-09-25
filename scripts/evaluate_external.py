#!/usr/bin/env python
"""Zero-shot evaluation on an external set with an explicit class mapping.

    # Raabin-WBC Test-B (neutrophils + lymphocytes only)
    python scripts/evaluate_external.py --checkpoint runs/.../best.pt --data-dir data/Raabin-WBC/TestB --preset raabin_testb

    # LISC, cropping each cell to its expert mask
    python scripts/evaluate_external.py --checkpoint runs/.../best.pt --data-dir data/LISC/Main_Dataset \
        --preset lisc --mask-dir data/LISC/Ground_Truth --crop-margin 0.1

No fine-tuning is performed.  The output JSON records the folder-to-class
mapping, the excluded folders, the number of evaluable images per class, the
unrestricted 5-way metrics and (when only a subset of classes is present) the
restricted-arg-max metrics.
"""
import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import torch
from torch.utils.data import DataLoader

from wbc.checkpoint import load_checkpoint
from wbc.data import DEFAULT_MAPPINGS, MappedImageFolder, build_transforms, parse_mapping, restricted_argmax
from wbc.metrics import accuracy_statistic, bootstrap_ci, compute_metrics, macro_f1_statistic
from wbc.utils import get_device, save_json


@torch.no_grad()
def predict(model, loader, device):
    ys, ps = [], []
    for x, y, _ in loader:
        logits = model(x.to(device)).float()
        ys.append(torch.as_tensor(y))
        ps.append(logits.softmax(1).cpu())
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--preset", choices=sorted(DEFAULT_MAPPINGS))
    ap.add_argument("--mapping", help='explicit "folder=TrainClass,..." mapping (overrides --preset)')
    ap.add_argument("--mask-dir", help="directory of binary masks; images are cropped to the mask bounding box")
    ap.add_argument("--mask-required", action="store_true")
    ap.add_argument("--crop-margin", type=float, default=0.10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--output")
    args = ap.parse_args()
    if not (args.preset or args.mapping):
        ap.error("--preset or --mapping is required")

    device = get_device()
    model, cfg, classes, _ = load_checkpoint(args.checkpoint, device)
    mapping = parse_mapping(args.mapping) if args.mapping else DEFAULT_MAPPINGS[args.preset]
    ds = MappedImageFolder(args.data_dir, classes, mapping, transform=build_transforms(cfg.data.img_size),
                           mask_dir=args.mask_dir, mask_required=args.mask_required, crop_margin=args.crop_margin)
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers)
    y_true, probs = predict(model, loader, device)
    present = ds.present_classes
    minority = [c for c in cfg.data.minority_classes if classes.index(c) in present]

    y_pred_full = probs.argmax(1)
    result = {
        "checkpoint": args.checkpoint,
        "fine_tuned": False,
        "dataset": ds.summary(),
        "images_without_mask": ds.no_mask,
        "unrestricted_5way": compute_metrics(y_true, y_pred_full, classes, minority, probs),
        "unrestricted_ci": {
            "accuracy": bootstrap_ci(y_true, y_pred_full, accuracy_statistic),
            "macro_f1_over_all_classes": bootstrap_ci(y_true, y_pred_full, macro_f1_statistic(len(classes))),
        },
    }
    # Macro metrics over the *present* classes only (absent classes have zero support and zero F1).
    pc = result["unrestricted_5way"]["per_class"]
    result["unrestricted_macro_f1_present_classes"] = float(np.mean([pc[classes[c]]["f1"] for c in present]))
    if len(present) < len(classes):
        y_pred_r = restricted_argmax(probs, present)
        result["restricted_argmax"] = compute_metrics(y_true, y_pred_r, classes, minority, None)
        result["restricted_macro_f1_present_classes"] = float(np.mean([result["restricted_argmax"]["per_class"][classes[c]]["f1"] for c in present]))
        result["restricted_ci"] = {"accuracy": bootstrap_ci(y_true, y_pred_r, accuracy_statistic)}
        result["predicted_as_absent_class"] = int(np.sum(~np.isin(y_pred_full, present)))

    path = Path(args.output) if args.output else Path(args.checkpoint).with_name("external_" + Path(args.data_dir).name + ".json")
    save_json(result, path)
    u = result["unrestricted_5way"]
    print(f"{ds.summary()['n_images']} images | present classes {ds.summary()['present_classes']} | excluded {ds.excluded}")
    print(f"unrestricted: acc {u['accuracy']:.4f} | macro-F1 (present classes) {result['unrestricted_macro_f1_present_classes']:.4f}")
    if "restricted_argmax" in result:
        r = result["restricted_argmax"]
        print(f"restricted  : acc {r['accuracy']:.4f} | macro-F1 (present classes) {result['restricted_macro_f1_present_classes']:.4f} | predicted as absent class: {result['predicted_as_absent_class']}")
    for c in present:
        m = u["per_class"][classes[c]]
        print(f"  {classes[c]:12s} P {m['precision']:.4f} R {m['recall']:.4f} F1 {m['f1']:.4f} n={m['support']}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
