#!/usr/bin/env python
"""Aggregate external-evaluation JSON files (written by evaluate_external.py) over seeds.

    python scripts/aggregate_external.py --pattern "runs/ablation_vit_eca_cf_alr/seed*/external_TestB.json" \
        --title "Raabin-WBC Test-B (no fine-tuning)" --out results/external_testB.md

Reports, as mean +/- SD over the matched files: accuracy and per-class
precision / recall / F1 of the unrestricted 5-way prediction, the macro-F1
over the classes present in the external set, the number of images predicted
as an absent class, and - when the set covers a subset of the classes - the
same quantities for the restricted arg-max.  The dataset summary (mapping,
excluded folders, counts, cropping) is copied from the first file.
"""
import argparse
import glob
from pathlib import Path
from typing import Dict, List

import _bootstrap  # noqa: F401
import numpy as np

from wbc.utils import format_table, load_json


def ms(values: List[float], scale: float = 100.0) -> str:
    v = np.asarray([x for x in values if x is not None], dtype=float) * scale
    if len(v) == 0:
        return "—"
    return f"{v.mean():.2f} ± {v.std(ddof=1) if len(v) > 1 else 0.0:.2f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pattern", required=True, help="glob of external_*.json files (one per seed)")
    ap.add_argument("--title", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    files = sorted(glob.glob(args.pattern))
    if not files:
        raise SystemExit(f"No files match {args.pattern}")
    res = [load_json(f) for f in files]
    ds = res[0]["dataset"]
    classes = res[0]["unrestricted_5way"]["class_names"]
    present = ds["present_classes"]

    lines = [f"# {args.title}", ""]
    if args.note:
        lines += [args.note, ""]
    lines += [
        f"Files: {len(files)} (one per seed). Evaluated **without fine-tuning**.",
        "",
        f"* Root: `{ds['root']}`",
        f"* Folder → class mapping: {ds['folder_to_class']}",
        f"* Excluded (unmapped) folders: {ds['excluded_folders'] or 'none'}",
        f"* Evaluable images per class: {ds['per_class']} (total {ds['n_images']})",
        f"* Crop to annotation mask: {ds['crop_to_mask']} (margin {ds['crop_margin']}); images without mask: {res[0].get('images_without_mask', 0)}",
        "",
        "## Unrestricted 5-way prediction",
        "",
    ]
    rows = [{
        "Accuracy (%)": ms([r["unrestricted_5way"]["accuracy"] for r in res]),
        "Macro-F1 over present classes (%)": ms([r["unrestricted_macro_f1_present_classes"] for r in res]),
        "Balanced accuracy over present classes (%)": ms([np.mean([r["unrestricted_5way"]["per_class"][c]["recall"] for c in present]) for r in res]),
        "Predicted as absent class (n)": ms([r.get("predicted_as_absent_class", 0) for r in res], scale=1.0),
    }]
    lines.append(format_table(rows, list(rows[0].keys())))
    lines += ["", "Per class (unrestricted):", ""]
    prow = [{
        "Class": c,
        "Precision (%)": ms([r["unrestricted_5way"]["per_class"][c]["precision"] for r in res]),
        "Recall (%)": ms([r["unrestricted_5way"]["per_class"][c]["recall"] for r in res]),
        "F1 (%)": ms([r["unrestricted_5way"]["per_class"][c]["f1"] for r in res]),
        "n": res[0]["unrestricted_5way"]["per_class"][c]["support"],
    } for c in present]
    lines.append(format_table(prow, ["Class", "Precision (%)", "Recall (%)", "F1 (%)", "n"]))

    if "restricted_argmax" in res[0]:
        lines += ["", f"## Restricted arg-max over the present classes {present}", ""]
        rows = [{
            "Accuracy (%)": ms([r["restricted_argmax"]["accuracy"] for r in res]),
            "Macro-F1 over present classes (%)": ms([r["restricted_macro_f1_present_classes"] for r in res]),
        }]
        lines.append(format_table(rows, list(rows[0].keys())))
        lines += ["", "Per class (restricted):", ""]
        prow = [{
            "Class": c,
            "Precision (%)": ms([r["restricted_argmax"]["per_class"][c]["precision"] for r in res]),
            "Recall (%)": ms([r["restricted_argmax"]["per_class"][c]["recall"] for r in res]),
            "F1 (%)": ms([r["restricted_argmax"]["per_class"][c]["f1"] for r in res]),
        } for c in present]
        lines.append(format_table(prow, ["Class", "Precision (%)", "Recall (%)", "F1 (%)"]))

    cm = np.mean([np.asarray(r["unrestricted_5way"]["confusion_matrix"], dtype=float) for r in res], axis=0)
    lines += ["", "Mean confusion matrix over seeds (rows = true class, columns = predicted class, unrestricted):", ""]
    crow = [{"true \\ pred": classes[i], **{classes[j]: f"{cm[i, j]:.1f}" for j in range(len(classes))}} for i in range(len(classes)) if cm[i].sum() > 0]
    lines.append(format_table(crow, ["true \\ pred", *classes]))
    lines.append("")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out} from {len(files)} files")


if __name__ == "__main__":
    main()
