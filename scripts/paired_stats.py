#!/usr/bin/env python
"""Paired statistics between two models, at run level and / or image level.

Run level (metric values over matched seeds):
    python scripts/paired_stats.py --runs-a runs/vit_eca_cf_alr --runs-b runs/resnet50 --metric macro_f1
    python scripts/paired_stats.py --values-a 96.1 96.4 96.0 96.3 96.2 --values-b 89.2 89.9 89.5 89.4 89.8

Image level (per-image correctness on the same test set):
    python scripts/paired_stats.py --predictions-a runs/vit_eca_cf_alr/seed0/test_predictions.csv \
                                   --predictions-b runs/resnet50/seed0/test_predictions.csv
"""
import argparse
import csv
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from wbc.stats import describe_design_limits, mcnemar_exact, paired_run_tests
from wbc.utils import load_json


def _collect(run_dir: Path, metric: str):
    vals, seeds = [], []
    for res in sorted(run_dir.glob("seed*/results.json")):
        r = load_json(res)
        vals.append(r["test"][metric])
        seeds.append(r["seed"])
    if not vals:
        raise SystemExit(f"No seed*/results.json under {run_dir}")
    return seeds, vals


def _read_predictions(path: Path):
    out = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            key = json.loads(r["path"]) if r["path"].startswith('"') else r["path"]
            out[key] = bool(int(r["correct"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-a")
    ap.add_argument("--runs-b")
    ap.add_argument("--metric", default="macro_f1")
    ap.add_argument("--values-a", nargs="*", type=float)
    ap.add_argument("--values-b", nargs="*", type=float)
    ap.add_argument("--predictions-a")
    ap.add_argument("--predictions-b")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    args = ap.parse_args()

    if args.runs_a and args.runs_b:
        sa, a = _collect(Path(args.runs_a), args.metric)
        sb, b = _collect(Path(args.runs_b), args.metric)
        if sa != sb:
            raise SystemExit(f"Seeds differ: {sa} vs {sb}; paired tests need matched seeds")
        res = paired_run_tests(a, b, args.label_a, args.label_b)
        print(json.dumps(res, indent=2))
        print(describe_design_limits(len(a)))
    elif args.values_a and args.values_b:
        res = paired_run_tests(args.values_a, args.values_b, args.label_a, args.label_b)
        print(json.dumps(res, indent=2))
        print(describe_design_limits(len(args.values_a)))
    if args.predictions_a and args.predictions_b:
        pa, pb = _read_predictions(Path(args.predictions_a)), _read_predictions(Path(args.predictions_b))
        keys = sorted(set(pa) & set(pb))
        if not keys:
            raise SystemExit("The two prediction files share no image paths")
        print(json.dumps(mcnemar_exact([pa[k] for k in keys], [pb[k] for k in keys]), indent=2))


if __name__ == "__main__":
    main()
