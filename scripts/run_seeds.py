#!/usr/bin/env python
"""Repeat experiments over matched seeds and compare them with paired tests.

    python scripts/run_seeds.py --configs configs/baselines/resnet50.yaml configs/default.yaml \
        --seeds 0 1 2 3 4 5 6 7 --metrics macro_f1 minority_f1 accuracy

The first config is the reference.  For every other config and every metric
the script reports mean +/- SD, the paired differences, the exact Wilcoxon
signed-rank p-value together with the smallest p-value the design can
produce, the paired t-test, the exact sign-flip permutation test, Cohen's d_z
and a bootstrap CI - and, per seed, the image-level McNemar test computed from
the per-image test predictions.  Everything is written to
<output_dir>/seeds_summary.{json,md}.
"""
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import _bootstrap  # noqa: F401
import numpy as np

from wbc.config import load_config
from wbc.stats import describe_design_limits, mcnemar_exact, paired_run_tests
from wbc.utils import format_table, load_json, save_json

import train as train_script


def _read_predictions(path: Path) -> Dict[str, bool]:
    out: Dict[str, bool] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            key = json.loads(r["path"]) if r["path"].startswith('"') else r["path"]
            out[key] = bool(int(r["correct"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    ap.add_argument("--metrics", nargs="+", default=["macro_f1", "minority_f1", "majority_f1", "accuracy", "balanced_accuracy"])
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--skip-existing", action="store_true", help="reuse results.json when present")
    args = ap.parse_args()

    runs: Dict[str, Dict[int, Dict]] = {}
    names: List[str] = []
    output_root = None
    for cfg_path in args.configs:
        base_cfg = load_config(cfg_path, args.overrides)
        names.append(base_cfg.experiment)
        output_root = output_root or Path(args.output_dir or base_cfg.output_dir)
        runs[base_cfg.experiment] = {}
        for seed in args.seeds:
            cfg = load_config(cfg_path, args.overrides)
            cfg.train.seed = seed
            out = output_root / cfg.experiment / f"seed{seed}"
            if args.skip_existing and (out / "results.json").exists():
                res = load_json(out / "results.json")
            else:
                res = train_script.run(cfg, out)
            runs[cfg.experiment][seed] = {"results": res, "dir": str(out)}

    ref = names[0]
    summary: Dict = {"seeds": args.seeds, "reference": ref, "experiments": {}, "comparisons": {},
                     "design_limits": describe_design_limits(len(args.seeds))}
    table_rows = []
    for name in names:
        per_metric = {}
        for m in args.metrics:
            vals = [runs[name][s]["results"]["test"][m] for s in args.seeds]
            per_metric[m] = {"values": vals, "mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0}
        per_metric["train_seconds"] = {"values": [runs[name][s]["results"]["train_seconds"] for s in args.seeds]}
        summary["experiments"][name] = per_metric
        row = {"experiment": name}
        for m in args.metrics:
            row[m] = f"{100 * per_metric[m]['mean']:.2f} +/- {100 * per_metric[m]['sd']:.2f}"
        row["train_s"] = f"{np.mean(per_metric['train_seconds']['values']):.0f}"
        table_rows.append(row)

    for name in names[1:]:
        comp: Dict = {}
        for m in args.metrics:
            a = [runs[name][s]["results"]["test"][m] for s in args.seeds]
            b = [runs[ref][s]["results"]["test"][m] for s in args.seeds]
            comp[m] = paired_run_tests(a, b, name, ref)
        mcn = []
        for s in args.seeds:
            pa = _read_predictions(Path(runs[name][s]["dir"]) / "test_predictions.csv")
            pb = _read_predictions(Path(runs[ref][s]["dir"]) / "test_predictions.csv")
            keys = sorted(set(pa) & set(pb))
            mcn.append({"seed": s, **mcnemar_exact([pa[k] for k in keys], [pb[k] for k in keys])})
        comp["mcnemar_per_seed"] = mcn
        summary["comparisons"][f"{name}_vs_{ref}"] = comp

    save_json(summary, output_root / "seeds_summary.json")
    md = ["# Repeated-seed summary", "", f"Seeds: {args.seeds}. Reference: `{ref}`.", "",
          format_table(table_rows, ["experiment", *args.metrics, "train_s"]), "",
          f"> {summary['design_limits']}", ""]
    for key, comp in summary["comparisons"].items():
        md.append(f"## {key}")
        md.append("")
        rows = []
        for m in args.metrics:
            c = comp[m]
            rows.append({
                "metric": m,
                "mean_diff": f"{100 * (c['mean_a'] - c['mean_b']):.2f}",
                "wilcoxon_p": f"{c['wilcoxon_exact']['p_value']:.4f} (floor {c['min_attainable_two_sided_p']:.4f})",
                "paired_t_p": f"{c.get('paired_t', {}).get('p_value', float('nan')):.4f}",
                "perm_p": f"{c['sign_flip_permutation']['p_value']:.4f}",
                "dz": f"{c['cohens_dz']:.2f}",
                "boot_ci": f"[{100 * c['bootstrap_mean_difference']['ci_low']:.2f}, {100 * c['bootstrap_mean_difference']['ci_high']:.2f}]",
            })
        md.append(format_table(rows, ["metric", "mean_diff", "wilcoxon_p", "paired_t_p", "perm_p", "dz", "boot_ci"]))
        md.append("")
        mrows = [{"seed": r["seed"], "b": r["b_a_right_b_wrong"], "c": r["c_a_wrong_b_right"], "p_exact": f"{r['p_value_exact']:.2e}"} for r in comp["mcnemar_per_seed"]]
        md.append("Image-level McNemar (per seed, same test images):")
        md.append("")
        md.append(format_table(mrows, ["seed", "b", "c", "p_exact"]))
        md.append("")
    (output_root / "seeds_summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
