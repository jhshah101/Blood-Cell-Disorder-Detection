#!/usr/bin/env python
"""Build manuscript-style tables from the run directories written by train.py.

    python scripts/build_tables.py --runs runs --out results --seeds 0 1 2 3 4 5 7 \
        --reference baseline_resnet50 \
        --baselines baseline_resnet18 baseline_resnet50 baseline_densenet121 ... \
        --ablation ablation_baseline_vit ablation_vit_eca_ws ... \
        --augmented-pairs baseline_resnet50:baseline_resnet50_augmented ablation_vit_eca_cf_alr:vit_eca_cf_alr_augmented \
        --sensitivity-prefix sens_ --trajectory ablation_vit_eca_cf_alr

Every number is the mean +/- SD over the matched seeds of the *test* metrics of
the checkpoint selected on *validation*; the selected epochs and the
validation-vs-test gap are tabulated as well.  Paired comparisons against the
reference experiment use the exact Wilcoxon signed-rank test (with its
attainable floor), the paired t-test, the exact sign-flip permutation test,
Cohen's d_z, a bootstrap CI of the mean difference and, per seed, McNemar's
exact test on the per-image test predictions.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import _bootstrap  # noqa: F401
import numpy as np

from wbc.stats import describe_design_limits, mcnemar_exact, paired_run_tests
from wbc.utils import format_table, load_json, save_json

OVERALL_KEYS = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "balanced_accuracy", "mcc", "majority_f1", "minority_f1", "f1_gap", "f1_balance_ratio"]


def load_runs(runs: Path, experiment: str, seeds: Sequence[int]) -> Dict[int, Dict]:
    out = {}
    for s in seeds:
        p = runs / experiment / f"seed{s}" / "results.json"
        if p.exists():
            out[s] = load_json(p)
    return out


def ms(values: Sequence[float], scale: float = 100.0, fmt: str = "{:.2f} ± {:.2f}") -> str:
    v = np.asarray(values, dtype=float) * scale
    if len(v) == 0:
        return "—"
    return fmt.format(v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0)


def pct(values: Sequence[float]) -> float:
    return float(np.mean(values) * 100.0) if len(values) else float("nan")


def read_predictions(path: Path) -> Dict[str, bool]:
    out: Dict[str, bool] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            key = json.loads(r["path"]) if r["path"].startswith('"') else r["path"]
            out[key] = bool(int(r["correct"]))
    return out


def per_class_table(runs_by_exp: Dict[str, Dict[int, Dict]], classes: List[str], title: str) -> str:
    rows = []
    for exp, runs in runs_by_exp.items():
        if not runs:
            continue
        for c in classes:
            rows.append({
                "Model": exp,
                "Class": c,
                "Precision (%)": ms([r["test"]["per_class"][c]["precision"] for r in runs.values()]),
                "Recall (%)": ms([r["test"]["per_class"][c]["recall"] for r in runs.values()]),
                "F1 (%)": ms([r["test"]["per_class"][c]["f1"] for r in runs.values()]),
                "n (test)": runs[next(iter(runs))]["test"]["per_class"][c]["support"],
            })
    return f"### {title}\n\n" + format_table(rows, ["Model", "Class", "Precision (%)", "Recall (%)", "F1 (%)", "n (test)"])


def overall_table(runs_by_exp: Dict[str, Dict[int, Dict]], title: str) -> str:
    rows = []
    for exp, runs in runs_by_exp.items():
        if not runs:
            continue
        t = [r["test"] for r in runs.values()]
        rows.append({
            "Model": exp,
            "n seeds": len(runs),
            "Accuracy (%)": ms([x["accuracy"] for x in t]),
            "Macro-P (%)": ms([x["macro_precision"] for x in t]),
            "Macro-R (%)": ms([x["macro_recall"] for x in t]),
            "Macro-F1 (%)": ms([x["macro_f1"] for x in t]),
            "Bal. acc. (%)": ms([x["balanced_accuracy"] for x in t]),
            "MCC": ms([x["mcc"] for x in t], scale=1.0, fmt="{:.3f} ± {:.3f}"),
            "Majority F1 (%)": ms([x.get("majority_f1", float("nan")) for x in t]),
            "Minority F1 (%)": ms([x.get("minority_f1", float("nan")) for x in t]),
            "Gap (pp)": ms([x.get("f1_gap", float("nan")) for x in t]),
            "Balance ratio": ms([x.get("f1_balance_ratio", float("nan")) for x in t], scale=1.0, fmt="{:.3f} ± {:.3f}"),
        })
    cols = ["Model", "n seeds", "Accuracy (%)", "Macro-P (%)", "Macro-R (%)", "Macro-F1 (%)", "Bal. acc. (%)", "MCC", "Majority F1 (%)", "Minority F1 (%)", "Gap (pp)", "Balance ratio"]
    return f"### {title}\n\n" + format_table(rows, cols)


def selection_table(runs_by_exp: Dict[str, Dict[int, Dict]]) -> str:
    rows = []
    for exp, runs in runs_by_exp.items():
        if not runs:
            continue
        rows.append({
            "Model": exp,
            "Selected epoch (mean ± SD)": ms([r["best_epoch"] for r in runs.values()], scale=1.0, fmt="{:.1f} ± {:.1f}"),
            "Val macro-F1 (%)": ms([r["val"]["macro_f1"] for r in runs.values()]),
            "Test macro-F1 (%)": ms([r["test"]["macro_f1"] for r in runs.values()]),
            "Val − test (pp)": f"{pct([r['val']['macro_f1'] for r in runs.values()]) - pct([r['test']['macro_f1'] for r in runs.values()]):+.2f}",
            "Train time (s)": ms([r["train_seconds"] for r in runs.values()], scale=1.0, fmt="{:.0f} ± {:.0f}"),
        })
    return format_table(rows, ["Model", "Selected epoch (mean ± SD)", "Val macro-F1 (%)", "Test macro-F1 (%)", "Val − test (pp)", "Train time (s)"])


def comparison_block(runs_a: Dict[int, Dict], runs_b: Dict[int, Dict], name_a: str, name_b: str, metrics: Sequence[str], runs_dir: Path) -> Dict:
    seeds = sorted(set(runs_a) & set(runs_b))
    out: Dict = {"seeds": seeds, "metrics": {}, "mcnemar": []}
    for m in metrics:
        a = [runs_a[s]["test"][m] for s in seeds]
        b = [runs_b[s]["test"][m] for s in seeds]
        out["metrics"][m] = paired_run_tests(a, b, name_a, name_b)
    for s in seeds:
        pa = read_predictions(runs_dir / name_a / f"seed{s}" / "test_predictions.csv")
        pb = read_predictions(runs_dir / name_b / f"seed{s}" / "test_predictions.csv")
        keys = sorted(set(pa) & set(pb))
        out["mcnemar"].append({"seed": s, **mcnemar_exact([pa[k] for k in keys], [pb[k] for k in keys])})
    return out


def comparison_markdown(comp: Dict, name_a: str, name_b: str) -> str:
    rows = []
    for m, c in comp["metrics"].items():
        rows.append({
            "Metric": m,
            f"{name_a} (%)": f"{100 * c['mean_a']:.2f} ± {100 * c['sd_a']:.2f}",
            f"{name_b} (%)": f"{100 * c['mean_b']:.2f} ± {100 * c['sd_b']:.2f}",
            "Mean diff (pp)": f"{100 * (c['mean_a'] - c['mean_b']):+.2f}",
            "Bootstrap 95% CI": f"[{100 * c['bootstrap_mean_difference']['ci_low']:+.2f}, {100 * c['bootstrap_mean_difference']['ci_high']:+.2f}]",
            "Wilcoxon exact p (floor)": f"{c['wilcoxon_exact']['p_value']:.4f} ({c['min_attainable_two_sided_p']:.4f})",
            "Paired t p": f"{c.get('paired_t', {}).get('p_value', float('nan')):.2e}",
            "Perm. p": f"{c['sign_flip_permutation']['p_value']:.4f}",
            "d_z": f"{c['cohens_dz']:.2f}",
        })
    md = format_table(rows, list(rows[0].keys())) if rows else ""
    mrows = [{"Seed": r["seed"], "b (A right, B wrong)": r["b_a_right_b_wrong"], "c (A wrong, B right)": r["c_a_wrong_b_right"], "McNemar exact p": f"{r['p_value_exact']:.2e}"} for r in comp["mcnemar"]]
    md += "\n\nImage-level McNemar test per seed (A = " + name_a + ", B = " + name_b + "):\n\n" + format_table(mrows, ["Seed", "b (A right, B wrong)", "c (A wrong, B right)", "McNemar exact p"])
    return md


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--reference", required=True, help="experiment name used as the baseline for paired tests")
    ap.add_argument("--baselines", nargs="*", default=[])
    ap.add_argument("--ablation", nargs="*", default=[])
    ap.add_argument("--extra", nargs="*", default=[], help="additional experiments to include in the comparison")
    ap.add_argument("--augmented-pairs", nargs="*", default=[], help="plain:augmented experiment-name pairs")
    ap.add_argument("--sensitivity-prefix", default=None)
    ap.add_argument("--trajectory", default=None, help="experiment whose ALR weight trajectory is plotted")
    ap.add_argument("--metrics", nargs="+", default=["macro_f1", "minority_f1", "majority_f1", "accuracy", "balanced_accuracy"])
    ap.add_argument("--note", default="", help="a sentence prepended to every table file (e.g. a data disclaimer)")
    args = ap.parse_args()

    runs_dir, out = Path(args.runs), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    note = (args.note + "\n\n") if args.note else ""
    all_names = list(dict.fromkeys(args.baselines + args.ablation + args.extra + [args.reference]))
    runs = {n: load_runs(runs_dir, n, args.seeds) for n in all_names}
    missing = [n for n, r in runs.items() if len(r) < len(args.seeds)]
    if missing:
        print("WARNING incomplete experiments:", {n: sorted(runs[n]) for n in missing})
    any_run = next(r for rr in runs.values() for r in rr.values())
    classes = any_run["class_names"]
    counts = any_run.get("counts", {})
    minority = any_run["test"].get("minority_classes", [])

    header = (f"Test metrics of the checkpoint selected on validation macro-F1, mean ± SD over seeds {args.seeds}. "
              f"Classes: {classes}; minority classes: {minority}. Train / val / test counts: {counts}.\n\n")

    # ---- Table 4: baselines --------------------------------------------------
    if args.baselines:
        b = {n: runs[n] for n in args.baselines}
        txt = note + "# CNN baselines (Table 4)\n\n" + header + per_class_table(b, classes, "Class-wise") + "\n\n" + overall_table(b, "Overall") + "\n\n### Selection transparency\n\n" + selection_table(b) + "\n"
        (out / "table4_baselines.md").write_text(txt, encoding="utf-8")

    # ---- Tables 5-8: ablation ------------------------------------------------
    if args.ablation:
        a = {n: runs[n] for n in args.ablation}
        best_cnn = None
        if args.baselines:
            best_cnn = max(args.baselines, key=lambda n: pct([r["test"].get("minority_f1", 0) for r in runs[n].values()]) if runs[n] else -1)
        rows = []
        for n in ([best_cnn] if best_cnn else []) + args.ablation:
            rr = runs[n]
            if not rr:
                continue
            t = [r["test"] for r in rr.values()]
            row = {
                "Model": n + (" (best CNN)" if n == best_cnn else ""),
                "Minority F1 (%)": ms([x.get("minority_f1") for x in t]),
                "Majority F1 (%)": ms([x.get("majority_f1") for x in t]),
                "Balance ratio": ms([x.get("f1_balance_ratio") for x in t], scale=1.0, fmt="{:.3f} ± {:.3f}"),
                "Macro-F1 (%)": ms([x["macro_f1"] for x in t]),
                "Accuracy (%)": ms([x["accuracy"] for x in t]),
                "Macro-P (%)": ms([x["macro_precision"] for x in t]),
                "Macro-R (%)": ms([x["macro_recall"] for x in t]),
            }
            if best_cnn and runs[best_cnn]:
                row["Δ minority F1 vs best CNN (pp)"] = f"{pct([x.get('minority_f1') for x in t]) - pct([r['test'].get('minority_f1') for r in runs[best_cnn].values()]):+.2f}"
            rows.append(row)
        t8 = format_table(rows, list(rows[0].keys())) if rows else ""
        txt = (note + "# Ablation of ViT-ECA-CF components (Tables 5, 6 and 8)\n\n" + header
               + per_class_table(a, classes, "Class-wise (Table 5)") + "\n\n" + overall_table(a, "Overall (Table 6)")
               + "\n\n### Minority / majority summary (Table 8 layout)\n\nΔ is an absolute difference in percentage points.\n\n" + t8
               + "\n\n### Selection transparency\n\n" + selection_table(a) + "\n")
        (out / "table5-8_ablation.md").write_text(txt, encoding="utf-8")

    # ---- Table 10: augmented vs non-augmented --------------------------------
    if args.augmented_pairs:
        rows, comps = [], {}
        pairs = [p.split(":") for p in args.augmented_pairs]
        for plain, aug in pairs:
            for setting, n in (("Non-augmented", plain), ("Augmented", aug)):
                rr = load_runs(runs_dir, n, args.seeds)
                runs[n] = rr
                if not rr:
                    continue
                t = [r["test"] for r in rr.values()]
                rows.append({
                    "Setting": setting, "Model": n,
                    "Accuracy (%)": ms([x["accuracy"] for x in t]), "Macro-P (%)": ms([x["macro_precision"] for x in t]),
                    "Macro-R (%)": ms([x["macro_recall"] for x in t]), "Macro-F1 (%)": ms([x["macro_f1"] for x in t]),
                    "Minority F1 (%)": ms([x.get("minority_f1") for x in t]),
                    "Train time (s)": ms([r["train_seconds"] for r in rr.values()], scale=1.0, fmt="{:.0f} ± {:.0f}"),
                })
        txt = note + "# Augmented vs non-augmented (Table 10)\n\n" + header + format_table(rows, list(rows[0].keys())) + "\n\n"
        ref_plain, ref_aug = pairs[0]
        for plain, aug in pairs[1:]:
            for setting, n, ref in (("Non-augmented", plain, ref_plain), ("Augmented", aug, ref_aug)):
                if runs.get(n) and runs.get(ref):
                    comp = comparison_block(runs[n], runs[ref], n, ref, args.metrics, runs_dir)
                    comps[f"{setting}: {n} vs {ref}"] = comp
                    txt += f"### {setting}: {n} vs {ref}\n\n" + comparison_markdown(comp, n, ref) + "\n\n"
        txt += f"> {describe_design_limits(len(args.seeds))}\n"
        (out / "table10_augmentation.md").write_text(txt, encoding="utf-8")
        save_json(comps, out / "table10_augmentation_stats.json")

    # ---- Statistics vs reference ---------------------------------------------
    comps = {}
    txt = note + f"# Paired statistics against `{args.reference}`\n\n" + header + f"> {describe_design_limits(len(args.seeds))}\n\n"
    for n in all_names:
        if n == args.reference or not runs[n] or not runs[args.reference]:
            continue
        comp = comparison_block(runs[n], runs[args.reference], n, args.reference, args.metrics, runs_dir)
        comps[n] = comp
        txt += f"## {n} vs {args.reference}\n\n" + comparison_markdown(comp, n, args.reference) + "\n\n"
    (out / "statistics.md").write_text(txt, encoding="utf-8")
    save_json(comps, out / "statistics.json")

    # ---- Sensitivity grid ----------------------------------------------------
    if args.sensitivity_prefix:
        rows = []
        for d in sorted(runs_dir.glob(args.sensitivity_prefix + "*")):
            rr = load_runs(runs_dir, d.name, args.seeds)
            if not rr:
                continue
            cfg = next(iter(rr.values()))["config"]["loss"]
            rows.append({
                "β": cfg["alr_beta"], "γ": cfg["alr_gamma"], "n seeds": len(rr),
                "Val macro-F1 (%)": ms([r["val"]["macro_f1"] for r in rr.values()]),
                "Val minority F1 (%)": ms([r["val"].get("minority_f1") for r in rr.values()]),
                "Val accuracy (%)": ms([r["val"]["accuracy"] for r in rr.values()]),
                "Test macro-F1 (%)": ms([r["test"]["macro_f1"] for r in rr.values()]),
                "Test minority F1 (%)": ms([r["test"].get("minority_f1") for r in rr.values()]),
            })
        txt = (note + "# ALR sensitivity to β and γ\n\nSelection of the operating constants must use the **validation** columns; "
               "the test columns are shown only for transparency.\n\n" + format_table(rows, list(rows[0].keys())) + "\n") if rows else ""
        if txt:
            (out / "alr_sensitivity.md").write_text(txt, encoding="utf-8")

    # ---- ALR trajectory ------------------------------------------------------
    if args.trajectory and runs.get(args.trajectory):
        r0 = runs[args.trajectory][min(runs[args.trajectory])]
        hist = r0.get("alr_history", [])
        if hist:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            epochs = [h["epoch"] for h in hist]
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            for i, c in enumerate(classes):
                axes[0].plot(epochs, [h["weights"][i] for h in hist], marker="o", label=c)
                axes[1].plot(epochs[1:], [h["loss"][i] for h in hist[1:]], marker="o", label=c)
                axes[2].plot(epochs[1:], [h["confidence"][i] for h in hist[1:]], marker="o", label=c)
            axes[0].axhline(1.0, color="k", lw=0.8, ls="--")
            axes[0].set_title("Class weight w_c (mean one)"); axes[1].set_title("Class loss L_c"); axes[2].set_title("Class confidence P_c")
            for ax in axes:
                ax.set_xlabel("epoch"); ax.grid(alpha=0.3)
            axes[0].legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(out / "alr_weight_trajectories.png", dpi=150)
            rows = [{"Epoch": h["epoch"], **{c: f"{h['weights'][i]:.3f}" for i, c in enumerate(classes)},
                     **{f"L_{c}": f"{h['loss'][i]:.3f}" if h["loss"] else "—" for i, c in enumerate(classes)},
                     **{f"P_{c}": f"{h['confidence'][i]:.3f}" if h["confidence"] else "—" for i, c in enumerate(classes)}} for h in hist]
            txt = (note + f"# ALR weight trajectory ({args.trajectory}, seed {min(runs[args.trajectory])})\n\n"
                   f"![trajectory](alr_weight_trajectories.png)\n\nWeights are rescaled to mean one after every update and clipped to "
                   f"[{r0['config']['loss']['alr_w_min']}, {r0['config']['loss']['alr_w_max']}]; β = {r0['config']['loss']['alr_beta']}, γ = {r0['config']['loss']['alr_gamma']}.\n\n"
                   + format_table(rows, list(rows[0].keys())) + "\n")
            (out / "alr_weight_trajectories.md").write_text(txt, encoding="utf-8")

    # ---- machine-readable summary -------------------------------------------
    summary = {n: {"seeds": sorted(rr), "test": {k: [r["test"].get(k) for r in rr.values()] for k in OVERALL_KEYS},
                   "val_macro_f1": [r["val"]["macro_f1"] for r in rr.values()], "best_epoch": [r["best_epoch"] for r in rr.values()],
                   "train_seconds": [r["train_seconds"] for r in rr.values()]} for n, rr in runs.items() if rr}
    save_json(summary, out / "summary.json")
    print(f"wrote tables to {out}")


if __name__ == "__main__":
    main()
