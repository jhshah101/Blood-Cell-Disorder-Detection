"""Classification metrics with *documented* aggregation rules.

Definitions used everywhere in this repository (and that the manuscript tables
must quote):

* accuracy            - fraction of correctly classified images
* precision / recall  - per class, one-vs-rest
* per-class F1        - harmonic mean of that class's precision and recall
* macro-F1            - unweighted arithmetic mean of the per-class F1 values
* weighted-F1         - support-weighted mean of the per-class F1 values
* balanced accuracy   - unweighted mean of the per-class recalls
* MCC, Cohen's kappa  - multiclass versions from scikit-learn
* majority-class F1   - unweighted mean of per-class F1 over the classes that
                        are *not* listed in ``minority_classes``
* minority-class F1   - unweighted mean of per-class F1 over ``minority_classes``
* F1 gap              - majority F1 - minority F1 (percentage points)
* F1 balance ratio    - minority F1 / majority F1 (manuscript Eq. 16)

Because macro-F1 is *by definition* the mean of the per-class F1 values, and
balanced accuracy the mean of the per-class recalls, the arithmetic identities
the reviewer checked hold automatically for every table produced from this
module.  Bootstrap confidence intervals are provided for the summary metrics.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)


def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
    minority_classes: Sequence[str] = (),
    y_prob: Optional[np.ndarray] = None,
) -> Dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    n_classes = len(class_names)
    labels = list(range(n_classes))

    p, r, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    minority_idx = [i for i, c in enumerate(class_names) if c in set(minority_classes)]
    majority_idx = [i for i in labels if i not in minority_idx]
    unknown = set(minority_classes) - set(class_names)
    if unknown:
        raise ValueError(f"minority_classes not in class_names: {sorted(unknown)}")

    out: Dict = {
        "n_samples": int(len(y_true)),
        "accuracy": float((y_true == y_pred).mean()) if len(y_true) else 0.0,
        "macro_precision": float(p.mean()),
        "macro_recall": float(r.mean()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float((f1 * support).sum() / max(support.sum(), 1)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if len(y_true) else 0.0,
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(y_true) else 0.0,
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)) if len(y_true) else 0.0,
        "per_class": {
            class_names[i]: {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in labels
        },
        "confusion_matrix": cm.tolist(),
        "class_names": list(class_names),
        "minority_classes": [class_names[i] for i in minority_idx],
        "majority_classes": [class_names[i] for i in majority_idx],
    }
    if minority_idx and majority_idx:
        maj = float(f1[majority_idx].mean())
        mino = float(f1[minority_idx].mean())
        out["majority_f1"] = maj
        out["minority_f1"] = mino
        out["f1_gap"] = maj - mino
        out["f1_balance_ratio"] = mino / maj if maj > 0 else float("nan")

    if y_prob is not None and len(y_true):
        y_prob = np.asarray(y_prob, dtype=np.float64)
        y_prob = y_prob / np.clip(y_prob.sum(axis=1, keepdims=True), 1e-12, None)  # float32 softmax rows may not sum to 1 exactly
        try:
            present = np.unique(y_true)
            if len(present) == n_classes:
                out["macro_auroc"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
            out["log_loss"] = float(log_loss(y_true, y_prob, labels=labels))
        except ValueError:
            pass
        out["mean_confidence"] = float(y_prob.max(axis=1).mean())
    return out


def summary_row(metrics: Dict, scale: float = 100.0) -> Dict[str, float]:
    """Flatten the headline numbers (in percent by default) for tables."""
    keys = [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "balanced_accuracy",
        "majority_f1",
        "minority_f1",
        "f1_gap",
    ]
    row = {k: metrics[k] * scale for k in keys if k in metrics}
    for k in ("mcc", "cohen_kappa", "f1_balance_ratio", "macro_auroc"):
        if k in metrics:
            row[k] = metrics[k]
    for cname, m in metrics.get("per_class", {}).items():
        row[f"f1_{cname}"] = m["f1"] * scale
        row[f"recall_{cname}"] = m["recall"] * scale
        row[f"precision_{cname}"] = m["precision"] * scale
    return row


def bootstrap_ci(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    statistic: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict[str, float]:
    """Percentile bootstrap CI of ``statistic(y_true, y_pred)`` over images."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[i] = statistic(y_true[idx], y_pred[idx])
    return {
        "point": float(statistic(y_true, y_pred)),
        "low": float(np.quantile(vals, alpha / 2)),
        "high": float(np.quantile(vals, 1 - alpha / 2)),
        "n_boot": n_boot,
    }


def macro_f1_statistic(n_classes: int) -> Callable[[np.ndarray, np.ndarray], float]:
    labels = list(range(n_classes))

    def _stat(yt: np.ndarray, yp: np.ndarray) -> float:
        _, _, f1, _ = precision_recall_fscore_support(yt, yp, labels=labels, zero_division=0)
        return float(f1.mean())

    return _stat


def accuracy_statistic(yt: np.ndarray, yp: np.ndarray) -> float:
    return float((yt == yp).mean())


def verify_aggregation_identities(metrics: Dict, tol: float = 1e-9) -> List[str]:
    """Return a list of violated identities (empty when the table is consistent)."""
    problems = []
    f1s = [m["f1"] for m in metrics["per_class"].values()]
    recalls = [m["recall"] for m in metrics["per_class"].values()]
    if abs(float(np.mean(f1s)) - metrics["macro_f1"]) > tol:
        problems.append("macro_f1 != mean(per-class f1)")
    if abs(float(np.mean(recalls)) - metrics["balanced_accuracy"]) > 1e-6:
        problems.append("balanced_accuracy != mean(per-class recall)")
    return problems
