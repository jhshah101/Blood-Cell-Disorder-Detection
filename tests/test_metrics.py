import numpy as np
import pytest

from wbc.metrics import bootstrap_ci, compute_metrics, macro_f1_statistic, summary_row, verify_aggregation_identities

CLASSES = ["Basophil", "Eosinophil", "Lymphocyte", "Monocyte", "Neutrophil"]
MINORITY = ["Basophil", "Eosinophil", "Monocyte"]


def _data():
    rng = np.random.default_rng(1)
    y_true = np.repeat(np.arange(5), [20, 40, 100, 30, 200])
    y_pred = y_true.copy()
    flip = rng.random(len(y_true)) < 0.15
    y_pred[flip] = rng.integers(0, 5, flip.sum())
    return y_true, y_pred


def test_macro_f1_is_mean_of_per_class_and_balanced_acc_is_mean_recall():
    y_true, y_pred = _data()
    m = compute_metrics(y_true, y_pred, CLASSES, MINORITY)
    f1s = [m["per_class"][c]["f1"] for c in CLASSES]
    recalls = [m["per_class"][c]["recall"] for c in CLASSES]
    assert abs(np.mean(f1s) - m["macro_f1"]) < 1e-12
    assert abs(np.mean(recalls) - m["balanced_accuracy"]) < 1e-9
    assert verify_aggregation_identities(m) == []


def test_majority_minority_aggregation_rule():
    y_true, y_pred = _data()
    m = compute_metrics(y_true, y_pred, CLASSES, MINORITY)
    mino = np.mean([m["per_class"][c]["f1"] for c in MINORITY])
    maj = np.mean([m["per_class"][c]["f1"] for c in ("Lymphocyte", "Neutrophil")])
    assert abs(m["minority_f1"] - mino) < 1e-12
    assert abs(m["majority_f1"] - maj) < 1e-12
    assert abs(m["f1_gap"] - (maj - mino)) < 1e-12
    assert abs(m["f1_balance_ratio"] - mino / maj) < 1e-12
    assert m["minority_classes"] == MINORITY


def test_unknown_minority_class_rejected():
    y_true, y_pred = _data()
    with pytest.raises(ValueError):
        compute_metrics(y_true, y_pred, CLASSES, ["Platelet"])


def test_probabilities_add_auroc_and_logloss():
    y_true, y_pred = _data()
    rng = np.random.default_rng(0)
    probs = rng.random((len(y_true), 5))
    probs[np.arange(len(y_true)), y_pred] += 2.0
    probs /= probs.sum(1, keepdims=True)
    m = compute_metrics(y_true, probs.argmax(1), CLASSES, MINORITY, probs)
    assert 0 <= m["macro_auroc"] <= 1 and m["log_loss"] > 0
    row = summary_row(m)
    assert "f1_Basophil" in row and abs(row["macro_f1"] - 100 * m["macro_f1"]) < 1e-9


def test_bootstrap_ci_contains_point():
    y_true, y_pred = _data()
    ci = bootstrap_ci(y_true, y_pred, macro_f1_statistic(5), n_boot=200)
    assert ci["low"] <= ci["point"] <= ci["high"]
