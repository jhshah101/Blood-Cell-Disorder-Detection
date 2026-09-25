"""Paired statistical comparison of two models.

Two different sampling units are supported and must never be confused:

1. **Run level** (``paired_run_tests``): the paired observations are the
   metric values of the two models over *n* matched random seeds (same split,
   same hyper-parameters).  With *n* non-zero paired differences the exact
   two-sided Wilcoxon signed-rank test enumerates 2**n sign assignments, so
   the smallest attainable two-sided p-value is ``2 / 2**n = 2**(1-n)``:

       n = 5  -> 0.0625      (cannot reach 0.05)
       n = 6  -> 0.03125
       n = 8  -> 0.0078125
       n = 11 -> 0.000977    (first n that can reach p < 0.001)

   ``min_two_sided_exact_p`` and ``min_runs_for_alpha`` expose these limits so
   that a reported p-value can be checked against the design that produced it.
   The paired t-test and the exact sign-flip permutation test on the mean
   difference are reported alongside, together with the effect size (Cohen's
   d_z) and a bootstrap CI of the mean difference.

2. **Image level** (``mcnemar_exact``): the paired observations are the
   per-image correctness indicators of the two models on the *same* test set.
   McNemar's exact test uses only the discordant pairs and, with thousands of
   test images, *can* legitimately produce p < 0.001.  This is the test to use
   when a strong per-sample claim is made; it must be reported with the
   discordant counts b and c.

Every function returns plain dictionaries so the numbers can be written to the
results files and quoted in the manuscript together with the raw paired values.
"""
from __future__ import annotations

import itertools
import math
from typing import Dict, Optional, Sequence

import numpy as np
from scipy import stats as sps


# --------------------------------------------------------------------------- #
# Design limits
# --------------------------------------------------------------------------- #
def min_two_sided_exact_p(n_nonzero_pairs: int) -> float:
    """Smallest two-sided p-value an exact signed-rank / sign-flip test can give."""
    if n_nonzero_pairs < 1:
        return 1.0
    return min(1.0, 2.0 ** (1 - n_nonzero_pairs))


def min_runs_for_alpha(alpha: float) -> int:
    """Smallest n such that ``min_two_sided_exact_p(n) <= alpha``."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    return int(math.ceil(1.0 - math.log2(alpha)))


# --------------------------------------------------------------------------- #
# Exact signed-rank test by enumeration (transparent, tie-aware ranks)
# --------------------------------------------------------------------------- #
def exact_wilcoxon_signed_rank(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("a and b must have the same length (paired observations)")
    d = a - b
    nz = d[d != 0]
    n = int(len(nz))
    if n == 0:
        return {"n_nonzero": 0, "w_plus": 0.0, "w_minus": 0.0, "p_value": 1.0, "min_attainable_p": 1.0}
    ranks = sps.rankdata(np.abs(nz))
    w_plus = float(ranks[nz > 0].sum())
    w_minus = float(ranks[nz < 0].sum())
    if n > 22:  # enumeration would be too large; fall back to scipy
        res = sps.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        return {
            "n_nonzero": n,
            "w_plus": w_plus,
            "w_minus": w_minus,
            "p_value": float(res.pvalue),
            "min_attainable_p": min_two_sided_exact_p(n),
            "method": "scipy",
        }
    signs = np.array(list(itertools.product([0.0, 1.0], repeat=n)))  # 2**n x n
    dist = signs @ ranks  # every attainable W+
    p_low = float(np.mean(dist <= w_plus))
    p_high = float(np.mean(dist >= w_plus))
    p_two = min(1.0, 2.0 * min(p_low, p_high))
    return {
        "n_nonzero": n,
        "w_plus": w_plus,
        "w_minus": w_minus,
        "p_value": p_two,
        "min_attainable_p": min_two_sided_exact_p(n),
        "method": "exact_enumeration",
    }


def sign_flip_permutation_test(a: Sequence[float], b: Sequence[float], n_perm: int = 20000, seed: int = 0) -> Dict[str, float]:
    """Exact (n <= 20) or Monte-Carlo sign-flip test on the mean paired difference."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return {"n_nonzero": 0, "mean_diff": 0.0, "p_value": 1.0}
    observed = abs(d.mean())
    if n <= 20:
        signs = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
        dist = np.abs(signs @ d) / n
        p = float(np.mean(dist >= observed - 1e-12))
        method = "exact"
    else:
        rng = np.random.default_rng(seed)
        flips = rng.choice([-1.0, 1.0], size=(n_perm, n))
        dist = np.abs(flips @ d) / n
        p = float((np.sum(dist >= observed - 1e-12) + 1) / (n_perm + 1))
        method = "monte_carlo"
    return {"n_nonzero": int(n), "mean_diff": float(d.mean()), "p_value": p, "method": method}


def bootstrap_mean_difference(a: Sequence[float], b: Sequence[float], n_boot: int = 10000, alpha: float = 0.05, seed: int = 0) -> Dict[str, float]:
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(d)
    means = np.array([d[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return {
        "mean_diff": float(d.mean()),
        "ci_low": float(np.quantile(means, alpha / 2)),
        "ci_high": float(np.quantile(means, 1 - alpha / 2)),
        "n_boot": n_boot,
    }


def cohens_dz(a: Sequence[float], b: Sequence[float]) -> float:
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    sd = d.std(ddof=1) if len(d) > 1 else 0.0
    return float(d.mean() / sd) if sd > 0 else float("inf") if d.mean() != 0 else 0.0


def paired_run_tests(a: Sequence[float], b: Sequence[float], label_a: str = "A", label_b: str = "B") -> Dict:
    """All run-level paired tests between two equally long metric sequences."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b):
        raise ValueError("Paired comparison requires the same number of runs for both models")
    out: Dict = {
        "label_a": label_a,
        "label_b": label_b,
        "n_runs": int(len(a)),
        "values_a": a.tolist(),
        "values_b": b.tolist(),
        "paired_differences": (a - b).tolist(),
        "mean_a": float(a.mean()),
        "sd_a": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        "mean_b": float(b.mean()),
        "sd_b": float(b.std(ddof=1)) if len(b) > 1 else 0.0,
        "wilcoxon_exact": exact_wilcoxon_signed_rank(a, b),
        "sign_flip_permutation": sign_flip_permutation_test(a, b),
        "bootstrap_mean_difference": bootstrap_mean_difference(a, b),
        "cohens_dz": cohens_dz(a, b),
        "min_attainable_two_sided_p": min_two_sided_exact_p(int(np.sum((a - b) != 0))),
    }
    if len(a) > 1:
        t = sps.ttest_rel(a, b)
        out["paired_t"] = {"t": float(t.statistic), "p_value": float(t.pvalue), "df": int(len(a) - 1)}
    return out


# --------------------------------------------------------------------------- #
# Image-level paired test
# --------------------------------------------------------------------------- #
def mcnemar_exact(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> Dict[str, float]:
    """Exact McNemar test on per-image correctness of two models on one test set."""
    ca = np.asarray(correct_a, dtype=bool)
    cb = np.asarray(correct_b, dtype=bool)
    if ca.shape != cb.shape:
        raise ValueError("Both models must be evaluated on the same images")
    b = int(np.sum(ca & ~cb))  # A right, B wrong
    c = int(np.sum(~ca & cb))  # A wrong, B right
    n = b + c
    if n == 0:
        p = 1.0
        mid_p = 1.0
    else:
        k = min(b, c)
        cdf = sps.binom.cdf(k, n, 0.5)
        p = min(1.0, 2.0 * cdf)
        mid_p = min(1.0, 2.0 * (cdf - 0.5 * sps.binom.pmf(k, n, 0.5)))
    return {
        "n_images": int(len(ca)),
        "acc_a": float(ca.mean()),
        "acc_b": float(cb.mean()),
        "b_a_right_b_wrong": b,
        "c_a_wrong_b_right": c,
        "discordant": n,
        "p_value_exact": float(p),
        "p_value_mid": float(mid_p),
    }


def describe_design_limits(n_runs: int) -> str:
    return (
        f"With n = {n_runs} paired runs the exact two-sided Wilcoxon signed-rank test "
        f"cannot report p < {min_two_sided_exact_p(n_runs):.4g}. "
        f"Reaching p < 0.05 needs n >= {min_runs_for_alpha(0.05)}; "
        f"p < 0.01 needs n >= {min_runs_for_alpha(0.01)}; "
        f"p < 0.001 needs n >= {min_runs_for_alpha(0.001)}."
    )
