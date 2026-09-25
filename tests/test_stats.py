import numpy as np
from scipy import stats as sps

from wbc.stats import (
    describe_design_limits,
    exact_wilcoxon_signed_rank,
    mcnemar_exact,
    min_runs_for_alpha,
    min_two_sided_exact_p,
    paired_run_tests,
    sign_flip_permutation_test,
)


def test_small_n_floor_of_exact_wilcoxon():
    assert min_two_sided_exact_p(5) == 0.0625
    assert min_two_sided_exact_p(6) == 0.03125
    assert min_two_sided_exact_p(11) < 0.001 < min_two_sided_exact_p(10)
    assert min_runs_for_alpha(0.05) == 6
    assert min_runs_for_alpha(0.01) == 8
    assert min_runs_for_alpha(0.001) == 11
    # five runs, every difference in the same direction: the most extreme case
    res = exact_wilcoxon_signed_rank([96.1, 96.4, 96.0, 96.3, 96.2], [89.2, 89.9, 89.5, 89.4, 89.8])
    assert res["n_nonzero"] == 5
    assert abs(res["p_value"] - 0.0625) < 1e-12
    assert "0.0625" in describe_design_limits(5)


def test_exact_wilcoxon_agrees_with_scipy_without_ties():
    rng = np.random.default_rng(3)
    for _ in range(5):
        a = rng.normal(0.3, 1, 9)
        b = rng.normal(0.0, 1, 9)
        ours = exact_wilcoxon_signed_rank(a, b)["p_value"]
        ref = sps.wilcoxon(a, b, alternative="two-sided", method="exact").pvalue
        assert abs(ours - ref) < 1e-9


def test_sign_flip_floor_and_symmetry():
    res = sign_flip_permutation_test([1, 2, 3, 4, 5], [0, 0, 0, 0, 0])
    assert abs(res["p_value"] - 0.0625) < 1e-12
    assert res["method"] == "exact"


def test_paired_run_tests_reports_everything():
    a = [0.961, 0.964, 0.960, 0.963, 0.962]
    b = [0.892, 0.899, 0.895, 0.894, 0.898]
    r = paired_run_tests(a, b, "proposed", "resnet50")
    assert r["n_runs"] == 5
    assert r["wilcoxon_exact"]["p_value"] == 0.0625
    assert r["min_attainable_two_sided_p"] == 0.0625
    assert r["paired_t"]["p_value"] < 1e-4
    assert r["cohens_dz"] > 5
    assert r["bootstrap_mean_difference"]["ci_low"] > 0


def test_mcnemar_exact():
    a = np.array([True] * 900 + [False] * 100)
    b = a.copy()
    b[:60] = False  # A right, B wrong on 60 images
    b[900:920] = True  # A wrong, B right on 20 images
    r = mcnemar_exact(a, b)
    assert r["b_a_right_b_wrong"] == 60 and r["c_a_wrong_b_right"] == 20
    assert r["p_value_exact"] < 0.001
    same = mcnemar_exact(a, a)
    assert same["p_value_exact"] == 1.0
