import json

import numpy as np
import pytest

from wbc.data.splits import make_split, save_split, split_indices


def _samples(n_per_class=(30, 60, 200)):
    samples, i = [], 0
    for c, n in enumerate(n_per_class):
        for k in range(n):
            samples.append((f"/data/Train/class{c}/img_{i:04d}.png", c))
            i += 1
    return samples


def test_stratified_split_preserves_class_proportions_and_is_deterministic():
    samples = _samples()
    classes = ["c0", "c1", "c2"]
    s1 = make_split(samples, classes, 0.2, 42, root="/data/Train")
    s2 = make_split(samples, classes, 0.2, 42, root="/data/Train")
    assert s1 == s2
    assert s1["n_train"] + s1["n_val"] == len(samples)
    for c, n in zip(classes, (30, 60, 200)):
        assert abs(s1["val_counts"][c] - 0.2 * n) <= 1
    assert not set(e["path"] for e in s1["train"]) & set(e["path"] for e in s1["val"])
    assert s1["train"][0]["path"].startswith("class0/")  # stored relative to root


def test_group_aware_split_keeps_groups_together():
    samples = _samples((20, 20, 40))
    groups = {}
    for p, _ in samples:
        # 8 images per slide
        idx = int(p.split("_")[-1].split(".")[0])
        groups[p.replace("/data/Train/", "")] = f"slide{idx // 8}"
    s = make_split(samples, ["c0", "c1", "c2"], 0.25, 0, root="/data/Train", groups=groups)
    g_train = {groups[e["path"]] for e in s["train"]}
    g_val = {groups[e["path"]] for e in s["val"]}
    assert s["group_aware"] and not (g_train & g_val)


def test_split_indices_roundtrip_and_error_on_missing(tmp_path):
    samples = _samples((10, 10, 10))
    classes = ["c0", "c1", "c2"]
    s = make_split(samples, classes, 0.3, 1, root="/data/Train")
    save_split(s, tmp_path / "split.json")
    loaded = json.loads((tmp_path / "split.json").read_text())
    tr, va = split_indices(loaded, samples, root="/data/Train")
    assert len(tr) == s["n_train"] and len(va) == s["n_val"] and not set(tr) & set(va)
    with pytest.raises(FileNotFoundError):
        split_indices(loaded, samples[:-3], root="/data/Train")


def test_val_fraction_bounds():
    with pytest.raises(ValueError):
        make_split(_samples(), ["c0", "c1", "c2"], 0.6, 0)
