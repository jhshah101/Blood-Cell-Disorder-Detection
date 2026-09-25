import torch
import torch.nn.functional as F

from wbc.config import LossConfig
from wbc.losses import AdaptiveLossReweighting, FocalLoss, build_criterion, initial_class_weights


def test_initial_weights_eq13():
    w = initial_class_weights([100, 50, 10], alpha=1.0, normalize=False)
    assert torch.allclose(w, torch.tensor([1.0, 2.0, 10.0]))
    w = initial_class_weights([100, 50, 10], alpha=0.5, normalize=False)
    assert torch.allclose(w, torch.tensor([1.0, 2.0**0.5, 10**0.5]))
    wn = initial_class_weights([100, 50, 10], alpha=1.0, normalize=True)
    assert abs(wn.mean().item() - 1.0) < 1e-6


def _fake_batch(easy_class_conf=0.95, hard_class_conf=0.4):
    # class 0 easy (high confidence, low loss); class 1 hard
    logits = torch.zeros(20, 2)
    targets = torch.tensor([0] * 10 + [1] * 10)
    e = torch.log(torch.tensor([easy_class_conf, 1 - easy_class_conf]))
    h = torch.log(torch.tensor([1 - hard_class_conf, hard_class_conf]))
    logits[:10] = e
    logits[10:] = h
    return logits, targets


def test_alr_without_normalisation_is_monotone_as_reviewer_noted():
    alr = AdaptiveLossReweighting(2, torch.ones(2), beta=0.5, gamma=0.5, normalize="none", w_min=1e-3, w_max=1e3)
    logits, t = _fake_batch()
    alr.accumulate(logits, t)
    alr.end_epoch(1)
    assert torch.all(alr.weights > 1.0)  # both weights increased: Eq. (14) alone can only grow


def test_alr_with_mean_one_normalisation_reduces_easy_class_weight():
    alr = AdaptiveLossReweighting(2, torch.ones(2), beta=0.5, gamma=0.5, normalize="mean_one", w_min=1e-3, w_max=1e3)
    logits, t = _fake_batch()
    alr.accumulate(logits, t)
    rec = alr.end_epoch(1)
    w = alr.weights
    assert abs(w.mean().item() - 1.0) < 1e-6
    assert w[0] < 1.0 < w[1]  # easy class decreases, hard class increases
    assert rec["loss"][0] < rec["loss"][1]
    assert rec["confidence"][0] > rec["confidence"][1]
    assert len(alr.history) == 2 and alr.history[0]["epoch"] == 0


def test_alr_clipping_and_momentum():
    alr = AdaptiveLossReweighting(2, torch.ones(2), beta=50.0, gamma=50.0, normalize="mean_one", w_min=0.5, w_max=1.5, momentum=0.0)
    logits, t = _fake_batch()
    alr.accumulate(logits, t)
    alr.end_epoch(1)
    assert alr.weights.min() >= 0.5 and alr.weights.max() <= 1.5
    alr2 = AdaptiveLossReweighting(2, torch.ones(2), beta=0.5, gamma=0.5, momentum=0.9, w_min=1e-3, w_max=1e3)
    alr2.accumulate(logits, t)
    alr2.end_epoch(1)
    assert torch.all((alr2.weights - 1.0).abs() < 0.2)  # heavily smoothed


def test_alr_forward_is_weighted_ce():
    alr = AdaptiveLossReweighting(2, torch.tensor([1.5, 0.5]))
    logits, t = _fake_batch()
    assert torch.allclose(alr(logits, t), F.cross_entropy(logits, t, weight=torch.tensor([1.5, 0.5])))


def test_focal_reduces_to_ce_at_gamma_zero():
    logits, t = _fake_batch()
    assert torch.allclose(FocalLoss(gamma=0.0)(logits, t), F.cross_entropy(logits, t), atol=1e-6)
    assert FocalLoss(gamma=2.0)(logits, t) < F.cross_entropy(logits, t)


def test_build_criterion_dispatch():
    counts = [200, 50, 10]
    dev = torch.device("cpu")
    assert isinstance(build_criterion(LossConfig(name="ce"), counts, dev), torch.nn.CrossEntropyLoss)
    wce = build_criterion(LossConfig(name="wce"), counts, dev)
    assert wce.weight is not None and wce.weight.argmax().item() == 2
    assert isinstance(build_criterion(LossConfig(name="focal"), counts, dev), FocalLoss)
    assert isinstance(build_criterion(LossConfig(name="alr"), counts, dev), AdaptiveLossReweighting)
