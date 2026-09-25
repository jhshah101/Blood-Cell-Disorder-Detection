"""Loss functions.

* ``ce``    - plain cross-entropy
* ``wce``   - cross-entropy with static class weights, manuscript Eq. (13)
* ``focal`` - focal loss (Lin et al.), optionally class-weighted
* ``alr``   - Adaptive Loss Reweighting, manuscript Eqs. (13)-(15) with the
              corrections that make it a *relative* reweighting

Adaptive Loss Reweighting
-------------------------
Initial weights (Eq. 13):     w_c^(0) = (N_max / N_c) ** alpha, rescaled to mean 1.

Per-epoch statistics (from the training forward passes, no extra pass):
    L_c  = mean per-sample *unweighted* cross-entropy of class c   (difficulty)
    P_c  = mean softmax probability of the true class over the samples of c
           that were classified correctly (0 if none)                (confidence)

Update (Eq. 14):              w~_c = w_c^(t) (1 + beta L_c)(1 + gamma (1 - P_c))

Because every factor is >= 1, Eq. (14) alone can only *increase* every weight,
which is the defect the reviewer identified.  The implementation therefore
adds two steps that the revised manuscript must state explicitly:

    normalisation:  w_c^(t+1) = C * w~_c / sum_j w~_j       (mean-one rescaling)
    clipping:       w_c^(t+1) <- clip(w_c^(t+1), w_min, w_max)

With the rescaling, the update is invariant to the loss scale and a class whose
multiplicative factor is *below the weighted average factor* sees its weight
decrease - which is exactly the intended behaviour for well-learned majority
classes.  Optional exponential smoothing (``momentum``) damps epoch-to-epoch
oscillation.  Every epoch's L_c, P_c, factors and resulting weights are stored
in ``history`` and written to disk by the trainer so the trajectory can be
reported.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LossConfig


def initial_class_weights(counts: Sequence[int], alpha: float = 1.0, normalize: bool = True) -> torch.Tensor:
    """Eq. (13): ``(N_max / N_c) ** alpha``, rescaled to mean one when ``normalize``."""
    c = torch.as_tensor(list(counts), dtype=torch.float32).clamp_min(1.0)
    w = (c.max() / c) ** float(alpha)
    if normalize:
        w = w * len(w) / w.sum()
    return w


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)
        if weight is not None:
            self.register_buffer("weight", weight.clone().float())
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits.float(), dim=1)
        logpt = logp.gather(1, targets[:, None]).squeeze(1)
        pt = logpt.exp()
        loss = -((1.0 - pt) ** self.gamma) * logpt
        if self.label_smoothing > 0:
            smooth = -logp.mean(dim=1)
            loss = (1 - self.label_smoothing) * loss + self.label_smoothing * smooth
        if self.weight is not None:
            w = self.weight[targets]
            return (loss * w).sum() / w.sum().clamp_min(1e-8)
        return loss.mean()


class AdaptiveLossReweighting(nn.Module):
    """Weighted cross-entropy whose class weights are re-estimated every epoch."""

    def __init__(
        self,
        num_classes: int,
        init_weights: torch.Tensor,
        beta: float = 0.5,
        gamma: float = 0.5,
        normalize: str = "mean_one",
        w_min: float = 0.2,
        w_max: float = 5.0,
        momentum: float = 0.0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        if init_weights.numel() != num_classes:
            raise ValueError("init_weights must have one entry per class")
        self.num_classes = int(num_classes)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.normalize = normalize
        self.w_min = float(w_min)
        self.w_max = float(w_max)
        self.momentum = float(momentum)
        self.label_smoothing = float(label_smoothing)
        self.register_buffer("weights", init_weights.clone().float())
        self.register_buffer("loss_sum", torch.zeros(num_classes))
        self.register_buffer("conf_sum", torch.zeros(num_classes))
        self.register_buffer("count", torch.zeros(num_classes))
        self.register_buffer("correct", torch.zeros(num_classes))
        self.history: List[Dict[str, List[float]]] = [
            {"epoch": 0, "weights": self.weights.tolist(), "loss": None, "confidence": None, "factor": None}
        ]

    # -- Eq. (15): the loss actually optimised -------------------------------
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits.float(), targets, weight=self.weights, label_smoothing=self.label_smoothing)

    # -- per-batch bookkeeping for L_c and P_c -------------------------------
    @torch.no_grad()
    def accumulate(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        logits = logits.detach().float()
        per_sample = F.cross_entropy(logits, targets, reduction="none")
        probs = logits.softmax(dim=1)
        conf = probs.gather(1, targets[:, None]).squeeze(1)
        correct = (probs.argmax(dim=1) == targets).float()
        ones = torch.ones_like(per_sample)
        self.loss_sum.index_add_(0, targets, per_sample)
        self.count.index_add_(0, targets, ones)
        self.conf_sum.index_add_(0, targets, conf * correct)
        self.correct.index_add_(0, targets, correct)

    @torch.no_grad()
    def reset_stats(self) -> None:
        for buf in (self.loss_sum, self.conf_sum, self.count, self.correct):
            buf.zero_()

    @torch.no_grad()
    def class_statistics(self):
        L = self.loss_sum / self.count.clamp_min(1.0)
        P = torch.where(self.correct > 0, self.conf_sum / self.correct.clamp_min(1.0), torch.zeros_like(self.conf_sum))
        return L, P

    # -- Eq. (14) + normalisation + clipping, called once per epoch ----------
    @torch.no_grad()
    def end_epoch(self, epoch: int) -> Dict[str, List[float]]:
        L, P = self.class_statistics()
        factor = (1.0 + self.beta * L) * (1.0 + self.gamma * (1.0 - P))
        new = self.weights * factor
        if self.normalize == "mean_one":
            new = new * self.num_classes / new.sum().clamp_min(1e-8)
        new = new.clamp(self.w_min, self.w_max)
        if self.momentum > 0:
            new = self.momentum * self.weights + (1.0 - self.momentum) * new
        self.weights.copy_(new)
        record = {
            "epoch": int(epoch),
            "weights": self.weights.tolist(),
            "loss": L.tolist(),
            "confidence": P.tolist(),
            "factor": factor.tolist(),
        }
        self.history.append(record)
        self.reset_stats()
        return record


def build_criterion(lcfg: LossConfig, class_counts: Sequence[int], device: torch.device) -> nn.Module:
    """Instantiate the loss named in ``lcfg`` from the *training-split* class counts."""
    n = len(class_counts)
    if lcfg.name == "ce":
        return nn.CrossEntropyLoss(label_smoothing=lcfg.label_smoothing)
    w = initial_class_weights(class_counts, alpha=lcfg.alpha).to(device)
    if lcfg.name == "wce":
        return nn.CrossEntropyLoss(weight=w, label_smoothing=lcfg.label_smoothing)
    if lcfg.name == "focal":
        return FocalLoss(gamma=lcfg.focal_gamma, weight=w, label_smoothing=lcfg.label_smoothing).to(device)
    if lcfg.name == "alr":
        return AdaptiveLossReweighting(
            n,
            w,
            beta=lcfg.alr_beta,
            gamma=lcfg.alr_gamma,
            normalize=lcfg.alr_normalize,
            w_min=lcfg.alr_w_min,
            w_max=lcfg.alr_w_max,
            momentum=lcfg.alr_momentum,
            label_smoothing=lcfg.label_smoothing,
        ).to(device)
    raise ValueError(f"Unknown loss {lcfg.name!r}")
