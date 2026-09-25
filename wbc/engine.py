"""Training / evaluation engine.

Protocol guarantees (these are the properties the reviewers asked for):

* the checkpoint is selected on the **validation** split only
  (``train.selection_metric``); the test loader is never touched before the
  final evaluation, which runs exactly once on the selected checkpoint;
* every epoch's learning rate, losses, validation metrics and - for ALR - the
  full class-weight vector are logged to ``history.csv``;
* per-image predictions with file paths are written for validation and test so
  that image-level paired tests (McNemar) between models are possible;
* the resolved configuration, environment, class counts of every split and the
  timing are stored in ``results.json`` together with the metrics.
"""
from __future__ import annotations

import copy
import json
import logging
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import Config, save_config, to_dict
from .losses import AdaptiveLossReweighting
from .metrics import accuracy_statistic, bootstrap_ci, compute_metrics, macro_f1_statistic, summary_row
from .utils import CSVLogger, environment_info, save_json

log = logging.getLogger("wbc")


def build_optimizer(cfg: Config, model: nn.Module) -> torch.optim.Optimizer:
    t = cfg.train
    params = [p for p in model.parameters() if p.requires_grad]
    if t.optimizer == "adam":
        return torch.optim.Adam(params, lr=t.lr, weight_decay=t.weight_decay)
    if t.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=t.lr, weight_decay=t.weight_decay)
    if t.optimizer == "sgd":
        return torch.optim.SGD(params, lr=t.lr, momentum=0.9, nesterov=True, weight_decay=t.weight_decay)
    raise ValueError(t.optimizer)


def build_scheduler(cfg: Config, optimizer: torch.optim.Optimizer, steps_per_epoch: int):
    t = cfg.train
    if t.scheduler == "none":
        return None
    total = max(1, t.epochs * steps_per_epoch)
    warm = max(0, int(t.warmup_epochs * steps_per_epoch))

    def lr_lambda(step: int) -> float:
        if warm > 0 and step < warm:
            return 0.01 + 0.99 * step / warm
        progress = (step - warm) / max(1, total - warm)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class Trainer:
    def __init__(
        self,
        cfg: Config,
        model: nn.Module,
        criterion: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader],
        class_names: List[str],
        device: torch.device,
        output_dir: Path,
        counts: Optional[Dict[str, List[int]]] = None,
        split_info: Optional[Dict] = None,
    ):
        self.cfg = cfg
        self.model = model.to(device)
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.class_names = list(class_names)
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        save_config(cfg, self.output_dir / "config.yaml")  # the resolved configuration always travels with the run
        self.counts = counts or {}
        self.split_info = split_info or {}
        self.optimizer = build_optimizer(cfg, self.model)
        self.scheduler = build_scheduler(cfg, self.optimizer, len(train_loader))
        self.use_amp = bool(cfg.train.amp and device.type == "cuda")
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        except (AttributeError, TypeError):  # torch < 2.3
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.history = CSVLogger(self.output_dir / "history.csv")
        self.is_alr = isinstance(criterion, AdaptiveLossReweighting)
        self.higher_is_better = cfg.train.selection_metric != "val_loss"
        self.best_score = -float("inf") if self.higher_is_better else float("inf")
        self.best_epoch = -1
        self.best_state: Optional[Dict[str, torch.Tensor]] = None
        self.epoch_times: List[float] = []

    # ------------------------------------------------------------------ #
    def _selection_value(self, val_metrics: Dict, val_loss: float) -> float:
        m = self.cfg.train.selection_metric
        if m == "val_loss":
            return val_loss
        return float(val_metrics[m])

    def _is_improvement(self, score: float) -> bool:
        return score > self.best_score if self.higher_is_better else score < self.best_score

    # ------------------------------------------------------------------ #
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        running, correct, seen = 0.0, 0, 0
        t0 = time.perf_counter()
        for step, (x, y, _) in enumerate(self.train_loader, start=1):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                logits = self.model(x)
                loss = self.criterion(logits, y)
            self.scaler.scale(loss).backward()
            if self.cfg.train.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scheduler is not None:
                self.scheduler.step()
            if self.is_alr:
                self.criterion.accumulate(logits, y)
            running += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            seen += x.size(0)
            if self.cfg.train.log_interval and step % self.cfg.train.log_interval == 0:
                log.info("epoch %d step %d/%d loss %.4f", epoch, step, len(self.train_loader), running / max(seen, 1))
        elapsed = time.perf_counter() - t0
        self.epoch_times.append(elapsed)
        return {"train_loss": running / max(seen, 1), "train_acc": correct / max(seen, 1), "epoch_seconds": elapsed}

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Tuple[Dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """Return ``(metrics, y_true, y_pred, probs, indices, unweighted_ce)``."""
        self.model.eval()
        ys, preds, probs, idxs = [], [], [], []
        loss_sum = 0.0
        for x, y, idx in loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                logits = self.model(x)
            logits = logits.float()
            loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
            p = logits.softmax(1)
            ys.append(y.cpu())
            preds.append(p.argmax(1).cpu())
            probs.append(p.cpu())
            idxs.append(idx)
        y_true = torch.cat(ys).numpy()
        y_pred = torch.cat(preds).numpy()
        prob = torch.cat(probs).numpy()
        indices = torch.cat(idxs).numpy()
        metrics = compute_metrics(y_true, y_pred, self.class_names, self.cfg.data.minority_classes, prob)
        return metrics, y_true, y_pred, prob, indices, loss_sum / max(len(y_true), 1)

    # ------------------------------------------------------------------ #
    def _write_predictions(self, name: str, loader: DataLoader, y_true, y_pred, prob, indices) -> None:
        paths = getattr(loader.dataset, "paths", None)
        header = ["index", "path", "y_true", "y_pred", "correct"] + [f"p_{c}" for c in self.class_names]
        lines = [",".join(header)]
        for i in range(len(y_true)):
            path = paths[int(indices[i])] if paths is not None else ""
            row = [str(int(indices[i])), json.dumps(path), str(int(y_true[i])), str(int(y_pred[i])), str(int(y_true[i] == y_pred[i]))]
            row += [f"{v:.6f}" for v in prob[i]]
            lines.append(",".join(row))
        (self.output_dir / f"{name}_predictions.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _checkpoint(self, path: Path, epoch: int, val_metrics: Dict) -> None:
        payload = {
            "model_state": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
            "classes": self.class_names,
            "config": to_dict(self.cfg),
            "epoch": epoch,
            "val_metrics": val_metrics,
        }
        if self.is_alr:
            payload["alr_history"] = self.criterion.history
        torch.save(payload, path)

    # ------------------------------------------------------------------ #
    def fit(self) -> Dict:
        cfg = self.cfg
        patience = cfg.train.early_stopping_patience
        bad_epochs = 0
        t_start = time.perf_counter()
        log.info("Training %s on %s | %d train / %d val images | %d classes", cfg.experiment, self.device,
                 len(self.train_loader.dataset), len(self.val_loader.dataset), len(self.class_names))
        for epoch in range(1, cfg.train.epochs + 1):
            train_stats = self.train_epoch(epoch)
            alr_record = self.criterion.end_epoch(epoch) if self.is_alr else None
            val_metrics, *_ , val_loss = self.evaluate(self.val_loader)
            score = self._selection_value(val_metrics, val_loss)
            row = {
                "epoch": epoch,
                "lr": self.optimizer.param_groups[0]["lr"],
                **train_stats,
                "val_loss": val_loss,
                **{f"val_{k}": v for k, v in summary_row(val_metrics).items()},
            }
            if alr_record is not None:
                row["class_weights"] = json.dumps([round(w, 5) for w in alr_record["weights"]])
                row["class_loss"] = json.dumps([round(v, 5) for v in alr_record["loss"]])
                row["class_confidence"] = json.dumps([round(v, 5) for v in alr_record["confidence"]])
            self.history.log(row)
            improved = self._is_improvement(score)
            log.info(
                "epoch %02d/%d | train_loss %.4f | val_loss %.4f | val_acc %.4f | val_macroF1 %.4f | val_minF1 %s | %s",
                epoch, cfg.train.epochs, train_stats["train_loss"], val_loss, val_metrics["accuracy"],
                val_metrics["macro_f1"], f"{val_metrics.get('minority_f1', float('nan')):.4f}",
                "* new best" if improved else "",
            )
            if improved:
                self.best_score, self.best_epoch, bad_epochs = score, epoch, 0
                self.best_state = copy.deepcopy(self.model.state_dict())
                self._checkpoint(self.output_dir / "best.pt", epoch, val_metrics)
            else:
                bad_epochs += 1
                if patience and bad_epochs >= patience:
                    log.info("Early stopping after %d epochs without improvement on the validation split", patience)
                    break
        train_seconds = time.perf_counter() - t_start
        self._checkpoint(self.output_dir / "last.pt", epoch, val_metrics)

        # ---- final evaluation on the selected checkpoint (test seen once) ----
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        val_metrics, yt, yp, pr, ix, val_loss = self.evaluate(self.val_loader)
        self._write_predictions("val", self.val_loader, yt, yp, pr, ix)
        results: Dict = {
            "experiment": cfg.experiment,
            "seed": cfg.train.seed,
            "config": to_dict(cfg),
            "environment": environment_info(),
            "class_names": self.class_names,
            "counts": self.counts,
            "split": self.split_info,
            "best_epoch": self.best_epoch,
            "selection_metric": cfg.train.selection_metric,
            "epochs_run": epoch,
            "train_seconds": train_seconds,
            "mean_epoch_seconds": float(np.mean(self.epoch_times)) if self.epoch_times else None,
            "val": val_metrics,
            "val_loss": val_loss,
        }
        if self.is_alr:
            results["alr_history"] = self.criterion.history
            results["final_class_weights"] = self.criterion.weights.tolist()
        if self.test_loader is not None:
            test_metrics, yt, yp, pr, ix, test_loss = self.evaluate(self.test_loader)
            self._write_predictions("test", self.test_loader, yt, yp, pr, ix)
            n_classes = len(self.class_names)
            results["test"] = test_metrics
            results["test_loss"] = test_loss
            results["test_ci"] = {
                "macro_f1": bootstrap_ci(yt, yp, macro_f1_statistic(n_classes), seed=cfg.train.seed),
                "accuracy": bootstrap_ci(yt, yp, accuracy_statistic, seed=cfg.train.seed),
            }
            log.info(
                "TEST (checkpoint from epoch %d, selected on validation %s) | acc %.4f | macroF1 %.4f | balAcc %.4f | minF1 %s | majF1 %s",
                self.best_epoch, cfg.train.selection_metric, test_metrics["accuracy"], test_metrics["macro_f1"],
                test_metrics["balanced_accuracy"], f"{test_metrics.get('minority_f1', float('nan')):.4f}",
                f"{test_metrics.get('majority_f1', float('nan')):.4f}",
            )
        save_json(results, self.output_dir / "results.json")
        return results
