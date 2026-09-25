"""End-to-end smoke test of the training protocol on a synthetic dataset."""
import csv
import json
from pathlib import Path

import torch

from wbc.checkpoint import load_checkpoint
from wbc.config import load_config
from wbc.data import build_datasets, build_loaders
from wbc.engine import Trainer
from wbc.losses import build_criterion
from wbc.models import build_model
from wbc.seed import set_seed


class CountingLoader:
    """Wraps a DataLoader and counts how many times it is iterated."""

    def __init__(self, loader):
        self.loader = loader
        self.iterations = 0
        self.dataset = loader.dataset

    def __iter__(self):
        self.iterations += 1
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)


def _cfg(tmp_path, tiny_imagefolder, **over):
    overrides = [
        f"data.train_dir={(tiny_imagefolder / 'Train').as_posix()}",
        f"data.test_dir={(tiny_imagefolder / 'TestA').as_posix()}",
        f"data.split_file={(tmp_path / 'split.json').as_posix()}",
        "data.img_size=64", "data.batch_size=8", "data.num_workers=0", "data.val_fraction=0.25",
        "data.minority_classes=[Basophil]",
        "model.pretrained=false", "model.embed_dim=32", "model.depth=1", "model.num_heads=4",
        "model.dropout=0.0", "model.drop_path=0.0",
        "train.epochs=2", "train.amp=false", "train.device=cpu", "train.log_interval=0",
        f"output_dir={(tmp_path / 'runs').as_posix()}",
    ] + [f"{k}={v}" for k, v in over.items()]
    return load_config(overrides=overrides)


def test_protocol_end_to_end(tmp_path, tiny_imagefolder):
    cfg = _cfg(tmp_path, tiny_imagefolder, **{"loss.name": "alr"})
    set_seed(0)
    device = torch.device("cpu")
    bundle = build_datasets(cfg)
    assert Path(cfg.data.split_file).exists()  # split file created and persisted
    assert bundle.class_names == ["Basophil", "Lymphocyte", "Neutrophil"]
    assert sum(bundle.train_counts) + sum(bundle.val_counts) == 40
    train_loader, val_loader, test_loader = build_loaders(cfg, bundle, 0, device)
    test_loader = CountingLoader(test_loader)

    model = build_model(cfg, 3)
    criterion = build_criterion(cfg.loss, bundle.train_counts, device)
    out = tmp_path / "runs" / "exp"
    trainer = Trainer(cfg, model, criterion, train_loader, val_loader, test_loader, bundle.class_names, device, out,
                      counts={"train": bundle.train_counts}, split_info=bundle.split_info)
    results = trainer.fit()

    # test set touched exactly once, after training
    assert test_loader.iterations == 1
    assert results["best_epoch"] >= 1
    assert "test" in results and "macro_f1" in results["test"]
    assert results["test_ci"]["macro_f1"]["low"] <= results["test_ci"]["macro_f1"]["point"]
    # ALR trajectory recorded
    assert len(results["alr_history"]) == 3  # initial + 2 epochs
    assert abs(sum(results["final_class_weights"]) / 3 - 1.0) < 1e-5
    # artefacts
    for name in ("config.yaml", "history.csv", "best.pt", "last.pt", "results.json", "val_predictions.csv", "test_predictions.csv"):
        assert (out / name).exists(), name
    with open(out / "history.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2 and "class_weights" in rows[0] and "val_macro_f1" in rows[0]
    with open(out / "test_predictions.csv", newline="") as fh:
        preds = list(csv.DictReader(fh))
    assert len(preds) == len(test_loader.dataset) and preds[0]["path"].strip('"').endswith(".png")

    # checkpoint rebuilds the identical network
    model2, cfg2, classes2, payload = load_checkpoint(out / "best.pt", device)
    assert classes2 == bundle.class_names and cfg2.model.embed_dim == 32
    x, _, _ = next(iter(val_loader))
    model.eval()
    with torch.no_grad():
        assert torch.allclose(model(x), model2(x), atol=1e-6)


def test_weighted_sampler_and_augmentation_paths(tmp_path, tiny_imagefolder):
    cfg = _cfg(tmp_path, tiny_imagefolder, **{"data.sampler": "weighted", "data.augment": "basic",
                                              "loss.name": "focal", "model.backbone": "resnet18",
                                              "train.selection_metric": "val_loss", "train.epochs": 1})
    device = torch.device("cpu")
    bundle = build_datasets(cfg)
    train_loader, val_loader, test_loader = build_loaders(cfg, bundle, 0, device)
    assert train_loader.sampler.__class__.__name__ == "WeightedRandomSampler"
    model = build_model(cfg, 3)
    criterion = build_criterion(cfg.loss, bundle.train_counts, device)
    results = Trainer(cfg, model, criterion, train_loader, val_loader, test_loader, bundle.class_names, device, tmp_path / "r").fit()
    assert results["selection_metric"] == "val_loss" and "test" in results
