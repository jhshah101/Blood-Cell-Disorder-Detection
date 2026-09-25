#!/usr/bin/env python
"""Parameters, FLOPs and latency of one or more models under one protocol.

    python scripts/benchmark.py --configs configs/default.yaml configs/baselines/resnet50.yaml --batch-sizes 1 32
    python scripts/benchmark.py --checkpoint runs/.../best.pt

Writes benchmark.json and prints a Markdown table.  The protocol (warm-up,
iterations, precision, device) is stored with the numbers.
"""
import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import torch

from wbc.benchmark import benchmark_model
from wbc.checkpoint import load_checkpoint
from wbc.config import load_config
from wbc.models import build_model
from wbc.utils import format_table, get_device, save_json


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs", nargs="*", default=[])
    ap.add_argument("--checkpoint")
    ap.add_argument("--num-classes", type=int, default=5)
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=[1])
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--output", default="benchmark.json")
    ap.add_argument("--no-pretrained", action="store_true", help="do not download ImageNet weights (cost is identical)")
    args = ap.parse_args()

    device = get_device()
    entries = []
    for path in args.configs:
        cfg = load_config(path)
        if args.no_pretrained:
            cfg.model.pretrained = False
        model = build_model(cfg, args.num_classes).to(device)
        entries.append((cfg.experiment, cfg.data.img_size, model))
    if args.checkpoint:
        model, cfg, classes, _ = load_checkpoint(args.checkpoint, device)
        entries.append((f"{cfg.experiment} ({Path(args.checkpoint).name})", cfg.data.img_size, model))
    if not entries:
        ap.error("give --configs and/or --checkpoint")

    report = {}
    rows = []
    for name, img_size, model in entries:
        r = benchmark_model(model, img_size, device, tuple(args.batch_sizes), args.warmup, args.iters)
        report[name] = r
        lat = r["latency"][0]
        rows.append({
            "model": name,
            "params_M": r["parameters"]["total"] / 1e6,
            "GFLOPs": r["flops"]["gflops"],
            "GMACs": r["flops"]["gmacs"],
            f"latency_ms_bs{lat['batch_size']}": lat["median_ms"],
        })
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    save_json(report, args.output)
    print(format_table(rows, list(rows[0].keys())))
    print(f"\nProtocol: {next(iter(report.values()))['protocol']}")
    print(f"Device: {device} | {next(iter(report.values()))['environment']}")


if __name__ == "__main__":
    main()
