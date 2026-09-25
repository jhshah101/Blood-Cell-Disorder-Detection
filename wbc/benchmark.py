"""Computational cost under a *stated* protocol.

Cross-paper latency numbers are not comparable unless hardware, software
stack, batch size, precision, warm-up and timing method are identical.  This
module measures every model of this repository under one explicit protocol and
records the protocol next to the numbers, so a table like the manuscript's
Table 11 can be built from measurements made on one machine.

* parameters : total and trainable
* FLOPs      : ``torch.utils.flop_counter.FlopCounterMode`` on one forward pass
               (reported as GFLOPs = 2 x GMACs, and GMACs separately, because
               the literature mixes the two conventions)
* latency    : batch size 1 (and optionally more), fp32, ``warmup`` untimed
               iterations, ``iters`` timed iterations, CUDA synchronised,
               median / mean / SD / p5 / p95 in milliseconds
"""
from __future__ import annotations

import statistics
import time
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .utils import environment_info


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}


def _sdpa_custom_mapping() -> Dict:
    """Register scaled-dot-product-attention kernels the stock counter may not know.

    ``FlopCounterMode`` ships formulas for the CUDA flash / efficient kernels
    but (depending on the torch version) not for the CPU flash kernel, which
    would make the QK^T and AV products of every attention layer count as zero.
    """
    try:
        from torch.utils.flop_counter import sdpa_flop_count
    except ImportError:  # pragma: no cover - very old torch
        return {}
    aten = torch.ops.aten
    mapping: Dict = {}

    # The counter keys its registry by overload *packet* and, through
    # ``shape_wrapper``, passes tensor *shapes* (not tensors) to the formula.
    def _cpu_flash(query_shape, key_shape, value_shape, *args, out_shape=None, **kwargs) -> int:
        return int(sdpa_flop_count(query_shape, key_shape, value_shape))

    for name in ("_scaled_dot_product_flash_attention_for_cpu", "_scaled_dot_product_fused_attention_overrideable"):
        op = getattr(aten, name, None)
        if op is not None:
            mapping[op] = _cpu_flash
    return mapping


def count_flops(model: nn.Module, input_shape=(1, 3, 224, 224), device: Optional[torch.device] = None) -> Dict[str, float]:
    """FLOPs of one forward pass.

    The count is taken with autograd *enabled* (no backward pass is run): under
    ``torch.no_grad`` ``nn.MultiheadAttention`` takes a fused fast path that the
    FLOP counter cannot see, which silently drops every attention projection
    from the total.  Latency, by contrast, is measured under ``no_grad`` because
    that is the inference regime.
    """
    from torch.utils.flop_counter import FlopCounterMode

    device = device or next(model.parameters()).device
    model.eval()
    x = torch.rand(*input_shape, device=device)
    counter = FlopCounterMode(display=False, custom_mapping=_sdpa_custom_mapping())
    with torch.enable_grad(), counter:
        model(x)
    flops = float(counter.get_total_flops())
    return {"gflops": flops / 1e9, "gmacs": flops / 2e9, "input_shape": list(input_shape)}


@torch.no_grad()
def measure_latency(
    model: nn.Module,
    input_shape=(1, 3, 224, 224),
    device: Optional[torch.device] = None,
    warmup: int = 20,
    iters: int = 100,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, float]:
    device = device or next(model.parameters()).device
    model.eval()
    x = torch.rand(*input_shape, device=device, dtype=dtype)
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32):
        for _ in range(warmup):
            model(x)
        sync()
        times: List[float] = []
        for _ in range(iters):
            t0 = time.perf_counter()
            model(x)
            sync()
            times.append((time.perf_counter() - t0) * 1000.0)
    times_sorted = sorted(times)
    return {
        "batch_size": int(input_shape[0]),
        "dtype": str(dtype).replace("torch.", ""),
        "warmup": warmup,
        "iters": iters,
        "median_ms": statistics.median(times),
        "mean_ms": statistics.fmean(times),
        "sd_ms": statistics.pstdev(times),
        "p5_ms": times_sorted[int(0.05 * (iters - 1))],
        "p95_ms": times_sorted[int(0.95 * (iters - 1))],
        "per_image_ms": statistics.median(times) / int(input_shape[0]),
    }


def benchmark_model(model: nn.Module, img_size: int = 224, device: Optional[torch.device] = None, batch_sizes=(1,), warmup: int = 20, iters: int = 100) -> Dict:
    device = device or next(model.parameters()).device
    out: Dict = {
        "environment": environment_info(),
        "device": str(device),
        "parameters": count_parameters(model),
        "flops": count_flops(model, (1, 3, img_size, img_size), device),
        "latency": [measure_latency(model, (b, 3, img_size, img_size), device, warmup, iters) for b in batch_sizes],
        "protocol": (
            "Single process, model.eval(), torch.no_grad(), random input, "
            f"{warmup} warm-up iterations discarded, {iters} timed iterations, "
            "CUDA synchronised before each stop-watch read, fp32 unless stated. "
            "Numbers from other papers measured under other protocols are not comparable."
        ),
    }
    return out
