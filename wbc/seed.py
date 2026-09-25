"""Reproducibility helpers.

``set_seed`` seeds Python, NumPy and PyTorch (CPU and every CUDA device) and,
when requested, switches PyTorch to deterministic kernels.  ``seed_worker`` and
``make_generator`` make DataLoader shuffling and worker-side randomness a pure
function of the run seed, so that two runs with the same seed see the same
mini-batches in the same order.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        # Required by cuBLAS for deterministic matmul on CUDA >= 10.2.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:  # very old torch without warn_only
            torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:  # noqa: ARG001 - signature fixed by torch
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g
