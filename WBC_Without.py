#!/usr/bin/env python
"""Compatibility entry point (the manuscript cites this file name).

The original ``WBC_Without.py`` (i) evaluated the *test* loader after every
epoch and kept the checkpoint with the best *test* macro-F1 - i.e. it selected
the model on the test set - (ii) used a hard-coded ECA kernel of 3 instead of
the adaptive rule (k = 5 for 512 channels), (iii) trained a 256-d / 4-layer /
8-head encoder with AdamW at 3e-4, and (iv) contained neither the colour
feature branch nor the adaptive loss reweighting that the manuscript
describes.

This file now delegates to the documented pipeline in ``scripts/train.py``:

    python WBC_Without.py                      # proposed model, configs/default.yaml
    python WBC_Without.py --legacy             # the original architecture, but with
                                               # validation-based model selection
    python WBC_Without.py --config <yaml> --set train.seed=3 ...

Model selection always uses the validation split; the test set is evaluated
once, after training, on the selected checkpoint.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))

from scripts import train as _train  # noqa: E402


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--legacy" in argv:
        argv.remove("--legacy")
        if "--config" not in argv:
            argv = ["--config", str(HERE / "configs" / "legacy" / "wbc_without_original.yaml")] + argv
    elif "--config" not in argv:
        argv = ["--config", str(HERE / "configs" / "default.yaml")] + argv
    _train.main(argv)


if __name__ == "__main__":
    main()
