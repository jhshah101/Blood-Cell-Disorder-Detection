#!/usr/bin/env python
"""Compatibility entry point for the former ``main.py`` (Hybrid CNN-ViT).

The original script hard-coded Windows data paths, created the positional
embedding lazily inside ``forward`` *after* the optimiser had been built (so
the embedding was never updated and the saved state could not be reloaded into
a fresh model), normalised the class weights to sum one, and trained for up to
200 epochs with early stopping on validation loss.

It now delegates to ``scripts/train.py``:

    python main.py                       # proposed model (configs/default.yaml)
    python main.py --legacy              # the original geometry and schedule, fixed
    python main.py --config <yaml> --set data.train_dir=/path/to/Train ...
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
            argv = ["--config", str(HERE / "configs" / "legacy" / "main_hybrid_cnn_vit.yaml")] + argv
    elif "--config" not in argv:
        argv = ["--config", str(HERE / "configs" / "default.yaml")] + argv
    _train.main(argv)


if __name__ == "__main__":
    main()
