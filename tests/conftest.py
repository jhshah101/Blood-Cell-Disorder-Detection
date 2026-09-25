import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def tiny_imagefolder(tmp_path):
    """Synthetic ImageFolder tree: three classes with imbalanced counts, plus a test tree."""
    rng = np.random.default_rng(0)
    classes = {"Basophil": 6, "Lymphocyte": 14, "Neutrophil": 20}
    for split, scale in (("Train", 1.0), ("TestA", 0.5)):
        for name, n in classes.items():
            d = tmp_path / split / name
            d.mkdir(parents=True)
            for i in range(max(2, int(n * scale))):
                base = {"Basophil": 40, "Lymphocyte": 120, "Neutrophil": 200}[name]
                arr = np.clip(rng.normal(base, 25, size=(48, 48, 3)), 0, 255).astype(np.uint8)
                Image.fromarray(arr).save(d / f"{name.lower()}_{i:03d}.png")
    return tmp_path
