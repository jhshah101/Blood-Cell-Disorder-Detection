"""External evaluation sets (Raabin Test-B, LISC) with explicit class mapping.

Rules that the manuscript must state and that this module enforces:

* The trained model is applied **without any fine-tuning**.
* The mapping from the external folder names to the five training classes is
  explicit (``mapping``), case-insensitive, and every folder that is not
  mapped is *excluded and counted* (LISC's mixed ``mixt`` folder is the
  canonical example).
* When the external set covers only a subset of the classes (Raabin Test-B
  has only neutrophils and lymphocytes) two evaluations are reported:

  - **unrestricted**: the 5-way arg-max; an image predicted as any absent
    class is an error, and the full 5-column confusion matrix is shown;
  - **restricted**: the arg-max over the present classes only, which is the
    2-class operating point a deployment would use.

  Both numbers are informative and neither should be presented alone.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

log = logging.getLogger("wbc")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

DEFAULT_MAPPINGS: Dict[str, Dict[str, str]] = {
    # Raabin-WBC Test-B: double-labelled neutrophils and lymphocytes only.
    "raabin_testb": {"neutrophil": "Neutrophil", "lymphocyte": "Lymphocyte"},
    # LISC main dataset folders.  ``mixt`` (mixed / unclassifiable) is excluded.
    "lisc": {
        "baso": "Basophil",
        "basophil": "Basophil",
        "eosi": "Eosinophil",
        "eosinophil": "Eosinophil",
        "lymp": "Lymphocyte",
        "lymphocyte": "Lymphocyte",
        "mono": "Monocyte",
        "monocyte": "Monocyte",
        "neut": "Neutrophil",
        "neutrophil": "Neutrophil",
    },
}


def parse_mapping(text: str) -> Dict[str, str]:
    """``"neut=Neutrophil,lymp=Lymphocyte"`` -> dict (keys lower-cased)."""
    out: Dict[str, str] = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"mapping entry {item!r} must look like folder=TrainClass")
        k, v = item.split("=", 1)
        out[k.strip().lower()] = v.strip()
    return out


def _mask_bbox(mask: np.ndarray, margin: float) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    h, w = y1 - y0 + 1, x1 - x0 + 1
    my, mx = int(round(h * margin)), int(round(w * margin))
    return max(0, x0 - mx), max(0, y0 - my), min(mask.shape[1], x1 + mx + 1), min(mask.shape[0], y1 + my + 1)


class MappedImageFolder(Dataset):
    """ImageFolder whose folder names are mapped onto the training class list.

    Optionally crops each image to the bounding box of a binary annotation
    mask (LISC style).  ``mask_dir`` is searched for a file whose stem starts
    with the image stem; when ``mask_required`` is False and no mask is found
    the full image is used and counted in ``no_mask``.
    """

    def __init__(
        self,
        root: str | Path,
        class_names: Sequence[str],
        mapping: Dict[str, str],
        transform=None,
        mask_dir: Optional[str | Path] = None,
        mask_required: bool = False,
        crop_margin: float = 0.10,
    ):
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        self.class_names = list(class_names)
        self.transform = transform
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.mask_required = mask_required
        self.crop_margin = crop_margin
        mapping = {k.lower(): v for k, v in mapping.items()}
        for v in mapping.values():
            if v not in self.class_names:
                raise ValueError(f"mapping target {v!r} is not one of the training classes {self.class_names}")
        self.samples: List[Tuple[str, int]] = []
        self.excluded: Dict[str, int] = {}
        self.no_mask: int = 0
        self.folder_to_class: Dict[str, str] = {}
        for sub in sorted(p for p in self.root.iterdir() if p.is_dir()):
            files = sorted(f for f in sub.rglob("*") if f.suffix.lower() in IMG_EXTS)
            target = mapping.get(sub.name.lower())
            if target is None:
                self.excluded[sub.name] = len(files)
                continue
            self.folder_to_class[sub.name] = target
            label = self.class_names.index(target)
            self.samples += [(str(f), label) for f in files]
        if not self.samples:
            raise RuntimeError(f"No images found under {self.root} for mapping {mapping}")
        self.targets = [y for _, y in self.samples]
        self.present_classes = sorted({y for y in self.targets})
        if self.excluded:
            log.warning("Excluded unmapped folders: %s", self.excluded)

    def _find_mask(self, image_path: Path) -> Optional[Path]:
        if self.mask_dir is None:
            return None
        cands = sorted(self.mask_dir.rglob(image_path.stem + "*"))
        cands = [c for c in cands if c.suffix.lower() in IMG_EXTS]
        return cands[0] if cands else None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        path, y = self.samples[i]
        img = Image.open(path).convert("RGB")
        mask_path = self._find_mask(Path(path))
        if mask_path is not None:
            m = np.asarray(Image.open(mask_path).convert("L")) > 0
            box = _mask_bbox(m, self.crop_margin)
            if box is not None:
                img = img.crop(box)
        elif self.mask_dir is not None:
            if self.mask_required:
                raise FileNotFoundError(f"No annotation mask for {path}")
            self.no_mask += 1
        if self.transform is not None:
            img = self.transform(img)
        return img, y, i

    def summary(self) -> Dict:
        counts = {self.class_names[c]: int(sum(1 for y in self.targets if y == c)) for c in self.present_classes}
        return {
            "root": str(self.root),
            "n_images": len(self.samples),
            "folder_to_class": self.folder_to_class,
            "excluded_folders": self.excluded,
            "per_class": counts,
            "present_classes": [self.class_names[c] for c in self.present_classes],
            "crop_to_mask": self.mask_dir is not None,
            "crop_margin": self.crop_margin,
        }


def restricted_argmax(probs: np.ndarray, allowed: Sequence[int]) -> np.ndarray:
    """Arg-max over ``allowed`` class indices only (subset operating point)."""
    allowed = list(allowed)
    sub = probs[:, allowed]
    return np.asarray(allowed)[sub.argmax(axis=1)]
