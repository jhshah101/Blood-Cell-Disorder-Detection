#!/usr/bin/env python
"""Generate a synthetic peripheral-blood-smear dataset with the Raabin-WBC structure.

The purpose is to exercise the *complete* experimental protocol (split, seeds,
ablation, external evaluation, statistics, benchmark, Grad-CAM) end to end on
a machine without the real data, producing every results file in exactly the
form the real experiment produces.  The images are procedurally drawn
leukocytes with class-specific morphology and colour:

    Lymphocyte  small cell, large round nucleus (high N:C), pale blue cytoplasm
    Monocyte    large cell, kidney-shaped nucleus, grey-blue cytoplasm, vacuoles
    Neutrophil  3-5 nuclear lobes joined by filaments, fine lilac granules
    Eosinophil  bilobed nucleus, coarse orange-red granules
    Basophil    coarse dark-purple granules obscuring a bilobed nucleus

on an RBC-strewn pink background, with per-image rotation, jitter, noise, blur
and colour-gain variation.  Three partitions mirror Raabin-WBC:

    Train   class counts = official Raabin training counts scaled by --scale
    TestA   class counts = official Test-A counts scaled by --scale
    TestB   neutrophils + lymphocytes only, drawn under an *acquisition shift*
            (warmer colour balance, lower magnification, stronger blur)

and a LISC-style set of full-field images (one leukocyte among red cells, a
binary expert mask, lower-case folder names including an unmapped ``mixt``
folder) exercises the external-evaluation path with mask cropping.

Nothing about the images is biological ground truth; every number obtained on
them is a placeholder that must be regenerated on the real data.
"""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

RAABIN_TRAIN = {"Neutrophil": 6231, "Lymphocyte": 2427, "Eosinophil": 744, "Monocyte": 561, "Basophil": 212}
RAABIN_TESTA = {"Neutrophil": 2660, "Lymphocyte": 1034, "Eosinophil": 322, "Monocyte": 234, "Basophil": 89}
TESTB = {"Neutrophil": 1500, "Lymphocyte": 750}  # Test-B is a two-class, shifted-acquisition set
LISC = {"neut": 50, "lymp": 52, "mono": 48, "eosi": 39, "baso": 53, "mixt": 8}  # LISC main-set folder names

NUCLEUS = (88, 48, 132)
CYTO = {
    "Lymphocyte": (176, 196, 232),
    "Monocyte": (186, 192, 214),
    "Neutrophil": (226, 206, 226),
    "Eosinophil": (232, 202, 204),
    "Basophil": (200, 182, 216),
}
CELL_RADIUS = {  # fraction of canvas half-size, mean and sd
    "Lymphocyte": (0.36, 0.04),
    "Monocyte": (0.56, 0.05),
    "Neutrophil": (0.46, 0.04),
    "Eosinophil": (0.46, 0.04),
    "Basophil": (0.44, 0.04),
}


def _jitter(rgb: Tuple[int, int, int], rng: random.Random, amount: int = 14) -> Tuple[int, int, int]:
    return tuple(int(max(0, min(255, c + rng.randint(-amount, amount)))) for c in rgb)


def _background(size: int, rng: random.Random, tint: Tuple[int, int, int]) -> Image.Image:
    img = Image.new("RGB", (size, size), _jitter(tint, rng, 10))
    d = ImageDraw.Draw(img, "RGBA")
    for _ in range(rng.randint(4, 9)):  # red blood cells, partially clipped
        r = rng.uniform(0.13, 0.19) * size
        cx, cy = rng.uniform(-r, size + r), rng.uniform(-r, size + r)
        col = _jitter((238, 178, 178), rng, 12) + (rng.randint(150, 220),)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col)
        inner = _jitter((246, 200, 200), rng, 8) + (rng.randint(80, 140),)
        d.ellipse([cx - 0.45 * r, cy - 0.45 * r, cx + 0.45 * r, cy + 0.45 * r], fill=inner)
    return img


def _dots(d: ImageDraw.ImageDraw, cx: float, cy: float, radius: float, n: int, r_range: Tuple[float, float], colour, rng: random.Random, alpha: int = 255) -> None:
    for _ in range(n):
        t = rng.uniform(0, 2 * math.pi)
        rr = radius * math.sqrt(rng.uniform(0, 1)) * 0.92
        x, y = cx + rr * math.cos(t), cy + rr * math.sin(t)
        r = rng.uniform(*r_range)
        d.ellipse([x - r, y - r, x + r, y + r], fill=_jitter(colour, rng, 10) + (alpha,))


def draw_leukocyte(cls: str, size: int, rng: random.Random, shift: bool = False) -> Image.Image:
    tint = (240, 214, 222) if not shift else (236, 220, 200)  # acquisition shift: yellowish cast
    img = _background(size, rng, tint)
    d = ImageDraw.Draw(img, "RGBA")
    half = size / 2
    mean_r, sd_r = CELL_RADIUS[cls]
    scale = 0.85 if shift else 1.0  # shift: lower magnification
    R = max(6.0, rng.gauss(mean_r, sd_r) * half * scale)
    cx, cy = half + rng.uniform(-0.12, 0.12) * half, half + rng.uniform(-0.12, 0.12) * half
    cyto = _jitter(CYTO[cls], rng, 12)
    ex, ey = R * rng.uniform(0.92, 1.08), R * rng.uniform(0.92, 1.08)
    d.ellipse([cx - ex, cy - ey, cx + ex, cy + ey], fill=cyto + (255,))
    nuc = _jitter(NUCLEUS, rng, 16)

    if cls == "Lymphocyte":
        rn = R * rng.uniform(0.74, 0.86)
        ox, oy = rng.uniform(-0.12, 0.12) * R, rng.uniform(-0.12, 0.12) * R
        d.ellipse([cx + ox - rn, cy + oy - rn * rng.uniform(0.9, 1.0), cx + ox + rn, cy + oy + rn], fill=nuc + (255,))
        _dots(d, cx, cy, R, rng.randint(0, 4), (0.8, 1.4), (150, 90, 170), rng)
    elif cls == "Monocyte":
        rn = R * rng.uniform(0.58, 0.68)
        ang = rng.uniform(0, 2 * math.pi)
        d.ellipse([cx - rn, cy - rn * 0.85, cx + rn, cy + rn * 0.85], fill=nuc + (255,))
        bx, by = cx + 0.75 * rn * math.cos(ang), cy + 0.75 * rn * math.sin(ang)
        rb = rn * rng.uniform(0.45, 0.6)  # the "bite" that makes the kidney shape
        d.ellipse([bx - rb, by - rb, bx + rb, by + rb], fill=cyto + (255,))
        _dots(d, cx, cy, R, rng.randint(4, 12), (1.2, 2.4), (230, 232, 240), rng, alpha=170)  # vacuoles
        _dots(d, cx, cy, R, rng.randint(6, 16), (0.6, 1.1), (170, 150, 200), rng)
    elif cls in ("Neutrophil", "Eosinophil", "Basophil"):
        lobes = rng.randint(3, 5) if cls == "Neutrophil" else 2
        rl = R * (rng.uniform(0.22, 0.28) if cls == "Neutrophil" else rng.uniform(0.30, 0.38))
        ang0 = rng.uniform(0, 2 * math.pi)
        pts: List[Tuple[float, float]] = []
        for i in range(lobes):
            a = ang0 + i * (2 * math.pi / lobes) * rng.uniform(0.75, 1.0)
            dist = R * rng.uniform(0.35, 0.5)
            pts.append((cx + dist * math.cos(a), cy + dist * math.sin(a)))
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            d.line([x0, y0, x1, y1], fill=nuc + (255,), width=max(1, int(R * 0.09)))  # filaments
        for x, y in pts:
            rr = rl * rng.uniform(0.85, 1.15)
            d.ellipse([x - rr, y - rr * rng.uniform(0.8, 1.0), x + rr, y + rr], fill=nuc + (255,))
        if cls == "Neutrophil":
            _dots(d, cx, cy, R, rng.randint(40, 80), (0.6, 1.1), (204, 174, 214), rng, alpha=200)
        elif cls == "Eosinophil":
            _dots(d, cx, cy, R, rng.randint(26, 48), (1.3, 2.3), (238, 122, 70), rng)
        else:  # Basophil: coarse dark granules over the whole cell, obscuring the nucleus
            _dots(d, cx, cy, R, rng.randint(34, 64), (1.4, 2.8), (52, 32, 112), rng)
    else:
        raise ValueError(cls)

    img = img.rotate(rng.uniform(0, 360), resample=Image.BILINEAR, fillcolor=_jitter(tint, rng, 6))
    blur = rng.uniform(0.0, 0.7) + (0.6 if shift else 0.0)
    if blur > 0.05:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    arr = np.asarray(img).astype(np.float32)
    gain = np.array([rng.uniform(0.92, 1.08), rng.uniform(0.92, 1.08), rng.uniform(0.92, 1.08)], dtype=np.float32)
    if shift:
        gain *= np.array([1.08, 1.0, 0.88], dtype=np.float32)  # warmer white balance
    arr = arr * gain + rng.uniform(-10, 10)
    arr += np.random.default_rng(rng.randint(0, 2**31 - 1)).normal(0, rng.uniform(2, 7), arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def write_folder(root: Path, counts: Dict[str, int], size: int, seed: int, shift: bool = False) -> None:
    rng = random.Random(seed)
    for cls, n in counts.items():
        d = root / cls
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            draw_leukocyte(cls, size, rng, shift).save(d / f"{cls.lower()}_{i:05d}.png")


def write_lisc(root: Path, counts: Dict[str, int], seed: int, field: Tuple[int, int] = (240, 192), cell: int = 72) -> None:
    """Full-field images with one leukocyte (or two, for ``mixt``) and a binary expert mask."""
    rng = random.Random(seed)
    names = {"neut": "Neutrophil", "lymp": "Lymphocyte", "mono": "Monocyte", "eosi": "Eosinophil", "baso": "Basophil"}
    img_root, mask_root = root / "Main_Dataset", root / "Ground_Truth"
    for folder, n in counts.items():
        (img_root / folder).mkdir(parents=True, exist_ok=True)
        (mask_root / folder).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            W, H = field
            base = _background(max(W, H), rng, (236, 210, 214)).crop((0, 0, W, H))
            mask = Image.new("L", (W, H), 0)
            classes = [names[folder]] if folder != "mixt" else rng.sample(list(names.values()), 2)
            for k, cls in enumerate(classes):
                patch = draw_leukocyte(cls, cell, rng)
                x = rng.randint(4, W - cell - 4) if k == 0 else rng.randint(4, W - cell - 4)
                y = rng.randint(4, H - cell - 4)
                circ = Image.new("L", (cell, cell), 0)
                ImageDraw.Draw(circ).ellipse([3, 3, cell - 3, cell - 3], fill=255)
                base.paste(patch, (x, y), circ)
                ImageDraw.Draw(mask).ellipse([x + 3, y + 3, x + cell - 3, y + cell - 3], fill=255)
            stem = f"{folder}_{i:03d}"
            base.save(img_root / folder / f"{stem}.bmp")
            mask.save(mask_root / folder / f"{stem}_expert.bmp")


def scaled(counts: Dict[str, int], scale: float, minimum: int = 8) -> Dict[str, int]:
    return {k: max(minimum, int(round(v * scale))) for k, v in counts.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/synthetic-raabin")
    ap.add_argument("--scale", type=float, default=0.125, help="fraction of the official Raabin counts")
    ap.add_argument("--size", type=int, default=96, help="stored crop size in pixels")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    out = Path(args.out)
    train, testa, testb = scaled(RAABIN_TRAIN, args.scale), scaled(RAABIN_TESTA, args.scale), scaled(TESTB, args.scale)
    print("Train ", train)
    print("TestA ", testa)
    print("TestB ", testb, "(acquisition shift)")
    print("LISC  ", LISC, "(full-field + masks)")
    write_folder(out / "Train", train, args.size, args.seed)
    write_folder(out / "TestA", testa, args.size, args.seed + 1)
    write_folder(out / "TestB", testb, args.size, args.seed + 2, shift=True)
    write_lisc(out / "LISC", LISC, args.seed + 3)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
