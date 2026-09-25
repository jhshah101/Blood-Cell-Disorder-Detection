"""FastAPI inference server for a ViT-ECA-CF checkpoint.

The server rebuilds the network from the checkpoint's own configuration, uses
the class list stored in the checkpoint (so the label order can never drift
from ``ImageFolder``'s alphabetical order), and applies exactly the input
pipeline used in training (resize -> [0, 1] tensor; normalisation lives inside
the model).

Environment variables
---------------------
WBC_CHECKPOINT     path to best.pt      (default: runs/vit_eca_cf_alr/seed0/best.pt)
WBC_DEVICE         auto | cpu | cuda    (default: auto)
WBC_CORS_ORIGINS   comma-separated list (default: the local Vite / CRA dev origins)
WBC_MAX_UPLOAD_MB  upload size limit    (default: 20)

Run
---
    uvicorn backend:app --host 127.0.0.1 --port 8000        (from Software/Backend)

This is a research prototype.  Its outputs are softmax scores of an image
classifier trained on a public dataset and are not a clinical device output.
"""
from __future__ import annotations

import io
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wbc.checkpoint import load_checkpoint  # noqa: E402
from wbc.data import build_transforms  # noqa: E402
from wbc.utils import get_device  # noqa: E402

log = logging.getLogger("wbc.backend")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

DISCLAIMER = (
    "Research prototype. Scores are softmax outputs of an image classifier trained on the public "
    "Raabin-WBC dataset; they are not calibrated probabilities and this service is not a medical device."
)
DEFAULT_ORIGINS = "http://localhost:8080,http://127.0.0.1:8080,http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000"


def _settings() -> Dict[str, Any]:
    return {
        "checkpoint": Path(os.getenv("WBC_CHECKPOINT", ROOT / "runs" / "vit_eca_cf_alr" / "seed0" / "best.pt")),
        "device": os.getenv("WBC_DEVICE", "auto"),
        "cors_origins": [o.strip() for o in os.getenv("WBC_CORS_ORIGINS", DEFAULT_ORIGINS).split(",") if o.strip()],
        "max_upload_bytes": int(float(os.getenv("WBC_MAX_UPLOAD_MB", "20")) * 1024 * 1024),
    }


SETTINGS = _settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    ckpt = SETTINGS["checkpoint"]
    app.state.model = None
    if not ckpt.exists():
        log.error("Checkpoint %s not found. Train a model (python scripts/train.py --config configs/default.yaml) "
                  "or set WBC_CHECKPOINT. /predict will answer 503 until then.", ckpt)
    else:
        device = get_device(SETTINGS["device"])
        model, cfg, classes, payload = load_checkpoint(ckpt, device)
        app.state.model = model
        app.state.cfg = cfg
        app.state.classes = classes
        app.state.device = device
        app.state.transform = build_transforms(cfg.data.img_size)
        app.state.checkpoint_meta = {
            "path": str(ckpt),
            "epoch": payload.get("epoch"),
            "validation": {k: payload["val_metrics"][k] for k in ("accuracy", "macro_f1", "balanced_accuracy", "minority_f1", "majority_f1") if k in payload.get("val_metrics", {})},
        }
        log.info("Loaded %s (%s, %d classes) on %s", ckpt, cfg.model.backbone, len(classes), device)
    yield


app = FastAPI(title="WBC classifier (ViT-ECA-CF + ALR)", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=SETTINGS["cors_origins"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _require_model():
    if getattr(app.state, "model", None) is None:
        raise HTTPException(status_code=503, detail="Model checkpoint not loaded; see server log.")


@app.get("/")
async def root() -> Dict[str, Any]:
    return {"service": "WBC classifier", "endpoints": ["/health", "/model-info", "/predict"], "docs": "/docs", "disclaimer": DISCLAIMER}


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"status": "ok", "model_loaded": getattr(app.state, "model", None) is not None}


@app.get("/model-info")
async def model_info() -> Dict[str, Any]:
    _require_model()
    cfg = app.state.cfg
    return {
        "experiment": cfg.experiment,
        "backbone": cfg.model.backbone,
        "use_eca": cfg.model.use_eca,
        "use_color_features": cfg.model.use_color_features,
        "loss": cfg.loss.name,
        "classes": app.state.classes,
        "img_size": cfg.data.img_size,
        "normalization": cfg.data.normalization,
        "augmentation": cfg.data.augment,
        "checkpoint": app.state.checkpoint_meta,
        "device": str(app.state.device),
        "disclaimer": DISCLAIMER,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict[str, Any]:
    _require_model()
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail=f"Unsupported content type {file.content_type!r}; upload an image.")
    data = await file.read()
    if len(data) > SETTINGS["max_upload_bytes"]:
        raise HTTPException(status_code=413, detail=f"File larger than {SETTINGS['max_upload_bytes'] // (1024 * 1024)} MB.")
    try:
        image = Image.open(io.BytesIO(data))
        image.verify()
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {exc}") from exc

    x = app.state.transform(image).unsqueeze(0).to(app.state.device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        probs = app.state.model(x).float().softmax(dim=1)[0].cpu()
    latency_ms = (time.perf_counter() - t0) * 1000.0

    classes: List[str] = app.state.classes
    order = torch.argsort(probs, descending=True)
    top_idx = int(order[0])
    return {
        "predicted_class": classes[top_idx],
        "confidence": float(probs[top_idx]),
        "probabilities": {classes[i]: float(probs[i]) for i in range(len(classes))},
        "top_k": [{"class": classes[int(i)], "probability": float(probs[int(i)])} for i in order],
        "image": {"filename": file.filename, "width": image.width, "height": image.height},
        "model": app.state.cfg.experiment,
        "latency_ms": latency_ms,
        "disclaimer": DISCLAIMER,
    }
