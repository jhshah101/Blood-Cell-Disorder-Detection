# Inference backend

A small FastAPI service that serves one trained checkpoint of the ViT-ECA-CF
model.

```bash
pip install -r ../../requirements.txt
set WBC_CHECKPOINT=..\..\runs\vit_eca_cf_alr\seed0\best.pt      # Windows
export WBC_CHECKPOINT=../../runs/vit_eca_cf_alr/seed0/best.pt   # Linux / macOS
uvicorn backend:app --host 127.0.0.1 --port 8000
```

| Endpoint      | Method | Purpose                                                                 |
|---------------|--------|-------------------------------------------------------------------------|
| `/health`     | GET    | liveness and whether the checkpoint is loaded                           |
| `/model-info` | GET    | backbone, class list (from the checkpoint), input size, validation score |
| `/predict`    | POST   | multipart `file` -> predicted class, softmax scores, top-k, latency     |

Design points that fix the previous version:

* the network is rebuilt from the configuration stored **inside** the checkpoint,
  so the served model is the trained model (the old server built a DenseNet-121
  while the training scripts trained other networks);
* class names come from the checkpoint and therefore follow `ImageFolder`'s
  alphabetical order (the old hard-coded list was in a different order, which
  silently mislabelled predictions);
* the input pipeline is identical to training (the old server normalised with
  0.5 / 0.5 while training used ImageNet statistics);
* invalid uploads return proper HTTP status codes instead of a 200 response
  with an `error` field, uploads are size-limited, CORS is restricted to the
  configured origins, and every answer carries a research-prototype disclaimer.
