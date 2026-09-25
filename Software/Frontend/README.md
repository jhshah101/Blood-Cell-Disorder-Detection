# Frontend – WBC classifier demo

Single-page React (Vite + TypeScript + shadcn/ui) client for the inference
backend in `../Backend`.

```bash
npm ci                       # install (package-lock.json is committed)
cp .env.example .env.local   # set VITE_API_URL if the backend is not on localhost:8000
npm run dev                  # http://localhost:8080
npm run build                # production bundle in dist/
```

What the page does:

* polls `GET /health` and shows whether the backend and its checkpoint are up;
* reads `GET /model-info` and displays facts about the loaded checkpoint
  (backbone, number of classes, validation macro-F1, input size) instead of
  hard-coded claims;
* validates the selected file (type, size) before uploading it to
  `POST /predict`, shows errors as toasts, aborts on timeout, and renders the
  per-class softmax scores sorted and rounded to one decimal;
* carries the research-prototype disclaimer returned by the backend.

The earlier version advertised a "99.2 % accuracy rate", "50K+ images
analysed" and "FDA compliant"; none of that was supported by anything in the
repository and it has been removed.
