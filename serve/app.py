"""FastAPI service: upload a photo of a part, get pass/fail and a heatmap.

    uvicorn serve.app:app --port 8000
"""

from __future__ import annotations

import base64
import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from defect.inference import Inspector

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(os.getenv("MODEL_DIR", ROOT / "models_onnx"))
MAX_BYTES = 10 * 1024 * 1024

app = FastAPI(title="Visual Defect Inspection", version="1.0.0")


@lru_cache
def inspector() -> Inspector:
    if not (MODEL_DIR / "patchcore_backbone.onnx").exists():
        raise HTTPException(503, f"No models in {MODEL_DIR}. Run `python scripts/download_models.py` or train and export them.")
    return Inspector(MODEL_DIR)


@app.get("/api/health")
def health():
    ready = (MODEL_DIR / "patchcore_backbone.onnx").exists()
    return {"status": "ok", "models_loaded": ready, "categories": inspector().categories if ready else []}


@app.get("/api/categories")
def categories():
    return inspector().categories


@app.post("/api/inspect")
async def inspect(file: UploadFile = File(...), category: str = Form(...), method: str = Form("patchcore")):
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "image larger than 10 MB")
    if method not in ("patchcore", "cnn"):
        raise HTTPException(422, "method must be 'patchcore' or 'cnn'")
    try:
        r = inspector().inspect(data, category, method)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except OSError as exc:  # PIL could not decode the upload
        raise HTTPException(400, f"could not read image: {exc}") from exc
    return {
        "category": r.category, "method": r.method, "verdict": r.verdict,
        "score": round(r.score, 4), "threshold": round(r.threshold, 4), "latency_ms": r.latency_ms,
        "heatmap_png_base64": base64.b64encode(r.heatmap_png).decode(),
    }


@app.get("/", include_in_schema=False)
def page():
    return FileResponse(Path(__file__).parent / "index.html")


# Page assets, sample parts (MVTec AD, CC BY-NC-SA 4.0) and the benchmark table.
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
