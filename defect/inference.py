"""ONNX Runtime inference: photo in, pass/fail and heatmap out. No PyTorch needed."""

from __future__ import annotations

import io
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from scipy.ndimage import gaussian_filter, zoom

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
RESIZE, CROP = 256, 224


def preprocess(img: Image.Image) -> tuple[np.ndarray, Image.Image]:
    """Same as training: shorter side to 256, centre crop 224, ImageNet normalisation."""
    img = img.convert("RGB")
    w, h = img.size
    s = RESIZE / min(w, h)
    img = img.resize((max(RESIZE, round(w * s)), max(RESIZE, round(h * s))), Image.BILINEAR)
    w, h = img.size
    left, top = (w - CROP) // 2, (h - CROP) // 2
    crop = img.crop((left, top, left + CROP, top + CROP))
    x = (np.asarray(crop, dtype=np.float32) / 255.0 - MEAN) / STD
    return x.transpose(2, 0, 1)[None].astype(np.float32), crop


def overlay(crop: Image.Image, heat: np.ndarray) -> bytes:
    """Blend a [0,1] heatmap over the image with a jet-like colour ramp; returns PNG bytes."""
    h = np.clip(heat, 0, 1)
    r = np.clip(1.5 - np.abs(4 * h - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * h - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * h - 1), 0, 1)
    color = (np.stack([r, g, b], -1) * 255).astype(np.uint8)
    blended = (0.55 * np.asarray(crop, dtype=np.float32) + 0.45 * color).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(blended).save(buf, format="PNG")
    return buf.getvalue()


@dataclass
class Result:
    category: str
    method: str
    verdict: str          # "pass" | "fail"
    score: float
    threshold: float
    heatmap_png: bytes
    latency_ms: float


class Inspector:
    def __init__(self, model_dir: Path) -> None:
        self.dir = Path(model_dir)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
        self._ort = lambda path: ort.InferenceSession(str(path), opts, providers=providers)
        self.backbone = self._ort(self.dir / "patchcore_backbone.onnx")
        self.categories = sorted(p.name for p in self.dir.iterdir() if (p / "classifier.onnx").exists())
        self._cache: dict = {}

    def _category(self, cat: str) -> dict:
        if cat not in self.categories:
            raise KeyError(f"unknown category {cat!r}; available: {self.categories}")
        if cat not in self._cache:
            d = self.dir / cat
            clf_meta = json.loads((d / "classifier.json").read_text())
            bank = np.load(d / "patchcore_bank.npy").astype(np.float32)
            self._cache[cat] = {
                "bank": bank, "bank_sq": (bank**2).sum(1),
                "pc_threshold": json.loads((d / "patchcore.json").read_text())["threshold"],
                "clf": self._ort(d / "classifier.onnx"),
                "fc": np.array(clf_meta["fc_weight"], dtype=np.float32), "clf_threshold": clf_meta["threshold"],
            }
        return self._cache[cat]

    def patchcore(self, x: np.ndarray, m: dict) -> tuple[float, np.ndarray]:
        f = self.backbone.run(None, {"image": x})[0][0]                  # (784, 1536)
        d2 = (f**2).sum(1)[:, None] + m["bank_sq"][None] - 2 * f @ m["bank"].T
        d = np.sqrt(np.maximum(d2.min(1), 0)).reshape(28, 28)
        heat = gaussian_filter(zoom(d, CROP / 28, order=1), sigma=4)
        return float(d.max()), heat

    def classifier(self, x: np.ndarray, m: dict) -> tuple[float, np.ndarray]:
        logits, feats = m["clf"].run(None, {"image": x})
        p = np.exp(logits[0] - logits[0].max())
        prob = float(p[1] / p.sum())
        w = m["fc"][1] - m["fc"][0]
        cam = np.maximum(np.tensordot(w, feats[0], axes=1), 0)        # (7, 7)
        return prob, zoom(cam, CROP / cam.shape[0], order=1)

    def inspect(self, image_bytes: bytes, category: str, method: str = "patchcore") -> Result:
        t0 = time.perf_counter()
        m = self._category(category)
        x, crop = preprocess(Image.open(io.BytesIO(image_bytes)))
        if method == "patchcore":
            score, heat = self.patchcore(x, m)
            thr = m["pc_threshold"]
            # Scale the heatmap so the decision threshold sits in the middle of the colour range.
            vis = heat / (2 * thr)
        elif method == "cnn":
            score, heat = self.classifier(x, m)
            thr = m["clf_threshold"]
            # Grad-CAM shows where the evidence for "defect" is; fade it by how confident the model is.
            vis = heat / (heat.max() + 1e-8) * score
        else:
            raise ValueError("method must be 'patchcore' or 'cnn'")
        return Result(category, method, "fail" if score >= thr else "pass", score, thr, overlay(crop, vis),
                      round((time.perf_counter() - t0) * 1000, 1))
