"""Per-category metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score


def best_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """Threshold that maximises F1 (chosen on the calibration split, never the holdout)."""
    p, r, t = precision_recall_curve(labels, scores)
    f1 = 2 * p * r / np.maximum(p + r, 1e-12)
    return float(t[np.argmax(f1[:-1])]) if len(t) else 0.5


def image_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    pred = (scores >= threshold).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    return {
        "image_auroc": float(roc_auc_score(labels, scores)) if len(set(labels)) > 1 else float("nan"),
        "f1": float(f1_score(labels, pred, zero_division=0)),
        "accuracy": (tp + tn) / len(labels),
        "recall": tp / max(tp + fn, 1),          # defects caught
        "false_reject_rate": fp / max(fp + tn, 1),  # good parts wrongly failed
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def pixel_auroc(masks: np.ndarray, maps: np.ndarray, max_pixels: int = 4_000_000, seed: int = 0) -> float:
    """AUROC over pixels: how well the heatmap separates defect pixels from the rest."""
    y = masks.reshape(-1).astype(np.uint8)
    s = maps.reshape(-1).astype(np.float32)
    if y.max() == 0:
        return float("nan")
    if len(y) > max_pixels:  # subsample for memory; AUROC is unbiased under uniform sampling
        idx = np.random.default_rng(seed).choice(len(y), max_pixels, replace=False)
        y, s = y[idx], s[idx]
    return float(roc_auc_score(y, s))
