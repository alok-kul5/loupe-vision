"""Unit tests that need neither the dataset nor a GPU."""

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import transforms as T

from defect import classifier as clf
from defect.data import MEAN, STD
from defect.inference import overlay, preprocess
from defect.metrics import best_f1_threshold, image_metrics, pixel_auroc
from defect.patchcore import gaussian_blur, greedy_coreset, nearest_distance


def test_coreset_covers_clusters():
    torch.manual_seed(0)
    centers = torch.tensor([[0.0, 0.0], [10.0, 10.0], [-10.0, 10.0]])
    pts = torch.cat([c + 0.1 * torch.randn(200, 2) for c in centers])
    idx = greedy_coreset(pts, ratio=3 / 600, proj_dim=2)
    chosen = pts[idx]
    # k-center greedy picks one point from each far-apart cluster.
    assert all(torch.cdist(c[None], chosen).min() < 1.0 for c in centers)


def test_nearest_distance_matches_bruteforce():
    q, b = torch.randn(50, 8), torch.randn(30, 8)
    expected = torch.cdist(q, b).min(1).values
    assert torch.allclose(nearest_distance(q, b, chunk=7), expected)


def test_gaussian_blur_preserves_mass_and_shape():
    m = torch.zeros(1, 32, 32)
    m[0, 16, 16] = 1.0
    out = gaussian_blur(m, sigma=2.0)
    assert out.shape == m.shape
    assert out.sum().item() == pytest.approx(1.0, abs=1e-3)
    assert out[0, 16, 16] == out.max()


def test_cam_equals_gradcam_for_gap_linear_head():
    torch.manual_seed(0)
    model = clf.DefectClassifier(pretrained=False).eval()
    x = torch.randn(2, 3, 224, 224)
    grad_cam = model.gradcam(x.clone())
    _, cam = clf.predict(model, x)
    assert torch.allclose(grad_cam, cam, atol=1e-4)


def test_metrics():
    labels = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    thr = best_f1_threshold(labels, scores)
    m = image_metrics(labels, scores, thr)
    assert m["image_auroc"] == 1.0 and m["f1"] == 1.0 and m["false_reject_rate"] == 0.0
    masks = np.zeros((2, 4, 4))
    masks[1, :2, :2] = 1
    maps = masks + 0.01 * np.random.default_rng(0).random(masks.shape)
    assert pixel_auroc(masks, maps) == 1.0
    assert np.isnan(pixel_auroc(np.zeros((1, 4, 4)), maps[:1]))


def test_numpy_preprocessing_matches_training_transform():
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 255, (300, 400, 3), dtype=np.uint8))
    ref = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(MEAN, STD)])(img).numpy()
    got, crop = preprocess(img)
    assert got.shape == (1, 3, 224, 224) and crop.size == (224, 224)
    # torchvision antialiases slightly differently from PIL; allow a small tolerance.
    assert np.abs(got[0] - ref).mean() < 0.02


def test_overlay_is_png():
    png = overlay(Image.new("RGB", (224, 224)), np.random.default_rng(0).random((224, 224)))
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
