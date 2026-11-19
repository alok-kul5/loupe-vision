"""The ONNX serving path, end to end, with tiny stand-in models."""

import base64
import io
import json

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image
from torch import nn

from defect.classifier import DefectClassifier
from defect.inference import Inspector


class TinyBackbone(nn.Module):
    """Same input/output contract as the PatchCore backbone: (B,3,224,224) -> (B,784,1536)."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1536, kernel_size=8, stride=8)

    def forward(self, x):
        f = self.conv(x)
        return f.permute(0, 2, 3, 1).reshape(x.shape[0], -1, 1536)


class TinyClassifier(nn.Module):
    """Same contract as the exported classifier: logits (B,2) and features (B,512,7,7)."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 512, kernel_size=32, stride=32)
        self.fc = nn.Linear(512, 2)

    def forward(self, x):
        a = self.conv(x)
        return self.fc(a.mean(dim=(2, 3))), a


def export(model, path, outputs):
    torch.onnx.export(model.eval(), (torch.randn(1, 3, 224, 224),), str(path), input_names=["image"], output_names=outputs,
                      opset_version=17, dynamo=False)


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    torch.manual_seed(0)
    d = tmp_path_factory.mktemp("models")
    backbone = TinyBackbone()
    export(backbone, d / "patchcore_backbone.onnx", ["features"])
    cat = d / "widget"
    cat.mkdir()
    # Memory bank = features of a plain grey image, so grey is "normal" and noise is anomalous.
    grey = torch.zeros(1, 3, 224, 224)
    with torch.no_grad():
        bank = backbone(grey)[0].numpy()
    np.save(cat / "patchcore_bank.npy", bank.astype(np.float16))
    (cat / "patchcore.json").write_text(json.dumps({"threshold": 1.0}))
    c = TinyClassifier()
    export(c, cat / "classifier.onnx", ["logits", "features"])
    (cat / "classifier.json").write_text(json.dumps({"threshold": 0.5, "fc_weight": c.fc.weight.tolist()}))
    return d


def png(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


# ImageNet mean colour normalises to ~0, matching the all-zero "grey" used for the bank.
NORMAL = png(np.tile(np.array([124, 116, 104], np.uint8), (256, 256, 1)))
NOISY = png(np.random.default_rng(1).integers(0, 255, (256, 256, 3), dtype=np.uint8))


def test_patchcore_flags_anomaly_and_passes_normal(model_dir):
    insp = Inspector(model_dir)
    assert insp.categories == ["widget"]
    ok = insp.inspect(NORMAL, "widget", "patchcore")
    bad = insp.inspect(NOISY, "widget", "patchcore")
    assert ok.verdict == "pass" and bad.verdict == "fail"
    assert bad.score > ok.score
    assert bad.heatmap_png[:4] == b"\x89PNG"


def test_classifier_path_runs(model_dir):
    r = Inspector(model_dir).inspect(NOISY, "widget", "cnn")
    assert 0.0 <= r.score <= 1.0 and r.verdict in ("pass", "fail")


def test_unknown_category(model_dir):
    with pytest.raises(KeyError):
        Inspector(model_dir).inspect(NORMAL, "spaceship")


def test_api(model_dir, monkeypatch):
    import serve.app as appmod

    monkeypatch.setattr(appmod, "MODEL_DIR", model_dir)
    appmod.inspector.cache_clear()
    c = TestClient(appmod.app)
    assert c.get("/api/categories").json() == ["widget"]
    r = c.post("/api/inspect", files={"file": ("x.png", NOISY, "image/png")}, data={"category": "widget", "method": "patchcore"})
    assert r.status_code == 200
    body = r.json()
    assert body["verdict"] == "fail"
    assert base64.b64decode(body["heatmap_png_base64"])[:4] == b"\x89PNG"
    assert c.post("/api/inspect", files={"file": ("x.png", b"not an image", "image/png")}, data={"category": "widget"}).status_code == 400
    assert c.post("/api/inspect", files={"file": ("x.png", NOISY, "image/png")}, data={"category": "nope"}).status_code == 404
    assert c.post("/api/inspect", files={"file": ("x.png", NOISY, "image/png")},
                  data={"category": "widget", "method": "magic"}).status_code == 422
    assert "Visual defect inspection" in c.get("/").text
    appmod.inspector.cache_clear()


def test_real_classifier_exports_with_matching_outputs(tmp_path):
    """The real ResNet-18 head (random weights) survives ONNX export unchanged."""
    import onnxruntime as ort

    torch.manual_seed(0)
    m = DefectClassifier(pretrained=False).eval()
    export(m, tmp_path / "c.onnx", ["logits", "features"])
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        ref_logits, ref_feats = m(x)
    logits, feats = ort.InferenceSession(str(tmp_path / "c.onnx")).run(None, {"image": x.numpy()})
    assert np.abs(logits - ref_logits.numpy()).max() < 1e-3
    assert np.abs(feats - ref_feats.numpy()).max() < 1e-3
