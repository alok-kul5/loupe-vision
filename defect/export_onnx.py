"""Export trained models to ONNX for serving without PyTorch.

    python -m defect.export_onnx

Writes to models_onnx/:
  patchcore_backbone.onnx      shared WideResNet-50 feature extractor (1,3,224,224) -> (1,784,1536)
  <category>/patchcore_bank.npy, patchcore.json (threshold)
  <category>/classifier.onnx   (1,3,224,224) -> logits (1,2), features (1,512,7,7)
  <category>/classifier.json   fc weights for gradient-free Grad-CAM, threshold
Each export is checked against PyTorch on a random input.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from .classifier import DefectClassifier
from .data import CATEGORIES
from .patchcore import FeatureExtractor

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "models"
DST = ROOT / "models_onnx"


def _export(model: torch.nn.Module, path: Path, outputs: list[str]) -> None:
    x = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, (x,), str(path), input_names=["image"], output_names=outputs,
                      dynamic_axes={"image": {0: "batch"}, **{o: {0: "batch"} for o in outputs}}, opset_version=17, dynamo=False)
    with torch.no_grad():
        ref = model(x)
    ref = ref if isinstance(ref, tuple) else (ref,)
    got = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, {"image": x.numpy()})
    for r, g in zip(ref, got, strict=True):
        err = float(np.abs(r.numpy() - g).max())
        assert err < 1e-3, f"{path.name}: ONNX output differs from PyTorch by {err}"


def main() -> None:
    DST.mkdir(exist_ok=True)
    _export(FeatureExtractor().eval(), DST / "patchcore_backbone.onnx", ["features"])
    exported = []
    for cat in CATEGORIES:
        src = SRC / cat
        if not (src / "classifier.pt").exists():
            continue
        out = DST / cat
        out.mkdir(exist_ok=True)
        shutil.copy(src / "patchcore_bank.npy", out / "patchcore_bank.npy")
        thr = json.loads((src / "patchcore_threshold.json").read_text())["threshold"]
        (out / "patchcore.json").write_text(json.dumps({"threshold": thr}))

        model = DefectClassifier(pretrained=False)
        model.load_state_dict(torch.load(src / "classifier.pt", map_location="cpu"))
        model.eval()
        _export(model, out / "classifier.onnx", ["logits", "features"])
        (out / "classifier.json").write_text(json.dumps({"threshold": 0.5, "fc_weight": model.fc.weight.detach().tolist()}))
        exported.append(cat)
    print(f"exported backbone + {len(exported)} categories to {DST}: {', '.join(exported)}")


if __name__ == "__main__":
    main()
