"""Download the exported ONNX models from this repo's GitHub release.

    python scripts/download_models.py               # latest tagged models
    python scripts/download_models.py --release v1.0.0
"""

import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

REPO = "alok-kul5/visual-defect-inspection"
DEST = Path(__file__).resolve().parent.parent / "models_onnx"

p = argparse.ArgumentParser()
p.add_argument("--release", default="v1.0.0")
a = p.parse_args()

url = f"https://github.com/{REPO}/releases/download/{a.release}/models_onnx.zip"
print(f"downloading {url}")
with urllib.request.urlopen(url) as r:
    data = r.read()
DEST.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(data)) as z:
    for name in z.namelist():
        target = (DEST / name).resolve()
        if DEST.resolve() not in target.parents and target != DEST.resolve():
            raise SystemExit(f"refusing to extract {name} outside {DEST}")
    z.extractall(DEST)
print(f"extracted {len(data) / 1e6:.0f} MB to {DEST}")
