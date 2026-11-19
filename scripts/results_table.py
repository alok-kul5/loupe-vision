"""Print the README results section from results/metrics.json, and build docs/examples.png.

    python scripts/results_table.py > results/RESULTS.md
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
m = json.loads((ROOT / "results" / "metrics.json").read_text())
cats = sorted(m)


def f(v, pct=False):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    return f"{100 * v:.0f}%" if pct else f"{v:.3f}"


def mean(method, key):
    vals = [m[c][method][key] for c in cats if not np.isnan(m[c][method][key])]
    return float(np.mean(vals))


lines = [
    "## Results",
    "",
    f"Holdout half of the MVTec AD test set, {len(cats)} categories, RTX 5050 laptop GPU.",
    "",
    "| | PatchCore (no defects seen) | ResNet-18 + Grad-CAM (trained on defects) |",
    "|---|---:|---:|",
    f"| **Mean image AUROC** | **{f(mean('patchcore', 'image_auroc'))}** | {f(mean('cnn', 'image_auroc'))} |",
    f"| **Mean pixel AUROC** (defect localisation) | **{f(mean('patchcore', 'pixel_auroc'))}** | {f(mean('cnn', 'pixel_auroc'))} |",
    f"| Mean F1 at the chosen threshold | {f(mean('patchcore', 'f1'))} | {f(mean('cnn', 'f1'))} |",
    f"| Defects caught (recall) | {f(mean('patchcore', 'recall'), True)} | {f(mean('cnn', 'recall'), True)} |",
    f"| Good parts wrongly rejected | {f(mean('patchcore', 'false_reject_rate'), True)} | {f(mean('cnn', 'false_reject_rate'), True)} |",
    "",
    "<details><summary>Per category</summary>",
    "",
    "| Category | PatchCore image | PatchCore pixel | PatchCore F1 | CNN image | CNN pixel | CNN F1 |",
    "|---|---:|---:|---:|---:|---:|---:|",
]
for c in cats:
    p, n = m[c]["patchcore"], m[c]["cnn"]
    lines.append(f"| {c.replace('_', ' ')} | {f(p['image_auroc'])} | {f(p['pixel_auroc'])} | {f(p['f1'])} | "
                 f"{f(n['image_auroc'])} | {f(n['pixel_auroc'])} | {f(n['f1'])} |")
lines += ["", "</details>", ""]

# Where each method is weakest, stated plainly.
worst_pc = sorted(cats, key=lambda c: m[c]["patchcore"]["image_auroc"])[:3]
cnn_wins = [c for c in cats if m[c]["cnn"]["image_auroc"] > m[c]["patchcore"]["image_auroc"] + 0.005]
weak = ", ".join(f"{c} ({f(m[c]['patchcore']['image_auroc'])})" for c in worst_pc)
lines.append(f"**Where PatchCore is weakest:** {weak}.")
if cnn_wins:
    lines.append(f"The classifier beats it on image AUROC in: {', '.join(cnn_wins)}, all categories where it trained on examples of the same defect types.")
else:
    lines.append("The classifier never beats it on image AUROC, despite having trained on real defects.")
print("\n".join(lines))

# Collage of example figures for the README.
figs = [ROOT / "results" / "examples" / f"{c}.png" for c in ("screw", "hazelnut", "carpet") if (ROOT / "results" / "examples" / f"{c}.png").exists()]
if figs:
    ims = [Image.open(p).convert("RGB") for p in figs]
    w = max(i.width for i in ims)
    out = Image.new("RGB", (w, sum(i.height for i in ims)), "white")
    y = 0
    for i in ims:
        out.paste(i, (0, y))
        y += i.height
    (ROOT / "docs").mkdir(exist_ok=True)
    out.save(ROOT / "docs" / "examples.png", optimize=True)
