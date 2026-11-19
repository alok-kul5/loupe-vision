"""Fit and evaluate both methods on every category, logging to MLflow.

    python -m defect.train                       # all 15 categories
    python -m defect.train --categories bottle screw
    mlflow ui --backend-store-uri sqlite:///mlflow.db   # browse the runs
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from . import classifier as clf  # noqa: E402
from .data import CATEGORIES, ImageSet, denormalize, list_train, split_test, train_augment  # noqa: E402
from .metrics import best_f1_threshold, image_metrics, pixel_auroc  # noqa: E402
from .patchcore import PatchCore  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
RESULTS = ROOT / "results"


def run_batches(fn, samples, batch_size=16):
    """Apply fn(images) -> (scores, maps) over a split; also return labels and masks."""
    loader = DataLoader(ImageSet(samples, with_masks=True), batch_size=batch_size, num_workers=0)
    scores, maps, labels, masks = [], [], [], []
    for x, y, m in loader:
        s, a = fn(x)
        scores.append(s.cpu().numpy())
        maps.append(a.cpu().numpy().astype(np.float16))
        labels.append(y.numpy())
        masks.append(m.squeeze(1).numpy().astype(np.uint8))
    return np.concatenate(scores), np.concatenate(maps), np.concatenate(labels), np.concatenate(masks)


def evaluate(fn, calib, hold) -> tuple[dict, dict]:
    cs, _, cl, _ = run_batches(fn, calib)
    thr = best_f1_threshold(cl, cs)
    hs, hm, hl, hk = run_batches(fn, hold)
    m = image_metrics(hl, hs, thr)
    m["pixel_auroc"] = pixel_auroc(hk, hm)
    m["threshold"] = thr
    return m, {"scores": hs, "maps": hm, "labels": hl, "masks": hk}


def example_figure(samples, out, title: str, path: Path, n: int = 4) -> None:
    """Defective holdout images with ground truth and each method's heatmap."""
    idx = [i for i, s in enumerate(samples) if s.label == 1][:n]
    cols = ["image", "ground truth", *out.keys()]
    fig, axes = plt.subplots(len(idx), len(cols), figsize=(2.4 * len(cols), 2.4 * len(idx)), squeeze=False)
    ds = ImageSet(samples, with_masks=True)
    for r, i in enumerate(idx):
        img, _, mask = ds[i]
        rgb = denormalize(img)
        axes[r][0].imshow(rgb)
        axes[r][1].imshow(mask[0], cmap="gray")
        for c, res in enumerate(out.values(), start=2):
            axes[r][c].imshow(rgb)
            axes[r][c].imshow(res["maps"][i].astype(np.float32), cmap="jet", alpha=0.45)
        axes[r][0].set_ylabel(samples[i].defect_type, fontsize=9)
    for c, name in enumerate(cols):
        axes[0][c].set_title(name, fontsize=10)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def run_category(cat: str, args, device: str) -> dict:
    train = list_train(cat)
    calib, hold = split_test(cat)
    out_dir = MODELS / cat
    out_dir.mkdir(parents=True, exist_ok=True)
    results, raw = {}, {}

    # --- PatchCore: only defect-free images ---------------------------------
    with mlflow.start_run(run_name=f"patchcore-{cat}", nested=True):
        mlflow.log_params({"method": "patchcore", "category": cat, "backbone": "wide_resnet50_2", "layers": "layer2+layer3",
                           "coreset_ratio": args.coreset, "image_size": 224, "train_images": len(train)})
        t0 = time.time()
        pc = PatchCore(device=device, coreset_ratio=args.coreset)
        info = pc.fit(DataLoader(ImageSet(train), batch_size=16, num_workers=0))
        fit_s = time.time() - t0
        m, raw["PatchCore"] = evaluate(pc.predict, calib, hold)
        m.update(fit_seconds=round(fit_s, 1), bank_size=info["bank_size"])
        mlflow.log_metrics({k: v for k, v in m.items() if isinstance(v, int | float) and not np.isnan(v)})
        pc.save(out_dir / "patchcore_bank.npy")
        (out_dir / "patchcore_threshold.json").write_text(json.dumps({"threshold": m["threshold"]}))
        results["patchcore"] = m
        del pc
        torch.cuda.empty_cache()

    # --- Classifier: train goods + calibration-half defects ------------------
    with mlflow.start_run(run_name=f"cnn-{cat}", nested=True):
        train_set = train + [s for s in calib if s.label == 1]
        mlflow.log_params({"method": "resnet18_finetune", "category": cat, "epochs": args.epochs, "image_size": 224,
                           "train_good": len(train), "train_defect": sum(s.label for s in train_set)})
        torch.manual_seed(0)
        model = clf.DefectClassifier()
        loader = clf.balanced_loader(ImageSet(train_set, transform=train_augment(cat)), [s.label for s in train_set])
        t0 = time.time()

        def log(epoch, loss, acc):
            mlflow.log_metrics({"train_loss": loss, "train_acc": acc}, step=epoch)

        model = clf.train_classifier(model, loader, args.epochs, device, log=log)
        fit_s = time.time() - t0

        def fn(x):
            return clf.predict(model, x.to(device))

        # Threshold: calibration goods + the defects it trained on would be biased,
        # so the CNN uses its natural 0.5 decision boundary.
        hs, hm, hl, hk = run_batches(fn, hold)
        m = image_metrics(hl, hs, 0.5)
        m.update(pixel_auroc=pixel_auroc(hk, hm), threshold=0.5, fit_seconds=round(fit_s, 1))
        raw["CNN + Grad-CAM"] = {"scores": hs, "maps": hm, "labels": hl, "masks": hk}
        mlflow.log_metrics({k: v for k, v in m.items() if isinstance(v, int | float) and not np.isnan(v)})
        torch.save(model.state_dict(), out_dir / "classifier.pt")
        results["cnn"] = m

    fig = RESULTS / "examples" / f"{cat}.png"
    fig.parent.mkdir(parents=True, exist_ok=True)
    example_figure(hold, raw, cat, fig)
    mlflow.log_artifact(str(fig))
    return results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--categories", nargs="*", default=CATEGORIES)
    # 1% keeps nearly all of the accuracy of 10% (per the paper) at a tenth of the memory and search time.
    p.add_argument("--coreset", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=12)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    RESULTS.mkdir(exist_ok=True)

    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("visual-defect-inspection")
    path = RESULTS / "metrics.json"
    all_results = json.loads(path.read_text()) if path.exists() else {}
    with mlflow.start_run(run_name=f"benchmark-{time.strftime('%Y%m%d-%H%M')}"):
        mlflow.log_params({"categories": ",".join(args.categories), "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu"})
        for cat in args.categories:
            t0 = time.time()
            all_results[cat] = run_category(cat, args, device)
            pcm, cnm = all_results[cat]["patchcore"], all_results[cat]["cnn"]
            print(f"{cat:<11} PatchCore AUROC {pcm['image_auroc']:.3f} / px {pcm['pixel_auroc']:.3f}   "
                  f"CNN AUROC {cnm['image_auroc']:.3f} / px {cnm['pixel_auroc']:.3f}   ({time.time() - t0:.0f}s)", flush=True)
            path.write_text(json.dumps(all_results, indent=2))
        for method in ("patchcore", "cnn"):
            for key in ("image_auroc", "pixel_auroc", "f1"):
                vals = [r[method][key] for r in all_results.values() if not np.isnan(r[method][key])]
                mlflow.log_metric(f"mean_{method}_{key}", float(np.mean(vals)))
        mlflow.log_artifact(str(path))


if __name__ == "__main__":
    main()
