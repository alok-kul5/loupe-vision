"""MVTec AD loading and the train / calibration / holdout split.

MVTec's own training set has only defect-free ("good") images. To compare an
anomaly detector with a supervised classifier fairly, each category's test
set is split in half, stratified by defect type, with a fixed seed:

* calibration half: used to pick decision thresholds, and as the defect
  examples the supervised classifier is allowed to train on;
* holdout half: used only for the final numbers of *both* methods.

PatchCore never sees a defect during fitting; the classifier sees the
calibration defects. Both are scored on the same holdout images.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

DATA_ROOT = Path(os.getenv("MVTEC_ROOT", Path(__file__).resolve().parent.parent / "data" / "mvtec"))
CATEGORIES = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut",
              "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"]
TEXTURES = {"carpet", "grid", "leather", "tile", "wood"}
RESIZE, CROP = 256, 224
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int          # 0 = good, 1 = defect
    defect_type: str    # "good" or e.g. "scratch"
    mask_path: Path | None


def image_transform() -> T.Compose:
    return T.Compose([T.Resize(RESIZE), T.CenterCrop(CROP), T.ToTensor(), T.Normalize(MEAN, STD)])


def mask_transform() -> T.Compose:
    return T.Compose([T.Resize(RESIZE, interpolation=T.InterpolationMode.NEAREST), T.CenterCrop(CROP), T.ToTensor()])


def train_augment(category: str) -> T.Compose:
    """Light augmentation for the classifier; textures can be flipped freely, objects only mildly."""
    flips = [T.RandomHorizontalFlip(), T.RandomVerticalFlip()] if category in TEXTURES else [T.RandomHorizontalFlip()]
    return T.Compose([T.Resize(RESIZE), T.RandomCrop(CROP), *flips,
                      T.ColorJitter(0.1, 0.1, 0.05), T.ToTensor(), T.Normalize(MEAN, STD)])


def list_train(category: str) -> list[Sample]:
    d = DATA_ROOT / category / "train" / "good"
    return [Sample(p, 0, "good", None) for p in sorted(d.glob("*.png"))]


def list_test(category: str) -> list[Sample]:
    out = []
    for d in sorted((DATA_ROOT / category / "test").iterdir()):
        for p in sorted(d.glob("*.png")):
            mask = None
            if d.name != "good":
                mask = DATA_ROOT / category / "ground_truth" / d.name / f"{p.stem}_mask.png"
            out.append(Sample(p, int(d.name != "good"), d.name, mask))
    return out


def split_test(category: str, seed: int = 0) -> tuple[list[Sample], list[Sample]]:
    """(calibration, holdout), stratified by defect type."""
    rng = random.Random(f"{category}-{seed}")
    calib, hold = [], []
    by_type: dict[str, list[Sample]] = {}
    for s in list_test(category):
        by_type.setdefault(s.defect_type, []).append(s)
    for _, items in sorted(by_type.items()):
        items = items[:]
        rng.shuffle(items)
        half = len(items) // 2
        calib += items[:half]
        hold += items[half:]
    return calib, hold


class ImageSet(Dataset):
    def __init__(self, samples: list[Sample], transform=None, with_masks: bool = False):
        self.samples = samples
        self.transform = transform or image_transform()
        self.with_masks = with_masks
        self.mtf = mask_transform()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        img = self.transform(Image.open(s.path).convert("RGB"))
        if not self.with_masks:
            return img, s.label
        if s.mask_path is not None and s.mask_path.exists():
            mask = (self.mtf(Image.open(s.mask_path).convert("L")) > 0.5).float()
        else:
            mask = torch.zeros(1, CROP, CROP)
        return img, s.label, mask


def denormalize(t: torch.Tensor) -> np.ndarray:
    """CHW normalized tensor -> HWC uint8 image."""
    x = t.detach().cpu().numpy().transpose(1, 2, 0) * np.array(STD) + np.array(MEAN)
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)
