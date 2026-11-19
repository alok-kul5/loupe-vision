"""PatchCore (Roth et al., "Towards Total Recall in Industrial Anomaly Detection", CVPR 2022).

1. Extract mid-level features (layer2 + layer3) of an ImageNet-pretrained
   WideResNet-50 for every patch of every *defect-free* training image.
2. Keep a small, representative subset of them (greedy coreset).
3. At test time a patch's anomaly score is its distance to the nearest kept
   patch; the image score is the highest patch score, and the patch scores
   upsampled to image size are the anomaly heatmap.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2


class FeatureExtractor(nn.Module):
    """WideResNet-50 truncated after layer3; returns locally aware patch features (B, H*W, C)."""

    def __init__(self) -> None:
        super().__init__()
        m = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1)
        self.layer2, self.layer3 = m.layer2, m.layer3
        self.pool = nn.AvgPool2d(3, stride=1, padding=1)  # neighbourhood aggregation (p=3)
        self.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f2 = self.layer2(self.stem(x))
        f3 = self.layer3(f2)
        f2, f3 = self.pool(f2), self.pool(f3)
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        f = torch.cat([f2, f3], dim=1)                 # (B, 1536, 28, 28) at 224px
        return f.permute(0, 2, 3, 1).reshape(f.shape[0], -1, f.shape[1])


def greedy_coreset(features: torch.Tensor, ratio: float, proj_dim: int = 128, seed: int = 0) -> torch.Tensor:
    """Indices of a k-center-greedy coreset, computed in a random low-dim projection for speed."""
    n = features.shape[0]
    k = max(1, int(n * ratio))
    g = torch.Generator(device=features.device).manual_seed(seed)
    proj = torch.randn(features.shape[1], proj_dim, device=features.device, generator=g) / math.sqrt(proj_dim)
    z = features @ proj
    selected = [int(torch.randint(n, (1,), generator=g, device=features.device))]
    min_d = torch.cdist(z, z[selected]).squeeze(1)
    for _ in range(k - 1):
        i = int(torch.argmax(min_d))
        selected.append(i)
        min_d = torch.minimum(min_d, torch.cdist(z, z[i : i + 1]).squeeze(1))
    return torch.tensor(selected, device=features.device)


def nearest_distance(queries: torch.Tensor, bank: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    out = []
    for i in range(0, queries.shape[0], chunk):
        out.append(torch.cdist(queries[i : i + chunk], bank).min(dim=1).values)
    return torch.cat(out)


def gaussian_blur(maps: torch.Tensor, sigma: float = 4.0) -> torch.Tensor:
    """(B, H, W) -> blurred, as in the paper's post-processing."""
    radius = int(3 * sigma)
    x = torch.arange(-radius, radius + 1, device=maps.device, dtype=maps.dtype)
    k = torch.exp(-(x**2) / (2 * sigma**2))
    k = k / k.sum()
    m = maps.unsqueeze(1)
    m = F.conv2d(F.pad(m, (radius, radius, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
    m = F.conv2d(F.pad(m, (0, 0, radius, radius), mode="reflect"), k.view(1, 1, -1, 1))
    return m.squeeze(1)


class PatchCore:
    def __init__(self, device: str = "cuda", coreset_ratio: float = 0.1) -> None:
        self.device = device
        self.coreset_ratio = coreset_ratio
        self.extractor = FeatureExtractor().to(device)
        self.bank: torch.Tensor | None = None
        self.grid: int | None = None

    @torch.no_grad()
    def fit(self, loader) -> dict:
        feats = []
        for x, _ in loader:
            f = self.extractor(x.to(self.device))
            self.grid = int(math.isqrt(f.shape[1]))
            feats.append(f.reshape(-1, f.shape[-1]))
        all_f = torch.cat(feats)
        idx = greedy_coreset(all_f, self.coreset_ratio)
        self.bank = all_f[idx].contiguous()
        return {"patches": int(all_f.shape[0]), "bank_size": int(self.bank.shape[0]), "feature_dim": int(all_f.shape[1])}

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Image scores (B,) and anomaly maps (B, H, W) at input resolution."""
        assert self.bank is not None, "call fit() first"
        x = x.to(self.device)
        f = self.extractor(x)
        b, n, c = f.shape
        d = nearest_distance(f.reshape(-1, c), self.bank).reshape(b, self.grid, self.grid)
        maps = F.interpolate(d.unsqueeze(1), size=x.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
        maps = gaussian_blur(maps)
        return d.reshape(b, -1).max(dim=1).values, maps

    def save(self, path) -> None:
        np.save(path, self.bank.cpu().numpy().astype(np.float16))

    def load(self, path) -> None:
        self.bank = torch.from_numpy(np.load(path).astype(np.float32)).to(self.device)
        self.grid = 28
