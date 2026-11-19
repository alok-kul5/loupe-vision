"""Supervised baseline: an ImageNet ResNet-18 fine-tuned to classify good vs defect.

Where it looks is shown with Grad-CAM. For a network that ends in global
average pooling followed by one linear layer (ResNet does), Grad-CAM for the
"defect" logit is exactly CAM: the last conv feature maps weighted by that
logit's fc weights. So the heatmap can be computed without gradients, which
lets the ONNX export serve it from plain feature maps.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.models import ResNet18_Weights, resnet18


class DefectClassifier(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3, m.layer4)
        self.fc = nn.Linear(512, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits (B,2), last conv feature maps (B,512,7,7))."""
        a = self.features(x)
        return self.fc(a.mean(dim=(2, 3))), a

    def gradcam(self, x: torch.Tensor) -> torch.Tensor:
        """Grad-CAM for the defect-vs-good logit margin via autograd, (B, H, W) in [0, 1]."""
        logits, a = self(x)
        a.retain_grad()
        (logits[:, 1] - logits[:, 0]).sum().backward()
        weights = a.grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * a).sum(1))
        return _normalize(F.interpolate(cam.unsqueeze(1), size=x.shape[-2:], mode="bilinear", align_corners=False).squeeze(1))


def cam_from_features(a: torch.Tensor, fc_weight: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Gradient-free CAM, equal to Grad-CAM for GAP + linear heads (up to a positive scale)."""
    w = (fc_weight[1] - fc_weight[0]).view(1, -1, 1, 1)  # direction of the defect-vs-good logit difference
    cam = F.relu((w * a).sum(1))
    return _normalize(F.interpolate(cam.unsqueeze(1), size=size, mode="bilinear", align_corners=False).squeeze(1))


def _normalize(m: torch.Tensor) -> torch.Tensor:
    flat = m.flatten(1)
    lo, hi = flat.min(1).values.view(-1, 1, 1), flat.max(1).values.view(-1, 1, 1)
    return (m - lo) / (hi - lo + 1e-8)


def balanced_loader(dataset, labels: list[int], batch_size: int = 32) -> DataLoader:
    """Defects are rare in the training data; sample both classes equally often."""
    counts = torch.bincount(torch.tensor(labels), minlength=2).float()
    w = (1.0 / counts)[torch.tensor(labels)]
    sampler = WeightedRandomSampler(w, num_samples=max(len(labels), 256), replacement=True)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0)


def train_classifier(model: DefectClassifier, loader: DataLoader, epochs: int, device: str, log=None) -> DefectClassifier:
    model.to(device).train()
    opt = torch.optim.AdamW([
        {"params": model.features.parameters(), "lr": 1e-4},
        {"params": model.fc.parameters(), "lr": 1e-3},
    ], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(loader))
    for epoch in range(epochs):
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)
        if log:
            log(epoch, loss_sum / total, correct / total)
    return model.eval()


@torch.no_grad()
def predict(model: DefectClassifier, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Defect probability (B,) and CAM heatmaps (B, H, W)."""
    logits, a = model(x)
    return logits.softmax(1)[:, 1], cam_from_features(a, model.fc.weight, x.shape[-2:])
