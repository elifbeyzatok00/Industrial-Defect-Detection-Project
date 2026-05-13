import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm

from config import (
    MODEL_NAME, PRETRAINED, FEATURE_DIM, PATCH_SIZE,
    IMAGE_SIZE, DEVICE, OUTPUT_DIR
)


def _get_device():
    if DEVICE == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class FeatureExtractor(nn.Module):
    """ResNet50 with output from layer2 + layer3 (mid-level features)."""

    def __init__(self):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1
                                   if PRETRAINED else None)
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1,
                                    backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        f2 = self.layer2(x)
        f3 = self.layer3(f2)
        return f2, f3


class PatchCore:
    """
    PatchCore anomaly detector.
    Stores patch-level features from the training set as a memory bank.
    Anomaly score for a test image = max nearest-neighbour distance over all patches.
    """

    def __init__(self, coreset_ratio=0.1):
        self.device = _get_device()
        self.extractor = FeatureExtractor().to(self.device).eval()
        self.memory_bank = None
        self.coreset_ratio = coreset_ratio
        print(f"  Device: {self.device}")

    # ------------------------------------------------------------------
    # Feature extraction helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _extract_patches(self, loader):
        """Return (N_patches, D) feature matrix for all images in loader."""
        all_patches = []
        for imgs, _, _ in tqdm(loader, desc="Extracting features"):
            imgs = imgs.to(self.device)
            f2, f3 = self.extractor(imgs)

            # Upsample f3 to match f2 spatial size, then concatenate
            f3_up = nn.functional.interpolate(f3, size=f2.shape[-2:], mode="bilinear",
                                              align_corners=False)
            feat = torch.cat([f2, f3_up], dim=1)  # (B, C, H, W)

            B, C, H, W = feat.shape
            feat = feat.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
            all_patches.append(feat.cpu().numpy())

        return np.concatenate(all_patches, axis=0)

    # ------------------------------------------------------------------
    # Greedy coreset subsampling (random approximation for speed)
    # ------------------------------------------------------------------

    def _coreset_subsample(self, patches):
        n = max(1, int(len(patches) * self.coreset_ratio))
        idx = np.random.choice(len(patches), size=n, replace=False)
        return patches[idx]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, train_loader):
        print("Building PatchCore memory bank...")
        patches = self._extract_patches(train_loader)
        self.memory_bank = self._coreset_subsample(patches)
        print(f"  Memory bank size: {self.memory_bank.shape[0]} patches "
              f"(coreset {self.coreset_ratio*100:.0f}%)")
        # Derive threshold from normal training scores
        train_scores = self.score_loader(train_loader)
        self._train_threshold = float(train_scores.mean() + 2 * train_scores.std())
        print(f"  Threshold (train)={self._train_threshold:.4f}")

    @torch.no_grad()
    def score_loader(self, test_loader):
        """Return per-image anomaly scores for all images in test_loader."""
        print("Scoring test images...")
        scores = []
        mem = torch.tensor(self.memory_bank, dtype=torch.float32).to(self.device)

        for imgs, _, _ in tqdm(test_loader, desc="PatchCore scoring"):
            imgs = imgs.to(self.device)
            f2, f3 = self.extractor(imgs)
            f3_up = nn.functional.interpolate(f3, size=f2.shape[-2:], mode="bilinear",
                                              align_corners=False)
            feat = torch.cat([f2, f3_up], dim=1)

            B, C, H, W = feat.shape
            patches = feat.permute(0, 2, 3, 1).reshape(B, H * W, C)

            for i in range(B):
                p = patches[i]  # (H*W, C)
                # Euclidean distance to memory bank
                dists = torch.cdist(p, mem)          # (H*W, M)
                min_dists = dists.min(dim=1).values  # (H*W,)
                scores.append(float(min_dists.max().item()))

        return np.array(scores)

    def predict(self, test_loader, threshold=None):
        scores = self.score_loader(test_loader)
        if threshold is None:
            threshold = self._auto_threshold(scores)
        predictions = (scores >= threshold).astype(int)
        return scores, predictions, threshold

    def _auto_threshold(self, scores=None):
        if hasattr(self, "_train_threshold"):
            return self._train_threshold
        return scores.mean() + 2 * scores.std()

    def save(self, path=None):
        path = path or OUTPUT_DIR / "memory_bank.npy"
        np.save(path, self.memory_bank)
        print(f"  Memory bank saved → {path}")

    def load(self, path=None):
        path = path or OUTPUT_DIR / "memory_bank.npy"
        self.memory_bank = np.load(path)
        print(f"  Memory bank loaded ← {path}")
