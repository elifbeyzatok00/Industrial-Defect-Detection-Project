import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

from config import (
    TRAIN_DIR, TEST_DIR, IMAGE_SIZE, OUTPUT_DIR
)


class GaussianBaselineDetector:
    """
    Baseline anomaly detector using Gaussian blur + per-pixel statistics.
    Builds a normal distribution model from training images.
    Anomaly score = mean absolute deviation from training pixel mean.
    """

    def __init__(self, blur_sigma=3):
        self.blur_sigma = blur_sigma
        self.mean_map = None
        self.std_map = None

    def _load_and_preprocess(self, path):
        img = np.array(Image.open(path).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE)),
                       dtype=np.float32) / 255.0
        return gaussian_filter(img, sigma=self.blur_sigma)

    def fit(self, train_paths):
        print("Fitting Gaussian baseline...")
        stack = np.stack([self._load_and_preprocess(p) for p in tqdm(train_paths)])
        self.mean_map = stack.mean(axis=0)
        self.std_map = stack.std(axis=0) + 1e-6
        # Compute threshold from normal training scores
        train_scores = np.array([self.score(p) for p in train_paths])
        self._train_threshold = train_scores.mean() + 2 * train_scores.std()
        print(f"  Fitted on {len(train_paths)} training images. "
              f"Threshold={self._train_threshold:.4f}")

    def score(self, image_path):
        """Return scalar anomaly score for one image."""
        img = self._load_and_preprocess(image_path)
        diff = np.abs(img - self.mean_map) / self.std_map
        return float(diff.mean())

    def score_map(self, image_path):
        """Return per-pixel anomaly heatmap (normalised 0-1)."""
        img = self._load_and_preprocess(image_path)
        diff = np.abs(img - self.mean_map) / self.std_map
        normalised = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
        return normalised

    def predict(self, test_paths, threshold=None):
        scores = [self.score(p) for p in tqdm(test_paths, desc="Baseline scoring")]
        if threshold is None:
            threshold = self._auto_threshold(scores)
        predictions = [1 if s >= threshold else 0 for s in scores]
        return np.array(scores), np.array(predictions), threshold

    def _auto_threshold(self, scores=None):
        """Threshold from training normal scores (set during fit)."""
        if hasattr(self, "_train_threshold"):
            return self._train_threshold
        arr = np.array(scores)
        return arr.mean() + 2 * arr.std()


def collect_image_paths(directory: Path):
    paths = sorted(directory.rglob("*.png")) + sorted(directory.rglob("*.jpg"))
    return paths


def run_baseline():
    train_paths = list(TRAIN_DIR.glob("*.png")) + list(TRAIN_DIR.glob("*.jpg"))

    detector = GaussianBaselineDetector(blur_sigma=3)
    detector.fit(train_paths)

    test_paths, true_labels = [], []
    good_dir = TEST_DIR / "good"
    if good_dir.exists():
        for p in sorted(good_dir.glob("*.png")) + sorted(good_dir.glob("*.jpg")):
            test_paths.append(p)
            true_labels.append(0)
    for d in sorted(TEST_DIR.iterdir()):
        if d.name == "good" or not d.is_dir():
            continue
        for p in sorted(d.glob("*.png")) + sorted(d.glob("*.jpg")):
            test_paths.append(p)
            true_labels.append(1)

    scores, predictions, threshold = detector.predict(test_paths)
    return scores, np.array(true_labels), predictions, threshold
