import os
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from config import (
    TRAIN_DIR, TEST_DIR, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS, DATA_CATEGORY
)


class MVTecDataset(Dataset):
    def __init__(self, image_paths, labels=None, transform=None):
        self.image_paths = image_paths
        self.labels = labels if labels is not None else [0] * len(image_paths)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label, str(self.image_paths[idx])


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, test_transform


def load_train_data(few_shot_ratio=1.0, train_dir=None):
    train_dir = Path(train_dir) if train_dir else TRAIN_DIR
    train_transform, _ = get_transforms()
    image_paths = sorted(train_dir.glob("*.png")) + sorted(train_dir.glob("*.jpg"))

    if not image_paths:
        raise FileNotFoundError(
            f"No images found in {train_dir}. "
            "Run generate_dummy_data() or download MVTec AD."
        )

    if few_shot_ratio < 1.0:
        n = max(1, int(len(image_paths) * few_shot_ratio))
        image_paths = image_paths[:n]

    dataset = MVTecDataset(image_paths, transform=train_transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=NUM_WORKERS)
    return loader, len(image_paths)


def load_test_data(test_dir=None):
    test_dir = Path(test_dir) if test_dir else TEST_DIR
    _, test_transform = get_transforms()
    image_paths = []
    labels = []

    good_dir = test_dir / "good"
    if good_dir.exists():
        for p in sorted(good_dir.glob("*.png")) + sorted(good_dir.glob("*.jpg")):
            image_paths.append(p)
            labels.append(0)

    for defect_dir in sorted(test_dir.iterdir()):
        if defect_dir.name == "good" or not defect_dir.is_dir():
            continue
        for p in sorted(defect_dir.glob("*.png")) + sorted(defect_dir.glob("*.jpg")):
            image_paths.append(p)
            labels.append(1)

    if not image_paths:
        raise FileNotFoundError(
            f"No test images found in {test_dir}. "
            "Run generate_dummy_data() or download MVTec AD."
        )

    dataset = MVTecDataset(image_paths, labels=labels, transform=test_transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=NUM_WORKERS)
    return loader, labels


def generate_dummy_data(n_train=70, n_good_test=20, n_defect_test=25):
    """Generate synthetic images for testing when MVTec AD is not available."""
    print("Generating dummy dataset...")

    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    good_test = TEST_DIR / "good"
    good_test.mkdir(parents=True, exist_ok=True)
    defect_test = TEST_DIR / "defect"
    defect_test.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)

    for i in range(n_train):
        img = rng.integers(180, 220, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        Image.fromarray(img).save(TRAIN_DIR / f"train_{i:04d}.png")

    for i in range(n_good_test):
        img = rng.integers(180, 220, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        Image.fromarray(img).save(good_test / f"good_{i:04d}.png")

    for i in range(n_defect_test):
        img = rng.integers(180, 220, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        # Add a visible defect patch
        x, y = rng.integers(30, 150, size=2)
        img[y:y+30, x:x+30] = rng.integers(0, 60, (30, 30, 3), dtype=np.uint8)
        Image.fromarray(img).save(defect_test / f"defect_{i:04d}.png")

    print(f"  Train: {n_train} images → {TRAIN_DIR}")
    print(f"  Test good: {n_good_test} images → {good_test}")
    print(f"  Test defect: {n_defect_test} images → {defect_test}")
