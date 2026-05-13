import os
from pathlib import Path

DATASET_URL = "https://www.mvtec.com/company/research/datasets/mvtec-ad"
DATASET_DIR = Path("./dataset")
DATA_CATEGORY = "metal_nut"
TRAIN_DIR = DATASET_DIR / DATA_CATEGORY / "train" / "good"
TEST_DIR = DATASET_DIR / DATA_CATEGORY / "test"

MODEL_NAME = "resnet50"
PRETRAINED = True
FEATURE_DIM = 2048
PATCH_SIZE = 7
IMAGE_SIZE = 224
DEVICE = "cuda"

BATCH_SIZE = 32
NUM_WORKERS = 4

THRESHOLD_START = 0.0
THRESHOLD_END = 100.0
THRESHOLD_STEP = 0.5

OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(exist_ok=True)
RESULTS_FILE = OUTPUT_DIR / "results.csv"
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)