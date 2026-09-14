from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
METADATA_DIR = PROJECT_ROOT / "metadata"
SPLITS_DIR = METADATA_DIR / "splits"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"

CLASS_NAMES = ("glioma", "meningioma", "notumor", "pituitary")
CLASS_TO_IDX = {name: index for index, name in enumerate(CLASS_NAMES)}
DISPLAY_NAMES = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No Tumor",
    "pituitary": "Pituitary",
}

SEED = 20260913
INPUT_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
BASELINE_MEAN = (0.5, 0.5, 0.5)
BASELINE_STD = (0.5, 0.5, 0.5)


def ensure_directories() -> None:
    for path in (METADATA_DIR, SPLITS_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR, METRICS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    requested = os.getenv("BRAIN_MRI_DEVICE", "auto").lower()
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_class_mapping() -> None:
    ensure_directories()
    payload = {
        "class_to_index": CLASS_TO_IDX,
        "index_to_class": {str(v): k for k, v in CLASS_TO_IDX.items()},
        "display_names": DISPLAY_NAMES,
    }
    (METADATA_DIR / "class_mapping.json").write_text(json.dumps(payload, indent=2) + "\n")
