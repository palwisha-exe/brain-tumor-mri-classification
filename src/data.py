from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from src.config import (
    BASELINE_MEAN,
    BASELINE_STD,
    IMAGENET_MEAN,
    IMAGENET_STD,
    INPUT_SIZE,
    PROJECT_ROOT,
    SEED,
    SPLITS_DIR,
)


def normalization_for(architecture: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if architecture == "efficientnet_b0":
        return IMAGENET_MEAN, IMAGENET_STD
    if architecture == "baseline_cnn":
        return BASELINE_MEAN, BASELINE_STD
    raise ValueError(f"Unknown architecture: {architecture}")


def build_transform(architecture: str, training: bool, input_size: int = INPUT_SIZE):
    mean, std = normalization_for(architecture)
    if training:
        spatial = [
            transforms.RandomResizedCrop(
                input_size,
                scale=(0.90, 1.0),
                ratio=(0.95, 1.05),
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(7, interpolation=InterpolationMode.BILINEAR, fill=0),
            transforms.ColorJitter(brightness=0.08, contrast=0.08),
        ]
    else:
        spatial = [
            transforms.Resize((input_size, input_size), interpolation=InterpolationMode.BILINEAR, antialias=True),
        ]
    return transforms.Compose([*spatial, transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])


class ManifestDataset(Dataset):
    def __init__(self, manifest: Path, architecture: str, training: bool = False, input_size: int = INPUT_SIZE):
        self.frame = pd.read_csv(manifest)
        self.transform = build_transform(architecture, training, input_size=input_size)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        path = PROJECT_ROOT / row["relative_path"]
        with Image.open(path) as image:
            image.load()
            rgb = image.convert("RGB")
            tensor = self.transform(rgb)
        return tensor, int(row["class_index"]), index


def make_loader(
    split: str,
    architecture: str,
    batch_size: int,
    training: bool = False,
    num_workers: int = 0,
    input_size: int = INPUT_SIZE,
) -> DataLoader:
    dataset = ManifestDataset(SPLITS_DIR / f"{split}.csv", architecture, training=training, input_size=input_size)
    generator = torch.Generator().manual_seed(SEED)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
        persistent_workers=num_workers > 0,
    )
