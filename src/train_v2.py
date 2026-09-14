from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from src.config import (
    CLASS_NAMES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    PROJECT_ROOT,
    SEED,
    SPLITS_DIR,
    get_device,
    set_seed,
)
from src.data import build_transform
from src.models import build_model


V2_MODEL_DIR = PROJECT_ROOT / "models" / "v2"
V2_METRICS_DIR = PROJECT_ROOT / "reports" / "metrics" / "v2"
V2_FIGURES_DIR = PROJECT_ROOT / "reports" / "figures" / "v2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def training_transform(input_size: int, augmentation: str):
    if augmentation == "v1_conservative":
        return build_transform("efficientnet_b0", training=True, input_size=input_size)
    if augmentation == "geometry_light":
        return transforms.Compose(
            [
                transforms.Resize(
                    (input_size, input_size),
                    interpolation=InterpolationMode.BILINEAR,
                    antialias=True,
                ),
                transforms.RandomAffine(
                    degrees=5,
                    translate=(0.025, 0.025),
                    scale=(0.98, 1.02),
                    interpolation=InterpolationMode.BILINEAR,
                    fill=0,
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.05, contrast=0.05),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    raise ValueError(f"Unknown augmentation profile: {augmentation}")


class SplitDataset(Dataset):
    def __init__(self, split: str, input_size: int, training: bool, augmentation: str):
        self.frame = pd.read_csv(SPLITS_DIR / f"{split}.csv")
        if split in {"train", "validation"}:
            # The original Kaggle Testing folder has already been inspected as a
            # diagnostic set. Keep only legitimate manifest rows whose source was
            # the original Training folder so none are fed back into Version 2
            # through either optimization or validation-based model selection.
            self.frame = self.frame[self.frame["original_split"] == "Training"].reset_index(drop=True)
        self.transform = (
            training_transform(input_size, augmentation)
            if training
            else build_transform("efficientnet_b0", training=False, input_size=input_size)
        )

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        with Image.open(PROJECT_ROOT / row["relative_path"]) as source:
            source.load()
            image = source.convert("RGB")
            tensor = self.transform(image)
        return tensor, int(row["class_index"]), index


class FocalLoss(nn.Module):
    def __init__(self, weight: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.register_buffer("weight", weight)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        cross_entropy = torch.nn.functional.cross_entropy(logits, target, weight=self.weight, reduction="none")
        target_probability = torch.softmax(logits, dim=1).gather(1, target[:, None]).squeeze(1)
        return (((1.0 - target_probability) ** self.gamma) * cross_entropy).mean()


def configure_unfreezing(model: nn.Module, unfrozen_blocks: int) -> int:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    if unfrozen_blocks:
        for block in model.features[-unfrozen_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def metrics_from_arrays(labels: np.ndarray, predictions: np.ndarray) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=np.arange(len(CLASS_NAMES)), zero_division=0
    )
    matrix = confusion_matrix(labels, predictions, labels=np.arange(len(CLASS_NAMES)))
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1.mean()),
        "per_class": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, name in enumerate(CLASS_NAMES)
        },
        "confusion_matrix": matrix.tolist(),
        "glioma_as_meningioma": int(matrix[0, 1]),
        "meningioma_as_glioma": int(matrix[1, 0]),
    }


def run_epoch(model, loader, criterion, device, optimizer=None) -> dict:
    training = optimizer is not None
    model.train(training)
    if training:
        for module in model.features.modules():
            if isinstance(module, nn.BatchNorm2d) and not any(
                parameter.requires_grad for parameter in module.parameters()
            ):
                module.eval()
    total_loss = 0.0
    labels_all, predictions_all = [], []
    for inputs, labels, _indices in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(inputs)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += float(loss.item()) * len(labels)
        labels_all.extend(labels.detach().cpu().tolist())
        predictions_all.extend(logits.argmax(1).detach().cpu().tolist())
    payload = metrics_from_arrays(np.asarray(labels_all), np.asarray(predictions_all))
    payload["loss"] = total_loss / len(loader.dataset)
    return payload


def predict(model, dataset, batch_size: int, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    probabilities, labels, indices = [], [], []
    model.eval()
    with torch.no_grad():
        for inputs, batch_labels, batch_indices in loader:
            probabilities.append(torch.softmax(model(inputs.to(device)), dim=1).cpu().numpy())
            labels.extend(batch_labels.numpy().tolist())
            indices.extend(batch_indices.numpy().tolist())
    return np.concatenate(probabilities), np.asarray(labels), np.asarray(indices)


def save_validation_artifacts(run_name: str, checkpoint_path: Path, batch_size: int, device) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model("efficientnet_b0", pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    dataset = SplitDataset("validation", checkpoint["input_size"], False, checkpoint["augmentation"])
    probabilities, labels, indices = predict(model, dataset, batch_size, device)
    predictions = probabilities.argmax(1)
    metrics = metrics_from_arrays(labels, predictions)
    metrics.update(
        {
            "run_name": run_name,
            "split": "validation",
            "samples": len(labels),
            "selection_used_this_split": True,
            "test_set_read_or_used": False,
            "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
        }
    )
    output = dataset.frame.iloc[indices].reset_index(drop=True).copy()
    output["true_class"] = [CLASS_NAMES[index] for index in labels]
    output["predicted_class"] = [CLASS_NAMES[index] for index in predictions]
    output["confidence"] = probabilities.max(1)
    output["correct"] = predictions == labels
    for index, name in enumerate(CLASS_NAMES):
        output[f"probability_{name}"] = probabilities[:, index]
    output.to_csv(V2_METRICS_DIR / f"{run_name}_validation_predictions.csv", index=False)
    (V2_METRICS_DIR / f"{run_name}_validation_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    matrix = np.asarray(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set(title=f"{run_name} — validation", xlabel="Predicted", ylabel="True")
    fig.tight_layout()
    fig.savefig(V2_FIGURES_DIR / f"{run_name}_validation_confusion_matrix.png", dpi=180)
    plt.close(fig)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Controlled EfficientNet-B0 Version 2 validation experiment.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--initial-checkpoint", type=Path, default=PROJECT_ROOT / "models" / "efficientnet_b0_frozen_best.pt")
    parser.add_argument("--input-size", type=int, choices=(160, 224, 256), required=True)
    parser.add_argument("--unfrozen-blocks", type=int, choices=range(0, 10), default=2)
    parser.add_argument(
        "--reset-classifier",
        action="store_true",
        help="Load only the ImageNet-derived feature extractor and initialize a fresh classifier.",
    )
    parser.add_argument("--augmentation", choices=("v1_conservative", "geometry_light"), default="v1_conservative")
    parser.add_argument("--loss", choices=("weighted_ce", "focal"), default="weighted_ce")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()

    V2_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    V2_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    V2_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = V2_MODEL_DIR / f"{args.run_name}_best.pt"
    protected_v1 = PROJECT_ROOT / "models" / "efficientnet_b0_finetuned_best.pt"
    if checkpoint_path.resolve() == protected_v1.resolve():
        raise ValueError("Version 2 output cannot overwrite the Version 1 checkpoint.")
    if checkpoint_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing experiment: {checkpoint_path}")

    set_seed(SEED)
    device = get_device()
    initial_checkpoint_path = args.initial_checkpoint if args.initial_checkpoint.is_absolute() else PROJECT_ROOT / args.initial_checkpoint
    initial = torch.load(initial_checkpoint_path, map_location="cpu", weights_only=False)
    if initial["architecture"] != "efficientnet_b0" or tuple(initial["class_names"]) != tuple(CLASS_NAMES):
        raise ValueError("Initial checkpoint architecture or class order is incompatible.")
    model = build_model("efficientnet_b0", pretrained=False)
    if args.reset_classifier:
        feature_state = {
            key.removeprefix("features."): value
            for key, value in initial["model_state_dict"].items()
            if key.startswith("features.")
        }
        model.features.load_state_dict(feature_state)
    else:
        model.load_state_dict(initial["model_state_dict"])
    trainable_parameters = configure_unfreezing(model, args.unfrozen_blocks)
    model.to(device)

    train_dataset = SplitDataset("train", args.input_size, True, args.augmentation)
    validation_dataset = SplitDataset("validation", args.input_size, False, args.augmentation)
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, generator=generator
    )
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    counts = train_dataset.frame["class_index"].value_counts().sort_index().to_numpy()
    class_weights = len(train_dataset) / (len(CLASS_NAMES) * counts)
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = (
        nn.CrossEntropyLoss(weight=weight_tensor)
        if args.loss == "weighted_ce"
        else FocalLoss(weight_tensor, gamma=args.focal_gamma)
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.3, patience=2)

    configuration = {
        "run_name": args.run_name,
        "version": 2,
        "architecture": "efficientnet_b0",
        "initial_checkpoint": str(initial_checkpoint_path.relative_to(PROJECT_ROOT)),
        "initial_checkpoint_sha256": sha256(initial_checkpoint_path),
        "input_size": args.input_size,
        "unfrozen_blocks": args.unfrozen_blocks,
        "classifier_reset": args.reset_classifier,
        "augmentation": args.augmentation,
        "loss": args.loss,
        "focal_gamma": args.focal_gamma if args.loss == "focal" else None,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "maximum_epochs": args.epochs,
        "patience": args.patience,
        "seed": SEED,
        "device": str(device),
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "test_set_read_or_used": False,
        "trainable_parameters": trainable_parameters,
        "class_weights": class_weights.tolist(),
    }
    (V2_METRICS_DIR / f"{args.run_name}_configuration.json").write_text(json.dumps(configuration, indent=2) + "\n")
    print(json.dumps(configuration, indent=2), flush=True)

    history = []
    best_score = -1.0
    stale = 0
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.time()
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        validation_metrics = run_epoch(model, validation_loader, criterion, device)
        scheduler.step(validation_metrics["macro_f1"])
        row = {
            "epoch": epoch,
            "epoch_seconds": time.time() - epoch_started,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "validation_loss": validation_metrics["loss"],
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_macro_f1": validation_metrics["macro_f1"],
            "validation_glioma_precision": validation_metrics["per_class"]["glioma"]["precision"],
            "validation_glioma_recall": validation_metrics["per_class"]["glioma"]["recall"],
            "validation_glioma_f1": validation_metrics["per_class"]["glioma"]["f1"],
            "validation_meningioma_precision": validation_metrics["per_class"]["meningioma"]["precision"],
            "validation_meningioma_recall": validation_metrics["per_class"]["meningioma"]["recall"],
            "validation_meningioma_f1": validation_metrics["per_class"]["meningioma"]["f1"],
            "validation_glioma_as_meningioma": validation_metrics["glioma_as_meningioma"],
            "validation_meningioma_as_glioma": validation_metrics["meningioma_as_glioma"],
            "validation_confusion_matrix": json.dumps(validation_metrics["confusion_matrix"]),
        }
        history.append(row)
        pd.DataFrame(history).to_csv(V2_METRICS_DIR / f"{args.run_name}_history.csv", index=False)
        print(json.dumps(row), flush=True)
        if validation_metrics["macro_f1"] > best_score + 1e-5:
            best_score = validation_metrics["macro_f1"]
            stale = 0
            torch.save(
                {
                    **configuration,
                    "stage": "frozen" if args.unfrozen_blocks == 0 else "finetune",
                    "model_state_dict": model.state_dict(),
                    "class_names": list(CLASS_NAMES),
                    "normalization_mean": IMAGENET_MEAN,
                    "normalization_std": IMAGENET_STD,
                    "best_validation_macro_f1": best_score,
                    "best_epoch": epoch,
                },
                checkpoint_path,
            )
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early_stopping epoch={epoch}", flush=True)
                break

    elapsed = time.time() - started
    best = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    summary = {
        **configuration,
        "epochs_completed": len(history),
        "best_epoch": best["best_epoch"],
        "best_validation_macro_f1": best["best_validation_macro_f1"],
        "training_time_seconds": elapsed,
        "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
        "checkpoint_sha256": sha256(checkpoint_path),
    }
    (V2_METRICS_DIR / f"{args.run_name}_training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    validation_metrics = save_validation_artifacts(args.run_name, checkpoint_path, args.batch_size, device)
    print(json.dumps({"training_summary": summary, "validation_metrics": validation_metrics}, indent=2), flush=True)


if __name__ == "__main__":
    main()
