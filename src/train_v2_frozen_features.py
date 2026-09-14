from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD, PROJECT_ROOT, SEED, get_device, set_seed
from src.models import build_model
from src.train_v2 import SplitDataset, V2_METRICS_DIR, V2_MODEL_DIR


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_features(model, loader, device):
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for inputs, batch_labels, _indices in loader:
            values = model.features(inputs.to(device))
            values = model.avgpool(values).flatten(1)
            features.append(values.cpu())
            labels.append(batch_labels)
    return torch.cat(features), torch.cat(labels)


def metrics(labels: list[int], predictions: list[int]) -> dict:
    labels_array = np.asarray(labels)
    predictions_array = np.asarray(predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels_array, predictions_array, labels=np.arange(len(CLASS_NAMES)), zero_division=0
    )
    matrix = confusion_matrix(labels_array, predictions_array, labels=np.arange(len(CLASS_NAMES)))
    return {
        "accuracy": float(accuracy_score(labels_array, predictions_array)),
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
    }


def run_head_epoch(head, loader, criterion, optimizer=None) -> dict:
    training = optimizer is not None
    head.train(training)
    total_loss = 0.0
    labels_all, predictions_all = [], []
    for features, labels in loader:
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = head(features)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += float(loss.item()) * len(labels)
        labels_all.extend(labels.tolist())
        predictions_all.extend(logits.argmax(1).tolist())
    result = metrics(labels_all, predictions_all)
    result["loss"] = total_loss / len(loader.dataset)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Version 2 frozen EfficientNet-B0 head training with cached features.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--input-size", type=int, choices=(160, 224, 256), required=True)
    parser.add_argument("--source-checkpoint", type=Path, default=PROJECT_ROOT / "models" / "efficientnet_b0_frozen_best.pt")
    parser.add_argument("--augmentation", choices=("v1_conservative", "geometry_light"), default="v1_conservative")
    parser.add_argument("--feature-batch-size", type=int, default=32)
    parser.add_argument("--head-batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=6)
    args = parser.parse_args()

    V2_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    V2_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = V2_MODEL_DIR / f"{args.run_name}_best.pt"
    if checkpoint_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing experiment: {checkpoint_path}")
    source_path = args.source_checkpoint if args.source_checkpoint.is_absolute() else PROJECT_ROOT / args.source_checkpoint

    set_seed(SEED)
    device = get_device()
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    if source["stage"] != "frozen":
        raise ValueError("The source checkpoint must have a backbone that remained frozen at ImageNet weights.")
    model = build_model("efficientnet_b0", pretrained=False)
    feature_state = {
        key.removeprefix("features."): value
        for key, value in source["model_state_dict"].items()
        if key.startswith("features.")
    }
    model.features.load_state_dict(feature_state)
    # The source classifier is intentionally not loaded because it saw original-Testing-source rows.
    model.to(device)
    for parameter in model.features.parameters():
        parameter.requires_grad = False

    train_source = SplitDataset("train", args.input_size, True, args.augmentation)
    validation_source = SplitDataset("validation", args.input_size, False, args.augmentation)
    train_loader = DataLoader(train_source, batch_size=args.feature_batch_size, shuffle=False, num_workers=0)
    validation_loader = DataLoader(validation_source, batch_size=args.feature_batch_size, shuffle=False, num_workers=0)
    started = time.time()
    print(f"Extracting {args.input_size}x{args.input_size} frozen features", flush=True)
    train_features, train_labels = extract_features(model, train_loader, device)
    validation_features, validation_labels = extract_features(model, validation_loader, device)
    extraction_seconds = time.time() - started
    print(f"features train={tuple(train_features.shape)} validation={tuple(validation_features.shape)}", flush=True)

    train_head_loader = DataLoader(
        TensorDataset(train_features, train_labels),
        batch_size=args.head_batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    validation_head_loader = DataLoader(
        TensorDataset(validation_features, validation_labels), batch_size=args.head_batch_size, shuffle=False
    )
    counts = torch.bincount(train_labels, minlength=len(CLASS_NAMES)).numpy()
    class_weights = len(train_labels) / (len(CLASS_NAMES) * counts)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32))
    head = model.classifier.cpu()
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.3, patience=2)

    configuration = {
        "run_name": args.run_name,
        "version": 2,
        "architecture": "efficientnet_b0",
        "stage": "frozen_cached_features",
        "source_checkpoint": str(source_path.relative_to(PROJECT_ROOT)),
        "source_checkpoint_sha256": sha256(source_path),
        "source_classifier_loaded": False,
        "input_size": args.input_size,
        "unfrozen_blocks": 0,
        "augmentation": args.augmentation,
        "loss": "weighted_ce",
        "learning_rate": args.learning_rate,
        "seed": SEED,
        "device": str(device),
        "train_samples": len(train_source),
        "validation_samples": len(validation_source),
        "test_set_read_or_used": False,
        "original_testing_source_used": False,
        "cached_training_views": 1,
        "class_weights": class_weights.tolist(),
    }
    (V2_METRICS_DIR / f"{args.run_name}_configuration.json").write_text(json.dumps(configuration, indent=2) + "\n")

    history = []
    best_score = -1.0
    stale = 0
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.time()
        train_metrics = run_head_epoch(head, train_head_loader, criterion, optimizer)
        validation_metrics = run_head_epoch(head, validation_head_loader, criterion)
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
            "validation_confusion_matrix": json.dumps(validation_metrics["confusion_matrix"]),
        }
        history.append(row)
        pd.DataFrame(history).to_csv(V2_METRICS_DIR / f"{args.run_name}_history.csv", index=False)
        print(json.dumps(row), flush=True)
        if validation_metrics["macro_f1"] > best_score + 1e-5:
            best_score = validation_metrics["macro_f1"]
            stale = 0
            model.classifier.load_state_dict(head.state_dict())
            torch.save(
                {
                    **configuration,
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

    elapsed_seconds = time.time() - started
    best = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    summary = {
        **configuration,
        "epochs_completed": len(history),
        "best_epoch": best["best_epoch"],
        "best_validation_macro_f1": best["best_validation_macro_f1"],
        "feature_extraction_seconds": extraction_seconds,
        "training_time_seconds": elapsed_seconds,
        "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
        "checkpoint_sha256": sha256(checkpoint_path),
    }
    (V2_METRICS_DIR / f"{args.run_name}_training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
