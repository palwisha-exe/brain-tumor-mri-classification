from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn

from src.config import (
    CLASS_NAMES,
    FIGURES_DIR,
    INPUT_SIZE,
    METRICS_DIR,
    MODELS_DIR,
    SEED,
    ensure_directories,
    get_device,
    set_seed,
)
from src.data import make_loader, normalization_for
from src.models import build_model, configure_trainable_layers


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    if training and hasattr(model, "features"):
        for module in model.features.modules():
            if isinstance(module, nn.BatchNorm2d) and not any(parameter.requires_grad for parameter in module.parameters()):
                module.eval()
    total_loss = 0.0
    labels_all, predictions_all = [], []
    for inputs, labels, _ in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(inputs)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        labels_all.extend(labels.detach().cpu().tolist())
        predictions_all.extend(logits.argmax(dim=1).detach().cpu().tolist())
    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(labels_all, predictions_all),
        "macro_f1": f1_score(labels_all, predictions_all, average="macro", zero_division=0),
    }


def plot_history(history: pd.DataFrame, run_name: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for metric, axis in zip(("loss", "accuracy", "macro_f1"), axes):
        axis.plot(history["epoch"], history[f"train_{metric}"], marker="o", label="Train")
        axis.plot(history["epoch"], history[f"validation_{metric}"], marker="o", label="Validation")
        axis.set(title=metric.replace("_", " ").title(), xlabel="Epoch", ylabel=metric.replace("_", " ").title())
        axis.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{run_name}_training_curves.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=("baseline_cnn", "efficientnet_b0"), required=True)
    parser.add_argument("--stage", choices=("baseline", "frozen", "finetune"), required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--input-size", type=int)
    args = parser.parse_args()

    ensure_directories()
    set_seed(SEED)
    device = get_device()
    input_size = args.input_size or (128 if args.architecture == "baseline_cnn" else INPUT_SIZE)
    pretrained = args.architecture == "efficientnet_b0" and args.stage == "frozen" and args.resume is None
    model = build_model(args.architecture, pretrained=pretrained)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
    configure_trainable_layers(model, args.architecture, args.stage)
    model.to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.3, patience=2)
    train_loader = make_loader("train", args.architecture, args.batch_size, training=True, input_size=input_size)
    validation_loader = make_loader("validation", args.architecture, args.batch_size, training=False, input_size=input_size)
    counts = train_loader.dataset.frame["class_index"].value_counts().sort_index().to_numpy()
    class_weights = len(train_loader.dataset) / (len(CLASS_NAMES) * counts)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))

    history = []
    best_score = -1.0
    epochs_without_improvement = 0
    checkpoint_path = MODELS_DIR / f"{args.run_name}_best.pt"
    start = time.time()
    print(f"device={device} trainable_parameters={sum(p.numel() for p in trainable):,}", flush=True)
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        validation_metrics = run_epoch(model, validation_loader, criterion, device)
        scheduler.step(validation_metrics["macro_f1"])
        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        score = validation_metrics["macro_f1"]
        if score > best_score + 1e-5:
            best_score = score
            epochs_without_improvement = 0
            mean, std = normalization_for(args.architecture)
            torch.save(
                {
                    "architecture": args.architecture,
                    "stage": args.stage,
                    "run_name": args.run_name,
                    "model_state_dict": model.state_dict(),
                    "class_names": list(CLASS_NAMES),
                    "input_size": input_size,
                    "normalization_mean": mean,
                    "normalization_std": std,
                    "seed": SEED,
                    "class_weights": class_weights.tolist(),
                    "best_validation_macro_f1": best_score,
                    "best_epoch": epoch,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"early_stopping epoch={epoch}", flush=True)
                break

    history_frame = pd.DataFrame(history)
    history_frame.to_csv(METRICS_DIR / f"{args.run_name}_history.csv", index=False)
    summary = {
        "run_name": args.run_name,
        "architecture": args.architecture,
        "stage": args.stage,
        "device": str(device),
        "seed": SEED,
        "epochs_completed": len(history),
        "best_validation_macro_f1": best_score,
        "best_epoch": int(torch.load(checkpoint_path, map_location="cpu", weights_only=False)["best_epoch"]),
        "elapsed_seconds": time.time() - start,
        "checkpoint": str(checkpoint_path.relative_to(checkpoint_path.parents[1])),
        "test_set_used": False,
        "class_weights": class_weights.tolist(),
        "input_size": input_size,
    }
    (METRICS_DIR / f"{args.run_name}_training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_history(history_frame, args.run_name)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
