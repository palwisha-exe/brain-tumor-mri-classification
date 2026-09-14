from __future__ import annotations

import argparse
import json
import time

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import CLASS_NAMES, FIGURES_DIR, METRICS_DIR, MODELS_DIR, SEED, ensure_directories, get_device, set_seed
from src.data import make_loader, normalization_for
from src.models import build_model, configure_trainable_layers


def extract_features(model, loader, device):
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for inputs, batch_labels, _ in loader:
            values = model.features(inputs.to(device))
            values = model.avgpool(values).flatten(1)
            features.append(values.cpu())
            labels.append(batch_labels)
    return torch.cat(features), torch.cat(labels)


def evaluate_head(head, loader, criterion):
    head.eval(); total_loss = 0.0; labels_all = []; predictions_all = []
    with torch.no_grad():
        for features, labels in loader:
            logits = head(features)
            total_loss += criterion(logits, labels).item() * len(labels)
            labels_all.extend(labels.tolist()); predictions_all.extend(logits.argmax(1).tolist())
    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(labels_all, predictions_all),
        "macro_f1": f1_score(labels_all, predictions_all, average="macro", zero_division=0),
    }


def main():
    parser = argparse.ArgumentParser(description="Frozen EfficientNet training using cached backbone representations.")
    parser.add_argument("--run-name", default="efficientnet_b0_frozen")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--head-batch-size", type=int, default=256)
    parser.add_argument("--input-size", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()
    ensure_directories(); set_seed(SEED); device = get_device(); started = time.time()
    model = build_model("efficientnet_b0", pretrained=True)
    configure_trainable_layers(model, "efficientnet_b0", "frozen")
    model.to(device)
    train_source = make_loader("train", "efficientnet_b0", args.feature_batch_size, training=True, input_size=args.input_size)
    validation_source = make_loader("validation", "efficientnet_b0", args.feature_batch_size, training=False, input_size=args.input_size)
    print(f"device={device} extracting frozen features at {args.input_size}x{args.input_size}", flush=True)
    train_features, train_labels = extract_features(model, train_source, device)
    validation_features, validation_labels = extract_features(model, validation_source, device)
    print(f"features train={tuple(train_features.shape)} validation={tuple(validation_features.shape)}", flush=True)

    train_loader = DataLoader(TensorDataset(train_features, train_labels), batch_size=args.head_batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(SEED))
    validation_loader = DataLoader(TensorDataset(validation_features, validation_labels), batch_size=args.head_batch_size)
    counts = torch.bincount(train_labels, minlength=len(CLASS_NAMES)).numpy()
    class_weights = len(train_labels) / (len(CLASS_NAMES) * counts)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32))
    head = model.classifier.cpu()
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.3, patience=2)
    checkpoint_path = MODELS_DIR / f"{args.run_name}_best.pt"
    history = []; best = -1.0; stale = 0
    for epoch in range(1, args.epochs + 1):
        head.train(); total_loss = 0.0; labels_all = []; predictions_all = []
        for features, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = head(features); loss = criterion(logits, labels); loss.backward(); optimizer.step()
            total_loss += loss.item() * len(labels)
            labels_all.extend(labels.tolist()); predictions_all.extend(logits.argmax(1).tolist())
        train_metrics = {
            "loss": total_loss / len(train_loader.dataset),
            "accuracy": accuracy_score(labels_all, predictions_all),
            "macro_f1": f1_score(labels_all, predictions_all, average="macro", zero_division=0),
        }
        validation_metrics = evaluate_head(head, validation_loader, criterion)
        scheduler.step(validation_metrics["macro_f1"])
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()},
               **{f"validation_{k}": v for k, v in validation_metrics.items()},
               "learning_rate": optimizer.param_groups[0]["lr"]}
        history.append(row); print(json.dumps(row), flush=True)
        if validation_metrics["macro_f1"] > best + 1e-5:
            best = validation_metrics["macro_f1"]; stale = 0
            model.classifier.load_state_dict(head.state_dict())
            mean, std = normalization_for("efficientnet_b0")
            torch.save({
                "architecture": "efficientnet_b0", "stage": "frozen", "run_name": args.run_name,
                "model_state_dict": model.state_dict(), "class_names": list(CLASS_NAMES), "input_size": args.input_size,
                "normalization_mean": mean, "normalization_std": std, "seed": SEED,
                "class_weights": class_weights.tolist(), "best_validation_macro_f1": best, "best_epoch": epoch,
                "frozen_feature_training": True,
            }, checkpoint_path)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early_stopping epoch={epoch}", flush=True); break
    frame = pd.DataFrame(history); frame.to_csv(METRICS_DIR / f"{args.run_name}_history.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for metric, axis in zip(("loss", "accuracy", "macro_f1"), axes):
        axis.plot(frame.epoch, frame[f"train_{metric}"], marker="o", label="Train")
        axis.plot(frame.epoch, frame[f"validation_{metric}"], marker="o", label="Validation")
        axis.set_title(metric.replace("_", " ").title()); axis.legend()
    fig.tight_layout(); fig.savefig(FIGURES_DIR / f"{args.run_name}_training_curves.png", dpi=180); plt.close(fig)
    best_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    summary = {
        "run_name": args.run_name, "architecture": "efficientnet_b0", "stage": "frozen",
        "device": str(device), "seed": SEED, "input_size": args.input_size, "epochs_completed": len(history),
        "best_epoch": best_checkpoint["best_epoch"], "best_validation_macro_f1": best,
        "elapsed_seconds": time.time() - started, "checkpoint": f"models/{checkpoint_path.name}",
        "test_set_used": False, "class_weights": class_weights.tolist(),
        "note": "Backbone representations were extracted once from a seeded dynamically augmented training view; only the classifier head was optimized in this stage.",
    }
    (METRICS_DIR / f"{args.run_name}_training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
