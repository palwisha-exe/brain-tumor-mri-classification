from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image, ImageDraw, ImageOps
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from src.config import CLASS_NAMES, DISPLAY_NAMES, FIGURES_DIR, METRICS_DIR, PROJECT_ROOT, ensure_directories, get_device
from src.data import make_loader
from src.models import build_model


def predict(checkpoint_path: Path, split: str, batch_size: int):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    architecture = checkpoint["architecture"]
    model = build_model(architecture, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    device = get_device()
    model.to(device).eval()
    loader = make_loader(split, architecture, batch_size, training=False, input_size=checkpoint["input_size"])
    probabilities, labels, indices = [], [], []
    with torch.no_grad():
        for inputs, batch_labels, batch_indices in loader:
            logits = model(inputs.to(device))
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            labels.extend(batch_labels.numpy().tolist())
            indices.extend(batch_indices.numpy().tolist())
    return checkpoint, loader.dataset.frame, np.concatenate(probabilities), np.asarray(labels), np.asarray(indices)


def metric_payload(labels: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray) -> dict:
    precision_per, recall_per, f1_per, support = precision_recall_fscore_support(
        labels, predictions, labels=np.arange(len(CLASS_NAMES)), zero_division=0
    )
    per_class = {
        class_name: {
            "precision": float(precision_per[index]),
            "recall": float(recall_per[index]),
            "f1": float(f1_per[index]),
            "support": int(support[index]),
        }
        for index, class_name in enumerate(CLASS_NAMES)
    }
    payload = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_precision": float(precision_score(labels, predictions, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(labels, predictions, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(labels, predictions, labels=np.arange(len(CLASS_NAMES))).tolist(),
        "classification_report": classification_report(
            labels, predictions, target_names=CLASS_NAMES, output_dict=True, zero_division=0
        ),
    }
    one_hot = label_binarize(labels, classes=np.arange(len(CLASS_NAMES)))
    payload["roc_auc_ovr_macro"] = float(roc_auc_score(one_hot, probabilities, average="macro", multi_class="ovr"))
    payload["roc_auc_ovr_weighted"] = float(roc_auc_score(one_hot, probabilities, average="weighted", multi_class="ovr"))
    payload["roc_auc_per_class"] = {
        class_name: float(roc_auc_score(one_hot[:, index], probabilities[:, index]))
        for index, class_name in enumerate(CLASS_NAMES)
    }
    return payload


def save_confusion(matrix: np.ndarray, run_name: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set(title=f"{run_name} — held-out test confusion matrix", xlabel="Predicted", ylabel="True")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{run_name}_test_confusion_matrix.png", dpi=180)
    plt.close(fig)


def save_roc(labels: np.ndarray, probabilities: np.ndarray, run_name: str) -> None:
    one_hot = label_binarize(labels, classes=np.arange(len(CLASS_NAMES)))
    fig, ax = plt.subplots(figsize=(7, 6))
    for index, class_name in enumerate(CLASS_NAMES):
        false_positive, true_positive, _ = roc_curve(one_hot[:, index], probabilities[:, index])
        ax.plot(false_positive, true_positive, label=f"{DISPLAY_NAMES[class_name]} (AUC={auc(false_positive, true_positive):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set(title=f"{run_name} — one-vs-rest ROC", xlabel="False-positive rate", ylabel="True-positive rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{run_name}_test_roc.png", dpi=180)
    plt.close(fig)


def save_error_montage(predictions: pd.DataFrame, run_name: str) -> None:
    errors = predictions[~predictions["correct"]].sort_values("confidence", ascending=False).head(20)
    cell_w, cell_h = 260, 285
    canvas = Image.new("RGB", (cell_w * 5, max(1, (len(errors) + 4) // 5) * cell_h), "white")
    draw = ImageDraw.Draw(canvas)
    for position, (_, row) in enumerate(errors.iterrows()):
        with Image.open(PROJECT_ROOT / row["relative_path"]) as image:
            tile = ImageOps.contain(image.convert("RGB"), (240, 220))
        x, y = (position % 5) * cell_w + 10, (position // 5) * cell_h + 5
        canvas.paste(tile, (x + (240 - tile.width) // 2, y))
        filename = Path(row["relative_path"]).name
        label = f"T: {row['true_class']}  P: {row['predicted_class']}\np={row['confidence']:.3f} | {filename}"
        draw.multiline_text((x, y + 225), label, fill="black", spacing=3)
    canvas.save(FIGURES_DIR / f"{run_name}_highest_confidence_errors.jpg", quality=92)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--split", default="test", choices=("validation", "test"))
    parser.add_argument("--batch-size", type=int, default=24)
    args = parser.parse_args()
    ensure_directories()
    checkpoint, manifest, probabilities, labels, indices = predict(args.checkpoint, args.split, args.batch_size)
    predictions = probabilities.argmax(axis=1)
    metrics = metric_payload(labels, predictions, probabilities)
    metrics.update(
        {
            "run_name": args.run_name,
            "architecture": checkpoint["architecture"],
            "checkpoint": str(args.checkpoint),
            "split": args.split,
            "samples": int(len(labels)),
            "selection_used_this_split": False,
        }
    )
    output = manifest.iloc[indices].reset_index(drop=True).copy()
    output["true_class"] = [CLASS_NAMES[index] for index in labels]
    output["predicted_class"] = [CLASS_NAMES[index] for index in predictions]
    output["confidence"] = probabilities.max(axis=1)
    output["correct"] = labels == predictions
    for index, class_name in enumerate(CLASS_NAMES):
        output[f"probability_{class_name}"] = probabilities[:, index]
    output.to_csv(METRICS_DIR / f"{args.run_name}_{args.split}_predictions.csv", index=False)
    (METRICS_DIR / f"{args.run_name}_{args.split}_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    if args.split == "test":
        save_confusion(np.asarray(metrics["confusion_matrix"]), args.run_name)
        save_roc(labels, probabilities, args.run_name)
        save_error_montage(output, args.run_name)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
