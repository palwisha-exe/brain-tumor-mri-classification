from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from src.config import CLASS_NAMES, DISPLAY_NAMES, PROJECT_ROOT, get_device, set_seed
from src.models import build_model
from src.train_v2 import SplitDataset


RUN_NAME = "v2_b0_224_last3_unfreeze"
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "v2" / f"{RUN_NAME}_best.pt"
SUMMARY_PATH = PROJECT_ROOT / "reports" / "metrics" / "v2" / f"{RUN_NAME}_training_summary.json"
V1_METRICS_PATH = PROJECT_ROOT / "reports" / "metrics" / "efficientnet_b0_finetuned_test_metrics.json"
V1_PREDICTIONS_PATH = PROJECT_ROOT / "reports" / "metrics" / "efficientnet_b0_finetuned_test_predictions.csv"
TEST_MANIFEST_PATH = PROJECT_ROOT / "metadata" / "splits" / "test.csv"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "metrics" / "v2"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures" / "v2"

PREDICTIONS_PATH = OUTPUT_DIR / "final_v2_test_predictions.csv"
METRICS_PATH = OUTPUT_DIR / "final_v2_test_metrics.json"
REPORT_CSV_PATH = OUTPUT_DIR / "final_v2_classification_report.csv"
REPORT_TEXT_PATH = OUTPUT_DIR / "final_v2_classification_report.txt"
ERRORS_PATH = OUTPUT_DIR / "final_v2_top20_high_confidence_errors.csv"
COMPARISON_CSV_PATH = OUTPUT_DIR / "final_v1_vs_v2_heldout_comparison.csv"
COMPARISON_JSON_PATH = OUTPUT_DIR / "final_v1_vs_v2_heldout_comparison.json"
CONFUSION_PATH = FIGURE_DIR / "final_v2_test_confusion_matrix.png"

EXPECTED_CHECKPOINT_SHA256 = "26e4b268c112533bf22b6e726625044b0d0ad774fa7d00bea2599e044c36e1d0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def confidence_statistics(values: pd.Series) -> dict:
    array = values.to_numpy(dtype=np.float64)
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "standard_deviation_population": float(np.std(array, ddof=0)),
        "minimum": float(np.min(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "maximum": float(np.max(array)),
    }


def verify_frozen_selection(checkpoint: dict, summary: dict) -> str:
    digest = sha256(CHECKPOINT_PATH)
    checks = {
        "checkpoint_sha256": digest == EXPECTED_CHECKPOINT_SHA256 == summary["checkpoint_sha256"],
        "run_name": checkpoint["run_name"] == RUN_NAME == summary["run_name"],
        "architecture": checkpoint["architecture"] == summary["architecture"] == "efficientnet_b0",
        "input_size": checkpoint["input_size"] == summary["input_size"] == 224,
        "unfrozen_blocks": checkpoint["unfrozen_blocks"] == summary["unfrozen_blocks"] == 3,
        "best_epoch": checkpoint["best_epoch"] == summary["best_epoch"] == 3,
        "class_mapping": tuple(checkpoint["class_names"]) == tuple(CLASS_NAMES),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Frozen Version 2 selection verification failed: {failed}")
    return digest


def main() -> None:
    output_paths = [
        PREDICTIONS_PATH,
        METRICS_PATH,
        REPORT_CSV_PATH,
        REPORT_TEXT_PATH,
        ERRORS_PATH,
        COMPARISON_CSV_PATH,
        COMPARISON_JSON_PATH,
        CONFUSION_PATH,
    ]
    existing = [str(path) for path in output_paths if path.exists()]
    if existing:
        raise FileExistsError(f"Final Version 2 test artifacts already exist; refusing to reevaluate: {existing}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    set_seed()

    summary = json.loads(SUMMARY_PATH.read_text())
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    checkpoint_digest = verify_frozen_selection(checkpoint, summary)

    dataset = SplitDataset("test", input_size=224, training=False, augmentation=checkpoint["augmentation"])
    manifest = pd.read_csv(TEST_MANIFEST_PATH)
    if len(dataset) != len(manifest) or not dataset.frame.equals(manifest):
        raise ValueError("The evaluation dataset does not exactly match the official test manifest.")
    if set(manifest["final_split"]) != {"test"}:
        raise ValueError("Official test manifest contains a non-test row.")
    observed_mapping = dict(
        manifest[["class_name", "class_index"]].drop_duplicates().sort_values("class_index").values.tolist()
    )
    expected_mapping = {name: index for index, name in enumerate(CLASS_NAMES)}
    if observed_mapping != expected_mapping:
        raise ValueError(f"Test-manifest class mapping mismatch: {observed_mapping}")

    v1_predictions = pd.read_csv(V1_PREDICTIONS_PATH)
    if len(v1_predictions) != len(manifest):
        raise ValueError("Version 1 predictions and official test manifest have different row counts.")
    if v1_predictions["relative_path"].tolist() != manifest["relative_path"].tolist():
        raise ValueError("Version 1 predictions are not aligned to the same official test manifest.")
    if v1_predictions["true_class"].tolist() != manifest["class_name"].tolist():
        raise ValueError("Version 1 true labels do not match the official test manifest.")

    device = get_device()
    model = build_model("efficientnet_b0", pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

    probability_batches: list[np.ndarray] = []
    labels: list[int] = []
    indices: list[int] = []
    with torch.no_grad():
        for inputs, batch_labels, batch_indices in loader:
            logits = model(inputs.to(device))
            probability_batches.append(torch.softmax(logits, dim=1).cpu().numpy())
            labels.extend(batch_labels.numpy().tolist())
            indices.extend(batch_indices.numpy().tolist())

    probabilities = np.concatenate(probability_batches)
    labels_array = np.asarray(labels)
    indices_array = np.asarray(indices)
    predicted = probabilities.argmax(axis=1)
    matrix = confusion_matrix(labels_array, predicted, labels=np.arange(len(CLASS_NAMES)))
    precision, recall, f1, support = precision_recall_fscore_support(
        labels_array, predicted, labels=np.arange(len(CLASS_NAMES)), zero_division=0
    )

    output = manifest.iloc[indices_array].reset_index(drop=True).copy()
    output["true_class"] = [CLASS_NAMES[index] for index in labels_array]
    output["predicted_class"] = [CLASS_NAMES[index] for index in predicted]
    output["confidence"] = probabilities.max(axis=1)
    output["correct"] = labels_array == predicted
    for index, class_name in enumerate(CLASS_NAMES):
        output[f"probability_{class_name}"] = probabilities[:, index]
    output.to_csv(PREDICTIONS_PATH, index=False)

    report_dict = classification_report(
        labels_array,
        predicted,
        labels=np.arange(len(CLASS_NAMES)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        labels_array,
        predicted,
        labels=np.arange(len(CLASS_NAMES)),
        target_names=CLASS_NAMES,
        digits=6,
        zero_division=0,
    )
    pd.DataFrame(report_dict).transpose().to_csv(REPORT_CSV_PATH, index_label="label")
    REPORT_TEXT_PATH.write_text(report_text)

    error_columns = [
        "relative_path",
        "true_class",
        "predicted_class",
        "confidence",
        *[f"probability_{name}" for name in CLASS_NAMES],
    ]
    top_errors = output.loc[~output["correct"], error_columns].sort_values(
        "confidence", ascending=False, kind="stable"
    ).head(20)
    top_errors.insert(0, "filename", top_errors["relative_path"].map(lambda value: Path(value).name))
    top_errors.to_csv(ERRORS_PATH, index=False)

    glioma_to_meningioma = int(matrix[0, 1])
    meningioma_to_glioma = int(matrix[1, 0])
    metrics = {
        "evaluation_name": "Final Version 2 official leakage-controlled held-out test evaluation",
        "run_name": RUN_NAME,
        "checkpoint": str(CHECKPOINT_PATH.relative_to(PROJECT_ROOT)),
        "checkpoint_sha256": checkpoint_digest,
        "selected_best_epoch": int(checkpoint["best_epoch"]),
        "architecture": checkpoint["architecture"],
        "input_size": int(checkpoint["input_size"]),
        "unfrozen_blocks_during_training": int(checkpoint["unfrozen_blocks"]),
        "class_names_in_output_index_order": list(CLASS_NAMES),
        "split": "test",
        "test_manifest": str(TEST_MANIFEST_PATH.relative_to(PROJECT_ROOT)),
        "test_manifest_sha256": sha256(TEST_MANIFEST_PATH),
        "samples": int(len(labels_array)),
        "selection_used_this_split": False,
        "retraining_or_tuning_performed": False,
        "accuracy": float(accuracy_score(labels_array, predicted)),
        "macro_precision": float(precision_score(labels_array, predicted, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(labels_array, predicted, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(labels_array, predicted, average="macro", zero_division=0)),
        "per_class": {
            class_name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, class_name in enumerate(CLASS_NAMES)
        },
        "confusion_matrix": matrix.tolist(),
        "glioma_to_meningioma_errors": glioma_to_meningioma,
        "meningioma_to_glioma_errors": meningioma_to_glioma,
        "glioma_meningioma_pairwise_errors": glioma_to_meningioma + meningioma_to_glioma,
        "confidence_statistics": {
            "correct_predictions": confidence_statistics(output.loc[output["correct"], "confidence"]),
            "incorrect_predictions": confidence_statistics(output.loc[~output["correct"], "confidence"]),
        },
        "artifacts": {
            "predictions_csv": str(PREDICTIONS_PATH.relative_to(PROJECT_ROOT)),
            "classification_report_csv": str(REPORT_CSV_PATH.relative_to(PROJECT_ROOT)),
            "classification_report_text": str(REPORT_TEXT_PATH.relative_to(PROJECT_ROOT)),
            "top_20_errors_csv": str(ERRORS_PATH.relative_to(PROJECT_ROOT)),
            "confusion_matrix_figure": str(CONFUSION_PATH.relative_to(PROJECT_ROOT)),
        },
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n")

    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        square=True,
        linewidths=0.6,
        xticklabels=[DISPLAY_NAMES[name] for name in CLASS_NAMES],
        yticklabels=[DISPLAY_NAMES[name] for name in CLASS_NAMES],
        ax=ax,
    )
    ax.set_title("Version 2 — Official Held-Out Test Confusion Matrix", pad=14, weight="bold")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    fig.tight_layout()
    fig.savefig(CONFUSION_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)

    v1 = json.loads(V1_METRICS_PATH.read_text())
    v1_matrix = np.asarray(v1["confusion_matrix"])
    if int(v1["samples"]) != len(labels_array) or int(v1_matrix.sum()) != len(labels_array):
        raise ValueError("Saved Version 1 metrics do not describe this same official test split.")
    comparison_rows = [
        {
            "version": "Version 1",
            "run_name": v1["run_name"],
            "checkpoint": v1["checkpoint"],
            "samples": int(v1["samples"]),
            "accuracy": float(v1["accuracy"]),
            "macro_precision": float(v1["macro_precision"]),
            "macro_recall": float(v1["macro_recall"]),
            "macro_f1": float(v1["macro_f1"]),
            "glioma_to_meningioma_errors": int(v1_matrix[0, 1]),
            "meningioma_to_glioma_errors": int(v1_matrix[1, 0]),
            "glioma_meningioma_pairwise_errors": int(v1_matrix[0, 1] + v1_matrix[1, 0]),
        },
        {
            "version": "Version 2",
            "run_name": RUN_NAME,
            "checkpoint": str(CHECKPOINT_PATH.relative_to(PROJECT_ROOT)),
            "samples": int(len(labels_array)),
            "accuracy": metrics["accuracy"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "macro_f1": metrics["macro_f1"],
            "glioma_to_meningioma_errors": glioma_to_meningioma,
            "meningioma_to_glioma_errors": meningioma_to_glioma,
            "glioma_meningioma_pairwise_errors": glioma_to_meningioma + meningioma_to_glioma,
        },
    ]
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(COMPARISON_CSV_PATH, index=False)
    delta = {
        metric: float(comparison_rows[1][metric] - comparison_rows[0][metric])
        for metric in ("accuracy", "macro_precision", "macro_recall", "macro_f1")
    }
    comparison_payload = {
        "comparison_name": "Version 1 vs selected Version 2 on the same official leakage-controlled held-out test split",
        "test_manifest": str(TEST_MANIFEST_PATH.relative_to(PROJECT_ROOT)),
        "test_manifest_sha256": metrics["test_manifest_sha256"],
        "samples": int(len(labels_array)),
        "selection_or_tuning_used_test_results": False,
        "models": comparison_rows,
        "v2_minus_v1": delta,
        "v2_improves_accuracy": bool(delta["accuracy"] > 0),
        "v2_improves_macro_f1": bool(delta["macro_f1"] > 0),
    }
    COMPARISON_JSON_PATH.write_text(json.dumps(comparison_payload, indent=2) + "\n")

    print(json.dumps({"metrics": metrics, "comparison": comparison_payload}, indent=2))


if __name__ == "__main__":
    main()
