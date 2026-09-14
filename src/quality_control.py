from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from src.config import (
    CLASS_NAMES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    METADATA_DIR,
    MODELS_DIR,
    PROJECT_ROOT,
    REPORTS_DIR,
    SPLITS_DIR,
)
from src.data import build_transform
from src.gradcam import GradCAM
from src.models import build_model, last_convolution


SELECTION_PATH = MODELS_DIR / "selected_model_v2.json"


def load_json(path: Path):
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check(condition: bool, name: str, results: list[dict]) -> None:
    if not condition:
        raise AssertionError(name)
    results.append({"check": name, "status": "pass"})


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the frozen Version 2 release without training or selection.")
    parser.add_argument("--output", type=Path, help="Optional JSON report path; omit for read-only verification.")
    args = parser.parse_args()

    results: list[dict] = []
    summary = load_json(METADATA_DIR / "preparation_summary.json")
    mapping = load_json(METADATA_DIR / "class_mapping.json")
    selection_record = load_json(SELECTION_PATH)
    selection = selection_record["selected"]

    check(summary["original_images"] == 7200, "original image count is 7,200", results)
    check(
        summary["total_included_unique"] + summary["total_excluded"] == 7200,
        "included plus excluded equals original count",
        results,
    )
    check(summary["method"] == "image-level leakage-controlled", "split wording is canonical", results)
    check(
        summary["patient_level_independence_verified"] is False,
        "patient-level independence is explicitly unverified",
        results,
    )
    check(
        mapping["class_to_index"] == {name: index for index, name in enumerate(CLASS_NAMES)},
        "class mapping is canonical",
        results,
    )

    frames = {split: pd.read_csv(SPLITS_DIR / f"{split}.csv") for split in ("train", "validation", "test")}
    check(
        {name: len(frame) for name, frame in frames.items()} == summary["final_split_counts"],
        "canonical manifest counts match preparation summary",
        results,
    )
    combined = pd.concat(frames.values(), ignore_index=True)
    check(len(combined) == 5968 and combined["relative_path"].is_unique, "manifests contain 5,968 unique paths", results)
    check(
        not combined["relative_path"].str.contains("aug", case=False).any(),
        "explicitly augmented filenames are absent from final manifests",
        results,
    )
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        check(
            set(frames[left]["pixel_sha256"]).isdisjoint(frames[right]["pixel_sha256"]),
            f"exact decoded hashes are disjoint: {left} vs {right}",
            results,
        )

    v2_train = frames["train"].loc[frames["train"]["original_split"] == "Training"]
    v2_validation = frames["validation"].loc[frames["validation"]["original_split"] == "Training"]
    check(len(v2_train) == 3221, "V2 Training-source-only training count is 3,221", results)
    check(len(v2_validation) == 690, "V2 Training-source-only validation count is 690", results)

    expected_selection = {
        "run_name": "v2_b0_224_last3_unfreeze",
        "architecture": "efficientnet_b0",
        "input_size": 224,
        "unfrozen_blocks": 3,
        "best_epoch": 3,
    }
    check(
        all(selection.get(key) == value for key, value in expected_selection.items()),
        "final V2 selection identity is frozen",
        results,
    )
    check(selection_record["test_set_used_for_selection"] is False, "V2 selection record excludes test selection", results)

    checkpoint_path = PROJECT_ROOT / selection["checkpoint"]
    checkpoint_digest = sha256(checkpoint_path)
    check(checkpoint_digest == selection["checkpoint_sha256"], "final V2 checkpoint SHA-256 matches", results)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    check(
        all(checkpoint.get(key) == value for key, value in expected_selection.items()),
        "checkpoint metadata matches final selection",
        results,
    )
    check(tuple(checkpoint["class_names"]) == tuple(CLASS_NAMES), "checkpoint class order matches", results)
    check(
        np.allclose(checkpoint["normalization_mean"], IMAGENET_MEAN)
        and np.allclose(checkpoint["normalization_std"], IMAGENET_STD),
        "checkpoint normalization matches the V2 pipeline",
        results,
    )

    metrics_path = PROJECT_ROOT / selection["test_metrics"]
    predictions_path = PROJECT_ROOT / selection["test_predictions"]
    metrics = load_json(metrics_path)
    predictions = pd.read_csv(predictions_path)
    check(metrics["checkpoint_sha256"] == checkpoint_digest, "V2 metrics identify the frozen checkpoint", results)
    check(metrics["test_manifest_sha256"] == sha256(SPLITS_DIR / "test.csv"), "V2 metrics identify the test manifest", results)
    check(metrics["samples"] == len(predictions) == len(frames["test"]) == 896, "V2 evaluation contains 896 images", results)
    check(
        predictions["relative_path"].tolist() == frames["test"]["relative_path"].tolist(),
        "V2 predictions align row-for-row with the test manifest",
        results,
    )

    class_to_index = {name: index for index, name in enumerate(CLASS_NAMES)}
    labels = predictions["true_class"].map(class_to_index).to_numpy()
    predicted = predictions["predicted_class"].map(class_to_index).to_numpy()
    matrix = confusion_matrix(labels, predicted, labels=range(len(CLASS_NAMES)))
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predicted, labels=range(len(CLASS_NAMES)), zero_division=0
    )
    check(np.isclose(accuracy_score(labels, predicted), metrics["accuracy"]), "V2 accuracy recomputes", results)
    check(np.isclose(precision.mean(), metrics["macro_precision"]), "V2 macro precision recomputes", results)
    check(np.isclose(recall.mean(), metrics["macro_recall"]), "V2 macro recall recomputes", results)
    check(np.isclose(f1.mean(), metrics["macro_f1"]), "V2 macro-F1 recomputes", results)
    check(matrix.tolist() == metrics["confusion_matrix"], "V2 confusion matrix recomputes", results)
    check(int(matrix[0, 1] + matrix[1, 0]) == 23, "V2 glioma/meningioma pairwise errors equal 23", results)

    expected_metrics = selection["verified_test_metrics"]
    check(
        expected_metrics["samples"] == 896
        and np.isclose(expected_metrics["accuracy"], 0.9375)
        and np.isclose(expected_metrics["macro_precision"], 0.9315602786058244)
        and np.isclose(expected_metrics["macro_recall"], 0.9425142564187508)
        and np.isclose(expected_metrics["macro_f1"], 0.9358515320873503),
        "deployment metrics match verified final values",
        results,
    )

    comparison = load_json(REPORTS_DIR / "metrics" / "v2" / "final_v1_vs_v2_heldout_comparison.json")
    check(comparison["samples"] == 896, "V1/V2 comparison uses 896 images", results)
    check(comparison["test_manifest_sha256"] == metrics["test_manifest_sha256"], "V1/V2 comparison uses same test manifest", results)
    check(comparison["models"][0]["version"] == "Version 1", "historical V1 is labeled", results)
    check(comparison["models"][1]["version"] == "Version 2", "final V2 is labeled", results)

    model = build_model(checkpoint["architecture"], pretrained=False).eval()
    model.load_state_dict(checkpoint["model_state_dict"])
    synthetic = Image.new("RGB", (341, 257), color=(64, 96, 128))
    tensor = build_transform("efficientnet_b0", training=False, input_size=224)(synthetic).unsqueeze(0)
    check(tuple(tensor.shape) == (1, 3, 224, 224), "V2 preprocessing produces 1x3x224x224", results)
    with torch.no_grad():
        normal_probabilities = torch.softmax(model(tensor), dim=1)[0].numpy()
    target_index = int(normal_probabilities.argmax())
    cam = GradCAM(model, last_convolution(model))
    try:
        heatmap, gradcam_probabilities = cam.generate(tensor, target_index)
    finally:
        cam.close()
    check(heatmap.shape == (224, 224), "V2 Grad-CAM uses 224x224 model input", results)
    check(
        np.allclose(normal_probabilities, gradcam_probabilities, rtol=1e-5, atol=1e-6),
        "V2 Grad-CAM and normal inference probabilities match",
        results,
    )

    required_docs = [
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "docs" / "brain_tumor_mri_research_report.md",
        PROJECT_ROOT / "docs" / "project_summary.md",
        PROJECT_ROOT / "docs" / "cv_project_entry.md",
        PROJECT_ROOT / "docs" / "interview_questions.md",
        PROJECT_ROOT / "docs" / "final_consolidated_report.md",
        PROJECT_ROOT / "docs" / "MODEL_CARD.md",
        PROJECT_ROOT / "docs" / "DATASET.md",
    ]
    text = "\n".join(path.read_text() for path in required_docs)
    check(all(path.is_file() and path.stat().st_size for path in required_docs), "release documentation is present", results)
    check("93.75%" in text and "93.59%" in text and "23" in text, "documentation contains final V2 results", results)
    check("patient-level independence could not be verified" in text.lower(), "documentation states patient limitation", results)
    check("completely untouched" in text, "documentation discloses project-lifetime test use", results)

    app_text = (PROJECT_ROOT / "app" / "app.py").read_text()
    check("selected_model_v2.json" in app_text, "Streamlit uses the V2 selection record", results)
    check("NOT a medical device" in app_text and "clinical diagnosis" in app_text, "app retains non-clinical disclaimer", results)

    report = {
        "status": "pass",
        "scope": "frozen Version 2 release verification; no training or model selection",
        "checks_passed": len(results),
        "checkpoint_sha256": checkpoint_digest,
        "checks": results,
        "limitations": [
            "Patient-level independence cannot be verified.",
            "V1 had previously been evaluated on the official test manifest before V2 development.",
            "This verification does not establish clinical validity or external generalization.",
        ],
    }
    if args.output:
        output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"PASS: {len(results)} checks. Saved {output.relative_to(PROJECT_ROOT)}")
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
