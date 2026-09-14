from __future__ import annotations

"""Historical Version 1 diagnostic retained for reproducibility."""

import json
import re
import tempfile
from collections import Counter
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from torch.utils.data import DataLoader

from src.config import CLASS_NAMES, METADATA_DIR, METRICS_DIR, MODELS_DIR, PROJECT_ROOT, REPORTS_DIR
from src.data import ManifestDataset, build_transform, normalization_for
from src.gradcam import GradCAM, overlay
from src.models import build_model, last_convolution


OUTPUT_DIR = REPORTS_DIR / "diagnostics" / "notumor_original_testing"


def natural_number(path: Path) -> int:
    match = re.search(r"(\d+)$", path.stem)
    return int(match.group(1)) if match else 10**9


def load_selected():
    selection = json.loads((MODELS_DIR / "selected_model.json").read_text())["selected"]
    checkpoint = torch.load(PROJECT_ROOT / selection["checkpoint"], map_location="cpu", weights_only=False)
    model = build_model(checkpoint["architecture"], pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return selection, checkpoint, model


def evaluation_pipeline_probabilities(paths: list[Path], checkpoint: dict, model) -> np.ndarray:
    frame = pd.DataFrame(
        {
            "relative_path": [str(path.relative_to(PROJECT_ROOT)) for path in paths],
            "class_name": "notumor",
            "class_index": CLASS_NAMES.index("notumor"),
        }
    )
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as handle:
        manifest_path = Path(handle.name)
        frame.to_csv(handle, index=False)
    try:
        dataset = ManifestDataset(
            manifest_path,
            checkpoint["architecture"],
            training=False,
            input_size=checkpoint["input_size"],
        )
        loader = DataLoader(dataset, batch_size=24, shuffle=False, num_workers=0)
        values = []
        with torch.no_grad():
            for tensors, _labels, _indices in loader:
                values.append(torch.softmax(model(tensors), dim=1).numpy())
        return np.concatenate(values)
    finally:
        manifest_path.unlink(missing_ok=True)


def streamlit_pipeline_probabilities(paths: list[Path], checkpoint: dict, model) -> tuple[np.ndarray, list[np.ndarray]]:
    transform = build_transform(
        checkpoint["architecture"], training=False, input_size=checkpoint["input_size"]
    )
    cam = GradCAM(model, last_convolution(model))
    values, heatmaps = [], []
    try:
        for path in paths:
            with Image.open(BytesIO(path.read_bytes())) as source:
                source.load()
                rgb = source.convert("RGB")
                tensor = transform(rgb).unsqueeze(0)
            with torch.no_grad():
                predicted_index = int(model(tensor).argmax(dim=1).item())
            with torch.enable_grad():
                heatmap, probabilities = cam.generate(tensor, predicted_index)
            values.append(probabilities)
            heatmaps.append(heatmap)
    finally:
        cam.close()
    return np.stack(values), heatmaps


def membership_table(paths: list[Path]) -> pd.DataFrame:
    audit = pd.read_csv(METADATA_DIR / "dataset_manifest.csv").set_index("relative_path", drop=False)
    rows = []
    for path in paths:
        relative = str(path.relative_to(PROJECT_ROOT))
        row = audit.loc[relative]
        rows.append(
            {
                "relative_path": relative,
                "manifest_status": row["status"],
                "final_split": row["final_split"],
                "exclusion_reason": row["exclusion_reason"] if pd.notna(row["exclusion_reason"]) else "",
                "duplicate_of": row["duplicate_of"] if pd.notna(row["duplicate_of"]) else "",
                "width": int(row["width"]),
                "height": int(row["height"]),
                "original_mode": row["original_mode"],
                "decoded_format": row["decoded_format"],
            }
        )
    return pd.DataFrame(rows)


def generate_error_gradcams(error_rows: pd.DataFrame, checkpoint: dict, model, limit: int = 6) -> list[dict]:
    chosen = error_rows.head(limit)
    if chosen.empty:
        return []
    transform = build_transform(
        checkpoint["architecture"], training=False, input_size=checkpoint["input_size"]
    )
    cam = GradCAM(model, last_convolution(model))
    outputs = []
    panels = []
    try:
        for _, row in chosen.iterrows():
            path = PROJECT_ROOT / row["relative_path"]
            with Image.open(path) as source:
                source.load()
                rgb = source.convert("RGB")
                tensor = transform(rgb).unsqueeze(0)
                target_index = CLASS_NAMES.index(row["predicted_class"])
                with torch.enable_grad():
                    heatmap, probabilities = cam.generate(tensor, target_index)
                result = overlay(rgb, heatmap).resize((320, 320))
                original = rgb.resize((320, 320))
            canvas = Image.new("RGB", (640, 366), "white")
            canvas.paste(original, (0, 0)); canvas.paste(result, (320, 0))
            draw = ImageDraw.Draw(canvas)
            label = f"{path.name}: true No Tumor -> {row['predicted_class']} ({row['confidence']:.1%})"
            draw.text((10, 332), label, fill="black")
            destination = OUTPUT_DIR / f"gradcam_{path.stem}_{row['predicted_class']}.png"
            canvas.save(destination)
            panels.append(canvas)
            outputs.append(
                {
                    "relative_path": row["relative_path"],
                    "predicted_class": row["predicted_class"],
                    "confidence": float(row["confidence"]),
                    "probabilities": {name: float(probabilities[i]) for i, name in enumerate(CLASS_NAMES)},
                    "output": str(destination.relative_to(PROJECT_ROOT)),
                }
            )
    finally:
        cam.close()
    sheet = Image.new("RGB", (1280, 366 * ((len(panels) + 1) // 2)), "white")
    for index, panel in enumerate(panels):
        sheet.paste(panel, ((index % 2) * 640, (index // 2) * 366))
    sheet.save(OUTPUT_DIR / "misclassified_notumor_gradcam_contact_sheet.png")
    return outputs


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selection, checkpoint, model = load_selected()
    source_paths = sorted((PROJECT_ROOT / "data" / "Testing" / "notumor").glob("*"), key=natural_number)
    source_paths = [path for path in source_paths if path.is_file()]
    sample_paths = source_paths[:20]

    evaluation_sample = evaluation_pipeline_probabilities(sample_paths, checkpoint, model)
    streamlit_sample, _sample_heatmaps = streamlit_pipeline_probabilities(sample_paths, checkpoint, model)
    absolute_difference = np.abs(evaluation_sample - streamlit_sample)

    all_probabilities = evaluation_pipeline_probabilities(source_paths, checkpoint, model)
    all_predictions = all_probabilities.argmax(axis=1)
    all_frame = membership_table(source_paths)
    all_frame["true_class"] = "notumor"
    all_frame["predicted_class"] = [CLASS_NAMES[index] for index in all_predictions]
    all_frame["confidence"] = all_probabilities.max(axis=1)
    for index, name in enumerate(CLASS_NAMES):
        all_frame[f"probability_{name}"] = all_probabilities[:, index]
    all_frame.to_csv(OUTPUT_DIR / "all_original_testing_notumor_predictions.csv", index=False)

    sample_frame = all_frame.iloc[:20].copy()
    for index, name in enumerate(CLASS_NAMES):
        sample_frame[f"evaluation_probability_{name}"] = evaluation_sample[:, index]
        sample_frame[f"streamlit_probability_{name}"] = streamlit_sample[:, index]
        sample_frame[f"absolute_difference_{name}"] = absolute_difference[:, index]
    sample_frame.to_csv(OUTPUT_DIR / "twenty_image_pipeline_comparison.csv", index=False)

    final_predictions = pd.read_csv(METRICS_DIR / f"{selection['run_name']}_test_predictions.csv")
    final_manifest = pd.read_csv(METADATA_DIR / "splits" / "test.csv")
    assert set(final_predictions["relative_path"]) == set(final_manifest["relative_path"])
    true_indices = final_predictions["true_class"].map({name: i for i, name in enumerate(CLASS_NAMES)}).to_numpy()
    predicted_indices = final_predictions["predicted_class"].map({name: i for i, name in enumerate(CLASS_NAMES)}).to_numpy()
    final_matrix = confusion_matrix(true_indices, predicted_indices, labels=range(len(CLASS_NAMES)))
    notumor_index = CLASS_NAMES.index("notumor")
    notumor_recall = recall_score(true_indices == notumor_index, predicted_indices == notumor_index)
    notumor_f1 = f1_score(true_indices == notumor_index, predicted_indices == notumor_index)

    errors = all_frame[all_frame["predicted_class"] != "notumor"].sort_values(
        ["confidence", "relative_path"], ascending=[False, True]
    )
    gradcams = generate_error_gradcams(errors, checkpoint, model)

    mean, std = normalization_for(checkpoint["architecture"])
    with Image.open(sample_paths[0]) as source:
        source.load(); rgb = source.convert("RGB")
        raw_array = np.asarray(rgb, dtype=np.float32) / 255.0
        tensor = build_transform(
            checkpoint["architecture"], training=False, input_size=checkpoint["input_size"]
        )(rgb)

    summary = {
        "selected_model": selection,
        "class_mapping": {str(index): name for index, name in enumerate(CLASS_NAMES)},
        "checkpoint_class_names": checkpoint["class_names"],
        "preprocessing": {
            "decode": "Pillow content decoding",
            "color_conversion": "RGB",
            "resize": [checkpoint["input_size"], checkpoint["input_size"]],
            "interpolation": "bilinear with antialiasing",
            "channel_order": "RGB / CHW tensor",
            "pre_normalization_pixel_range_first_sample": [float(raw_array.min()), float(raw_array.max())],
            "normalization_mean": list(mean),
            "normalization_std": list(std),
            "post_normalization_range_first_sample": [float(tensor.min()), float(tensor.max())],
            "tensor_shape": [1, *list(tensor.shape)],
            "normalization_count": 1,
        },
        "twenty_image_sample": {
            "selection": "Te-no_1.jpg through Te-no_20.jpg, natural numeric order",
            "prediction_counts": dict(Counter(sample_frame["predicted_class"])),
            "maximum_absolute_probability_difference": float(absolute_difference.max()),
            "all_probabilities_match_at_atol_1e-6": bool(np.allclose(evaluation_sample, streamlit_sample, atol=1e-6, rtol=1e-6)),
            "membership_counts": dict(Counter(sample_frame["final_split"])),
        },
        "all_original_testing_notumor": {
            "images": len(all_frame),
            "prediction_counts": dict(Counter(all_frame["predicted_class"])),
            "membership_counts": dict(Counter(all_frame["final_split"])),
            "exclusion_reason_counts": dict(Counter(value for value in all_frame["exclusion_reason"] if value)),
        },
        "final_leakage_controlled_test": {
            "images": len(final_predictions),
            "confusion_matrix_class_order": list(CLASS_NAMES),
            "confusion_matrix": final_matrix.tolist(),
            "notumor_support": int((true_indices == notumor_index).sum()),
            "notumor_recall": float(notumor_recall),
            "notumor_f1": float(notumor_f1),
            "source": str((METRICS_DIR / f"{selection['run_name']}_test_predictions.csv").relative_to(PROJECT_ROOT)),
        },
        "gradcam_cases": gradcams,
    }
    (OUTPUT_DIR / "diagnostic_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
