from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from matplotlib import colormaps
import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.config import CLASS_NAMES, FIGURES_DIR, METRICS_DIR, MODELS_DIR, PROJECT_ROOT, ensure_directories, get_device
from src.data import build_transform
from src.models import build_model, last_convolution


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self.forward_handle = target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, _module, _inputs, output):
        self.activations = output.detach()
        if output.requires_grad:
            output.register_hook(self._save_gradient)

    def _save_gradient(self, gradient):
        self.gradients = gradient.detach()

    def generate(self, tensor, target_index: int):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(tensor)
        logits[:, target_index].sum().backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        heatmap = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        heatmap = torch.nn.functional.interpolate(heatmap, tensor.shape[-2:], mode="bilinear", align_corners=False)
        heatmap = heatmap[0, 0]
        heatmap -= heatmap.min()
        heatmap /= heatmap.max().clamp_min(1e-8)
        return heatmap.cpu().numpy(), torch.softmax(logits.detach(), dim=1)[0].cpu().numpy()

    def close(self):
        self.forward_handle.remove()


def overlay(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.42) -> Image.Image:
    base = image.convert("RGB")
    scalar_map = Image.fromarray(np.asarray(heatmap, dtype=np.float32), mode="F")
    resized = scalar_map.resize(base.size, resample=Image.Resampling.LANCZOS)
    display_heatmap = np.clip(np.asarray(resized, dtype=np.float32), 0.0, 1.0)
    colored = (colormaps["jet"](display_heatmap)[..., :3] * 255).astype(np.uint8)
    return Image.blend(base, Image.fromarray(colored), alpha)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_cases(predictions: pd.DataFrame) -> pd.DataFrame:
    chosen = []
    used = set()
    for class_name in CLASS_NAMES:
        subset = predictions[(predictions["true_class"] == class_name) & predictions["correct"]]
        for label, ordered in (
            ("correct_high_confidence", subset.sort_values("confidence", ascending=False)),
            ("correct_lower_confidence", subset.sort_values("confidence", ascending=True)),
        ):
            if not ordered.empty:
                row = ordered.iloc[0].copy(); row["case_type"] = label
                if row["relative_path"] not in used:
                    chosen.append(row); used.add(row["relative_path"])
    errors = predictions[~predictions["correct"]]
    for label, ordered in (
        ("incorrect_high_confidence", errors.sort_values("confidence", ascending=False)),
        ("incorrect_lower_confidence", errors.sort_values("confidence", ascending=True)),
    ):
        for _, row in ordered.head(4).iterrows():
            row = row.copy(); row["case_type"] = label
            if row["relative_path"] not in used:
                chosen.append(row); used.add(row["relative_path"])
    return pd.DataFrame(chosen)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, default=MODELS_DIR / "selected_model_v2.json")
    args = parser.parse_args()
    ensure_directories()
    selection = json.loads(args.selection.read_text())["selected"]
    checkpoint_path = PROJECT_ROOT / selection["checkpoint"]
    if selection.get("checkpoint_sha256") and sha256(checkpoint_path) != selection["checkpoint_sha256"]:
        raise ValueError("Checkpoint checksum does not match the selection record.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if tuple(checkpoint["class_names"]) != tuple(CLASS_NAMES):
        raise ValueError("Checkpoint class order does not match the project mapping.")
    architecture = checkpoint["architecture"]
    model = build_model(architecture, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    device = get_device(); model.to(device).eval()
    cam = GradCAM(model, last_convolution(model))
    predictions_path = (
        PROJECT_ROOT / selection["test_predictions"]
        if selection.get("test_predictions")
        else METRICS_DIR / f"{selection['run_name']}_test_predictions.csv"
    )
    predictions = pd.read_csv(predictions_path)
    cases = choose_cases(predictions)
    transform = build_transform(architecture, training=False, input_size=checkpoint["input_size"])
    metadata = []
    version_name = "v2" if args.selection.name == "selected_model_v2.json" else "v1"
    output_dir = FIGURES_DIR / version_name / "gradcam_cases"
    output_dir.mkdir(parents=True, exist_ok=True)
    for order, (_, row) in enumerate(cases.iterrows(), start=1):
        path = PROJECT_ROOT / row["relative_path"]
        with Image.open(path) as source:
            rgb = source.convert("RGB")
            tensor = transform(rgb).unsqueeze(0).to(device)
            predicted_index = CLASS_NAMES.index(row["predicted_class"])
            heatmap, probabilities = cam.generate(tensor, predicted_index)
            result = overlay(rgb, heatmap)
        output_path = output_dir / f"{order:02d}_{row['case_type']}_{row['true_class']}_{row['predicted_class']}.png"
        result.save(output_path)
        metadata.append(
            {
                "case_type": row["case_type"],
                "relative_path": row["relative_path"],
                "true_class": row["true_class"],
                "predicted_class": row["predicted_class"],
                "confidence": float(row["confidence"]),
                "gradcam_target": row["predicted_class"],
                "output": str(output_path.relative_to(PROJECT_ROOT)),
                **{f"probability_{name}": float(probabilities[i]) for i, name in enumerate(CLASS_NAMES)},
            }
        )
    cam.close()
    manifest_path = FIGURES_DIR / version_name / "gradcam_case_manifest.csv"
    pd.DataFrame(metadata).to_csv(manifest_path, index=False)
    print(f"Saved {len(metadata)} Grad-CAM cases to {output_dir}")


if __name__ == "__main__":
    main()
