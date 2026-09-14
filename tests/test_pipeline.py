import json
import hashlib

import torch
from PIL import Image

from src.config import CLASS_NAMES, METADATA_DIR, MODELS_DIR, PROJECT_ROOT, SPLITS_DIR
from src.data import build_transform
from src.gradcam import GradCAM
from src.models import build_model, last_convolution


def test_class_mapping_is_canonical():
    mapping = json.loads((METADATA_DIR / "class_mapping.json").read_text())
    assert mapping["class_to_index"] == {name: index for index, name in enumerate(CLASS_NAMES)}


def test_eval_preprocessing_shapes_and_range():
    rgb = Image.new("RGB", (341, 257), color=(64, 96, 128))
    baseline = build_transform("baseline_cnn", training=False, input_size=128)(rgb)
    efficientnet = build_transform("efficientnet_b0", training=False, input_size=224)(rgb)
    assert baseline.shape == (3, 128, 128)
    assert efficientnet.shape == (3, 224, 224)
    assert torch.isfinite(baseline).all() and torch.isfinite(efficientnet).all()


def test_model_output_shapes():
    sample = torch.zeros(2, 3, 224, 224)
    for architecture in ("baseline_cnn", "efficientnet_b0"):
        model = build_model(architecture, pretrained=False).eval()
        with torch.no_grad():
            output = model(sample)
        assert output.shape == (2, len(CLASS_NAMES))


def test_selected_checkpoint_inference_and_gradcam():
    selection = json.loads((MODELS_DIR / "selected_model_v2.json").read_text())["selected"]
    checkpoint_path = PROJECT_ROOT / selection["checkpoint"]
    assert hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() == selection["checkpoint_sha256"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["run_name"] == "v2_b0_224_last3_unfreeze"
    assert checkpoint["input_size"] == 224
    assert checkpoint["unfrozen_blocks"] == 3
    assert checkpoint["best_epoch"] == 3
    assert checkpoint["class_names"] == list(CLASS_NAMES)
    model = build_model(checkpoint["architecture"], pretrained=False).eval()
    model.load_state_dict(checkpoint["model_state_dict"])
    image = Image.new("RGB", (341, 257), color=(64, 96, 128))
    tensor = build_transform(
        checkpoint["architecture"], training=False, input_size=checkpoint["input_size"]
    )(image).unsqueeze(0)
    with torch.no_grad():
        normal_probabilities = torch.softmax(model(tensor), dim=1)[0].numpy()
    predicted_index = int(normal_probabilities.argmax())
    cam = GradCAM(model, last_convolution(model))
    heatmap, probabilities = cam.generate(tensor, target_index=predicted_index)
    cam.close()
    assert heatmap.shape == (checkpoint["input_size"], checkpoint["input_size"])
    assert probabilities.shape == (len(CLASS_NAMES),)
    assert __import__("numpy").allclose(probabilities, normal_probabilities, rtol=1e-5, atol=1e-6)
    assert abs(float(probabilities.sum()) - 1.0) < 1e-5
    assert 0.0 <= float(heatmap.min()) <= float(heatmap.max()) <= 1.0


def test_streamlit_app_uses_verified_v2_artifacts_and_preserves_original_display():
    selection = json.loads((MODELS_DIR / "selected_model_v2.json").read_text())["selected"]
    metrics = json.loads((PROJECT_ROOT / selection["test_metrics"]).read_text())
    assert metrics["samples"] == 896
    assert metrics["accuracy"] == 0.9375
    assert abs(metrics["macro_precision"] - 0.9315602786058244) < 1e-12
    assert abs(metrics["macro_recall"] - 0.9425142564187508) < 1e-12
    assert abs(metrics["macro_f1"] - 0.9358515320873503) < 1e-12
    app_text = (PROJECT_ROOT / "app" / "app.py").read_text()
    assert 'selected_model_v2.json' in app_text
    assert 'original_image.size' in app_text
    assert 'Image.Resampling.LANCZOS' in app_text
    assert 'Patient-level independence could not be verified' in app_text
    assert 'NOT a medical device' in app_text
    assert "It is not the model's overall held-out test accuracy" in app_text
