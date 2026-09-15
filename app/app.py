from __future__ import annotations

import hashlib
import json
import os
import sys
from io import BytesIO
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))

import numpy as np
import pandas as pd
import streamlit as st
import torch
from matplotlib import colormaps
from PIL import Image

sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CLASS_NAMES, DISPLAY_NAMES, IMAGENET_MEAN, IMAGENET_STD, MODELS_DIR  # noqa: E402
from src.data import build_transform  # noqa: E402
from src.gradcam import GradCAM  # noqa: E402
from src.models import build_model, last_convolution  # noqa: E402
from scripts.fetch_final_model import DEFAULT_URL, EXPECTED_SHA256, ensure_final_model  # noqa: E402


st.set_page_config(
    page_title="Brain MRI AI — Tumor Classification & Explainability",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

FINAL_SELECTION_PATH = MODELS_DIR / "selected_model_v2.json"

st.markdown(
    """
    <style>
      :root {
        --canvas: #06111c;
        --canvas-deep: #030a11;
        --panel: #0b1b2a;
        --panel-raised: #102536;
        --panel-soft: #132b3d;
        --line: #243b4d;
        --line-strong: #315269;
        --text: #eef5f8;
        --text-soft: #c5d2dc;
        --muted: #91a6b5;
        --accent: #42b8c7;
        --accent-soft: #183b49;
        --success: #67c7a5;
        --warning: #e5b65c;
      }
      html, body, [class*="css"] { font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
      .stApp {
        background:
          radial-gradient(circle at 70% -20%, rgba(37,102,126,.18), transparent 35%),
          var(--canvas);
        color: var(--text);
      }
      [data-testid="stHeader"] { background: rgba(6,17,28,.88); border-bottom: 1px solid rgba(49,82,105,.45); }
      .block-container { max-width: 1380px; padding-top: 2.5rem; padding-bottom: 3rem; }
      [data-testid="stAppViewContainer"] h1,
      [data-testid="stAppViewContainer"] h2,
      [data-testid="stAppViewContainer"] h3,
      [data-testid="stAppViewContainer"] h4 { color: var(--text) !important; letter-spacing: -.02em; }
      [data-testid="stAppViewContainer"] p,
      [data-testid="stAppViewContainer"] label,
      [data-testid="stAppViewContainer"] li { color: var(--text-soft); }
      [data-testid="stCaptionContainer"],
      [data-testid="stCaptionContainer"] p {
        color: #a9bac6 !important;
        opacity: 1 !important;
      }
      [data-testid="stSidebar"] {
        background: #081624;
        border-right: 1px solid var(--line);
      }
      [data-testid="stSidebar"] hr { border-color: var(--line); }
      .sidebar-brand {
        display: flex; align-items: center; gap: .75rem;
        padding: .25rem 0 .65rem;
      }
      .sidebar-mark {
        width: 2.25rem; height: 2.25rem; display: grid; place-items: center;
        background: var(--accent-soft); border: 1px solid #2a6573; border-radius: 8px;
        color: #83d6df !important; font-weight: 800; font-size: .8rem; letter-spacing: .04em;
      }
      .sidebar-title { color: var(--text) !important; font-weight: 750; font-size: 1rem; line-height: 1.2; }
      .sidebar-subtitle { color: var(--muted) !important; font-size: .74rem; margin-top: .12rem; }
      .hero {
        display: flex; align-items: center; justify-content: space-between; gap: 2rem;
        padding: 1.25rem 1.45rem; margin-bottom: 1rem;
        border: 1px solid var(--line-strong); border-radius: 12px;
        background: linear-gradient(110deg, #0b1b2a 0%, #0d2232 70%, #102e3c 100%);
        box-shadow: 0 14px 34px rgba(0,0,0,.22);
      }
      .hero-badge {
        color: #72cbd5 !important; margin-bottom: .35rem;
        font-size: .68rem; font-weight: 750; letter-spacing: .14em; text-transform: uppercase;
      }
      .hero h1 { margin: 0; color: var(--text) !important; font-size: clamp(1.65rem, 3vw, 2.25rem); line-height: 1.12; }
      .hero p { max-width: 760px; margin: .42rem 0 0; color: var(--text-soft) !important; font-size: .92rem; }
      .system-status {
        min-width: 170px; display: flex; align-items: center; gap: .65rem;
        padding: .7rem .85rem; border: 1px solid #315064; border-radius: 9px;
        background: rgba(4,13,21,.45);
      }
      .status-dot { width: .55rem; height: .55rem; border-radius: 50%; background: var(--success); box-shadow: 0 0 0 4px rgba(103,199,165,.12); }
      .status-copy small { display: block; color: var(--muted); font-size: .62rem; letter-spacing: .12em; }
      .status-copy strong { display: block; color: var(--text); margin-top: .15rem; font-size: .82rem; }
      .section-kicker { color: #6ec8d2; font-weight: 750; letter-spacing: .12em; text-transform: uppercase; font-size: .68rem; margin-top: 1.35rem; }
      .result-card {
        background: linear-gradient(145deg, #102536, #0d202f); border: 1px solid var(--line-strong);
        border-radius: 11px; padding: 1.25rem 1.35rem; margin: .2rem 0 .8rem;
        box-shadow: 0 12px 26px rgba(0,0,0,.18);
      }
      .result-label { color: var(--muted); font-size: .7rem; font-weight: 700; letter-spacing: .11em; text-transform: uppercase; }
      .result-name { color: var(--text); font-size: 1.9rem; line-height: 1.2; font-weight: 760; margin: .32rem 0; }
      .result-confidence { color: #74d0da; font-size: .98rem; font-weight: 680; }
      .prob-row { display:flex; justify-content:space-between; align-items:center; margin-top:.7rem; color:var(--text-soft); font-weight:620; }
      .fine-print { color: var(--muted); font-size: .84rem; line-height: 1.5; }
      .app-footer {
        margin-top: 3rem; padding: 1.15rem 0 .25rem; border-top: 1px solid var(--line);
        text-align: center; color: var(--muted); line-height: 1.45;
      }
      .app-footer .footer-name { color: var(--text-soft); font-size: .86rem; font-weight: 650; }
      .app-footer .footer-project { color: var(--muted); font-size: .82rem; }
      .app-footer .footer-tools { color: #6f8493; font-size: .72rem; margin-top: .22rem; }
      div[data-testid="stFileUploader"] {
        background: var(--panel); color: var(--text); border: 1px solid var(--line-strong);
        border-radius: 11px; padding: .65rem;
      }
      div[data-testid="stFileUploader"] * { color: var(--text-soft) !important; }
      [data-testid="stFileUploaderDropzone"] {
        background: #0d202f !important;
        border: 1px dashed #3b6377 !important;
        border-radius: 8px !important;
      }
      [data-testid="stFileUploaderDropzoneInstructions"],
      [data-testid="stFileUploaderDropzoneInstructions"] span,
      [data-testid="stFileUploaderDropzoneInstructions"] small {
        color: var(--text-soft) !important;
        opacity: 1 !important;
      }
      [data-testid="stFileUploaderDropzoneInstructions"] svg {
        color: var(--accent) !important;
        fill: var(--accent) !important;
      }
      div[data-testid="stMetric"] {
        background: var(--panel); border: 1px solid var(--line); border-radius: 9px; padding: .8rem .9rem;
        min-height: 104px;
      }
      div[data-testid="stMetric"] [data-testid="stMetricLabel"] p { color: var(--muted) !important; font-size: .75rem; }
      div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: var(--text) !important; }
      .stButton > button {
        min-height: 2.85rem; border-radius: 8px; font-weight: 700;
        box-shadow: 0 8px 20px rgba(0,0,0,.22);
      }
      [data-testid="stBaseButton-primary"] {
        background: #237f91 !important; border: 1px solid #49aebe !important; color: #ffffff !important;
      }
      [data-testid="stBaseButton-primary"] * { color: #ffffff !important; }
      [data-testid="stBaseButton-secondary"] {
        background: #172c3b !important; border: 1px solid #426174 !important; color: var(--text) !important;
      }
      [data-testid="stBaseButton-secondary"] * { color: var(--text) !important; }
      [data-testid="stProgressBar"] > div { background: #1a3040 !important; }
      [data-testid="stProgressBar"] > div > div { background-color: var(--accent) !important; }
      button[role="tab"] { background: transparent !important; }
      button[role="tab"] p { color: var(--muted) !important; font-weight: 650; }
      button[role="tab"][aria-selected="true"] p { color: #8bd9e2 !important; }
      button[role="tab"][aria-selected="true"] { border-bottom-color: var(--accent) !important; }
      div[data-testid="stImage"] { background: var(--canvas-deep); border: 1px solid var(--line); border-radius: 10px; padding: .65rem; }
      div[data-testid="stImage"] img { border-radius: 6px; }
      div[data-testid="stExpander"] {
        color: var(--text); border: 1px solid var(--line); border-radius: 9px;
        overflow: hidden; background: var(--panel) !important;
      }
      div[data-testid="stExpander"] details { background: var(--panel) !important; }
      div[data-testid="stExpander"] summary {
        background: var(--panel-raised) !important;
        color: var(--text-soft) !important;
        border-radius: 8px;
      }
      div[data-testid="stExpander"] summary:hover { background: var(--panel-soft) !important; }
      div[data-testid="stExpander"] summary *,
      div[data-testid="stExpander"] summary p,
      div[data-testid="stExpander"] summary span,
      div[data-testid="stExpander"] summary svg {
        color: var(--text-soft) !important;
        fill: currentColor !important;
        opacity: 1 !important;
      }
      div[data-testid="stExpander"] details > div,
      div[data-testid="stExpander"] details > div * { color: var(--text-soft) !important; }
      [data-testid="stSidebar"] h1,
      [data-testid="stSidebar"] h2,
      [data-testid="stSidebar"] h3,
      [data-testid="stSidebar"] h4,
      [data-testid="stSidebar"] p,
      [data-testid="stSidebar"] label,
      [data-testid="stSidebar"] li { color: var(--text-soft) !important; }
      [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
      [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
        color: #a9bac6 !important;
        opacity: 1 !important;
      }
      [data-testid="stSidebar"] div[data-testid="stExpander"] {
        background: var(--panel) !important;
        border-color: var(--line) !important;
      }
      [data-testid="stSidebar"] div[data-testid="stExpander"] details,
      [data-testid="stSidebar"] div[data-testid="stExpander"] summary {
        background: var(--panel-raised) !important;
      }
      [data-testid="stSidebar"] div[data-testid="stExpander"] details > div {
        background: var(--panel) !important;
      }
      [data-testid="stSidebar"] div[data-testid="stExpander"] summary:hover {
        background: var(--panel-soft) !important;
      }
      [data-testid="stSidebar"] div[data-testid="stExpander"] summary *,
      [data-testid="stSidebar"] div[data-testid="stExpander"] details > div * {
        color: var(--text-soft) !important;
        fill: currentColor !important;
        opacity: 1 !important;
      }
      [data-testid="stAlert"] { background: #102536 !important; border: 1px solid var(--line-strong) !important; border-radius: 9px !important; }
      [data-testid="stAlert"] * { color: var(--text-soft) !important; }
      .sidebar-warning {
        display: flex; gap: .7rem; align-items: flex-start;
        margin: .4rem 0 .85rem; padding: .8rem .85rem;
        border: 1px solid #765f32; border-radius: 8px;
        background: #2b2519; color: #f5d99f !important;
        font-weight: 620; line-height: 1.45;
      }
      .sidebar-warning span { color: #f5d99f !important; }
      .sidebar-warning .warning-icon { font-size: 1.15rem; line-height: 1.35; }
      hr { border-color: var(--line) !important; }
      [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }
      @media (max-width: 700px) {
        .block-container { padding-top: 2rem; }
        .hero { align-items: flex-start; flex-direction: column; padding: 1rem; }
        .system-status { min-width: 0; width: 100%; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_footer() -> None:
    st.markdown(
        """
        <footer class="app-footer">
          <div class="footer-name">Developed by Palwisha Mirani</div>
          <div class="footer-project">Brain Tumor MRI Classification &amp; Explainable AI</div>
          <div class="footer-tools">Built with Python, PyTorch, Streamlit &amp; Grad-CAM</div>
        </footer>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_selected_model():
    selection = json.loads(FINAL_SELECTION_PATH.read_text())["selected"]
    if selection.get("checkpoint_sha256") != EXPECTED_SHA256:
        raise ValueError("The selected checkpoint digest does not match the approved final V2 release asset.")
    checkpoint_path = PROJECT_ROOT / selection["checkpoint"]
    try:
        ensure_final_model(destination=checkpoint_path, url=DEFAULT_URL)
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            "The approved final V2 checkpoint could not be downloaded and verified. "
            "Analysis is unavailable until the verified release asset can be loaded."
        ) from exc
    checkpoint_digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if checkpoint_digest != selection["checkpoint_sha256"]:
        raise ValueError("The final Version 2 checkpoint checksum does not match its frozen selection record.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected_metadata = {
        "run_name": selection["run_name"],
        "architecture": selection["architecture"],
        "input_size": selection["input_size"],
        "unfrozen_blocks": selection["unfrozen_blocks"],
        "best_epoch": selection["best_epoch"],
    }
    mismatches = [key for key, expected in expected_metadata.items() if checkpoint.get(key) != expected]
    if mismatches:
        raise ValueError(f"The final Version 2 checkpoint metadata does not match its selection record: {mismatches}")
    if tuple(checkpoint.get("class_names", ())) != tuple(CLASS_NAMES):
        raise ValueError("The final Version 2 checkpoint class mapping is incompatible with the application.")
    if not np.allclose(checkpoint.get("normalization_mean"), IMAGENET_MEAN) or not np.allclose(
        checkpoint.get("normalization_std"), IMAGENET_STD
    ):
        raise ValueError("The final Version 2 checkpoint normalization is incompatible with the application.")
    model = build_model(checkpoint["architecture"], pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint, selection


@st.cache_data
def load_verified_test_performance():
    selection = json.loads(FINAL_SELECTION_PATH.read_text())["selected"]
    metrics_path = PROJECT_ROOT / selection["test_metrics"]
    metrics = json.loads(metrics_path.read_text())
    if metrics.get("run_name") != selection["run_name"]:
        raise ValueError("The test metrics do not match the selected model run.")
    if metrics.get("architecture") != selection["architecture"]:
        raise ValueError("The test metrics do not match the selected architecture.")
    if metrics.get("checkpoint") != selection["checkpoint"]:
        raise ValueError("The test metrics do not match the selected checkpoint.")
    if metrics.get("checkpoint_sha256") != selection["checkpoint_sha256"]:
        raise ValueError("The test metrics do not match the frozen checkpoint checksum.")
    if metrics.get("split") != "test":
        raise ValueError("The saved metrics are not labeled as held-out test results.")
    required = ("accuracy", "macro_f1", "macro_precision", "macro_recall", "per_class", "confusion_matrix")
    if any(key not in metrics for key in required):
        raise ValueError("The held-out test artifact is missing required metrics.")
    expected_metrics = selection["verified_test_metrics"]
    for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
        if not np.isclose(metrics[key], expected_metrics[key], rtol=0.0, atol=1e-12):
            raise ValueError(f"The saved held-out test {key} does not match the frozen deployment record.")
    if metrics.get("samples") != expected_metrics["samples"]:
        raise ValueError("The held-out test sample count does not match the frozen deployment record.")
    return metrics, str(metrics_path.relative_to(PROJECT_ROOT))


def resize_heatmap_for_display(heatmap: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize a model-resolution Grad-CAM map to the original image size for display only."""
    scalar_map = Image.fromarray(np.asarray(heatmap, dtype=np.float32), mode="F")
    resized = scalar_map.resize(size, resample=Image.Resampling.LANCZOS)
    return np.clip(np.asarray(resized, dtype=np.float32), 0.0, 1.0)


def make_heatmap_image(display_heatmap: np.ndarray) -> Image.Image:
    colored = (colormaps["turbo"](display_heatmap)[..., :3] * 255).astype(np.uint8)
    return Image.fromarray(colored)


def overlay_on_original(image: Image.Image, display_heatmap: np.ndarray, alpha: float = 0.42) -> Image.Image:
    """Blend a display-resolution heatmap onto the untouched-size RGB upload."""
    base = image.convert("RGB")
    expected_shape = (base.height, base.width)
    if display_heatmap.shape != expected_shape:
        raise ValueError(f"Display heatmap shape {display_heatmap.shape} does not match image shape {expected_shape}.")
    colored = (colormaps["jet"](display_heatmap)[..., :3] * 255).astype(np.uint8)
    return Image.blend(base, Image.fromarray(colored), alpha)


def analyze(image: Image.Image) -> dict:
    model, checkpoint, selection = load_selected_model()
    original_image = image.convert("RGB")
    transform = build_transform(
        checkpoint["architecture"], training=False, input_size=checkpoint["input_size"]
    )
    # This checkpoint-sized tensor is exclusively for model inference. The UI retains and
    # displays original_image at the uploaded pixel dimensions.
    tensor = transform(original_image).unsqueeze(0)
    with torch.no_grad():
        normal_probabilities = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
        predicted_index = int(normal_probabilities.argmax())

    cam = GradCAM(model, last_convolution(model))
    try:
        with torch.enable_grad():
            heatmap, gradcam_probabilities = cam.generate(tensor, predicted_index)
    finally:
        cam.close()

    if not np.allclose(normal_probabilities, gradcam_probabilities, rtol=1e-5, atol=1e-6):
        maximum_difference = float(np.max(np.abs(normal_probabilities - gradcam_probabilities)))
        raise RuntimeError(
            f"Grad-CAM and normal inference probabilities diverged (maximum absolute difference {maximum_difference:.3e})."
        )

    display_heatmap = resize_heatmap_for_display(heatmap, original_image.size)
    predicted_class = CLASS_NAMES[predicted_index]
    return {
        "predicted_class": predicted_class,
        "probabilities": normal_probabilities,
        "overlay": overlay_on_original(original_image, display_heatmap),
        "heatmap": make_heatmap_image(display_heatmap),
        "original_dimensions": original_image.size,
        "model_input_dimensions": (checkpoint["input_size"], checkpoint["input_size"]),
        "display_dimensions": original_image.size,
        "selection": selection,
        "checkpoint": checkpoint,
        "gradcam_probability_max_abs_difference": float(
            np.max(np.abs(normal_probabilities - gradcam_probabilities))
        ),
    }


try:
    test_performance, test_metrics_source = load_verified_test_performance()
except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
    test_performance = None
    test_metrics_source = None
    performance_error = str(exc)

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
          <div class="sidebar-mark">MR</div>
          <div>
            <div class="sidebar-title">Brain MRI AI</div>
            <div class="sidebar-subtitle">Tumor Classification &amp; Explainability</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown("**Analysis Engine**")
    st.write("EfficientNet-B0")
    st.caption("224×224 input")
    st.markdown("**Classes**")
    st.write("Glioma · Meningioma · No Tumor · Pituitary")
    st.markdown("**Evaluation**")
    st.write("Leakage-controlled image-level evaluation")
    st.divider()
    st.markdown(
        '<div class="sidebar-warning"><span class="warning-icon">⚠️</span>'
        '<span>Research &amp; educational use only.</span></div>',
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="hero">
      <div>
        <div class="hero-badge">MRI classification workspace</div>
        <h1>Brain MRI Review</h1>
        <p>Review the model classification, class probabilities, and attention overlay in one imaging-focused workspace.</p>
      </div>
      <div class="system-status">
        <span class="status-dot"></span>
        <div class="status-copy"><small>MODEL STATUS</small><strong>READY</strong></div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.error(
    "This application is for research and educational demonstration only. It is NOT a medical device and must not be used for clinical diagnosis, treatment, triage, or patient-care decisions.",
    icon="🚫",
)

st.markdown('<div class="section-kicker">System validation</div>', unsafe_allow_html=True)
st.header("Model Performance")
if test_performance is None:
    st.warning(
        "Verified final held-out test metrics are unavailable. Final test evaluation is required before performance can be displayed. "
        f"Artifact check: {performance_error}"
    )
else:
    metric_1, metric_2, metric_3, metric_4, metric_5 = st.columns(5)
    metric_1.metric("Held-out Test Accuracy", f"{test_performance['accuracy']:.2%}")
    metric_2.metric("Macro Precision", f"{test_performance['macro_precision']:.2%}")
    metric_3.metric("Macro Recall", f"{test_performance['macro_recall']:.2%}")
    metric_4.metric("Macro F1-score", f"{test_performance['macro_f1']:.2%}")
    metric_5.metric("Held-out Images", f"{test_performance['samples']:,}")
    st.info(
        "Evaluated on a leakage-controlled held-out image-level test set (896 images). "
        "Patient-level independence could not be verified due to unavailable patient identifiers.",
        icon="ℹ️",
    )

    per_class_frame = pd.DataFrame(
        [
            {
                "Class": DISPLAY_NAMES[class_name],
                "F1-score": test_performance["per_class"][class_name]["f1"],
                "Support": test_performance["per_class"][class_name]["support"],
            }
            for class_name in CLASS_NAMES
        ]
    )
    confusion_frame = pd.DataFrame(
        test_performance["confusion_matrix"],
        index=[f"True {DISPLAY_NAMES[name]}" for name in CLASS_NAMES],
        columns=[f"Pred. {DISPLAY_NAMES[name]}" for name in CLASS_NAMES],
    )
    with st.expander("Detailed Performance"):
        class_panel, confusion_panel = st.columns([0.8, 1.2], gap="large")
        with class_panel:
            st.markdown("#### Per-class F1 scores")
            formatted_per_class = per_class_frame.copy()
            formatted_per_class["F1-score"] = formatted_per_class["F1-score"].map(lambda value: f"{value:.2%}")
            st.dataframe(formatted_per_class, hide_index=True, width="stretch")
        with confusion_panel:
            st.markdown("#### Confusion matrix")
            st.caption("Rows are true classes; columns are model predictions.")
            st.dataframe(confusion_frame, width="stretch")

st.markdown('<div class="section-kicker">01 · Study intake</div>', unsafe_allow_html=True)
st.subheader("Open an MRI image")
st.caption("Accepted formats: JPG, JPEG, and PNG. The image is decoded by content and converted to RGB.")

uploaded = st.file_uploader("MRI image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded is None:
    st.info("Upload an image to enable analysis. Your original file is not modified.", icon="📤")
    st.markdown("#### Workspace ready")
    col1, col2, col3 = st.columns(3)
    col1.metric("Output classes", "4")
    col2.metric("Analysis engine", "EfficientNet-B0")
    col3.metric("Attention review", "Grad-CAM")
    render_footer()
    st.stop()

raw_bytes = uploaded.getvalue()
upload_id = hashlib.sha256(raw_bytes).hexdigest()
try:
    source_image = Image.open(BytesIO(raw_bytes))
    source_image.load()
    original_mode = source_image.mode
    image = source_image.convert("RGB")
except Exception:
    st.error("This file could not be decoded as an image. Please choose a valid JPG or PNG file.")
    render_footer()
    st.stop()

preview, controls = st.columns([1.18, 0.82], gap="large")
with preview:
    st.image(image, caption=f"Original-resolution MRI · {image.width} × {image.height} pixels", width="content")
with controls:
    st.markdown("#### Study details")
    st.write(f"**File:** `{uploaded.name}`")
    st.write(f"**Dimensions:** {image.width} × {image.height} pixels")
    st.write(f"**Decoded mode:** {original_mode} → RGB")
    st.caption("The model will resize this image to its checkpoint-recorded 224×224 input and apply ImageNet normalization once.")
    run_analysis = st.button("Run image review", type="primary", width="stretch")
    st.caption("Analysis runs locally with the saved selected checkpoint.")

if run_analysis:
    with st.spinner("Running classification and generating Grad-CAM…"):
        try:
            st.session_state.analysis_result = analyze(image)
            st.session_state.analysis_upload_id = upload_id
        except FileNotFoundError:
            st.error("The selected checkpoint is unavailable. Restore the saved model artifact and try again.")
            render_footer()
            st.stop()
        except Exception as exc:
            st.error(f"Analysis could not be completed: {exc}")
            render_footer()
            st.stop()

if st.session_state.get("analysis_upload_id") != upload_id:
    st.info("Select **Run image review** to generate a model output for this image.")
    render_footer()
    st.stop()

result = st.session_state.analysis_result
probabilities = result["probabilities"]
predicted_class = result["predicted_class"]
predicted_index = CLASS_NAMES.index(predicted_class)
confidence = float(probabilities[predicted_index])
probability_frame = pd.DataFrame(
    {
        "Class": [DISPLAY_NAMES[name] for name in CLASS_NAMES],
        "Probability": [float(probabilities[index]) for index in range(len(CLASS_NAMES))],
    }
).sort_values("Probability", ascending=False, ignore_index=True)

st.divider()
st.markdown('<div class="section-kicker">02 · Model output</div>', unsafe_allow_html=True)
st.subheader("Image Classification Result")

summary, probability_panel = st.columns([0.9, 1.1], gap="large")
with summary:
    st.markdown(
        f"""
        <div class="result-card">
          <div class="result-label">Classification for this uploaded MRI</div>
          <div class="result-name">{DISPLAY_NAMES[predicted_class]}</div>
          <div class="result-confidence">Model confidence for this MRI: {confidence:.1%}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "This confidence is the probability assigned to this individual prediction. "
        "It is not the model's overall held-out test accuracy."
    )
    if test_performance is not None:
        st.info(
            f"Individual-image confidence: {confidence:.1%} · Separate held-out test accuracy: "
            f"{test_performance['accuracy']:.2%}",
            icon="ℹ️",
        )

with probability_panel:
    st.markdown("#### Class probabilities")
    for index, class_name in enumerate(CLASS_NAMES):
        value = float(probabilities[index])
        st.markdown(
            f'<div class="prob-row"><span>{DISPLAY_NAMES[class_name]}</span><span>{value:.1%}</span></div>',
            unsafe_allow_html=True,
        )
        st.progress(value)

st.markdown("#### Class prediction bar chart")
st.bar_chart(
    probability_frame,
    x="Class",
    y="Probability",
    color="#087F8C",
    horizontal=True,
    height=280,
)

with st.expander("View probability table"):
    display_frame = probability_frame.copy()
    display_frame["Probability"] = display_frame["Probability"].map(lambda value: f"{value:.2%}")
    st.dataframe(display_frame, hide_index=True, width="stretch")

st.divider()
st.markdown('<div class="section-kicker">03 · Image review</div>', unsafe_allow_html=True)
st.subheader("Attention Overlay Review")
st.caption("Warmer colors indicate image regions that more strongly influenced the displayed class score.")

original_tab, overlay_tab, heatmap_tab = st.tabs(["Original MRI", "Grad-CAM overlay", "Attention map"])
with original_tab:
    st.image(
        image,
        caption=f"Original uploaded image · {image.width} × {image.height} pixels",
        width="content",
    )
with overlay_tab:
    st.image(
        result["overlay"],
        caption=(
            f"Grad-CAM overlay for {DISPLAY_NAMES[predicted_class]} · rendered at original resolution "
            f"({result['display_dimensions'][0]} × {result['display_dimensions'][1]} pixels)"
        ),
        width="content",
    )
with heatmap_tab:
    st.image(
        result["heatmap"],
        caption=(
            "Coarse model attention map · calculated at model resolution and interpolated "
            "to the original image dimensions for visualization"
        ),
        width="content",
    )

st.warning(
    "Grad-CAM is a qualitative, coarse model-attention visualization. It is not validated tumor segmentation, anatomical localization, or evidence that the prediction is medically grounded.",
    icon="⚠️",
)

with st.expander("How to interpret this result responsibly"):
    st.markdown(
        """
        - Review all four probabilities, not only the largest value.
        - Treat attention outside the brain, on borders, or on text/artifacts as possible shortcut sensitivity.
        - A plausible-looking heatmap does not validate the classification.
        - This model was evaluated only on an internal image-level split; patient-level independence is unknown.
        - Never use the output to confirm or exclude a medical condition.
        """
    )

st.caption(
    f"Model: {result['selection']['run_name']} · Architecture: {result['checkpoint']['architecture']} · "
    f"Input: {result['checkpoint']['input_size']}×{result['checkpoint']['input_size']} · Device: CPU"
)

render_footer()
