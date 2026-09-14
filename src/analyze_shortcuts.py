from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image, ImageDraw, ImageOps
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import FIGURES_DIR, METADATA_DIR, PROJECT_ROOT, REPORTS_DIR, SEED, ensure_directories


def save_artifact_review(frame: pd.DataFrame) -> None:
    candidates = pd.concat(
        [
            frame.nlargest(6, "border_to_center_ratio"),
            frame.nsmallest(6, "border_to_center_ratio"),
            frame.nlargest(6, "bright_border_fraction"),
            frame.sort_values("aspect_ratio").iloc[[0, 1, -2, -1]],
        ]
    ).drop_duplicates("relative_path").head(20)
    cell_w, cell_h = 260, 290
    canvas = Image.new("RGB", (cell_w * 5, cell_h * 4), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (_, row) in enumerate(candidates.iterrows()):
        with Image.open(PROJECT_ROOT / row["relative_path"]) as source:
            tile = ImageOps.contain(source.convert("RGB"), (240, 230))
        x = (index % 5) * cell_w + 10
        y = (index // 5) * cell_h + 5
        canvas.paste(tile, (x + (240 - tile.width) // 2, y))
        label = f"{row['class_name']} | {row['filename']}\nAR={row['aspect_ratio']:.2f} B/C={row['border_to_center_ratio']:.2f}"
        draw.multiline_text((x, y + 235), label, fill="black", spacing=3)
    canvas.save(FIGURES_DIR / "artifact_review_extremes.jpg", quality=92)


def main() -> None:
    ensure_directories()
    frame = pd.read_csv(METADATA_DIR / "dataset_manifest.csv")
    included = frame[frame["status"].eq("included")].copy()

    categorical = ["original_mode", "decoded_format", "has_exif", "original_split"]
    numeric = [
        "width", "height", "aspect_ratio", "file_size_bytes", "mean_intensity", "std_intensity",
        "border_mean", "center_mean", "border_to_center_ratio", "dark_border_fraction", "bright_border_fraction",
    ]
    features = categorical + numeric
    train = included[included["final_split"].eq("train")]
    validation = included[included["final_split"].eq("validation")]
    processor = ColumnTransformer(
        [("numeric", StandardScaler(), numeric), ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical)]
    )
    probe = Pipeline(
        [("features", processor), ("classifier", LogisticRegression(max_iter=2000, random_state=SEED, class_weight="balanced"))]
    )
    probe.fit(train[features], train["class_name"])
    predictions = probe.predict(validation[features])
    report = classification_report(validation["class_name"], predictions, output_dict=True, zero_division=0)
    result = {
        "purpose": "Quantify class-predictive acquisition/source signals without using image anatomy.",
        "features": features,
        "validation_accuracy": accuracy_score(validation["class_name"], predictions),
        "validation_macro_f1": f1_score(validation["class_name"], predictions, average="macro"),
        "classification_report": report,
        "interpretation": "Performance above the majority-class baseline indicates potential shortcut signal; it is not a medical imaging classifier.",
    }
    (REPORTS_DIR / "shortcut_probe.json").write_text(json.dumps(result, indent=2) + "\n")

    stats = included.groupby(["class_name", "final_split"])[numeric].agg(["mean", "std", "median"])
    stats.to_csv(REPORTS_DIR / "shortcut_feature_statistics.csv")
    for column in categorical:
        pd.crosstab(included["class_name"], included[column]).to_csv(REPORTS_DIR / f"shortcut_{column}_crosstab.csv")

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    sns.boxplot(data=included, x="class_name", y="aspect_ratio", ax=axes[0], showfliers=False)
    sns.boxplot(data=included, x="class_name", y="border_to_center_ratio", ax=axes[1], showfliers=False)
    sns.boxplot(data=included, x="class_name", y="file_size_bytes", ax=axes[2], showfliers=False)
    for axis in axes:
        axis.tick_params(axis="x", rotation=25)
    axes[0].set_title("Aspect ratio by class")
    axes[1].set_title("Border/center intensity ratio")
    axes[2].set_title("File size by class")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "shortcut_signal_distributions.png", dpi=180)
    plt.close(fig)
    save_artifact_review(included)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
