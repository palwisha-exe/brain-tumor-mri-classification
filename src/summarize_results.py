from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import FIGURES_DIR, METRICS_DIR, REPORTS_DIR, ensure_directories


def main() -> None:
    ensure_directories()
    run_names = ("baseline_cnn", "efficientnet_b0_finetuned")
    rows = []
    for run_name in run_names:
        metrics = json.loads((METRICS_DIR / f"{run_name}_test_metrics.json").read_text())
        rows.append(
            {
                "model": run_name,
                "accuracy": metrics["accuracy"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "roc_auc_ovr_macro": metrics["roc_auc_ovr_macro"],
            }
        )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(REPORTS_DIR / "v1_model_comparison.csv", index=False)

    final = pd.read_csv(METRICS_DIR / "efficientnet_b0_finetuned_test_predictions.csv")
    errors = final[~final["correct"]].copy()
    confusions = (
        errors.groupby(["true_class", "predicted_class"]).size().sort_values(ascending=False).rename("count").reset_index()
    )
    confusions.to_csv(REPORTS_DIR / "v1_final_model_error_confusions.csv", index=False)
    error_summary = {
        "test_images": int(len(final)),
        "correct": int(final["correct"].sum()),
        "incorrect": int((~final["correct"]).sum()),
        "error_rate": float((~final["correct"]).mean()),
        "mean_confidence_correct": float(final.loc[final["correct"], "confidence"].mean()),
        "mean_confidence_incorrect": float(errors["confidence"].mean()),
        "incorrect_at_confidence_ge_0_90": int((errors["confidence"] >= 0.90).sum()),
        "incorrect_at_confidence_ge_0_80": int((errors["confidence"] >= 0.80).sum()),
        "dominant_confusions": confusions.head(8).to_dict(orient="records"),
    }
    (REPORTS_DIR / "v1_final_model_error_summary.json").write_text(json.dumps(error_summary, indent=2) + "\n")

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(data=final, x="confidence", hue="correct", bins=20, multiple="layer", ax=ax)
    ax.set_title("Selected model confidence: correct vs incorrect test predictions")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "v1_final_model_confidence_errors.png", dpi=180)
    plt.close(fig)
    print(comparison.to_string(index=False))
    print(json.dumps(error_summary, indent=2))


if __name__ == "__main__":
    main()
