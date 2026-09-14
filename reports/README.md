# Report Artifact Index

## Final Version 2

- reports/metrics/v2/final_v2_test_metrics.json
- reports/metrics/v2/final_v2_test_predictions.csv
- reports/metrics/v2/final_v2_classification_report.csv
- reports/metrics/v2/final_v2_top20_high_confidence_errors.csv
- reports/metrics/v2/final_v1_vs_v2_heldout_comparison.json
- reports/figures/v2/final_v2_test_confusion_matrix.png
- reports/v2_quality_control.json

These are the final deployed-model evaluation and release-verification artifacts.

## Historical Version 1

The following files are retained as historical V1 evidence and are not final V2 results:

- reports/v1_model_comparison.csv
- reports/v1_final_model_error_summary.json
- reports/v1_final_model_error_confusions.csv
- reports/figures/v1_final_model_confidence_errors.png
- reports/v1_gradcam_analysis.md
- reports/figures/v1_gradcam_case_manifest.csv
- reports/v1_original_testing_predictions.csv
- reports/v1_original_testing_misclassifications.csv
- reports/figures/v1_original_testing_confusion_matrix.png
- reports/v1_quality_control.json
- reports/diagnostics/notumor_original_testing/

The original Kaggle Testing-folder outputs were diagnostic evaluations of V1. They are not the project's official held-out test result and must not replace final V2 metrics.

## Image redistribution

MRI sample montages, error contact sheets, and saved Grad-CAM overlays remain local and are excluded from the proposed public Git staging set because exact source-image redistribution rights are unverified. Aggregate statistical figures and confusion matrices remain eligible for release.
