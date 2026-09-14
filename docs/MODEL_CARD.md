# Model Card: Final EfficientNet-B0 V2

## Summary

The final model is a four-class PyTorch image classifier for research and educational demonstration. It predicts one of: glioma, meningioma, no tumor, or pituitary.

It is not a medical device and is not intended for diagnosis, screening, treatment, triage, prognosis, or patient-care decisions.

## Architecture and selection

- Architecture: EfficientNet-B0
- Initialization lineage: ImageNet-pretrained EfficientNet features
- Input: 224×224 RGB
- Fine-tuning: final three backbone blocks plus classifier trainable
- Trainable parameters during final run: 3,160,864
- Loss: class-weighted cross-entropy
- Training augmentation: conservative dynamic augmentation, training only
- Selected checkpoint: epoch 3
- Primary selection criterion: validation macro-F1
- Selection data: 690 Training-source-only validation images
- Training data: 3,221 Training-source-only images
- Checkpoint SHA-256: `26e4b268c112533bf22b6e726625044b0d0ad774fa7d00bea2599e044c36e1d0`

The interrupted epoch 4 resumed without optimizer, scheduler, or augmentation RNG state. It did not replace the checkpoint; the selected epoch-3 checkpoint was completed before the interruption and was unaffected.

## Input preprocessing

1. Decode image content with Pillow.
2. Convert consistently to RGB.
3. Resize to 224×224 with bilinear interpolation and antialiasing.
4. Convert pixels to a PyTorch tensor.
5. Normalize exactly once with ImageNet mean `(0.485, 0.456, 0.406)` and standard deviation `(0.229, 0.224, 0.225)`.

Output index mapping is glioma=0, meningioma=1, no tumor=2, pituitary=3.

## Held-out image-level performance

The frozen selected checkpoint was evaluated on the project's fixed 896-image leakage-controlled test manifest.

| Metric | Value |
|---|---:|
| Accuracy | 93.75% |
| Macro precision | 93.16% |
| Macro recall | 94.25% |
| Macro F1 | 93.59% |

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Glioma | 97.96% | 89.89% | 93.75% | 267 |
| Meningioma | 89.29% | 90.09% | 89.69% | 222 |
| No Tumor | 88.75% | 99.30% | 93.73% | 143 |
| Pituitary | 96.63% | 97.73% | 97.18% | 264 |

Confusion matrix, with rows=true and columns=predicted in the class order above:

```text
[[240, 19,  6,  2],
 [  4,200, 11,  7],
 [  0,  1,142,  0],
 [  1,  4,  1,258]]
```

There were 56 errors, including 19 glioma→meningioma and four meningioma→glioma errors. Mean maximum confidence was 95.82% for correct predictions and 73.99% for incorrect predictions.

Historical V1 achieved 89.96% accuracy and 90.25% macro-F1 with 60 direct glioma↔meningioma errors on the same manifest. V2 achieved 93.75%, 93.59%, and 23 respectively.

## Evaluation caveat

V2 checkpoint selection used validation only; no V2 candidate was selected using official test performance. V1 had already been evaluated on the same official test manifest before V2 development, however. The cohort is not completely untouched across the project's full lifetime. The results are internal image-level estimates, not external clinical validation.

## Intended use

- reproducibility demonstrations;
- education about leakage control and model evaluation;
- portfolio review;
- research hypothesis generation and failure analysis.

## Out-of-scope use

- any clinical decision;
- confirming or excluding a tumor;
- patient triage or treatment recommendations;
- use as segmentation, localization, grading, or prognosis;
- deployment on unseen institutions or populations without new validation;
- interpreting softmax confidence as clinical certainty.

## Known limitations and risks

- Patient/study identifiers were unavailable; patient-level independence could not be verified.
- Dataset source version, acquisition sites, demographics, diagnostic reference standards, and exact license are not proven from local records.
- No external, prospective, multi-site, subgroup, calibration, human-comparator, or clinical-workflow evaluation was performed.
- Source files contain heterogeneous sizes, modes, formats, borders, and presentation artifacts.
- A non-anatomical feature probe achieved 70.84% validation accuracy, indicating strong shortcut sensitivity.
- Perceptual duplicate rules are heuristics and cannot establish patient identity.
- Meningioma has the lowest final per-class F1, and glioma/meningioma confusion remains present.
- Incorrect predictions can have high confidence; the maximum confidence among final test errors was 99.67%.
- Four output classes do not represent the full clinical differential diagnosis.

## Explainability

The application computes Grad-CAM using the final model input and resizes the heatmap to the original uploaded image for display. Normal and Grad-CAM inference probabilities must agree numerically.

Grad-CAM is a qualitative, coarse, architecture-dependent post-hoc attention map. It is not validated tumor segmentation, anatomical localization, causal explanation, or evidence that the model uses medically appropriate features. Historical V1 review found examples of border, skull, extracranial, and broad non-specific attention; V1 findings are retained as historical evidence and are not relabeled as V2 results.

## Artifacts

- Selection record: `models/selected_model_v2.json`
- Test metrics: `reports/metrics/v2/final_v2_test_metrics.json`
- Predictions: `reports/metrics/v2/final_v2_test_predictions.csv`
- Classification report: `reports/metrics/v2/final_v2_classification_report.csv`
- Confusion matrix: `reports/figures/v2/final_v2_test_confusion_matrix.png`
- V1/V2 comparison: `reports/metrics/v2/final_v1_vs_v2_heldout_comparison.json`

The checkpoint is planned as a separately hosted release asset and must pass the recorded SHA-256 check before use.
