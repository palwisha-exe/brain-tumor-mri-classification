# Historical Version 1 diagnostic: original Testing/notumor predictions

> This diagnostic used the historical 160×160 Version 1 EfficientNet-B0 checkpoint. It is not a final Version 2 evaluation and must not replace the project's official V2 held-out metrics.

## Scope and controls

This investigation used the existing selected EfficientNet-B0 checkpoint and the existing manifests and saved held-out predictions. No model was retrained or changed, and no dataset file or split was modified. The deterministic 20-image sample was `Te-no_1.jpg` through `Te-no_20.jpg` in natural numeric order.

## Root-cause conclusion

The reported behavior is **not caused by a class-index mapping error or a Streamlit preprocessing mismatch**. The saved evaluation pipeline and a byte-upload implementation of the Streamlit pipeline produced the same four probabilities for every tested image, with a maximum absolute difference of `6.780028343200684e-07` (all values matched with `atol=1e-6, rtol=1e-6`). The immediate cause is genuine behavior of the saved model on a small subset of the source images.

The original Kaggle-style `data/Testing` folder is not the project's held-out test set. The leakage-controlled split was rebuilt across the original folders, so images from `data/Testing/notumor` can be in final training, validation, test, or exclusions. Folder-wide, the selected model predicted No Tumor for 389 of 400 images (97.25%); the 11 errors were 6 Glioma and 5 Meningioma. A manually chosen small group can therefore appear substantially worse than the complete folder.

The underlying scientific cause of the individual errors cannot be established from this dataset. The evidence is consistent with sensitivity to image appearance and source/shortcut features, and possibly ambiguous source labels. Patient/study identifiers, acquisition metadata, and clinical ground truth are unavailable, so these possibilities cannot be resolved here.

## 1. Class-index mapping

The class order is consistent in all checked components:

| Output index | Internal class | Display label |
|---:|---|---|
| 0 | `glioma` | Glioma |
| 1 | `meningioma` | Meningioma |
| 2 | `notumor` | No Tumor |
| 3 | `pituitary` | Pituitary |

Verified sources:

- `metadata/class_mapping.json`
- `src/config.py` (`CLASS_NAMES` and `CLASS_TO_IDX`)
- `metadata/splits/train.csv`, `validation.csv`, and `test.csv`
- `models/efficientnet_b0_finetuned_best.pt` (`class_names`)
- `src/evaluate.py`
- `app/app.py`

All manifest rows also have `class_index` consistent with their class name.

## 2. Preprocessing comparison

Both paths perform the same operations:

- Pillow content decoding
- conversion to RGB
- bilinear resize with antialiasing to 160 × 160
- RGB channel order, converted to a CHW tensor
- `ToTensor`, producing values in `[0, 1]`
- one ImageNet normalization with mean `(0.485, 0.456, 0.406)` and standard deviation `(0.229, 0.224, 0.225)`
- final input shape `[1, 3, 160, 160]`

For the first diagnostic sample, the pre-normalization range was `[0.0, 1.0]` and the post-normalization range was `[-2.1179039478302, 2.517995834350586]`. No channel-order mismatch or double normalization was found.

Pipeline A used `ManifestDataset(..., training=False)` exactly as held-out evaluation does. Pipeline B read each file into bytes, decoded it through `BytesIO`, converted it to RGB, and called the same `build_transform(...)` and Grad-CAM probability path as Streamlit.

## 3. Deterministic 20-image comparison

Predictions for `Te-no_1.jpg` through `Te-no_20.jpg`:

| Predicted class | Count |
|---|---:|
| Glioma | 2 |
| Meningioma | 0 |
| No Tumor | 18 |
| Pituitary | 0 |

The evaluation-pipeline probabilities are below; the last column is the largest absolute difference from the corresponding Streamlit-pipeline probability for that image:

| Image | Glioma | Meningioma | No Tumor | Pituitary | Prediction | Final status | Max difference |
|---|---:|---:|---:|---:|---|---|---:|
| `Te-no_1.jpg` | 0.000065 | 0.000093 | 0.999697 | 0.000145 | No Tumor | train | 6.98e-10 |
| `Te-no_2.jpg` | 0.001256 | 0.000579 | 0.998109 | 0.000056 | No Tumor | test | 2.91e-10 |
| `Te-no_3.jpg` | 0.000128 | 0.000058 | 0.999663 | 0.000151 | No Tumor | train | 8.73e-10 |
| `Te-no_4.jpg` | 0.004679 | 0.001085 | 0.992396 | 0.001840 | No Tumor | excluded | 3.96e-08 |
| `Te-no_5.jpg` | 0.000133 | 0.000012 | 0.999854 | 0.000001 | No Tumor | excluded | 6.26e-10 |
| `Te-no_6.jpg` | 0.845695 | 0.042463 | 0.111606 | 0.000237 | Glioma | excluded | 2.76e-07 |
| `Te-no_7.jpg` | 0.000097 | 0.000176 | 0.999696 | 0.000031 | No Tumor | excluded | 1.29e-09 |
| `Te-no_8.jpg` | 0.832236 | 0.035731 | 0.131561 | 0.000472 | Glioma | train | 4.77e-07 |
| `Te-no_9.jpg` | 0.062272 | 0.042713 | 0.894720 | 0.000296 | No Tumor | train | 1.19e-07 |
| `Te-no_10.jpg` | 0.034675 | 0.001697 | 0.962974 | 0.000654 | No Tumor | train | 2.76e-07 |
| `Te-no_11.jpg` | 0.000600 | 0.000362 | 0.998517 | 0.000521 | No Tumor | test | 2.62e-09 |
| `Te-no_12.jpg` | 0.002974 | 0.003006 | 0.993238 | 0.000782 | No Tumor | train | 1.19e-07 |
| `Te-no_13.jpg` | 0.002695 | 0.000201 | 0.997048 | 0.000056 | No Tumor | train | 1.19e-07 |
| `Te-no_14.jpg` | 0.005222 | 0.002069 | 0.992529 | 0.000180 | No Tumor | train | 4.70e-08 |
| `Te-no_15.jpg` | 0.035985 | 0.000786 | 0.962430 | 0.000799 | No Tumor | train | 6.78e-07 |
| `Te-no_16.jpg` | 0.000124 | 0.000011 | 0.999864 | 0.000001 | No Tumor | train | 1.06e-09 |
| `Te-no_17.jpg` | 0.008737 | 0.008403 | 0.981554 | 0.001306 | No Tumor | validation | 1.19e-07 |
| `Te-no_18.jpg` | 0.020463 | 0.016760 | 0.958661 | 0.004116 | No Tumor | test | 1.19e-07 |
| `Te-no_19.jpg` | 0.028614 | 0.028734 | 0.941174 | 0.001477 | No Tumor | test | 2.98e-07 |
| `Te-no_20.jpg` | 0.000560 | 0.000082 | 0.999316 | 0.000041 | No Tumor | train | 7.22e-09 |

The two Glioma predictions were:

- `Te-no_6.jpg`: Glioma 84.57%; excluded as a redundant high-similarity perceptual duplicate of `Te-no_384.jpg`.
- `Te-no_8.jpg`: Glioma 83.22%; included in final training.

Membership of the 20 images was: 11 final training, 1 validation, 4 final test, and 4 excluded. The complete image-by-image probabilities from both pipelines and their absolute differences are saved in `twenty_image_pipeline_comparison.csv`.

## 4. Complete original Testing/notumor folder

For context, all 400 source-folder images were tested:

| Predicted class | Count | Percentage |
|---|---:|---:|
| Glioma | 6 | 1.50% |
| Meningioma | 5 | 1.25% |
| No Tumor | 389 | 97.25% |
| Pituitary | 0 | 0.00% |

Their final-manifest membership was:

| Final status | Count | Percentage |
|---|---:|---:|
| Training | 200 | 50.00% |
| Validation | 50 | 12.50% |
| Held-out test | 41 | 10.25% |
| Excluded | 109 | 27.25% |

All 109 exclusions in this source folder were documented as `redundant_high_similarity_perceptual_duplicate`.

The 11 errors and final-manifest membership were:

| Image | Prediction | Confidence | Final status | Note |
|---|---|---:|---|---|
| `Te-no_150.jpg` | Glioma | 90.82% | validation | retained representative |
| `Te-no_6.jpg` | Glioma | 84.57% | excluded | perceptual duplicate of `Te-no_384.jpg` |
| `Te-no_384.jpg` | Glioma | 84.06% | test | retained duplicate-group representative |
| `Te-no_8.jpg` | Glioma | 83.22% | train | retained representative |
| `Te-no_96.jpg` | Glioma | 73.09% | train | retained representative |
| `Te-no_265.jpg` | Glioma | 57.80% | train | retained representative |
| `Te-no_38.jpg` | Meningioma | 55.14% | validation | retained representative |
| `Te-no_29.jpg` | Meningioma | 50.69% | train | retained representative |
| `Te-no_352.jpg` | Meningioma | 53.35% | excluded | perceptual duplicate of `Te-no_29.jpg` |
| `Te-no_365.jpg` | Meningioma | 36.18% | train | retained representative |
| `Te-no_121.jpg` | Meningioma | 51.25% | test | retained representative; exact decoded duplicate excluded elsewhere |

`Te-no_96.jpg` and `Te-no_265.jpg` are visually similar and have a 256-bit perceptual-hash distance of 20, but were not assigned to the same duplicate component under the project's fixed conservative threshold. Both are in training, so this pair does not cross an evaluation boundary.

## 5. Final leakage-controlled held-out test

The saved held-out prediction artifact contains exactly the same 896 paths as `metadata/splits/test.csv`. With class order Glioma, Meningioma, No Tumor, Pituitary, its confusion matrix is:

| True / predicted | Glioma | Meningioma | No Tumor | Pituitary |
|---|---:|---:|---:|---:|
| Glioma | 229 | 32 | 4 | 2 |
| Meningioma | 28 | 181 | 5 | 8 |
| No Tumor | 2 | 2 | 139 | 0 |
| Pituitary | 3 | 4 | 0 | 257 |

For No Tumor on this held-out split:

- support: 143
- recall: `0.972027972027972` (97.20%)
- F1-score: `0.9553264604810997` (95.53%)
- error distribution: 2 predicted Glioma, 2 predicted Meningioma, 139 predicted No Tumor, 0 predicted Pituitary

These values were recalculated from `reports/metrics/efficientnet_b0_finetuned_test_predictions.csv`, not from the original `data/Testing` folder, and agree with `reports/metrics/efficientnet_b0_finetuned_test_metrics.json`.

## 6. Grad-CAM review of six No Tumor → Glioma errors

The six highest-confidence Glioma errors were reviewed in `misclassified_notumor_gradcam_contact_sheet.png`:

- `Te-no_6.jpg`, `Te-no_384.jpg`, and `Te-no_8.jpg`: activation is broad and centered over intracranial tissue, especially the central/deep-brain region. It is not sharply localized and therefore does not establish a medically meaningful focus.
- `Te-no_96.jpg` and `Te-no_265.jpg`: activation is concentrated on a conspicuous bright central intracranial feature. The model appears to be responding to that image feature, but the available label and missing clinical metadata do not establish what the feature represents.
- `Te-no_150.jpg`: strongest activation is within the lower central intracranial/posterior-fossa region, with some spread toward the inferior skull/background boundary.

Overall, these six maps are mostly brain-centered rather than exclusively background-centered, but they are coarse and include boundary sensitivity. They do not validate the classifications, localize a tumor, or rule out shortcut learning. Prior project findings—including strong metadata-only shortcut predictability—remain important evidence that source/acquisition signals can influence this model.

## 7. Action decision

No Streamlit inference fix is warranted: the two pipelines match numerically and use the same class order and transform. The app and checkpoint were left unchanged. Improving these predictions would require a new, explicitly approved modeling/data investigation, ideally with patient/study identifiers, acquisition metadata, independent external data, and clinically reviewed labels; it must not be achieved by changing the displayed mapping or post-processing probabilities.

## Reproducible artifacts

- `src/diagnose_notumor.py` — diagnostic runner
- `reports/diagnostics/notumor_original_testing/diagnostic_summary.json` — machine-readable summary
- `reports/diagnostics/notumor_original_testing/twenty_image_pipeline_comparison.csv` — per-image pipeline comparison
- `reports/diagnostics/notumor_original_testing/all_original_testing_notumor_predictions.csv` — all 400 predictions and membership
- `reports/diagnostics/notumor_original_testing/misclassified_notumor_gradcam_contact_sheet.png` — six reviewed error cases
