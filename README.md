# Brain Tumor Classification from MRI Using Deep Learning and Explainable AI

A reproducible four-class MRI image-classification research project with dataset-integrity auditing, leakage-controlled splitting, a scratch CNN baseline, two EfficientNetB0 development versions, held-out evaluation, error analysis, Grad-CAM, and a Streamlit demonstration.

> **Research and educational use only.** This software is not a medical device and must not be used for diagnosis, treatment, triage, or patient-care decisions. Patient and study identifiers were unavailable, so patient-level independence could not be verified.

## Final model

The deployed model is the validation-selected **Version 2 EfficientNet-B0** checkpoint. It uses 224×224 RGB input, ImageNet normalization, and was trained with the last three backbone blocks unfrozen. Epoch 3 was selected using validation macro-F1 only.

| Model | Role | Input | Validation macro-F1 | Test accuracy | Test macro-F1 | Glioma ↔ Meningioma errors |
|---|---|---:|---:|---:|---:|---:|
| Scratch CNN | Baseline | 128×128 | 0.8585 | 0.8705 | 0.8719 | — |
| EfficientNet-B0 V1 | Historical comparison | 160×160 | 0.9148¹ | 0.8996 | 0.9025 | 60 |
| **EfficientNet-B0 V2** | **Final deployed model** | **224×224** | **0.9701²** | **0.9375** | **0.9359** | **23** |

¹ V1's original 895-image validation result. ² V2's 690-image Training-source-only validation subset. Validation values are not a same-cohort comparison in this table; the controlled comparison on the same 690 images is in `reports/metrics/v2/version2_validation_comparison.md`.

Final V2 performance on the 896-image leakage-controlled held-out image-level test split:

| Metric | Value |
|---|---:|
| Accuracy | 93.75% |
| Macro precision | 93.16% |
| Macro recall | 94.25% |
| Macro F1 | 93.59% |

V1 had previously been evaluated on this same official test manifest before V2 development. V2 checkpoint selection used validation only and did not read the official test set, but the test cohort should not be described as completely untouched over the full lifetime of the project.

## Dataset integrity and splits

The local source dataset contains 7,200 images across glioma, meningioma, no-tumor, and pituitary folders. Its supplied Training/Testing boundary was rejected after decoded-pixel hashing found 111 exact duplicate groups crossing that boundary and 203 explicitly pre-generated augmented filenames, including files in Testing.

The deterministic preparation pipeline used seed `20260913`, excluded 1,232 augmented, redundant, perceptually duplicated, or label-conflicting files, and retained 5,968 representative images:

| Split | Glioma | Meningioma | No Tumor | Pituitary | Total |
|---|---:|---:|---:|---:|---:|
| Train | 1,243 | 1,034 | 671 | 1,229 | 4,177 |
| Validation | 266 | 222 | 144 | 263 | 895 |
| Test | 267 | 222 | 143 | 264 | 896 |

No exact decoded-image hash crosses these canonical splits. Because the original Kaggle Testing folder had already been used diagnostically, V2 training and selection further restricted the train and validation manifests to rows originating in the source Training folder: 3,221 training images and 690 validation images. The official test split remained fixed at 896 images.

See `docs/DATASET.md` for provenance, attribution status, redistribution restrictions, and the distinction between source-page claims and this project's empirical audit.

## Methods

- Pillow decodes content and converts every input to RGB.
- The final V2 transform resizes to 224×224 using bilinear interpolation with antialiasing, converts to a tensor, and applies ImageNet mean `(0.485, 0.456, 0.406)` and standard deviation `(0.229, 0.224, 0.225)` exactly once.
- Conservative dynamic augmentation is applied only during training.
- Weighted cross-entropy addresses post-deduplication class imbalance.
- Validation macro-F1 was the primary selection criterion. The official test set was not used to select among V2 checkpoints.
- The final three-block run was interrupted during epoch 4. Epoch 4 resumed without optimizer, scheduler, or augmentation RNG state. This did not affect the selected epoch-3 checkpoint, which had completed before the interruption.

## Repository layout

```text
app/             Streamlit research demonstration
data/            local source dataset; never distributed or tracked
docs/            research report, dataset statement, model card, portfolio material
metadata/        audit, exclusion, class-mapping, and split records
models/          selection metadata; checkpoints remain local except release asset
notebooks/       executed Phase 1 exploration notebook
reports/         metrics, predictions, diagnostics, and non-image figures
scripts/         release verification, model retrieval, and opt-in training records
src/             preparation, training, evaluation, Grad-CAM, and QC code
tests/           manifest, preprocessing, checkpoint, inference, and Grad-CAM checks
```

## Setup and demo

Python 3.9.6 and the pinned dependencies were used for the recorded experiments.

```bash
git clone <repository-url>
cd brain-tumor-mri-ai
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The final checkpoint is distributed as a separate [GitHub Release asset](https://github.com/palwisha-exe/brain-tumor-mri-classification/releases/tag/v2.0.0). Retrieve and verify it with:

```bash
python scripts/fetch_final_model.py
streamlit run app/app.py
```

The helper rejects any file whose SHA-256 is not:

```text
26e4b268c112533bf22b6e726625044b0d0ad774fa7d00bea2599e044c36e1d0
```

No checkpoint has been uploaded by this preparation step. See `models/README.md`.

## Verify the frozen release

Normal verification never trains or selects a model:

```bash
source .venv/bin/activate
bash scripts/reproduce.sh
```

Historical training is separate, expensive, and explicitly opt-in; see `scripts/reproduce_training.sh`.

## Explainability and limitations

The deployed app computes Grad-CAM from the final V2 model and resizes the heatmap to the original uploaded image for display. Prediction probabilities are checked against normal inference. Grad-CAM is qualitative post-hoc attention—not tumor segmentation, validated localization, or evidence of clinical reasoning.

Historical saved Grad-CAM images were generated with V1 and are labeled accordingly. They showed mixed behavior, including attention to borders, skull, extracranial regions, or broad non-specific anatomy. A non-anatomical shortcut probe achieved 70.84% validation accuracy, indicating substantial dataset-specific shortcut risk.

Additional limitations include unavailable patient/study identifiers, unverified patient-level independence, no external or prospective testing, uncertain source-label reference standards, heterogeneous image properties, heuristic perceptual-duplicate rules, one fixed split, no confidence intervals, and no clinical calibration or expert explanation validation.

## Rights and citation

Original source code authored for this repository is available under the MIT License; see `LICENSE`. That license does not relicense the source MRI dataset, third-party datasets or components, pretrained weights, or separately governed MRI-derived material. The source MRI dataset and MRI-derived visual assets are not distributed. The precise local dataset version and applicable license could not be proven from retained local records; see `docs/DATASET.md`, `THIRD_PARTY_NOTICES.md`, and `CITATION.cff`.

## References

- Tan, M., & Le, Q. V. (2019). [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://proceedings.mlr.press/v97/tan19a.html). ICML/PMLR.
- Selvaraju, R. R., et al. (2017). [Grad-CAM: Visual Explanations From Deep Networks via Gradient-Based Localization](https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html). ICCV.
- Tejani, A. S., et al. (2024). [Checklist for Artificial Intelligence in Medical Imaging (CLAIM): 2024 Update](https://pubs.rsna.org/doi/10.1148/ryai.240300). *Radiology: Artificial Intelligence*.
- PyTorch. [Reproducibility documentation](https://docs.pytorch.org/docs/stable/notes/randomness.html).
