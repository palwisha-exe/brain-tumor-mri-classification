# Project Summary

## Outcome

This project built a reproducible four-class brain MRI research classifier with dataset-integrity auditing, leakage-controlled split construction, a scratch CNN baseline, two EfficientNet-B0 development versions, error analysis, Grad-CAM, automated verification, and a Streamlit demonstration.

The final deployed model is EfficientNet-B0 Version 2: 224×224 input, ImageNet normalization, last three backbone blocks unfrozen, and epoch 3 selected using validation macro-F1 only.

## Data and leakage control

The 7,200-image supplied Training/Testing boundary was rejected because 111 exact decoded-image duplicate groups crossed it and augmented filenames occurred in Testing. The preparation pipeline excluded 1,232 files and retained canonical manifests containing 4,177 training, 895 validation, and 896 test images with no exact decoded duplicate crossing a split.

V2 training and model selection used only source-Training rows within the canonical manifests: 3,221 training and 690 validation images. The final official test set remained 896 images. The split is image-level leakage-controlled; patient-level independence could not be verified because patient/study identifiers were unavailable.

## Results

| Model | Status | Test accuracy | Macro precision | Macro recall | Macro F1 | Glioma ↔ Meningioma errors |
|---|---|---:|---:|---:|---:|---:|
| Scratch CNN | Baseline | 87.05% | 87.55% | 88.21% | 87.19% | — |
| EfficientNet-B0 V1 | Historical | 89.96% | 90.06% | 90.46% | 90.25% | 60 |
| **EfficientNet-B0 V2** | **Final** | **93.75%** | **93.16%** | **94.25%** | **93.59%** | **23** |

All test results use the same 896-image official leakage-controlled manifest. V1 had already been evaluated on this manifest before V2 development. V2 checkpoint selection did not use test data, but the cohort is not described as completely untouched across the project's entire lifetime.

## Explainability and limitations

The Streamlit application uses the final V2 checkpoint and exact 224×224 preprocessing. It preserves the uploaded image at original resolution, displays all four probabilities separately from overall model performance, and renders a Grad-CAM overlay at the original display dimensions.

Saved historical Grad-CAM cases belong to V1 and showed mixed attention, including skull, border, extracranial, and broad non-specific regions. Grad-CAM is not segmentation or validated localization. A non-anatomical shortcut probe reached 70.84% validation accuracy, reinforcing the risk of dataset-specific shortcuts.

The system has no patient-level, external, prospective, clinical-workflow, calibration, or regulatory validation and must not be used for clinical decisions.

## Run locally

```bash
cd brain-tumor-mri-ai
source .venv/bin/activate
python scripts/fetch_final_model.py --url <final-checkpoint-release-url>
streamlit run app/app.py
```

The checkpoint release URL remains unavailable until a repository release is approved and created.
