# Brain Tumor Classification from MRI Using Deep Learning and Explainable AI

## Abstract

This study develops and internally tests four-class brain MRI image classifiers while treating dataset integrity as a first-order methodological concern. The supplied 7,200-image Training/Testing boundary contained 111 exact decoded-image duplicate groups crossing it, 203 explicitly pre-generated augmented files, internal redundancy, heterogeneous image properties, and no patient/study identifiers. A deterministic image-level leakage-control process excluded 1,232 files and produced canonical manifests of 4,177 training, 895 validation, and 896 test images with no exact decoded-image hashes crossing splits. A scratch CNN, historical 160×160 EfficientNet-B0 Version 1, and controlled Version 2 candidates were evaluated. The final V2 checkpoint used 224×224 input with the last three backbone blocks unfrozen and was selected at epoch 3 using validation macro-F1 only. On the 896-image official held-out manifest it achieved 0.9375 accuracy, 0.9316 macro precision, 0.9425 macro recall, and 0.9359 macro-F1. Historical V1 achieved 0.8996 accuracy and 0.9025 macro-F1 on the same manifest; direct glioma–meningioma errors decreased from 60 to 23. Patient-level independence, dataset-version provenance, external generalization, clinical validity, and reliable tumor localization remain unverified.

## Introduction

Brain tumor classification from MRI is a useful test bed for transfer learning, robustness analysis, and explainability. It is also vulnerable to misleading performance when duplicates, augmented copies, correlated slices, or source artifacts cross evaluation boundaries. This project therefore treats split construction, retained evidence, and limitations as core methodological components.

The task is single-image classification into glioma, meningioma, no tumor, or pituitary. It is not segmentation, grading, prognosis, treatment selection, triage, or a clinical decision-support system.

## Research Motivation

The project asks whether a reproducible leakage-controlled workflow can support a transparent internal comparison and whether errors and post-hoc explanations reveal potential shortcut learning. It emphasizes immutable source files, saved manifests and exclusions, fixed seeds, validation-only checkpoint selection, retained predictions, and explicit caveats.

## Dataset

The local data tree contains 7,200 files under `data/Training` and `data/Testing`: 1,400 training and 400 testing images per class. All 7,200 files decoded successfully. Content inspection identified 7,196 JPEG and four PNG files despite every filename using a `.jpg` suffix. Original modes included 4,129 RGB, 3,067 grayscale, three RGBA, and one palette image. Widths ranged from 150 to 1,375 pixels and heights from 167 to 1,446 across 447 dimension pairs; 1,533 images were non-square.

No CSV, DICOM, patient ID, study ID, acquisition protocol, or grouping metadata was supplied. Twenty-two images contained limited EXIF tags, but the public manifest stores tag names rather than EXIF values and no patient identifier was identified.

The directory shape and filenames strongly match the Masoud Nickparvar *Brain Tumor MRI Dataset* on Kaggle, but retained local records do not prove the exact source URL, dataset version, download date, or license applicable when downloaded. These fields remain explicitly unverified in `docs/DATASET.md`; the source data are not redistributed.

## Dataset Integrity Audit

Raw-file SHA-256 found no byte-identical pair across the supplied boundary, while decoded-image comparison found 111 cross-boundary groups, 121 train/test pairs, and 229 involved files. Within original splits, raw hashing found 153 duplicate groups containing 340 files. Filenames marked 100 Training and 103 Testing meningioma images as pre-generated augmentations.

These empirical results describe the local files and remain separate from any statements on the current dataset-hosting page. The original dataset was never modified, renamed, moved, or overwritten. Decisions are recorded in `metadata/dataset_manifest.csv`, `metadata/exclusions.csv`, and `metadata/preparation_summary.json`.

## Leakage Prevention and Data Partitions

With seed `20260913`, preparation excluded all 203 explicitly augmented files. Duplicate components were constructed from exact decoded RGB equality and visually reviewed DCT perceptual-hash, correlation, and local-structure criteria. Every class-conflicting component was excluded; otherwise one deterministic representative was retained. This removed 263 redundant exact copies, 756 threshold-qualified perceptual copies, and 10 files from label-conflicting components.

The 5,968 representatives were stratified 70/15/15:

| Split | Glioma | Meningioma | No Tumor | Pituitary | Total |
|---|---:|---:|---:|---:|---:|
| Train | 1,243 | 1,034 | 671 | 1,229 | 4,177 |
| Validation | 266 | 222 | 144 | 263 | 895 |
| Test | 267 | 222 | 143 | 264 | 896 |

No exact decoded-image hash crosses these canonical splits. Perceptual screening cannot prove patient identity or exclude related slices. The partitions are therefore described only as **image-level leakage-controlled**. Patient-level independence could not be verified.

Before V2, the original Kaggle Testing folder was used for a separate diagnostic evaluation of V1. To avoid feeding that inspected source into later optimization or model selection, V2 used only rows whose `original_split` was Training: 3,221 training images and 690 validation images. The official 896-image test manifest remained unchanged.

## Preprocessing

Pillow decodes file content and converts all images to RGB. Validation and test transforms are deterministic.

- Scratch CNN: 128×128 input; channel mean and standard deviation of 0.5.
- Historical V1 EfficientNet-B0: 160×160; ImageNet normalization.
- Final V2 EfficientNet-B0: 224×224 using bilinear interpolation with antialiasing; ImageNet mean `(0.485, 0.456, 0.406)` and standard deviation `(0.229, 0.224, 0.225)`.

Normalization occurs exactly once. Conservative training-only augmentation uses a restricted random resized crop, horizontal flip, rotation up to 7°, and mild brightness/contrast jitter. No dynamic augmentation is applied to validation or test images.

## Methodology

Class order is glioma=0, meningioma=1, no tumor=2, and pituitary=3. Weighted cross-entropy addresses the moderate imbalance after deduplication. Random seeds were set for Python, NumPy, and PyTorch; deterministic backend behavior was requested where supported. Exact cross-platform bitwise reproducibility is not guaranteed.

The recorded environment was Python 3.9.6, PyTorch 2.8.0, torchvision 0.23.0, and CPU execution. Configuration, history, checkpoint hashes, predictions, and metrics are retained in `metadata/`, `models/`, and `reports/metrics/`.

## Baseline CNN

The scratch CNN contains four convolutional stages, global aggregation, dropout, and a four-class head. It has 98,196 trainable parameters and used 128×128 images. Its best original validation macro-F1 was 0.8585. On the official test manifest it achieved 0.8705 accuracy and 0.8719 macro-F1.

## Historical Version 1

V1 used ImageNet-pretrained EfficientNet-B0, a frozen-backbone stage, then controlled fine-tuning of the final two feature blocks at 160×160. Its original validation macro-F1 was 0.9148. On the official 896-image test manifest it achieved 0.8996 accuracy, 0.9006 macro precision, 0.9046 macro recall, and 0.9025 macro-F1. It made 90 errors, including 60 direct glioma↔meningioma errors.

V1 is preserved as a historical comparison and is not the deployed model.

## Version 2 Development and Selection

V2 experiments used the 3,221/690 Training-source-only optimization/validation subsets. Controlled experiments compared the existing 160×160 setup with 224×224 input and compared unfreezing two versus three final backbone blocks while holding the loss and conservative augmentation policy fixed.

The selected candidate was EfficientNet-B0 at 224×224 with the last three backbone blocks unfrozen, learning rate `5e-5`, batch size 16, weighted cross-entropy, and 3,160,864 trainable parameters. Epoch 3 achieved validation accuracy 0.9681 and macro-F1 0.9701, the highest validation macro-F1 among the declared candidates.

The process was interrupted during epoch 4. That epoch resumed from the epoch-3 model weights without optimizer, scheduler, or augmentation RNG state. Epoch 4 did not replace the checkpoint. The selected epoch-3 checkpoint was completed before the interruption and was unaffected.

## Test-Use Disclosure

V2 checkpoint selection used validation only and did not read the official test set. After selection was frozen, the chosen epoch-3 checkpoint was evaluated once on the fixed 896-image official test manifest.

However, V1 had already been evaluated on this same official manifest before V2 development. Knowledge of historical test behavior existed at the project level. The test cohort is therefore not described as completely untouched across the entire lifetime of the project, even though no V2 candidate was selected or tuned using its results. A genuinely external patient-grouped cohort is required for stronger confirmation.

## Final Held-Out Results

| Model | Accuracy | Macro precision | Macro recall | Macro-F1 | Glioma ↔ Meningioma errors |
|---|---:|---:|---:|---:|---:|
| Scratch CNN | 0.8705 | 0.8755 | 0.8821 | 0.8719 | — |
| EfficientNet-B0 V1, historical | 0.8996 | 0.9006 | 0.9046 | 0.9025 | 60 |
| **EfficientNet-B0 V2, final** | **0.9375** | **0.9316** | **0.9425** | **0.9359** | **23** |

Final V2 per-class results:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Glioma | 0.9796 | 0.8989 | 0.9375 | 267 |
| Meningioma | 0.8929 | 0.9009 | 0.8969 | 222 |
| No Tumor | 0.8875 | 0.9930 | 0.9373 | 143 |
| Pituitary | 0.9663 | 0.9773 | 0.9718 | 264 |

V2 correctly classified 840 of 896 images and made 56 errors. The confusion matrix, in class order glioma, meningioma, no tumor, pituitary, is:

```text
[[240, 19,  6,  2],
 [  4,200, 11,  7],
 [  0,  1,142,  0],
 [  1,  4,  1,258]]
```

These are point estimates from one internal image-level split; confidence intervals were not calculated.

## Error Analysis

The 23 direct glioma↔meningioma errors comprise 19 glioma→meningioma and four meningioma→glioma predictions. Other errors include glioma/no-tumor and meningioma/no-tumor confusions. Meningioma has the lowest V2 class F1, 0.8969.

Mean maximum confidence was 0.9582 for correct predictions and 0.7399 for incorrect predictions; the highest-confidence incorrect prediction was 0.9967. Confidence must not be interpreted as clinical certainty.

## Grad-CAM

The Streamlit application generates Grad-CAM from the final V2 model using its 224×224 input and resizes the resulting heatmap to the uploaded image's original dimensions for display. Normal and Grad-CAM inference probabilities are checked for numerical agreement.

The 16 saved offline Grad-CAM cases under the V1 namespace were created with the historical model. Their review found mixed attention: some broad intracranial emphasis and several examples emphasizing skull, image borders, extracranial regions, or non-specific anatomy. These observations align with the shortcut-signal probe but cannot be transferred automatically to every V2 prediction.

Grad-CAM is coarse post-hoc attention. It is not validated tumor segmentation, anatomical localization, causal explanation, or proof that a prediction is medically grounded.

## Shortcut Analysis

A multinomial logistic-regression probe using image properties and source-related features—but no spatial anatomy—achieved 0.7084 validation accuracy and 0.7158 macro-F1, compared with a 0.2972 majority-class baseline. Dimensions, formats, modes, original source split, intensities, and border statistics are therefore strongly class-correlated. Model performance may partially reflect these shortcuts.

## Limitations

1. Patient and study identifiers were unavailable; patient-level independence cannot be verified.
2. Exact local dataset provenance, version, download date, license, acquisition sites, demographics, and diagnostic reference standards are unverified.
3. The test set is internally held out at image level, not external or prospective.
4. V1 test results were known before V2 development, although V2 selection itself used validation only.
5. Perceptual duplicate thresholds are heuristics and cannot identify all related slices.
6. Images are heterogeneous 2D exports without sequence or acquisition metadata.
7. Shortcut signals are strong and Grad-CAM behavior is not clinically validated.
8. Results use one fixed split and one selected training trajectory; no confidence intervals or multi-seed uncertainty estimates were produced.
9. The four output categories do not cover the clinical differential diagnosis.
10. No calibration, subgroup, external-site, prospective, human-comparator, or workflow evaluation was performed.

## Ethical and Clinical Considerations

The application is a research and educational demonstration only. It must not be used to diagnose, exclude, or treat disease. Uploaded-image confidence is not clinical certainty. Public distribution excludes the source MRI dataset and MRI-derived figures until provenance and redistribution rights are confirmed.

## Future Work

Future work requires patient/study-grouped development data, verified reference standards, acquisition metadata, an external site, uncertainty and calibration analysis, subgroup robustness checks, expert review, and explanation validation against appropriate annotations. Those steps require new data and governance rather than further optimization on the present test set.

## Conclusion

The final V2 model improves internal image-level test accuracy from 89.96% to 93.75%, macro-F1 from 90.25% to 93.59%, and direct glioma–meningioma errors from 60 to 23 relative to historical V1 on the same 896-image manifest. These results are reproducible from saved predictions and checkpoint metadata, but remain constrained by unverified patient independence, provenance, shortcut sensitivity, prior project-level knowledge of the test cohort, and absence of external clinical testing.

## References

1. Tan M, Le QV. [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://proceedings.mlr.press/v97/tan19a.html). Proceedings of Machine Learning Research. 2019;97:6105–6114.
2. Selvaraju RR, et al. [Grad-CAM: Visual Explanations From Deep Networks via Gradient-Based Localization](https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html). ICCV. 2017:618–626.
3. Mongan J, Moy L, Kahn CE Jr. [Checklist for Artificial Intelligence in Medical Imaging (CLAIM): A Guide for Authors and Reviewers](https://pubs.rsna.org/doi/10.1148/ryai.2020200029). Radiology: Artificial Intelligence. 2020;2(2):e200029.
4. Tejani AS, et al. [Checklist for Artificial Intelligence in Medical Imaging (CLAIM): 2024 Update](https://pubs.rsna.org/doi/10.1148/ryai.240300). Radiology: Artificial Intelligence. 2024;6(4):e240300.
5. PyTorch. [Reproducibility documentation](https://docs.pytorch.org/docs/stable/notes/randomness.html). Accessed 2026-09-13.
