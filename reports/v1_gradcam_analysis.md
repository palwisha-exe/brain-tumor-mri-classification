# Historical Version 1 Grad-CAM Review

These artifacts belong to the historical 160×160 EfficientNet-B0 Version 1 checkpoint, not the final deployed V2 model. Sixteen test cases were generated: a high-confidence and a lower-confidence correct prediction for each class, four high-confidence errors, and four lower-confidence errors. Exact case paths and probabilities are in `figures/v1_gradcam_case_manifest.csv`.

## Findings

- Correct glioma and meningioma examples sometimes had broad central or basal intracranial activation that plausibly overlapped abnormal-looking tissue, although the maps were coarse.
- A high-confidence correct no-tumor example emphasized skull and peripheral regions more than central anatomy.
- A high-confidence correct pituitary example showed broad superior-brain and extracranial emphasis rather than a discrete sellar focus.
- In a high-confidence glioma→no-tumor error, the conspicuous abnormal region remained relatively cool while other/background areas received stronger activation.
- Other high-confidence glioma/meningioma errors produced broad, non-specific central or superior maps that could not justify the predicted class.
- Lower-confidence examples likewise did not establish a consistent class-specific anatomical pattern.

## Interpretation

The maps provide qualitative evidence that the classifier uses a mixture of potentially relevant intracranial features and dataset-specific peripheral or presentation cues. This aligns with the independent shortcut probe, which achieved 70.84% validation accuracy from non-spatial acquisition/source features. It is therefore unsafe to interpret high confidence as reliable anatomical reasoning.

Grad-CAM is not validated tumor segmentation. No lesion masks, patient metadata, or expert localization ratings were available, so these maps cannot establish whether the model localized pathology correctly. They should be used for hypothesis generation and failure review only.

The Streamlit application now generates Grad-CAM from final V2 at inference time. These historical V1 observations must not be relabeled as V2 findings.
