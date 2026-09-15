# Final Consolidated Project Report

1. **Built:** immutable-data audit, deterministic leakage-controlled manifests, reusable preprocessing, scratch CNN, historical EfficientNet-B0 V1, final EfficientNet-B0 V2, held-out evaluation, error analysis, shortcut probe, Grad-CAM, Streamlit demo, tests, and research documentation.
2. **Final dataset preparation:** 5,968 retained images in canonical 4,177/895/896 train/validation/test manifests using seed `20260913`; no exact decoded-image hash crosses a split.
3. **V2 development subset:** 3,221 training and 690 validation images whose source folder was Training. The original Kaggle Testing source was not fed into V2 optimization or selection.
4. **Excluded:** 1,232 images—203 pre-generated augmentations, 263 redundant exact decoded copies, 756 threshold-qualified perceptual copies, and 10 images from label-conflicting components.
5. **Baseline CNN:** 87.05% test accuracy and 87.19% macro-F1.
6. **Historical V1:** EfficientNet-B0 at 160×160; 89.96% test accuracy, 90.25% macro-F1, and 60 glioma↔meningioma errors.
7. **Final V2:** EfficientNet-B0 at 224×224 with the last three blocks unfrozen; epoch 3 selected by validation macro-F1. On 896 official test images it achieved 93.75% accuracy, 93.16% macro precision, 94.25% macro recall, and 93.59% macro-F1.
8. **Final per-class V2 F1:** glioma 93.75%, meningioma 89.69%, no tumor 93.73%, pituitary 97.18%.
9. **Major V2 errors:** 56/896 incorrect; 19 glioma→meningioma and four meningioma→glioma, for 23 pairwise errors.
10. **Selection disclosure:** V2 checkpoint selection used validation only. V1 had already been evaluated on the same official test manifest before V2 development, so the test cohort was not completely untouched across the entire project lifetime.
11. **Interruption disclosure:** epoch 4 resumed without optimizer, scheduler, or augmentation RNG state. The selected epoch-3 checkpoint was complete before the interruption and was unaffected.
12. **Grad-CAM:** live explanations use V2. The 16 saved historical cases are V1 artifacts and showed mixed intracranial and shortcut-sensitive attention. Grad-CAM is not tumor segmentation or validated localization.
13. **Streamlit:** the final app uses the frozen V2 checkpoint, 224×224 preprocessing, original-resolution display, probability/Grad-CAM parity checks, and prominent non-clinical disclaimers.
14. **Limitations:** no patient/study IDs, patient-level independence unverified, provenance/version not proven from local records, no external or prospective testing, heuristic perceptual matching, shortcut signals, one fixed split, no confidence intervals, and no clinical or explanation validation.
15. **Review first:** `README.md`, `docs/brain_tumor_mri_research_report.md`, `docs/MODEL_CARD.md`, `docs/DATASET.md`, and `reports/metrics/v2/final_v2_test_metrics.json`.
16. **Demo:** retrieve the verified `v2.0.0` release checkpoint with `python scripts/fetch_final_model.py`, then run `streamlit run app/app.py`.
17. **External work required:** confirm exact dataset provenance/license and obtain patient-grouped, clinically adjudicated external data for stronger evaluation.
