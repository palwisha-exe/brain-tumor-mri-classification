# CV Project Entry

## Concise version

**Brain Tumor MRI Classification with Leakage Control and Explainable AI** — Built a reproducible PyTorch pipeline for four-class MRI classification; audited 7,200 images, detected exact cross-split leakage and pre-generated test augmentations, and constructed a documented 5,968-image leakage-controlled split. Compared a scratch CNN with two EfficientNet-B0 development versions. The validation-selected final 224×224 model achieved 93.75% accuracy and 93.59% macro-F1 on an 896-image held-out image-level test split, versus 89.96% and 90.25% for historical V1. Added error analysis, shortcut-signal analysis, Grad-CAM, integrity tests, and a Streamlit research demo, while explicitly documenting the absence of patient IDs and clinical validation.

## Extended portfolio version

- Identified 111 exact decoded-image duplicate groups crossing the supplied Training/Testing boundary and excluded 1,232 augmented, redundant, perceptually duplicated, or label-conflicting files without changing source data.
- Created seeded canonical manifests of 4,177/895/896 images with zero exact decoded-image hashes crossing splits; correctly described the result as image-level leakage-controlled, not patient-level independent.
- Used 3,221 Training-source-only training images and 690 Training-source-only validation images for V2 model development.
- Selected the final EfficientNet-B0 checkpoint at epoch 3 by validation macro-F1, using 224×224 RGB input, ImageNet normalization, weighted cross-entropy, conservative training-only augmentation, and the last three backbone blocks unfrozen.
- Improved held-out accuracy from 89.96% for historical V1 to 93.75% for V2 and macro-F1 from 90.25% to 93.59%; glioma↔meningioma errors decreased from 60 to 23 on the same 896-image test manifest.
- Preserved full predictions, checkpoint hashes, per-class results, confusion matrices, and a transparent disclosure that V1 had already been evaluated on the official test manifest before V2 development.
- Built a Streamlit upload demonstration with original-resolution display, four-class probabilities, Grad-CAM, and prominent non-clinical-use warnings.

## Interview-safe framing

Say “internal performance on a leakage-controlled held-out image-level split,” not “diagnostic accuracy.” V2 selection itself used validation only, but the official test cohort was not untouched over the entire project lifetime because V1 had previously been evaluated on it. Patient-level independence, external generalization, label provenance, and clinical validity remain unverified.
