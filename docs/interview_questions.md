# Interview Questions and Evidence-Based Answers

## Why did you reject the provided train/test split?

Decoded-image hashing found 111 exact duplicate groups and 121 pairs crossing it. The Testing directory also contained 103 filenames explicitly marked as augmented. Raw file hashing missed the cross-boundary copies because their encodings differed.

## What does “image-level leakage-controlled” mean?

Exact and threshold-qualified perceptual duplicate groups were excluded or reduced to one representative before splitting, and no exact decoded-image hash crosses the canonical splits. This does not mean patient-independent: patient and study identifiers were unavailable, so related images or different slices from the same person may still cross splits.

## What data did Version 2 use?

The canonical manifests contain 4,177 training, 895 validation, and 896 test images. Because the source Kaggle Testing folder had already been inspected diagnostically, V2 optimization and selection used only canonical train/validation rows originating in the source Training folder: 3,221 training and 690 validation images. Its final evaluation used the unchanged 896-image official test manifest.

## Why EfficientNet-B0 and 224×224 input?

EfficientNet-B0 provides a computationally efficient transfer-learning baseline. V1 used 160×160 input and the last two blocks. Controlled validation experiments found the highest V2 validation macro-F1 at 224×224 with the last three blocks unfrozen. The change was accepted on validation evidence, not test results.

## How was Version 2 selected?

Epoch 3 of the 224×224 three-block run achieved the highest validation macro-F1, 0.9701, on the 690-image Training-source-only validation subset. Validation accuracy was 0.9681. The official test set was not read to select among V2 checkpoints.

## Was the official test set completely untouched?

Not across the entire lifetime of the project. V1 had already been evaluated on the same official 896-image manifest before V2 development. V2 selection itself used validation only, and V2 was evaluated once after freezing, but the strongest claim of a project-wide never-seen test cohort would be inaccurate.

## What were the final results?

On 896 official held-out images, V2 achieved 93.75% accuracy, 93.16% macro precision, 94.25% macro recall, and 93.59% macro-F1. Historical V1 achieved 89.96% accuracy and 90.25% macro-F1 on the same manifest.

## Did V2 reduce glioma–meningioma confusion?

Yes on the official test manifest. Direct glioma↔meningioma errors decreased from 60 for V1 to 23 for V2: 19 glioma→meningioma and four meningioma→glioma. This remains an important error mode and does not establish clinical reliability.

## What was the final per-class performance?

V2 F1 scores were 0.9375 for glioma, 0.8969 for meningioma, 0.9373 for no tumor, and 0.9718 for pituitary. Meningioma remained the lowest-F1 class.

## How did you avoid double normalization?

The final pipeline decodes with Pillow, converts to RGB, resizes to 224×224 with bilinear interpolation and antialiasing, converts to a tensor, then applies ImageNet mean and standard deviation once. The model does not contain a second input-normalization layer.

## What happened during the interrupted final experiment?

Epochs 1–3 completed normally and epoch 3 was already the saved best checkpoint. The process was interrupted during epoch 4. That epoch resumed from epoch-3 weights without optimizer, scheduler, or augmentation RNG state. Epoch 4 did not win, and the selected uninterrupted epoch-3 checkpoint was unaffected.

## What did Grad-CAM show?

The app generates Grad-CAM from V2. The retained saved case review was created earlier with V1 and is labeled historical. That review showed mixed behavior: some intracranial emphasis, but also skull, border, extracranial, and broad non-specific attention. Grad-CAM is qualitative post-hoc attention—not tumor segmentation or validated anatomical localization.

## What evidence suggests shortcut learning?

A logistic-regression probe using dimensions, aspect ratio, file size, mode, format, EXIF presence, original split, intensity, and border statistics—but no spatial anatomy—reached 70.84% validation accuracy and 0.7158 macro-F1. This demonstrates strong class-correlated source properties.

## Is confidence a measure of clinical certainty?

No. It is the model's softmax probability for one input under its learned internal distribution. V2 made 56 test errors, including high-confidence errors. Confidence does not measure diagnosis validity, calibration under distribution shift, or patient risk.

## Is the Streamlit app clinically usable?

No. It is a research and portfolio demonstration. It lacks verified patient-level separation, external and prospective testing, clinical workflow evaluation, calibration analysis, regulatory review, and coverage beyond four output labels.

## What would be required next?

Use patient/study identifiers to build group-independent cohorts, verify labels and acquisition metadata, reserve an external site, pre-register the analysis, estimate uncertainty, assess calibration and subgroup robustness, and validate explanations against expert annotations. Study-level 3D or multi-sequence modeling would require appropriate source data.
