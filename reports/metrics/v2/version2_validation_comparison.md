# Version 2 validation comparison

Status: historical validation-selection record, complete through the predeclared 224x224 three-block comparison. The selected epoch-3 checkpoint was subsequently evaluated once on the official held-out test split; final test results are reported separately in `final_v2_test_metrics.json`.

This comparison uses the same 690-image, Training-source-only validation subset for all reported metrics. The original Kaggle `Testing` source and the project's official held-out test split were not used for Version 2 selection. Version 1 metrics were recomputed from its frozen checkpoint on this same validation subset; Version 1 was not retrained.

| Candidate | Validation accuracy | Validation macro-F1 | Glioma recall | Meningioma recall | Glioma <-> Meningioma errors | Recorded training time | Trainable parameters | Best epoch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Version 1: EfficientNet-B0, 160x160, last two blocks | 92.46% | 92.56% | 89.52% | 89.71% | 25 | 29m 07.5s | 1,134,516 | 2 |
| Version 2: EfficientNet-B0, 160x160, last two blocks | 93.77% | 93.93% | 91.43% | 90.86% | 23 | 22m 18.7s | 1,134,516 | 6 |
| Version 2: EfficientNet-B0, 224x224, last two blocks | 96.23% | 96.21% | 97.14% | 93.71% | 12 | 47m 14.3s | 1,134,516 | 5 |
| **Version 2: EfficientNet-B0, 224x224, last three blocks** | **96.81%** | **97.01%** | **93.81%** | **97.14%** | **17** | **6h 46m 01.4s*** | **3,160,864** | **3** |

Primary selection result: the 224x224 three-block candidate has the highest validation macro-F1 (97.01%) and is therefore the selected Version 2 candidate under the predeclared primary criterion.

Relative to the 224x224 two-block candidate, the selected three-block checkpoint improves macro-F1 by 0.80 percentage points and accuracy by 0.58 points. It improves meningioma recall by 3.43 points, but reduces glioma recall by 3.33 points and increases direct glioma/meningioma errors from 12 to 17. This tradeoff must not be hidden.

## Three-block completion record

- Epochs 1-3 were complete before continuation; the earlier process had been interrupted during epoch 4.
- Only epoch 4 was completed during the continuation.
- Epoch 4 validation accuracy was 96.81% and macro-F1 was 96.97%; it did not replace the epoch-3 checkpoint.
- The saved checkpoint did not include optimizer, scheduler, or random-generator state. Epoch 4 therefore resumed from the epoch-3 model weights with a fresh AdamW state and a recorded continuation seed of `20260917`. This does not affect the selected checkpoint, which remains the uninterrupted epoch-3 checkpoint.
- *The three-block recorded wall time is not directly comparable: saved epoch times include long host pause/interruption intervals (especially epochs 2 and 3). It is retained exactly as recorded rather than replaced by an estimate.

## Configuration changes

- All Version 2 runs used only the legitimate Training-source subset: 3,221 training images and 690 validation images.
- All runs used EfficientNet-B0, weighted cross-entropy, the conservative Version 1 dynamic augmentation policy, ImageNet normalization, fixed seed `20260913`, and validation macro-F1 for selection.
- The 160x160 control kept the last two EfficientNet blocks trainable with learning rate `1e-4` and batch size 24.
- The 224x224 two-block run changed input resolution to 224x224 and batch size to 16, while retaining learning rate `1e-4`.
- The 224x224 three-block run unfroze one additional block, reduced learning rate to `5e-5`, retained batch size 16, and used a predeclared four-epoch maximum.
- At the point documented by this validation-selection record, no official held-out test evaluation or Streamlit update had yet been performed. The later one-time held-out evaluation and deployment integration did not alter this selection decision. No new architecture, loss, augmentation policy, or 256x256 experiment was performed.

## Validation confusion matrices

Class order: glioma, meningioma, notumor, pituitary.

Version 1:

```text
[[188, 21, 1, 0],
 [  4,157, 2,12],
 [  4,  2,87, 1],
 [  2,  3, 0,206]]
```

Version 2, 160x160, last two blocks:

```text
[[192, 17, 0, 1],
 [  6,159, 2, 8],
 [  2,  1,89, 2],
 [  1,  3, 0,207]]
```

Version 2, 224x224, last two blocks:

```text
[[204, 6, 0, 0],
 [  6,164, 0, 5],
 [  4, 1,88, 1],
 [  1, 2, 0,208]]
```

Version 2, 224x224, last three blocks (selected epoch 3):

```text
[[197,13, 0, 0],
 [  4,170,0, 1],
 [  1, 0,92, 1],
 [  0, 2, 0,209]]
```
