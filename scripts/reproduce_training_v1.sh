#!/usr/bin/env bash
set -euo pipefail

if [[ "${ALLOW_EXPENSIVE_TRAINING:-}" != "YES" ]]; then
  echo "Historical V1 training is disabled by default."
  echo "Run only in a fresh experimental clone with ALLOW_EXPENSIVE_TRAINING=YES."
  exit 2
fi

# Run from the repository root. The original data/ tree is read-only input.
python -m src.prepare_data
python -m src.analyze_shortcuts
python -m src.audit_near_duplicates

# Training is intentionally explicit and expensive. Skip these commands when
# inspecting the supplied completed checkpoints and metrics.
python -m src.train --architecture baseline_cnn --stage baseline --run-name baseline_cnn --epochs 15 --input-size 128
python -m src.train_frozen_features
python -m src.train --architecture efficientnet_b0 --run-name efficientnet_b0_finetuned --stage finetune --epochs 6 --patience 3 --learning-rate 1e-4 --resume models/efficientnet_b0_frozen_best.pt --input-size 160

# Selection uses validation metrics only. Evaluation follows selection.
python -m src.evaluate --checkpoint models/baseline_cnn_best.pt --run-name baseline_cnn --split validation
python -m src.evaluate --checkpoint models/efficientnet_b0_finetuned_best.pt --run-name efficientnet_b0_finetuned --split validation
python -m src.select_model \
  --candidate baseline_cnn=models/baseline_cnn_best.pt \
  --candidate efficientnet_b0_finetuned=models/efficientnet_b0_finetuned_best.pt
python -m src.evaluate --checkpoint models/baseline_cnn_best.pt --run-name baseline_cnn --split test
python -m src.evaluate --checkpoint models/efficientnet_b0_finetuned_best.pt --run-name efficientnet_b0_finetuned --split test
python -m src.summarize_results
python -m src.gradcam --selection models/selected_model.json

echo "Historical V1 pipeline complete. Run final V2 verification separately after V2 artifacts exist."
