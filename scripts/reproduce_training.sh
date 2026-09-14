#!/usr/bin/env bash
set -euo pipefail

if [[ "${ALLOW_EXPENSIVE_TRAINING:-}" != "YES" ]]; then
  echo "Historical training is disabled by default."
  echo "Run only in a fresh experimental clone with ALLOW_EXPENSIVE_TRAINING=YES."
  exit 2
fi

echo "This archival procedure is expensive and writes new experiment artifacts."
echo "It must not be run in the frozen release directory."

bash scripts/reproduce_training_v1.sh

python -m src.train_v2_frozen_features \
  --run-name v2_b0_160_frozen_cached --input-size 160
python -m src.train_v2 \
  --run-name v2_b0_160_last2_control \
  --initial-checkpoint models/v2/v2_b0_160_frozen_cached_best.pt \
  --input-size 160 --unfrozen-blocks 2 --epochs 6 --batch-size 24 \
  --learning-rate 1e-4 --patience 3
python -m src.train_v2_frozen_features \
  --run-name v2_b0_224_frozen_cached --input-size 224
python -m src.train_v2 \
  --run-name v2_b0_224_last2_resolution \
  --initial-checkpoint models/v2/v2_b0_224_frozen_cached_best.pt \
  --input-size 224 --unfrozen-blocks 2 --epochs 6 --batch-size 16 \
  --learning-rate 1e-4 --patience 3
python -m src.train_v2 \
  --run-name v2_b0_224_last3_unfreeze \
  --initial-checkpoint models/v2/v2_b0_224_frozen_cached_best.pt \
  --input-size 224 --unfrozen-blocks 3 --epochs 4 --batch-size 16 \
  --learning-rate 5e-5 --patience 3

echo "Historical run configurations are the source of truth under reports/metrics/v2/."
echo "The recorded epoch-4 interruption cannot be reproduced bit-for-bit; see its resume note."
