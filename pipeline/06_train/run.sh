#!/usr/bin/env bash
# Training: dataset-aware bucketed DDP fine-tuning of the cross-encoder.
set -euo pipefail
cd "$(dirname "$0")/../.."
set -a; source .env; set +a

# One-time bucket preparation (idempotent — script no-ops if outputs exist).
python pipeline/06_train/prepare_buckets.py \
  --input-dir "${COMPILED_DATA_DIR}" \
  --output-dir "${TRAIN_DATASET_PATH}" \
  --num-buckets "$(python -c 'import tomllib; print(tomllib.loads(open("configs/thesis/06_train.toml").read())["num_buckets"])')"

torchrun --nproc_per_node="${NPROC_PER_NODE:-8}" \
  pipeline/06_train/train.py \
  --config configs/thesis/06_train.toml \
  --train-dataset "${TRAIN_DATASET_PATH}" \
  --eval-dataset "${EVAL_DATASET_PATH}" \
  --output-dir "${TRAIN_OUTPUT_DIR}" \
  "$@"
