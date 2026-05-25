#!/usr/bin/env bash
# Default example: skip stages 2-5 (require embedding API).
# Starts from examples/data/mini_compiled.parquet and runs training + evaluation.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -f examples/.env ]]; then
    cp examples/.env.example examples/.env
fi
set -a; source examples/.env; set +a

mkdir -p "${TRAIN_DATASET_PATH}" "${EVAL_DATASET_PATH}" "${TRAIN_OUTPUT_DIR}" "${TRAIN_LOG_DIR}"

# Bucket the pre-compiled mini training parquet
python pipeline/06_train/prepare_buckets.py \
  --input-dir examples/data \
  --output-dir "${TRAIN_DATASET_PATH}" \
  --num-buckets 4

# Use the same parquet for eval (synthetic example only)
cp -R "${TRAIN_DATASET_PATH}/." "${EVAL_DATASET_PATH}/"

torchrun --nproc_per_node=1 \
  pipeline/06_train/train.py \
  --config examples/configs/06_train.toml \
  --train-dataset "${TRAIN_DATASET_PATH}" \
  --eval-dataset "${EVAL_DATASET_PATH}" \
  --output-dir "${TRAIN_OUTPUT_DIR}"

bash evaluation/run.sh
echo "examples/run.sh: done — see ${EVAL_RESULTS_PATH}"
