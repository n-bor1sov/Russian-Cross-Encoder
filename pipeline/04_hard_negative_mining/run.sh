#!/usr/bin/env bash
# Hard negative mining: margin-based selection from per-dataset FAISS index.
set -euo pipefail
cd "$(dirname "$0")/../.."
set -a; source .env; set +a

python pipeline/04_hard_negative_mining/mine.py \
  --config configs/thesis/04_hnm.toml \
  --input-shards "${HNM_INPUT_DIR:-${FILTERING_DATA_ROOT}/filtered}" \
  --embeddings-dir "${EMBEDDINGS_DIR}/filtered" \
  --output "${HNM_ROOT}" \
  "$@"
