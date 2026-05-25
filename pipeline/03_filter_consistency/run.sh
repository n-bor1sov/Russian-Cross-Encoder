#!/usr/bin/env bash
# Consistency filtering: keep pairs whose positive is retrievable.
set -euo pipefail
cd "$(dirname "$0")/../.."
set -a; source .env; set +a

# Step A: score retrieval_rank and virtual_rank for every pair
python pipeline/03_filter_consistency/score.py \
  --config configs/thesis/03_filter_consistency.toml \
  --shards-root "${SPLIT_DATA_DIR}" \
  --embeddings-dir "${EMBEDDINGS_DIR}" \
  --output "${FILTERING_DATA_ROOT}"

# Step B: apply the rank filter (Eq. 5) to produce the surviving set
python pipeline/03_filter_consistency/filter_by_rank.py \
  --config configs/thesis/03_filter_consistency.toml \
  --input-dir "${FILTERING_DATA_ROOT}" \
  --output-dir "${FILTERING_DATA_ROOT}/filtered"

# Step C: rebuild per-dataset embedding indices for survivors
python pipeline/03_filter_consistency/restore_embeddings.py \
  --shards-root "${FILTERING_DATA_ROOT}/filtered" \
  --embeddings-dir "${EMBEDDINGS_DIR}" \
  --output-dir "${EMBEDDINGS_DIR}/filtered"

# Pass-through CLI overrides apply only to score.py (the most likely target).
