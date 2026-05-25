#!/usr/bin/env bash
# Embedding: per-dataset FAISS index over query and passage embeddings.
set -euo pipefail
cd "$(dirname "$0")/../.."
set -a; source .env; set +a

python pipeline/02_embed/embed_shards.py \
  --config configs/thesis/02_embed.toml \
  --input-dir "${SPLIT_DATA_DIR}" \
  --output-dir "${EMBEDDINGS_DIR}" \
  "$@"
