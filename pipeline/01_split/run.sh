#!/usr/bin/env bash
# Splitting: bipartite-component train/val split.
set -euo pipefail
cd "$(dirname "$0")/../.."
set -a; source .env; set +a

python pipeline/01_split/split.py \
  --config configs/thesis/01_split.toml \
  --input-dir "${UNIFIED_DATA_DIR}" \
  --output-dir "${SPLIT_DATA_DIR}" \
  "$@"
