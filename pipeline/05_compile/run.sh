#!/usr/bin/env bash
# Compilation: cross-lingual augmentation and final training-parquet materialization.
set -euo pipefail
cd "$(dirname "$0")/../.."
set -a; source .env; set +a

python pipeline/05_compile/compile.py \
  --config configs/thesis/05_compile.toml \
  --filtered-root "${FILTERING_DATA_ROOT}/filtered" \
  --hnm-root "${HNM_ROOT}" \
  --output "${COMPILED_DATA_DIR}" \
  "$@"
