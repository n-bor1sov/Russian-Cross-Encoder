#!/usr/bin/env bash
# Full pipeline on synthetic data. Requires EMBEDDING_API_BASE_URL to be set.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -f examples/.env ]]; then
    cp examples/.env.example examples/.env
fi
set -a; source examples/.env; set +a

if [[ -z "${EMBEDDING_API_BASE_URL:-}" ]]; then
    echo "EMBEDDING_API_BASE_URL is not set in examples/.env — full example needs an embedding endpoint."
    exit 1
fi

mkdir -p "${SPLIT_DATA_DIR}" "${EMBEDDINGS_DIR}" "${FILTERING_DATA_ROOT}" "${HNM_ROOT}" "${COMPILED_DATA_DIR}"

CFG_DIR=examples/configs

python pipeline/01_split/split.py --config "${CFG_DIR}/01_split.toml" \
    --input-dir "${UNIFIED_DATA_DIR}" --output-dir "${SPLIT_DATA_DIR}"
python pipeline/02_embed/embed_shards.py --config "${CFG_DIR}/02_embed.toml" \
    --input-dir "${SPLIT_DATA_DIR}" --output-dir "${EMBEDDINGS_DIR}"
python pipeline/03_filter_consistency/score.py --config "${CFG_DIR}/03_filter_consistency.toml" \
    --shards-root "${SPLIT_DATA_DIR}" --embeddings-dir "${EMBEDDINGS_DIR}" --output "${FILTERING_DATA_ROOT}"
python pipeline/03_filter_consistency/filter_by_rank.py --config "${CFG_DIR}/03_filter_consistency.toml" \
    --input-dir "${FILTERING_DATA_ROOT}" --output-dir "${FILTERING_DATA_ROOT}/filtered"
python pipeline/04_hard_negative_mining/mine.py --config "${CFG_DIR}/04_hnm.toml" \
    --input-shards "${FILTERING_DATA_ROOT}/filtered" \
    --embeddings-dir "${EMBEDDINGS_DIR}" --output "${HNM_ROOT}"
python pipeline/05_compile/compile.py --config "${CFG_DIR}/05_compile.toml" \
    --filtered-root "${FILTERING_DATA_ROOT}/filtered" \
    --hnm-root "${HNM_ROOT}" --output "${COMPILED_DATA_DIR}"

bash examples/run.sh
