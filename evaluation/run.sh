#!/usr/bin/env bash
# Evaluation: BM25 top-100 → reranker → NDCG@10 aggregated across RusBEIR.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a

python evaluation/generate_bm25_top100.py \
  --config configs/thesis/eval.toml \
  --output "${EVAL_BM25_TOP100_PATH}"

python evaluation/evaluate_reranker_top100.py \
  --config configs/thesis/eval.toml \
  --model "${TRAIN_OUTPUT_DIR}/best" \
  --top100 "${EVAL_BM25_TOP100_PATH}" \
  --output "${EVAL_RESULTS_PATH}" \
  "$@"
