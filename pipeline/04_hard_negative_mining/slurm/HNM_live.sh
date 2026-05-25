#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/home/jovyan/new-pvc/cross_encoders/new_faiss/bin/python}"
GPU_COUNT="${GPU_COUNT:-8}"
EMBEDDINGS_ROOT="${EMBEDDINGS_ROOT:-/home/jovyan/new-pvc/cross_encoders/nik_datasets/parquets/consistency_filtering_data/restored_embeddings}"
EMBEDDING_FORMAT="${EMBEDDING_FORMAT:-sqlite}"
MODEL_PATH="${MODEL_PATH:-/home/jovyan/new-pvc/cross_encoders/nik_datasets/models/qwen-tokenizer}"
API_BASE_URL="${API_BASE_URL:-https://airun.dmp.x5.ru/x5-airun-embed-qwen-rnd/v1}"
API_KEY="${API_KEY:-${OPENAI_API_KEY:-aboba}}"
API_MODEL="${API_MODEL:-x5-airun-embed-qwen-rnd}"
EMBEDDING_BATCH_SIZE_QUERY="${EMBEDDING_BATCH_SIZE_QUERY:-4096}"
EMBEDDING_BATCH_SIZE_PASSAGE="${EMBEDDING_BATCH_SIZE_PASSAGE:-512}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
N_PARALLEL="${N_PARALLEL:-2}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
LOG_DIR="${LOG_DIR:-./logs/hnm}"

mkdir -p "$LOG_DIR"

echo "Python command: $PYTHON_BIN"
echo "Python path: $("$PYTHON_BIN" -c 'import sys; print(sys.executable)')"

if (( GPU_COUNT <= 0 )); then
  echo "GPU_COUNT must be positive, got: $GPU_COUNT" >&2
  exit 1
fi

JOBS=()
FAILED=0

run_hnm_job() {
  local split="$1"
  local out_dir="$2"
  local name="$3"
  local path="$4"
  local embedding_mode="$5"
  local gpu_id="$6"
  local log_file

  log_file="$LOG_DIR/${split}_${name}.log"

  status=0
  cmd=(
    "$PYTHON_BIN" ../mine.py
    -d "$path"
    -o "$out_dir"
    -n "$name"
    --gpu-id "$gpu_id"
    --embedding-dimension 1024
    --log-level "$LOG_LEVEL"
  )

  if [[ "$embedding_mode" == "restored" ]]; then
    cmd+=(
      --embeddings-root "$EMBEDDINGS_ROOT"
      --embedding-format "$EMBEDDING_FORMAT"
    )
  elif [[ "$embedding_mode" == "live" ]]; then
    cmd+=(
      --model-path "$MODEL_PATH"
      --api-base-url "$API_BASE_URL"
      --api-key "$API_KEY"
      --api-model "$API_MODEL"
      --embedding-batch-size-query "$EMBEDDING_BATCH_SIZE_QUERY"
      --embedding-batch-size-passage "$EMBEDDING_BATCH_SIZE_PASSAGE"
      --max-tokens "$MAX_TOKENS"
      --n-parallel "$N_PARALLEL"
    )
  else
    echo "Unsupported embedding mode: $embedding_mode" >&2
    return 1
  fi

  echo "Started $split/$name on GPU $gpu_id; log: $log_file"
  if {
    echo "========================================"
    echo "Running HNM for dataset: $name"
    echo "Split: $split"
    echo "GPU: $gpu_id"
    echo "Embedding mode: $embedding_mode"
    echo "Path: $path"
    echo "Output dir: $out_dir"
    echo "Python command: $PYTHON_BIN"
    echo "Python path: $("$PYTHON_BIN" -c 'import sys; print(sys.executable)')"
    if [[ "$embedding_mode" == "restored" ]]; then
      echo "Embeddings root: $EMBEDDINGS_ROOT"
    else
      echo "Model path: $MODEL_PATH"
      echo "API model: $API_MODEL"
    fi
    echo "Log level: $LOG_LEVEL"
    echo "Started at: $(date -Is)"
    echo "========================================"

    "${cmd[@]}"
  } >"$log_file" 2>&1; then
    status=0
  else
    status=$?
  fi

  {
    echo "========================================"
    echo "Finished dataset: $name"
    echo "Split: $split"
    echo "GPU: $gpu_id"
    echo "Exit code: $status"
    echo "Finished at: $(date -Is)"
    echo "========================================"
  } >>"$log_file"

  return "$status"
}

enqueue_hnm_job() {
  local split="$1"
  local out_dir="$2"
  local name="$3"
  local path="$4"
  local embedding_mode="$5"
  local log_file="$LOG_DIR/${split}_${name}.log"

  JOBS+=("${split}"$'\t'"${out_dir}"$'\t'"${name}"$'\t'"${path}"$'\t'"${embedding_mode}")
  echo "Queued $split/$name; log: $log_file"
}

launch_queued_job() {
  local job_index="$1"
  local gpu_id="$2"
  local job="${JOBS[$job_index]}"
  local split
  local out_dir
  local name
  local path
  local embedding_mode
  local status

  IFS=$'\t' read -r split out_dir name path embedding_mode <<<"$job"

  (
    if run_hnm_job "$split" "$out_dir" "$name" "$path" "$embedding_mode" "$gpu_id"; then
      status=0
    else
      status=$?
    fi
    printf '%s\t%s\n' "$gpu_id" "$status" >&8
    exit "$status"
  ) &
}

run_queued_jobs() {
  local done_fifo
  local gpu_id
  local status
  local next_job=0
  local running=0
  local total_jobs="${#JOBS[@]}"
  local initial_workers

  if (( total_jobs == 0 )); then
    echo "No HNM jobs queued."
    return
  fi

  done_fifo="$(mktemp -u)"
  mkfifo "$done_fifo"
  exec 8<>"$done_fifo"
  rm -f "$done_fifo"

  initial_workers="$GPU_COUNT"
  if (( total_jobs < initial_workers )); then
    initial_workers="$total_jobs"
  fi

  echo "Running $total_jobs queued jobs with up to $GPU_COUNT GPUs"
  for ((gpu_id = 0; gpu_id < initial_workers; gpu_id++)); do
    launch_queued_job "$next_job" "$gpu_id"
    next_job=$((next_job + 1))
    running=$((running + 1))
  done

  while (( running > 0 )); do
    IFS=$'\t' read -r gpu_id status <&8
    running=$((running - 1))
    if [[ "$status" != "0" ]]; then
      FAILED=1
      echo "HNM job on GPU $gpu_id failed with exit code $status. Check logs in $LOG_DIR" >&2
    fi

    if (( next_job < total_jobs )); then
      launch_queued_job "$next_job" "$gpu_id"
      next_job=$((next_job + 1))
      running=$((running + 1))
    fi
  done

  exec 8>&-
}

run_split() {
  local split="$1"
  local out_dir="$2"
  local embedding_mode="$3"
  shift 3
  local datasets=("$@")
  local ds
  local name
  local path

  mkdir -p "$out_dir"
  echo "Starting split '$split' with ${#datasets[@]} datasets, up to $GPU_COUNT parallel jobs, embedding_mode=$embedding_mode"

  for ds in "${datasets[@]}"; do
    name="${ds%%:*}"
    path="${ds#*:}"
    enqueue_hnm_job "$split" "$out_dir" "$name" "$path" "$embedding_mode"
  done
}

TRAIN_FILTERED_TOP_30_DATASETS=(
  "habr_qna:/home/jovyan/new-pvc/cross_encoders/nik_datasets/parquets/consistency_filtering_data/filtered_top_30/habr_qna.parquet"
)

# VALIDATION_DATASETS=(
#   "mmarco:/home/jovyan/new-pvc/cross_encoders/nik_datasets/parquets/cleaned_datasets/splits/validation/mmarco.parquet"
#   "9111_questions:/home/jovyan/new-pvc/cross_encoders/nik_datasets/parquets/cleaned_datasets/splits/validation/legal_9111_unified.parquet"
#   "habr_qna:/home/jovyan/new-pvc/cross_encoders/nik_datasets/parquets/cleaned_datasets/splits/validation/habr_qna_unified.parquet"
#   "nq:/home/jovyan/new-pvc/cross_encoders/nik_datasets/parquets/cleaned_datasets/splits/validation/nq_unified.parquet"
#   "ru_news:/home/jovyan/new-pvc/cross_encoders/nik_datasets/parquets/cleaned_datasets/splits/validation/ru_news_cleaned.parquet"
#   "ru_sci_bench:/home/jovyan/new-pvc/cross_encoders/nik_datasets/parquets/cleaned_datasets/splits/validation/ru_sci_bench_merged.parquet"
#   "ru_sci_bench_ru:/home/jovyan/new-pvc/cross_encoders/nik_datasets/parquets/cleaned_datasets/splits/validation/ru_sci_bench_ru.parquet"
#   "ru_stackoverflow:/home/jovyan/new-pvc/cross_encoders/nik_datasets/parquets/cleaned_datasets/splits/validation/ru_stackoverflow.parquet"
#   "swim_ir:/home/jovyan/new-pvc/cross_encoders/nik_datasets/parquets/cleaned_datasets/splits/validation/swim_ir_ru_en_unified.parquet"
#   "yandex_q:/home/jovyan/new-pvc/cross_encoders/nik_datasets/parquets/cleaned_datasets/splits/validation/yandex_q_merged.parquet"
#   "habr:/home/jovyan/new-pvc/cross_encoders/nik_datasets/parquets/cleaned_datasets/splits/validation/habr.parquet"
# )

run_split "train_filtered_top_30_live" "./mined_negatives/train/filtered_top_30_live" "live" "${TRAIN_FILTERED_TOP_30_DATASETS[@]}"
# run_split "validation" "./mined_negatives/validation" "live" "${VALIDATION_DATASETS[@]}"

run_queued_jobs

if (( FAILED != 0 )); then
  echo "Some HNM jobs failed. Check logs in $LOG_DIR" >&2
  exit 1
fi

echo "All datasets processed successfully."

# Ex: nohup bash HNM.sh > hnm_run.log 2>&1 &
