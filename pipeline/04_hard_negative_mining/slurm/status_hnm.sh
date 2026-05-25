#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="${LOG_DIR:-./logs/hnm}"
TAIL_LINES="${TAIL_LINES:-5}"

echo "HNM status at $(date -Is)"
echo "Log dir: $LOG_DIR"
echo

mapfile -t HNM_PIDS < <(
  pgrep -f '(^|[ /])(HNM\.sh|mine\.py)([ ]|$)' \
    | sort -u \
    | grep -v "^$$$" \
    | grep -v "^$PPID$" \
    || true
)

if (( ${#HNM_PIDS[@]} == 0 )); then
  echo "No running HNM processes found."
  exit 0
fi

echo "Running HNM processes:"
ps -o pid,ppid,stat,etime,command -p "$(IFS=,; echo "${HNM_PIDS[*]}")"
echo

find_log_for_command() {
  local command="$1"
  local dataset=""
  local output_dir=""
  local split=""
  local candidate=""

  dataset="$(sed -nE 's/.*(^|[[:space:]])-n[[:space:]]+([^[:space:]]+).*/\2/p' <<<"$command")"
  output_dir="$(sed -nE 's/.*(^|[[:space:]])-o[[:space:]]+([^[:space:]]+).*/\2/p' <<<"$command")"

  if [[ -z "$dataset" ]]; then
    return 1
  fi

  case "$output_dir" in
    *mined_negatives/train/filtered_top_30*) split="train_filtered_top_30" ;;
    *mined_negatives/train/filtered_top_5*) split="train_filtered_top_5" ;;
    *mined_negatives/train/restored_top30*) split="train_restored_top30" ;;
    *mined_negatives/validation*) split="validation" ;;
  esac

  if [[ -n "$split" ]]; then
    candidate="$LOG_DIR/${split}_${dataset}.log"
    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  fi

  candidate="$(ls -t "$LOG_DIR"/*_"$dataset".log 2>/dev/null | head -n 1 || true)"
  if [[ -n "$candidate" ]]; then
    printf '%s\n' "$candidate"
    return 0
  fi

  return 1
}

echo "Last $TAIL_LINES log lines for active mine.py miners:"
echo

found_active_miner=0
for pid in "${HNM_PIDS[@]}"; do
  command="$(ps -o command= -p "$pid" || true)"
  if [[ "$command" != *"mine.py"* ]]; then
    continue
  fi

  found_active_miner=1
  log_file="$(find_log_for_command "$command" || true)"

  echo
  echo "================================================================================"
  echo "PID: $pid"
  echo "================================================================================"
  echo "Command: $command"
  if [[ -z "$log_file" ]]; then
    echo "Log: not found"
    echo "================================================================================"
    continue
  fi

  echo "Log: $log_file"
  tail -n "$TAIL_LINES" "$log_file"
  echo "================================================================================"
done

if (( found_active_miner == 0 )); then
  echo "No active mine.py miner processes found. HNM.sh may still be waiting or finishing."
fi

echo
echo "================================================================================"
echo "End of HNM status at $(date -Is)"
echo "================================================================================"
