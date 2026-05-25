#!/usr/bin/env bash
set -euo pipefail

DRY_RUN="${DRY_RUN:-0}"
TERM_TIMEOUT_SECONDS="${TERM_TIMEOUT_SECONDS:-15}"

PATTERNS=(
  '(^|[ /])HNM\.sh([ ]|$)'
  '(^|[ /])mine\.py([ ]|$)'
)

collect_pids() {
  local pattern
  local pid

  for pattern in "${PATTERNS[@]}"; do
    pgrep -f "$pattern" || true
  done | sort -u | while read -r pid; do
    [[ -z "$pid" ]] && continue
    [[ "$pid" == "$$" ]] && continue
    [[ "$pid" == "$PPID" ]] && continue
    printf '%s\n' "$pid"
  done
}

mapfile -t PIDS < <(collect_pids)

if (( ${#PIDS[@]} == 0 )); then
  echo "No HNM processes found."
  exit 0
fi

echo "Found HNM processes:"
ps -o pid,ppid,stat,etime,command -p "$(IFS=,; echo "${PIDS[*]}")"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN=1, not killing anything."
  exit 0
fi

echo "Sending SIGTERM..."
kill -TERM "${PIDS[@]}" 2>/dev/null || true

deadline=$((SECONDS + TERM_TIMEOUT_SECONDS))
while (( SECONDS < deadline )); do
  mapfile -t REMAINING < <(collect_pids)
  if (( ${#REMAINING[@]} == 0 )); then
    echo "All HNM processes stopped."
    exit 0
  fi
  sleep 1
done

mapfile -t REMAINING < <(collect_pids)
if (( ${#REMAINING[@]} > 0 )); then
  echo "Some HNM processes are still running, sending SIGKILL:"
  ps -o pid,ppid,stat,etime,command -p "$(IFS=,; echo "${REMAINING[*]}")"
  kill -KILL "${REMAINING[@]}" 2>/dev/null || true
fi

echo "Done."
