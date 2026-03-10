#!/bin/zsh
set -euo pipefail
# Some environments enable BG_NICE and print noisy "nice(...) failed" for background jobs.
setopt NO_BG_NICE 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNNER="${SCRIPT_DIR}/fill_to_1000_parallel.sh"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${SCRIPT_DIR}/runs/parallel_${RUN_ID}"
LOG_FILE="${RUN_DIR}/run.log"

if [[ ! -x "${RUNNER}" ]]; then
  echo "Runner is not executable: ${RUNNER}"
  echo "Try: chmod +x stable_runtime/fill_to_1000_parallel.sh"
  exit 2
fi

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  stable_runtime/start_fill_to_1000_parallel_detached.sh [runner options]

Example:
  ./stable_runtime/start_fill_to_1000_parallel_detached.sh \
    --db "/Users/dmitrijfabarisov/Projects/Mango analyse/stable_runtime/semi_prod_400.db" \
    --target-done 1000 \
    --transcribe-limit 6 \
    --resolve-limit 10 \
    --analyze-limit 10 \
    --progress-every 20

All additional options are passed to stable_runtime/fill_to_1000_parallel.sh.
EOF
  exit 0
fi

mkdir -p "${RUN_DIR}"

nohup "${RUNNER}" --run-id "${RUN_ID}" "$@" > "${LOG_FILE}" 2>&1 &
PID="$!"
echo "${PID}" > "${RUN_DIR}/run.pid"
echo "parallel_${RUN_ID}" > "${SCRIPT_DIR}/runs/last_parallel_run_id.txt"

sleep 1
if ! kill -0 "${PID}" 2>/dev/null; then
  echo "Parallel fill run failed to start."
  echo "  run_id: parallel_${RUN_ID}"
  echo "  log: ${LOG_FILE}"
  echo
  echo "Last log lines:"
  tail -n 20 "${LOG_FILE}" 2>/dev/null || true
  exit 1
fi

echo "Started parallel fill run"
echo "  run_id: parallel_${RUN_ID}"
echo "  pid: ${PID}"
echo "  log: ${LOG_FILE}"
echo
echo "Monitor:"
echo "  tail -f \"${LOG_FILE}\""
echo "  tail -f \"${SCRIPT_DIR}/runs/$(cat "${SCRIPT_DIR}/runs/last_parallel_run_id.txt")/post_worker.log\""
echo
echo "Stop:"
echo "  kill ${PID}"
