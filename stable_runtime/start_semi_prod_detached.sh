#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNNER="${SCRIPT_DIR}/semi_prod_run.sh"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${SCRIPT_DIR}/runs/${RUN_ID}"
LOG_FILE="${RUN_DIR}/run.log"

if [[ ! -x "${RUNNER}" ]]; then
  echo "Runner is not executable: ${RUNNER}"
  exit 2
fi

if [[ $# -eq 0 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  stable_runtime/start_semi_prod_detached.sh --input <recordings_dir> --output <transcripts_dir> [runner options]

Example:
  ./stable_runtime/start_semi_prod_detached.sh \
    --input "/path/to/mango_export" \
    --output "/path/to/transcripts_semi_prod" \
    --target 400 \
    --stage-limit 30 \
    --qc-every 50

All additional options are passed to stable_runtime/semi_prod_run.sh.
EOF
  exit 0
fi

mkdir -p "${RUN_DIR}"

nohup "${RUNNER}" --run-id "${RUN_ID}" "$@" > "${LOG_FILE}" 2>&1 &
PID="$!"
echo "${PID}" > "${RUN_DIR}/run.pid"

echo "Started semi-prod run"
echo "  run_id: ${RUN_ID}"
echo "  pid: ${PID}"
echo "  log: ${LOG_FILE}"
echo
echo "Monitor:"
echo "  tail -f \"${LOG_FILE}\""
echo
echo "Stop:"
echo "  kill ${PID}"
