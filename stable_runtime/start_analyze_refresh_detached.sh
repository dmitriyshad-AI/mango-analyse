#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_PYTHON="${SCRIPT_DIR}/venv_stable/bin/python"

if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "Stable runtime is missing: ${VENV_PYTHON}"
  echo "Rebuild stable runtime first."
  exit 1
fi

DB_PATH="${PROJECT_ROOT}/mango_mvp.db"
BATCH_LIMIT=20
CONCURRENCY=4
PROVIDER="${ANALYZE_PROVIDER:-codex_cli}"
MODEL="${CODEX_ANALYZE_MODEL:-gpt-5.4-mini}"
REASONING="${CODEX_REASONING_EFFORT:-medium}"
RESET_LIMIT=200000
SKIP_RESET=0
REBUILD_THRESHOLD=100
WORKBOOK_PATH="${PROJECT_ROOT}/sales_workbook.xlsx"
EXPORT_POLL_SEC=15
WORKER_IDLE_CYCLES=30
TARGET_DONE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --db)
      DB_PATH="$2"
      shift 2
      ;;
    --batch-limit)
      BATCH_LIMIT="$2"
      shift 2
      ;;
    --concurrency)
      CONCURRENCY="$2"
      shift 2
      ;;
    --provider)
      PROVIDER="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --reasoning)
      REASONING="$2"
      shift 2
      ;;
    --skip-reset)
      SKIP_RESET=1
      shift
      ;;
    --target-done)
      TARGET_DONE="$2"
      shift 2
      ;;
    --rebuild-threshold)
      REBUILD_THRESHOLD="$2"
      shift 2
      ;;
    --workbook-out)
      WORKBOOK_PATH="$2"
      shift 2
      ;;
    --export-poll-sec)
      EXPORT_POLL_SEC="$2"
      shift 2
      ;;
    --worker-idle-cycles)
      WORKER_IDLE_CYCLES="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

mkdir -p "${SCRIPT_DIR}/runs"
RUN_ID="analyze_refresh_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${SCRIPT_DIR}/runs/${RUN_ID}"
LOG_FILE="${RUN_DIR}/run.log"
CONTROLLER_SCRIPT="${RUN_DIR}/controller.sh"
WORKBOOK_MARKER="${RUN_DIR}/workbook_exported.count"
mkdir -p "${RUN_DIR}"

export PATH="${PROJECT_ROOT}/.local/bin:${PATH}"

cat >"${CONTROLLER_SCRIPT}" <<EOF
#!/bin/zsh
set -euo pipefail
cd "${PROJECT_ROOT}"
export PATH="${PROJECT_ROOT}/.local/bin:\${PATH}"
export PYTHONPATH=src
export DATABASE_URL="sqlite:///${DB_PATH}"
export ANALYZE_PROVIDER="${PROVIDER}"
export CODEX_ANALYZE_MODEL="${MODEL}"
export CODEX_REASONING_EFFORT="${REASONING}"
export CODEX_CLI_COMMAND="${CODEX_CLI_COMMAND:-codex}"
export ANALYZE_MAX_ATTEMPTS="\${ANALYZE_MAX_ATTEMPTS:-3}"

count_state() {
  "${VENV_PYTHON}" - <<'PY'
import datetime
import os
import sqlite3

db_path = os.environ["DB_PATH"]
now = datetime.datetime.now(datetime.timezone.utc).isoformat(sep=" ")
max_attempts = int(os.environ.get("ANALYZE_MAX_ATTEMPTS", "3") or "3")
conn = sqlite3.connect(db_path)
cur = conn.cursor()
cur.execute(
    """
    SELECT count(*)
      FROM call_records
     WHERE transcription_status='done'
       AND (resolve_status IN ('done','skipped') OR resolve_status IS NULL)
       AND dead_letter_stage IS NULL
       AND analysis_status IN ('pending','failed')
       AND analyze_attempts < ?
       AND (next_retry_at IS NULL OR next_retry_at <= ?)
    """,
    (max_attempts, now),
)
eligible = int(cur.fetchone()[0] or 0)
cur.execute("SELECT count(*) FROM call_records WHERE analysis_status='in_progress'")
in_progress = int(cur.fetchone()[0] or 0)
cur.execute(
    """
    SELECT count(*)
      FROM call_records
     WHERE transcription_status='done'
       AND (resolve_status IN ('done','skipped') OR resolve_status IS NULL)
       AND dead_letter_stage IS NULL
       AND analysis_status='done'
    """
)
done_count = int(cur.fetchone()[0] or 0)
cur.execute(
    """
    SELECT count(*)
      FROM call_records
     WHERE transcription_status='done'
       AND (resolve_status IN ('done','skipped') OR resolve_status IS NULL)
       AND dead_letter_stage IS NULL
       AND analysis_status='failed'
    """
)
failed_count = int(cur.fetchone()[0] or 0)
cur.execute("SELECT count(*) FROM call_records WHERE dead_letter_stage='analyze'")
dead_count = int(cur.fetchone()[0] or 0)
conn.close()
print(f"{eligible}|{in_progress}|{done_count}|{failed_count}|{dead_count}")
PY
}

parse_processed() {
  ANALYZE_JSON="\$1" "${VENV_PYTHON}" - <<'PY'
import json
import os
import sys

raw = os.environ.get("ANALYZE_JSON", "").strip()
try:
    payload = json.loads(raw) if raw else {}
except Exception:
    print(0)
    raise SystemExit(0)
print(int(payload.get("processed", 0) or 0))
PY
}

analyze_worker() {
  local worker_idx="\$1"
  local idle=0
  echo "[worker-\${worker_idx}] start batch_limit=${BATCH_LIMIT}"
  while true; do
    local state eligible in_progress done_count failed_count dead_count remaining_claim effective_limit
    state="\$(count_state)"
    IFS='|' read -r eligible in_progress done_count failed_count dead_count <<< "\${state}"
    if [[ "${TARGET_DONE}" -gt 0 ]]; then
      remaining_claim=\$(( ${TARGET_DONE} - done_count - in_progress ))
      if [[ "\${remaining_claim}" -le 0 ]]; then
        echo "[worker-\${worker_idx}] target reached or fully claimed"
        break
      fi
      effective_limit="${BATCH_LIMIT}"
      if [[ "\${remaining_claim}" -lt "\${effective_limit}" ]]; then
        effective_limit="\${remaining_claim}"
      fi
    else
      effective_limit="${BATCH_LIMIT}"
    fi

    if [[ "\${eligible}" -eq 0 && "\${in_progress}" -eq 0 ]]; then
      echo "[worker-\${worker_idx}] no more work"
      break
    fi

    local output rc processed
    output=\$("${VENV_PYTHON}" -m mango_mvp.cli analyze --limit "\${effective_limit}") || rc=\$?
    rc=\${rc:-0}
    echo "\${output}"
    processed=\$(parse_processed "\${output}")

    if [[ "\${rc}" -ne 0 ]]; then
      echo "[worker-\${worker_idx}] analyze rc=\${rc}"
      idle=\$((idle + 1))
      sleep 2
      continue
    fi

    if [[ "\${processed}" -le 0 ]]; then
      idle=\$((idle + 1))
      if [[ "\${idle}" -ge "${WORKER_IDLE_CYCLES}" ]]; then
        state="\$(count_state)"
        IFS='|' read -r eligible in_progress done_count failed_count dead_count <<< "\${state}"
        if [[ "\${eligible}" -eq 0 && "\${in_progress}" -eq 0 ]]; then
          echo "[worker-\${worker_idx}] idle exit"
          break
        fi
        idle=0
      fi
      sleep 2
      continue
    fi

    idle=0
    sleep 1
  done
  echo "[worker-\${worker_idx}] done"
}

exporter_loop() {
  local last_exported=0
  if [[ -f "${WORKBOOK_MARKER}" ]]; then
    last_exported=\$(cat "${WORKBOOK_MARKER}" 2>/dev/null || echo 0)
  fi
  echo "[exporter] start rebuild_threshold=${REBUILD_THRESHOLD}"
  while true; do
    local state eligible in_progress done_count failed_count dead_count should_export
    state="\$(count_state)"
    IFS='|' read -r eligible in_progress done_count failed_count dead_count <<< "\${state}"
    echo "[state] eligible=\${eligible} in_progress=\${in_progress} done=\${done_count} failed=\${failed_count} dead=\${dead_count}"

    should_export=0
    if [[ "\${done_count}" -ge "${REBUILD_THRESHOLD}" && \$((done_count - last_exported)) -ge "${REBUILD_THRESHOLD}" ]]; then
      should_export=1
    fi
    if [[ "\${eligible}" -eq 0 && "\${in_progress}" -eq 0 && "\${done_count}" -gt "\${last_exported}" ]]; then
      should_export=1
    fi

    if [[ "\${should_export}" -eq 1 ]]; then
      "${VENV_PYTHON}" -m mango_mvp.cli export-sales-workbook --out "${WORKBOOK_PATH}" --only-done --limit 200000
      echo "\${done_count}" > "${WORKBOOK_MARKER}"
      last_exported="\${done_count}"
      echo "[\$(date '+%F %T')] workbook_exported=${WORKBOOK_PATH} done_count=\${done_count}"
    fi

    if [[ "${TARGET_DONE}" -gt 0 && "\${done_count}" -ge "${TARGET_DONE}" && "\${in_progress}" -eq 0 ]]; then
      break
    fi
    if [[ "\${eligible}" -eq 0 && "\${in_progress}" -eq 0 ]]; then
      break
    fi
    sleep "${EXPORT_POLL_SEC}"
  done
  echo "[exporter] done"
}

child_pids=()
cleanup() {
  trap - EXIT INT TERM
  for pid in "\${child_pids[@]:-}"; do
    kill "\${pid}" 2>/dev/null || true
  done
  wait || true
}
trap cleanup EXIT INT TERM

export DB_PATH="${DB_PATH}"
echo "[start] run_id=${RUN_ID}"
echo "[start] db=${DB_PATH}"
echo "[start] provider=${PROVIDER}"
echo "[start] model=${MODEL}"
echo "[start] reasoning=${REASONING}"
echo "[start] batch_limit=${BATCH_LIMIT}"
echo "[start] concurrency=${CONCURRENCY}"
echo "[start] skip_reset=${SKIP_RESET}"
echo "[start] target_done=${TARGET_DONE}"
echo "[start] rebuild_threshold=${REBUILD_THRESHOLD}"
echo "[start] export_poll_sec=${EXPORT_POLL_SEC}"
echo "[start] workbook_out=${WORKBOOK_PATH}"

if [[ "${SKIP_RESET}" -eq 0 ]]; then
  "${VENV_PYTHON}" -m mango_mvp.cli reset-analysis \\
    --statuses done,failed,pending,dead,in_progress \\
    --only-terminal-resolve \\
    --clear-json \\
    --clear-error \\
    --limit "${RESET_LIMIT}"
fi

for worker_idx in \$(seq 1 "${CONCURRENCY}"); do
  analyze_worker "\${worker_idx}" >> "${LOG_FILE}" 2>&1 &
  child_pids+=("\$!")
done

exporter_loop >> "${LOG_FILE}" 2>&1 &
child_pids+=("\$!")

wait
echo "[done] run_id=${RUN_ID}"
EOF

chmod +x "${CONTROLLER_SCRIPT}"
PID=$(PY_CONTROLLER="${CONTROLLER_SCRIPT}" PY_LOG="${LOG_FILE}" PY_CWD="${PROJECT_ROOT}" python3 - <<'PY'
import os
import subprocess

controller = os.environ["PY_CONTROLLER"]
log_file = os.environ["PY_LOG"]
cwd = os.environ["PY_CWD"]
with open(log_file, "ab", buffering=0) as log:
    proc = subprocess.Popen(
        ["/bin/zsh", controller],
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        close_fds=True,
        cwd=cwd,
    )
print(proc.pid)
PY
)

echo "Started analyze refresh run"
echo "  run_id: ${RUN_ID}"
echo "  pid: ${PID}"
echo "  log: ${LOG_FILE}"
echo
echo "Monitor:"
echo "  tail -f \"${LOG_FILE}\""
echo
echo "Stop:"
echo "  kill ${PID}"
