#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_PYTHON="${SCRIPT_DIR}/venv_stable/bin/python"
RUN_ROOT="${PROJECT_ROOT}/stable_runtime/recent_window_20260319_20260326_mini"
SOURCE_DB="${PROJECT_ROOT}/stable_runtime/history_cohort_20260319_20260326/history_cohort_20260319_20260326_asr.db"

if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "Stable runtime is missing: ${VENV_PYTHON}"
  exit 1
fi

export PATH="/Applications/Codex.app/Contents/Resources:${PROJECT_ROOT}/.local/bin:${PATH}"
unset PYTHONPATH || true

"${VENV_PYTHON}" "${PROJECT_ROOT}/scripts/prepare_date_window_subset.py" \
  --source-db "${SOURCE_DB}" \
  --out-root "${RUN_ROOT}" \
  --start-date "2026-03-19" \
  --end-date "2026-03-26" \
  >/dev/null

DB_PATH="${RUN_ROOT}/recent_window_20260319_20260326_mini.db"
TRANSCRIPTS_DIR="${RUN_ROOT}/transcripts"
RESOLVE_LOG="${RUN_ROOT}/resolve.log"
ANALYZE_LOG="${RUN_ROOT}/analyze.log"

rm -f "${RESOLVE_LOG}" "${ANALYZE_LOG}" "${RUN_ROOT}/resolve.pid" "${RUN_ROOT}/analyze.pid"

COMMON_ENV=(
  "DATABASE_URL=sqlite:////${DB_PATH}"
  "TRANSCRIPT_EXPORT_DIR=${TRANSCRIPTS_DIR}"
  "RESOLVE_LLM_PROVIDER=codex_cli"
  "RESOLVE_DIALOGUE_MODE=dialogue"
  "CODEX_RESOLVE_MODEL=gpt-5.4-mini"
  "ANALYZE_PROVIDER=codex_cli"
  "CODEX_ANALYZE_MODEL=gpt-5.4-mini"
  "CODEX_REASONING_EFFORT=medium"
)

nohup env "${COMMON_ENV[@]}" \
  "${SCRIPT_DIR}/run-cli.sh" worker --stage-limit 20 --stages resolve --poll-sec 10 --max-idle-cycles 30 \
  >"${RESOLVE_LOG}" 2>&1 &
echo $! > "${RUN_ROOT}/resolve.pid"

nohup env "${COMMON_ENV[@]}" \
  "${SCRIPT_DIR}/run-cli.sh" worker --stage-limit 20 --stages analyze --poll-sec 10 --max-idle-cycles 30 \
  >"${ANALYZE_LOG}" 2>&1 &
echo $! > "${RUN_ROOT}/analyze.pid"

echo "run_root=${RUN_ROOT}"
echo "db=${DB_PATH}"
echo "manifest=${RUN_ROOT}/selection_manifest.json"
echo "resolve_log=${RESOLVE_LOG}"
echo "analyze_log=${ANALYZE_LOG}"
echo "resolve_pid=$(cat "${RUN_ROOT}/resolve.pid")"
echo "analyze_pid=$(cat "${RUN_ROOT}/analyze.pid")"
