#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="${PROJECT_ROOT}/stable_runtime/history_remaining_excl_done_20260407"
DB_PATH="${ROOT_DIR}/remaining_history_3762.db"
RUN_ID="resolve4_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${ROOT_DIR}/runs/${RUN_ID}"
STAGE_LIMIT="${STAGE_LIMIT:-10}"
POLL_SEC="${POLL_SEC:-10}"
MAX_IDLE_CYCLES="${MAX_IDLE_CYCLES:-60}"
PRIMARY_VENV_PYTHON="${SCRIPT_DIR}/venv_stable/bin/python"
FALLBACK_VENV_PYTHON="${SCRIPT_DIR}/venv_stable.broken_20260407/bin/python"

if [[ -x "${PRIMARY_VENV_PYTHON}" ]]; then
  VENV_PYTHON="${PRIMARY_VENV_PYTHON}"
elif [[ -x "${FALLBACK_VENV_PYTHON}" ]]; then
  VENV_PYTHON="${FALLBACK_VENV_PYTHON}"
else
  echo "Stable runtime is missing."
  echo "Expected one of:"
  echo "  ${PRIMARY_VENV_PYTHON}"
  echo "  ${FALLBACK_VENV_PYTHON}"
  exit 1
fi

if [[ ! -f "${DB_PATH}" ]]; then
  echo "Missing DB: ${DB_PATH}"
  exit 1
fi

mkdir -p "${RUN_DIR}"

export DATABASE_URL="sqlite:///${DB_PATH}"
export RESOLVE_LLM_PROVIDER="codex_cli"
export CODEX_RESOLVE_MODEL="gpt-5.4-mini"
export PYTHONPATH="${PROJECT_ROOT}/src"
export PATH="/Applications/Codex.app/Contents/Resources:${PROJECT_ROOT}/.local/bin:${PATH}"

pids=()
for idx in 1 2 3 4; do
  log_path="${RUN_DIR}/resolve_${idx}.log"
  nohup "${VENV_PYTHON}" -u -m mango_mvp.cli worker \
    --stage-limit "${STAGE_LIMIT}" \
    --stages resolve \
    --poll-sec "${POLL_SEC}" \
    --max-idle-cycles "${MAX_IDLE_CYCLES}" \
    > "${log_path}" 2>&1 &
  pids+=("$!")
done

{
  echo "run_id=${RUN_ID}"
  echo "db=${DB_PATH}"
  echo "run_dir=${RUN_DIR}"
  echo "stage_limit=${STAGE_LIMIT}"
  echo "poll_sec=${POLL_SEC}"
  echo "max_idle_cycles=${MAX_IDLE_CYCLES}"
  echo "pids=${(j:,:)pids}"
} | tee "${RUN_DIR}/run_info.txt"

printf '%s\n' "${pids[@]}" > "${RUN_DIR}/pids.txt"

echo "run_id=${RUN_ID}"
echo "run_dir=${RUN_DIR}"
echo "db=${DB_PATH}"
echo "pids=${(j:,:)pids}"
