#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WAVE_ROOT="${PROJECT_ROOT}/stable_runtime/messages28_phone_history_asr_20260408"
PRIMARY_VENV_PYTHON="${SCRIPT_DIR}/venv_stable/bin/python"
FALLBACK_VENV_PYTHON="${SCRIPT_DIR}/venv_stable.broken_20260407/bin/python"

if [[ -x "${PRIMARY_VENV_PYTHON}" ]]; then
  VENV_PYTHON="${PRIMARY_VENV_PYTHON}"
elif [[ -x "${FALLBACK_VENV_PYTHON}" ]]; then
  VENV_PYTHON="${FALLBACK_VENV_PYTHON}"
else
  echo "Stable runtime is missing."
  exit 1
fi

mkdir -p "${WAVE_ROOT}/transcripts"

export PATH="/Applications/Codex.app/Contents/Resources:${PROJECT_ROOT}/.local/bin:${PATH}"
export PYTHONPATH="${PROJECT_ROOT}/src"

"${VENV_PYTHON}" "${PROJECT_ROOT}/scripts/requeue_secondary_backfill.py" \
  --db "${WAVE_ROOT}/messages28_phone_history_asr_20260408.db" \
  --provider gigaam \
  --only-exhausted

export MANGO_UI_PROJECT_DIR="${PROJECT_ROOT}"
export MANGO_UI_RECORDINGS_DIR="${WAVE_ROOT}/batch_phone_history"
export MANGO_UI_DATABASE_PATH="${WAVE_ROOT}/messages28_phone_history_asr_20260408.db"
export MANGO_UI_EXPORT_DIR="${WAVE_ROOT}/transcripts"
export MANGO_UI_BACKEND_PYTHON="${VENV_PYTHON}"
export MANGO_UI_USE_PROJECT_SRC="1"
export MANGO_UI_SIMPLE_MODE="0"
export MANGO_UI_STAGE_LIMIT="20"
export MANGO_UI_TRANSCRIBE_MODE="dual"
export MANGO_UI_TRANSCRIBE_PROVIDER="mlx"
export MANGO_UI_SECONDARY_PROVIDER="gigaam"
export MANGO_UI_MERGE_PROVIDER="rule"
export MANGO_UI_MONO_ROLE_ASSIGNMENT_MODE="rule"
export MANGO_UI_PIPELINE_STAGE_TRANSCRIBE="1"
export MANGO_UI_PIPELINE_STAGE_BACKFILL="1"
export MANGO_UI_PIPELINE_STAGE_RESOLVE="0"
export MANGO_UI_PIPELINE_STAGE_ANALYZE="0"
export MANGO_UI_PIPELINE_STAGE_SYNC="0"

exec "${VENV_PYTHON}" -m mango_mvp.gui
