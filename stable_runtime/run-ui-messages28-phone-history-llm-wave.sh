#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WAVE_ROOT="${PROJECT_ROOT}/stable_runtime/messages28_phone_history_llm_wave_20260409"
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

mkdir -p "${WAVE_ROOT}/transcripts"

export PATH="/Applications/Codex.app/Contents/Resources:${PROJECT_ROOT}/.local/bin:${PATH}"
export PYTHONPATH="${PROJECT_ROOT}/src"

export MANGO_UI_PROJECT_DIR="${PROJECT_ROOT}"
export MANGO_UI_RECORDINGS_DIR="${WAVE_ROOT}/batch_llm_wave"
export MANGO_UI_DATABASE_PATH="${WAVE_ROOT}/messages28_phone_history_llm_wave_20260409.db"
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
export MANGO_UI_RESOLVE_LLM_PROVIDER="codex_cli"
export MANGO_UI_RESOLVE_DIALOGUE_MODE="dialogue"
export MANGO_UI_ANALYZE_PROVIDER="codex_cli"
export MANGO_UI_CODEX_RESOLVE_MODEL="gpt-5.4-mini"
export MANGO_UI_CODEX_ANALYZE_MODEL="gpt-5.4-mini"
export MANGO_UI_PIPELINE_STAGE_TRANSCRIBE="0"
export MANGO_UI_PIPELINE_STAGE_BACKFILL="0"
export MANGO_UI_PIPELINE_STAGE_RESOLVE="1"
export MANGO_UI_PIPELINE_STAGE_ANALYZE="1"
export MANGO_UI_PIPELINE_STAGE_SYNC="0"

exec "${VENV_PYTHON}" -m mango_mvp.gui
