#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_PYTHON="${SCRIPT_DIR}/venv_stable/bin/python"
WAVE_ROOT="${PROJECT_ROOT}/stable_runtime/top20_core_wave1_20260331"

if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "Stable runtime is missing: ${VENV_PYTHON}"
  echo "Rebuild stable runtime first."
  exit 1
fi

mkdir -p "${WAVE_ROOT}/transcripts"

export PATH="/Applications/Codex.app/Contents/Resources:${PROJECT_ROOT}/.local/bin:${PATH}"
unset PYTHONPATH || true

export MANGO_UI_RECORDINGS_DIR="${WAVE_ROOT}/batch_llm_wave"
export MANGO_UI_DATABASE_PATH="${WAVE_ROOT}/top_20_llm_wave.db"
export MANGO_UI_EXPORT_DIR="${WAVE_ROOT}/transcripts"
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
