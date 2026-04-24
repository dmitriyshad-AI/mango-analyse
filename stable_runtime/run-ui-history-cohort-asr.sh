#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_PYTHON="${SCRIPT_DIR}/venv_stable/bin/python"
COHORT_ROOT="${PROJECT_ROOT}/stable_runtime/history_cohort_20260319_20260326"

if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "Stable runtime is missing: ${VENV_PYTHON}"
  echo "Rebuild stable runtime first."
  exit 1
fi

export PATH="/Applications/Codex.app/Contents/Resources:${PROJECT_ROOT}/.local/bin:${PATH}"
unset PYTHONPATH || true

export MANGO_UI_RECORDINGS_DIR="${COHORT_ROOT}/batch_full_history"
export MANGO_UI_DATABASE_PATH="${COHORT_ROOT}/history_cohort_20260319_20260326_asr.db"
export MANGO_UI_EXPORT_DIR="${COHORT_ROOT}/history_cohort_20260319_20260326_transcripts"
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
