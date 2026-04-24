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

export PATH="/Applications/Codex.app/Contents/Resources:${PROJECT_ROOT}/.local/bin:${PATH}"
export LLM_CACHE_ENABLED="${LLM_CACHE_ENABLED:-0}"
unset PYTHONPATH || true

exec "${VENV_PYTHON}" -m mango_mvp.gui
