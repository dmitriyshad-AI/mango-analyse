#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PRIMARY_VENV_PYTHON="${SCRIPT_DIR}/venv_stable/bin/python"
FALLBACK_VENV_PYTHON="${SCRIPT_DIR}/venv_stable.broken_20260407/bin/python"

_is_working_ui_python() {
  local py="$1"
  [[ -x "${py}" ]] || return 1
  PYTHONPATH="${PROJECT_ROOT}/src" "${py}" - <<'PY' >/dev/null 2>&1
import importlib
for mod in ("sqlalchemy", "mango_mvp.cli", "mango_mvp.gui"):
    importlib.import_module(mod)
PY
}

if _is_working_ui_python "${PRIMARY_VENV_PYTHON}"; then
  VENV_PYTHON="${PRIMARY_VENV_PYTHON}"
elif _is_working_ui_python "${FALLBACK_VENV_PYTHON}"; then
  VENV_PYTHON="${FALLBACK_VENV_PYTHON}"
else
  echo "Stable runtime is missing."
  echo "Expected one of:"
  echo "  ${PRIMARY_VENV_PYTHON}"
  echo "  ${FALLBACK_VENV_PYTHON}"
  exit 1
fi

export PATH="/Applications/Codex.app/Contents/Resources:${PROJECT_ROOT}/.local/bin:${PATH}"
unset PYTHONPATH || true

exec "${VENV_PYTHON}" -m mango_mvp.gui
