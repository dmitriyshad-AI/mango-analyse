#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${SCRIPT_DIR}/venv_stable"
LOCK_FILE="${SCRIPT_DIR}/requirements-lock.txt"

PYTHON_BIN="${PYTHON_BIN:-/opt/homebrew/bin/python3.12}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

if [[ "${MANGO_STABLE_SMOKE_ONLY:-0}" == "1" ]]; then
  "${VENV_DIR}/bin/python" -c "import tkinter, mango_mvp.gui, mango_mvp.cli"
  "${SCRIPT_DIR}/run-cli.sh" --help >/dev/null
  echo "Stable snapshot smoke check passed."
  exit 0
fi

if [[ -f "${LOCK_FILE}" ]]; then
  FILTERED_LOCK="$(mktemp -t mango_stable_lock.XXXXXX)"
  grep -Ev "^(mango-call-mvp([[:space:]]|==|@)|-e[[:space:]])" "${LOCK_FILE}" > "${FILTERED_LOCK}"
  "${VENV_DIR}/bin/pip" install -r "${FILTERED_LOCK}"
  rm -f "${FILTERED_LOCK}"
else
  "${VENV_DIR}/bin/pip" install -r "${PROJECT_ROOT}/requirements.txt" \
    -r "${PROJECT_ROOT}/requirements-local-whisper.txt" \
    -r "${PROJECT_ROOT}/requirements-local-dual-asr.txt"
fi

"${VENV_DIR}/bin/pip" install --no-deps --no-build-isolation "${PROJECT_ROOT}"

"${VENV_DIR}/bin/pip" freeze | grep -Ev "^(mango-call-mvp([[:space:]]|==|@)|-e[[:space:]])" > "${LOCK_FILE}"
date "+%Y-%m-%d %H:%M:%S %z" > "${SCRIPT_DIR}/SNAPSHOT_CREATED_AT.txt"

echo "Stable snapshot rebuilt:"
echo "  VENV: ${VENV_DIR}"
echo "  LOCK: ${SCRIPT_DIR}/requirements-lock.txt"
