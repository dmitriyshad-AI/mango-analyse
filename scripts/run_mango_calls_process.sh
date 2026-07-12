#!/bin/zsh
set -euo pipefail

ROOT="$(cd "$(dirname "${0}")/.." && pwd)"
CONFIG="${1:?config path is required}"
ENV_FILE="${2:?env file path is required}"
COMMAND="${3:?process command is required}"

if [[ "${COMMAND}" != "process-a" && "${COMMAND}" != "process-b" ]]; then
  print -u2 '{"status":"failed","stop_reason":"unknown_process_command"}'
  exit 2
fi
if [[ ! -f "${CONFIG}" || ! -f "${ENV_FILE}" ]]; then
  print -u2 '{"status":"failed","stop_reason":"config_or_env_missing"}'
  exit 2
fi

set -a
source "${ENV_FILE}"
set +a

PYTHON_EXECUTABLE="$(/usr/bin/python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["python_executable"])' "${CONFIG}")"
if [[ ! -x "${PYTHON_EXECUTABLE}" ]]; then
  print -u2 '{"status":"failed","stop_reason":"configured_python_missing"}'
  exit 2
fi

exec "${PYTHON_EXECUTABLE}" "${ROOT}/scripts/run_mango_calls_pipeline.py" \
  --config "${CONFIG}" "${COMMAND}"
