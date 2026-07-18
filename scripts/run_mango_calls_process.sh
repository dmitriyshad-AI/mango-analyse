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

PYTHON_EXECUTABLE="$(/usr/bin/plutil -extract python_executable raw -o - "${CONFIG}")"
if [[ ! -x "${PYTHON_EXECUTABLE}" ]]; then
  print -u2 '{"status":"failed","stop_reason":"configured_python_missing"}'
  exit 2
fi

set +e
OUTPUT="$("${PYTHON_EXECUTABLE}" "${ROOT}/scripts/run_mango_calls_pipeline.py" \
  --config "${CONFIG}" "${COMMAND}")"
RC=$?
set -e
print -r -- "${OUTPUT}"
if (( RC != 0 )); then
  exit "${RC}"
fi

if [[ "${COMMAND}" == "process-a" ]]; then
  set +e
  STATUS="$(print -r -- "${OUTPUT}" | "${PYTHON_EXECUTABLE}" -c '
import json, sys
text = sys.stdin.read()
decoder = json.JSONDecoder()
last = None
for index, char in enumerate(text):
    if char != "{":
        continue
    try:
        value, end = decoder.raw_decode(text[index:])
    except json.JSONDecodeError:
        continue
    if isinstance(value, dict) and not text[index + end:].strip():
        last = value
if last is None:
    raise SystemExit(2)
print(last.get("status", ""))
')"
  PARSE_RC=$?
  set -e
  if (( PARSE_RC != 0 )); then
    print -u2 '{"status":"failed","stop_reason":"process_a_status_parse_failed"}'
    exit 3
  fi
  if [[ "${STATUS}" == "ok" ]]; then
    exec /bin/launchctl kickstart "gui/$(/usr/bin/id -u)/com.mango.calls-process-b"
  fi
fi
