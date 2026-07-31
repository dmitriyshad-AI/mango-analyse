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
if [[ "${COMMAND}" != "process-a" ]] && (( RC != 0 )); then
  exit "${RC}"
fi

if [[ "${COMMAND}" == "process-a" ]]; then
  set +e
  DOWNSTREAM_READY="$(print -r -- "${OUTPUT}" | "${PYTHON_EXECUTABLE}" -c '
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
print("true" if last.get("downstream_ready") is True else "false")
')"
  PARSE_RC=$?
  set -e
  if (( PARSE_RC != 0 )); then
    print -u2 '{"status":"failed","stop_reason":"process_a_status_parse_failed"}'
    exit 3
  fi
  if [[ "${DOWNSTREAM_READY}" == "true" ]]; then
    /bin/launchctl kickstart "gui/$(/usr/bin/id -u)/com.mango.calls-process-b" || exit $?
  fi
fi
if [[ "${COMMAND}" == "process-b" && -n "${MANGO_CALLS_DAILY_EXPORT_OUT:-}" ]]; then
  PROCESS_B_STATE="$(print -r -- "${OUTPUT}" | "${PYTHON_EXECUTABLE}" -c '
import json, sys
text = sys.stdin.read(); decoder = json.JSONDecoder(); last = None
for index, char in enumerate(text):
    if char != "{": continue
    try: value, end = decoder.raw_decode(text[index:])
    except json.JSONDecodeError: continue
    if isinstance(value, dict) and not text[index + end:].strip(): last = value
if last is None: raise SystemExit(2)
print(str(last.get("status") or "") + "|" + str(last.get("stop_reason") or ""))
')" || exit 3
  if [[ "${PROCESS_B_STATE}" != "ok|" && "${PROCESS_B_STATE}" != "idle|drop_unchanged" ]]; then
    exit "${RC}"
  fi
  PIPELINE_ROOT="$(/usr/bin/plutil -extract pipeline_root raw -o - "${CONFIG}")"
  "${PYTHON_EXECUTABLE}" "${ROOT}/scripts/export_daily_mango_calls_resolve.py" \
    --ready-db "${PIPELINE_ROOT}/drop/mango_calls_ready.sqlite" \
    --working-db "${PIPELINE_ROOT}/working/mango_calls_pipeline.sqlite" \
    --out "${MANGO_CALLS_DAILY_EXPORT_OUT}"
fi
exit "${RC}"
