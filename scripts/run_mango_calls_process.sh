#!/bin/zsh
set -euo pipefail

ROOT="$(cd "$(dirname "${0}")/.." && pwd)"
CONFIG="${1:?config path is required}"
ENV_FILE="${2:?env file path is required}"
COMMAND="${3:?process command is required}"

if [[ "${COMMAND}" != "process-a" && "${COMMAND}" != "process-a-worker" && "${COMMAND}" != "process-b" && "${COMMAND}" != "process-b-pull" ]]; then
  print -u2 '{"status":"failed","stop_reason":"unknown_process_command"}'
  exit 2
fi
if [[ ! -f "${CONFIG}" || ! -f "${ENV_FILE}" ]]; then
  print -u2 '{"status":"failed","stop_reason":"config_or_env_missing"}'
  exit 2
fi
CONFIG_META="$(/usr/bin/stat -f '%u:%Lp' "${CONFIG}")"
CONFIG_UID="${CONFIG_META%%:*}"
CONFIG_MODE="${CONFIG_META##*:}"
if [[ "${CONFIG_UID}" != "$(/usr/bin/id -u)" ]] || (( (8#${CONFIG_MODE} & 8#022) != 0 )); then
  print -u2 '{"status":"failed","stop_reason":"config_file_permissions_are_unsafe"}'
  exit 2
fi
if [[ "$(/usr/bin/stat -f '%u:%Lp' "${ENV_FILE}")" != "$(/usr/bin/id -u):600" ]]; then
  print -u2 '{"status":"failed","stop_reason":"env_file_must_be_owner_only_0600"}'
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

PIPELINE_COMMAND="${COMMAND}"
[[ "${COMMAND}" == "process-a-worker" ]] && PIPELINE_COMMAND="process-a"
[[ "${COMMAND}" == "process-b-pull" ]] && PIPELINE_COMMAND="process-b"

verify_split_revision() {
  local expected="${MANGO_CALLS_EXPECTED_CODE_SHA:-}" actual dirty
  if [[ ! "${expected}" =~ '^[0-9a-f]{40}$' ]]; then
    print -u2 '{"status":"failed","stop_reason":"split_code_sha_missing_or_invalid"}'
    return 4
  fi
  actual="$(/usr/bin/git -C "${ROOT}" rev-parse HEAD 2>/dev/null)" || return 4
  dirty="$(/usr/bin/git -C "${ROOT}" status --porcelain --untracked-files=all)" || return 4
  if [[ "${actual}" != "${expected}" || -n "${dirty}" ]]; then
    print -u2 '{"status":"failed","stop_reason":"split_code_revision_mismatch_or_dirty"}'
    return 4
  fi
}

if [[ "${COMMAND}" == "process-a-worker" || "${COMMAND}" == "process-b-pull" ]]; then
  verify_split_revision
fi

publish_daily_report() {
  if [[ -n "${MANGO_CALLS_GOOGLE_DRIVE_FOLDER_ID:-}" || -n "${GOOGLE_APPLICATION_CREDENTIALS:-}" ]]; then
    if [[ -z "${MANGO_CALLS_DAILY_EXPORT_OUT:-}" || -z "${MANGO_CALLS_GOOGLE_DRIVE_FOLDER_ID:-}" || -z "${GOOGLE_APPLICATION_CREDENTIALS:-}" ]]; then
      print -u2 '{"status":"failed","stop_reason":"google_publish_config_incomplete"}'
      return 4
    fi
  fi
  [[ -n "${MANGO_CALLS_DAILY_EXPORT_OUT:-}" ]] || return 0
  local pipeline_root ready_db working_db report_day
  local -a export_args
  pipeline_root="$(/usr/bin/plutil -extract pipeline_root raw -o - "${CONFIG}")"
  ready_db="${pipeline_root}/drop/mango_calls_ready.sqlite"
  working_db="${pipeline_root}/working/mango_calls_pipeline.sqlite"
  [[ "${1:-}" == "sealed" ]] && working_db="${ready_db}"
  report_day="$(TZ=Europe/Moscow /bin/date -v-1d +%F)"
  export_args=(--ready-db "${ready_db}" --working-db "${working_db}" \
    --out "${MANGO_CALLS_DAILY_EXPORT_OUT}" --day "${report_day}")
  [[ "${1:-}" == "sealed" ]] && export_args+=(--sealed-only)
  "${PYTHON_EXECUTABLE}" "${ROOT}/scripts/export_daily_mango_calls_resolve.py" "${export_args[@]}"
  if [[ -n "${MANGO_CALLS_GOOGLE_DRIVE_FOLDER_ID:-}" ]]; then
    "${PYTHON_EXECUTABLE}" "${ROOT}/scripts/publish_daily_mango_calls_google.py" \
      --report-root "${MANGO_CALLS_DAILY_EXPORT_OUT}" --folder-id "${MANGO_CALLS_GOOGLE_DRIVE_FOLDER_ID}" \
      --credentials "${GOOGLE_APPLICATION_CREDENTIALS}" --day "${report_day}" \
      --execute --confirmation UPLOAD_MANGO_DAILY_REPORT
  fi
}

if [[ "${COMMAND}" == "process-b-pull" ]]; then
  if [[ -z "${MANGO_CALLS_REMOTE_HOST:-}" || -z "${MANGO_CALLS_REMOTE_DROP_ROOT:-}" || -z "${MANGO_CALLS_REMOTE_INCOMING_ROOT:-}" ]]; then
    print -u2 '{"status":"failed","stop_reason":"remote_pull_config_incomplete"}'
    exit 4
  fi
  PIPELINE_ROOT="$(/usr/bin/plutil -extract pipeline_root raw -o - "${CONFIG}")"
  "${PYTHON_EXECUTABLE}" "${ROOT}/scripts/pull_mango_calls_drop_remote.py" \
    --remote-host "${MANGO_CALLS_REMOTE_HOST}" --remote-drop-root "${MANGO_CALLS_REMOTE_DROP_ROOT}" \
    --incoming-root "${MANGO_CALLS_REMOTE_INCOMING_ROOT}" --pipeline-root "${PIPELINE_ROOT}" \
    --config "${CONFIG}" --execute --confirmation PULL_MANGO_CALLS_REMOTE_DROP
  exit $?
fi

set +e
OUTPUT="$("${PYTHON_EXECUTABLE}" "${ROOT}/scripts/run_mango_calls_pipeline.py" \
  --config "${CONFIG}" "${PIPELINE_COMMAND}")"
RC=$?
set -e
print -r -- "${OUTPUT}"
if [[ "${PIPELINE_COMMAND}" != "process-a" ]] && (( RC != 0 )); then
  exit "${RC}"
fi

if [[ "${PIPELINE_COMMAND}" == "process-a" ]]; then
  set +e
  PROCESS_A_STATE="$(print -r -- "${OUTPUT}" | "${PYTHON_EXECUTABLE}" -c '
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
print(str(last.get("status") or "") + "|" + ("true" if last.get("downstream_ready") is True else "false"))
')"
  PARSE_RC=$?
  set -e
  if (( PARSE_RC != 0 )); then
    print -u2 '{"status":"failed","stop_reason":"process_a_status_parse_failed"}'
    exit 3
  fi
  PROCESS_A_STATUS="${PROCESS_A_STATE%%|*}"
  DOWNSTREAM_READY="${PROCESS_A_STATE##*|}"
  if [[ "${DOWNSTREAM_READY}" == "true" ]]; then
    if [[ "${COMMAND}" == "process-a-worker" ]]; then
      [[ "${PROCESS_A_STATUS}" == "ok" ]] && publish_daily_report sealed
    else
      /bin/launchctl kickstart "gui/$(/usr/bin/id -u)/com.mango.calls-process-b" || exit $?
    fi
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
    exit 1
  fi
  publish_daily_report
fi
exit "${RC}"
