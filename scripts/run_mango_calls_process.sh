#!/bin/zsh
set -euo pipefail

ROOT="$(cd "$(dirname "${0}")/.." && pwd)"
CONFIG="${1:?config path is required}"
ENV_FILE="${2:?env file path is required}"
COMMAND="${3:?process command is required}"

if [[ "${COMMAND}" != "process-a" && "${COMMAND}" != "process-a-worker" \
    && "${COMMAND}" != "process-b" && "${COMMAND}" != "process-b-worker" \
    && "${COMMAND}" != "process-b-pull" \
    && "${COMMAND}" != "capture" && "${COMMAND}" != "capture-worker" \
    && "${COMMAND}" != "pipeline" && "${COMMAND}" != "pipeline-worker" \
    && "${COMMAND}" != "watchdog" && "${COMMAND}" != "watchdog-worker" \
    && "${COMMAND}" != "publication-current" \
    && "${COMMAND}" != "publication-close" \
    && "${COMMAND}" != "publication-alert" \
    && "${COMMAND}" != "publication-status" ]]; then
  print -u2 '{"status":"failed","stop_reason":"unknown_process_command"}'
  exit 2
fi
if [[ ! -f "${CONFIG}" || ! -f "${ENV_FILE}" || -L "${CONFIG}" || -L "${ENV_FILE}" ]]; then
  print -u2 '{"status":"failed","stop_reason":"config_or_env_missing"}'
  exit 2
fi
CONFIG_META="$(/usr/bin/stat -f '%u:%Lp' "${CONFIG}")"
CONFIG_UID="${CONFIG_META%%:*}"
CONFIG_MODE="${CONFIG_META##*:}"
if [[ "${CONFIG_UID}" != "$(/usr/bin/id -u)" || "${CONFIG_MODE}" != "600" ]]; then
  print -u2 '{"status":"failed","stop_reason":"config_file_must_be_owner_only_0600"}'
  exit 2
fi
if [[ "$(/usr/bin/stat -f '%u:%Lp' "${ENV_FILE}")" != "$(/usr/bin/id -u):600" ]]; then
  print -u2 '{"status":"failed","stop_reason":"env_file_must_be_owner_only_0600"}'
  exit 2
fi

PYTHON_EXECUTABLE="$(/usr/bin/plutil -extract python_executable raw -o - "${CONFIG}" 2>/dev/null)" || {
  print -u2 '{"status":"failed","stop_reason":"config_missing_python_executable"}'
  exit 2
}
if [[ ! -x "${PYTHON_EXECUTABLE}" ]]; then
  print -u2 '{"status":"failed","stop_reason":"configured_python_missing"}'
  exit 2
fi
ENV_READER_PYTHON="/usr/bin/python3"
[[ -x "${ENV_READER_PYTHON}" ]] || {
  print -u2 '{"status":"failed","stop_reason":"env_reader_python_missing"}'; exit 2;
}
PIPELINE_ROOT="$(/usr/bin/plutil -extract pipeline_root raw -o - "${CONFIG}" 2>/dev/null)" || {
  print -u2 '{"status":"failed","stop_reason":"config_missing_pipeline_root"}'
  exit 2
}
ENV_EXPORTS="$("${ENV_READER_PYTHON}" "${ROOT}/scripts/mango_calls_env.py" --export-lines "${ENV_FILE}" 2>/dev/null)" || {
  print -u2 '{"status":"failed","stop_reason":"worker_env_invalid"}'; exit 2;
}
for inherited_name in ${(k)parameters}; do
  if [[ "${inherited_name}" == MANGO_* || "${inherited_name}" == GOOGLE_APPLICATION_CREDENTIALS ]]; then
    unset "${inherited_name}"
  fi
done
while IFS= read -r item; do
  [[ -n "${item}" ]] && export "${item}"
done <<< "${ENV_EXPORTS}"
if [[ "${COMMAND}" == "process-a-worker" || "${COMMAND}" == "process-b-worker" \
    || "${COMMAND}" == "process-b-pull" \
    || "${COMMAND}" == "capture-worker" || "${COMMAND}" == "pipeline-worker" \
    || "${COMMAND}" == "watchdog-worker" \
    || "${COMMAND}" == publication-* ]]; then
  if [[ -z "${MANGO_CALLS_PIPELINE_ROOT:-}" || "${MANGO_CALLS_PIPELINE_ROOT}" != "${PIPELINE_ROOT}" ]]; then
    print -u2 '{"status":"failed","stop_reason":"pipeline_root_config_env_mismatch"}'
    exit 2
  fi
  "${ENV_READER_PYTHON}" - "${PIPELINE_ROOT}" "${HOME}" <<'PY' >/dev/null 2>&1 || {
import os, sys
from pathlib import Path
raw = os.path.abspath(sys.argv[1])
candidate = os.path.realpath(raw)
owner_raw = os.path.abspath(os.path.join(sys.argv[2], ".mango_local"))
owner_local = os.path.realpath(owner_raw)
assert owner_raw == owner_local
assert raw == candidate and candidate != owner_local
assert os.path.commonpath((candidate, owner_local)) == owner_local
current = Path(candidate)
while True:
    assert not (current / ".git").exists()
    if str(current) == owner_local:
        break
    current = current.parent
PY
    print -u2 '{"status":"failed","stop_reason":"pipeline_root_outside_owner_local_root_or_symlink"}'
    exit 2
  }
fi

PIPELINE_COMMAND="${COMMAND}"
[[ "${COMMAND}" == "process-a-worker" ]] && PIPELINE_COMMAND="process-a"
[[ "${COMMAND}" == "process-b-worker" ]] && PIPELINE_COMMAND="process-b"
[[ "${COMMAND}" == "process-b-pull" ]] && PIPELINE_COMMAND="process-b"
[[ "${COMMAND}" == "capture-worker" ]] && PIPELINE_COMMAND="capture"
[[ "${COMMAND}" == "pipeline-worker" ]] && PIPELINE_COMMAND="pipeline"
[[ "${COMMAND}" == "watchdog-worker" ]] && PIPELINE_COMMAND="watchdog"

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

if [[ "${COMMAND}" == "process-a-worker" || "${COMMAND}" == "process-b-worker" \
    || "${COMMAND}" == "process-b-pull" \
    || "${COMMAND}" == "capture-worker" || "${COMMAND}" == "pipeline-worker" \
    || "${COMMAND}" == "watchdog-worker" \
    || "${COMMAND}" == publication-* ]]; then
  verify_split_revision
fi

if [[ "${COMMAND}" == publication-* ]]; then
  COORDINATOR_COMMAND="${COMMAND#publication-}"
  [[ "${COORDINATOR_COMMAND}" == "current" ]] && COORDINATOR_COMMAND="current-plan"
  [[ "${COORDINATOR_COMMAND}" == "close" ]] && COORDINATOR_COMMAND="daily-close"
  [[ "${COORDINATOR_COMMAND}" == "alert" ]] && COORDINATOR_COMMAND="daily-alert"
  [[ "${COORDINATOR_COMMAND}" == "status" ]] && COORDINATOR_COMMAND="daily-status"
  exec "${PYTHON_EXECUTABLE}" \
    "${ROOT}/scripts/run_mango_calls_publication_coordinator.py" \
    --config "${CONFIG}" "${COORDINATOR_COMMAND}"
fi

if [[ "${COMMAND}" == "process-b-pull" ]]; then
  if [[ -z "${MANGO_CALLS_REMOTE_HOST:-}" || -z "${MANGO_CALLS_REMOTE_DROP_ROOT:-}" || -z "${MANGO_CALLS_REMOTE_INCOMING_ROOT:-}" ]]; then
    print -u2 '{"status":"failed","stop_reason":"remote_pull_config_incomplete"}'
    exit 4
  fi
  [[ -n "${MANGO_CALLS_REMOTE_SSH_KEY:-}" && -n "${MANGO_CALLS_REMOTE_KNOWN_HOSTS:-}" ]] || {
    print -u2 '{"status":"failed","stop_reason":"remote_ssh_files_incomplete"}'; exit 4;
  }
  typeset -a ssh_args
  ssh_args=(--identity-file "${MANGO_CALLS_REMOTE_SSH_KEY}" --known-hosts "${MANGO_CALLS_REMOTE_KNOWN_HOSTS}")
  "${PYTHON_EXECUTABLE}" "${ROOT}/scripts/pull_mango_calls_drop_remote.py" \
    --remote-host "${MANGO_CALLS_REMOTE_HOST}" --remote-drop-root "${MANGO_CALLS_REMOTE_DROP_ROOT}" \
    --incoming-root "${MANGO_CALLS_REMOTE_INCOMING_ROOT}" --pipeline-root "${PIPELINE_ROOT}" \
    --config "${CONFIG}" "${ssh_args[@]}" --execute --confirmation PULL_MANGO_CALLS_REMOTE_DROP
  exit $?
fi

set +e
if [[ "${COMMAND}" == "pipeline-worker" && -x /usr/bin/caffeinate ]]; then
  OUTPUT="$(/usr/bin/caffeinate -dimsu -- "${PYTHON_EXECUTABLE}" \
    "${ROOT}/scripts/run_mango_calls_pipeline.py" --config "${CONFIG}" "${PIPELINE_COMMAND}")"
else
  OUTPUT="$("${PYTHON_EXECUTABLE}" "${ROOT}/scripts/run_mango_calls_pipeline.py" \
    --config "${CONFIG}" "${PIPELINE_COMMAND}")"
fi
RC=$?
set -e
print -r -- "${OUTPUT}"
if [[ "${PIPELINE_COMMAND}" != "process-a" ]] && (( RC != 0 )); then
  exit "${RC}"
fi
if [[ "${COMMAND}" == "pipeline-worker" ]]; then
  "${PYTHON_EXECUTABLE}" \
    "${ROOT}/scripts/run_mango_calls_publication_coordinator.py" \
    --config "${CONFIG}" current-plan
  "${PYTHON_EXECUTABLE}" \
    "${ROOT}/scripts/run_mango_calls_publication_coordinator.py" \
    --config "${CONFIG}" daily-close
  exit $?
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
  if [[ "${DOWNSTREAM_READY}" == "true" && "${COMMAND}" != "process-a-worker" ]]; then
    /bin/launchctl kickstart "gui/$(/usr/bin/id -u)/com.mango.calls-process-b" || exit $?
  fi
fi
exit "${RC}"
