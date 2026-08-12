#!/bin/zsh
set -euo pipefail
umask 077

ROOT="$(cd "$(dirname "${0}")/.." && pwd)"
CONFIG="${1:?config path is required}"
ENV_FILE="${2:?env file path is required}"
COMMAND="${3:?process command is required}"

if [[ "${COMMAND}" != "process-a" && "${COMMAND}" != "process-a-worker" \
    && "${COMMAND}" != "process-b" && "${COMMAND}" != "process-b-worker" \
    && "${COMMAND}" != "process-b-pull" \
    && "${COMMAND}" != "capture" && "${COMMAND}" != "capture-worker" \
    && "${COMMAND}" != "pipeline" && "${COMMAND}" != "pipeline-worker" \
    && "${COMMAND}" != "controlled-one" && "${COMMAND}" != "controlled-one-worker" \
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
ENV_READER_PYTHON="/usr/bin/python3"
[[ -x "${ENV_READER_PYTHON}" ]] || {
  print -u2 '{"status":"failed","stop_reason":"env_reader_python_missing"}'; exit 2;
}
RUNTIME_CONFIG_FIELDS="$(MANGO_CALLS_EXPECTED_CONFIG_SHA256= \
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${ROOT}/src" \
  "${ENV_READER_PYTHON}" -m mango_mvp.productization.mango_calls_config \
  "${CONFIG}" 2>/dev/null)" || {
  print -u2 '{"status":"failed","stop_reason":"config_file_must_be_owner_only_0600_or_invalid"}'
  exit 2
}
typeset -a RUNTIME_CONFIG_LINES
RUNTIME_CONFIG_LINES=("${(@f)RUNTIME_CONFIG_FIELDS}")
if (( ${#RUNTIME_CONFIG_LINES[@]} != 5 )); then
  print -u2 '{"status":"failed","stop_reason":"config_file_must_be_owner_only_0600_or_invalid"}'
  exit 2
fi
PYTHON_EXECUTABLE="${RUNTIME_CONFIG_LINES[1]}"
PIPELINE_ROOT="${RUNTIME_CONFIG_LINES[2]}"
STRICT_RUNTIME="${RUNTIME_CONFIG_LINES[3]}"
CONFIG_SHA256="${RUNTIME_CONFIG_LINES[5]}"
if [[ ! "${CONFIG_SHA256}" =~ '^[0-9a-f]{64}$' ]]; then
  print -u2 '{"status":"failed","stop_reason":"config_snapshot_sha_invalid"}'
  exit 2
fi
if [[ ! -x "${PYTHON_EXECUTABLE}" ]]; then
  print -u2 '{"status":"failed","stop_reason":"configured_python_missing"}'
  exit 2
fi
if [[ "${STRICT_RUNTIME}" == "true" ]] \
    && [[ "${COMMAND}" == "process-a" || "${COMMAND}" == "process-b" \
        || "${COMMAND}" == "capture" || "${COMMAND}" == "pipeline" \
        || "${COMMAND}" == "controlled-one" || "${COMMAND}" == "watchdog" ]]; then
  print -u2 '{"status":"failed","stop_reason":"strict_runtime_requires_guarded_worker_command"}'
  exit 2
fi
set +e
ENV_EXPORTS="$("${ENV_READER_PYTHON}" "${ROOT}/scripts/mango_calls_env.py" \
  --export-lines "${ENV_FILE}" 2>/dev/null)"
ENV_PARSE_RC=$?
set -e
if (( ENV_PARSE_RC == 3 )); then
  print -u2 '{"status":"failed","stop_reason":"env_file_must_be_owner_only_0600"}'
  exit 2
fi
if (( ENV_PARSE_RC != 0 )); then
  print -u2 '{"status":"failed","stop_reason":"worker_env_invalid"}'
  exit 2
fi
for inherited_name in ${(k)parameters}; do
  if [[ "${inherited_name}" == MANGO_* || "${inherited_name}" == GOOGLE_APPLICATION_CREDENTIALS ]]; then
    unset "${inherited_name}"
  fi
done
while IFS= read -r item; do
  [[ -n "${item}" ]] && export "${item}"
done <<< "${ENV_EXPORTS}"
export MANGO_CALLS_EXPECTED_CONFIG_SHA256="${CONFIG_SHA256}"
if [[ "${COMMAND}" == "process-a-worker" || "${COMMAND}" == "process-b-worker" \
    || "${COMMAND}" == "process-b-pull" \
    || "${COMMAND}" == "capture-worker" || "${COMMAND}" == "pipeline-worker" \
    || "${COMMAND}" == "controlled-one-worker" \
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
[[ "${COMMAND}" == "controlled-one-worker" ]] && PIPELINE_COMMAND="controlled-one"
[[ "${COMMAND}" == "watchdog-worker" ]] && PIPELINE_COMMAND="watchdog"

verify_split_revision() {
  local expected="${MANGO_CALLS_EXPECTED_CODE_SHA:-}" actual dirty top unsafe_index
  typeset -a safe_git
  safe_git=(/usr/bin/env -i HOME="${HOME}" PATH="/usr/bin:/bin" \
    /usr/bin/git -c core.fsmonitor=false -c core.untrackedCache=false -C "${ROOT}")
  if [[ ! "${expected}" =~ '^[0-9a-f]{40}$' ]]; then
    print -u2 '{"status":"failed","stop_reason":"split_code_sha_missing_or_invalid"}'
    return 4
  fi
  top="$("${safe_git[@]}" rev-parse --show-toplevel 2>/dev/null)" || return 4
  [[ "${top:A}" == "${ROOT:A}" ]] || return 4
  actual="$("${safe_git[@]}" rev-parse HEAD 2>/dev/null)" || return 4
  unsafe_index="$("${safe_git[@]}" ls-files -v | /usr/bin/awk \
    'substr($0,1,2) != "H " { print; exit }')" || return 4
  dirty="$("${safe_git[@]}" status --porcelain=v1 --untracked-files=all)" || return 4
  "${safe_git[@]}" diff-files --quiet --ignore-submodules=none -- || return 4
  "${safe_git[@]}" diff-index --cached --quiet --ignore-submodules=none HEAD -- || return 4
  if [[ "${actual}" != "${expected}" || -n "${dirty}" || -n "${unsafe_index}" ]]; then
    print -u2 '{"status":"failed","stop_reason":"split_code_revision_mismatch_or_dirty"}'
    return 4
  fi
}

if [[ "${COMMAND}" == "process-a-worker" || "${COMMAND}" == "process-b-worker" \
    || "${COMMAND}" == "process-b-pull" \
    || "${COMMAND}" == "capture-worker" || "${COMMAND}" == "pipeline-worker" \
    || "${COMMAND}" == "controlled-one-worker" \
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
if [[ ("${COMMAND}" == "pipeline-worker" || "${COMMAND}" == "controlled-one-worker") \
    && -x /usr/bin/caffeinate ]]; then
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
  set +e
  CURRENT_OUTPUT="$("${PYTHON_EXECUTABLE}" \
    "${ROOT}/scripts/run_mango_calls_publication_coordinator.py" \
    --config "${CONFIG}" current-plan)"
  CURRENT_RC=$?
  print -r -- "${CURRENT_OUTPUT}"
  DAILY_OUTPUT="$("${PYTHON_EXECUTABLE}" \
    "${ROOT}/scripts/run_mango_calls_publication_coordinator.py" \
    --config "${CONFIG}" daily-close)"
  DAILY_RC=$?
  print -r -- "${DAILY_OUTPUT}"
  set -e
  if (( CURRENT_RC != 0 )); then
    exit "${CURRENT_RC}"
  fi
  exit "${DAILY_RC}"
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
