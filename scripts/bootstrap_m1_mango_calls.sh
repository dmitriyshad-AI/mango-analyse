#!/bin/zsh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODE="${1:-check}"
CONFIG="${MANGO_CALLS_CONFIG:-$HOME/.mango_local/mango_calls_two_processes/config.json}"
ENV_FILE="${MANGO_CALLS_ENV_FILE:-$HOME/.mango_secrets/mango_calls_m1_worker.env}"
VENV="${MANGO_CALLS_VENV:-$HOME/.mango_local/mango_calls_runtime/venv}"
ENV_READER_PYTHON=""

print_plan() {
  cat <<'JSON'
{"schema":"m1_mango_calls_bootstrap_v2","system_packages":["git","ffmpeg","node","python@3.12"],"python_requirements":["requirements.txt","requirements-local-whisper.txt","requirements-local-dual-asr.txt"],"required_access":["Mango API","Codex CLI subscription login"],"report_access":["Tallanto read-only","Tallanto contacts CSV","Yandex Disk local folder"],"optional_publish_access":["private Google Drive folder","Google service account"],"starts_services":false,"runs_asr":false,"runs_resolve_analyze":false}
JSON
}

install_packages() {
  [[ "${CONFIRM_M1_PACKAGE_INSTALL:-}" == "INSTALL_M1_MANGO_CALLS_PACKAGES" ]] || {
    print -u2 'STOP: set CONFIRM_M1_PACKAGE_INSTALL=INSTALL_M1_MANGO_CALLS_PACKAGES'; exit 3;
  }
  [[ "$(uname -m)" == "arm64" ]] || { print -u2 'STOP: M1/M-series arm64 host required'; exit 3; }
  command -v brew >/dev/null || { print -u2 'STOP: install Homebrew from brew.sh first'; exit 3; }
  brew install git ffmpeg node python@3.12
  local python
  python="$(brew --prefix python@3.12)/bin/python3.12"
  "$python" -m venv "$VENV"
  "$VENV/bin/python" -m pip install --upgrade pip setuptools wheel
  "$VENV/bin/python" -m pip install -e "${ROOT}[dev]" \
    -r "$ROOT/requirements-local-whisper.txt" \
    -r "$ROOT/requirements-local-dual-asr.txt"
  npm install -g @openai/codex@0.142.3
  print 'Packages installed. Services were not installed or started.'
}

has_env_key() {
  env_value "$1" "$2" >/dev/null
}

env_value() {
  local file="$1" key="$2"
  [[ -x "$ENV_READER_PYTHON" && -f "$file" ]] || return 1
  "$ENV_READER_PYTHON" "$ROOT/scripts/mango_calls_env.py" "$file" --get "$key" 2>/dev/null
}

owner_only() {
  [[ -f "$1" && ! -L "$1" && "$(/usr/bin/stat -f '%u:%Lp' "$1")" == "$(id -u):600" ]]
}

owner_dir_only() {
  [[ -d "$1" && "$(/usr/bin/stat -f '%u:%Lp' "$1")" == "$(id -u):700" ]]
}

inside_dir() {
  "$ENV_READER_PYTHON" - "$1" "$2" <<'PY' >/dev/null 2>&1
import os, sys
candidate, root = map(os.path.realpath, sys.argv[1:])
assert os.path.commonpath((candidate, root)) == root and candidate != root
PY
}

check_host() {
  local python="" pipeline_root="" codex_binary="" imports=false config_valid=false platform_ok=false
  local mango=false codex_auth=false codex_version=false tallanto=false google=false google_config_valid=false revision=false env_valid=false
  if owner_only "$CONFIG"; then
    python="$(/usr/bin/plutil -extract python_executable raw -o - "$CONFIG" 2>/dev/null || true)"
    pipeline_root="$(/usr/bin/plutil -extract pipeline_root raw -o - "$CONFIG" 2>/dev/null || true)"
    codex_binary="$(/usr/bin/plutil -extract codex_binary raw -o - "$CONFIG" 2>/dev/null || true)"
  fi
  [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]] && platform_ok=true
  if [[ -n "$python" && -x "$python" ]]; then
    ENV_READER_PYTHON="$python"
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$ROOT/src" "$python" - "$CONFIG" <<'PY' >/dev/null 2>&1 && config_valid=true || true
import sys
from pathlib import Path
from mango_mvp.customer_timeline.calls_two_processes import CallsTwoProcessesConfig
assert sys.version_info[:2] == (3, 12)
CallsTwoProcessesConfig.from_json(Path(sys.argv[1]))
PY
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$ROOT/src" "$python" - <<'PY' >/dev/null 2>&1 && imports=true || true
import sqlalchemy, dotenv, mlx_whisper, gigaam, openpyxl, google.auth, mango_mvp.cli
PY
  fi
  if [[ -x "$python" ]] && owner_only "$ENV_FILE" && inside_dir "$ENV_FILE" "$HOME/.mango_secrets" \
      && "$python" "$ROOT/scripts/mango_calls_env.py" "$ENV_FILE" >/dev/null 2>&1; then
    env_valid=true
  fi
  if owner_only "$ENV_FILE" && has_env_key "$ENV_FILE" MANGO_OFFICE_API_KEY \
      && has_env_key "$ENV_FILE" MANGO_OFFICE_API_SALT; then mango=true; fi
  if [[ -n "$codex_binary" && -x "$codex_binary" ]]; then
    [[ "$("$codex_binary" --version 2>/dev/null || true)" == "codex-cli 0.142.3" ]] && codex_version=true
    "$codex_binary" login status >/dev/null 2>&1 && codex_auth=true || true
  fi
  local tallanto_env_path="$(env_value "$ENV_FILE" MANGO_CALLS_TALLANTO_ENV || true)"
  if owner_only "$tallanto_env_path" && inside_dir "$tallanto_env_path" "$HOME/.mango_secrets" \
      && has_env_key "$tallanto_env_path" CRM_TALLANTO_BASE_URL \
      && has_env_key "$tallanto_env_path" CRM_TALLANTO_API_TOKEN; then tallanto=true; fi
  local google_path="$(env_value "$ENV_FILE" GOOGLE_APPLICATION_CREDENTIALS || true)"
  local google_folder_id="$(env_value "$ENV_FILE" MANGO_CALLS_GOOGLE_DRIVE_FOLDER_ID || true)"
  if [[ -z "$google_path" && -z "$google_folder_id" ]]; then
    google_config_valid=true
  elif [[ -x "$python" ]] && owner_only "$google_path" && inside_dir "$google_path" "$HOME/.mango_secrets" \
      && [[ -n "$google_folder_id" ]]; then
    "$python" - "$google_path" <<'PY' >/dev/null 2>&1 && google=true || true
import json, sys
data=json.load(open(sys.argv[1], encoding="utf-8"))
assert data.get("type") == "service_account" and data.get("client_email") and data.get("private_key")
PY
    [[ "$google" == true ]] && google_config_valid=true
  fi
  if owner_only "$ENV_FILE" && has_env_key "$ENV_FILE" MANGO_CALLS_EXPECTED_CODE_SHA; then
    local expected actual dirty
    expected="$(env_value "$ENV_FILE" MANGO_CALLS_EXPECTED_CODE_SHA)"
    actual="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || true)"
    dirty="$(git -C "$ROOT" status --porcelain --untracked-files=all 2>/dev/null || true)"
    [[ "$expected" == "$actual" && -z "$dirty" ]] && revision=true
  fi
  local ffmpeg=false ffprobe=false skills=false yandex=false tallanto_export=false disk_space_ok=false snapshot_as_of=false
  local secrets_dir_owner_only=false pipeline_root_owner_only=false pipeline_root_under_owner_local=false pipeline_root_matches_env=false conflicting_services_loaded=false pipeline_lock_held=false host_preflight_passed=false
  command -v ffmpeg >/dev/null && ffmpeg=true
  command -v ffprobe >/dev/null && ffprobe=true
  if [[ -x "$python" ]]; then
    "$python" - "$ROOT/docs/m1_calls_handoff_20260801/codex_profile_manifest_20260801.json" "$HOME/.codex/skills" <<'PY' >/dev/null 2>&1 && skills=true || true
import json, sys
from pathlib import Path
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
root = Path(sys.argv[2])
assert all((root / name / "SKILL.md").is_file() for name in manifest["local_skills"])
PY
  fi
  local tallanto_export_path="$(env_value "$ENV_FILE" MANGO_CALLS_TALLANTO_EXPORT || true)"
  local yandex_path="$(env_value "$ENV_FILE" MANGO_CALLS_DAILY_EXPORT_OUT || true)"
  if [[ -x "$python" && -n "$tallanto_export_path" ]] && owner_only "$tallanto_export_path" && [[ -s "$tallanto_export_path" ]]; then
    "$python" - "$tallanto_export_path" <<'PY' >/dev/null 2>&1 && tallanto_export=true || true
import csv, sys
with open(sys.argv[1], encoding="utf-8-sig", newline="") as stream:
    headers = set(next(csv.reader(stream)))
assert {"ID", "Имя", "Фамилия", "ФИО родителя", "Тел. (родителя)", "Тел. (доп.)"} <= headers
PY
  fi
  local snapshot_value="$(env_value "$ENV_FILE" MANGO_CALLS_TALLANTO_SNAPSHOT_AS_OF || true)"
  if [[ -x "$python" && -n "$snapshot_value" ]]; then
    "$python" - "$snapshot_value" <<'PY' >/dev/null 2>&1 && snapshot_as_of=true || true
from datetime import datetime
import sys
value = datetime.fromisoformat(sys.argv[1])
assert value.tzinfo is not None and value <= datetime.now(value.tzinfo)
PY
  fi
  if [[ -d "$yandex_path" && ! -L "$yandex_path" && -w "$yandex_path" \
      && -f "$yandex_path/.mango_calls_yandex_target" \
      && "$(<"$yandex_path/.mango_calls_yandex_target")" == "mango-calls-yandex-v1" ]]; then
    yandex=true
  fi
  local available_kib=""
  [[ -d "$pipeline_root" ]] && available_kib="$(df -Pk "$pipeline_root" 2>/dev/null | /usr/bin/awk 'NR==2 {print $4}')"
  [[ "$available_kib" == <-> && "$available_kib" -ge 41943040 ]] && disk_space_ok=true
  owner_dir_only "$HOME/.mango_secrets" && secrets_dir_owner_only=true
  [[ -n "$pipeline_root" ]] && owner_dir_only "$pipeline_root" && pipeline_root_owner_only=true
  if [[ -x "$python" && -n "$pipeline_root" ]]; then
    "$python" - "$pipeline_root" "$HOME" <<'PY' >/dev/null 2>&1 && pipeline_root_under_owner_local=true || true
import os, sys
from pathlib import Path
raw = os.path.abspath(sys.argv[1]); candidate = os.path.realpath(raw)
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
  fi
  [[ "$(env_value "$ENV_FILE" MANGO_CALLS_PIPELINE_ROOT || true)" == "$pipeline_root" ]] && pipeline_root_matches_env=true
  for label in com.mango.calls-process-a com.mango.calls-process-b com.mango.calls-two-processes; do
    /bin/launchctl print "gui/$(id -u)/$label" >/dev/null 2>&1 && conflicting_services_loaded=true || true
  done
  if [[ -x "$python" && -d "$pipeline_root/locks" ]]; then
    "$python" - "$pipeline_root/locks" <<'PY' >/dev/null 2>&1 || pipeline_lock_held=true
import fcntl, sys
from pathlib import Path
for name in ("process_a.lock", "process_b.lock"):
    path = Path(sys.argv[1]) / name
    if not path.is_file():
        continue
    with path.open("rb") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
PY
  fi
  if [[ "$platform_ok" == true && "$config_valid" == true && "$env_valid" == true && "$imports" == true && "$ffmpeg" == true \
      && "$ffprobe" == true && "$mango" == true && "$codex_version" == true && "$codex_auth" == true \
      && "$tallanto" == true && "$tallanto_export" == true && "$snapshot_as_of" == true \
      && "$google_config_valid" == true && "$yandex" == true && "$secrets_dir_owner_only" == true \
      && "$pipeline_root_owner_only" == true && "$pipeline_root_under_owner_local" == true && "$pipeline_root_matches_env" == true \
      && "$disk_space_ok" == true && "$revision" == true && "$conflicting_services_loaded" == false \
      && "$pipeline_lock_held" == false ]]; then host_preflight_passed=true; fi
  printf '{"schema":"m1_mango_calls_host_check_v2","platform_ok":%s,"config_owner_only_and_valid":%s,"worker_env_owner_only_and_valid":%s,"python_3_12_imports":%s,"ffmpeg_present":%s,"ffprobe_present":%s,"mango_credentials_present":%s,"configured_codex_0_142_3":%s,"codex_login_present":%s,"tallanto_credentials_present":%s,"tallanto_export_owner_only":%s,"tallanto_snapshot_as_of_present":%s,"google_publish_enabled":%s,"google_config_valid":%s,"yandex_target_verified":%s,"developer_profile_ready":%s,"secrets_dir_owner_only":%s,"pipeline_root_owner_only":%s,"pipeline_root_under_owner_local":%s,"pipeline_root_matches_env":%s,"disk_space_ok":%s,"clean_expected_revision":%s,"conflicting_services_loaded":%s,"pipeline_lock_held":%s,"network_access_verified":false,"host_preflight_passed":%s,"runtime_ready":false,"services_started_by_this_script":false}\n' \
    "$platform_ok" "$config_valid" "$env_valid" \
    "$imports" "$ffmpeg" "$ffprobe" "$mango" "$codex_version" "$codex_auth" "$tallanto" "$tallanto_export" "$snapshot_as_of" "$google" "$google_config_valid" "$yandex" "$skills" \
    "$secrets_dir_owner_only" "$pipeline_root_owner_only" "$pipeline_root_under_owner_local" "$pipeline_root_matches_env" "$disk_space_ok" "$revision" "$conflicting_services_loaded" "$pipeline_lock_held" "$host_preflight_passed"
  [[ "$host_preflight_passed" == true ]]
}

case "$MODE" in
  plan) print_plan ;;
  install) install_packages ;;
  check) check_host ;;
  *) print -u2 'usage: bootstrap_m1_mango_calls.sh [plan|install|check]'; exit 2 ;;
esac
