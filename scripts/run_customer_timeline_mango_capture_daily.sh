#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APPLY=0
LOCK_DIR="${ROOT}/.codex_local/staging/daily_capture/mango_capture.lock"
MANIFEST="${ROOT}/.codex_local/staging/daily_capture/mango_capture_manifest.json"
COMMAND_FILE="${MANGO_CAPTURE_COMMAND_FILE:-}"
HOLD_LOCK_SECONDS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --apply)
      APPLY=1
      shift
      ;;
    --lock-dir)
      LOCK_DIR="$2"
      shift 2
      ;;
    --manifest)
      MANIFEST="$2"
      shift 2
      ;;
    --command-file)
      COMMAND_FILE="$2"
      shift 2
      ;;
    --hold-lock-seconds)
      HOLD_LOCK_SECONDS="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

mkdir -p "$(dirname "${LOCK_DIR}")" "$(dirname "${MANIFEST}")"
if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  printf '{"schema_version":"customer_timeline_mango_capture_driver_v1","status":"locked","lock_dir":"%s","writes_prod":false,"writes_crm":false,"runs_llm":false,"runs_asr":false}\n' "${LOCK_DIR}" | tee "${MANIFEST}"
  exit 75
fi
trap 'rmdir "${LOCK_DIR}" 2>/dev/null || true' EXIT

if [[ "${HOLD_LOCK_SECONDS}" != "0" ]]; then
  sleep "${HOLD_LOCK_SECONDS}"
fi

if [[ "${APPLY}" != "1" ]]; then
  printf '{"schema_version":"customer_timeline_mango_capture_driver_v1","status":"dry_run","apply":false,"command_file":"%s","writes_prod":false,"writes_crm":false,"runs_llm":false,"runs_asr":false}\n' "${COMMAND_FILE}" | tee "${MANIFEST}"
  exit 0
fi

if [[ -z "${COMMAND_FILE}" || ! -f "${COMMAND_FILE}" ]]; then
  printf '{"schema_version":"customer_timeline_mango_capture_driver_v1","status":"not_configured","reason":"MANGO_CAPTURE_COMMAND_FILE is required for apply","writes_prod":false,"writes_crm":false,"runs_llm":false,"runs_asr":false}\n' | tee "${MANIFEST}"
  exit 78
fi

bash "${COMMAND_FILE}" > "${MANIFEST}"
