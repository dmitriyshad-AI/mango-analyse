#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APPLY=0
LOCK_DIR="${ROOT}/.codex_local/staging/daily_capture/mail_capture.lock"
MANIFEST="${ROOT}/.codex_local/staging/daily_capture/mail_capture_manifest.json"
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
  printf '{"schema_version":"customer_timeline_mail_capture_driver_v1","status":"locked","lock_dir":"%s","writes_prod":false,"writes_crm":false,"runs_llm":false}\n' "${LOCK_DIR}" | tee "${MANIFEST}"
  exit 75
fi
trap 'rmdir "${LOCK_DIR}" 2>/dev/null || true' EXIT

if [[ "${HOLD_LOCK_SECONDS}" != "0" ]]; then
  sleep "${HOLD_LOCK_SECONDS}"
fi

if [[ "${APPLY}" != "1" ]]; then
  printf '{"schema_version":"customer_timeline_mail_capture_driver_v1","status":"dry_run","apply":false,"command":"scripts/build_customer_timeline_nightly_dv2_sources.py","writes_prod":false,"writes_crm":false,"runs_llm":false}\n' | tee "${MANIFEST}"
  exit 0
fi

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${ROOT}/src" python3 "${ROOT}/scripts/build_customer_timeline_nightly_dv2_sources.py" \
  --out-root "${ROOT}/.codex_local/staging/nightly_dv2_sources" \
  --service-config-out "${ROOT}/.codex_local/staging/nightly_service/customer_timeline_nightly_service_dv2_config.json" \
  > "${MANIFEST}"

