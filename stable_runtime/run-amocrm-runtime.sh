#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNTIME_DIR="${PROJECT_ROOT}/stable_runtime/amocrm_runtime"
PRIVATE_ENV_FILE="${RUNTIME_DIR}/.env.private"
HANDOFF_ENV_FILE="${PROJECT_ROOT}/prod_runtime_transfer/.env.private"

mkdir -p "${RUNTIME_DIR}"

if [[ -f "${PRIVATE_ENV_FILE}" ]]; then
  set -a
  source "${PRIVATE_ENV_FILE}"
  set +a
elif [[ -f "${HANDOFF_ENV_FILE}" ]]; then
  set -a
  source "${HANDOFF_ENV_FILE}"
  set +a
fi

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export AI_OFFICE_API_KEY="${AI_OFFICE_API_KEY:-mango-amocrm-runtime-local}"
export DATABASE_URL="${DATABASE_URL:-sqlite:///${RUNTIME_DIR}/amo_runtime.db}"
export CRM_AMO_MODE="${CRM_AMO_MODE:-http}"
export CRM_AMO_BASE_URL="${CRM_AMO_BASE_URL:-${AMOCRM_BASE_URL:-}}"
export CRM_AMO_API_TOKEN="${CRM_AMO_API_TOKEN:-${AMOCRM_ACCESS_TOKEN:-}}"
export CRM_AMO_OAUTH_ACCOUNT_BASE_URL="${CRM_AMO_OAUTH_ACCOUNT_BASE_URL:-${CRM_AMO_BASE_URL:-}}"
export API_HOST="${API_HOST:-0.0.0.0}"
export API_PORT="${API_PORT:-8010}"

if [[ -z "${CRM_AMO_BASE_URL:-}" ]]; then
  echo "CRM_AMO_BASE_URL is not configured. Put it in ${PRIVATE_ENV_FILE} or prod_runtime_transfer/.env.private." >&2
  exit 1
fi

if [[ "${DATABASE_URL}" == sqlite:* ]]; then
  mkdir -p "${RUNTIME_DIR}"
fi

exec python3 -m uvicorn mango_mvp.amocrm_runtime.main:app --host "${API_HOST}" --port "${API_PORT}"
