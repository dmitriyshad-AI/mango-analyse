#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TUNNEL_SCRIPT="${SCRIPT_DIR}/start-amocrm-shared-db-tunnel.sh"
LOG_DIR="${PROJECT_ROOT}/stable_runtime/amocrm_runtime/logs"
LOCAL_HOST="${AMOCRM_SHARED_DB_LOCAL_HOST:-127.0.0.1}"
LOCAL_PORT="${AMOCRM_SHARED_DB_LOCAL_PORT:-15432}"
WAIT_SECONDS="${AMOCRM_SHARED_DB_TUNNEL_WAIT_SECONDS:-12}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_PATH="${LOG_DIR}/shared_db_tunnel_${STAMP}.log"
PID_PATH="${LOG_DIR}/shared_db_tunnel_latest.pid"

mkdir -p "${LOG_DIR}"

port_is_open() {
  if command -v nc >/dev/null 2>&1; then
    nc -z "${LOCAL_HOST}" "${LOCAL_PORT}" >/dev/null 2>&1
    return $?
  fi
  (echo >"/dev/tcp/${LOCAL_HOST}/${LOCAL_PORT}") >/dev/null 2>&1
}

if port_is_open; then
  echo "AMO shared DB tunnel is already up on ${LOCAL_HOST}:${LOCAL_PORT}."
  lsof -nP -iTCP:"${LOCAL_PORT}" -sTCP:LISTEN 2>/dev/null || true
  exit 0
fi

if [[ ! -x "${TUNNEL_SCRIPT}" ]]; then
  echo "Tunnel script is missing or not executable: ${TUNNEL_SCRIPT}" >&2
  exit 1
fi

echo "Starting AMO shared DB tunnel on ${LOCAL_HOST}:${LOCAL_PORT}..."
echo "Log: ${LOG_PATH}"

nohup "${TUNNEL_SCRIPT}" >"${LOG_PATH}" 2>&1 &
PID="$!"
echo "${PID}" >"${PID_PATH}"

for _ in $(seq 1 "${WAIT_SECONDS}"); do
  if port_is_open; then
    echo "AMO shared DB tunnel is up on ${LOCAL_HOST}:${LOCAL_PORT}."
    echo "PID: ${PID}"
    echo "PID file: ${PID_PATH}"
    exit 0
  fi
  if ! kill -0 "${PID}" >/dev/null 2>&1; then
    echo "Tunnel process exited before the port opened." >&2
    echo "Log tail:" >&2
    tail -80 "${LOG_PATH}" >&2 || true
    exit 2
  fi
  sleep 1
done

echo "Tunnel process is running, but ${LOCAL_HOST}:${LOCAL_PORT} did not open within ${WAIT_SECONDS}s." >&2
echo "PID: ${PID}" >&2
echo "Log tail:" >&2
tail -80 "${LOG_PATH}" >&2 || true
exit 3
