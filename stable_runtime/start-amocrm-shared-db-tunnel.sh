#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
KEY_PATH="${PROJECT_ROOT}/stable_runtime/amocrm_runtime/ssh/id_ed25519_mango_runtime"
REMOTE_HOST="${AMOCRM_SHARED_DB_SSH_HOST:-151.242.88.24}"
REMOTE_USER="${AMOCRM_SHARED_DB_SSH_USER:-root}"
LOCAL_PORT="${AMOCRM_SHARED_DB_LOCAL_PORT:-15432}"
REMOTE_PORT="${AMOCRM_SHARED_DB_REMOTE_PORT:-15432}"
REMOTE_BIND="${AMOCRM_SHARED_DB_REMOTE_BIND:-127.0.0.1}"

if [[ ! -f "${KEY_PATH}" ]]; then
  echo "SSH key not found: ${KEY_PATH}" >&2
  exit 1
fi

exec ssh \
  -o IdentitiesOnly=yes \
  -o ConnectTimeout=10 \
  -o TCPKeepAlive=yes \
  -o ServerAliveInterval=15 \
  -o ServerAliveCountMax=2 \
  -o ExitOnForwardFailure=yes \
  -o StrictHostKeyChecking=accept-new \
  -i "${KEY_PATH}" \
  -N \
  -L "${LOCAL_PORT}:${REMOTE_BIND}:${REMOTE_PORT}" \
  "${REMOTE_USER}@${REMOTE_HOST}"
