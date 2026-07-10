#!/bin/zsh
set -euo pipefail

ROOT="$(cd "$(dirname "${0}")/.." && pwd)"
CONFIG="${1:?config path is required}"
ENV_FILE="${2:?env file path is required}"

if [[ ! -f "${CONFIG}" || ! -f "${ENV_FILE}" ]]; then
  print -u2 '{"status":"failed","stop_reason":"config_or_env_missing"}'
  exit 2
fi

set -a
source "${ENV_FILE}"
set +a

exec /usr/bin/python3 "${ROOT}/scripts/run_mango_calls_pipeline.py" \
  --config "${CONFIG}" cycle
