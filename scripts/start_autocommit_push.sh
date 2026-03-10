#!/usr/bin/env bash
set -euo pipefail

INTERVAL_SEC="${1:-120}"
BRANCH="${2:-main}"
PID_FILE=".git/autocommit_push.pid"
LOG_FILE="${AUTOCOMMIT_LOG_FILE:-.git/autocommit_push.log}"

if [[ ! -d ".git" ]]; then
  echo "Not a git repository: $(pwd)"
  exit 2
fi

if [[ -f "${PID_FILE}" ]]; then
  PID="$(cat "${PID_FILE}" 2>/dev/null || true)"
  if [[ -n "${PID}" ]] && kill -0 "${PID}" 2>/dev/null; then
    echo "Already running with PID ${PID}"
    echo "Log: ${LOG_FILE}"
    exit 0
  fi
fi

nohup bash scripts/autocommit_push_loop.sh "${INTERVAL_SEC}" "${BRANCH}" >/dev/null 2>&1 &
sleep 1

if [[ -f "${PID_FILE}" ]]; then
  PID="$(cat "${PID_FILE}")"
  if kill -0 "${PID}" 2>/dev/null; then
    echo "Auto commit/push started"
    echo "  pid: ${PID}"
    echo "  interval: ${INTERVAL_SEC}s"
    echo "  branch: ${BRANCH}"
    echo "  log: ${LOG_FILE}"
    exit 0
  fi
fi

echo "Failed to start auto commit/push. Check shell and repo state."
exit 1
