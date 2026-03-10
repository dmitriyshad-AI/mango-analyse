#!/usr/bin/env bash
set -euo pipefail

PID_FILE=".git/autocommit_push.pid"

if [[ ! -f "${PID_FILE}" ]]; then
  echo "Auto commit/push is not running (no pid file)."
  exit 0
fi

PID="$(cat "${PID_FILE}" 2>/dev/null || true)"
if [[ -z "${PID}" ]]; then
  rm -f "${PID_FILE}" 2>/dev/null || true
  echo "Stale pid file removed."
  exit 0
fi

if kill -0 "${PID}" 2>/dev/null; then
  kill "${PID}" 2>/dev/null || true
  sleep 1
fi

if kill -0 "${PID}" 2>/dev/null; then
  kill -9 "${PID}" 2>/dev/null || true
fi

rm -f "${PID_FILE}" 2>/dev/null || true
echo "Auto commit/push stopped."
