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

if [[ ! "${INTERVAL_SEC}" =~ ^[0-9]+$ ]] || [[ "${INTERVAL_SEC}" -lt 10 ]]; then
  echo "Interval must be integer >= 10 sec"
  exit 2
fi

if [[ -f "${PID_FILE}" ]]; then
  OLD_PID="$(cat "${PID_FILE}" 2>/dev/null || true)"
  if [[ -n "${OLD_PID}" ]] && kill -0 "${OLD_PID}" 2>/dev/null; then
    echo "Already running with PID ${OLD_PID}"
    exit 0
  fi
fi

echo "$$" > "${PID_FILE}"
touch "${LOG_FILE}"

cleanup() {
  rm -f "${PID_FILE}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[$(date '+%Y-%m-%d %H:%M:%S %z')] started (interval=${INTERVAL_SEC}s branch=${BRANCH})" >> "${LOG_FILE}"

while true; do
  HAS_STAGED=0
  HAS_UNSTAGED=0
  HAS_UNTRACKED=0

  if ! git diff --cached --quiet; then
    HAS_STAGED=1
  fi
  if ! git diff --quiet; then
    HAS_UNSTAGED=1
  fi
  if [[ -n "$(git ls-files --others --exclude-standard)" ]]; then
    HAS_UNTRACKED=1
  fi

  if [[ "${HAS_STAGED}" -eq 1 || "${HAS_UNSTAGED}" -eq 1 || "${HAS_UNTRACKED}" -eq 1 ]]; then
    TS="$(date '+%Y-%m-%d %H:%M:%S %z')"
    MSG="auto: checkpoint ${TS}"
    {
      echo "[$TS] changes detected -> commit+push"
      git add -A
      if git commit -m "${MSG}"; then
        if git push origin "${BRANCH}"; then
          echo "[$TS] pushed to origin/${BRANCH}"
        else
          echo "[$TS] push failed (will retry on next cycle)"
        fi
      else
        echo "[$TS] commit skipped"
      fi
    } >> "${LOG_FILE}" 2>&1 || true
  fi

  sleep "${INTERVAL_SEC}"
done
