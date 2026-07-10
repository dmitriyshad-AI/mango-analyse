#!/bin/zsh
set -euo pipefail

REAL_CODEX="${MANGO_CODEX_REAL_BIN:-/opt/homebrew/bin/codex}"

if [[ "${1:-}" == "exec" ]]; then
  shift
  exec "${REAL_CODEX}" exec \
    --disable apps \
    --disable plugins \
    --disable browser_use \
    --disable computer_use \
    --disable in_app_browser \
    "$@"
fi

exec "${REAL_CODEX}" "$@"
