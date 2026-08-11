#!/bin/zsh
set -euo pipefail

REAL_CODEX="${MANGO_CODEX_REAL_BIN:-/opt/homebrew/bin/codex}"

if [[ "${1:-}" == "exec" ]]; then
  shift
  typeset -a ephemeral_args
  ephemeral_args=(--ephemeral)
  for arg in "$@"; do
    [[ "${arg}" == "--ephemeral" ]] && ephemeral_args=() && break
  done
  exec "${REAL_CODEX}" exec \
    "${ephemeral_args[@]}" \
    --disable apps \
    --disable plugins \
    --disable browser_use \
    --disable computer_use \
    --disable in_app_browser \
    "$@"
fi

exec "${REAL_CODEX}" "$@"
