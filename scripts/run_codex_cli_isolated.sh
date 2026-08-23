#!/bin/zsh
set -euo pipefail

REAL_CODEX="${MANGO_CODEX_REAL_BIN:-/opt/homebrew/bin/codex}"

if [[ "${1:-}" == "exec" ]]; then
  CODEX_HOME_VALUE="${CODEX_HOME:-}"
  PROCESS_HOME="${MANGO_CODEX_PROCESS_HOME:-}"
  PROCESS_TMPDIR="${MANGO_CODEX_PROCESS_TMPDIR:-}"
  PATH_VALUE="${PATH:-/usr/bin:/bin}"
  [[ -n "${CODEX_HOME_VALUE}" && -d "${CODEX_HOME_VALUE}" && ! -L "${CODEX_HOME_VALUE}" ]] || exit 64
  [[ -n "${PROCESS_HOME}" && -d "${PROCESS_HOME}" && ! -L "${PROCESS_HOME}" ]] || exit 64
  [[ -n "${PROCESS_TMPDIR}" && -d "${PROCESS_TMPDIR}" && ! -L "${PROCESS_TMPDIR}" ]] || exit 64
  shift
  typeset -a ephemeral_args
  ephemeral_args=(--ephemeral)
  for arg in "$@"; do
    [[ "${arg}" == "--ephemeral" ]] && ephemeral_args=() && break
  done
  exec /usr/bin/env -i \
    HOME="${PROCESS_HOME}" \
    CODEX_HOME="${CODEX_HOME_VALUE}" \
    PATH="${PATH_VALUE}" \
    TMPDIR="${PROCESS_TMPDIR}" \
    LANG="${LANG:-en_US.UTF-8}" \
    LC_ALL="${LC_ALL:-en_US.UTF-8}" \
    NO_COLOR=1 \
    "${REAL_CODEX}" exec \
    "${ephemeral_args[@]}" \
    --disable apps \
    --disable plugins \
    --disable browser_use \
    --disable computer_use \
    --disable in_app_browser \
    "$@"
fi

exec "${REAL_CODEX}" "$@"
