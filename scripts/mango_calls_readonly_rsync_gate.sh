#!/bin/zsh
set -euo pipefail

ALLOWED_ROOT="${1:?allowed drop root is required}"
VALIDATE_ONLY="${2:-}"
[[ "${ALLOWED_ROOT}" == /* && "${ALLOWED_ROOT}" != *..* ]] || exit 126
[[ -n "${SSH_ORIGINAL_COMMAND:-}" ]] || exit 126

typeset -a words
words=("${(z)SSH_ORIGINAL_COMMAND}")
typeset -a expected
expected=(rsync --server --sender -g -l -o -p -D -r -t --dirs .)
(( ${#words} == 13 )) || exit 126
[[ "${(j: :)words[1,12]}" == "${(j: :)expected}" ]] || exit 126
[[ "${words[13]}" == "${ALLOWED_ROOT}/mango_calls_ready.sqlite" \
   || "${words[13]}" == "${ALLOWED_ROOT}/mango_calls_ready.manifest.json" ]] || exit 126
[[ -f "${words[13]}" && ! -L "${words[13]}" ]] || exit 126

[[ "${VALIDATE_ONLY}" == "--validate-only" ]] && { print -- ok; exit 0; }
words[1]="/usr/bin/rsync"
exec "${words[@]}"
