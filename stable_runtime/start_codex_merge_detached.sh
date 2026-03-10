#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNNER="${SCRIPT_DIR}/start_semi_prod_detached.sh"
PROFILE="${SCRIPT_DIR}/profiles/semi_prod_dual_asr_codex_merge.env"

if [[ ! -x "${RUNNER}" ]]; then
  echo "Runner is not executable: ${RUNNER}"
  exit 2
fi
if [[ ! -f "${PROFILE}" ]]; then
  echo "Profile not found: ${PROFILE}"
  exit 2
fi

if [[ $# -eq 0 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  stable_runtime/start_codex_merge_detached.sh --input <recordings_dir> --output <transcripts_dir> [runner options]

This runs dual-ASR with Codex CLI merge on blocks of calls in detached mode.
Requires `codex login` done beforehand.

Example:
  CODEX_MERGE_MODEL=gpt-5.4 \
  ./stable_runtime/start_codex_merge_detached.sh \
    --input "/path/to/mango_export" \
    --output "/path/to/transcripts_codex_merge" \
    --db "/path/to/codex_merge.db" \
    --target 300 \
    --stage-limit 20 \
    --qc-every 50
EOF
  exit 0
fi

exec "${RUNNER}" --profile "${PROFILE}" "$@"
