#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Compatibility entrypoint: there is exactly one live Wappi policy.
exec "$ROOT/scripts/start_wappi_draft_loop_phase1b_live.sh" "$@"
