#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${DRAFT_LOOP_PYTHON_BIN:-${HOME}/.mango_local/draft_loop/venv/bin/python}"
[[ -x "$PYTHON_BIN" ]] || PYTHON_BIN="$(command -v python3)"
cd "$ROOT"

LOG_DIR="${DRAFT_LOOP_LOG_DIR:-$HOME/.mango_local/draft_loop}"
mkdir -p "$LOG_DIR"
exec >>"$LOG_DIR/poller_phase1b.log" 2>&1

HEAD="$(git rev-parse HEAD)"
EXPECTED_HEAD="${DRAFT_LOOP_EXPECTED_HEAD:-}"
if [[ -n "$EXPECTED_HEAD" && "$HEAD" != "$EXPECTED_HEAD" ]]; then
  echo "Refusing Phase 1b start: HEAD=$HEAD expected=$EXPECTED_HEAD"
  exit 78
fi
if ! git diff --quiet -- || ! git diff --cached --quiet --; then
  echo "Refusing Phase 1b start: tracked code is dirty"
  exit 78
fi
if [[ -n "$(git ls-files --others --exclude-standard -- src scripts deploy)" ]]; then
  echo "Refusing Phase 1b start: untracked code exists under src/scripts/deploy"
  exit 78
fi

STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
WRAPPER_SHA256="$(shasum -a 256 "${BASH_SOURCE[0]}" | awk '{print $1}')"
echo "[$STARTED_AT] starting Phase 1b Wappi draft-loop from $ROOT at $HEAD"

export CODEX_HOME="${CODEX_HOME:-$HOME/.mango_local/codex_wappi_draft_loop_v1}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=src

export ENFORCE_CANONICAL_PROFILE=1
export TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1
export TELEGRAM_BOT_SAFE_MEMORY_STEP_GUARD=1
export TELEGRAM_BOT_SAFE_CRM_CONTEXT="${TELEGRAM_BOT_SAFE_CRM_CONTEXT:-0}"
export TELEGRAM_TIMELINE_MEMORY_IN_PROMPT="${TELEGRAM_TIMELINE_MEMORY_IN_PROMPT:-0}"
export TELEGRAM_DIRECT_PATH_FORMAT_GUIDANCE=1
export TELEGRAM_DIRECT_PATH_SCOPE_OVERCLAIM_GUARD=0

AUTO_PAIRS_FILE="${DRAFT_LOOP_AUTO_PAIRS_FILE:-$HOME/.mango_local/draft_loop/empty_auto_pairs.json}"
if [[ ! -f "$AUTO_PAIRS_FILE" ]]; then
  mkdir -p "$(dirname "$AUTO_PAIRS_FILE")"
  printf '{"pairs":[]}\n' >"$AUTO_PAIRS_FILE"
fi

TIMELINE_ARGS=()
if [[ "$TELEGRAM_BOT_SAFE_CRM_CONTEXT" == "1" ]]; then
  : "${CUSTOMER_TIMELINE_DB:?CUSTOMER_TIMELINE_DB is required when Timeline context is enabled}"
  TIMELINE_ARGS=(--customer-timeline-db "$CUSTOMER_TIMELINE_DB" --customer-timeline-tenant "${CUSTOMER_TIMELINE_TENANT:-foton}")
fi

MANIFEST="$LOG_DIR/phase1b_startup_manifest.json"
export DRAFT_LOOP_STARTUP_MANIFEST="$MANIFEST"
export DRAFT_LOOP_RUNTIME_HEAD="$HEAD"
export DRAFT_LOOP_WRAPPER_SHA256="$WRAPPER_SHA256"

exec "$PYTHON_BIN" scripts/run_amo_wappi_draft_loop.py \
  --loop \
  --live-write \
  --interval-sec "${DRAFT_LOOP_INTERVAL_SEC:-45}" \
  --model "${DRAFT_LOOP_MODEL:-gpt-5.5}" \
  --reasoning "${DRAFT_LOOP_REASONING:-high}" \
  "${TIMELINE_ARGS[@]}" \
  --auto-pairs-file "$AUTO_PAIRS_FILE"
