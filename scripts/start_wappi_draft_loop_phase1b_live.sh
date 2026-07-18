#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/Resources/Python.app/Contents/MacOS/Python"
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
export DRAFT_LOOP_AUTO_RESOLVER=0

export ENFORCE_CANONICAL_PROFILE=1
export TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1
export TELEGRAM_BOT_SAFE_MEMORY_STEP_GUARD=1
export TELEGRAM_BOT_SAFE_CRM_CONTEXT=1
export TELEGRAM_TIMELINE_MEMORY_IN_PROMPT=1
export TELEGRAM_DIRECT_PATH_FORMAT_GUIDANCE=1
export TELEGRAM_DIRECT_PATH_SCOPE_OVERCLAIM_GUARD=0

AUTO_PAIRS_FILE="${DRAFT_LOOP_AUTO_PAIRS_FILE:-$HOME/.mango_local/draft_loop/empty_auto_pairs.json}"
if [[ ! -f "$AUTO_PAIRS_FILE" ]]; then
  mkdir -p "$(dirname "$AUTO_PAIRS_FILE")"
  printf '{"pairs":[]}\n' >"$AUTO_PAIRS_FILE"
fi

CUSTOMER_TIMELINE_DB="${CUSTOMER_TIMELINE_DB:-/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_timeline/customer_timeline_prod_20260621/customer_timeline.sqlite}"

MANIFEST="$LOG_DIR/phase1b_startup_manifest.json"
MANIFEST_TMP="$MANIFEST.$$"
{
  printf '{\n'
  printf '  "schema_version": "wappi_phase1b_startup_v1",\n'
  printf '  "started_at": "%s",\n' "$STARTED_AT"
  printf '  "cwd": "%s",\n' "$ROOT"
  printf '  "head": "%s",\n' "$HEAD"
  printf '  "wrapper_sha256": "%s",\n' "$WRAPPER_SHA256"
  printf '  "profile": "pilot_gold_v1",\n'
  printf '  "pair_mode": "manual_only",\n'
  printf '  "memory_step_guard": true,\n'
  printf '  "bot_safe_crm_context": true,\n'
  printf '  "timeline_memory_in_prompt": true,\n'
  printf '  "format_guidance": true,\n'
  printf '  "scope_overclaim_guard": false\n'
  printf '}\n'
} >"$MANIFEST_TMP"
mv "$MANIFEST_TMP" "$MANIFEST"

exec "$PYTHON_BIN" scripts/run_amo_wappi_draft_loop.py \
  --loop \
  --live-write \
  --interval-sec "${DRAFT_LOOP_INTERVAL_SEC:-45}" \
  --model "${DRAFT_LOOP_MODEL:-gpt-5.5}" \
  --reasoning "${DRAFT_LOOP_REASONING:-high}" \
  --customer-timeline-db "$CUSTOMER_TIMELINE_DB" \
  --customer-timeline-tenant "${CUSTOMER_TIMELINE_TENANT:-foton}" \
  --auto-pairs-file "$AUTO_PAIRS_FILE"
