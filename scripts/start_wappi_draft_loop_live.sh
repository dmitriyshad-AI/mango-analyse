#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LOG_DIR="${DRAFT_LOOP_LOG_DIR:-/Users/dmitrijfabarisov/.mango_local/draft_loop}"
mkdir -p "$LOG_DIR"
exec >>"$LOG_DIR/poller.log" 2>&1
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] starting Wappi draft-loop from $ROOT"

export CODEX_HOME="${CODEX_HOME:-/Users/dmitrijfabarisov/.mango_local/codex_wappi_draft_loop_v1}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export PYTHONPATH="${PYTHONPATH:-src}"
export DRAFT_LOOP_AUTO_RESOLVER="${DRAFT_LOOP_AUTO_RESOLVER:-0}"
AUTO_PAIRS_FILE="${DRAFT_LOOP_AUTO_PAIRS_FILE:-/Users/dmitrijfabarisov/.mango_local/draft_loop/empty_auto_pairs.json}"
mkdir -p "$(dirname "$AUTO_PAIRS_FILE")"
if [[ ! -f "$AUTO_PAIRS_FILE" ]]; then
  printf '{"pairs":[]}\n' > "$AUTO_PAIRS_FILE"
fi

export TELEGRAM_TIMELINE_MEMORY_IN_PROMPT="${TELEGRAM_TIMELINE_MEMORY_IN_PROMPT:-1}"
export TELEGRAM_BOT_SAFE_CRM_CONTEXT="${TELEGRAM_BOT_SAFE_CRM_CONTEXT:-1}"
export TELEGRAM_BOT_SAFE_CRM_CONTEXT_DB="${TELEGRAM_BOT_SAFE_CRM_CONTEXT_DB:-/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_timeline/customer_timeline_prod_20260621/customer_timeline.sqlite}"
export TELEGRAM_BOT_SAFE_CRM_CONTEXT_TENANT="${TELEGRAM_BOT_SAFE_CRM_CONTEXT_TENANT:-foton}"

export TELEGRAM_DIRECT_PATH_PILOT_CONFIG="${TELEGRAM_DIRECT_PATH_PILOT_CONFIG:-pilot_gold_v1}"
export TELEGRAM_P0_MODEL_LED="${TELEGRAM_P0_MODEL_LED:-1}"
export TELEGRAM_PROSE_MODEL_LED="${TELEGRAM_PROSE_MODEL_LED:-1}"
export TELEGRAM_FACT_VENUE_SCOPE="${TELEGRAM_FACT_VENUE_SCOPE:-1}"
export TELEGRAM_AUTONOMY_SCOPE_PRECISION="${TELEGRAM_AUTONOMY_SCOPE_PRECISION:-1}"

exec python3 scripts/run_amo_wappi_draft_loop.py \
  --loop \
  --live-write \
  --interval-sec "${DRAFT_LOOP_INTERVAL_SEC:-45}" \
  --model "${DRAFT_LOOP_MODEL:-gpt-5.5}" \
  --reasoning "${DRAFT_LOOP_REASONING:-high}" \
  --customer-timeline-db "$TELEGRAM_BOT_SAFE_CRM_CONTEXT_DB" \
  --customer-timeline-tenant "$TELEGRAM_BOT_SAFE_CRM_CONTEXT_TENANT" \
  --auto-pairs-file "$AUTO_PAIRS_FILE"
