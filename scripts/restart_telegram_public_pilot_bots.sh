#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/dmitrijfabarisov/Projects/Mango analyse"
ENV_FILE="${1:-/Users/dmitrijfabarisov/.codex/mango_telegram_pilot_bots.env}"
RUNTIME_DIR="$ROOT/.codex_local/telegram_pilot_bots/runtime"
PID_FILE="$RUNTIME_DIR/public_pilot_bots.pid"
LOG_FILE="$RUNTIME_DIR/public_pilot_bots_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$RUNTIME_DIR"
cd "$ROOT"

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE" || true)"
  if [[ -n "${old_pid:-}" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "Stopping existing bot process: $old_pid"
    kill "$old_pid"
    for _ in {1..20}; do
      if ! kill -0 "$old_pid" 2>/dev/null; then
        break
      fi
      sleep 0.5
    done
    if kill -0 "$old_pid" 2>/dev/null; then
      echo "Process did not stop gracefully, sending TERM again: $old_pid"
      kill "$old_pid" || true
    fi
  fi
fi

pkill -f "scripts/run_telegram_public_pilot_bots.py --mode poll" 2>/dev/null || true

echo "Starting Telegram pilot bots..."
echo "Env file: $ENV_FILE"
echo "Log file: $LOG_FILE"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src nohup python3 scripts/run_telegram_public_pilot_bots.py \
  --env-file "$ENV_FILE" \
  --mode poll \
  --brand all \
  > "$LOG_FILE" 2>&1 &

new_pid="$!"
echo "$new_pid" > "$PID_FILE"

sleep 2
if kill -0 "$new_pid" 2>/dev/null; then
  echo "Telegram pilot bots started. PID: $new_pid"
  echo "Tail logs:"
  tail -20 "$LOG_FILE" || true
else
  echo "Telegram pilot bots failed to start. Log:"
  cat "$LOG_FILE" || true
  exit 1
fi
