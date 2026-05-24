#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/dmitrijfabarisov/Projects/Mango analyse"
ENV_FILE="${1:-/Users/dmitrijfabarisov/.codex/mango_telegram_pilot_bots.env}"
RUNTIME_DIR="$ROOT/.codex_local/telegram_pilot_bots/runtime"
PID_FILE="$RUNTIME_DIR/public_pilot_bots.pid"
LOG_FILE="$RUNTIME_DIR/public_pilot_bots_$(date +%Y%m%d_%H%M%S).log"
SCREEN_NAME="mango_telegram_pilot_bots"
LAUNCHER_FILE="$RUNTIME_DIR/public_pilot_bots_launcher.sh"

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
if command -v screen >/dev/null 2>&1; then
  screen -S "$SCREEN_NAME" -X quit 2>/dev/null || true
fi

echo "Starting Telegram pilot bots..."
echo "Env file: $ENV_FILE"
echo "Log file: $LOG_FILE"

if command -v screen >/dev/null 2>&1; then
  cat > "$LAUNCHER_FILE" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$ROOT"
echo "\$\$" > "$PID_FILE"
exec env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/run_telegram_public_pilot_bots.py --env-file "$ENV_FILE" --mode poll --brand all > "$LOG_FILE" 2>&1
EOF
  chmod +x "$LAUNCHER_FILE"
  screen -dmS "$SCREEN_NAME" "$LAUNCHER_FILE"
  sleep 0.5
  new_pid="$(cat "$PID_FILE" || true)"
else
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src nohup python3 scripts/run_telegram_public_pilot_bots.py \
    --env-file "$ENV_FILE" \
    --mode poll \
    --brand all \
    > "$LOG_FILE" 2>&1 &

  new_pid="$!"
  echo "$new_pid" > "$PID_FILE"
fi

sleep 2
if kill -0 "$new_pid" 2>/dev/null; then
  echo "Telegram pilot bots started. PID: $new_pid"
  if command -v screen >/dev/null 2>&1; then
    echo "Screen session: $SCREEN_NAME"
  fi
  echo "Tail logs:"
  tail -20 "$LOG_FILE" || true
else
  echo "Telegram pilot bots failed to start. Log:"
  cat "$LOG_FILE" || true
  exit 1
fi
