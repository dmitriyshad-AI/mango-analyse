#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/dmitrijfabarisov/Projects/Mango analyse"
DB_PATH="$ROOT/stable_runtime/messages28_phone_history_asr_20260408/messages28_phone_history_asr_20260408.db"

if [[ -x "$ROOT/stable_runtime/venv_stable/bin/python" ]]; then
  PY_BIN="$ROOT/stable_runtime/venv_stable/bin/python"
else
  PY_BIN="$ROOT/stable_runtime/venv_stable.broken_20260407/bin/python"
fi

cd "$ROOT"
export PYTHONPATH="$ROOT/src"
export DATABASE_URL="sqlite:////Users/dmitrijfabarisov/Projects/Mango analyse/stable_runtime/messages28_phone_history_asr_20260408/messages28_phone_history_asr_20260408.db"
export TRANSCRIBE_PROVIDER="mlx"
export DUAL_TRANSCRIBE_ENABLED="1"
export SECONDARY_TRANSCRIBE_PROVIDER="gigaam"

"$PY_BIN" scripts/requeue_secondary_backfill.py --db "$DB_PATH" --provider gigaam --only-exhausted
exec "$PY_BIN" -u -m mango_mvp.cli worker --stage-limit 8 --stages backfill-second-asr --poll-sec 10 --max-idle-cycles 120
