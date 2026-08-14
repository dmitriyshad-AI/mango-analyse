#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BRAND="${1:-}"
MODE="${2:-}"
if [[ "$BRAND" != "foton" && "$BRAND" != "unpk" ]]; then
  echo "Usage: $0 {foton|unpk} [--render-only|--run]" >&2
  exit 2
fi
if [[ -n "$MODE" && "$MODE" != "--render-only" && "$MODE" != "--run" && "$MODE" != "--status" && "$MODE" != "--smoke" && "$MODE" != "--stop" && "$MODE" != "--rollback" ]]; then
  echo "Usage: $0 {foton|unpk} [--render-only|--run|--status|--smoke|--stop|--rollback]" >&2
  exit 2
fi

LABEL="com.mango.telegram-ai-agent.${BRAND}"
SOURCE="${MANGO_TELEGRAM_LAUNCHD_SOURCE:-${ROOT}/deploy/telegram_ai_agent/com.mango.telegram-ai-agent.plist.template}"
TARGET="${MANGO_TELEGRAM_LAUNCHD_PLIST:-${HOME}/Library/LaunchAgents/${LABEL}.plist}"
ROLLBACK="${MANGO_TELEGRAM_LAUNCHD_ROLLBACK:-${HOME}/.mango_local/telegram_ai_agent/${LABEL}.rollback.plist}"
PYTHON_BIN="${MANGO_TELEGRAM_PYTHON_BIN:-${HOME}/.mango_local/telegram_ai_agent/venv/bin/python}"
[[ -x "$PYTHON_BIN" ]] || PYTHON_BIN="$(command -v python3)"

if [[ "$MODE" == "--run" ]]; then
  cd "$ROOT"
  HEAD="$(git rev-parse HEAD)"
  EXPECTED="${MANGO_TELEGRAM_EXPECTED_HEAD:-}"
  [[ -n "$EXPECTED" && "$HEAD" == "$EXPECTED" ]] || { echo "Refusing Telegram start: HEAD=$HEAD expected=$EXPECTED" >&2; exit 78; }
  git diff --quiet -- && git diff --cached --quiet -- || { echo "Refusing Telegram start: tracked code is dirty" >&2; exit 78; }
  [[ -z "$(git ls-files --others --exclude-standard -- src scripts deploy)" ]] || { echo "Refusing Telegram start: untracked code exists" >&2; exit 78; }
  ENV_FILE="${MANGO_TELEGRAM_ENV_FILE:-${HOME}/.codex/mango_telegram_pilot_bots.env}"
  [[ -f "$ENV_FILE" ]] || { echo "Telegram env file not found: $ENV_FILE" >&2; exit 78; }
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
  export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ENFORCE_CANONICAL_PROFILE=1
  MANIFEST_DIR="${HOME}/.mango_local/telegram_ai_agent"
  mkdir -p "$MANIFEST_DIR"
  export MANGO_TELEGRAM_RUNTIME_STATUS="$MANIFEST_DIR/${BRAND}_startup_manifest.json"
  KB_PATH="$ROOT/product_data/knowledge_base/kb_release_20260813_v6_8_owner_approved/kb_release_v3_snapshot.json"
  KB_SHA256="$(shasum -a 256 "$KB_PATH" | awk '{print $1}')"
  "$PYTHON_BIN" -c 'import json,os,sys; from datetime import datetime,timezone; from pathlib import Path; Path(sys.argv[1]).write_text(json.dumps({"status":"starting","brand":sys.argv[2],"head":sys.argv[3],"pid":os.getppid(),"started_at":datetime.now(timezone.utc).isoformat(),"code_root":sys.argv[4],"kb_path":sys.argv[5],"kb_sha256":sys.argv[6],"state_path":sys.argv[7]},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")' "$MANGO_TELEGRAM_RUNTIME_STATUS" "$BRAND" "$HEAD" "$ROOT" "$KB_PATH" "$KB_SHA256" "$MANIFEST_DIR/${BRAND}_dialogue_state.json"
  exec "$PYTHON_BIN" scripts/run_telegram_ai_agent.py --brand "$BRAND"
fi

EXPECTED_HEAD="$(git -C "$ROOT" rev-parse HEAD)"
STATUS_PATH="${HOME}/.mango_local/telegram_ai_agent/${BRAND}_startup_manifest.json"
KB_PATH="$ROOT/product_data/knowledge_base/kb_release_20260813_v6_8_owner_approved/kb_release_v3_snapshot.json"
KB_SHA256="$(shasum -a 256 "$KB_PATH" | awk '{print $1}')"
RENDERED="$(mktemp "${TMPDIR:-/tmp}/mango-telegram-launchd.XXXXXX")"
trap 'rm -f "$RENDERED"' EXIT
"$PYTHON_BIN" - "$SOURCE" "$RENDERED" "$ROOT" "$EXPECTED_HEAD" "$HOME" "$BRAND" "$LABEL" <<'PY'
from pathlib import Path
import plistlib
import sys
from xml.sax.saxutils import escape

source, target = Path(sys.argv[1]), Path(sys.argv[2])
values = {
    "__MANGO_CODE_ROOT__": str(Path(sys.argv[3]).resolve()),
    "__MANGO_EXPECTED_HEAD__": sys.argv[4],
    "__MANGO_HOME__": str(Path(sys.argv[5]).resolve()),
    "__MANGO_BRAND__": sys.argv[6],
    "__MANGO_LABEL__": sys.argv[7],
}
rendered = source.read_text(encoding="utf-8")
for marker, value in values.items():
    rendered = rendered.replace(marker, escape(value))
if "__MANGO_" in rendered:
    raise SystemExit("unresolved Telegram launchd placeholder")
plistlib.loads(rendered.encode("utf-8"))
target.write_text(rendered, encoding="utf-8")
PY
plutil -lint "$RENDERED" >/dev/null
if [[ "$MODE" == "--render-only" ]]; then
  cat "$RENDERED"
  exit 0
fi

DOMAIN="gui/$(id -u)"
status_ok() {
  "$PYTHON_BIN" - "$STATUS_PATH" "$EXPECTED_HEAD" "$KB_SHA256" "$BRAND" "$ROOT" <<'PY'
import json,os,sys
from datetime import datetime,timezone
from pathlib import Path

path=Path(sys.argv[1])
try:
    payload=json.loads(path.read_text(encoding="utf-8"))
    stamp=datetime.fromisoformat(str(payload["last_cycle_at"]).replace("Z", "+00:00"))
    age=(datetime.now(timezone.utc)-stamp.astimezone(timezone.utc)).total_seconds()
except (OSError,ValueError,KeyError,TypeError,json.JSONDecodeError):
    raise SystemExit(1)
pid=int(payload.get("pid") or 0)
try:
    os.kill(pid, 0)
    pid_alive=pid>0
except (OSError, ValueError):
    pid_alive=False
ok=(payload.get("status")=="ok" and payload.get("head")==sys.argv[2]
    and payload.get("kb_sha256")==sys.argv[3] and payload.get("brand")==sys.argv[4]
    and Path(str(payload.get("code_root") or "")).resolve()==Path(sys.argv[5]).resolve()
    and pid_alive and 0<=age<=300)
raise SystemExit(0 if ok else 1)
PY
}
if [[ "$MODE" == "--stop" ]]; then
  launchctl bootout "${DOMAIN}/${LABEL}" >/dev/null 2>&1 || true
  exit 0
fi
if [[ "$MODE" == "--status" || "$MODE" == "--smoke" ]]; then
  launchctl print "${DOMAIN}/${LABEL}" >/dev/null 2>&1 || exit 4
  status_ok || exit 4
  cat "$STATUS_PATH"
  exit 0
fi
if [[ "$MODE" == "--rollback" ]]; then
  [[ -f "$ROLLBACK" ]] || { echo "No previous plist to restore" >&2; exit 4; }
  launchctl bootout "${DOMAIN}/${LABEL}" >/dev/null 2>&1 || true
  install -m 0644 "$ROLLBACK" "$TARGET"
  launchctl bootstrap "$DOMAIN" "$TARGET"
  exit 0
fi
install -d -m 0755 "$(dirname "$TARGET")" "$(dirname "$ROLLBACK")"
HAD_PLIST=0
WAS_LOADED=0
if [[ -f "$TARGET" ]]; then
  plutil -lint "$TARGET" >/dev/null
  install -m 0644 "$TARGET" "$ROLLBACK"
  HAD_PLIST=1
fi
launchctl print "${DOMAIN}/${LABEL}" >/dev/null 2>&1 && WAS_LOADED=1
launchctl bootout "${DOMAIN}/${LABEL}" >/dev/null 2>&1 || true
install -m 0644 "$RENDERED" "$TARGET"
rm -f "$STATUS_PATH"
rollback() {
  launchctl bootout "${DOMAIN}/${LABEL}" >/dev/null 2>&1 || true
  if [[ "$HAD_PLIST" == "1" ]]; then
    install -m 0644 "$ROLLBACK" "$TARGET"
    [[ "$WAS_LOADED" == "1" ]] && launchctl bootstrap "$DOMAIN" "$TARGET" >/dev/null 2>&1 || true
  else
    rm -f "$TARGET"
  fi
}
if ! launchctl bootstrap "$DOMAIN" "$TARGET"; then
  rollback
  echo "Telegram launchd bootstrap failed; previous plist restored" >&2
  exit 3
fi
for _ in {1..45}; do
  if launchctl print "${DOMAIN}/${LABEL}" >/dev/null 2>&1 && status_ok; then
    exit 0
  fi
  sleep 1
done
rollback
echo "Telegram service did not publish a fresh verified heartbeat" >&2
exit 4
