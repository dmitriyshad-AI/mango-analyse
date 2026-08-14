#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="${DRAFT_LOOP_LAUNCHD_LABEL:-com.mango.wappi-draft-loop}"
PLIST_SOURCE="${DRAFT_LOOP_LAUNCHD_SOURCE:-${ROOT}/deploy/wappi_draft_loop/${LABEL}.plist.template}"
PLIST_TARGET="${DRAFT_LOOP_LAUNCHD_PLIST:-${HOME}/Library/LaunchAgents/${LABEL}.plist}"
ROLLBACK_PLIST="${DRAFT_LOOP_LAUNCHD_ROLLBACK:-${HOME}/.mango_local/draft_loop/${LABEL}.rollback.plist}"
MODE="${1:-}"
PYTHON_BIN="${DRAFT_LOOP_PYTHON_BIN:-${HOME}/.mango_local/draft_loop/venv/bin/python}"
[[ -x "$PYTHON_BIN" ]] || PYTHON_BIN="$(command -v python3)"

if [[ -n "$MODE" && "$MODE" != "--render-only" && "$MODE" != "--status" && "$MODE" != "--smoke" && "$MODE" != "--stop" && "$MODE" != "--rollback" ]]; then
  echo "Usage: $0 [--render-only|--status|--smoke|--stop|--rollback]" >&2
  exit 2
fi

EXPECTED_HEAD="$(git -C "$ROOT" rev-parse HEAD)"
RENDERED_PLIST="$(mktemp "${TMPDIR:-/tmp}/mango-wappi-launchd.XXXXXX")"
trap 'rm -f "$RENDERED_PLIST"' EXIT

"$PYTHON_BIN" - "$PLIST_SOURCE" "$RENDERED_PLIST" "$ROOT" "$EXPECTED_HEAD" "$HOME" <<'PY'
from pathlib import Path
import plistlib
import sys
from xml.sax.saxutils import escape

source = Path(sys.argv[1])
target = Path(sys.argv[2])
code_root = sys.argv[3]
expected_head = sys.argv[4]
home = sys.argv[5]
rendered = source.read_text(encoding="utf-8")
rendered = rendered.replace("__MANGO_CODE_ROOT__", escape(str(Path(code_root).resolve())))
rendered = rendered.replace("__MANGO_EXPECTED_HEAD__", escape(expected_head))
rendered = rendered.replace("__MANGO_HOME__", escape(str(Path(home).resolve())))
if "__MANGO_" in rendered:
    raise SystemExit("unresolved Wappi launchd placeholder")
plistlib.loads(rendered.encode("utf-8"))
target.write_text(rendered, encoding="utf-8")
PY

plutil -lint "$RENDERED_PLIST" >/dev/null
if [[ "$MODE" == "--render-only" ]]; then
  cat "$RENDERED_PLIST"
  exit 0
fi

DOMAIN="gui/$(id -u)"
if [[ "$MODE" == "--stop" ]]; then
  launchctl bootout "${DOMAIN}/${LABEL}" >/dev/null 2>&1 || true
  exit 0
fi
if [[ "$MODE" == "--status" || "$MODE" == "--smoke" ]]; then
  launchctl print "${DOMAIN}/${LABEL}" >/dev/null 2>&1 || exit 4
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$ROOT/src" "$PYTHON_BIN" "$ROOT/scripts/skills/live_truth.py" \
    --root "$ROOT" --expect-head "run_amo_wappi_draft_loop.py=$EXPECTED_HEAD" --no-write
  exit $?
fi
if [[ "$MODE" == "--rollback" ]]; then
  [[ -f "$ROLLBACK_PLIST" ]] || { echo "No previous Wappi plist to restore" >&2; exit 4; }
  launchctl bootout "${DOMAIN}/${LABEL}" >/dev/null 2>&1 || true
  install -m 0644 "$ROLLBACK_PLIST" "$PLIST_TARGET"
  launchctl bootstrap "$DOMAIN" "$PLIST_TARGET"
  exit 0
fi
WAS_LOADED=0
HAD_PLIST=0

wait_until_unloaded() {
  for _ in {1..50}; do
    launchctl print "${DOMAIN}/${LABEL}" >/dev/null 2>&1 || return 0
    sleep 0.1
  done
  return 1
}

install -d -m 0755 "$(dirname "${PLIST_TARGET}")" "$(dirname "${ROLLBACK_PLIST}")"
if [[ -f "$PLIST_TARGET" ]]; then
  plutil -lint "$PLIST_TARGET" >/dev/null
  install -m 0644 "$PLIST_TARGET" "$ROLLBACK_PLIST"
  HAD_PLIST=1
fi
if launchctl print "${DOMAIN}/${LABEL}" >/dev/null 2>&1; then
  WAS_LOADED=1
  launchctl bootout "${DOMAIN}/${LABEL}"
  if ! wait_until_unloaded; then
    echo "Wappi launchd job did not unload" >&2
    exit 3
  fi
fi
install -m 0644 "${RENDERED_PLIST}" "${PLIST_TARGET}"

rollback() {
  launchctl bootout "${DOMAIN}/${LABEL}" >/dev/null 2>&1 || true
  wait_until_unloaded || true
  if [[ "$HAD_PLIST" == "1" ]]; then
    install -m 0644 "$ROLLBACK_PLIST" "$PLIST_TARGET"
    if [[ "$WAS_LOADED" == "1" ]]; then
      launchctl bootstrap "$DOMAIN" "$PLIST_TARGET" || true
    fi
  else
    rm -f "$PLIST_TARGET"
  fi
}

if ! launchctl bootstrap "$DOMAIN" "$PLIST_TARGET"; then
  rollback
  echo "Wappi launchd bootstrap failed; previous plist restored" >&2
  exit 3
fi

if ! "$PYTHON_BIN" - <<'PY'
from __future__ import annotations

import subprocess
import sys
import time


def pids() -> list[str]:
    out = subprocess.run(["ps", "-axo", "pid=,command="], check=False, text=True, stdout=subprocess.PIPE).stdout
    result: list[str] = []
    for raw in out.splitlines():
        line = raw.strip()
        if "scripts/run_amo_wappi_draft_loop.py" not in line:
            continue
        if "wappi_draft_loop_ops.py" in line:
            continue
        result.append(line.split(maxsplit=1)[0])
    return result


deadline = time.monotonic() + 45
last: list[str] = []
while time.monotonic() < deadline:
    last = pids()
    if len(last) == 1:
        print({"status": "ok", "pid": last[0]})
        raise SystemExit(0)
    if len(last) > 1:
        print({"status": "multiple_processes", "pids": last}, file=sys.stderr)
        raise SystemExit(3)
    time.sleep(1)

print({"status": "not_started", "pids": last}, file=sys.stderr)
raise SystemExit(4)
PY
then
  rollback
  echo "Wappi draft-loop smoke failed; previous plist restored" >&2
  exit 4
fi

if ! PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$ROOT/src" "$PYTHON_BIN" "$ROOT/scripts/skills/live_truth.py" \
  --root "$ROOT" \
  --expect-head "run_amo_wappi_draft_loop.py=$EXPECTED_HEAD" \
  --no-write; then
  rollback
  echo "Wappi live truth check failed; previous plist restored" >&2
  exit 5
fi
