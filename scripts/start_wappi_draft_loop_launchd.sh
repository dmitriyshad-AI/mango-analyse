#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="${DRAFT_LOOP_LAUNCHD_LABEL:-com.mango.wappi-draft-loop}"
PLIST_SOURCE="${DRAFT_LOOP_LAUNCHD_SOURCE:-${ROOT}/deploy/wappi_draft_loop/${LABEL}.plist.template}"
PLIST_TARGET="${DRAFT_LOOP_LAUNCHD_PLIST:-${HOME}/Library/LaunchAgents/${LABEL}.plist}"
ROLLBACK_PLIST="${DRAFT_LOOP_LAUNCHD_ROLLBACK:-${HOME}/.mango_local/draft_loop/${LABEL}.rollback.plist}"
MODE="${1:-}"
PYTHON_BIN="/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/Resources/Python.app/Contents/MacOS/Python"
[[ -x "$PYTHON_BIN" ]] || PYTHON_BIN="$(command -v python3)"

if [[ -n "$MODE" && "$MODE" != "--render-only" ]]; then
  echo "Usage: $0 [--render-only]" >&2
  exit 2
fi

EXPECTED_HEAD="$(git -C "$ROOT" rev-parse HEAD)"
RENDERED_PLIST="$(mktemp "${TMPDIR:-/tmp}/mango-wappi-launchd.XXXXXX")"
trap 'rm -f "$RENDERED_PLIST"' EXIT

"$PYTHON_BIN" - "$PLIST_SOURCE" "$RENDERED_PLIST" "$ROOT" "$EXPECTED_HEAD" <<'PY'
from pathlib import Path
import plistlib
import sys
from xml.sax.saxutils import escape

source = Path(sys.argv[1])
target = Path(sys.argv[2])
code_root = sys.argv[3]
expected_head = sys.argv[4]
rendered = source.read_text(encoding="utf-8")
rendered = rendered.replace("__MANGO_CODE_ROOT__", escape(str(Path(code_root).resolve())))
rendered = rendered.replace("__MANGO_EXPECTED_HEAD__", escape(expected_head))
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
WAS_LOADED=0
HAD_PLIST=0
install -d -m 0755 "$(dirname "${PLIST_TARGET}")" "$(dirname "${ROLLBACK_PLIST}")"
if [[ -f "$PLIST_TARGET" ]]; then
  plutil -lint "$PLIST_TARGET" >/dev/null
  install -m 0644 "$PLIST_TARGET" "$ROLLBACK_PLIST"
  HAD_PLIST=1
fi
if launchctl print "${DOMAIN}/${LABEL}" >/dev/null 2>&1; then
  WAS_LOADED=1
  launchctl bootout "${DOMAIN}/${LABEL}"
fi
install -m 0644 "${RENDERED_PLIST}" "${PLIST_TARGET}"

rollback() {
  launchctl bootout "${DOMAIN}/${LABEL}" >/dev/null 2>&1 || true
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
