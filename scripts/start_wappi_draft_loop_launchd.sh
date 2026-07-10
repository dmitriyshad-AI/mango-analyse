#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="${DRAFT_LOOP_LAUNCHD_LABEL:-com.mango.wappi-draft-loop}"
PLIST_SOURCE="${DRAFT_LOOP_LAUNCHD_SOURCE:-${ROOT}/deploy/wappi_draft_loop/${LABEL}.plist.template}"
PLIST_TARGET="${DRAFT_LOOP_LAUNCHD_PLIST:-${HOME}/Library/LaunchAgents/${LABEL}.plist}"

install -d -m 0755 "$(dirname "${PLIST_TARGET}")"
install -m 0644 "${PLIST_SOURCE}" "${PLIST_TARGET}"

if launchctl print "gui/$(id -u)/${LABEL}" >/dev/null 2>&1; then
  launchctl kickstart -k "gui/$(id -u)/${LABEL}"
else
  launchctl bootstrap "gui/$(id -u)" "${PLIST_TARGET}"
fi

python3 - <<'PY'
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
