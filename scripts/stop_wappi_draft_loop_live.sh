#!/usr/bin/env bash
set -euo pipefail

screen -S mango_draft_loop -X quit >/dev/null 2>&1 || true
sleep 2

python3 - <<'PY'
from __future__ import annotations

import os
import signal
import subprocess
import time

own_pid = os.getpid()
out = subprocess.run(["ps", "-axo", "pid=,command="], text=True, capture_output=True, check=False).stdout
pids: list[int] = []
for raw in out.splitlines():
    line = raw.strip()
    if not line:
        continue
    try:
        pid_raw, command = line.split(maxsplit=1)
        pid = int(pid_raw)
    except ValueError:
        continue
    if pid == own_pid:
        continue
    if "scripts/run_amo_wappi_draft_loop.py" in command:
        pids.append(pid)

for pid in pids:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass

if pids:
    time.sleep(3)

out = subprocess.run(["ps", "-axo", "pid=,command="], text=True, capture_output=True, check=False).stdout
remaining: list[int] = []
for raw in out.splitlines():
    line = raw.strip()
    if not line:
        continue
    try:
        pid_raw, command = line.split(maxsplit=1)
        pid = int(pid_raw)
    except ValueError:
        continue
    if pid == own_pid:
        continue
    if "scripts/run_amo_wappi_draft_loop.py" in command:
        remaining.append(pid)

for pid in remaining:
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass

print({"terminated": pids, "killed": remaining})
PY
