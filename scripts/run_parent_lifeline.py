#!/usr/bin/env python3
"""Execute a command and kill this group if the real parent disappears."""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys


def main() -> int:
    if len(sys.argv) < 2:
        return 64
    raw_fd = os.getenv("MANGO_CALLS_CONTROLLED_LIFELINE_FD", "")
    if not re.fullmatch(r"[0-9]{1,6}", raw_fd):
        return 65
    read_fd = int(raw_fd)
    sentinel = os.fork()
    if sentinel == 0:
        try:
            while os.read(read_fd, 1):
                pass
            os.killpg(os.getpgrp(), signal.SIGKILL)
        except BaseException:
            os._exit(70)
        os._exit(0)
    os.close(read_fd)
    try:
        return subprocess.call(sys.argv[1:])
    finally:
        try:
            os.kill(sentinel, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            os.waitpid(sentinel, 0)
        except ChildProcessError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
