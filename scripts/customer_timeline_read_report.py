#!/usr/bin/env python3
"""Build a read-only customer timeline API/report snapshot.

This command opens customer_timeline.sqlite in SQLite read-only mode. It does
not call CRM, Tallanto, messengers, ASR, R+A, or stable_runtime DBs.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mango_mvp.customer_timeline.read_api import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
