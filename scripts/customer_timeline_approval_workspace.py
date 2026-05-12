#!/usr/bin/env python3
"""Build a static read-only approval workspace for customer_timeline.sqlite."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mango_mvp.customer_timeline.approval_workspace import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
