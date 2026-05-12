#!/usr/bin/env python3
"""Build a manager-reviewed channel draft from an approved Customer Timeline context pack."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mango_mvp.customer_timeline.channel_preview_from_pack import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
