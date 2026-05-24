#!/usr/bin/env python3
from __future__ import annotations

try:
    from telegram_pilot_daily_report import main
except ImportError:  # pragma: no cover - import path used by pytest/package contexts
    from scripts.telegram_pilot_daily_report import main


if __name__ == "__main__":
    raise SystemExit(main())
