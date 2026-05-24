#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Optional, Sequence

from mango_mvp.channels.telegram_pilot_reporting import build_pilot_daily_report


DEFAULT_DB_PATH = Path(".codex_local/telegram_pilot/telegram_pilot.sqlite")
DEFAULT_P0_REGISTER_PATH = Path(".codex_local/telegram_pilot/p0_incident_register.csv")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build a local Telegram pilot daily report.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--p0-register", type=Path, default=DEFAULT_P0_REGISTER_PATH)
    args = parser.parse_args(argv)

    out_dir = args.out_dir or Path("audits/_inbox") / f"telegram_pilot_daily_{str(args.date).replace('-', '')}"
    report = build_pilot_daily_report(args.db, args.date, out_dir=out_dir, p0_register_path=args.p0_register)
    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
