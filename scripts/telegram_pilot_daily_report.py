#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Optional, Sequence

from mango_mvp.channels.telegram_pilot_metrics import build_daily_metrics, write_daily_metrics_report


DEFAULT_DB_PATH = Path(".codex_local/telegram_pilot/telegram_pilot.sqlite")
DEFAULT_REPORT_DIR = Path(".codex_local/telegram_pilot/reports")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build a local Telegram pilot daily report.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args(argv)

    metrics = build_daily_metrics(args.db, args.date)
    paths = write_daily_metrics_report(metrics, args.out_dir)
    print(
        json.dumps(
            {
                "metrics": metrics.to_json_dict(),
                "outputs": paths,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
