#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from mango_mvp.channels.telegram_pilot_reporting import import_employee_feedback_csv


DEFAULT_DB_PATH = Path(".codex_local/telegram_pilot/telegram_pilot.sqlite")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Import employee Telegram pilot feedback CSV into local pilot store.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--actor", default="employee_review_sheet")
    args = parser.parse_args(argv)

    summary = import_employee_feedback_csv(args.db, args.csv, actor=args.actor)
    print(json.dumps(summary.to_json_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 1 if summary.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
