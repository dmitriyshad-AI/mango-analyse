#!/usr/bin/env python3
"""Run Customer Timeline Stage 3 maintenance on a non-production SQLite DB."""

from __future__ import annotations

import argparse
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from mango_mvp.customer_timeline.safety import (
    guard_customer_timeline_output_path,
    is_customer_timeline_prod_path,
)
from mango_mvp.customer_timeline.stage3_maintenance import (
    Stage3MaintenanceConfig,
    run_stage3_maintenance,
)
from mango_mvp.customer_timeline.store import json_dumps


REPORT_NAME = "stage3_maintenance_report.json"
COUNTED_TABLES = ("derived_signals", "customer_objections_v1", "family_links_v1")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage 3 maintenance on a staging Customer Timeline SQLite DB.")
    parser.add_argument("--db-path", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help=f"Report directory; writes {REPORT_NAME}.")
    parser.add_argument("--canonical-calls-db", type=Path)
    parser.add_argument("--as-of", help="Timezone-aware ISO timestamp for reproducible signal recompute.")
    parser.add_argument("--apply", action="store_true", help="Write to the staging DB. Default is dry-run.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    db_path = args.db_path.expanduser().resolve(strict=False)
    if not db_path.is_file():
        raise FileNotFoundError(f"timeline DB does not exist: {db_path}")
    if is_customer_timeline_prod_path(db_path):
        raise ValueError(f"refusing to run Stage 3 CLI on prod timeline path: {db_path}")

    allowed_root = db_path.parent
    output_dir = guard_customer_timeline_output_path(args.output.expanduser(), allowed_root)
    if is_customer_timeline_prod_path(output_dir):
        raise ValueError(f"refusing to write Stage 3 report to prod-like path: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    report = dict(
        run_stage3_maintenance(
            Stage3MaintenanceConfig(
                timeline_db_path=db_path,
                allowed_root=allowed_root,
                out_dir=output_dir,
                canonical_calls_db_path=args.canonical_calls_db,
                apply=bool(args.apply),
                signal_as_of=_parse_as_of(args.as_of),
            )
        )
    )
    report["table_counts"] = _table_counts(db_path)
    report_path = output_dir / REPORT_NAME
    report_path.write_text(json_dumps(report), encoding="utf-8")
    print(json_dumps(report))
    return 0


def _table_counts(db_path: Path) -> dict[str, int]:
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as con:
        tables = {
            str(row[0])
            for row in con.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name IN (?, ?, ?)",
                COUNTED_TABLES,
            )
        }
        return {
            table: int(con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]) if table in tables else 0
            for table in COUNTED_TABLES
        }


def _parse_as_of(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("--as-of must be timezone-aware")
    return parsed.astimezone(timezone.utc)


if __name__ == "__main__":
    raise SystemExit(main())
