from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from mango_mvp.customer_profile.builder import CustomerProfileBuilder, CustomerProfileBuildOptions
from mango_mvp.customer_profile.store import CustomerProfileSQLiteStore
from mango_mvp.customer_timeline.read_api import mask_phone


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build deterministic customer profiles from read-only timeline.")
    parser.add_argument("--timeline-db", required=True)
    parser.add_argument("--profiles-db", required=True)
    parser.add_argument("--master-calls-db")
    parser.add_argument("--tenant-id", default="foton")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true")
    group.add_argument("--customer-id", action="append")
    group.add_argument("--phone")
    parser.add_argument("--show-phone", action="store_true", help="Print active profile fields for --phone with masked phone.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    options = CustomerProfileBuildOptions(
        timeline_db=Path(args.timeline_db),
        profiles_db=Path(args.profiles_db),
        master_calls_db=Path(args.master_calls_db) if args.master_calls_db else None,
        tenant_id=args.tenant_id,
        customer_ids=tuple(args.customer_id or ()),
        phone=args.phone,
    )
    if not args.all and not args.customer_id and not args.phone:
        raise SystemExit("Specify --all, --customer-id, or --phone")
    report = CustomerProfileBuilder(options).build()
    if args.show_phone and args.phone:
        with CustomerProfileSQLiteStore(options.profiles_db) as store:
            print(json.dumps({"phone": mask_phone(args.phone), "report": report}, ensure_ascii=False, indent=2))
            for profile_id in selected_profile_ids(options.profiles_db):
                safe_fields = [safe_field_preview(dict(row)) for row in store.active_fields(profile_id)]
                print(json.dumps({"profile_id": profile_id, "active_fields": safe_fields}, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def selected_profile_ids(profiles_db: Path) -> list[str]:
    import sqlite3

    con = sqlite3.connect(profiles_db)
    try:
        return [str(row[0]) for row in con.execute("SELECT profile_id FROM customer_profiles ORDER BY profile_id").fetchall()]
    finally:
        con.close()


def safe_field_preview(row: dict) -> dict:
    return {
        "field": row.get("field"),
        "has_value": bool(row.get("value")),
        "value_len": len(str(row.get("value") or "")),
        "child_key": row.get("child_key"),
        "brand": row.get("brand"),
        "event_at": row.get("event_at"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
