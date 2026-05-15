#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mango_mvp.deal_aware.amo_rollback import (  # noqa: E402
    ROLLBACK_CONFIRMATION,
    RetryPolicy,
    load_snapshot_rows,
    load_successful_rollback_keys,
    rollback_summary,
    run_rollback,
    write_rollback_outputs,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rollback deal-aware AMO lead AI fields from a pre-write snapshot.")
    parser.add_argument("--live-run-root", required=True)
    parser.add_argument("--pre-write-snapshot", default="")
    parser.add_argument("--live-report", default="")
    parser.add_argument("--field-catalog-cache", default="")
    parser.add_argument("--rollback-confirmation", default="")
    parser.add_argument("--max-rollback-rows", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--delay-ms", type=int, default=750)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--resume-from-report", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    live_run_root = Path(args.live_run_root).expanduser().resolve()
    snapshot_path = Path(args.pre_write_snapshot).expanduser().resolve() if args.pre_write_snapshot else live_run_root / "pre_write_snapshot.jsonl"
    resume_path = Path(args.resume_from_report).expanduser().resolve() if args.resume_from_report else None
    apply = bool(args.apply)
    if args.dry_run and args.apply:
        print("Refusing rollback: choose either --dry-run or --apply, not both.", file=sys.stderr)
        return 2
    if apply and args.rollback_confirmation != ROLLBACK_CONFIRMATION:
        print(f"Refusing rollback apply: --rollback-confirmation must be {ROLLBACK_CONFIRMATION!r}.", file=sys.stderr)
        return 2

    snapshot_rows = load_snapshot_rows(snapshot_path)
    resume_keys = load_successful_rollback_keys(resume_path)

    from scripts.write_amo_ready_contacts import _load_env_files, _preflight_runtime_db  # noqa: PLC0415

    _load_env_files()
    from mango_mvp.amocrm_runtime.amo_integration import fetch_lead, send_lead_custom_field_update  # noqa: PLC0415
    from mango_mvp.amocrm_runtime.db import SessionLocal  # noqa: PLC0415

    session = SessionLocal()
    try:
        ok, error = _preflight_runtime_db(session)
        if not ok:
            summary = {
                "schema_version": "deal_aware_amo_rollback_report_v1",
                "apply": apply,
                "snapshot_path": str(snapshot_path),
                "preflight_failed": True,
                "preflight_error": error,
                "evaluated_rows": 0,
                "status_counts": {},
            }
            write_rollback_outputs(live_run_root, rows=[], summary=summary, apply=apply)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 2

        def fetcher(lead_id: int) -> dict:
            return fetch_lead(session, lead_id=lead_id)

        def updater(*, lead_id: int, field_payload: dict) -> dict:
            return send_lead_custom_field_update(session, lead_id=lead_id, field_payload=field_payload)

        def progress_writer(current_rows: list[dict]) -> None:
            current_summary = rollback_summary(
                rows=current_rows,
                snapshot_path=snapshot_path,
                apply=apply,
                max_rollback_rows=args.max_rollback_rows,
            )
            current_summary["batch_size"] = args.batch_size
            current_summary["delay_ms"] = args.delay_ms
            current_summary["max_retries"] = args.max_retries
            write_rollback_outputs(live_run_root, rows=current_rows, summary=current_summary, apply=apply)

        rows = run_rollback(
            snapshot_rows=snapshot_rows,
            fetch_lead=fetcher,
            send_update=updater,
            apply=apply,
            confirmation=args.rollback_confirmation,
            max_rows=args.max_rollback_rows,
            batch_size=args.batch_size,
            retry_policy=RetryPolicy(max_retries=args.max_retries, delay_ms=args.delay_ms),
            resume_success_keys=resume_keys,
            progress_writer=progress_writer,
        )
        if apply:
            session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    summary = rollback_summary(rows=rows, snapshot_path=snapshot_path, apply=apply, max_rollback_rows=args.max_rollback_rows)
    summary["batch_size"] = args.batch_size
    summary["delay_ms"] = args.delay_ms
    summary["max_retries"] = args.max_retries
    summary["live_report"] = str(Path(args.live_report).expanduser().resolve()) if args.live_report else ""
    summary["field_catalog_cache"] = str(Path(args.field_catalog_cache).expanduser().resolve()) if args.field_catalog_cache else ""
    write_rollback_outputs(live_run_root, rows=rows, summary=summary, apply=apply)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary.get("failed", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
