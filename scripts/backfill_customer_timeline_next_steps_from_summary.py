#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from mango_mvp.customer_timeline.ids import stable_digest
from mango_mvp.customer_timeline.next_step_resolver import extract_next_step_action
from mango_mvp.customer_timeline.store import json_dumps, json_loads, scrub_timeline_persisted_json


NEXT_STEP_SOURCE = "summary_extractor_v1"


@dataclass(frozen=True)
class BackfillReport:
    timeline_db: str
    applied: bool
    total_mango_calls: int
    eligible_mango_calls: int
    skipped_existing_structured: int
    extracted: int
    updated: int
    no_extraction: int
    examples: list[Mapping[str, Any]] = field(default_factory=list)

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill deterministic next_step fields from mango_call summary on a test-copy timeline DB."
    )
    parser.add_argument("--timeline-db", required=True, type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--report-out", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    db_path = args.timeline_db.expanduser().resolve()
    _guard_safe_db_path(db_path)
    report = backfill_next_steps_from_summary(db_path, apply=args.apply, limit=args.limit)
    text = json.dumps(report.to_json_dict(), ensure_ascii=False, indent=2, sort_keys=True)
    if args.report_out:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def backfill_next_steps_from_summary(db_path: Path, *, apply: bool, limit: int | None = None) -> BackfillReport:
    if not db_path.exists():
        raise FileNotFoundError(db_path)

    updated = extracted = skipped_existing = no_extraction = eligible = 0
    examples: list[Mapping[str, Any]] = []
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        total_calls = int(
            con.execute("SELECT COUNT(*) FROM timeline_events WHERE event_type = 'mango_call'").fetchone()[0]
        )
        sql = """
            SELECT event_id, record_json
            FROM timeline_events
            WHERE event_type = 'mango_call'
            ORDER BY event_at ASC, event_id ASC
        """
        params: list[Any] = []
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        rows = con.execute(sql, params).fetchall()

        for row in rows:
            event = dict(json_loads(row["record_json"]))
            if _has_structured_next_step(event):
                skipped_existing += 1
                continue
            eligible += 1
            action = extract_next_step_action(event)
            if not action:
                no_extraction += 1
                continue
            extracted += 1
            payload = _with_backfilled_next_step(event, action)
            record_hash = stable_digest(scrub_timeline_persisted_json(payload))
            if apply:
                con.execute(
                    """
                    UPDATE timeline_events
                    SET record_json = ?, record_hash = ?
                    WHERE event_id = ?
                    """,
                    (json_dumps(payload), record_hash, row["event_id"]),
                )
                updated += 1
            if len(examples) < 20:
                examples.append(
                    {
                        "event_id": row["event_id"],
                        "customer_id": event.get("customer_id"),
                        "event_at": event.get("event_at"),
                        "summary": event.get("summary"),
                        "extracted_next_step": action,
                    }
                )
        if apply:
            con.commit()

    return BackfillReport(
        timeline_db=str(db_path),
        applied=apply,
        total_mango_calls=total_calls,
        eligible_mango_calls=eligible,
        skipped_existing_structured=skipped_existing,
        extracted=extracted,
        updated=updated,
        no_extraction=no_extraction,
        examples=examples,
    )


def _guard_safe_db_path(db_path: Path) -> None:
    if "stable_runtime" in db_path.parts:
        raise SystemExit("Refusing to backfill a DB under stable_runtime; use an isolated test copy.")


def _has_structured_next_step(event: Mapping[str, Any]) -> bool:
    record = event.get("record") if isinstance(event.get("record"), Mapping) else {}
    call_analysis = record.get("call_analysis") if isinstance(record.get("call_analysis"), Mapping) else {}
    values = (
        call_analysis.get("next_step"),
        record.get("next_step"),
        record.get("recommended_action"),
        event.get("next_step"),
        event.get("recommended_action"),
    )
    return any(str(value or "").strip() for value in values)


def _with_backfilled_next_step(event: Mapping[str, Any], action: str) -> Mapping[str, Any]:
    payload = dict(event)
    record = dict(payload.get("record") if isinstance(payload.get("record"), Mapping) else {})
    record["next_step"] = action
    record["next_step_source"] = NEXT_STEP_SOURCE
    record["next_step_extracted_from"] = "summary"
    payload["record"] = record
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
