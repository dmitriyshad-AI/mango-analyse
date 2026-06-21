#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mango_mvp.customer_timeline.ids import stable_digest  # noqa: E402
from mango_mvp.customer_timeline.mail_stage2_ingest import file_sha256  # noqa: E402
from mango_mvp.customer_timeline.store import scrub_timeline_persisted_json  # noqa: E402


DEFAULT_DB = Path(
    "/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_timeline/"
    "customer_timeline_prod_20260621/customer_timeline.sqlite"
)
DEFAULT_STAGE2_FULL = Path(
    "/Users/dmitrijfabarisov/Projects/Mango analyse/_external_handoffs/mail_archive_2026-05-12/"
    "regru_edu/full_all_mail_combined_20260513/stage2_email_ingest_20260620/"
    "stage2_full_corpus_events.jsonl"
)
DEFAULT_STAGE2_DELTA = Path(
    "/Users/dmitrijfabarisov/Projects/Mango analyse/_external_handoffs/mail_archive_2026-06-20/"
    "regru_edu/incremental_20260513_to_20260620/stage2_delta_ingest_20260621/"
    "stage2_delta_full_events.jsonl"
)
DEFAULT_ARCHIVE_ROOTS = (
    Path("/Users/dmitrijfabarisov/Projects/Mango analyse/_external_handoffs/mail_archive_2026-05-12/regru_edu"),
    Path("/Users/dmitrijfabarisov/Projects/Mango analyse/_external_handoffs/mail_archive_2026-06-20/regru_edu"),
)
SOURCE_REF_RE = re.compile(r"^mail_stage2:([^:]+):(\d+):")


def parse_dt(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = parsedate_to_datetime(text)
        except (TypeError, ValueError, IndexError, OverflowError):
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def parse_stage2_event_date(event: Mapping[str, Any]) -> tuple[datetime | None, str]:
    for key in ("date_iso", "date_first", "date_last", "date", "message_date_iso"):
        parsed = parse_dt(event.get(key))
        if parsed is not None:
            return parsed, f"stage2_{key}"
    return None, ""


def load_stage2_events(source_refs: list[str], event_paths: tuple[Path, ...]) -> dict[str, Mapping[str, Any]]:
    by_file = {path.name: path for path in event_paths}
    wanted: dict[str, set[int]] = defaultdict(set)
    for source_ref in source_refs:
        match = SOURCE_REF_RE.match(source_ref)
        if not match:
            continue
        wanted[match.group(1)].add(int(match.group(2)))
    loaded: dict[str, Mapping[str, Any]] = {}
    for file_name, line_numbers in wanted.items():
        path = by_file.get(file_name)
        if path is None:
            continue
        remaining = set(line_numbers)
        with path.open(encoding="utf-8") as fh:
            for line_number, line in enumerate(fh, start=1):
                if line_number not in remaining:
                    continue
                source_ref_prefix = f"mail_stage2:{file_name}:{line_number}:"
                event = json.loads(line)
                loaded[source_ref_prefix] = event
                remaining.remove(line_number)
                if not remaining:
                    break
    result: dict[str, Mapping[str, Any]] = {}
    for source_ref in source_refs:
        match = SOURCE_REF_RE.match(source_ref)
        if not match:
            continue
        prefix = f"mail_stage2:{match.group(1)}:{int(match.group(2))}:"
        event = loaded.get(prefix)
        if event is not None:
            result[source_ref] = event
    return result


def load_archive_date_index(message_shas: set[str], archive_roots: tuple[Path, ...]) -> dict[str, tuple[datetime, str]]:
    result: dict[str, tuple[datetime, str]] = {}
    if not message_shas:
        return result
    archive_dbs: list[Path] = []
    for root in archive_roots:
        archive_dbs.extend(sorted(root.rglob("mail_archive.sqlite")))
    for db_path in archive_dbs:
        if len(result) == len(message_shas):
            break
        try:
            con = sqlite3.connect(db_path)
            con.row_factory = sqlite3.Row
            has_messages = con.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='messages'"
            ).fetchone()
            if not has_messages:
                con.close()
                continue
            placeholders = ",".join("?" for _ in message_shas)
            rows = con.execute(
                f"""
                SELECT sha256, message_date_iso, message_date_header, first_ingested_at, updated_at, mailbox
                FROM messages
                WHERE sha256 IN ({placeholders})
                """,
                tuple(message_shas),
            ).fetchall()
            con.close()
        except sqlite3.Error:
            continue
        for row in rows:
            sha = str(row["sha256"])
            if sha in result:
                continue
            for key, source in (
                ("message_date_iso", "archive_message_date_iso"),
                ("message_date_header", "archive_message_date_header"),
                ("first_ingested_at", "archive_first_ingested_at_no_message_date"),
                ("updated_at", "archive_updated_at_no_message_date"),
            ):
                parsed = parse_dt(row[key])
                if parsed is not None:
                    result[sha] = (parsed, source)
                    break
    return result


def json_dumps(payload: Mapping[str, Any]) -> str:
    safe = scrub_timeline_persisted_json(dict(payload))
    return json.dumps(safe, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def update_payload_date(payload: Mapping[str, Any], new_iso: str, source: str, repaired_at: str) -> tuple[str, str]:
    item = dict(payload)
    previous = str(item.get("event_at") or "")
    item["event_at"] = new_iso
    metadata = dict(item.get("metadata") or {})
    metadata["mail_stage2_date_repair"] = {
        "previous_event_at": previous,
        "source": source,
        "repaired_at": repaired_at,
    }
    item["metadata"] = metadata
    record = dict(item.get("record") or {})
    record["date_repair"] = metadata["mail_stage2_date_repair"]
    item["record"] = record
    safe = scrub_timeline_persisted_json(item)
    return json.dumps(safe, ensure_ascii=False, sort_keys=True, separators=(",", ":")), stable_digest(safe)


def create_backup(db_path: Path) -> Mapping[str, Any]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = db_path.parent / "backups" / f"before_mail_stage2_date_repair_{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    backup_db = backup_dir / db_path.name
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as source, sqlite3.connect(backup_db) as target:
        source.backup(target)
    manifest = {
        "kind": "timeline_backup",
        "purpose": "mail_stage2_date_repair",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_db_path": str(db_path),
        "backup_db_path": str(backup_db),
        "backup_sha256": file_sha256(backup_db),
        "source_sha256_after_backup": file_sha256(db_path),
    }
    manifest_path = backup_dir / "backup_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return {**manifest, "manifest_path": str(manifest_path)}


def repair_dates(
    *,
    db_path: Path,
    event_paths: tuple[Path, ...],
    archive_roots: tuple[Path, ...],
    dry_run: bool,
) -> Mapping[str, Any]:
    repaired_at = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            """
            SELECT event_id, source_id, source_ref, opportunity_id, event_at, record_json
            FROM timeline_events
            WHERE source_system = 'mail_archive_stage2'
              AND event_at <= '1970-01-02'
            ORDER BY source_ref
            """
        ).fetchall()
    source_refs = [str(row["source_ref"]) for row in rows]
    events = load_stage2_events(source_refs, event_paths)
    archive_dates = load_archive_date_index({str(row["source_id"]) for row in rows}, archive_roots)
    planned: list[dict[str, Any]] = []
    counts: Counter[str] = Counter({"candidate_events": len(rows)})
    for row in rows:
        event = events.get(str(row["source_ref"]))
        parsed = None
        source = ""
        if event is not None:
            parsed, source = parse_stage2_event_date(event)
        if parsed is None:
            archive_item = archive_dates.get(str(row["source_id"]))
            if archive_item is not None:
                parsed, source = archive_item
        if parsed is None:
            counts["unresolved"] += 1
            continue
        new_iso = parsed.isoformat()
        if new_iso == row["event_at"]:
            counts["already_ok"] += 1
            continue
        planned.append(
            {
                "event_id": row["event_id"],
                "source_id": row["source_id"],
                "source_ref": row["source_ref"],
                "opportunity_id": row["opportunity_id"],
                "previous_event_at": row["event_at"],
                "new_event_at": new_iso,
                "date_source": source,
                "record_json": row["record_json"],
            }
        )
        counts[f"source_{source}"] += 1
    report: dict[str, Any] = {
        "schema_version": "mail_stage2_date_repair_v1",
        "mode": "dry_run" if dry_run else "apply",
        "created_at": repaired_at,
        "timeline_db": str(db_path),
        "event_paths": [str(path) for path in event_paths],
        "archive_roots": [str(path) for path in archive_roots],
        "counts": {**counts, "planned_updates": len(planned)},
        "sample_updates": [
            {key: item[key] for key in ("event_id", "source_ref", "previous_event_at", "new_event_at", "date_source")}
            for item in planned[:20]
        ],
    }
    if dry_run:
        return report
    backup = create_backup(db_path)
    report["backup"] = backup
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        con.execute("BEGIN")
        for item in planned:
            payload = json.loads(str(item["record_json"]))
            payload_json, record_hash = update_payload_date(
                payload,
                str(item["new_event_at"]),
                str(item["date_source"]),
                repaired_at,
            )
            con.execute(
                """
                UPDATE timeline_events
                SET event_at = ?, record_json = ?, record_hash = ?
                WHERE event_id = ?
                """,
                (item["new_event_at"], payload_json, record_hash, item["event_id"]),
            )
            con.execute(
                "UPDATE bot_context_chunks SET event_at = ? WHERE event_id = ?",
                (item["new_event_at"], item["event_id"]),
            )
        con.execute(
            """
            UPDATE customer_opportunities
            SET opened_at = (
                SELECT MIN(e.event_at)
                FROM timeline_events e
                WHERE e.opportunity_id = customer_opportunities.opportunity_id
                  AND e.source_system = 'mail_archive_stage2'
            )
            WHERE source_system = 'mail_archive_stage2'
              AND opportunity_id IN (
                SELECT DISTINCT opportunity_id
                FROM timeline_events
                WHERE source_system = 'mail_archive_stage2'
                  AND opportunity_id IS NOT NULL
              )
            """
        )
        remaining_1970 = int(
            con.execute(
                """
                SELECT count(*)
                FROM timeline_events
                WHERE source_system = 'mail_archive_stage2'
                  AND event_at <= '1970-01-02'
                """
            ).fetchone()[0]
        )
        unsafe_chunks = int(
            con.execute(
                """
                SELECT count(*)
                FROM bot_context_chunks
                WHERE source_system = 'mail_archive_stage2'
                  AND (allowed_for_bot != 0 OR requires_manager_review != 1)
                """
            ).fetchone()[0]
        )
        con.commit()
    report["validation"] = {
        "remaining_1970_mail_stage2_events": remaining_1970,
        "unsafe_bot_context_chunks": unsafe_chunks,
        "pass": remaining_1970 == 0 and unsafe_chunks == 0,
    }
    report_path = db_path.parent / "mail_stage2_date_repair_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair 1970 event_at values for mail_stage2 timeline events.")
    parser.add_argument("--timeline-db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--event-jsonl", type=Path, action="append")
    parser.add_argument("--archive-root", type=Path, action="append")
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    event_paths = tuple(args.event_jsonl or (DEFAULT_STAGE2_FULL, DEFAULT_STAGE2_DELTA))
    archive_roots = tuple(args.archive_root or DEFAULT_ARCHIVE_ROOTS)
    report = repair_dates(
        db_path=args.timeline_db.expanduser().resolve(strict=False),
        event_paths=tuple(path.expanduser().resolve(strict=False) for path in event_paths),
        archive_roots=tuple(path.expanduser().resolve(strict=False) for path in archive_roots),
        dry_run=not args.apply,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.get("validation", {}).get("pass", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
