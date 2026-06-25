#!/usr/bin/env python3
"""Build read-only Mango call increment JSONL for customer_timeline."""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from mango_mvp.customer_timeline.contracts import IdentityMatchClass
from mango_mvp.customer_timeline.ids import stable_digest
from mango_mvp.utils.phone import normalize_phone


MANGO_INCREMENT_PRODUCER_SCHEMA_VERSION = "mango_call_timeline_increment_v1"
MANGO_SOURCE_SYSTEM = "mango_processed_summary"
MANGO_EVENT_TYPE = "mango_call"
DEFAULT_TENANT_ID = "foton"
HISTORY_CALL_TYPES = {"sales_call", "existing_client_progress", "technical_call"}
UNSAFE_LINK_CLASSES = {
    IdentityMatchClass.AMBIGUOUS.value,
    IdentityMatchClass.DUPLICATE.value,
    IdentityMatchClass.UNMATCHED.value,
}


@dataclass(frozen=True)
class IdentityResolution:
    match_class: str
    customer_id: str | None
    reason: str
    candidate_count: int


@dataclass(frozen=True)
class SourceRow:
    source_kind: str
    source_db: str
    row_id: str
    source_call_id: str | None
    source_filename: str | None
    source_file: str | None
    started_at: str
    phone: str | None
    manager_name: str | None
    direction: str | None
    duration_sec: float | None
    analysis_json: str
    amocrm_contact_id: str | None = None
    amocrm_lead_id: str | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Mango call increment JSONL from already analyzed local DBs.")
    parser.add_argument("--timeline-db", required=True, help="Existing customer_timeline DB, opened read-only for identity links.")
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--report-out", required=True)
    parser.add_argument("--tenant-id", default=DEFAULT_TENANT_ID)
    parser.add_argument("--canonical-db", action="append", default=[], help="Read canonical_calls from this DB. Can repeat.")
    parser.add_argument("--package-root", action="append", default=[], help="Read package-local call_records DB under this root. Can repeat.")
    parser.add_argument("--package-db", action="append", default=[], help="Read package-local call_records from this DB. Can repeat.")
    parser.add_argument("--since")
    parser.add_argument("--until")
    parser.add_argument("--limit", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.monotonic()
    since = parse_optional_datetime(args.since)
    until = parse_optional_datetime(args.until)
    out_jsonl = Path(args.out_jsonl)
    report_out = Path(args.report_out)
    rows: list[SourceRow] = []
    for db in args.canonical_db:
        rows.extend(read_ready_call_rows(Path(db), table="canonical_calls", source_kind="canonical_calls"))
    duplicate_base_ids: set[str] = set()
    package_dbs = [Path(item) for item in args.package_db]
    for root in args.package_root:
        package_dbs.extend(discover_package_call_dbs(Path(root)))
    for db in package_dbs:
        rows.extend(read_ready_call_rows(db, table="call_records", source_kind="call_records"))
        duplicate_base_ids.update(
            read_duplicate_source_ids(db, table="call_records", source_kind="call_records", since=since, until=until)
        )

    all_filtered = filter_rows(rows, since=since, until=until)
    duplicate_base_ids.update(duplicate_source_ids(all_filtered))
    filtered = list(all_filtered)
    filtered.sort(key=lambda item: parse_source_datetime(item.started_at) or datetime.min.replace(tzinfo=timezone.utc))
    if args.limit is not None:
        filtered = filtered[: max(0, args.limit)]
    events: list[Mapping[str, Any]] = []
    resolution_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    call_type_counts: Counter[str] = Counter()
    examples: list[Mapping[str, Any]] = []
    with open_timeline_ro(Path(args.timeline_db)) as timeline:
        for row in filtered:
            analysis = parse_json_object(row.analysis_json)
            if not analysis:
                continue
            call_type = analysis_call_type(analysis)
            call_type_counts[call_type or "unknown"] += 1
            resolution = resolve_phone_identity(timeline, args.tenant_id, row.phone)
            resolution_counts[resolution.match_class] += 1
            source_counts[row.source_kind] += 1
            event = build_event_payload(
                row,
                analysis,
                tenant_id=args.tenant_id,
                resolution=resolution,
                duplicate_base_ids=duplicate_base_ids,
                call_type=call_type,
            )
            events.append(event)
            if len(examples) < 10:
                examples.append(
                    {
                        "source_kind": row.source_kind,
                        "source_id": event["call_id"],
                        "started_at": event["call_at"],
                        "phone_masked": mask_phone(row.phone),
                        "match_class": resolution.match_class,
                        "customer_id_present": bool(resolution.customer_id),
                        "call_type": call_type,
                    }
                )

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
    report = {
        "schema_version": MANGO_INCREMENT_PRODUCER_SCHEMA_VERSION,
        "safety": {
            "read_only_sources": True,
            "writes_amo": False,
            "writes_tallanto": False,
            "writes_crm": False,
            "runs_asr": False,
            "runs_analyze": False,
            "writes_timeline_db": False,
        },
        "inputs": {
            "timeline_db": str(Path(args.timeline_db)),
            "canonical_db": [str(Path(item)) for item in args.canonical_db],
            "package_root": [str(Path(item)) for item in args.package_root],
            "package_db": [str(item) for item in package_dbs],
            "since": args.since,
            "until": args.until,
            "limit": args.limit,
        },
        "output_jsonl": str(out_jsonl),
        "rows_read": len(rows),
        "rows_selected": len(filtered),
        "events_written": len(events),
        "source_counts": dict(source_counts),
        "identity_resolution_counts": dict(resolution_counts),
        "call_type_counts": dict(call_type_counts),
        "examples": examples,
        "duration_seconds": round(time.monotonic() - started, 3),
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def discover_package_call_dbs(root: Path) -> list[Path]:
    candidates: list[Path] = []
    summary = root / "RA_FINAL_SUMMARY.json"
    if summary.exists():
        payload = parse_json_object(summary.read_text(encoding="utf-8"))
        db_path = payload.get("database") or payload.get("db_path")
        if db_path:
            candidates.append(Path(str(db_path)))
    candidates.extend(sorted(root.glob("asr_ui_batch/*.sqlite")))
    result: list[Path] = []
    seen: set[Path] = set()
    for item in candidates:
        resolved = item.expanduser().resolve(strict=False)
        if resolved.exists() and resolved not in seen:
            result.append(resolved)
            seen.add(resolved)
    return result


def read_ready_call_rows(path: Path, *, table: str, source_kind: str) -> list[SourceRow]:
    if not path.exists():
        raise FileNotFoundError(path)
    with sqlite3.connect(ro_uri(path), uri=True) as con:
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA query_only = ON")
        if not table_exists(con, table):
            return []
        cols = table_columns(con, table)
        if "analysis_json" not in cols:
            return []
        query = f"SELECT * FROM {table} WHERE analysis_status = 'done' AND analysis_json IS NOT NULL AND analysis_json != ''"
        result: list[SourceRow] = []
        for raw in con.execute(query):
            row = dict(raw)
            analysis = parse_json_object(str(row.get("analysis_json") or ""))
            if not analysis:
                continue
            started_at = first_text(row, "started_at", "call_at", "event_at")
            if not started_at:
                continue
            row_id = first_text(row, "canonical_call_id", "id", "source_call_id", "source_filename")
            if not row_id:
                continue
            result.append(
                SourceRow(
                    source_kind=source_kind,
                    source_db=str(path),
                    row_id=row_id,
                    source_call_id=first_text(row, "source_call_id"),
                    source_filename=first_text(row, "source_filename"),
                    source_file=first_text(row, "source_file"),
                    started_at=started_at,
                    phone=first_text(row, "phone", "client_phone", "normalized_phone", "Телефон клиента"),
                    manager_name=first_text(row, "manager_name", "Менеджер"),
                    direction=first_text(row, "direction", "Направление звонка"),
                    duration_sec=float_or_none(row.get("duration_sec") or row.get("Длительность, сек")),
                    analysis_json=str(row.get("analysis_json") or ""),
                    amocrm_contact_id=first_text(row, "amocrm_contact_id"),
                    amocrm_lead_id=first_text(row, "amocrm_lead_id"),
                )
            )
        return result


def read_duplicate_source_ids(
    path: Path,
    *,
    table: str,
    source_kind: str,
    since: datetime | None,
    until: datetime | None,
) -> set[str]:
    if source_kind == "canonical_calls" or not path.exists():
        return set()
    with sqlite3.connect(ro_uri(path), uri=True) as con:
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA query_only = ON")
        if not table_exists(con, table):
            return set()
        rows: list[SourceRow] = []
        for raw in con.execute(f"SELECT * FROM {table}"):
            row = dict(raw)
            started_at = first_text(row, "started_at", "call_at", "event_at")
            if not started_at:
                continue
            row_id = first_text(row, "canonical_call_id", "id", "source_call_id", "source_filename")
            if not row_id:
                continue
            rows.append(
                SourceRow(
                    source_kind=source_kind,
                    source_db=str(path),
                    row_id=row_id,
                    source_call_id=first_text(row, "source_call_id"),
                    source_filename=first_text(row, "source_filename"),
                    source_file=first_text(row, "source_file"),
                    started_at=started_at,
                    phone=first_text(row, "phone", "client_phone", "normalized_phone", "Телефон клиента"),
                    manager_name=first_text(row, "manager_name", "Менеджер"),
                    direction=first_text(row, "direction", "Направление звонка"),
                    duration_sec=float_or_none(row.get("duration_sec") or row.get("Длительность, сек")),
                    analysis_json=str(row.get("analysis_json") or ""),
                    amocrm_contact_id=first_text(row, "amocrm_contact_id"),
                    amocrm_lead_id=first_text(row, "amocrm_lead_id"),
                )
            )
    return duplicate_source_ids(filter_rows(rows, since=since, until=until))


def build_event_payload(
    row: SourceRow,
    analysis: Mapping[str, Any],
    *,
    tenant_id: str,
    resolution: IdentityResolution,
    duplicate_base_ids: set[str],
    call_type: str,
) -> Mapping[str, Any]:
    source_id = stable_call_source_id(row, duplicate_base_ids=duplicate_base_ids)
    summary = "" if call_type == "non_conversation" else text_value(analysis.get("history_summary") or analysis.get("summary"))
    payload: dict[str, Any] = {
        "source_system": MANGO_SOURCE_SYSTEM,
        "event_type": MANGO_EVENT_TYPE,
        "tenant_id": tenant_id,
        "call_id": source_id,
        "provider_call_id": source_id,
        "original_call_id": row.source_call_id or row.row_id,
        "source_ref": f"mango:{source_id}",
        "source_db": row.source_db,
        "source_row_id": row.row_id,
        "source_filename": row.source_filename,
        "source_file": row.source_file,
        "phone": normalize_phone(row.phone or ""),
        "call_at": normalize_datetime_text(row.started_at),
        "event_at": normalize_datetime_text(row.started_at),
        "updated_at": normalize_datetime_text(row.started_at),
        "manager_name": row.manager_name,
        "direction": normalize_direction(row.direction),
        "duration_sec": row.duration_sec,
        "summary": summary,
        "analysis_summary": summary,
        "call_type": call_type,
        "analysis_json": analysis,
        "amocrm_contact_id": row.amocrm_contact_id,
        "amocrm_lead_id": row.amocrm_lead_id,
        "identity_authority": "existing_timeline_increment",
        "identity_resolved_by_increment": True,
        "match_class": resolution.match_class,
        "identity_resolution_reason": resolution.reason,
        "identity_candidate_count": resolution.candidate_count,
        "allowed_for_bot": False,
        "requires_manager_review": True,
        "confidence": 0.95 if resolution.customer_id else 0.55,
    }
    if resolution.customer_id:
        payload["customer_id"] = resolution.customer_id
        payload["resolved_customer_id"] = resolution.customer_id
    return payload


def resolve_phone_identity(con: sqlite3.Connection, tenant_id: str, phone: str | None) -> IdentityResolution:
    normalized_phone = normalize_phone(phone or "")
    if not normalized_phone:
        return IdentityResolution(IdentityMatchClass.UNMATCHED.value, None, "missing_phone", 0)
    rows = con.execute(
        """
        SELECT customer_id, match_class
        FROM identity_links
        WHERE tenant_id = ?
          AND link_type IN ('phone', 'mango_client_phone')
          AND link_value = ?
        """,
        (tenant_id, normalized_phone),
    ).fetchall()
    customer_ids = sorted({str(row["customer_id"]) for row in rows if row["customer_id"]})
    classes = {str(row["match_class"] or "").strip() for row in rows}
    if not rows:
        return IdentityResolution(IdentityMatchClass.UNMATCHED.value, None, "no_identity_link", 0)
    if len(customer_ids) == 1 and not (classes & UNSAFE_LINK_CLASSES):
        return IdentityResolution(IdentityMatchClass.STRONG_UNIQUE.value, customer_ids[0], "single_existing_customer", 1)
    reason = "multiple_existing_customers" if len(customer_ids) > 1 else "unsafe_existing_link_class"
    return IdentityResolution(IdentityMatchClass.AMBIGUOUS.value, None, reason, len(customer_ids))


def stable_call_source_id(row: SourceRow, *, duplicate_base_ids: set[str]) -> str:
    if row.source_kind == "canonical_calls":
        return str(row.row_id)
    base = f"provider:{row.source_call_id or row.row_id}"
    if base not in duplicate_base_ids:
        return base
    suffix = stable_digest({"source_filename": row.source_filename, "started_at": row.started_at})[:12]
    return f"{base}:{suffix}"


def duplicate_source_ids(rows: Sequence[SourceRow]) -> set[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        if row.source_kind == "canonical_calls":
            continue
        counts[f"provider:{row.source_call_id or row.row_id}"] += 1
    return {source_id for source_id, count in counts.items() if count > 1}


def filter_rows(rows: Iterable[SourceRow], *, since: datetime | None, until: datetime | None) -> list[SourceRow]:
    result: list[SourceRow] = []
    for row in rows:
        started = parse_source_datetime(row.started_at)
        if started is None:
            continue
        if since is not None and started < since:
            continue
        if until is not None and started >= until:
            continue
        result.append(row)
    return result


def analysis_call_type(analysis: Mapping[str, Any]) -> str:
    quality_current = analysis.get("call_quality_current")
    if isinstance(quality_current, Mapping):
        value = text_value(quality_current.get("call_type"))
        if value:
            return value
    quality_flags = analysis.get("quality_flags")
    if isinstance(quality_flags, Mapping):
        value = text_value(quality_flags.get("call_type"))
        if value:
            return value
    return text_value(analysis.get("call_type")) or "unknown"


def normalize_direction(value: str | None) -> str:
    text = text_value(value).lower()
    if text in {"outbound", "out", "исходящий"}:
        return "outbound"
    if text in {"inbound", "in", "входящий"}:
        return "inbound"
    return text or "inbound"


def open_timeline_ro(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(ro_uri(path), uri=True)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA query_only = ON")
    return con


def ro_uri(path: Path) -> str:
    return path.expanduser().resolve(strict=False).as_uri() + "?mode=ro"


def table_exists(con: sqlite3.Connection, table: str) -> bool:
    row = con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
    return row is not None


def table_columns(con: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in con.execute(f"PRAGMA table_info({table})")}


def first_text(row: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = row.get(key)
        text = text_value(value)
        if text:
            return text
    return None


def text_value(value: Any) -> str:
    return str(value or "").strip()


def float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_json_object(raw: str) -> Mapping[str, Any]:
    try:
        parsed = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, Mapping) else {}


def parse_optional_datetime(raw: str | None) -> datetime | None:
    return parse_source_datetime(raw) if raw else None


def parse_source_datetime(raw: str | None) -> datetime | None:
    text = text_value(raw)
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def normalize_datetime_text(raw: str) -> str:
    parsed = parse_source_datetime(raw)
    if parsed is None:
        return text_value(raw)
    return parsed.isoformat()


def mask_phone(phone: str | None) -> str | None:
    normalized = normalize_phone(phone or "")
    if not normalized:
        return None
    return normalized[:3] + "***" + normalized[-2:]


if __name__ == "__main__":
    raise SystemExit(main())
