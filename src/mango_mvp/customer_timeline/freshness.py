from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping


_CURSOR_GROUPS = {
    "amocrm_snapshot": ("amo_leads_updated_at", "amo_contacts_updated_at"),
    "amocrm_event": ("amo_events_created_at",),
    "tallanto_snapshot": ("tallanto_cards_daily",),
    "tallanto_attendance_api": ("tallanto_attendance_api",),
}
_REQUIRED_DATA_BOUNDARY = {"tallanto_snapshot", "tallanto_crm_call", "tallanto_attendance_api"}
MANAGER_REQUIRED_SOURCE_SYSTEMS = (
    "amocrm_snapshot",
    "amocrm_event",
    "mail_archive_stage2",
    "mango_processed_summary",
    "tallanto_snapshot",
    "tallanto_crm_call",
    "tallanto_attendance_api",
    "wappi_telegram",
    "wappi_max",
)


def source_freshness_rows(
    con: sqlite3.Connection,
    *,
    tenant_id: str = "foton",
    expected_sources: tuple[str, ...] = (),
) -> list[Mapping[str, Any]]:
    """Return data boundary, import time and event time as separate facts."""
    event_rows = con.execute(
        """
        SELECT source_system, MAX(event_at) AS max_event_at, COUNT(*) AS events
        FROM timeline_events
        WHERE tenant_id = ?
        GROUP BY source_system
        """,
        (tenant_id,),
    ).fetchall()
    cursor_rows = []
    if _table_exists(con, "ingestion_cursors"):
        cursor_rows = con.execute(
            """
            SELECT source_system, last_cursor_ts, updated_at
            FROM ingestion_cursors
            WHERE tenant_id = ?
            """,
            (tenant_id,),
        ).fetchall()
    cursors = {str(row["source_system"]): dict(row) for row in cursor_rows}
    successful_imports: dict[str, str] = {}
    if _table_exists(con, "ingestion_runs"):
        payment_proof = (
            "(source_system != 'tallanto_crm_call' OR source_ref = 'tallanto_api:get_entry_list')"
            if _column_exists(con, "ingestion_runs", "source_ref")
            else "source_system != 'tallanto_crm_call'"
        )
        successful_imports = {
            str(row["source_system"]): str(row["imported_at"])
            for row in con.execute(
                f"""
                SELECT source_system, MAX(finished_at) AS imported_at
                FROM ingestion_runs
                WHERE tenant_id = ?
                  AND run_kind IN ('timeline_import', 'tallanto_attendance_api_increment')
                  AND status = 'completed'
                  AND finished_at IS NOT NULL
                  AND {payment_proof}
                GROUP BY source_system
                """,
                (tenant_id,),
            ).fetchall()
        }

    result: list[Mapping[str, Any]] = []
    event_by_source = {str(row["source_system"]): row for row in event_rows}
    cursor_aliases = {
        cursor_name: source_system
        for source_system, cursor_names in _CURSOR_GROUPS.items()
        for cursor_name in cursor_names
    }
    source_systems = set(event_by_source) | set(successful_imports)
    source_systems.update(cursor_aliases.get(name, name) for name in cursors)
    source_systems.update(expected_sources)
    for source_system in source_systems:
        event_row = event_by_source.get(source_system)
        preferred = _CURSOR_GROUPS.get(source_system, (source_system,))
        selected = [cursors[name] for name in preferred if name in cursors]
        cursor_complete = len(selected) == len(preferred)
        if not selected and source_system in cursors:
            selected = [cursors[source_system]]
            cursor_complete = True
        imported_candidates = [
            successful_imports[name]
            for name in (source_system, *preferred)
            if name in successful_imports
        ]
        result.append(
            {
                "source_system": source_system,
                "cursor_at": min((str(row["last_cursor_ts"]) for row in selected), default=None),
                "cursor_updated_at": min((str(row["updated_at"]) for row in selected), default=None),
                "imported_at": max(imported_candidates, default=None),
                "max_event_at": event_row["max_event_at"] if event_row is not None else None,
                "events": int(event_row["events"] or 0) if event_row is not None else 0,
                "cursor_complete": cursor_complete,
                "cursor_sources": [str(row["source_system"]) for row in selected],
                "expected": source_system in expected_sources,
                "missing": event_row is None and not selected and source_system not in successful_imports,
            }
        )
    return sorted(
        result,
        key=lambda row: (str(row.get("cursor_at") or ""), str(row.get("max_event_at") or ""), str(row["source_system"])),
        reverse=True,
    )


def manager_freshness_gate(
    rows: list[Mapping[str, Any]],
    *,
    now: datetime | None = None,
    max_import_age_hours: float = 36.0,
    max_cursor_check_age_hours: float = 36.0,
) -> Mapping[str, Any]:
    checked_at = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    blockers: list[Mapping[str, str]] = []
    for row in rows:
        if not row.get("expected"):
            continue
        source = str(row.get("source_system") or "unknown")
        if row.get("missing"):
            blockers.append({"source_system": source, "reason": "missing"})
            continue
        if source in _CURSOR_GROUPS and not row.get("cursor_complete"):
            blockers.append({"source_system": source, "reason": "cursor_incomplete"})
        elif source in _CURSOR_GROUPS:
            cursor_updated_at = _parse_datetime(row.get("cursor_updated_at"))
            if cursor_updated_at is None:
                blockers.append({"source_system": source, "reason": "cursor_check_missing"})
            elif cursor_updated_at > checked_at + timedelta(minutes=5):
                blockers.append({"source_system": source, "reason": "cursor_check_in_future"})
            elif checked_at - cursor_updated_at > timedelta(hours=max_cursor_check_age_hours):
                blockers.append({"source_system": source, "reason": "cursor_check_stale"})
        imported_at = _parse_datetime(row.get("imported_at"))
        if imported_at is None:
            blockers.append({"source_system": source, "reason": "successful_import_missing"})
            continue
        if imported_at > checked_at + timedelta(minutes=5):
            blockers.append({"source_system": source, "reason": "imported_at_in_future"})
        elif checked_at - imported_at > timedelta(hours=max_import_age_hours):
            blockers.append({"source_system": source, "reason": "successful_import_stale"})
        max_event_at = _parse_datetime(row.get("max_event_at"))
        if max_event_at is not None and max_event_at > checked_at + timedelta(minutes=5):
            blockers.append({"source_system": source, "reason": "max_event_at_in_future"})
        if source in _REQUIRED_DATA_BOUNDARY and max_event_at is None:
            blockers.append({"source_system": source, "reason": "data_boundary_missing"})
    return {
        "passed": not blockers,
        "checked_at": checked_at.isoformat(),
        "max_import_age_hours": max_import_age_hours,
        "max_cursor_check_age_hours": max_cursor_check_age_hours,
        "blockers": blockers,
    }


def _parse_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _table_exists(con: sqlite3.Connection, table: str) -> bool:
    return con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone() is not None


def _column_exists(con: sqlite3.Connection, table: str, column: str) -> bool:
    return any(str(row[1]) == column for row in con.execute(f"PRAGMA table_info({table})"))


__all__ = ["MANAGER_REQUIRED_SOURCE_SYSTEMS", "manager_freshness_gate", "source_freshness_rows"]
