from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import AbstractSet, Any, Mapping, Sequence

from mango_mvp.customer_timeline.ids import stable_digest


CANONICAL_CALLS_SOURCE_KIND = "canonical_calls"
SUPPORTED_CALL_TABLES = frozenset({"canonical_calls", "call_records"})


@dataclass(frozen=True)
class CallSourceSnapshot:
    path: Path
    table: str
    columns: frozenset[str]
    ready_rows: tuple[Mapping[str, Any], ...]
    all_rows: tuple[Mapping[str, Any], ...]


def read_call_source_snapshot(
    path: Path,
    *,
    table: str | None = None,
    include_all_rows: bool = False,
    ready_columns: Sequence[str] | None = None,
    all_columns: Sequence[str] | None = None,
    require_analysis: bool | None = True,
    require_datetime: bool | None = True,
) -> CallSourceSnapshot:
    """Read one consistent, read-only snapshot shared by producer and derived readers."""

    db = Path(path).expanduser().resolve(strict=False)
    if not db.exists() or not db.is_file():
        raise FileNotFoundError(db)
    if table is not None and table not in SUPPORTED_CALL_TABLES:
        raise ValueError(f"unsupported calls table: {table}")
    with closing(sqlite3.connect(f"{db.as_uri()}?mode=ro", uri=True)) as con:
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA query_only = ON")
        con.execute("PRAGMA busy_timeout = 30000")
        con.execute("BEGIN")
        present = {
            str(row[0])
            for row in con.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name IN ('canonical_calls', 'call_records')"
            ).fetchall()
        }
        if table is None:
            if len(present) != 1:
                raise ValueError(
                    "calls DB must contain exactly one supported source table: canonical_calls or call_records"
                )
            selected = next(iter(present))
        else:
            selected = table
            if selected not in present:
                raise ValueError(f"required call table missing in {db}: {selected}")
        columns = frozenset(str(row[1]) for row in con.execute(f"PRAGMA table_info({selected})"))
        analysis_required = selected == "call_records" if require_analysis is None else require_analysis
        datetime_required = selected == "call_records" if require_datetime is None else require_datetime
        predicate = "1 = 1"
        if analysis_required:
            missing = {"analysis_status", "analysis_json"} - columns
            if missing:
                raise ValueError(f"required call columns missing in {db}: {sorted(missing)}")
            predicate = "analysis_status = 'done' AND analysis_json IS NOT NULL AND analysis_json != ''"
            invalid = con.execute(
                f"""
                SELECT rowid
                FROM {selected}
                WHERE {predicate}
                  AND (
                    json_valid(analysis_json) = 0
                    OR CASE
                         WHEN json_valid(analysis_json) = 1
                         THEN json_type(analysis_json) != 'object' OR json(analysis_json) = '{{}}'
                         ELSE 1
                       END
                  )
                LIMIT 1
                """
            ).fetchone()
            if invalid is not None:
                raise ValueError(f"invalid done analysis_json in {db}: rowid={invalid[0]}")
        ready_select = _select_clause(columns, ready_columns)
        ready_rows = tuple(
            dict(row)
            for row in con.execute(
                f"""
                SELECT {ready_select} FROM {selected}
                WHERE {predicate}
                """
            )
        )
        all_rows = (
            tuple(
                dict(row)
                for row in con.execute(
                    f"SELECT {_select_clause(columns, all_columns)} FROM {selected}"
                )
            )
            if include_all_rows
            else ()
        )
    for row in ready_rows:
        row_id = _first_text(row, "canonical_call_id", "id", "source_call_id", "source_filename")
        if not row_id:
            raise ValueError(f"missing done call id in {db}")
        if datetime_required and not _first_text(row, "started_at", "call_at", "event_at"):
            raise ValueError(f"missing done call datetime in {db}: {row_id}")
    return CallSourceSnapshot(
        path=db,
        table=selected,
        columns=columns,
        ready_rows=ready_rows,
        all_rows=all_rows,
    )


def _select_clause(columns: AbstractSet[str], requested: Sequence[str] | None) -> str:
    if requested is None:
        return "*"
    selected = tuple(dict.fromkeys(name for name in requested if name in columns))
    if not selected:
        raise ValueError("call source projection does not select any existing columns")
    return ", ".join(f'"{name}"' for name in selected)


def _first_text(row: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return None


def parse_call_started_at(value: str | datetime | None) -> datetime | None:
    """Parse provider call time; the legacy provider's naive values are UTC."""

    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value or "").strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def normalize_call_started_at(value: str | datetime | None) -> str:
    parsed = parse_call_started_at(value)
    return parsed.isoformat() if parsed is not None else str(value or "").strip()


def call_source_base_id(
    *,
    source_kind: str,
    row_id: str,
    source_call_id: str | None,
) -> str:
    """Return the exact source-id base shared by the call producer and readers."""

    if source_kind == CANONICAL_CALLS_SOURCE_KIND:
        return str(row_id)
    return f"provider:{source_call_id or row_id}"


def stable_call_source_id(
    *,
    source_kind: str,
    row_id: str,
    source_call_id: str | None,
    source_filename: str | None,
    started_at: str,
    duplicate_base_ids: AbstractSet[str],
) -> str:
    base = call_source_base_id(
        source_kind=source_kind,
        row_id=row_id,
        source_call_id=source_call_id,
    )
    if source_kind == CANONICAL_CALLS_SOURCE_KIND or base not in duplicate_base_ids:
        return base
    suffix = stable_digest({"source_filename": source_filename, "started_at": started_at})[:12]
    return f"{base}:{suffix}"


def build_call_lineage(
    *,
    source_kind: str,
    source_db: str,
    row_id: str,
    source_call_id: str | None,
    source_filename: str | None,
    started_at: str,
    duplicate_base_ids: AbstractSet[str],
    updated_at: str | None = None,
) -> dict[str, str | None]:
    source_id = stable_call_source_id(
        source_kind=source_kind,
        row_id=row_id,
        source_call_id=source_call_id,
        source_filename=source_filename,
        started_at=started_at,
        duplicate_base_ids=duplicate_base_ids,
    )
    normalized_at = normalize_call_started_at(started_at)
    normalized_updated_at = normalize_call_started_at(updated_at) if updated_at else normalized_at
    return {
        "call_id": source_id,
        "provider_call_id": source_id,
        "original_call_id": source_call_id or row_id,
        "source_ref": f"mango:{source_id}",
        "source_db": source_db,
        "source_row_id": row_id,
        "source_filename": source_filename,
        "call_at": normalized_at,
        "event_at": normalized_at,
        "updated_at": normalized_updated_at,
    }
