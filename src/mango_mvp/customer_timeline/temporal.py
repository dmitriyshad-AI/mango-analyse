from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence


def parse_aware_utc(value: Any) -> datetime | None:
    """Parse only timestamps carrying an explicit timezone."""

    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(
                f"{text[:-1]}+00:00" if text.endswith("Z") else text
            )
        except (TypeError, ValueError):
            return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(timezone.utc)


def normalize_aware_utc(value: datetime, field_name: str = "as_of") -> datetime:
    parsed = parse_aware_utc(value)
    if parsed is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return parsed


def semantic_record_at_or_before(
    record: Mapping[str, Any],
    timestamp_fields: Sequence[str],
    *,
    as_of: datetime,
    allow_missing: bool = True,
) -> bool:
    """Apply one strict cutoff; present invalid or naive timestamps fail closed."""

    cutoff = normalize_aware_utc(as_of)
    for field in timestamp_fields:
        value = record.get(field)
        if value in (None, ""):
            if allow_missing:
                continue
            return False
        parsed = parse_aware_utc(value)
        if parsed is None or parsed > cutoff:
            return False
    return True


def register_temporal_sql_functions(con: sqlite3.Connection) -> None:
    """Register deterministic UTC predicates used before SQL ordering and limits."""

    def cutoff_pair(value: Any, cutoff: Any) -> tuple[datetime | None, datetime | None]:
        return parse_aware_utc(value), parse_aware_utc(cutoff)

    def at_or_before(value: Any, cutoff: Any) -> int:
        parsed, parsed_cutoff = cutoff_pair(value, cutoff)
        return int(parsed is not None and parsed_cutoff is not None and parsed <= parsed_cutoff)

    def at_or_after(value: Any, cutoff: Any) -> int:
        parsed, parsed_cutoff = cutoff_pair(value, cutoff)
        return int(parsed is not None and parsed_cutoff is not None and parsed >= parsed_cutoff)

    def after(value: Any, cutoff: Any) -> int:
        parsed, parsed_cutoff = cutoff_pair(value, cutoff)
        return int(parsed is not None and parsed_cutoff is not None and parsed > parsed_cutoff)

    def epoch(value: Any) -> float | None:
        parsed = parse_aware_utc(value)
        return parsed.timestamp() if parsed is not None else None

    def conflict_created_by(value: Any, cutoff: Any) -> int:
        parsed, parsed_cutoff = cutoff_pair(value, cutoff)
        if parsed_cutoff is None:
            return 1
        return 1 if parsed is None else int(parsed <= parsed_cutoff)

    def conflict_resolved_by(value: Any, cutoff: Any) -> int:
        parsed, parsed_cutoff = cutoff_pair(value, cutoff)
        return int(parsed is not None and parsed_cutoff is not None and parsed <= parsed_cutoff)

    con.create_function("mango_tz_at_or_before", 2, at_or_before, deterministic=True)
    con.create_function("mango_tz_at_or_after", 2, at_or_after, deterministic=True)
    con.create_function("mango_tz_after", 2, after, deterministic=True)
    con.create_function("mango_tz_epoch", 1, epoch, deterministic=True)
    con.create_function("mango_conflict_created_by", 2, conflict_created_by, deterministic=True)
    con.create_function("mango_conflict_resolved_by", 2, conflict_resolved_by, deterministic=True)


__all__ = (
    "normalize_aware_utc",
    "parse_aware_utc",
    "register_temporal_sql_functions",
    "semantic_record_at_or_before",
)
