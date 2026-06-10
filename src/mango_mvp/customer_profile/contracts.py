from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence


CUSTOMER_PROFILE_SCHEMA_VERSION = "customer_profile_v1"


SOURCE_CONFIDENCE = {
    "tallanto_snapshot": 0.95,
    "amocrm_snapshot": 0.9,
    "mango_processed_summary": 0.8,
    "channel_snapshot": 0.65,
}


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def require_text(value: str | None, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{name} is required")
    return text


def normalize_brand(value: str | None) -> str:
    text = str(value or "").strip().lower()
    return text if text in {"foton", "unpk", "unknown"} else "unknown"


def stable_field_id(
    *,
    profile_id: str,
    field: str,
    value: str,
    child_key: str,
    source_system: str,
    source_ref: str,
    event_at: str,
) -> str:
    payload = {
        "profile_id": profile_id,
        "field": field,
        "value": value,
        "child_key": child_key,
        "source_system": source_system,
        "source_ref": source_ref,
        "event_at": event_at,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ProfileFieldCandidate:
    profile_id: str
    field: str
    value: str
    source_system: str
    source_ref: str
    event_at: datetime
    child_key: str = ""
    brand: str = "unknown"
    quote: str = ""
    field_id: str | None = None
    superseded_by: str = ""

    def __post_init__(self) -> None:
        profile_id = require_text(self.profile_id, "profile_id")
        field = require_text(self.field, "field")
        value = require_text(self.value, "value")
        source_system = require_text(self.source_system, "source_system")
        source_ref = require_text(self.source_ref, "source_ref")
        if self.event_at.tzinfo is None or self.event_at.utcoffset() is None:
            raise ValueError("event_at must be timezone-aware")
        event_at = self.event_at.astimezone(timezone.utc)
        child_key = str(self.child_key or "").strip()
        quote = str(self.quote or "").strip()[:200]
        object.__setattr__(self, "profile_id", profile_id)
        object.__setattr__(self, "field", field)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "source_system", source_system)
        object.__setattr__(self, "source_ref", source_ref)
        object.__setattr__(self, "event_at", event_at)
        object.__setattr__(self, "child_key", child_key)
        object.__setattr__(self, "brand", normalize_brand(self.brand))
        object.__setattr__(self, "quote", quote)
        object.__setattr__(
            self,
            "field_id",
            self.field_id
            or stable_field_id(
                profile_id=profile_id,
                field=field,
                value=value,
                child_key=child_key,
                source_system=source_system,
                source_ref=source_ref,
                event_at=event_at.isoformat(),
            ),
        )


@dataclass(frozen=True)
class ProfileSnapshot:
    profile_id: str
    tenant_id: str
    primary_phone: str = ""
    display_name: str = ""
    source_event_count: int = 0
    last_event_at: datetime | None = None


def apply_superseded_rules(fields: Sequence[ProfileFieldCandidate]) -> tuple[ProfileFieldCandidate, ...]:
    grouped: dict[tuple[str, str, str], list[ProfileFieldCandidate]] = {}
    for field in fields:
        grouped.setdefault((field.profile_id, field.field, field.child_key), []).append(field)

    superseded: dict[str, str] = {}
    for items in grouped.values():
        ordered = sorted(
            items,
            key=lambda item: (
                item.event_at,
                SOURCE_CONFIDENCE.get(item.source_system, 0.5),
                item.source_ref,
            ),
        )
        winner = ordered[-1]
        for item in ordered[:-1]:
            if item.value != winner.value:
                superseded[item.field_id or ""] = winner.field_id or ""

    return tuple(replace(item, superseded_by=superseded.get(item.field_id or "", "")) for item in fields)


def profile_contract_inventory() -> Mapping[str, Any]:
    return {
        "schema_version": CUSTOMER_PROFILE_SCHEMA_VERSION,
        "storage": "product_data/customer_profiles/customer_profiles.sqlite",
        "rebuild_model": "deterministic_full_rebuild",
        "field_origin_required": True,
        "conflict_rule": "later_event_wins_then_source_confidence_then_source_ref",
    }
