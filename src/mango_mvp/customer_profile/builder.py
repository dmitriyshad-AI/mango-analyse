from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from mango_mvp.customer_profile.contracts import (
    ProfileFieldCandidate,
    ProfileSnapshot,
    apply_superseded_rules,
    normalize_brand,
)
from mango_mvp.customer_profile.store import CustomerProfileSQLiteStore, sha256_file
from mango_mvp.utils.phone import normalize_phone


@dataclass(frozen=True)
class CustomerProfileBuildOptions:
    timeline_db: Path
    profiles_db: Path
    master_calls_db: Path | None = None
    tenant_id: str = "foton"
    customer_ids: Sequence[str] = ()
    phone: str | None = None
    build_id: str | None = None


class CustomerProfileBuilder:
    def __init__(self, options: CustomerProfileBuildOptions) -> None:
        self.options = options

    def build(self) -> Mapping[str, Any]:
        started_at = datetime.now(timezone.utc)
        build_id = self.options.build_id or f"profile_build_{started_at.strftime('%Y%m%dT%H%M%SZ')}"
        timeline = sqlite3.connect(f"file:{self.options.timeline_db}?mode=ro", uri=True)
        timeline.row_factory = sqlite3.Row
        try:
            profile_ids = self._select_profile_ids(timeline)
            profiles = self._load_profile_snapshots(timeline, profile_ids)
            fields = list(self._fields_from_timeline(timeline, profile_ids))
            if self.options.master_calls_db:
                fields.extend(self._fields_from_master_calls(timeline, profile_ids))
            fields = list(apply_superseded_rules(fields))
        finally:
            timeline.close()

        with CustomerProfileSQLiteStore(self.options.profiles_db) as store:
            result = store.replace_profiles(
                build_id=build_id,
                built_at=started_at,
                timeline_db_path=self.options.timeline_db,
                timeline_db_sha256=sha256_file(self.options.timeline_db),
                profiles=profiles,
                fields=fields,
                notes="deterministic rebuild from timeline and optional master calls",
            )
            summary = store.summary()
        return {
            **result,
            "summary": summary,
            "timeline_db": str(self.options.timeline_db),
            "profiles_db": str(self.options.profiles_db),
            "master_calls_db": str(self.options.master_calls_db) if self.options.master_calls_db else None,
            "unmatched_calls": getattr(self, "_unmatched_calls", 0),
            "ambiguous_calls": getattr(self, "_ambiguous_calls", 0),
        }

    def _select_profile_ids(self, con: sqlite3.Connection) -> list[str]:
        if self.options.customer_ids:
            return list(dict.fromkeys(self.options.customer_ids))
        if self.options.phone:
            phone = normalize_phone(self.options.phone)
            if not phone:
                return []
            rows = con.execute(
                """
                SELECT DISTINCT customer_id FROM identity_links
                WHERE tenant_id = ? AND link_type IN ('phone', 'mango_client_phone')
                  AND link_value = ? AND customer_id IS NOT NULL
                ORDER BY customer_id
                """,
                (self.options.tenant_id, phone),
            ).fetchall()
            return [str(row["customer_id"]) for row in rows]
        rows = con.execute(
            "SELECT customer_id FROM customer_identities WHERE tenant_id = ? ORDER BY customer_id",
            (self.options.tenant_id,),
        ).fetchall()
        return [str(row["customer_id"]) for row in rows]

    def _load_profile_snapshots(self, con: sqlite3.Connection, profile_ids: Sequence[str]) -> list[ProfileSnapshot]:
        if not profile_ids:
            return []
        placeholders = ",".join("?" for _ in profile_ids)
        rows = con.execute(
            f"""
            SELECT customer_id, tenant_id, COALESCE(primary_phone, '') AS primary_phone,
                   COALESCE(display_name, '') AS display_name
            FROM customer_identities
            WHERE customer_id IN ({placeholders})
            ORDER BY customer_id
            """,
            tuple(profile_ids),
        ).fetchall()
        counts = self._event_counts(con, profile_ids)
        return [
            ProfileSnapshot(
                profile_id=str(row["customer_id"]),
                tenant_id=str(row["tenant_id"]),
                primary_phone=str(row["primary_phone"] or ""),
                display_name=str(row["display_name"] or ""),
                source_event_count=counts.get(str(row["customer_id"]), {}).get("count", 0),
                last_event_at=counts.get(str(row["customer_id"]), {}).get("last_event_at"),
            )
            for row in rows
        ]

    def _event_counts(self, con: sqlite3.Connection, profile_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
        if not profile_ids:
            return {}
        placeholders = ",".join("?" for _ in profile_ids)
        rows = con.execute(
            f"""
            SELECT customer_id, count(*) AS event_count, max(event_at) AS last_event_at
            FROM timeline_events
            WHERE customer_id IN ({placeholders})
            GROUP BY customer_id
            """,
            tuple(profile_ids),
        ).fetchall()
        result: dict[str, dict[str, Any]] = {}
        for row in rows:
            result[str(row["customer_id"])] = {
                "count": int(row["event_count"] or 0),
                "last_event_at": parse_dt(row["last_event_at"]),
            }
        return result

    def _fields_from_timeline(self, con: sqlite3.Connection, profile_ids: Sequence[str]) -> Iterable[ProfileFieldCandidate]:
        if not profile_ids:
            return
        placeholders = ",".join("?" for _ in profile_ids)
        rows = con.execute(
            f"""
            SELECT customer_id, event_type, event_at, source_system, source_ref, record_json, summary
            FROM timeline_events
            WHERE customer_id IN ({placeholders})
              AND event_type IN ('tallanto_student_snapshot', 'amo_deal_stage', 'amo_contact_snapshot')
            ORDER BY event_at, source_ref
            """,
            tuple(profile_ids),
        ).fetchall()
        for row in rows:
            payload = event_payload(row)
            brand = brand_from_payload(payload)
            for field, value in timeline_field_values(str(row["event_type"]), payload):
                yield ProfileFieldCandidate(
                    profile_id=str(row["customer_id"]),
                    field=field,
                    value=value,
                    source_system=str(row["source_system"]),
                    source_ref=str(row["source_ref"]),
                    event_at=parse_dt(row["event_at"]) or datetime.now(timezone.utc),
                    brand=brand,
                )

    def _phone_to_profile_ids(self, con: sqlite3.Connection, profile_ids: Sequence[str]) -> dict[str, list[str]]:
        if not profile_ids:
            return {}
        placeholders = ",".join("?" for _ in profile_ids)
        rows = con.execute(
            f"""
            SELECT customer_id, link_value FROM identity_links
            WHERE customer_id IN ({placeholders})
              AND link_type IN ('phone', 'mango_client_phone')
            ORDER BY customer_id
            """,
            tuple(profile_ids),
        ).fetchall()
        result: dict[str, list[str]] = {}
        for row in rows:
            phone = normalize_phone(str(row["link_value"] or ""))
            if phone:
                result.setdefault(phone, []).append(str(row["customer_id"]))
                result.setdefault(last10(phone), []).append(str(row["customer_id"]))
        return {key: sorted(set(value)) for key, value in result.items()}

    def _fields_from_master_calls(self, timeline: sqlite3.Connection, profile_ids: Sequence[str]) -> Iterable[ProfileFieldCandidate]:
        phone_map = self._phone_to_profile_ids(timeline, profile_ids)
        if not phone_map or not self.options.master_calls_db:
            return
        brand_index = self._mango_brand_index(timeline, profile_ids)
        master = sqlite3.connect(f"file:{self.options.master_calls_db}?mode=ro", uri=True)
        master.row_factory = sqlite3.Row
        unmatched = 0
        ambiguous = 0
        try:
            table = "canonical_calls" if table_exists(master, "canonical_calls") else "call_records"
            rows = master.execute(
                f"""
                SELECT * FROM {table}
                WHERE analysis_status = 'done'
                  AND analysis_json IS NOT NULL
                ORDER BY started_at, rowid
                """
            ).fetchall()
            for row in rows:
                phone = normalize_phone(str(row["phone"] or ""))
                profile_ids_for_phone = phone_map.get(phone) or phone_map.get(last10(phone)) or []
                if not phone or not profile_ids_for_phone:
                    unmatched += 1
                    continue
                if len(set(profile_ids_for_phone)) > 1:
                    ambiguous += 1
                    continue
                analysis = parse_json(row["analysis_json"])
                event_at = parse_dt(row["started_at"]) or datetime.now(timezone.utc)
                source_id = str(row["canonical_call_id"] if "canonical_call_id" in row.keys() else row["id"])
                source_ref = f"mango:{source_id}"
                brand = brand_index.get(source_ref) or brand_index.get(source_id) or "unknown"
                for profile_id in profile_ids_for_phone:
                    yield from call_analysis_fields(
                        profile_id=profile_id,
                        analysis=analysis,
                        source_ref=source_ref,
                        event_at=event_at,
                        brand=brand,
                    )
        finally:
            master.close()
            self._unmatched_calls = unmatched
            self._ambiguous_calls = ambiguous

    def _mango_brand_index(self, con: sqlite3.Connection, profile_ids: Sequence[str]) -> dict[str, str]:
        if not profile_ids:
            return {}
        placeholders = ",".join("?" for _ in profile_ids)
        rows = con.execute(
            f"""
            SELECT source_id, source_ref, record_json FROM timeline_events
            WHERE customer_id IN ({placeholders}) AND event_type = 'mango_call'
            """,
            tuple(profile_ids),
        ).fetchall()
        result: dict[str, str] = {}
        for row in rows:
            payload = parse_json(row["record_json"])
            brand = brand_from_payload(payload)
            result[str(row["source_id"])] = brand
            result[str(row["source_ref"])] = brand
        return result


def table_exists(con: sqlite3.Connection, table: str) -> bool:
    return con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone() is not None


def parse_json(raw: Any) -> Mapping[str, Any]:
    if isinstance(raw, Mapping):
        return raw
    try:
        value = json.loads(raw or "{}")
    except (TypeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, Mapping) else {}


def parse_dt(raw: Any) -> datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        value = datetime.fromisoformat(text)
    except ValueError:
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def event_payload(row: sqlite3.Row) -> Mapping[str, Any]:
    record = parse_json(row["record_json"])
    nested = record.get("record") if isinstance(record.get("record"), Mapping) else record
    payload = nested.get("payload") if isinstance(nested.get("payload"), Mapping) else nested
    return payload if isinstance(payload, Mapping) else {}


def brand_from_payload(payload: Mapping[str, Any]) -> str:
    if isinstance(payload.get("metadata"), Mapping):
        brand = payload["metadata"].get("brand")
        if brand:
            return normalize_brand(str(brand))
    if isinstance(payload.get("record"), Mapping):
        brand = payload["record"].get("brand")
        if brand:
            return normalize_brand(str(brand))
    return normalize_brand(str(payload.get("brand") or payload.get("brand_hint") or "unknown"))


def timeline_field_values(event_type: str, payload: Mapping[str, Any]) -> Iterable[tuple[str, str]]:
    if event_type == "tallanto_student_snapshot":
        for key, field in (
            ("balance", "tallanto_balance"),
            ("group", "tallanto_group"),
            ("group_name", "tallanto_group"),
            ("course", "tallanto_group"),
            ("last_payment", "payment_fact"),
        ):
            value = text(payload.get(key))
            if value:
                yield field, value
    if event_type in {"amo_deal_stage", "amo_contact_snapshot"}:
        for key, field in (("stage", "amo_stage"), ("status", "amo_status"), ("target_product", "target_product")):
            value = text(payload.get(key))
            if value:
                yield field, value


def call_analysis_fields(
    *,
    profile_id: str,
    analysis: Mapping[str, Any],
    source_ref: str,
    event_at: datetime,
    brand: str,
) -> Iterable[ProfileFieldCandidate]:
    structured = analysis.get("structured_fields") if isinstance(analysis.get("structured_fields"), Mapping) else {}
    children = structured.get("children") if isinstance(structured.get("children"), list) else None
    if children:
        for idx, child in enumerate(children, start=1):
            if isinstance(child, Mapping):
                yield from child_fields(profile_id, child, f"child_{idx}", source_ref, event_at, brand)
    else:
        child: dict[str, Any] = {}
        people = structured.get("people") if isinstance(structured.get("people"), Mapping) else {}
        student = structured.get("student") if isinstance(structured.get("student"), Mapping) else {}
        interests = structured.get("interests") if isinstance(structured.get("interests"), Mapping) else {}
        if people.get("child_fio"):
            child["child_name"] = people.get("child_fio")
        if student.get("grade_current"):
            child["grade"] = student.get("grade_current")
        if interests.get("subjects"):
            child["subject"] = joined(interests.get("subjects"))
        if child:
            inferred_key = f"child_{stable_child_key(child)}"
            yield from child_fields(profile_id, child, inferred_key, source_ref, event_at, brand)

    people = structured.get("people") if isinstance(structured.get("people"), Mapping) else {}
    interests = structured.get("interests") if isinstance(structured.get("interests"), Mapping) else {}
    next_step = structured.get("next_step") if isinstance(structured.get("next_step"), Mapping) else {}
    for field, value in (
        ("parent_name", people.get("parent_fio")),
        ("format", joined(interests.get("format"))),
        ("target_product", analysis.get("target_product")),
        ("next_step", next_step.get("action") or analysis.get("next_step")),
        ("objection", joined(analysis.get("objections") or structured.get("objections"))),
    ):
        value_text = text(value)
        if value_text:
            yield ProfileFieldCandidate(
                profile_id=profile_id,
                field=field,
                value=value_text,
                child_key="",
                source_system="mango_processed_summary",
                source_ref=source_ref,
                event_at=event_at,
                brand=brand,
            )


def child_fields(
    profile_id: str,
    child: Mapping[str, Any],
    child_key: str,
    source_ref: str,
    event_at: datetime,
    brand: str,
) -> Iterable[ProfileFieldCandidate]:
    for field, value in (
        ("child_name", child.get("child_name") or child.get("name")),
        ("grade", child.get("grade") or child.get("grade_current")),
        ("subject", child.get("subject") or joined(child.get("subjects"))),
    ):
        value_text = text(value)
        if value_text:
            yield ProfileFieldCandidate(
                profile_id=profile_id,
                field=field,
                value=value_text,
                child_key=child_key,
                source_system="mango_processed_summary",
                source_ref=source_ref,
                event_at=event_at,
                brand=brand,
            )


def text(value: Any) -> str:
    return str(value or "").strip()


def joined(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return "; ".join(str(item).strip() for item in value if str(item).strip())
    return text(value)


def last10(phone: str) -> str:
    digits = "".join(ch for ch in phone if ch.isdigit())
    return digits[-10:] if len(digits) >= 10 else digits


def stable_child_key(child: Mapping[str, Any]) -> str:
    raw = text(child.get("child_name") or child.get("name") or child.get("grade") or child.get("subject") or "unknown").lower()
    import hashlib

    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]
