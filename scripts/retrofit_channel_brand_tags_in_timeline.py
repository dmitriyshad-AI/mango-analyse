#!/usr/bin/env python3
"""Retrofit Telegram/WhatsApp brand tags in a local customer timeline DB.

Dry-run by default. The script only mutates the local SQLite DB passed via
--timeline-db when --apply is set; it does not call AMO, Tallanto, Wappi,
Telegram, ASR, R+A, or any network API.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import quote

from mango_mvp.customer_timeline.ids import normalize_key, stable_digest, stable_prefixed_id
from mango_mvp.customer_timeline.safety import guard_customer_timeline_output_path
from mango_mvp.customer_timeline.store import (
    CUSTOMER_TIMELINE_SQLITE_SCHEMA_VERSION,
    guard_customer_timeline_sqlite_path,
    json_dumps,
    scrub_timeline_persisted_json,
)


SCHEMA_VERSION = "channel_brand_retrofit_v1"
SOURCE_SYSTEM = "channel_snapshot"
SUPPORTED_CHANNELS = {"telegram", "whatsapp"}


@dataclass(frozen=True)
class RetrofitChannelBrandConfig:
    timeline_db: Path
    allowed_root: Path
    tenant_id: str = "foton"
    telegram_brand: str = "unpk"
    whatsapp_brand: str = "unpk"
    apply: bool = False
    actor: str = "channel_brand_retrofit"

    def __post_init__(self) -> None:
        root = Path(self.allowed_root).expanduser().resolve(strict=False)
        db_path = guard_customer_timeline_output_path(
            guard_customer_timeline_sqlite_path(Path(self.timeline_db).expanduser()),
            root,
        )
        object.__setattr__(self, "allowed_root", root)
        object.__setattr__(self, "timeline_db", db_path)
        object.__setattr__(self, "tenant_id", normalize_key(self.tenant_id, "tenant_id"))
        object.__setattr__(self, "telegram_brand", normalize_brand(self.telegram_brand))
        object.__setattr__(self, "whatsapp_brand", normalize_brand(self.whatsapp_brand))
        object.__setattr__(self, "actor", normalize_key(self.actor, "actor"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Retrofit local channel brand tags in customer_timeline.sqlite.")
    parser.add_argument("--timeline-db", required=True)
    parser.add_argument("--allowed-root", default=".")
    parser.add_argument("--tenant-id", default="foton")
    parser.add_argument("--telegram-brand", default="unpk", choices=("foton", "unpk", "unknown"))
    parser.add_argument("--whatsapp-brand", default="unpk", choices=("foton", "unpk", "unknown"))
    parser.add_argument("--apply", action="store_true", help="Write changes into the local timeline DB.")
    parser.add_argument("--actor", default="channel_brand_retrofit")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_retrofit_channel_brand_tags(
        RetrofitChannelBrandConfig(
            timeline_db=Path(args.timeline_db),
            allowed_root=Path(args.allowed_root),
            tenant_id=args.tenant_id,
            telegram_brand=args.telegram_brand,
            whatsapp_brand=args.whatsapp_brand,
            apply=bool(args.apply),
            actor=args.actor,
        )
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["validation_ok"] else 1


def run_retrofit_channel_brand_tags(config: RetrofitChannelBrandConfig) -> Mapping[str, Any]:
    if config.apply:
        con = sqlite3.connect(config.timeline_db)
    else:
        uri = f"file:{quote(str(config.timeline_db.resolve(strict=False)), safe='/:')}?mode=ro&immutable=1"
        con = sqlite3.connect(uri, uri=True)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    if not config.apply:
        con.execute("PRAGMA query_only = ON")
    try:
        before = channel_brand_snapshot(con, config.tenant_id)
        planned = collect_channel_changes(con, config)
        if config.apply and planned:
            apply_changes(con, config, planned)
        after = channel_brand_snapshot(con, config.tenant_id)
    finally:
        con.close()

    changed_events = sum(1 for change in planned if change.table == "timeline_events")
    changed_chunks = sum(1 for change in planned if change.table == "bot_context_chunks")
    return {
        "schema_version": SCHEMA_VERSION,
        "validation_ok": True,
        "dry_run": not config.apply,
        "mode": "apply" if config.apply else "dry_run_preview",
        "timeline_db": str(config.timeline_db),
        "tenant_id": config.tenant_id,
        "brands": {"telegram": config.telegram_brand, "whatsapp": config.whatsapp_brand},
        "changed": {
            "timeline_events": changed_events,
            "bot_context_chunks": changed_chunks,
            "total": changed_events + changed_chunks,
        },
        "channels": count_planned_by_channel(planned),
        "before": before,
        "after": after,
        "safety": {
            "local_sqlite_only": True,
            "write_requires_apply": True,
            "write_crm": False,
            "write_tallanto": False,
            "send_messenger": False,
            "run_asr": False,
            "run_ra": False,
            "mutate_stable_runtime": False,
        },
    }


@dataclass(frozen=True)
class PlannedChange:
    table: str
    key_column: str
    key: str
    tenant_id: str
    channel: str
    before_hash: str
    after_hash: str
    payload: Mapping[str, Any]
    fts: Mapping[str, Any]


def collect_channel_changes(con: sqlite3.Connection, config: RetrofitChannelBrandConfig) -> list[PlannedChange]:
    changes: list[PlannedChange] = []
    if table_exists(con, "timeline_events"):
        for row in con.execute(
            """
            SELECT event_id, tenant_id, customer_id, opportunity_id, event_type, source_system,
                   event_at, subject, text_preview, summary, record_hash, record_json
            FROM timeline_events
            WHERE tenant_id = ? AND source_system = ?
            ORDER BY event_id
            """,
            (config.tenant_id, SOURCE_SYSTEM),
        ):
            payload = json_loads(row["record_json"])
            channel = event_channel(payload)
            if channel not in SUPPORTED_CHANNELS:
                continue
            updated = retrofit_payload(payload, channel, brand_for_channel(config, channel))
            add_change_if_needed(
                changes,
                table="timeline_events",
                key_column="event_id",
                key=str(row["event_id"]),
                tenant_id=str(row["tenant_id"]),
                channel=channel,
                before_hash=str(row["record_hash"]),
                payload=updated,
                fts=dict(row),
            )
    if table_exists(con, "bot_context_chunks"):
        for row in con.execute(
            """
            SELECT chunk_id, tenant_id, customer_id, opportunity_id, event_id, event_at,
                   record_hash, record_json
            FROM bot_context_chunks
            WHERE tenant_id = ? AND source_system = ?
            ORDER BY chunk_id
            """,
            (config.tenant_id, SOURCE_SYSTEM),
        ):
            payload = json_loads(row["record_json"])
            channel = chunk_channel(payload)
            if channel not in SUPPORTED_CHANNELS:
                continue
            updated = retrofit_chunk_payload(payload, channel, brand_for_channel(config, channel))
            add_change_if_needed(
                changes,
                table="bot_context_chunks",
                key_column="chunk_id",
                key=str(row["chunk_id"]),
                tenant_id=str(row["tenant_id"]),
                channel=channel,
                before_hash=str(row["record_hash"]),
                payload=updated,
                fts=dict(row),
            )
    return changes


def add_change_if_needed(
    changes: list[PlannedChange],
    *,
    table: str,
    key_column: str,
    key: str,
    tenant_id: str,
    channel: str,
    before_hash: str,
    payload: Mapping[str, Any],
    fts: Mapping[str, Any],
) -> None:
    after_hash = payload_hash(payload)
    if after_hash == before_hash:
        return
    changes.append(
        PlannedChange(
            table=table,
            key_column=key_column,
            key=key,
            tenant_id=tenant_id,
            channel=channel,
            before_hash=before_hash,
            after_hash=after_hash,
            payload=payload,
            fts=fts,
        )
    )


def apply_changes(
    con: sqlite3.Connection,
    config: RetrofitChannelBrandConfig,
    changes: Sequence[PlannedChange],
) -> None:
    next_seq = int(con.execute("SELECT COALESCE(MAX(seq), 0) + 1 FROM audit_log").fetchone()[0]) if table_exists(con, "audit_log") else 1
    created_at = datetime.now(timezone.utc)
    with con:
        for change in changes:
            con.execute(
                f"UPDATE {change.table} SET record_hash = ?, record_json = ? WHERE {change.key_column} = ?",
                (change.after_hash, json_dumps(change.payload), change.key),
            )
            if table_exists(con, "audit_log"):
                append_audit_log(con, config, change, next_seq, created_at)
                next_seq += 1
        rebuild_fts(con)


def rebuild_fts(con: sqlite3.Connection) -> None:
    if table_exists(con, "timeline_event_fts"):
        con.execute("DELETE FROM timeline_event_fts")
        for row in con.execute(
            """
            SELECT tenant_id, event_id, customer_id, opportunity_id, event_type, source_system,
                   event_at, subject, text_preview, summary, record_hash, record_json
            FROM timeline_events
            ORDER BY event_id
            """
        ):
            payload = json_loads(row["record_json"])
            con.execute(
                """
                INSERT INTO timeline_event_fts (
                  tenant_id, event_id, customer_id, opportunity_id, event_type,
                  source_system, event_at, subject, text_preview, summary, record_text
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["tenant_id"],
                    row["event_id"],
                    row["customer_id"],
                    row["opportunity_id"],
                    row["event_type"],
                    row["source_system"],
                    row["event_at"],
                    row["subject"],
                    row["text_preview"],
                    row["summary"],
                    json_dumps({"hash": row["record_hash"], "record": payload.get("record") or {}}),
                ),
            )
    if table_exists(con, "bot_context_chunk_fts"):
        con.execute("DELETE FROM bot_context_chunk_fts")
        for row in con.execute(
            """
            SELECT tenant_id, chunk_id, customer_id, opportunity_id, event_id, event_at,
                   record_hash, record_json
            FROM bot_context_chunks
            ORDER BY chunk_id
            """
        ):
            payload = json_loads(row["record_json"])
            con.execute(
                """
                INSERT INTO bot_context_chunk_fts (
                  tenant_id, chunk_id, customer_id, opportunity_id, event_id,
                  event_at, text, summary
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["tenant_id"],
                    row["chunk_id"],
                    row["customer_id"],
                    row["opportunity_id"],
                    row["event_id"],
                    row["event_at"] or "",
                    payload.get("text"),
                    f"{payload.get('summary') or ''} {row['record_hash']}",
                ),
            )


def append_audit_log(
    con: sqlite3.Connection,
    config: RetrofitChannelBrandConfig,
    change: PlannedChange,
    seq: int,
    created_at: datetime,
) -> None:
    audit_id = stable_prefixed_id(
        "timeline_audit",
        {
            "seq": seq,
            "tenant_id": change.tenant_id,
            "action": "retrofit_channel_brand",
            "entity_type": singular_record_type(change.table),
            "entity_id": change.key,
            "created_at": created_at.isoformat(),
        },
    )
    record = {
        "schema_version": CUSTOMER_TIMELINE_SQLITE_SCHEMA_VERSION,
        "audit_id": audit_id,
        "tenant_id": change.tenant_id,
        "action": "retrofit_channel_brand",
        "entity_type": singular_record_type(change.table),
        "entity_id": change.key,
        "actor": config.actor,
        "created_at": created_at.isoformat(),
        "ingestion_run_id": None,
        "before_hash": change.before_hash,
        "after_hash": change.after_hash,
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "channel": change.channel,
            "table": change.table,
        },
    }
    con.execute(
        """
        INSERT INTO audit_log (
          seq, audit_id, tenant_id, action, entity_type, entity_id, actor,
          created_at, ingestion_run_id, before_hash, after_hash, record_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            seq,
            audit_id,
            change.tenant_id,
            "retrofit_channel_brand",
            singular_record_type(change.table),
            change.key,
            config.actor,
            created_at.isoformat(),
            None,
            change.before_hash,
            change.after_hash,
            json_dumps(record),
        ),
    )


def retrofit_payload(payload: Mapping[str, Any], channel: str, brand: str) -> Mapping[str, Any]:
    updated = json.loads(json.dumps(payload, ensure_ascii=False))
    record = ensure_mapping(updated, "record")
    message = ensure_mapping(record, "message")
    metadata = ensure_mapping(updated, "metadata")
    message["brand_hint"] = brand
    metadata["brand"] = brand
    if channel == "whatsapp":
        message["channel_shared"] = True
        metadata["channel_shared"] = True
    return updated


def retrofit_chunk_payload(payload: Mapping[str, Any], channel: str, brand: str) -> Mapping[str, Any]:
    updated = json.loads(json.dumps(payload, ensure_ascii=False))
    metadata = ensure_mapping(updated, "metadata")
    metadata["brand"] = brand
    if channel == "whatsapp":
        metadata["channel_shared"] = True
    tags = updated.get("relevance_tags")
    if not isinstance(tags, list):
        tags = []
    updated["relevance_tags"] = retrofit_tags(tuple(str(item) for item in tags), channel=channel, brand=brand)
    return updated


def retrofit_tags(tags: Sequence[str], *, channel: str, brand: str) -> list[str]:
    cleaned: list[str] = []
    for tag in tags:
        normalized = str(tag or "").strip().lower()
        if not normalized or normalized in {"unknown", "brand:unknown"}:
            continue
        if normalized.startswith("brand:") and channel == "telegram":
            continue
        cleaned.append(normalized)
    cleaned.append(f"brand:{brand}" if channel == "telegram" else brand)
    if channel == "whatsapp":
        cleaned.append("channel_shared:true")
    return list(dict.fromkeys(cleaned))


def event_channel(payload: Mapping[str, Any]) -> str:
    record = payload.get("record") if isinstance(payload.get("record"), Mapping) else {}
    message = record.get("message") if isinstance(record.get("message"), Mapping) else {}
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else {}
    return normalize_channel(message.get("channel") or metadata.get("channel"))


def chunk_channel(payload: Mapping[str, Any]) -> str:
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else {}
    tags = payload.get("relevance_tags") if isinstance(payload.get("relevance_tags"), list) else []
    channel = normalize_channel(metadata.get("channel"))
    if channel:
        return channel
    normalized_tags = {str(tag or "").strip().lower() for tag in tags}
    if "telegram" in normalized_tags:
        return "telegram"
    if "whatsapp" in normalized_tags:
        return "whatsapp"
    return ""


def channel_brand_snapshot(con: sqlite3.Connection, tenant_id: str) -> Mapping[str, Any]:
    events = {"telegram": {"rows": 0, "brand": {}, "channel_shared": 0}, "whatsapp": {"rows": 0, "brand": {}, "channel_shared": 0}}
    chunks = {"telegram": {"rows": 0, "brand": {}, "channel_shared": 0}, "whatsapp": {"rows": 0, "brand": {}, "channel_shared": 0}}
    if table_exists(con, "timeline_events"):
        for row in con.execute(
            "SELECT record_json FROM timeline_events WHERE tenant_id = ? AND source_system = ?",
            (tenant_id, SOURCE_SYSTEM),
        ):
            payload = json_loads(row["record_json"])
            channel = event_channel(payload)
            if channel not in events:
                continue
            metadata = payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else {}
            brand = str(metadata.get("brand") or "unknown")
            events[channel]["rows"] += 1
            events[channel]["brand"][brand] = events[channel]["brand"].get(brand, 0) + 1
            if bool(metadata.get("channel_shared")):
                events[channel]["channel_shared"] += 1
    if table_exists(con, "bot_context_chunks"):
        for row in con.execute(
            "SELECT record_json FROM bot_context_chunks WHERE tenant_id = ? AND source_system = ?",
            (tenant_id, SOURCE_SYSTEM),
        ):
            payload = json_loads(row["record_json"])
            channel = chunk_channel(payload)
            if channel not in chunks:
                continue
            metadata = payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else {}
            brand = str(metadata.get("brand") or "unknown")
            chunks[channel]["rows"] += 1
            chunks[channel]["brand"][brand] = chunks[channel]["brand"].get(brand, 0) + 1
            if bool(metadata.get("channel_shared")) or {
                "channel_shared",
                "channel_shared:true",
            } & {
                str(tag or "").strip().lower()
                for tag in (payload.get("relevance_tags") if isinstance(payload.get("relevance_tags"), list) else [])
            }:
                chunks[channel]["channel_shared"] += 1
    return {"timeline_events": events, "bot_context_chunks": chunks}


def count_planned_by_channel(planned: Sequence[PlannedChange]) -> Mapping[str, Mapping[str, int]]:
    result = {channel: {"timeline_events": 0, "bot_context_chunks": 0, "total": 0} for channel in sorted(SUPPORTED_CHANNELS)}
    for change in planned:
        result[change.channel][change.table] += 1
        result[change.channel]["total"] += 1
    return result


def brand_for_channel(config: RetrofitChannelBrandConfig, channel: str) -> str:
    return config.telegram_brand if channel == "telegram" else config.whatsapp_brand


def ensure_mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        value = {}
        parent[key] = value
    return value


def payload_hash(payload: Mapping[str, Any]) -> str:
    return stable_digest(scrub_timeline_persisted_json(payload))


def json_loads(value: str) -> Mapping[str, Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("timeline record_json must contain an object")
    return parsed


def normalize_brand(value: Any) -> str:
    brand = str(value or "unknown").strip().lower()
    if brand not in {"foton", "unpk", "unknown"}:
        raise ValueError(f"unsupported brand: {value!r}")
    return brand


def normalize_channel(value: Any) -> str:
    channel = str(value or "").strip().lower()
    return channel if channel in SUPPORTED_CHANNELS else ""


def table_exists(con: sqlite3.Connection, table_name: str) -> bool:
    row = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'virtual') AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def singular_record_type(table: str) -> str:
    return "timeline_event" if table == "timeline_events" else "bot_context_chunk"


if __name__ == "__main__":
    raise SystemExit(main())
