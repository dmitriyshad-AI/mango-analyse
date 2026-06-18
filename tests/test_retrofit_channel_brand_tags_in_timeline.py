from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from mango_mvp.customer_timeline.contracts import (
    BotContextChunk,
    CustomerIdentity,
    IdentityStatus,
    TimelineDirection,
    TimelineEvent,
    TimelineEventType,
)
from mango_mvp.customer_timeline.ids import stable_digest
from mango_mvp.customer_timeline.store import CustomerTimelineSQLiteStore, scrub_timeline_persisted_json

from scripts.retrofit_channel_brand_tags_in_timeline import (
    RetrofitChannelBrandConfig,
    run_retrofit_channel_brand_tags,
)


NOW = datetime(2026, 6, 11, 9, 0, tzinfo=timezone.utc)


def test_retrofit_channel_brand_tags_is_scoped_and_idempotent(tmp_path: Path) -> None:
    timeline_db, ids = seed_timeline(tmp_path)
    max_event_hash_before = record_hash(timeline_db, "timeline_events", "event_id", ids["max_event_id"])
    max_chunk_hash_before = record_hash(timeline_db, "bot_context_chunks", "chunk_id", ids["max_chunk_id"])
    audit_count_before = count_rows(timeline_db, "audit_log")

    dry_run = run_retrofit_channel_brand_tags(
        RetrofitChannelBrandConfig(timeline_db=timeline_db, allowed_root=tmp_path, apply=False)
    )

    assert dry_run["dry_run"] is True
    assert dry_run["changed"]["timeline_events"] == 2
    assert dry_run["changed"]["bot_context_chunks"] == 2
    assert event_payload(timeline_db, ids["telegram_event_id"])["metadata"]["brand"] == "unknown"

    first = run_retrofit_channel_brand_tags(
        RetrofitChannelBrandConfig(timeline_db=timeline_db, allowed_root=tmp_path, apply=True)
    )
    audit_count_after_first = count_rows(timeline_db, "audit_log")
    second = run_retrofit_channel_brand_tags(
        RetrofitChannelBrandConfig(timeline_db=timeline_db, allowed_root=tmp_path, apply=True)
    )

    telegram_event = event_payload(timeline_db, ids["telegram_event_id"])
    whatsapp_event = event_payload(timeline_db, ids["whatsapp_event_id"])
    telegram_chunk = chunk_payload(timeline_db, ids["telegram_chunk_id"])
    whatsapp_chunk = chunk_payload(timeline_db, ids["whatsapp_chunk_id"])
    max_event = event_payload(timeline_db, ids["max_event_id"])
    max_chunk = chunk_payload(timeline_db, ids["max_chunk_id"])

    assert first["dry_run"] is False
    assert first["changed"]["total"] == 4
    assert second["changed"]["total"] == 0
    assert count_rows(timeline_db, "audit_log") == audit_count_after_first
    assert audit_count_after_first == audit_count_before + 4

    assert telegram_event["metadata"]["brand"] == "unpk"
    assert telegram_event["record"]["message"]["brand_hint"] == "unpk"
    assert "channel_shared" not in telegram_event["metadata"]
    assert "channel_shared" not in telegram_event["record"]["message"]
    assert "brand:unpk" in telegram_chunk["relevance_tags"]
    assert "channel_shared:true" not in telegram_chunk["relevance_tags"]

    assert whatsapp_event["metadata"]["brand"] == "unpk"
    assert whatsapp_event["metadata"]["channel_shared"] is True
    assert whatsapp_event["record"]["message"]["brand_hint"] == "unpk"
    assert whatsapp_event["record"]["message"]["channel_shared"] is True
    assert whatsapp_chunk["metadata"]["channel_shared"] is True
    assert "brand:unpk" in whatsapp_chunk["relevance_tags"]
    assert "unpk" not in whatsapp_chunk["relevance_tags"]
    assert "channel_shared:true" in whatsapp_chunk["relevance_tags"]

    assert max_event["metadata"]["brand"] == "unknown"
    assert max_chunk["metadata"]["brand"] == "unknown"
    assert record_hash(timeline_db, "timeline_events", "event_id", ids["max_event_id"]) == max_event_hash_before
    assert record_hash(timeline_db, "bot_context_chunks", "chunk_id", ids["max_chunk_id"]) == max_chunk_hash_before
    assert_hash_matches_payload(timeline_db, "timeline_events", "event_id", ids["telegram_event_id"])
    assert_hash_matches_payload(timeline_db, "bot_context_chunks", "chunk_id", ids["whatsapp_chunk_id"])


def seed_timeline(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    db = tmp_path / "customer_timeline.sqlite"
    ids: dict[str, str] = {}
    with CustomerTimelineSQLiteStore(db, allowed_root=tmp_path) as store:
        customer = CustomerIdentity(
            tenant_id="foton",
            customer_id="customer-1",
            identity_status=IdentityStatus.STRONG,
            display_name="Customer",
            primary_phone="+79990000000",
            source_ref="seed",
            first_seen_at=NOW,
            last_seen_at=NOW,
        )
        store.upsert_customer(customer, actor="test")
        for channel, event_type, tags in (
            ("telegram", TimelineEventType.TELEGRAM_MESSAGE, ("telegram", "message", "brand:unknown")),
            ("whatsapp", TimelineEventType.WHATSAPP_MESSAGE, ("whatsapp", "message", "unknown")),
            ("max", TimelineEventType.MAX_MESSAGE, ("max", "message", "unknown")),
        ):
            event = TimelineEvent(
                tenant_id="foton",
                customer_id=customer.customer_id,
                event_type=event_type,
                event_at=NOW,
                source_system="channel_snapshot",
                source_id=f"{channel}:thread:1",
                source_ref=f"{channel}:thread:1",
                direction=TimelineDirection.INBOUND,
                text_preview=f"{channel} text",
                record={"message": {"channel": channel, "brand_hint": "unknown", "text": f"{channel} text"}},
                metadata={"brand": "unknown", "channel": channel},
                created_at=NOW,
            )
            store.upsert_event(event, actor="test")
            chunk = BotContextChunk(
                tenant_id="foton",
                customer_id=customer.customer_id,
                chunk_type="channel_message",
                text=f"{channel} text",
                event_id=event.event_id,
                source_system="channel_snapshot",
                source_ref=event.source_ref,
                event_at=NOW,
                relevance_tags=tags,
                metadata={"brand": "unknown", "channel": channel},
                created_at=NOW,
            )
            store.upsert_bot_context_chunk(chunk, actor="test")
            ids[f"{channel}_event_id"] = event.event_id
            ids[f"{channel}_chunk_id"] = chunk.chunk_id
    return db, ids


def event_payload(db_path: Path, event_id: str) -> dict:
    return stored_payload(db_path, "timeline_events", "event_id", event_id)


def chunk_payload(db_path: Path, chunk_id: str) -> dict:
    return stored_payload(db_path, "bot_context_chunks", "chunk_id", chunk_id)


def stored_payload(db_path: Path, table: str, key_column: str, key: str) -> dict:
    with sqlite3.connect(db_path) as con:
        raw = con.execute(f"SELECT record_json FROM {table} WHERE {key_column} = ?", (key,)).fetchone()[0]
    return json.loads(raw)


def record_hash(db_path: Path, table: str, key_column: str, key: str) -> str:
    with sqlite3.connect(db_path) as con:
        return str(con.execute(f"SELECT record_hash FROM {table} WHERE {key_column} = ?", (key,)).fetchone()[0])


def count_rows(db_path: Path, table: str) -> int:
    with sqlite3.connect(db_path) as con:
        return int(con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def assert_hash_matches_payload(db_path: Path, table: str, key_column: str, key: str) -> None:
    payload = stored_payload(db_path, table, key_column, key)
    assert record_hash(db_path, table, key_column, key) == stable_digest(scrub_timeline_persisted_json(payload))
