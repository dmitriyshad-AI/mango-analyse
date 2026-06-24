from __future__ import annotations

from datetime import datetime, timezone

from mango_mvp.customer_timeline.amo_incremental import (
    event_summary,
    fetch_events_source,
)
from mango_mvp.customer_timeline.ingestion import TimelineSourceRecord
from mango_mvp.customer_timeline.nightly_incremental import (
    AmoEventNormalizer,
    IncrementalSourceConfig,
    normalizer_for_source,
)


NOW = datetime(2026, 6, 24, 8, 0, tzinfo=timezone.utc)


class FakeAmoClient:
    def __init__(self, payload):
        self.payload = payload

    def amo_api_get(self, *, path, params=None, limit=50):
        assert path == "events"
        assert "filter[created_at][from]" in (params or {})
        return self.payload


def test_amo_event_normalizer_creates_manager_review_raw_chunk() -> None:
    normalizer = AmoEventNormalizer(tenant_id="foton")

    batch = normalizer.normalize(
        TimelineSourceRecord(
            source_system="amo_events_created_at",
            source_ref="amocrm:event:evt-1",
            observed_at=NOW,
            payload={
                "event_id": "evt-1",
                "customer_id": "customer:test",
                "entity_type": "lead",
                "entity_id": "lead-1",
                "amo_event_type": "common_note_added",
                "created_at": NOW.isoformat(),
                "source_body_status": "note_body_missing",
                "summary": "AMO common_note_added for lead; body missing",
            },
        )
    )

    assert len(batch.events) == 1
    assert batch.events[0].event_type.value == "amo_note"
    assert batch.events[0].record["source_body_status"] == "note_body_missing"
    assert len(batch.bot_context_chunks) == 1
    assert batch.bot_context_chunks[0].allowed_for_bot is False
    assert batch.bot_context_chunks[0].requires_manager_review is True


def test_amo_event_normalizer_requires_customer_id() -> None:
    normalizer = AmoEventNormalizer(tenant_id="foton")

    batch = normalizer.normalize(
        TimelineSourceRecord(
            source_system="amo_events_created_at",
            source_ref="amocrm:event:evt-2",
            observed_at=NOW,
            payload={
                "event_id": "evt-2",
                "entity_type": "lead",
                "entity_id": "lead-2",
                "amo_event_type": "incoming_chat_message",
                "created_at": NOW.isoformat(),
            },
        )
    )

    assert batch.events == ()
    assert batch.bot_context_chunks == ()


def test_normalizer_dispatch_supports_amo_snapshot_and_amo_event() -> None:
    snapshot = normalizer_for_source(
        IncrementalSourceConfig(
            name="lead_cards",
            source_system="amo_leads_updated_at",
            path="dummy.jsonl",
            normalizer="amo_snapshot",
        )
    )
    event = normalizer_for_source(
        IncrementalSourceConfig(
            name="events",
            source_system="amo_events_created_at",
            path="dummy.jsonl",
            normalizer="amo_event",
        )
    )

    assert snapshot.source_system == "amocrm_snapshot"
    assert event.source_system == "amocrm_event"


def test_fetch_events_source_marks_unmatched_and_ambiguous() -> None:
    payload = {
        "_embedded": {
            "events": [
                {"id": 1, "type": "incoming_chat_message", "entity_type": "lead", "entity_id": 10, "created_at": 1782250000},
                {"id": 2, "type": "common_note_added", "entity_type": "lead", "entity_id": 20, "created_at": 1782250001},
                {"id": 3, "type": "incoming_mail", "entity_type": "contact", "entity_id": 30, "created_at": 1782250002},
                {"id": 4, "type": "entity_linked", "entity_type": "lead", "entity_id": 10, "created_at": 1782250003},
            ]
        }
    }
    config = type("Config", (), {"page_limit": 10, "max_pages": 1, "sleep_sec": 0.0})()

    rows, stats = fetch_events_source(
        FakeAmoClient(payload),
        from_ts=NOW,
        link_index={
            ("amo_lead_id", "10"): ("customer:lead-10",),
            ("amo_lead_id", "20"): ("customer:a", "customer:b"),
        },
        config=config,
    )

    assert len(rows) == 1
    assert rows[0]["customer_id"] == "customer:lead-10"
    assert rows[0]["source_body_status"] == "event_only"
    assert stats["skipped"]["ambiguous"] == 1
    assert stats["skipped"]["unmatched"] == 1
    assert stats["skipped"]["unsupported_type"] == 1
    assert event_summary({"type": "common_note_added", "entity_type": "lead"}, body_status="note_body_missing").endswith("body missing")
