from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from mango_mvp.customer_timeline import (
    BotContextChunk,
    CustomerIdentity,
    CustomerTimelineSQLiteStore,
    Stage3MaintenanceConfig,
    TimelineDirection,
    TimelineEvent,
    TimelineEventType,
    run_stage3_maintenance,
)


NOW = datetime(2026, 7, 3, 12, 0, tzinfo=timezone.utc)


def _identity() -> CustomerIdentity:
    return CustomerIdentity(
        tenant_id="foton",
        identity_status="strong",
        display_name="Тестовый клиент",
        primary_phone="+79161234567",
        primary_email="client@example.com",
        first_seen_at=NOW,
        last_seen_at=NOW,
        touch_count=1,
        created_at=NOW,
        updated_at=NOW,
    )


def _email_event(customer: CustomerIdentity | None, *, source_id: str, preview: str) -> TimelineEvent:
    return TimelineEvent(
        tenant_id="foton",
        customer_id=customer.customer_id if customer else None,
        event_type=TimelineEventType.EMAIL_MESSAGE,
        event_at=NOW,
        source_system="mail_archive_stage2",
        source_id=source_id,
        direction=TimelineDirection.INBOUND,
        subject="Заявка с сайта",
        text_preview=preview,
        summary="Клиент уточнил расписание группы и попросил ответить.",
        importance=2,
        match_status="strong_unique" if customer else "unmatched",
        confidence=0.9 if customer else None,
        created_at=NOW,
        record={"message_sha256": source_id},
    )


def _mail_chunk(event: TimelineEvent, *, text: str) -> BotContextChunk:
    return BotContextChunk(
        tenant_id=event.tenant_id,
        customer_id=event.customer_id or "",
        event_id=event.event_id,
        source_ref=event.source_ref,
        source_system=event.source_system,
        chunk_type="email_message",
        text=text,
        summary=event.summary or "",
        event_at=event.event_at,
        freshness_score=0.7,
        relevance_tags=("email",),
        allowed_for_bot=False,
        requires_manager_review=True,
        created_at=event.created_at,
    )


def test_stage3_soft_deletes_only_attributed_duplicates_and_keeps_fts_clean(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        customer = _identity()
        store.upsert_customer(customer)
        first = _email_event(customer, source_id="a" * 64, preview="Клиент спрашивает расписание.")
        duplicate = replace(
            _email_event(customer, source_id="b" * 64, preview="Клиент спрашивает расписание повторно."),
            created_at=NOW + timedelta(seconds=1),
        )
        same_key_different_preview = replace(
            _email_event(customer, source_id="f" * 64, preview="Клиент просит подобрать другую группу."),
            created_at=NOW + timedelta(seconds=1),
        )
        none_first = _email_event(None, source_id="c" * 64, preview="Безымянная web-форма 1.")
        none_second = replace(
            _email_event(None, source_id="d" * 64, preview="Безымянная web-форма 2."),
            created_at=NOW + timedelta(seconds=1),
        )
        store.upsert_event(first)
        store.upsert_event(duplicate)
        store.upsert_event(same_key_different_preview)
        store.upsert_event(none_first)
        store.upsert_event(none_second)
        store.upsert_bot_context_chunk(_mail_chunk(first, text="уникальныйдубль первый"))
        store.upsert_bot_context_chunk(_mail_chunk(duplicate, text="уникальныйдубль второй"))
        store.upsert_bot_context_chunk(_mail_chunk(same_key_different_preview, text="не дубль по preview"))
        content_key = store._con.execute(  # noqa: SLF001 - test fixture prepares historical duplicate rows.
            "SELECT content_key FROM timeline_events WHERE event_id = ?",
            (first.event_id,),
        ).fetchone()[0]
        store._con.execute(  # noqa: SLF001
            "UPDATE timeline_events SET content_key = ? WHERE event_id IN (?, ?, ?)",
            (content_key, same_key_different_preview.event_id, none_first.event_id, none_second.event_id),
        )
        store._con.execute(  # noqa: SLF001
            "UPDATE timeline_events SET content_key = ?, text_preview = ? WHERE event_id = ?",
            (content_key, first.text_preview, duplicate.event_id),
        )
        store._con.commit()  # noqa: SLF001

    report = run_stage3_maintenance(
        Stage3MaintenanceConfig(
            timeline_db_path=db_path,
            allowed_root=tmp_path,
            out_dir=tmp_path / "out",
            apply=True,
        )
    )

    assert report["duplicate_plan"]["groups"] == 1
    assert report["duplicate_plan"]["duplicate_events"] == 1
    assert report["duplicate_plan"]["none_customer_groups_report_only"]["groups"] == 1
    assert report["duplicate_plan"]["mixed_preview_groups_report_only"] == 1
    assert report["soft_delete"]["superseded_events"] == 1
    assert report["soft_delete"]["superseded_chunks"] == 1
    assert report["final_checks"]["fts_superseded_counts"] == {
        "timeline_event_fts_superseded": 0,
        "timeline_event_fts_keys_superseded": 0,
        "bot_context_chunk_fts_superseded": 0,
    }

    with sqlite3.connect(db_path) as con:
        hidden = con.execute(
            "SELECT superseded_by FROM timeline_events WHERE event_id = ?",
            (duplicate.event_id,),
        ).fetchone()[0]
        none_hidden = con.execute(
            "SELECT count(*) FROM timeline_events WHERE event_id IN (?, ?) AND superseded_by IS NOT NULL",
            (none_first.event_id, none_second.event_id),
        ).fetchone()[0]
        different_preview_hidden = con.execute(
            "SELECT superseded_by FROM timeline_events WHERE event_id = ?",
            (same_key_different_preview.event_id,),
        ).fetchone()[0]
        mail_allowed = con.execute(
            "SELECT count(*) FROM bot_context_chunks WHERE source_system = 'mail_archive_stage2' AND allowed_for_bot != 0"
        ).fetchone()[0]
    assert hidden == first.event_id
    assert none_hidden == 0
    assert different_preview_hidden is None
    assert mail_allowed == 0


def test_stage3_chunk_label_backfill_is_conservative_for_raw_mail(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        customer = _identity()
        store.upsert_customer(customer)
        event = _email_event(customer, source_id="e" * 64, preview="Письмо про расписание.")
        store.upsert_event(event)
        store.upsert_bot_context_chunk(_mail_chunk(event, text="Текст письма"))

    report = run_stage3_maintenance(
        Stage3MaintenanceConfig(
            timeline_db_path=db_path,
            allowed_root=tmp_path,
            out_dir=tmp_path / "out",
            apply=True,
        )
    )

    assert report["chunk_label_backfill"]["counts"]["chunks_updated"] == 1
    with sqlite3.connect(db_path) as con:
        row = con.execute("SELECT record_json FROM bot_context_chunks").fetchone()
    payload = __import__("json").loads(row[0])
    assert payload["allowed_for_bot"] is False
    assert payload["requires_manager_review"] is True
    assert payload["metadata"]["client_safe"] is False
    assert payload["metadata"]["client_safe_reason"] == "stage2_mail_manager_review_pending"
    assert payload["metadata"]["client_safe_policy_version"] == "cs_v1"
    assert payload["metadata"]["memory_status"] == "manager_review_required"


def test_stage3_hardens_legacy_mail_chunk_columns_and_json(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        customer = _identity()
        store.upsert_customer(customer)
        event = _email_event(customer, source_id="9" * 64, preview="Старое письмо.")
        store.upsert_event(event)
        store.upsert_bot_context_chunk(_mail_chunk(event, text="Полный текст старого письма"))

    with sqlite3.connect(db_path) as con:
        row = con.execute("SELECT chunk_id, record_json FROM bot_context_chunks").fetchone()
        payload = json.loads(row[1])
        payload["allowed_for_bot"] = True
        payload["requires_manager_review"] = False
        con.execute(
            """
            UPDATE bot_context_chunks
            SET allowed_for_bot = 1, requires_manager_review = 0, record_json = ?
            WHERE chunk_id = ?
            """,
            (json.dumps(payload, ensure_ascii=False), row[0]),
        )
        con.commit()

    report = run_stage3_maintenance(
        Stage3MaintenanceConfig(
            timeline_db_path=db_path,
            allowed_root=tmp_path,
            out_dir=tmp_path / "out",
            apply=True,
        )
    )

    assert report["mail_stage2_visibility_hardening"]["updated_chunks"] == 1
    assert report["after"]["mail_stage2_chunks_allowed"] == 0
    assert report["after"]["mail_stage2_chunks_without_review"] == 0
    with sqlite3.connect(db_path) as con:
        allowed, review, record_json = con.execute(
            "SELECT allowed_for_bot, requires_manager_review, record_json FROM bot_context_chunks"
        ).fetchone()
    payload = json.loads(record_json)
    assert (allowed, review) == (0, 1)
    assert payload["allowed_for_bot"] is False
    assert payload["requires_manager_review"] is True
