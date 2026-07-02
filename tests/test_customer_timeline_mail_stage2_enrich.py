from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mango_mvp.customer_timeline import CustomerIdentity, CustomerTimelineSQLiteStore, IdentityStatus
from mango_mvp.customer_timeline.contracts import BotContextChunk, TimelineDirection, TimelineEvent, TimelineEventType
from mango_mvp.customer_timeline.mail_stage2_enrich import (
    MAIL_STAGE2_ENRICH_SCHEMA_VERSION,
    MailStage2ExistingEnrichConfig,
    enrich_existing_mail_stage2_from_archives,
)


NOW = datetime(2026, 7, 3, 12, 0, tzinfo=timezone.utc)
TENANT = "foton"
CUSTOMER = "customer:mail"
MESSAGE_SHA = "a" * 64


def test_enrich_existing_mail_stage2_updates_event_chunk_and_keeps_manager_only(tmp_path: Path) -> None:
    db = tmp_path / "timeline.sqlite"
    archive = build_archive(tmp_path, text="Полный текст письма про олимпиаду, оплату и расписание.")
    store = CustomerTimelineSQLiteStore(db, allowed_root=tmp_path)
    try:
        store.upsert_customer(CustomerIdentity(tenant_id=TENANT, customer_id=CUSTOMER, identity_status=IdentityStatus.STRONG))
        event = TimelineEvent(
            tenant_id=TENANT,
            customer_id=CUSTOMER,
            event_type=TimelineEventType.EMAIL_MESSAGE,
            event_at=NOW,
            source_system="mail_archive_stage2",
            source_id=MESSAGE_SHA,
            direction=TimelineDirection.INBOUND,
            subject="Письмо",
            text_preview="Письмо",
            summary="Письмо",
            match_status="strong_unique",
            record={"message_sha256": MESSAGE_SHA},
            created_at=NOW,
        )
        store.upsert_event(event)
        store.upsert_bot_context_chunk(
            BotContextChunk(
                tenant_id=TENANT,
                customer_id=CUSTOMER,
                event_id=event.event_id,
                source_system="mail_archive_stage2",
                source_ref=event.source_ref,
                chunk_type="email_message",
                text="Письмо",
                summary="Письмо",
                event_at=NOW,
                allowed_for_bot=False,
                requires_manager_review=True,
                created_at=NOW,
            )
        )
    finally:
        store.close()

    config = MailStage2ExistingEnrichConfig(
        timeline_db_path=db,
        allowed_root=tmp_path,
        archive_db_paths=(archive,),
        out_dir=tmp_path / "out",
    )
    first = enrich_existing_mail_stage2_from_archives(config)
    second = enrich_existing_mail_stage2_from_archives(config)

    assert first["counts"]["events_updated"] == 1
    assert first["counts"]["chunks_updated"] == 1
    assert second["counts"]["events_duplicate"] == 1
    assert second["counts"]["chunks_duplicate"] == 1
    with sqlite3.connect(db) as con:
        con.row_factory = sqlite3.Row
        event_row = con.execute("SELECT summary, record_json FROM timeline_events WHERE source_id = ?", (MESSAGE_SHA,)).fetchone()
        chunk_row = con.execute(
            "SELECT allowed_for_bot, requires_manager_review, record_json FROM bot_context_chunks WHERE source_system = 'mail_archive_stage2'"
        ).fetchone()
        fts_hits = con.execute("SELECT count(*) FROM bot_context_chunk_fts WHERE bot_context_chunk_fts MATCH 'олимпиаду'").fetchone()[0]
    assert "олимпиаду" in event_row["summary"]
    assert "full_clean_text" in event_row["record_json"]
    assert MAIL_STAGE2_ENRICH_SCHEMA_VERSION in event_row["record_json"]
    assert chunk_row["allowed_for_bot"] == 0
    assert chunk_row["requires_manager_review"] == 1
    assert "олимпиаду" in chunk_row["record_json"]
    assert fts_hits >= 1


def test_enrich_existing_mail_stage2_rejects_prod_path(tmp_path: Path) -> None:
    prod_like = tmp_path / "customer_timeline_prod_20260621" / "customer_timeline.sqlite"
    prod_like.parent.mkdir(parents=True)
    prod_like.write_bytes(b"not sqlite")

    config = MailStage2ExistingEnrichConfig(
        timeline_db_path=prod_like,
        allowed_root=tmp_path,
        archive_db_paths=(tmp_path / "missing.sqlite",),
        out_dir=tmp_path / "out",
    )
    with pytest.raises(ValueError, match="refusing to enrich prod"):
        enrich_existing_mail_stage2_from_archives(config)


def build_archive(tmp_path: Path, *, text: str) -> Path:
    text_path = tmp_path / "_external_handoffs" / "mail_archive" / "extracted_text" / "mail.txt"
    text_path.parent.mkdir(parents=True)
    text_path.write_text(text, encoding="utf-8")
    archive = tmp_path / "mail_archive.sqlite"
    with sqlite3.connect(archive) as con:
        con.execute(
            """
            CREATE TABLE messages (
              sha256 TEXT PRIMARY KEY,
              extracted_text_path TEXT,
              extracted_text_chars INTEGER
            )
            """
        )
        con.execute(
            "INSERT INTO messages (sha256, extracted_text_path, extracted_text_chars) VALUES (?, ?, ?)",
            (MESSAGE_SHA, str(text_path), len(text)),
        )
        con.commit()
    return archive
