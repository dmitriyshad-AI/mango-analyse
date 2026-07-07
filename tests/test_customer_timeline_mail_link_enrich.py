from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from mango_mvp.customer_timeline.a2_mail_ingest import A2V3_MAIL_SOURCE_SYSTEM
from mango_mvp.customer_timeline.contracts import (
    CustomerIdentity,
    IdentityLink,
    IdentityStatus,
    TimelineDirection,
    TimelineEvent,
)
from mango_mvp.customer_timeline.mail_link_enrich import MailLinkEnrichConfig, run_mail_link_enrich
from mango_mvp.customer_timeline.store import CustomerTimelineSQLiteStore


NOW = datetime(2026, 7, 7, 12, 0, tzinfo=timezone.utc)


def _seed_customer_with_links(db_path: Path, allowed_root: Path) -> None:
    with CustomerTimelineSQLiteStore(db_path, allowed_root=allowed_root) as store:
        store.upsert_customer(
            CustomerIdentity(
                tenant_id="foton",
                customer_id="customer:phone",
                identity_status=IdentityStatus.STRONG,
                display_name="Phone Parent",
                primary_phone="+79161234567",
            )
        )
        store.upsert_customer(
            CustomerIdentity(
                tenant_id="foton",
                customer_id="customer:email",
                identity_status=IdentityStatus.STRONG,
                display_name="Email Parent",
                primary_email="parent@example.com",
            )
        )
        store.upsert_identity_link(
            IdentityLink(
                tenant_id="foton",
                customer_id="customer:phone",
                link_type="phone",
                link_value="+79161234567",
                source_system="test",
                source_ref="test",
                confidence=0.95,
            )
        )
        store.upsert_identity_link(
            IdentityLink(
                tenant_id="foton",
                customer_id="customer:email",
                link_type="email",
                link_value="parent@example.com",
                source_system="test",
                source_ref="test",
                confidence=0.95,
            )
        )


def _seed_pending_event(db_path: Path, allowed_root: Path, *, sha: str, source_file: Path, subject: str) -> None:
    with CustomerTimelineSQLiteStore(db_path, allowed_root=allowed_root) as store:
        event = TimelineEvent(
            tenant_id="foton",
            event_type="email_message",
            event_at=NOW,
            source_system=A2V3_MAIL_SOURCE_SYSTEM,
            source_id=sha,
            source_ref=f"mail_stage2:test:{sha[:16]}",
            direction=TimelineDirection.INBOUND,
            match_status="unmatched",
            subject=subject,
            text_preview="Входящее письмо.",
            summary="Родитель спрашивает про обучение.",
            record={
                "payload": {
                    "source_file": str(source_file),
                    "full_clean_text": "Родитель спрашивает про обучение.",
                    "brand": "unknown",
                }
            },
            metadata={"pending_attribution": True},
            created_at=NOW,
        )
        store.upsert_event(event, actor="test")


def _write_archive(tmp_path: Path, *, sha: str, email: str, text: str) -> tuple[Path, Path]:
    handoff = tmp_path / "handoff"
    source_file = handoff / "stage2_delta_ingest" / "stage2_delta_full_events.jsonl"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("", encoding="utf-8")
    archive_db = handoff / "archive" / "mail_archive.sqlite"
    archive_db.parent.mkdir(parents=True)
    text_path = handoff / "archive" / f"{sha}.txt"
    text_path.write_text(text, encoding="utf-8")
    with sqlite3.connect(archive_db) as con:
        con.executescript(
            """
            CREATE TABLE messages (
              sha256 TEXT PRIMARY KEY,
              subject TEXT,
              extracted_text_path TEXT
            );
            CREATE TABLE message_participants (
              message_sha256 TEXT,
              header_name TEXT,
              display_name TEXT,
              email_normalized TEXT,
              domain TEXT
            );
            """
        )
        con.execute(
            "INSERT INTO messages VALUES (?, ?, ?)",
            (sha, "Запись в Фотон", str(text_path)),
        )
        con.execute(
            "INSERT INTO message_participants VALUES (?, 'from', 'Parent', ?, 'example.com')",
            (sha, email),
        )
    return source_file, archive_db


def test_mail_link_enrich_links_phone_from_signature_without_opening_bot_visibility(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    _seed_customer_with_links(db_path, tmp_path)
    sha = "a" * 64
    source_file, _ = _write_archive(
        tmp_path,
        sha=sha,
        email="parent@example.com",
        text="Здравствуйте, интересует Фотон.\n\nС уважением,\nродитель\n+7 916 123-45-67",
    )
    _seed_pending_event(db_path, tmp_path, sha=sha, source_file=source_file, subject="Фотон")

    report = run_mail_link_enrich(
        MailLinkEnrichConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            out_dir=tmp_path / "out",
            apply=True,
        )
    )

    assert report["counts"]["planned.strong"] == 1
    assert report["safety"]["allowed_for_bot_before"] == report["safety"]["allowed_for_bot_after"] == 0
    assert report["safety"]["mail_stage2_allowed_for_bot_before"] == report["safety"]["mail_stage2_allowed_for_bot_after"] == 0
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as con:
        con.row_factory = sqlite3.Row
        event = con.execute("SELECT customer_id, match_status, record_json FROM timeline_events WHERE source_id=?", (sha,)).fetchone()
        chunk = con.execute(
            "SELECT allowed_for_bot, requires_manager_review FROM bot_context_chunks WHERE event_id=?",
            (json.loads(event["record_json"])["event_id"],),
        ).fetchone()
    assert event["customer_id"] == "customer:phone"
    assert event["match_status"] == "strong_unique"
    assert json.loads(event["record_json"])["metadata"]["fresh_relink"] is True
    assert tuple(chunk) == (0, 1)


def test_mail_link_enrich_keeps_email_only_and_body_phone_as_weak_pending(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    _seed_customer_with_links(db_path, tmp_path)
    sha = "b" * 64
    body_phone_then_tail = "\n".join(
        ["Телефон из текста +7 916 123-45-67 не должен быть сильной привязкой."]
        + [f"строка без телефона {index}" for index in range(20)]
    )
    source_file, _ = _write_archive(
        tmp_path,
        sha=sha,
        email="parent@example.com",
        text=body_phone_then_tail,
    )
    _seed_pending_event(db_path, tmp_path, sha=sha, source_file=source_file, subject="Запись")

    report = run_mail_link_enrich(
        MailLinkEnrichConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            out_dir=tmp_path / "out",
            apply=True,
        )
    )

    assert report["counts"]["planned.weak_email"] == 1
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as con:
        con.row_factory = sqlite3.Row
        event = con.execute("SELECT customer_id, match_status, record_json FROM timeline_events WHERE source_id=?", (sha,)).fetchone()
        chunks = con.execute("SELECT COUNT(*) FROM bot_context_chunks").fetchone()[0]
    payload = json.loads(event["record_json"])
    assert event["customer_id"] is None
    assert event["match_status"] == "unmatched"
    assert payload["metadata"]["pending_reason"] == "weak_email_only"
    assert chunks == 0
