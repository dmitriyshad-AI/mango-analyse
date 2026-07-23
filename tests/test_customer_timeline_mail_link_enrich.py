from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

import mango_mvp.customer_timeline.mail_link_enrich as mail_link_enrich_module
from mango_mvp.customer_timeline.a2_mail_ingest import A2V3_MAIL_SOURCE_SYSTEM
from mango_mvp.customer_timeline.contracts import (
    CustomerIdentity,
    IdentityLink,
    IdentityStatus,
    TimelineDirection,
    TimelineEvent,
)
from mango_mvp.customer_timeline.mail_link_enrich import (
    MailLinkEnrichConfig,
    _contact_from_archive_row,
    run_mail_link_enrich,
)
from mango_mvp.customer_timeline.store import CustomerTimelineSQLiteStore


NOW = datetime(2026, 7, 7, 12, 0, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    ("stored_direction", "participants", "expected_email"),
    (
        (
            "outbound",
            (
                {"header_name": "from", "email_normalized": "parent@example.com", "domain": "example.com"},
                {"header_name": "to", "email_normalized": "edu@kmipt.ru", "domain": "kmipt.ru"},
            ),
            "parent@example.com",
        ),
        (
            "inbound",
            (
                {"header_name": "from", "email_normalized": "edu@kmipt.ru", "domain": "kmipt.ru"},
                {"header_name": "to", "email_normalized": "parent@example.com", "domain": "example.com"},
            ),
            "parent@example.com",
        ),
    ),
)
def test_mail_contact_uses_headers_when_stored_direction_is_wrong(
    stored_direction: str,
    participants: tuple[dict[str, str], ...],
    expected_email: str,
) -> None:
    contact = _contact_from_archive_row(stored_direction, {"participants": participants, "text": ""})

    assert contact.contact_email == expected_email


def _seed_customer_with_links(
    db_path: Path,
    allowed_root: Path,
    *,
    email_source_system: str = "test",
) -> None:
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
                source_system=email_source_system,
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


def _seed_pending_event_with_archive_db(
    db_path: Path,
    allowed_root: Path,
    *,
    sha: str,
    archive_db: Path,
    subject: str,
) -> None:
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
                "stage2_enrich_archive_db": str(archive_db),
                "full_clean_text": "Родитель спрашивает про обучение.",
                "brand": "unknown",
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


def _write_thread_archive(
    tmp_path: Path,
    *,
    target_sha: str,
    reference_messages: list[tuple[str, str]],
) -> Path:
    archive_dir = tmp_path / "thread_archive"
    archive_db = archive_dir / "mail_archive.sqlite"
    raw_dir = archive_dir / "raw_eml" / target_sha[:2]
    raw_dir.mkdir(parents=True)
    target_message_id = "target@example.test"
    references = " ".join(f"<{message_id}>" for _, message_id in reference_messages)
    raw_path = raw_dir / f"{target_sha}.eml"
    raw_path.write_text(
        "\r\n".join(
            (
                f"Message-ID: <{target_message_id}>",
                f"In-Reply-To: <{reference_messages[0][1]}>",
                f"References: {references}",
                "From: edu@kmipt.ru",
                "To: edu@kmipt.ru",
                "Subject: Re: Запись",
                "",
                "Ответ",
            )
        ),
        encoding="utf-8",
    )
    with sqlite3.connect(archive_db) as con:
        con.executescript(
            """
            CREATE TABLE messages (
              sha256 TEXT PRIMARY KEY,
              message_id TEXT,
              subject TEXT,
              raw_eml_path TEXT,
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
            "INSERT INTO messages VALUES (?, ?, 'Re: Запись', ?, NULL)",
            (target_sha, f"<{target_message_id}>", str(raw_path)),
        )
        con.executemany(
            "INSERT INTO messages VALUES (?, ?, 'Запись', '', NULL)",
            [(sha, f"<{message_id}>") for sha, message_id in reference_messages],
        )
    return archive_db


def _seed_linked_mail_event(
    db_path: Path,
    allowed_root: Path,
    *,
    sha: str,
    customer_id: str,
) -> None:
    with CustomerTimelineSQLiteStore(db_path, allowed_root=allowed_root) as store:
        store.upsert_event(
            TimelineEvent(
                tenant_id="foton",
                customer_id=customer_id,
                event_type="email_message",
                event_at=NOW,
                source_system=A2V3_MAIL_SOURCE_SYSTEM,
                source_id=sha,
                direction=TimelineDirection.INBOUND,
                match_status="strong_unique",
                confidence=0.95,
                subject="Запись",
                record={"payload": {"brand": "unknown"}},
                created_at=NOW,
            ),
            actor="test",
        )


def _seed_trusted_customer_brand(db_path: Path, *, customer_id: str, brand: str) -> None:
    with sqlite3.connect(db_path) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS a2v3_customer_brand_profiles (
              tenant_id TEXT NOT NULL,
              customer_id TEXT NOT NULL,
              brand TEXT NOT NULL,
              source TEXT NOT NULL,
              reason TEXT NOT NULL
            )
            """
        )
        con.execute(
            "INSERT INTO a2v3_customer_brand_profiles VALUES ('foton', ?, ?, 'customer_history', 'single_known_brand_in_history')",
            (customer_id, brand),
        )


@pytest.mark.parametrize(("customer_brand", "expected_outcome"), (("foton", "strong"), ("unpk", "blocked")))
def test_mail_link_enrich_blocks_trusted_cross_brand_phone_match(
    tmp_path: Path,
    customer_brand: str,
    expected_outcome: str,
) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    _seed_customer_with_links(db_path, tmp_path)
    _seed_trusted_customer_brand(db_path, customer_id="customer:phone", brand=customer_brand)
    sha = "9" * 64
    source_file, _ = _write_archive(
        tmp_path,
        sha=sha,
        email="unknown@example.com",
        text="Здравствуйте, интересует Фотон.\n\n+7 916 123-45-67",
    )
    _seed_pending_event(db_path, tmp_path, sha=sha, source_file=source_file, subject="Фотон")

    report = run_mail_link_enrich(
        MailLinkEnrichConfig(timeline_db=db_path, allowed_root=tmp_path, out_dir=tmp_path / "out", apply=True)
    )

    assert report["counts"][f"planned.{expected_outcome}"] == 1
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as con:
        event = con.execute(
            "SELECT customer_id, match_status, record_json FROM timeline_events WHERE source_id=?",
            (sha,),
        ).fetchone()
    if expected_outcome == "strong":
        assert event[0] == "customer:phone"
    else:
        assert event[0] is None
        assert event[1] == "ambiguous"
        assert json.loads(event[2])["metadata"]["pending_reason"] == "cross_brand_signal"


def test_mail_link_enrich_blocks_phone_email_customer_contradiction(tmp_path: Path) -> None:
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

    assert report["counts"]["planned.blocked"] == 1
    assert report["safety"]["allowed_for_bot_before"] == report["safety"]["allowed_for_bot_after"] == 0
    assert report["safety"]["mail_stage2_allowed_for_bot_before"] == report["safety"]["mail_stage2_allowed_for_bot_after"] == 0
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as con:
        con.row_factory = sqlite3.Row
        event = con.execute("SELECT customer_id, match_status, record_json FROM timeline_events WHERE source_id=?", (sha,)).fetchone()
        chunks = con.execute("SELECT COUNT(*) FROM bot_context_chunks").fetchone()[0]
    assert event["customer_id"] is None
    assert event["match_status"] == "ambiguous"
    assert json.loads(event["record_json"])["metadata"]["fresh_relink"] is True
    assert json.loads(event["record_json"])["metadata"]["pending_reason"] == "phone_email_customer_conflict"
    assert chunks == 0


@pytest.mark.parametrize(
    ("email_source_system", "expected_outcome"),
    (("test", "strong"), ("mail_archive_stage2", "blocked")),
)
def test_mail_link_enrich_resolves_shared_family_phone_only_with_external_strong_email(
    tmp_path: Path,
    email_source_system: str,
    expected_outcome: str,
) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    _seed_customer_with_links(db_path, tmp_path, email_source_system=email_source_system)
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        store.upsert_identity_link(
            IdentityLink(
                tenant_id="foton",
                customer_id="customer:email",
                link_type="phone",
                link_value="+79161234567",
                source_system="test",
                source_ref="family-test",
                confidence=0.95,
            )
        )
    sha = "d" * 64
    source_file, _ = _write_archive(
        tmp_path,
        sha=sha,
        email="parent@example.com",
        text="Здравствуйте.\n\nС уважением,\n+7 916 123-45-67",
    )
    _seed_pending_event(db_path, tmp_path, sha=sha, source_file=source_file, subject="Фотон")

    report = run_mail_link_enrich(
        MailLinkEnrichConfig(timeline_db=db_path, allowed_root=tmp_path, out_dir=tmp_path / "out", apply=True)
    )

    assert report["counts"][f"planned.{expected_outcome}"] == 1
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as con:
        event = con.execute("SELECT customer_id, match_status, record_json FROM timeline_events WHERE source_id=?", (sha,)).fetchone()
    if expected_outcome == "strong":
        assert event[0] == "customer:email"
        assert event[1] == "strong_unique"
        assert json.loads(event[2])["metadata"]["mail_link_enrich"]["reason"] == "shared_phone_strong_email_intersection"
    else:
        assert event[0] is None
        assert event[1] == "ambiguous"
        assert json.loads(event[2])["metadata"]["pending_reason"] == "phone_multiple_customers"


@pytest.mark.parametrize("match_class", ("inferred", "manual"))
def test_mail_link_enrich_does_not_promote_non_authoritative_phone_link(
    tmp_path: Path,
    match_class: str,
) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        store.upsert_customer(
            CustomerIdentity(
                tenant_id="foton",
                customer_id="customer:inferred",
                identity_status=IdentityStatus.STRONG,
                display_name="Inferred parent",
            )
        )
        store.upsert_identity_link(
            IdentityLink(
                tenant_id="foton",
                customer_id="customer:inferred",
                link_type="phone",
                link_value="+79161234567",
                source_system="mail_archive_stage2",
                source_ref="inferred-test",
                match_class=match_class,
                confidence=0.6,
            )
        )
    sha = "e" * 64
    source_file, _ = _write_archive(
        tmp_path,
        sha=sha,
        email="unlinked@example.com",
        text="Здравствуйте.\n\nС уважением,\n+7 916 123-45-67",
    )
    _seed_pending_event(db_path, tmp_path, sha=sha, source_file=source_file, subject="Фотон")

    report = run_mail_link_enrich(
        MailLinkEnrichConfig(timeline_db=db_path, allowed_root=tmp_path, out_dir=tmp_path / "out", apply=True)
    )

    assert report["counts"]["planned.unmatched"] == 1
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as con:
        event = con.execute("SELECT customer_id, match_status, record_json FROM timeline_events WHERE source_id=?", (sha,)).fetchone()
        chunks = con.execute("SELECT COUNT(*) FROM bot_context_chunks").fetchone()[0]
    assert event[0] is None
    assert event[1] == "unmatched"
    assert json.loads(event[2])["metadata"]["pending_reason"] == "phone_non_authoritative_identity_link"
    assert chunks == 0


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


def test_mail_link_enrich_reads_stage2_archive_db_without_legacy_payload(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    _seed_customer_with_links(db_path, tmp_path)
    sha = "c" * 64
    _, archive_db = _write_archive(
        tmp_path,
        sha=sha,
        email="parent@example.com",
        text="Здравствуйте, интересует Фотон.",
    )
    _seed_pending_event_with_archive_db(db_path, tmp_path, sha=sha, archive_db=archive_db, subject="Фотон")

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
    payload = json.loads(event["record_json"])
    assert event["customer_id"] is None
    assert event["match_status"] == "unmatched"
    assert payload["metadata"]["pending_reason"] == "weak_email_only"
    assert payload["metadata"]["mail_link_enrich"]["reason"] == "email_unique_identity_link"
    assert payload["record"]["payload"]["contact_email_hash"]


def test_mail_link_enrich_reconsiders_old_pending_after_identity_refresh(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    _seed_customer_with_links(db_path, tmp_path)
    sha = "f" * 64
    source_file, _ = _write_archive(
        tmp_path,
        sha=sha,
        email="unlinked@example.com",
        text="Здравствуйте.\n\nС уважением,\n+7 916 123-45-67",
    )
    _seed_pending_event(db_path, tmp_path, sha=sha, source_file=source_file, subject="Фотон")
    with sqlite3.connect(db_path) as con:
        row = con.execute("SELECT record_json FROM timeline_events WHERE source_id=?", (sha,)).fetchone()
        payload = json.loads(row[0])
        payload["metadata"]["pending_reason"] = "no_strong_identity_match"
        con.execute(
            "UPDATE timeline_events SET record_json=? WHERE source_id=?",
            (json.dumps(payload, ensure_ascii=False, sort_keys=True), sha),
        )

    default_report = run_mail_link_enrich(
        MailLinkEnrichConfig(timeline_db=db_path, allowed_root=tmp_path, out_dir=tmp_path / "default")
    )
    reconsidered = run_mail_link_enrich(
        MailLinkEnrichConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            out_dir=tmp_path / "reconsidered",
            reconsider_pending=True,
        )
    )

    assert default_report["target_events"] == 0
    assert reconsidered["target_events"] == 1
    assert reconsidered["counts"]["planned.strong"] == 1


def test_mail_link_enrich_never_reconsiders_cross_brand_pending(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    sha = "9" * 64
    source_file, _ = _write_archive(
        tmp_path,
        sha=sha,
        email="parent@example.com",
        text="Здравствуйте.",
    )
    _seed_pending_event(db_path, tmp_path, sha=sha, source_file=source_file, subject="Другой бренд")
    with sqlite3.connect(db_path) as con:
        row = con.execute("SELECT record_json FROM timeline_events WHERE source_id=?", (sha,)).fetchone()
        payload = json.loads(row[0])
        payload["metadata"]["pending_reason"] = "cross_brand_signal"
        con.execute(
            "UPDATE timeline_events SET record_json=? WHERE source_id=?",
            (json.dumps(payload, ensure_ascii=False, sort_keys=True), sha),
        )

    report = run_mail_link_enrich(
        MailLinkEnrichConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            out_dir=tmp_path / "out",
            reconsider_pending=True,
        )
    )

    assert report["target_events"] == 0


def test_mail_link_enrich_reconsiders_ambiguous_after_identity_refresh(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    _seed_customer_with_links(db_path, tmp_path)
    sha = "7" * 64
    source_file, _ = _write_archive(
        tmp_path,
        sha=sha,
        email="unlinked@example.com",
        text="Здравствуйте.\n\nС уважением,\n+7 916 123-45-67",
    )
    _seed_pending_event(db_path, tmp_path, sha=sha, source_file=source_file, subject="Фотон")
    with sqlite3.connect(db_path) as con:
        row = con.execute("SELECT record_json FROM timeline_events WHERE source_id=?", (sha,)).fetchone()
        payload = json.loads(row[0])
        payload["metadata"]["pending_reason"] = "shared_family_contact"
        con.execute(
            "UPDATE timeline_events SET match_status='ambiguous', record_json=? WHERE source_id=?",
            (json.dumps(payload, ensure_ascii=False, sort_keys=True), sha),
        )

    default_report = run_mail_link_enrich(
        MailLinkEnrichConfig(timeline_db=db_path, allowed_root=tmp_path, out_dir=tmp_path / "default")
    )
    reconsidered = run_mail_link_enrich(
        MailLinkEnrichConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            out_dir=tmp_path / "reconsidered",
            reconsider_pending=True,
        )
    )

    assert default_report["target_events"] == 0
    assert reconsidered["target_events"] == 1
    assert reconsidered["counts"]["planned.strong"] == 1


def test_mail_link_enrich_uses_explicit_archive_fallback_for_deleted_old_path(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    _seed_customer_with_links(db_path, tmp_path)
    sha = "8" * 64
    source_file, archive_db = _write_archive(
        tmp_path,
        sha=sha,
        email="unlinked@example.com",
        text="Здравствуйте.\n\nС уважением,\n+7 916 123-45-67",
    )
    _seed_pending_event(db_path, tmp_path, sha=sha, source_file=source_file, subject="Фотон")
    source_file.unlink()

    report = run_mail_link_enrich(
        MailLinkEnrichConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            out_dir=tmp_path / "out",
            fallback_archive_dbs=(archive_db,),
        )
    )

    assert report["target_events"] == 1
    assert report["counts"]["planned.strong"] == 1


def test_mail_link_enrich_promotes_unique_tallanto_email_without_phone(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    _seed_customer_with_links(db_path, tmp_path, email_source_system="tallanto_snapshot")
    sha = "7" * 64
    source_file, _ = _write_archive(
        tmp_path,
        sha=sha,
        email="parent@example.com",
        text="Здравствуйте, интересует обучение.",
    )
    _seed_pending_event(db_path, tmp_path, sha=sha, source_file=source_file, subject="Фотон")

    report = run_mail_link_enrich(
        MailLinkEnrichConfig(timeline_db=db_path, allowed_root=tmp_path, out_dir=tmp_path / "out")
    )

    assert report["counts"]["planned.strong"] == 1
    assert report["counts"]["reason.strong_external_email_identity_link"] == 1


def test_mail_link_enrich_revokes_link_after_identity_becomes_ambiguous(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    _seed_customer_with_links(db_path, tmp_path, email_source_system="tallanto_snapshot")
    sha = "0" * 64
    source_file, _ = _write_archive(
        tmp_path,
        sha=sha,
        email="parent@example.com",
        text="Здравствуйте, интересует обучение.",
    )
    _seed_pending_event(db_path, tmp_path, sha=sha, source_file=source_file, subject="Фотон")
    first = run_mail_link_enrich(
        MailLinkEnrichConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            out_dir=tmp_path / "first",
            apply=True,
        )
    )
    assert first["apply"]["counts"]["created_chunks"] == 1
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        store.upsert_identity_link(
            IdentityLink(
                tenant_id="foton",
                customer_id="customer:email",
                link_type="email",
                link_value="parent@example.com",
                source_system="tallanto_snapshot",
                source_ref="test",
                confidence=0.5,
                match_class="ambiguous",
            ),
            actor="test",
        )

    second = run_mail_link_enrich(
        MailLinkEnrichConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            out_dir=tmp_path / "second",
            apply=True,
            reconsider_pending=True,
        )
    )

    assert second["apply"]["counts"]["revoked_chunks"] == 1
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as con:
        event = con.execute(
            "SELECT customer_id, match_status FROM timeline_events WHERE source_id=?",
            (sha,),
        ).fetchone()
        active_chunks = con.execute(
            """
            SELECT count(*)
            FROM bot_context_chunks
            WHERE event_id=(SELECT event_id FROM timeline_events WHERE source_id=?)
              AND superseded_by IS NULL
            """,
            (sha,),
        ).fetchone()[0]
    assert event == (None, "unmatched")
    assert active_chunks == 0


def test_mail_link_enrich_keeps_mail_derived_email_weak(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    _seed_customer_with_links(db_path, tmp_path, email_source_system=A2V3_MAIL_SOURCE_SYSTEM)
    sha = "a" * 64
    source_file, _ = _write_archive(
        tmp_path,
        sha=sha,
        email="parent@example.com",
        text="Здравствуйте, интересует обучение.",
    )
    _seed_pending_event(db_path, tmp_path, sha=sha, source_file=source_file, subject="Фотон")

    report = run_mail_link_enrich(
        MailLinkEnrichConfig(timeline_db=db_path, allowed_root=tmp_path, out_dir=tmp_path / "out")
    )

    assert report["counts"]["planned.weak_email"] == 1
    assert report["counts"]["reason.email_unique_identity_link"] == 1


def test_mail_link_enrich_requires_contact_to_confirm_rfc_thread_anchor(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    _seed_customer_with_links(db_path, tmp_path)
    anchor_sha = "b" * 64
    target_sha = "c" * 64
    archive_db = _write_thread_archive(
        tmp_path,
        target_sha=target_sha,
        reference_messages=[(anchor_sha, "anchor@example.test")],
    )
    _seed_linked_mail_event(db_path, tmp_path, sha=anchor_sha, customer_id="customer:phone")
    _seed_pending_event_with_archive_db(
        db_path,
        tmp_path,
        sha=target_sha,
        archive_db=archive_db,
        subject="Re: Запись",
    )

    report = run_mail_link_enrich(
        MailLinkEnrichConfig(timeline_db=db_path, allowed_root=tmp_path, out_dir=tmp_path / "out")
    )

    assert report["counts"]["planned.unmatched"] == 1
    assert report["counts"]["reason.inbound_no_external_from"] == 1


def test_mail_link_enrich_uses_rfc_thread_when_contact_confirms_customer(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    _seed_customer_with_links(db_path, tmp_path)
    anchor_sha = "6" * 64
    target_sha = "7" * 64
    archive_db = _write_thread_archive(
        tmp_path,
        target_sha=target_sha,
        reference_messages=[(anchor_sha, "anchor@example.test")],
    )
    with sqlite3.connect(archive_db) as con:
        con.execute(
            "INSERT INTO message_participants VALUES (?, 'from', 'Parent', 'parent@example.com', 'example.com')",
            (target_sha,),
        )
    _seed_linked_mail_event(db_path, tmp_path, sha=anchor_sha, customer_id="customer:email")
    _seed_pending_event_with_archive_db(
        db_path,
        tmp_path,
        sha=target_sha,
        archive_db=archive_db,
        subject="Re: Запись",
    )

    report = run_mail_link_enrich(
        MailLinkEnrichConfig(timeline_db=db_path, allowed_root=tmp_path, out_dir=tmp_path / "out")
    )

    assert report["counts"]["planned.strong"] == 1
    assert report["counts"]["reason.strong_thread_header_identity_link"] == 1


def test_mail_link_enrich_blocks_rfc_thread_with_two_customers(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    _seed_customer_with_links(db_path, tmp_path)
    first_sha = "d" * 64
    second_sha = "e" * 64
    target_sha = "f" * 64
    archive_db = _write_thread_archive(
        tmp_path,
        target_sha=target_sha,
        reference_messages=[
            (first_sha, "first@example.test"),
            (second_sha, "second@example.test"),
        ],
    )
    _seed_linked_mail_event(db_path, tmp_path, sha=first_sha, customer_id="customer:phone")
    _seed_linked_mail_event(db_path, tmp_path, sha=second_sha, customer_id="customer:email")
    _seed_pending_event_with_archive_db(
        db_path,
        tmp_path,
        sha=target_sha,
        archive_db=archive_db,
        subject="Re: Запись",
    )

    report = run_mail_link_enrich(
        MailLinkEnrichConfig(timeline_db=db_path, allowed_root=tmp_path, out_dir=tmp_path / "out")
    )

    assert report["counts"]["planned.blocked"] == 1
    assert report["counts"]["reason.thread_customer_conflict"] == 1


def test_mail_link_enrich_checks_all_duplicate_message_id_anchors(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    _seed_customer_with_links(db_path, tmp_path)
    first_sha = "1" * 64
    second_sha = "2" * 64
    target_sha = "3" * 64
    archive_db = _write_thread_archive(
        tmp_path,
        target_sha=target_sha,
        reference_messages=[
            (first_sha, "duplicate@example.test"),
            (second_sha, "duplicate@example.test"),
        ],
    )
    _seed_linked_mail_event(db_path, tmp_path, sha=first_sha, customer_id="customer:phone")
    _seed_linked_mail_event(db_path, tmp_path, sha=second_sha, customer_id="customer:email")
    _seed_pending_event_with_archive_db(
        db_path,
        tmp_path,
        sha=target_sha,
        archive_db=archive_db,
        subject="Re: Запись",
    )

    report = run_mail_link_enrich(
        MailLinkEnrichConfig(timeline_db=db_path, allowed_root=tmp_path, out_dir=tmp_path / "out")
    )

    assert report["counts"]["planned.blocked"] == 1
    assert report["counts"]["reason.thread_customer_conflict"] == 1


def test_mail_link_enrich_does_not_pick_child_from_shared_family_contact(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    _seed_customer_with_links(db_path, tmp_path)
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        store.upsert_identity_link(
            IdentityLink(
                tenant_id="foton",
                customer_id="customer:phone",
                link_type="email",
                link_value="parent@example.com",
                source_system="tallanto_snapshot",
                source_ref="family",
                confidence=0.5,
                match_class="ambiguous",
            ),
            actor="test",
        )
    anchor_sha = "4" * 64
    target_sha = "5" * 64
    archive_db = _write_thread_archive(
        tmp_path,
        target_sha=target_sha,
        reference_messages=[(anchor_sha, "family-thread@example.test")],
    )
    with sqlite3.connect(archive_db) as con:
        con.execute(
            "INSERT INTO message_participants VALUES (?, 'from', 'Parent', 'parent@example.com', 'example.com')",
            (target_sha,),
        )
    _seed_linked_mail_event(db_path, tmp_path, sha=anchor_sha, customer_id="customer:phone")
    _seed_pending_event_with_archive_db(
        db_path,
        tmp_path,
        sha=target_sha,
        archive_db=archive_db,
        subject="Re: Запись",
    )

    report = run_mail_link_enrich(
        MailLinkEnrichConfig(timeline_db=db_path, allowed_root=tmp_path, out_dir=tmp_path / "out")
    )

    assert report["counts"]["planned.blocked"] == 1
    assert report["counts"]["reason.thread_contact_ambiguous"] == 1


def test_mail_link_enrich_blocks_links_to_partial_customer(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    _seed_customer_with_links(db_path, tmp_path)
    with sqlite3.connect(db_path) as con:
        con.execute("UPDATE customer_identities SET identity_status='partial' WHERE customer_id='customer:phone'")
    sha = "6" * 64
    source_file, _ = _write_archive(
        tmp_path,
        sha=sha,
        email="unlinked@example.com",
        text="Здравствуйте.\n\nС уважением,\n+7 916 123-45-67",
    )
    _seed_pending_event(db_path, tmp_path, sha=sha, source_file=source_file, subject="Фотон")

    report = run_mail_link_enrich(
        MailLinkEnrichConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            out_dir=tmp_path / "out",
            aggregate_only=True,
        )
    )

    assert report["counts"]["planned.blocked"] == 1
    assert report["counts"]["reason.phone_customer_identity_not_strong"] == 1
    assert not (tmp_path / "out" / "mail_link_enrich_decisions.jsonl").exists()
    assert (tmp_path / "out").stat().st_mode & 0o777 == 0o700


def test_mail_link_enrich_apply_aggregate_only_exposes_no_row_ids(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    _seed_customer_with_links(db_path, tmp_path)
    sha = "5" * 64
    source_file, _ = _write_archive(
        tmp_path,
        sha=sha,
        email="unlinked@example.com",
        text="Здравствуйте.\n\nС уважением,\n+7 916 123-45-67",
    )
    _seed_pending_event(db_path, tmp_path, sha=sha, source_file=source_file, subject="Фотон")
    with sqlite3.connect(db_path) as con:
        event_id = con.execute("SELECT event_id FROM timeline_events WHERE source_id=?", (sha,)).fetchone()[0]

    report = run_mail_link_enrich(
        MailLinkEnrichConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            out_dir=tmp_path / "out",
            apply=True,
            aggregate_only=True,
        )
    )

    report_path = tmp_path / "out" / "mail_link_enrich_apply_report.json"
    report_text = report_path.read_text(encoding="utf-8")
    assert report["apply"] == {"counts": report["apply"]["counts"]}
    assert not (tmp_path / "out" / "mail_link_enrich_decisions.jsonl").exists()
    assert event_id not in report_text
    assert "customer:phone" not in report_text
    with sqlite3.connect(db_path) as con:
        run_payload = json.loads(
            con.execute(
                "SELECT record_json FROM ingestion_runs WHERE run_kind='mail_link_enrich'"
            ).fetchone()[0]
        )
    assert run_payload["metadata"] == {"counts": report["apply"]["counts"]}
    assert event_id not in json.dumps(run_payload["metadata"], sort_keys=True)
    assert "customer:phone" not in json.dumps(run_payload["metadata"], sort_keys=True)


def test_mail_link_enrich_rolls_back_run_and_data_on_apply_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    _seed_customer_with_links(db_path, tmp_path)
    sha = "7" * 64
    source_file, _ = _write_archive(
        tmp_path,
        sha=sha,
        email="unlinked@example.com",
        text="Здравствуйте.\n\nС уважением,\n+7 916 123-45-67",
    )
    _seed_pending_event(db_path, tmp_path, sha=sha, source_file=source_file, subject="Фотон")

    def fail_update(*_args: object, **_kwargs: object) -> TimelineEvent:
        raise RuntimeError("forced apply failure")

    monkeypatch.setattr(mail_link_enrich_module, "_updated_event_from_decision", fail_update)
    with pytest.raises(RuntimeError, match="forced apply failure"):
        run_mail_link_enrich(
            MailLinkEnrichConfig(
                timeline_db=db_path,
                allowed_root=tmp_path,
                out_dir=tmp_path / "out",
                apply=True,
                aggregate_only=True,
            )
        )

    with sqlite3.connect(db_path) as con:
        assert con.execute("SELECT COUNT(*) FROM ingestion_runs WHERE run_kind='mail_link_enrich'").fetchone()[0] == 0
        event_payload = json.loads(
            con.execute("SELECT record_json FROM timeline_events WHERE source_id=?", (sha,)).fetchone()[0]
        )
    assert event_payload["customer_id"] is None
