from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

import mango_mvp.customer_timeline.mail_link_enrich as mail_link_enrich_module
import scripts.run_mail_link_enrich as mail_link_enrich_runner
from mango_mvp.customer_timeline.a2_mail_ingest import A2V3_MAIL_SOURCE_SYSTEM
from mango_mvp.customer_timeline.contracts import (
    CustomerIdentity,
    IdentityLink,
    IdentityStatus,
    TimelineDirection,
    TimelineEvent,
)
from mango_mvp.customer_timeline.family_graph import FamilyGraphConfig, build_family_graph
from mango_mvp.customer_timeline.mail_link_enrich import (
    MailLinkEnrichConfig,
    _contact_from_archive_row,
    run_mail_link_enrich,
)
from mango_mvp.customer_timeline.store import CustomerTimelineSQLiteStore


NOW = datetime(2026, 7, 7, 12, 0, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    ("flag", "reconsider_pending", "revalidate_existing_strong"),
    (
        ("--reconsider-pending", True, False),
        ("--revalidate-existing-strong", False, True),
    ),
)
def test_mail_link_enrich_runner_separates_pending_and_strong_modes(
    monkeypatch: pytest.MonkeyPatch,
    flag: str,
    reconsider_pending: bool,
    revalidate_existing_strong: bool,
) -> None:
    configs: list[MailLinkEnrichConfig] = []
    monkeypatch.setattr(
        mail_link_enrich_runner,
        "run_mail_link_enrich",
        lambda config: configs.append(config) or {"safety": {}},
    )

    assert mail_link_enrich_runner.main([flag]) == 0
    assert configs[0].reconsider_pending is reconsider_pending
    assert configs[0].revalidate_existing_strong is revalidate_existing_strong


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
    # NOTE: source_file/archive_db are fixed (non-sha-parameterized) paths by
    # design: production has exactly one stage2 delta file and one shared
    # archive db per allowed_root, and several tests call this helper more
    # than once against the same tmp_path to seed multiple messages into one
    # shared archive (mirrors real multi-message archives). mkdir(exist_ok)
    # + CREATE TABLE IF NOT EXISTS make repeated calls safe/idempotent; the
    # single-call tests are unaffected since both are no-ops on first call.
    handoff = tmp_path / "handoff"
    source_file = handoff / "stage2_delta_ingest" / "stage2_delta_full_events.jsonl"
    source_file.parent.mkdir(parents=True, exist_ok=True)
    source_file.write_text("", encoding="utf-8")
    archive_db = handoff / "archive" / "mail_archive.sqlite"
    archive_db.parent.mkdir(parents=True, exist_ok=True)
    text_path = handoff / "archive" / f"{sha}.txt"
    text_path.write_text(text, encoding="utf-8")
    with sqlite3.connect(archive_db) as con:
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS messages (
              sha256 TEXT PRIMARY KEY,
              subject TEXT,
              extracted_text_path TEXT
            );
            CREATE TABLE IF NOT EXISTS message_participants (
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


def _write_tallanto_identity_db(
    path: Path,
    *,
    email: str,
    tallanto_id: str,
    candidate_key: str,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as con:
        con.executescript(
            """
            CREATE TABLE identity_values (
              kind TEXT,
              value TEXT,
              match_class TEXT,
              candidate_count INTEGER
            );
            CREATE TABLE identity_candidates (
              candidate_key TEXT,
              tallanto_id TEXT
            );
            CREATE TABLE identity_links (
              kind TEXT,
              value TEXT,
              candidate_key TEXT
            );
            """
        )
        con.execute("INSERT INTO identity_values VALUES ('email', ?, 'strong_unique', 1)", (email,))
        con.execute("INSERT INTO identity_candidates VALUES (?, ?)", (candidate_key, tallanto_id))
        con.execute("INSERT INTO identity_links VALUES ('email', ?, ?)", (email, candidate_key))
    return path


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


@pytest.mark.parametrize(("customer_brand", "expected_authorized"), (("foton", True), ("unpk", False)))
def test_mail_link_enrich_keeps_identity_but_separates_cross_brand_context(
    tmp_path: Path,
    customer_brand: str,
    expected_authorized: bool,
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

    assert report["counts"]["planned.strong"] == 1
    assert report["counts"]["transition.unmatched.strong"] == 1
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as con:
        event = con.execute(
            "SELECT customer_id, match_status, record_json FROM timeline_events WHERE source_id=?",
            (sha,),
        ).fetchone()
    assert event[0] == "customer:phone"
    assert event[1] == "strong_unique"
    assert json.loads(event[2])["metadata"]["brand_context_authorized"] is expected_authorized


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
    (("tallanto_snapshot", "strong"), ("mail_archive_stage2", "blocked")),
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


def test_mail_link_enrich_never_promotes_name_match_to_strong_without_exact_email_or_phone(tmp_path: Path) -> None:
    """BLOK C2 lock-in: a sender display_name that exactly (let alone fuzzily)
    matches a known customer's display_name must never by itself produce a
    strong/family_strong link. Only exact email, exact phone, an already-proven
    family chain, or an intersection of exact keys may do that (см. ТЗ: «Нечёткое
    имя — НЕ strong»). The message below carries the identical display_name as an
    existing strong customer but an unrelated, unlinked email and no phone at all,
    so it must stay unlinked."""
    db_path = tmp_path / "customer_timeline.sqlite"
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        store.upsert_customer(
            CustomerIdentity(
                tenant_id="foton",
                customer_id="customer:knownname",
                identity_status=IdentityStatus.STRONG,
                display_name="Иванова Мария Петровна",
                primary_email="realparent@example.com",
            )
        )
        store.upsert_identity_link(
            IdentityLink(
                tenant_id="foton",
                customer_id="customer:knownname",
                link_type="email",
                link_value="realparent@example.com",
                source_system="test",
                source_ref="test",
                confidence=0.95,
            )
        )
    sha = "b" * 64
    handoff = tmp_path / "handoff"
    source_file = handoff / "stage2_delta_ingest" / "stage2_delta_full_events.jsonl"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("", encoding="utf-8")
    archive_db = handoff / "archive" / "mail_archive.sqlite"
    archive_db.parent.mkdir(parents=True)
    text_path = handoff / "archive" / f"{sha}.txt"
    text_path.write_text("Здравствуйте, интересует Фотон для ребёнка.", encoding="utf-8")
    with sqlite3.connect(archive_db) as con:
        con.executescript(
            """
            CREATE TABLE messages (sha256 TEXT PRIMARY KEY, subject TEXT, extracted_text_path TEXT);
            CREATE TABLE message_participants (
              message_sha256 TEXT, header_name TEXT, display_name TEXT, email_normalized TEXT, domain TEXT
            );
            """
        )
        con.execute("INSERT INTO messages VALUES (?, ?, ?)", (sha, "Запись в Фотон", str(text_path)))
        con.execute(
            "INSERT INTO message_participants VALUES (?, 'from', ?, ?, 'unlinked.example')",
            (sha, "Иванова Мария Петровна", "unlinked-sender@unlinked.example"),
        )
    _seed_pending_event(db_path, tmp_path, sha=sha, source_file=source_file, subject="Фотон")

    report = run_mail_link_enrich(
        MailLinkEnrichConfig(timeline_db=db_path, allowed_root=tmp_path, out_dir=tmp_path / "out", apply=True)
    )

    assert report["counts"].get("planned.strong", 0) == 0
    assert report["counts"].get("planned.family_strong", 0) == 0
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as con:
        con.row_factory = sqlite3.Row
        event = con.execute(
            "SELECT customer_id, match_status FROM timeline_events WHERE source_id=?", (sha,)
        ).fetchone()
    assert event["customer_id"] is None
    assert event["match_status"] != "strong_unique"


def test_mail_link_enrich_apply_rerun_on_same_input_is_idempotent(tmp_path: Path) -> None:
    """BLOK C2 idempotency: revalidating an already-linked mail event on an
    unchanged DB must not flip customer_id/match_status or create new chunks."""
    db_path = tmp_path / "customer_timeline.sqlite"
    _seed_customer_with_links(db_path, tmp_path)
    sha = "e" * 64
    source_file, _ = _write_archive(
        tmp_path,
        sha=sha,
        email="unknown@example.com",
        text="Здравствуйте, интересует Фотон.\n\n+7 916 123-45-67",
    )
    _seed_pending_event(db_path, tmp_path, sha=sha, source_file=source_file, subject="Фотон")

    first = run_mail_link_enrich(
        MailLinkEnrichConfig(timeline_db=db_path, allowed_root=tmp_path, out_dir=tmp_path / "out1", apply=True)
    )
    assert first["counts"]["planned.strong"] == 1
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as con:
        con.row_factory = sqlite3.Row
        after_first = dict(
            con.execute(
                "SELECT customer_id, match_status FROM timeline_events WHERE source_id=?", (sha,)
            ).fetchone()
        )
        chunks_after_first = con.execute("SELECT COUNT(*) FROM bot_context_chunks").fetchone()[0]

    second = run_mail_link_enrich(
        MailLinkEnrichConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            out_dir=tmp_path / "out2",
            apply=True,
            revalidate_existing_strong=True,
        )
    )
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as con:
        con.row_factory = sqlite3.Row
        after_second = dict(
            con.execute(
                "SELECT customer_id, match_status FROM timeline_events WHERE source_id=?", (sha,)
            ).fetchone()
        )
        chunks_after_second = con.execute("SELECT COUNT(*) FROM bot_context_chunks").fetchone()[0]

    assert second["target_events"] == 1
    assert after_second == after_first
    assert chunks_after_second == chunks_after_first
    assert second["safety"]["allowed_for_bot_changed"] is False
    assert second["safety"]["mail_stage2_allowed_for_bot_changed"] is False


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


@pytest.mark.parametrize(
    ("link_type", "link_value", "email", "text"),
    (
        ("email", "new-parent@example.com", "new-parent@example.com", "Здравствуйте, интересует обучение."),
        ("phone", "+79161234567", "unlinked@example.com", "Здравствуйте.\n\nС уважением,\n+7 916 123-45-67"),
    ),
)
def test_mail_link_enrich_reconsiders_old_pending_after_tallanto_identity_refresh(
    tmp_path: Path,
    link_type: str,
    link_value: str,
    email: str,
    text: str,
) -> None:
    db_path = tmp_path / ".codex_local" / "staging" / "customer_timeline.sqlite"
    db_path.parent.mkdir(parents=True)
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        store.upsert_customer(
            CustomerIdentity(
                tenant_id="foton",
                customer_id="customer:tallanto",
                identity_status=IdentityStatus.STRONG,
                display_name="Tallanto Parent",
            )
        )
    sha = "f" * 64
    source_file, _ = _write_archive(
        tmp_path,
        sha=sha,
        email=email,
        text=text,
    )
    _seed_pending_event(db_path, tmp_path, sha=sha, source_file=source_file, subject="Фотон")
    initial = run_mail_link_enrich(
        MailLinkEnrichConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            out_dir=tmp_path / "initial",
            apply=True,
        )
    )
    assert initial["counts"]["planned.unmatched"] == 1

    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        store.upsert_identity_link(
            IdentityLink(
                tenant_id="foton",
                customer_id="customer:tallanto",
                link_type=link_type,
                link_value=link_value,
                source_system="tallanto_snapshot",
                source_ref="tallanto:student:new",
                match_class="strong_unique",
                confidence=1.0,
            )
        )

    not_reconsidered = run_mail_link_enrich(
        MailLinkEnrichConfig(timeline_db=db_path, allowed_root=tmp_path, out_dir=tmp_path / "not_reconsidered")
    )
    reconsidered = run_mail_link_enrich(
        MailLinkEnrichConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            out_dir=tmp_path / "reconsidered",
            apply=True,
            reconsider_pending=True,
        )
    )
    rerun = run_mail_link_enrich(
        MailLinkEnrichConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            out_dir=tmp_path / "rerun",
            apply=True,
            reconsider_pending=True,
        )
    )

    assert not_reconsidered["target_events"] == 0
    assert reconsidered["target_events"] == 1
    assert reconsidered["counts"]["planned.strong"] == 1
    assert rerun["target_events"] == 0
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as con:
        event = con.execute(
            "SELECT customer_id, match_status FROM timeline_events WHERE source_id=?",
            (sha,),
        ).fetchone()
        event_count = con.execute(
            "SELECT COUNT(*) FROM timeline_events WHERE source_id=?",
            (sha,),
        ).fetchone()[0]
    assert event == ("customer:tallanto", "strong_unique")
    assert event_count == 1


def test_mail_link_enrich_reconsider_pending_does_not_select_existing_strong(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    _seed_customer_with_links(db_path, tmp_path)
    sha = "1" * 64
    source_file, _ = _write_archive(
        tmp_path,
        sha=sha,
        email="unlinked@example.com",
        text="Здравствуйте.\n\nС уважением,\n+7 916 123-45-67",
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
    assert first["counts"]["planned.strong"] == 1

    reconsidered = run_mail_link_enrich(
        MailLinkEnrichConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            out_dir=tmp_path / "reconsidered",
            apply=True,
            reconsider_pending=True,
        )
    )

    assert reconsidered["target_events"] == 0
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as con:
        event = con.execute(
            "SELECT customer_id, match_status FROM timeline_events WHERE source_id=?",
            (sha,),
        ).fetchone()
    assert event == ("customer:phone", "strong_unique")


def test_mail_link_enrich_reconsiders_cross_brand_identity_but_keeps_context_blocked(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    _seed_customer_with_links(db_path, tmp_path, email_source_system="tallanto_snapshot")
    _seed_trusted_customer_brand(db_path, customer_id="customer:email", brand="unpk")
    sha = "9" * 64
    source_file, _ = _write_archive(
        tmp_path,
        sha=sha,
        email="parent@example.com",
        text="Здравствуйте.",
    )
    _seed_pending_event(db_path, tmp_path, sha=sha, source_file=source_file, subject="Фотон")
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
            apply=True,
        )
    )

    assert report["target_events"] == 1
    assert report["counts"]["planned.strong"] == 1
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as con:
        event = con.execute(
            "SELECT customer_id, match_status, record_json FROM timeline_events WHERE source_id=?",
            (sha,),
        ).fetchone()
    assert event[0] == "customer:email"
    assert event[1] == "strong_unique"
    assert json.loads(event[2])["metadata"]["brand_context_authorized"] is False


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
        MailLinkEnrichConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            out_dir=tmp_path / "out",
            aggregate_only=True,
        )
    )

    assert report["counts"]["planned.strong"] == 1
    assert report["counts"]["reason.strong_external_email_identity_link"] == 1
    assert report["breakdown"]["exact_tallanto"] == 1


def test_mail_link_enrich_breakdown_counts_exact_amo_email_separately(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    _seed_customer_with_links(db_path, tmp_path, email_source_system="amocrm_snapshot")
    sha = "9" * 64
    source_file, _ = _write_archive(
        tmp_path,
        sha=sha,
        email="parent@example.com",
        text="Здравствуйте, интересует обучение.",
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

    assert report["counts"]["planned.strong"] == 1
    assert report["breakdown"]["exact_amo"] == 1
    assert report["breakdown"]["exact_tallanto"] == 0


def test_mail_link_enrich_promotes_shared_tallanto_parent_email_to_family(tmp_path: Path) -> None:
    db_path = tmp_path / ".codex_local" / "staging" / "customer_timeline.sqlite"
    db_path.parent.mkdir(parents=True)
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        for index, customer_id in enumerate(("customer:child-a", "customer:child-b"), start=1):
            store.upsert_customer(
                CustomerIdentity(
                    tenant_id="foton",
                    customer_id=customer_id,
                    identity_status=IdentityStatus.STRONG,
                    display_name=f"Child {index}",
                    primary_email="parent@example.com",
                )
            )
            store.upsert_identity_link(
                IdentityLink(
                    tenant_id="foton",
                    customer_id=customer_id,
                    link_type="tallanto_student_id",
                    link_value=f"tallanto-{index}",
                    source_system="tallanto_snapshot",
                    source_ref=f"tallanto:{index}",
                    confidence=1.0,
                )
            )
            store.upsert_identity_link(
                IdentityLink(
                    tenant_id="foton",
                    customer_id=customer_id,
                    link_type="email",
                    link_value="parent@example.com",
                    source_system="tallanto_snapshot",
                    source_ref=f"tallanto:{index}",
                    confidence=0.8,
                    match_class="ambiguous",
                )
            )
            store.upsert_event(
                TimelineEvent(
                    tenant_id="foton",
                    customer_id=customer_id,
                    event_type="tallanto_student_snapshot",
                    event_at=NOW,
                    source_system="tallanto_snapshot",
                    source_id=f"tallanto-{index}",
                    direction="system",
                    match_status="strong_unique",
                    record={"payload": {"parent_fio": "Ирина Иванова"}},
                )
            )
        store.upsert_identity_link(
            IdentityLink(
                tenant_id="foton",
                customer_id="customer:child-b",
                link_type="email",
                link_value="parent@example.com",
                source_system="amocrm_snapshot",
                source_ref="amo:parent",
                confidence=1.0,
            )
        )
    sha = "b" * 64
    source_file, _ = _write_archive(
        tmp_path,
        sha=sha,
        email="parent@example.com",
        text="Здравствуйте, интересует обучение для детей.",
    )
    _seed_pending_event(db_path, tmp_path, sha=sha, source_file=source_file, subject="Фотон")
    build_family_graph(FamilyGraphConfig(timeline_db=db_path, allowed_root=tmp_path, apply=True))

    report = run_mail_link_enrich(
        MailLinkEnrichConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            out_dir=tmp_path / "out",
            apply=True,
        )
    )

    assert report["counts"]["planned.family_strong"] == 1
    assert report["counts"]["reason.strong_tallanto_family_identity_link"] == 1
    assert report["breakdown"]["family"] == 1
    with sqlite3.connect(db_path) as con:
        event = json.loads(con.execute("SELECT record_json FROM timeline_events WHERE source_id=?", (sha,)).fetchone()[0])
    assert event["customer_id"] is None
    assert event["metadata"]["family_id"].startswith("family:")


# --- D2 rule 4: an email shared by two *different* families is an evidenced
# conflict, not a first-match; D2 rule 7: exact_amo/exact_tallanto/
# thread_propagation/ambiguous/unmatched breakdown. ---


def test_mail_link_enrich_two_families_sharing_one_email_is_conflict_not_first_match(tmp_path: Path) -> None:
    db_path = tmp_path / ".codex_local" / "staging" / "customer_timeline.sqlite"
    db_path.parent.mkdir(parents=True)
    shared_email = "shared-office@example.com"
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        for index, (customer_id, family_id) in enumerate(
            (("customer:family-a", "family:A"), ("customer:family-b", "family:B")), start=1
        ):
            store.upsert_customer(
                CustomerIdentity(
                    tenant_id="foton",
                    customer_id=customer_id,
                    identity_status=IdentityStatus.STRONG,
                    display_name=f"Family {index}",
                    primary_email=shared_email,
                )
            )
            store.upsert_identity_link(
                IdentityLink(
                    tenant_id="foton",
                    customer_id=customer_id,
                    link_type="email",
                    link_value=shared_email,
                    source_system="tallanto_snapshot",
                    source_ref=f"tallanto:{index}",
                    match_class="strong_unique",
                    confidence=1.0,
                )
            )
        con = store._con
        con.executemany(
            "INSERT INTO family_members_v1 "
            "(tenant_id,family_id,customer_id,membership_status,confidence,reason,created_at,updated_at,record_hash,record_json) "
            "VALUES ('foton',?,?,'confident','high','test','2026-07-24T00:00:00+00:00',"
            "'2026-07-24T00:00:00+00:00','test','{}')",
            (("family:A", "customer:family-a"), ("family:B", "customer:family-b")),
        )
        con.commit()
    sha = "c" * 64
    source_file, _ = _write_archive(
        tmp_path,
        sha=sha,
        email=shared_email,
        text="Здравствуйте, интересует обучение.",
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
    assert report["counts"]["reason.email_multiple_families_conflict"] == 1
    assert report["breakdown"]["ambiguous"] == 1
    with sqlite3.connect(db_path) as con:
        event = json.loads(con.execute("SELECT record_json FROM timeline_events WHERE source_id=?", (sha,)).fetchone()[0])
    # Never a first-match: no customer picked for either family.
    assert event["customer_id"] is None
    assert event["customer_id"] not in {"customer:family-a", "customer:family-b"}


def test_mail_link_enrich_breakdown_separates_exact_amo_from_exact_tallanto(tmp_path: Path) -> None:
    db_path = tmp_path / ".codex_local" / "staging" / "customer_timeline.sqlite"
    db_path.parent.mkdir(parents=True)
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        store.upsert_customer(
            CustomerIdentity(
                tenant_id="foton",
                customer_id="customer:amo-parent",
                identity_status=IdentityStatus.STRONG,
                display_name="AMO Parent",
                primary_email="amo-parent@example.com",
            )
        )
        store.upsert_identity_link(
            IdentityLink(
                tenant_id="foton",
                customer_id="customer:amo-parent",
                link_type="email",
                link_value="amo-parent@example.com",
                source_system="amocrm_snapshot",
                source_ref="amo:parent",
                match_class="strong_unique",
                confidence=1.0,
            )
        )
        store.upsert_customer(
            CustomerIdentity(
                tenant_id="foton",
                customer_id="customer:tallanto-parent",
                identity_status=IdentityStatus.STRONG,
                display_name="Tallanto Parent",
                primary_email="tallanto-parent@example.com",
            )
        )
        store.upsert_identity_link(
            IdentityLink(
                tenant_id="foton",
                customer_id="customer:tallanto-parent",
                link_type="email",
                link_value="tallanto-parent@example.com",
                source_system="tallanto_snapshot",
                source_ref="tallanto:parent",
                match_class="strong_unique",
                confidence=1.0,
            )
        )
    amo_sha = "d" * 64
    amo_file, _ = _write_archive(tmp_path, sha=amo_sha, email="amo-parent@example.com", text="Вопрос про Фотон.")
    _seed_pending_event(db_path, tmp_path, sha=amo_sha, source_file=amo_file, subject="Фотон")
    tallanto_sha = "e" * 64
    tallanto_file, _ = _write_archive(
        tmp_path, sha=tallanto_sha, email="tallanto-parent@example.com", text="Вопрос про занятия."
    )
    _seed_pending_event(db_path, tmp_path, sha=tallanto_sha, source_file=tallanto_file, subject="Фотон")

    report = run_mail_link_enrich(
        MailLinkEnrichConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            out_dir=tmp_path / "out",
            aggregate_only=True,
        )
    )

    assert report["breakdown"]["exact_amo"] == 1
    assert report["breakdown"]["exact_tallanto"] == 1
    assert report["breakdown"]["ambiguous"] == 0
    assert report["breakdown"]["unmatched"] == 0
    assert sum(report["breakdown"].values()) == report["target_events"]


def test_mail_link_enrich_thread_propagation_is_its_own_breakdown_bucket(tmp_path: Path) -> None:
    # Re-uses the already-covered "strong contact confirms rfc thread"
    # fixture pattern (see test_mail_link_enrich_uses_rfc_thread_when_
    # contact_confirms_customer) purely to assert the new D2 rule 7
    # breakdown bucketing: a thread_header_identity_link decision must land
    # in breakdown["thread_propagation"], not exact_amo/exact_tallanto.
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
    assert report["breakdown"]["thread_propagation"] == 1


def test_mail_link_enrich_uses_historical_tallanto_email_for_same_student(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        store.upsert_customer(
            CustomerIdentity(
                tenant_id="foton",
                customer_id="customer:historical",
                identity_status=IdentityStatus.STRONG,
                display_name="Historical Parent",
            )
        )
        store.upsert_identity_link(
            IdentityLink(
                tenant_id="foton",
                customer_id="customer:historical",
                link_type="tallanto_student_id",
                link_value="tallanto-old-1",
                source_system="tallanto_snapshot",
                source_ref="tallanto:current",
                confidence=1.0,
            )
        )
    identity_db = _write_tallanto_identity_db(
        tmp_path / "identity.sqlite",
        email="old-parent@example.com",
        tallanto_id="tallanto-old-1",
        candidate_key="candidate-old-1",
    )
    sha = "c" * 64
    source_file, _ = _write_archive(
        tmp_path,
        sha=sha,
        email="old-parent@example.com",
        text="Здравствуйте, хотим вернуться к занятиям.",
    )
    _seed_pending_event(db_path, tmp_path, sha=sha, source_file=source_file, subject="Фотон")

    report = run_mail_link_enrich(
        MailLinkEnrichConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            out_dir=tmp_path / "out",
            tallanto_identity_dbs=(identity_db,),
            apply=True,
        )
    )

    assert report["counts"]["planned.strong"] == 1
    assert report["counts"]["reason.strong_historical_tallanto_email_identity_link"] == 1
    assert report["historical_tallanto_identity"]["usable_email_values"] == 1
    with sqlite3.connect(db_path) as con:
        event = json.loads(con.execute("SELECT record_json FROM timeline_events WHERE source_id=?", (sha,)).fetchone()[0])
    assert event["customer_id"] == "customer:historical"


def test_mail_link_enrich_blocks_conflicting_historical_tallanto_email(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        for index in (1, 2):
            customer_id = f"customer:historical-{index}"
            store.upsert_customer(
                CustomerIdentity(
                    tenant_id="foton",
                    customer_id=customer_id,
                    identity_status=IdentityStatus.STRONG,
                    display_name=f"Historical Parent {index}",
                )
            )
            store.upsert_identity_link(
                IdentityLink(
                    tenant_id="foton",
                    customer_id=customer_id,
                    link_type="tallanto_student_id",
                    link_value=f"tallanto-old-{index}",
                    source_system="tallanto_snapshot",
                    source_ref=f"tallanto:{index}",
                    confidence=1.0,
                )
            )
            store.upsert_event(
                TimelineEvent(
                    tenant_id="foton",
                    customer_id=customer_id,
                    event_type="tallanto_student_snapshot",
                    event_at=NOW,
                    source_system="tallanto_snapshot",
                    source_id=f"family-student-{index}",
                    direction="system",
                    match_status="strong_unique",
                    record={"payload": {"parent_fio": "Ирина Иванова"}},
                )
            )
    identity_dbs = tuple(
        _write_tallanto_identity_db(
            tmp_path / f"identity-{index}.sqlite",
            email="old-parent@example.com",
            tallanto_id=f"tallanto-old-{index}",
            candidate_key=f"candidate-old-{index}",
        )
        for index in (1, 2)
    )
    sha = "d" * 64
    source_file, _ = _write_archive(
        tmp_path,
        sha=sha,
        email="old-parent@example.com",
        text="Здравствуйте, интересует обучение.",
    )
    _seed_pending_event(db_path, tmp_path, sha=sha, source_file=source_file, subject="Фотон")

    report = run_mail_link_enrich(
        MailLinkEnrichConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            out_dir=tmp_path / "out",
            tallanto_identity_dbs=identity_dbs,
            apply=True,
        )
    )

    assert report["counts"].get("planned.strong", 0) == 0
    assert report["counts"]["reason.historical_tallanto_email_cross_family"] == 1
    with sqlite3.connect(db_path) as con:
        event = json.loads(con.execute("SELECT record_json FROM timeline_events WHERE source_id=?", (sha,)).fetchone()[0])
    assert event["customer_id"] is None


def test_mail_link_enrich_accepts_historical_email_for_one_tallanto_family(tmp_path: Path) -> None:
    db_path = tmp_path / ".codex_local" / "staging" / "customer_timeline.sqlite"
    db_path.parent.mkdir(parents=True)
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        for index in (1, 2):
            customer_id = f"customer:family-child-{index}"
            store.upsert_customer(
                CustomerIdentity(
                    tenant_id="foton",
                    customer_id=customer_id,
                    identity_status=IdentityStatus.STRONG,
                )
            )
            store.upsert_identity_link(
                IdentityLink(
                    tenant_id="foton",
                    customer_id=customer_id,
                    link_type="tallanto_student_id",
                    link_value=f"family-student-{index}",
                    source_system="tallanto_snapshot",
                    source_ref=f"tallanto:{index}",
                    confidence=1.0,
                )
            )
            store.upsert_identity_link(
                IdentityLink(
                    tenant_id="foton",
                    customer_id=customer_id,
                    link_type="phone",
                    link_value="+79990000001",
                    source_system="tallanto_snapshot",
                    source_ref=f"tallanto:{index}",
                    confidence=1.0,
                )
            )
            store.upsert_event(
                TimelineEvent(
                    tenant_id="foton",
                    customer_id=customer_id,
                    event_type="tallanto_student_snapshot",
                    event_at=NOW,
                    source_system="tallanto_snapshot",
                    source_id=f"family-student-{index}",
                    direction="system",
                    match_status="strong_unique",
                    record={"payload": {"parent_fio": "Ирина Иванова"}},
                )
            )
    identity_dbs = tuple(
        _write_tallanto_identity_db(
            tmp_path / f"family-identity-{index}.sqlite",
            email="family-parent@example.com",
            tallanto_id=f"family-student-{index}",
            candidate_key=f"family-candidate-{index}",
        )
        for index in (1, 2)
    )
    sha = "e" * 64
    source_file, _ = _write_archive(
        tmp_path,
        sha=sha,
        email="family-parent@example.com",
        text="Здравствуйте, вопрос по занятиям детей.",
    )
    _seed_pending_event(db_path, tmp_path, sha=sha, source_file=source_file, subject="Фотон")
    build_family_graph(FamilyGraphConfig(timeline_db=db_path, allowed_root=tmp_path, apply=True))

    report = run_mail_link_enrich(
        MailLinkEnrichConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            out_dir=tmp_path / "out",
            tallanto_identity_dbs=identity_dbs,
            apply=True,
        )
    )

    assert report["counts"]["planned.family_strong"] == 1
    assert report["counts"]["reason.strong_tallanto_family_identity_link"] == 1
    assert report["historical_tallanto_identity"]["same_family_values"] == 1
    with sqlite3.connect(db_path) as con:
        event = json.loads(con.execute("SELECT record_json FROM timeline_events WHERE source_id=?", (sha,)).fetchone()[0])
    assert event["customer_id"] is None
    assert event["metadata"]["family_id"].startswith("family:")


def test_mail_link_enrich_ignores_historical_id_from_non_tallanto_source(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        store.upsert_customer(
            CustomerIdentity(
                tenant_id="foton",
                customer_id="customer:mail-derived",
                identity_status=IdentityStatus.STRONG,
            )
        )
        store.upsert_identity_link(
            IdentityLink(
                tenant_id="foton",
                customer_id="customer:mail-derived",
                link_type="tallanto_student_id",
                link_value="not-current-tallanto",
                source_system="mail_archive_stage2",
                source_ref="mail:derived",
                confidence=1.0,
            )
        )
    identity_db = _write_tallanto_identity_db(
        tmp_path / "identity.sqlite",
        email="historical-only@example.com",
        tallanto_id="not-current-tallanto",
        candidate_key="mail-derived-candidate",
    )
    sha = "f" * 64
    source_file, _ = _write_archive(
        tmp_path,
        sha=sha,
        email="historical-only@example.com",
        text="Здравствуйте, вопрос по обучению.",
    )
    _seed_pending_event(db_path, tmp_path, sha=sha, source_file=source_file, subject="Фотон")

    report = run_mail_link_enrich(
        MailLinkEnrichConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            out_dir=tmp_path / "out",
            tallanto_identity_dbs=(identity_db,),
            apply=True,
        )
    )

    assert report["historical_tallanto_identity"]["usable_email_values"] == 0
    assert report["counts"].get("planned.strong", 0) == 0
    with sqlite3.connect(db_path) as con:
        event = json.loads(con.execute("SELECT record_json FROM timeline_events WHERE source_id=?", (sha,)).fetchone()[0])
    assert event["customer_id"] is None


def test_mail_link_enrich_rechecks_old_strong_link_without_enrich_metadata(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    _seed_customer_with_links(db_path, tmp_path, email_source_system="tallanto_snapshot")
    sha = "9" * 64
    source_file, _ = _write_archive(
        tmp_path,
        sha=sha,
        email="parent@example.com",
        text="Здравствуйте, интересует обучение.",
    )
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        store.upsert_customer(
            CustomerIdentity(
                tenant_id="foton",
                customer_id="customer:old-wrong-link",
                identity_status=IdentityStatus.STRONG,
            )
        )
        store.upsert_event(
            TimelineEvent(
                tenant_id="foton",
                customer_id="customer:old-wrong-link",
                event_type="email_message",
                event_at=NOW,
                source_system=A2V3_MAIL_SOURCE_SYSTEM,
                source_id=sha,
                source_ref=f"mail_stage2:test:{sha[:16]}",
                direction=TimelineDirection.INBOUND,
                match_status="strong_unique",
                subject="Фотон",
                text_preview="Входящее письмо.",
                summary="Родитель спрашивает про обучение.",
                record={
                    "payload": {
                        "source_file": str(source_file),
                        "full_clean_text": "Родитель спрашивает про обучение.",
                        "brand": "foton",
                    }
                },
                metadata={},
                created_at=NOW,
            ),
            actor="test",
        )

    pending_only = run_mail_link_enrich(
        MailLinkEnrichConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            out_dir=tmp_path / "pending_only",
            reconsider_pending=True,
        )
    )
    assert pending_only["target_events"] == 0

    report = run_mail_link_enrich(
        MailLinkEnrichConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            out_dir=tmp_path / "out",
            revalidate_existing_strong=True,
            apply=True,
        )
    )

    assert report["counts"]["reason.existing_customer_identity_conflict"] == 1
    with sqlite3.connect(db_path) as con:
        row = con.execute(
            "SELECT customer_id, match_status FROM timeline_events WHERE source_id=?",
            (sha,),
        ).fetchone()
    assert row == (None, "ambiguous")


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
            revalidate_existing_strong=True,
        )
    )

    assert second["counts"]["planned.weak_email"] == 1
    assert second["counts"]["reason.email_ambiguous_identity_link"] == 1
    assert second["apply"]["counts"]["revoked_chunks"] == 1
    assert second["apply"]["counts"].get("not_revalidated_events", 0) == 0
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


def test_mail_link_enrich_does_not_lower_existing_strong_when_archive_is_missing(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    _seed_customer_with_links(db_path, tmp_path)
    sha = "2" * 64
    source_file, archive_db = _write_archive(
        tmp_path,
        sha=sha,
        email="unlinked@example.com",
        text="Здравствуйте.\n\nС уважением,\n+7 916 123-45-67",
    )
    _seed_pending_event(db_path, tmp_path, sha=sha, source_file=source_file, subject="Фотон")
    first = run_mail_link_enrich(
        MailLinkEnrichConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            out_dir=tmp_path / "first",
            fallback_archive_dbs=(archive_db,),
            apply=True,
        )
    )
    assert first["counts"]["planned.strong"] == 1
    archive_db.unlink()

    second = run_mail_link_enrich(
        MailLinkEnrichConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            out_dir=tmp_path / "second",
            revalidate_existing_strong=True,
            apply=True,
        )
    )

    assert second["counts"]["planned.not_revalidated_archive_missing"] == 1
    assert second["apply"]["counts"]["not_revalidated_events"] == 1
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as con:
        assert con.execute(
            "SELECT customer_id, match_status FROM timeline_events WHERE source_id=?", (sha,)
        ).fetchone() == ("customer:phone", "strong_unique")


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


def test_mail_link_enrich_accepts_exact_tallanto_email_for_partial_family(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    _seed_customer_with_links(db_path, tmp_path, email_source_system="tallanto_snapshot")
    with sqlite3.connect(db_path) as con:
        con.execute("UPDATE customer_identities SET identity_status='partial' WHERE customer_id='customer:email'")
    sha = "6" * 64
    source_file, _ = _write_archive(
        tmp_path,
        sha=sha,
        email="parent@example.com",
        text="Здравствуйте, интересует обучение.",
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

    assert report["counts"]["planned.strong"] == 1
    assert report["counts"]["reason.strong_external_email_identity_link"] == 1
    assert not (tmp_path / "out" / "mail_link_enrich_decisions.jsonl").exists()
    assert (tmp_path / "out").stat().st_mode & 0o777 == 0o700


def test_mail_link_enrich_uses_exact_tallanto_email_when_same_phone_customer_is_partial(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    _seed_customer_with_links(db_path, tmp_path, email_source_system="tallanto_snapshot")
    with sqlite3.connect(db_path) as con:
        con.execute(
            "UPDATE identity_links SET customer_id='customer:phone' "
            "WHERE customer_id='customer:email' AND link_type='email'"
        )
        con.execute("UPDATE customer_identities SET identity_status='partial' WHERE customer_id='customer:phone'")
    sha = "7" * 64
    source_file, _ = _write_archive(
        tmp_path,
        sha=sha,
        email="parent@example.com",
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

    assert report["counts"]["planned.strong"] == 1
    assert report["counts"]["reason.shared_phone_strong_email_intersection"] == 1


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
