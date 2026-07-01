from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from mango_mvp.customer_timeline.a2_mail_ingest import (
    A2V3MailIngestConfig,
    apply_a2v3_mail_ingest,
    build_local_client_review,
    create_test_db_backup,
    plan_a2v3_mail_ingest,
    validate_a2v3_mail_ingest,
)
from mango_mvp.customer_timeline.contracts import CustomerIdentity, IdentityLink, IdentityStatus, TimelineDirection, TimelineEvent
from mango_mvp.customer_timeline.store import CustomerTimelineSQLiteStore


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


def _row(
    sha: str,
    *,
    email: str = "parent@example.com",
    status: str = "usable_memory",
    event_type: str = "application",
    brand: str = "foton",
    amount_kind: str = "",
    amount_rub: object = None,
    thread_id: str = "thread-a",
) -> dict[str, object]:
    return {
        "message_sha256": sha,
        "date_iso": "2026-07-01T10:00:00+00:00",
        "direction": "inbound",
        "brand": brand,
        "brand_source": "test",
        "contact_email": email,
        "contact_phone": None,
        "contact_name": "Parent",
        "contact_ambiguous": False,
        "contact_missing": False,
        "subject_full": "Запись на курс",
        "full_clean_text": "Письмо о записи на курс.",
        "summary_payload": {
            "summary": "Родитель спрашивает о записи на курс.",
            "topic": "Запись",
            "event_type": event_type,
            "money_direction": "none",
            "amount_kind": amount_kind,
            "amount_rub": amount_rub,
            "next_step": "Ответить по условиям записи.",
        },
        "quality": {
            "memory_status": status,
            "quality_flags": [],
            "requires_human_confirmation": False,
            "safe_next_step_note": "Ответить по условиям записи.",
            "thread_id": thread_id,
            "thread_basis": "message_id",
            "money_amounts_rub": [],
            "amount_uncertain": False,
        },
    }


def _make_prod_db(tmp_path: Path) -> Path:
    db = tmp_path / "prod.sqlite"
    store = CustomerTimelineSQLiteStore(db, allowed_root=tmp_path)
    customer = CustomerIdentity(
        tenant_id="foton",
        customer_id="customer:known",
        identity_status=IdentityStatus.STRONG,
        display_name="Known",
        primary_email="known@example.com",
    )
    ambiguous = CustomerIdentity(
        tenant_id="foton",
        customer_id="customer:ambiguous",
        identity_status=IdentityStatus.AMBIGUOUS,
        display_name="Ambiguous",
        primary_email="amb@example.com",
    )
    store.upsert_customer(customer)
    store.upsert_customer(ambiguous)
    store.upsert_identity_link(
        IdentityLink(
            tenant_id="foton",
            customer_id="customer:known",
            link_type="email",
            link_value="known@example.com",
            source_system="canonical",
            source_ref="test",
            match_class="strong_unique",
            confidence=0.95,
        )
    )
    store.upsert_identity_link(
        IdentityLink(
            tenant_id="foton",
            customer_id="customer:known",
            link_type="tallanto_student_id",
            link_value="T-1",
            source_system="canonical",
            source_ref="test",
            match_class="strong_unique",
            confidence=0.95,
        )
    )
    store.upsert_identity_link(
        IdentityLink(
            tenant_id="foton",
            customer_id="customer:ambiguous",
            link_type="email",
            link_value="amb@example.com",
            source_system="canonical",
            source_ref="test",
            match_class="strong_unique",
            confidence=0.95,
        )
    )
    inbound = TimelineEvent(
        tenant_id="foton",
        customer_id="customer:known",
        event_type="email_message",
        event_at=customer.created_at,
        source_system="mail_archive",
        source_id="history-in",
        direction=TimelineDirection.INBOUND,
        match_status="strong_unique",
    )
    outbound = TimelineEvent(
        tenant_id="foton",
        customer_id="customer:known",
        event_type="email_message",
        event_at=customer.created_at,
        source_system="mail_archive",
        source_id="history-out",
        direction=TimelineDirection.OUTBOUND,
        match_status="strong_unique",
    )
    store.upsert_event(inbound)
    store.upsert_event(outbound)
    store.close()
    return db


def _make_identity_db(tmp_path: Path) -> Path:
    db = tmp_path / "identity.sqlite"
    with sqlite3.connect(db) as con:
        con.executescript(
            """
            CREATE TABLE identity_values (
              kind TEXT NOT NULL,
              value TEXT NOT NULL,
              match_class TEXT NOT NULL,
              candidate_count INTEGER NOT NULL,
              PRIMARY KEY (kind, value)
            );
            CREATE TABLE identity_candidates (
              candidate_key TEXT PRIMARY KEY,
              tallanto_id TEXT
            );
            CREATE TABLE identity_links (
              kind TEXT NOT NULL,
              value TEXT NOT NULL,
              candidate_key TEXT NOT NULL,
              source_columns_json TEXT NOT NULL,
              PRIMARY KEY (kind, value, candidate_key)
            );
            """
        )
        con.execute("INSERT INTO identity_values VALUES ('email', 'parent@example.com', 'strong_unique', 1)")
        con.execute("INSERT INTO identity_candidates VALUES ('candidate-1', 'T-1')")
        con.execute("INSERT INTO identity_links VALUES ('email', 'parent@example.com', 'candidate-1', '[]')")
        con.execute("INSERT INTO identity_values VALUES ('email', 'shared@example.com', 'ambiguous', 2)")
        con.commit()
    return db


def _config(tmp_path: Path, rows: list[dict[str, object]]) -> A2V3MailIngestConfig:
    return A2V3MailIngestConfig(
        input_jsonl=_write_jsonl(tmp_path / "a2v3.jsonl", rows),
        prod_timeline_db=_make_prod_db(tmp_path),
        timeline_db_path=tmp_path / "test_timeline.sqlite",
        allowed_root=tmp_path,
        out_dir=tmp_path / "out",
        tallanto_identity_db=_make_identity_db(tmp_path),
        tenant_id="foton",
        source_ref="test-a2v3",
    )


def test_a2v3_mail_ingest_uses_tallanto_email_and_adds_email_link(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        [_row("a" * 64, email="parent@example.com", event_type="payment", amount_kind="actual", amount_rub=50000)],
    )
    CustomerTimelineSQLiteStore(config.timeline_db_path, allowed_root=tmp_path).close()

    plans, report = plan_a2v3_mail_ingest(config)

    assert report["counts"]["linked"] == 1
    assert plans[0].resolution.method == "tallanto_email"
    assert plans[0].identity_links[0].link_type.value == "email"
    assert plans[0].bot_visible is True

    backup = create_test_db_backup(config)
    first = apply_a2v3_mail_ingest(config, backup_manifest_path=Path(str(backup["manifest_path"])))
    second = apply_a2v3_mail_ingest(config, backup_manifest_path=Path(str(backup["manifest_path"])))

    assert first["counts"]["created_events"] == 1
    assert first["counts"]["created_bot_visible_chunks"] == 1
    assert second["selected_events"] == 0
    with sqlite3.connect(config.timeline_db_path) as con:
        assert con.execute("SELECT allowed_for_bot, requires_manager_review FROM bot_context_chunks").fetchall() == [(1, 0)]
        assert con.execute("SELECT link_type, link_value FROM identity_links WHERE link_type='email'").fetchall() == [
            ("email", "parent@example.com")
        ]
        fact = con.execute(
            """
            SELECT event_type_detail, amount_kind, amount_rub, email_brand, customer_brand,
                   customer_brand_source, bot_visible
            FROM a2v3_mail_event_facts
            """
        ).fetchone()
        assert fact == ("payment", "actual", 50000.0, "foton", "foton", "a2v3_email_content", 1)


def test_a2v3_mail_ingest_blocks_ambiguous_identity_and_non_usable_memory(tmp_path: Path) -> None:
    rows = [
        _row("b" * 64, email="amb@example.com"),
        _row("c" * 64, email="known@example.com", status="needs_thread_context"),
    ]
    config = _config(tmp_path, rows)
    CustomerTimelineSQLiteStore(config.timeline_db_path, allowed_root=tmp_path).close()

    plans, report = plan_a2v3_mail_ingest(config)

    by_sha = {plan.event.source_id: plan for plan in plans}
    assert by_sha["b" * 64].resolution.outcome == "blocked"
    assert by_sha["b" * 64].chunk is None
    assert by_sha["c" * 64].resolution.outcome == "linked"
    assert by_sha["c" * 64].bot_visible is False
    assert by_sha["c" * 64].chunk is not None
    assert by_sha["c" * 64].chunk.allowed_for_bot is False
    assert by_sha["c" * 64].chunk.requires_manager_review is True
    assert report["counts"]["blocked"] == 1
    assert report["counts"]["linked"] == 1


def test_a2v3_mail_ingest_blocks_bot_visibility_when_customer_brand_unknown(tmp_path: Path) -> None:
    config = _config(tmp_path, [_row("g" * 64, email="known@example.com", brand="none")])
    CustomerTimelineSQLiteStore(config.timeline_db_path, allowed_root=tmp_path).close()

    plans, report = plan_a2v3_mail_ingest(config)

    assert report["counts"]["linked"] == 1
    assert report["counts"]["bot_gate.customer_brand_unknown"] == 1
    assert plans[0].customer_brand == "unknown"
    assert plans[0].bot_visible is False


def test_a2v3_mail_ingest_checks_prod_dedupe_across_both_mail_namespaces(tmp_path: Path) -> None:
    sha = "d" * 64
    config = _config(tmp_path, [_row(sha, email="known@example.com")])
    store = CustomerTimelineSQLiteStore(config.prod_timeline_db, allowed_root=tmp_path)
    try:
        event = TimelineEvent(
            tenant_id="foton",
            customer_id="customer:known",
            event_type="email_message",
            event_at=CustomerIdentity(tenant_id="foton", identity_status="strong", customer_id="tmp").created_at,
            source_system="mail_archive",
            source_id=sha,
            direction="inbound",
            match_status="strong_unique",
        )
        store.upsert_event(event)
    finally:
        store.close()
    CustomerTimelineSQLiteStore(config.timeline_db_path, allowed_root=tmp_path).close()

    validation = validate_a2v3_mail_ingest(config)
    backup = create_test_db_backup(config)
    applied = apply_a2v3_mail_ingest(config, backup_manifest_path=Path(str(backup["manifest_path"])))

    assert validation["would_skip_prod_duplicate_events"] == 1
    assert applied["selected_events"] == 1
    assert applied["counts"]["prod_duplicate_events_observed"] == 1
    assert applied["counts"]["created_events"] == 1


def test_a2v3_local_client_review_includes_linked_and_unresolved_rows(tmp_path: Path) -> None:
    rows = [
        _row("e" * 64, email="parent@example.com"),
        _row("f" * 64, email="missing@example.com"),
    ]
    config = _config(tmp_path, rows)
    CustomerTimelineSQLiteStore(config.timeline_db_path, allowed_root=tmp_path).close()

    plans, _ = plan_a2v3_mail_ingest(config)
    review = build_local_client_review(config, plans)

    assert review["rows"] == 2
    assert review["new_email_events"] == 2
    assert review["linked_groups"] == 1
    assert review["unresolved_groups"] == 1
    review_rows = [
        json.loads(line)
        for line in Path(str(review["jsonl"])).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    linked = next(row for row in review_rows if row["customer_id"] == "customer:known")
    unresolved = next(row for row in review_rows if row["customer_id"] is None)

    assert linked["existing_before"]["event_count"] == 2
    assert linked["new_email_events"][0]["resolution"] == "linked"
    assert linked["combined_timeline_count"] == 3
    assert linked["combined_timeline"][0]["source"] == "new_a2v3_email"
    assert linked["current_next_step"]["schema_version"] == "customer_timeline_next_step_resolution_v1"

    assert unresolved["resolution"] == {"unmatched": 1}
    assert unresolved["combined_timeline_count"] == 1
    assert unresolved["combined_timeline"][0]["source"] == "new_a2v3_email"
    assert unresolved["current_next_step"]["reason_code"] == "unresolved_identity"
