from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from mango_mvp.customer_timeline.a2_mail_ingest import (
    A2V3MailIngestConfig,
    apply_a2v3_mail_ingest,
    build_local_client_review,
    create_test_db_backup,
    ensure_not_prod_apply_path,
    current_contact_brand,
    plan_a2v3_mail_ingest,
    reconcile_a2v3_event_facts,
    validate_a2v3_mail_ingest,
)
from mango_mvp.customer_timeline.contracts import (
    BotContextChunk,
    CustomerIdentity,
    CustomerOpportunity,
    IdentityLink,
    IdentityStatus,
    OpportunityType,
    TimelineDirection,
    TimelineEvent,
)
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
    assert plans[0].bot_eligible_candidate is True

    backup = create_test_db_backup(config)
    first = apply_a2v3_mail_ingest(config, backup_manifest_path=Path(str(backup["manifest_path"])))
    second = apply_a2v3_mail_ingest(config, backup_manifest_path=Path(str(backup["manifest_path"])))

    assert first["counts"]["created_events"] == 1
    assert first["counts"]["created_bot_eligible_candidate_chunks"] == 1
    assert second["selected_events"] == 0
    with sqlite3.connect(config.timeline_db_path) as con:
        assert con.execute("SELECT allowed_for_bot, requires_manager_review FROM bot_context_chunks").fetchall() == [(0, 1)]
        event_record = json.loads(con.execute("SELECT record_json FROM timeline_events").fetchone()[0])
        assert "bot_visibility" not in event_record["record"]
        assert event_record["record"]["bot_eligible_candidate"] == {
            "eligible": True,
            "qualified": True,
            "reason": "usable_linked_qualified",
        }
        assert con.execute("SELECT link_type, link_value FROM identity_links WHERE link_type='email'").fetchall() == [
            ("email", "parent@example.com")
        ]
        fact = con.execute(
            """
            SELECT event_type_detail, amount_kind, amount_rub, email_brand, customer_brand,
                   customer_brand_source, bot_visible, client_safe, client_safe_reason,
                   sensitivity_tags_json
            FROM a2v3_mail_event_facts
            """
        ).fetchone()
        assert fact[:7] == ("payment", "actual", 50000.0, "foton", "foton", "a2v3_email_content", 1)
        assert fact[7:9] == (0, "sensitive_money")
        assert "sensitive_money" in json.loads(fact[9])
        chunk_record = json.loads(con.execute("SELECT record_json FROM bot_context_chunks").fetchone()[0])
        assert "Письмо о записи на курс." in chunk_record["text"]
        assert "Ответить по условиям записи." not in chunk_record["text"]
        assert chunk_record["metadata"]["client_safe"] is False
        assert chunk_record["metadata"]["client_safe_reason"] == "sensitive_money"
        profile = con.execute(
            """
            SELECT brand, source, reason
            FROM a2v3_customer_brand_profiles
            WHERE customer_id = 'customer:known'
            """
        ).fetchone()
        assert profile == ("foton", "a2v3_email_content", "history_unknown_email_content_known")


def test_a2v3_mail_ingest_content_duplicate_skips_chunk_and_facts(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        [
            _row("h" * 64, email="parent@example.com"),
            _row("i" * 64, email="parent@example.com"),
        ],
    )
    CustomerTimelineSQLiteStore(config.timeline_db_path, allowed_root=tmp_path).close()

    validation = validate_a2v3_mail_ingest(config)
    backup = create_test_db_backup(config)
    applied = apply_a2v3_mail_ingest(config, backup_manifest_path=Path(str(backup["manifest_path"])))

    assert validation["would_create_events"] == 1
    assert validation["would_skip_content_events_in_input"] == 1
    assert applied["selected_events"] == 1
    assert applied["counts"]["created_events"] == 1
    assert applied["counts"]["created_chunks"] == 1
    assert applied["counts"]["upserted_a2v3_event_facts"] == 1
    with sqlite3.connect(config.timeline_db_path) as con:
        assert con.execute("SELECT count(*) FROM timeline_events").fetchone()[0] == 1
        assert con.execute("SELECT count(*) FROM bot_context_chunks").fetchone()[0] == 1
        assert con.execute("SELECT count(*) FROM a2v3_mail_event_facts").fetchone()[0] == 1


def test_a2v3_mail_ingest_reconciles_facts_for_existing_event_after_partial_crash(tmp_path: Path) -> None:
    config = _config(tmp_path, [_row("j" * 64, email="parent@example.com")])
    CustomerTimelineSQLiteStore(config.timeline_db_path, allowed_root=tmp_path).close()
    plans, _ = plan_a2v3_mail_ingest(config)
    plan = plans[0]
    store = CustomerTimelineSQLiteStore(config.timeline_db_path, allowed_root=tmp_path)
    try:
        assert plan.customer is not None
        assert plan.opportunity is not None
        store.upsert_customer(plan.customer)
        store.upsert_opportunity(plan.opportunity)
        store.upsert_event(plan.event)
    finally:
        store.close()

    with sqlite3.connect(config.timeline_db_path) as con:
        assert con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='a2v3_mail_event_facts'"
        ).fetchone() is None

    backup = create_test_db_backup(config)
    applied = apply_a2v3_mail_ingest(config, backup_manifest_path=Path(str(backup["manifest_path"])))

    assert applied["selected_events"] == 0
    assert applied["counts"]["upserted_a2v3_event_facts"] == 1
    assert applied["counts"]["reconciled_a2v3_event_facts"] == 1
    with sqlite3.connect(config.timeline_db_path) as con:
        assert con.execute("SELECT count(*) FROM timeline_events").fetchone()[0] == 1
        assert con.execute("SELECT count(*) FROM a2v3_mail_event_facts").fetchone()[0] == 1


def test_reconcile_a2v3_event_facts_is_idempotent(tmp_path: Path) -> None:
    config = _config(tmp_path, [_row("k" * 64, email="parent@example.com")])
    CustomerTimelineSQLiteStore(config.timeline_db_path, allowed_root=tmp_path).close()
    plans, _ = plan_a2v3_mail_ingest(config)
    backup = create_test_db_backup(config)
    apply_a2v3_mail_ingest(config, backup_manifest_path=Path(str(backup["manifest_path"])))

    first = reconcile_a2v3_event_facts(config.timeline_db_path, plans, allowed_root=config.allowed_root)
    second = reconcile_a2v3_event_facts(config.timeline_db_path, plans, allowed_root=config.allowed_root)

    assert first == {"plans": 1, "events_found": 1, "facts_upserted": 1}
    assert second == {"plans": 1, "events_found": 1, "facts_upserted": 1}
    with sqlite3.connect(config.timeline_db_path) as con:
        assert con.execute("SELECT count(*) FROM a2v3_mail_event_facts").fetchone()[0] == 1


def test_a2v3_mail_ingest_rich_chunk_keeps_thread_context_without_manager_note(tmp_path: Path) -> None:
    row = _row("n" * 64, email="parent@example.com")
    row["full_clean_text"] = "Актуальный вопрос про расписание и стоимость."
    row["thread_context"] = "Предыдущее письмо: обсуждали 8 класс и воскресенье."
    row["thread_context_source"] = "raw_body_split_thread_context"
    config = _config(tmp_path, [row])
    CustomerTimelineSQLiteStore(config.timeline_db_path, allowed_root=tmp_path).close()

    plan = plan_a2v3_mail_ingest(config)[0][0]

    assert plan.chunk is not None
    assert "Актуальный вопрос про расписание" in plan.chunk.text
    assert "Контекст переписки" in plan.chunk.text
    assert "Предыдущее письмо" in plan.chunk.text
    assert "Ответить по условиям записи." not in plan.chunk.text
    assert plan.chunk.metadata["chunk_rich_text"] is True
    assert plan.chunk.allowed_for_bot is False
    assert plan.chunk.requires_manager_review is True


def test_a2v3_mail_ingest_ignores_unproven_thread_context(tmp_path: Path) -> None:
    row = _row("s" * 64, email="parent@example.com")
    row["full_clean_text"] = "Актуальный вопрос."
    row["thread_context"] = "Ручной хвост без raw-body provenance."
    config = _config(tmp_path, [row])
    CustomerTimelineSQLiteStore(config.timeline_db_path, allowed_root=tmp_path).close()

    plan = plan_a2v3_mail_ingest(config)[0][0]

    assert plan.chunk is not None
    assert "Актуальный вопрос." in plan.chunk.text
    assert "Ручной хвост" not in plan.chunk.text
    assert "Контекст переписки" not in plan.chunk.text


def test_a2v3_mail_ingest_legacy_summary_chunk_still_omits_manager_note(tmp_path: Path) -> None:
    config = replace(_config(tmp_path, [_row("o" * 64, email="parent@example.com")]), chunk_rich_text=False)
    CustomerTimelineSQLiteStore(config.timeline_db_path, allowed_root=tmp_path).close()

    plan = plan_a2v3_mail_ingest(config)[0][0]

    assert plan.chunk is not None
    assert "Родитель спрашивает о записи" in plan.chunk.text
    assert "Ответить по условиям записи." not in plan.chunk.text


def test_a2v3_customer_brand_profile_uses_dominant_history_not_email_domain(tmp_path: Path) -> None:
    config = _config(tmp_path, [_row("p" * 64, email="known@example.com", brand="none")])
    store = CustomerTimelineSQLiteStore(config.prod_timeline_db, allowed_root=tmp_path)
    try:
        for index in range(4):
            store.upsert_opportunity(
                CustomerOpportunity(
                    tenant_id="foton",
                    customer_id="customer:known",
                    opportunity_type=OpportunityType.AMO_DEAL,
                    source_system="amo",
                    source_id=f"foton-{index}",
                    status="observed",
                    product_context={"brand": "foton"},
                )
            )
        store.upsert_opportunity(
            CustomerOpportunity(
                tenant_id="foton",
                customer_id="customer:known",
                opportunity_type=OpportunityType.AMO_DEAL,
                source_system="amo",
                source_id="unpk-1",
                status="observed",
                product_context={"brand": "unpk"},
            )
        )
    finally:
        store.close()
    CustomerTimelineSQLiteStore(config.timeline_db_path, allowed_root=tmp_path).close()

    plans, report = plan_a2v3_mail_ingest(config)

    assert plans[0].customer_brand == "foton"
    assert plans[0].customer_brand_reason == "dominant_brand_history"
    assert report["counts"]["bot_gate.usable_linked_qualified"] == 1
    with sqlite3.connect(config.prod_timeline_db) as con:
        con.row_factory = sqlite3.Row
        brand, reason = current_contact_brand(
            con,
            tenant_id="foton",
            customer_id="customer:known",
            as_of=plans[0].event.event_at,
        )
    assert (brand, reason) == ("foton", "current_opportunity_brand")


def test_current_contact_brand_does_not_infer_from_email_domain(tmp_path: Path) -> None:
    db = tmp_path / "timeline.sqlite"
    store = CustomerTimelineSQLiteStore(db, allowed_root=tmp_path)
    try:
        store.upsert_customer(
            CustomerIdentity(tenant_id="foton", customer_id="customer:domain", identity_status=IdentityStatus.STRONG)
        )
        store.upsert_event(
            TimelineEvent(
                tenant_id="foton",
                customer_id="customer:domain",
                event_type="email_message",
                event_at=CustomerIdentity(tenant_id="foton", identity_status="strong", customer_id="tmp").created_at,
                source_system="mail_archive",
                source_id="domain-only",
                direction=TimelineDirection.INBOUND,
                subject="edu@kmipt.ru",
                summary="В письме есть только служебный домен.",
                match_status="strong_unique",
            )
        )
    finally:
        store.close()
    with sqlite3.connect(db) as con:
        con.row_factory = sqlite3.Row
        assert current_contact_brand(con, tenant_id="foton", customer_id="customer:domain") == (
            "unknown",
            "stale_brand_signal",
        )


def test_a2v3_customer_brand_history_ignores_email_domains(tmp_path: Path) -> None:
    config = _config(tmp_path, [_row("r" * 64, email="known@example.com", brand="none")])
    store = CustomerTimelineSQLiteStore(config.prod_timeline_db, allowed_root=tmp_path)
    try:
        store.upsert_event(
            TimelineEvent(
                tenant_id="foton",
                customer_id="customer:known",
                event_type="email_message",
                event_at=CustomerIdentity(tenant_id="foton", identity_status="strong", customer_id="tmp").created_at,
                source_system="mail_archive",
                source_id="domain-history",
                direction=TimelineDirection.INBOUND,
                subject="noreply@cdpofoton.ru",
                summary="Служебный адрес cdpofoton.ru без смыслового бренда.",
                match_status="strong_unique",
            )
        )
    finally:
        store.close()
    CustomerTimelineSQLiteStore(config.timeline_db_path, allowed_root=tmp_path).close()

    plans, report = plan_a2v3_mail_ingest(config)

    assert plans[0].customer_brand == "unknown"
    assert plans[0].customer_brand_reason == "no_known_brand_in_history"
    assert report["counts"]["bot_gate.customer_brand_unknown"] == 1


def test_customer_purchases_v1_scaffold_does_not_use_email_amounts(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        [_row("q" * 64, email="parent@example.com", event_type="payment", amount_kind="actual", amount_rub=77777)],
    )
    store = CustomerTimelineSQLiteStore(config.timeline_db_path, allowed_root=tmp_path)
    try:
        store.upsert_customer(
            CustomerIdentity(tenant_id="foton", customer_id="customer:known", identity_status=IdentityStatus.STRONG)
        )
        store.upsert_opportunity(
            CustomerOpportunity(
                tenant_id="foton",
                customer_id="customer:known",
                opportunity_type=OpportunityType.AMO_DEAL,
                source_system="amo",
                source_id="paid-deal",
                status="Оплата получена",
                product_context={"brand": "foton"},
            )
        )
    finally:
        store.close()

    backup = create_test_db_backup(config)
    apply_a2v3_mail_ingest(config, backup_manifest_path=Path(str(backup["manifest_path"])))

    with sqlite3.connect(config.timeline_db_path) as con:
        row = con.execute(
            """
            SELECT total_in, total_out, deals_cnt, computability, sources_json
            FROM customer_purchases_v1
            WHERE customer_id = 'customer:known'
            """
        ).fetchone()
        assert row[:4] == (None, None, 1, "not_computable_missing_primary_amounts")
        assert json.loads(row[4])["email_amounts_used"] is False


def test_a2v3_apply_path_guard_requires_allowed_root_and_rejects_prod_or_symlink(tmp_path: Path) -> None:
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    ensure_not_prod_apply_path(allowed_root / "timeline.sqlite", allowed_root=allowed_root)

    with pytest.raises(ValueError, match="outside allowed_root"):
        ensure_not_prod_apply_path(tmp_path / "outside.sqlite", allowed_root=allowed_root)

    with pytest.raises(ValueError, match="prod timeline"):
        ensure_not_prod_apply_path(
            allowed_root / "customer_timeline_prod_20260621" / "customer_timeline.sqlite",
            allowed_root=allowed_root,
        )

    outside = tmp_path / "outside_real.sqlite"
    outside.write_text("", encoding="utf-8")
    symlink = allowed_root / "linked.sqlite"
    symlink.symlink_to(outside)
    with pytest.raises(ValueError, match="outside allowed_root"):
        ensure_not_prod_apply_path(symlink, allowed_root=allowed_root)


def test_a2v3_mail_ingest_requires_explicit_enrich_for_hash_mismatch(tmp_path: Path) -> None:
    config = _config(tmp_path, [_row("l" * 64, email="parent@example.com")])
    CustomerTimelineSQLiteStore(config.timeline_db_path, allowed_root=tmp_path).close()
    plans, _ = plan_a2v3_mail_ingest(config)
    plan = plans[0]
    old_event = replace(plan.event, summary="Старая сжатая версия письма.")
    store = CustomerTimelineSQLiteStore(config.timeline_db_path, allowed_root=tmp_path)
    try:
        assert plan.customer is not None
        assert plan.opportunity is not None
        store.upsert_customer(plan.customer)
        store.upsert_opportunity(plan.opportunity)
        store.upsert_event(old_event)
    finally:
        store.close()

    backup = create_test_db_backup(config)
    with pytest.raises(ValueError, match="enrich_existing required"):
        apply_a2v3_mail_ingest(config, backup_manifest_path=Path(str(backup["manifest_path"])))

    assert (config.out_dir / "enrich_existing_required.json").exists()
    with sqlite3.connect(config.timeline_db_path) as con:
        assert con.execute("SELECT summary FROM timeline_events").fetchone()[0] == "Старая сжатая версия письма."
        assert con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='a2v3_mail_event_facts'"
        ).fetchone() is None


def test_a2v3_mail_ingest_enrich_existing_preserves_metadata_and_updates_summary(tmp_path: Path) -> None:
    base_config = _config(tmp_path, [_row("m" * 64, email="parent@example.com")])
    config = replace(base_config, enrich_existing=True)
    CustomerTimelineSQLiteStore(config.timeline_db_path, allowed_root=tmp_path).close()
    plans, _ = plan_a2v3_mail_ingest(config)
    plan = plans[0]
    old_event = replace(
        plan.event,
        summary="Старая сжатая версия письма.",
        metadata={"fresh_relink": True, "existing_only": "keep"},
        record={"legacy_field": "keep", "bot_visibility": {"allowed_for_bot": True}},
    )
    store = CustomerTimelineSQLiteStore(config.timeline_db_path, allowed_root=tmp_path)
    try:
        assert plan.customer is not None
        assert plan.opportunity is not None
        store.upsert_customer(plan.customer)
        store.upsert_opportunity(plan.opportunity)
        store.upsert_event(old_event)
    finally:
        store.close()

    backup = create_test_db_backup(config)
    applied = apply_a2v3_mail_ingest(config, backup_manifest_path=Path(str(backup["manifest_path"])))

    assert applied["counts"]["enrich_existing_events"] == 1
    assert applied["counts"].get("created_events", 0) == 0
    assert (config.out_dir / "enrich_existing_diff.json").exists()
    with sqlite3.connect(config.timeline_db_path) as con:
        row = con.execute("SELECT summary, record_json FROM timeline_events").fetchone()
        payload = json.loads(row[1])
        assert row[0] == "Родитель спрашивает о записи на курс."
        assert payload["metadata"]["fresh_relink"] is True
        assert payload["metadata"]["existing_only"] == "keep"
        assert payload["record"]["legacy_field"] == "keep"
        assert "bot_visibility" not in payload["record"]
        assert payload["record"]["bot_eligible_candidate"]["eligible"] is True
        assert con.execute("SELECT count(*) FROM a2v3_mail_event_facts").fetchone()[0] == 1


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
    assert by_sha["c" * 64].bot_eligible_candidate is False
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
    assert plans[0].bot_eligible_candidate is False


def test_store_rejects_mail_archive_stage2_bot_visible_chunk(tmp_path: Path) -> None:
    store = CustomerTimelineSQLiteStore(tmp_path / "timeline.sqlite", allowed_root=tmp_path)
    try:
        customer = CustomerIdentity(tenant_id="foton", customer_id="customer:known", identity_status=IdentityStatus.STRONG)
        store.upsert_customer(customer)
        with pytest.raises(ValueError, match="mail_archive_stage2 bot context chunks"):
            store.upsert_bot_context_chunk(
                BotContextChunk(
                    tenant_id="foton",
                    customer_id="customer:known",
                    chunk_type="email_message",
                    text="Почтовый чанк не должен открываться боту на Э1.",
                    source_system="mail_archive_stage2",
                    source_ref="test",
                    allowed_for_bot=True,
                    requires_manager_review=False,
                )
            )
    finally:
        store.close()


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
