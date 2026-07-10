from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from mango_mvp.customer_timeline import BotContextChunk, CustomerIdentity, CustomerTimelineSQLiteStore, TimelineEvent
from mango_mvp.customer_timeline.stage4b_bot_opening import (
    STAGE4B_OPENING_POLICY_VERSION,
    Stage4BBotOpeningConfig,
    run_stage4b_bot_opening,
)


NOW = datetime(2026, 7, 3, 12, 0, tzinfo=timezone.utc)


def test_stage4b_opens_only_linked_non_empty_mail_chunks_and_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        customer = CustomerIdentity(
            tenant_id="foton",
            identity_status="strong",
            customer_id="customer:known",
            primary_email="client@example.com",
            created_at=NOW,
            updated_at=NOW,
        )
        store.upsert_customer(customer)
        open_event = _mail_event(customer, "open", "Клиент спрашивает расписание и стоимость Фотона.")
        empty_event = _mail_event(customer, "empty", "Пустое письмо.")
        hidden_event = _mail_event(customer, "hidden", "Старое письмо.")
        store.upsert_event(open_event)
        store.upsert_event(empty_event)
        store.upsert_event(hidden_event)
        store.upsert_bot_context_chunk(_mail_chunk(open_event, text="Фотон. Расписание: суббота 12.15-14.15, цена 59 000 руб."))
        store.upsert_bot_context_chunk(_mail_chunk(empty_event, text="Временно непустой текст."))
        store.upsert_bot_context_chunk(_mail_chunk(hidden_event, text="Этот чанк будет superseded."))
        store._con.executescript(  # noqa: SLF001 - fixture creates the A2 facts side table slice.
            """
            CREATE TABLE a2v3_mail_event_facts (
              message_sha256 TEXT PRIMARY KEY,
              event_id TEXT NOT NULL,
              client_safe INTEGER NOT NULL,
              client_safe_reason TEXT NOT NULL DEFAULT 'no_sensitive_signals',
              sensitivity_tags_json TEXT NOT NULL DEFAULT '[]',
              bot_visible INTEGER NOT NULL DEFAULT 0
            );
            """
        )
        store._con.execute(  # noqa: SLF001
            """
            INSERT INTO a2v3_mail_event_facts(
              message_sha256, event_id, client_safe, client_safe_reason, sensitivity_tags_json, bot_visible
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("open", open_event.event_id, 1, "no_sensitive_signals", "[]", 1),
        )
        store._con.execute(  # noqa: SLF001 - test fixture creates historical empty text.
            "UPDATE bot_context_chunks SET record_json = json_set(record_json, '$.text', '') WHERE event_id = ?",
            (empty_event.event_id,),
        )
        store._con.execute(  # noqa: SLF001 - test fixture creates historical superseded chunk.
            "UPDATE bot_context_chunks SET superseded_by = ? WHERE event_id = ?",
            (open_event.event_id, hidden_event.event_id),
        )
        store._con.commit()  # noqa: SLF001

    config = Stage4BBotOpeningConfig(
        timeline_db_path=db_path,
        allowed_root=tmp_path,
        out_dir=tmp_path / "out",
        apply=True,
        allow_test_paths=True,
    )
    first = run_stage4b_bot_opening(config)
    second = run_stage4b_bot_opening(config)

    assert first["plan"]["candidate_chunks"] == 1
    assert first["apply"]["chunks_updated"] == 1
    assert first["after"]["mail_stage2_chunks_bot_visible"] == 1
    assert first["final_checks"]["candidate_review_violations_after"] == 0
    assert second["plan"]["already_open"] == 1
    assert second["apply"]["chunks_updated"] == 0

    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        rows = {
            row["event_id"]: row
            for row in con.execute(
                "SELECT event_id, allowed_for_bot, requires_manager_review, superseded_by, record_json FROM bot_context_chunks"
            )
        }
    opened = rows[open_event.event_id]
    payload = json.loads(opened["record_json"])
    assert opened["allowed_for_bot"] == 1
    assert opened["requires_manager_review"] == 0
    assert payload["metadata"]["memory_status"] == "usable_memory"
    assert payload["metadata"]["client_safe"] is False
    assert payload["metadata"]["bot_memory_allowed"] is True
    assert payload["metadata"]["bot_memory_policy_version"] == STAGE4B_OPENING_POLICY_VERSION
    assert "foton" in payload["metadata"]["sensitivity_tags"]
    assert "money" in payload["metadata"]["sensitivity_tags"]
    assert "schedule" in payload["metadata"]["sensitivity_tags"]
    assert rows[empty_event.event_id]["allowed_for_bot"] == 0
    assert rows[empty_event.event_id]["requires_manager_review"] == 1
    assert rows[hidden_event.event_id]["allowed_for_bot"] == 0
    assert rows[hidden_event.event_id]["requires_manager_review"] == 1


def test_stage4b_opens_only_strong_linked_channel_chunks_without_open_conflict(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        clean_customer = CustomerIdentity(
            tenant_id="foton",
            identity_status="strong",
            customer_id="customer:channel-clean",
            created_at=NOW,
            updated_at=NOW,
        )
        conflict_customer = CustomerIdentity(
            tenant_id="foton",
            identity_status="strong",
            customer_id="customer:channel-conflict",
            created_at=NOW,
            updated_at=NOW,
        )
        wappi_customer = CustomerIdentity(
            tenant_id="foton",
            identity_status="strong",
            customer_id="customer:wappi-clean",
            created_at=NOW,
            updated_at=NOW,
        )
        for customer in (clean_customer, conflict_customer, wappi_customer):
            store.upsert_customer(customer)

        telegram_event = _channel_event(
            clean_customer,
            "telegram_history",
            "telegram-open",
            match_status="strong_unique",
            text="Фотон. Клиент спрашивает, когда следующее занятие.",
        )
        conflict_event = _channel_event(
            conflict_customer,
            "telegram_history",
            "telegram-conflict",
            match_status="strong_unique",
            text="Фотон. Этот клиент с открытым конфликтом.",
        )
        wappi_event = _channel_event(
            wappi_customer,
            "wappi_max",
            "wappi-open",
            match_status="manual",
            text="Фотон. Клиент уточнил адрес занятия.",
        )
        pending_event = _channel_event(
            clean_customer,
            "telegram_history",
            "telegram-pending",
            match_status="ambiguous",
            text="Фотон. Неоднозначный telegram не должен открыться.",
        )
        for event in (telegram_event, conflict_event, wappi_event, pending_event):
            store.upsert_event(event)
            store.upsert_bot_context_chunk(_channel_chunk(event, text=event.summary or ""))
        store.record_conflict(
            "foton",
            conflict_type="tallanto_identity_ambiguous",
            entity_refs=(conflict_customer.customer_id,),
            status="open",
        )

    config = Stage4BBotOpeningConfig(
        timeline_db_path=db_path,
        allowed_root=tmp_path,
        out_dir=tmp_path / "out",
        apply=True,
        allow_test_paths=True,
    )
    report = run_stage4b_bot_opening(config)

    assert report["plan"]["source_system_counts"] == {"telegram_history": 1, "wappi_max": 1}
    assert report["apply"]["chunks_updated"] == 2
    assert report["plan"]["skipped"]["channel_events_not_openable_identity"] == 2
    assert report["final_checks"]["opened_disallowed_identity_after"] == 0

    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        rows = {
            row["event_id"]: row
            for row in con.execute(
                "SELECT event_id, allowed_for_bot, requires_manager_review, record_json FROM bot_context_chunks"
            )
        }
    assert rows[telegram_event.event_id]["allowed_for_bot"] == 1
    assert rows[telegram_event.event_id]["requires_manager_review"] == 0
    telegram_payload = json.loads(rows[telegram_event.event_id]["record_json"])
    assert {"channel", "telegram_history", "bot_visible", "foton"}.issubset(set(telegram_payload["relevance_tags"]))
    assert rows[wappi_event.event_id]["allowed_for_bot"] == 1
    assert rows[wappi_event.event_id]["requires_manager_review"] == 0
    assert rows[conflict_event.event_id]["allowed_for_bot"] == 0
    assert rows[conflict_event.event_id]["requires_manager_review"] == 1
    assert rows[pending_event.event_id]["allowed_for_bot"] == 0
    assert rows[pending_event.event_id]["requires_manager_review"] == 1


def test_stage4b_opens_only_strong_unique_mango_processed_summary_chunks(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        customer = CustomerIdentity(
            tenant_id="foton",
            identity_status="strong",
            customer_id="customer:mango-strong",
            created_at=NOW,
            updated_at=NOW,
        )
        partial_customer = CustomerIdentity(
            tenant_id="foton",
            identity_status="partial",
            customer_id="customer:mango-partial",
            created_at=NOW,
            updated_at=NOW,
        )
        mismatch_customer = CustomerIdentity(
            tenant_id="foton",
            identity_status="strong",
            customer_id="customer:mango-mismatch",
            created_at=NOW,
            updated_at=NOW,
        )
        ambiguous_identity_customer = CustomerIdentity(
            tenant_id="foton",
            identity_status="ambiguous",
            customer_id="customer:mango-ambiguous-identity",
            created_at=NOW,
            updated_at=NOW,
        )
        store.upsert_customer(customer)
        store.upsert_customer(partial_customer)
        store.upsert_customer(mismatch_customer)
        store.upsert_customer(ambiguous_identity_customer)
        strong_event = _mango_call_event(
            customer,
            "mango-strong",
            match_status="strong_unique",
            text="Звонок: клиент обсуждал курс по физике для 8 класса.",
            brand="foton",
        )
        ambiguous_event = _mango_call_event(
            customer,
            "mango-ambiguous",
            match_status="ambiguous",
            text="Фотон. Неоднозначный звонок не должен открыться.",
            brand="foton",
        )
        unmatched_event = _mango_call_event(
            customer,
            "mango-unmatched",
            match_status="unmatched",
            text="Фотон. Непривязанный звонок не должен открыться.",
            brand="foton",
        )
        partial_event = _mango_call_event(
            partial_customer,
            "mango-partial",
            match_status="strong_unique",
            text="Фотон. Partial identity у клиента не закрывает strong_unique звонок.",
            brand="foton",
        )
        unknown_brand_event = _mango_call_event(
            customer,
            "mango-unknown-brand",
            match_status="strong_unique",
            text="Звонок без бренда должен открыться как общий телефонный контекст.",
            brand="unknown",
        )
        wrong_chunk_type_event = _mango_call_event(
            customer,
            "mango-wrong-chunk-type",
            match_status="strong_unique",
            text="Mango source с неправильным chunk_type не должен открыться.",
            brand="unknown",
        )
        mismatch_event = _mango_call_event(
            customer,
            "mango-customer-mismatch",
            match_status="strong_unique",
            text="Mango chunk с другим customer_id не должен открыться.",
            brand="unknown",
        )
        ambiguous_identity_event = _mango_call_event(
            ambiguous_identity_customer,
            "mango-ambiguous-identity",
            match_status="strong_unique",
            text="Mango strong_unique не должен открыться при ambiguous customer identity.",
            brand="unknown",
        )
        for event in (
            strong_event,
            ambiguous_event,
            unmatched_event,
            partial_event,
            unknown_brand_event,
            wrong_chunk_type_event,
            mismatch_event,
            ambiguous_identity_event,
        ):
            store.upsert_event(event)
            if event is wrong_chunk_type_event:
                store.upsert_bot_context_chunk(_mango_call_chunk(event, text=event.summary or "", chunk_type="wrong_call_summary"))
            elif event is mismatch_event:
                store.upsert_bot_context_chunk(
                    _mango_call_chunk(event, text=event.summary or "", customer_id=mismatch_customer.customer_id)
                )
            else:
                store.upsert_bot_context_chunk(_mango_call_chunk(event, text=event.summary or ""))

    report = run_stage4b_bot_opening(
        Stage4BBotOpeningConfig(
            timeline_db_path=db_path,
            allowed_root=tmp_path,
            out_dir=tmp_path / "out",
            apply=True,
            allow_test_paths=True,
        )
    )

    assert report["plan"]["source_system_counts"] == {"mango_processed_summary": 3}
    assert report["apply"]["chunks_updated"] == 3
    assert report["after"]["mango_processed_summary_chunks_bot_visible"] == 3
    assert report["final_checks"]["opened_mango_processed_non_strong_after"] == 0
    assert report["final_checks"]["opened_disallowed_identity_after"] == 0
    assert report["final_checks"]["opened_unknown_brand_non_call_after"] == 0
    assert report["final_checks"]["opened_mango_processed_unknown_brand_after"] == 1
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        rows = {
            row["event_id"]: row
            for row in con.execute(
                "SELECT event_id, allowed_for_bot, requires_manager_review, record_json FROM bot_context_chunks"
            )
        }
    assert rows[strong_event.event_id]["allowed_for_bot"] == 1
    assert rows[strong_event.event_id]["requires_manager_review"] == 0
    payload = json.loads(rows[strong_event.event_id]["record_json"])
    assert {"call", "mango_processed_summary", "bot_visible", "foton"}.issubset(set(payload["relevance_tags"]))
    assert "brand_unknown" not in set(payload["relevance_tags"])
    assert rows[ambiguous_event.event_id]["allowed_for_bot"] == 0
    assert rows[unmatched_event.event_id]["allowed_for_bot"] == 0
    assert rows[partial_event.event_id]["allowed_for_bot"] == 1
    assert rows[unknown_brand_event.event_id]["allowed_for_bot"] == 1
    unknown_payload = json.loads(rows[unknown_brand_event.event_id]["record_json"])
    assert {"call", "mango_processed_summary", "bot_visible", "brand_unknown"}.issubset(
        set(unknown_payload["relevance_tags"])
    )
    assert rows[wrong_chunk_type_event.event_id]["allowed_for_bot"] == 0
    assert rows[mismatch_event.event_id]["allowed_for_bot"] == 0
    assert rows[ambiguous_identity_event.event_id]["allowed_for_bot"] == 0


def test_stage4b_retracts_previously_opened_non_strong_or_conflicted_chunks(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        partial_customer = CustomerIdentity(
            tenant_id="foton",
            identity_status="partial",
            customer_id="customer:partial-rich",
            created_at=NOW,
            updated_at=NOW,
        )
        conflicted_customer = CustomerIdentity(
            tenant_id="foton",
            identity_status="strong",
            customer_id="customer:conflicted-rich",
            created_at=NOW,
            updated_at=NOW,
        )
        other_tenant_customer = CustomerIdentity(
            tenant_id="unpk",
            identity_status="partial",
            customer_id="customer:other-tenant-rich",
            created_at=NOW,
            updated_at=NOW,
        )
        store.upsert_customer(partial_customer)
        store.upsert_customer(conflicted_customer)
        store.upsert_customer(other_tenant_customer)
        partial_event = _mail_event(partial_customer, "partial-rich", "Фотон. Partial identity уже ошибочно открыт.")
        conflict_event = _mail_event(conflicted_customer, "conflicted-rich", "Фотон. Conflict ref уже ошибочно открыт.")
        other_tenant_event = _mail_event(other_tenant_customer, "other-tenant-rich", "УНПК. Соседний tenant не трогать.")
        store.upsert_event(partial_event)
        store.upsert_event(conflict_event)
        store.upsert_event(other_tenant_event)
        store.upsert_bot_context_chunk(_mail_chunk(partial_event, text=partial_event.summary or ""))
        store.upsert_bot_context_chunk(_mail_chunk(conflict_event, text=conflict_event.summary or ""))
        store.upsert_bot_context_chunk(_mail_chunk(other_tenant_event, text=other_tenant_event.summary or ""))
        store.record_conflict(
            "foton",
            conflict_type="shared_family_phone",
            entity_refs=("phone_hash:test", "customer:customer:conflicted-rich"),
            status="open",
        )
        store._con.execute(  # noqa: SLF001 - simulate a previous bad opening that must be retracted.
            """
            UPDATE bot_context_chunks
            SET allowed_for_bot = 1,
                requires_manager_review = 0,
                record_json = json_set(record_json, '$.metadata.e4b_bot_opening.opened', true)
            WHERE event_id IN (?, ?, ?)
            """,
            (partial_event.event_id, conflict_event.event_id, other_tenant_event.event_id),
        )
        store._con.commit()  # noqa: SLF001

    config = Stage4BBotOpeningConfig(
        timeline_db_path=db_path,
        allowed_root=tmp_path,
        out_dir=tmp_path / "out",
        apply=True,
        allow_test_paths=True,
    )
    report = run_stage4b_bot_opening(config)

    assert report["plan"]["candidate_chunks"] == 0
    assert report["apply"]["chunks_retracted_not_openable"] == 2
    assert report["final_checks"]["opened_disallowed_identity_after"] == 0
    with sqlite3.connect(db_path) as con:
        rows = {
            row[0]: row
            for row in con.execute(
                "SELECT event_id, tenant_id, allowed_for_bot, requires_manager_review FROM bot_context_chunks"
            )
        }
        assert rows[partial_event.event_id][2:] == (0, 1)
        assert rows[conflict_event.event_id][2:] == (0, 1)
        assert rows[other_tenant_event.event_id][1:] == ("unpk", 1, 0)


def test_stage4b_keeps_a2_client_unsafe_mail_manager_only(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        customer = CustomerIdentity(
            tenant_id="foton",
            identity_status="strong",
            customer_id="customer:unsafe-mail",
            created_at=NOW,
            updated_at=NOW,
        )
        store.upsert_customer(customer)
        unsafe_event = _mail_event(customer, "unsafe", "Фотон. Клиент прислал пароль от кабинета.")
        safe_event = _mail_event(customer, "safe", "Фотон. Клиент спрашивает расписание.")
        money_event = _mail_event(customer, "money", "Фотон. Клиент уточнил оплату 59 000 руб.")
        medical_event = _mail_event(customer, "medical", "Фотон. Оплата есть, но в письме есть медицинские детали.")
        store.upsert_event(unsafe_event)
        store.upsert_event(safe_event)
        store.upsert_event(money_event)
        store.upsert_event(medical_event)
        store.upsert_bot_context_chunk(_mail_chunk(unsafe_event, text=unsafe_event.summary or ""))
        store.upsert_bot_context_chunk(_mail_chunk(safe_event, text=safe_event.summary or ""))
        store.upsert_bot_context_chunk(_mail_chunk(money_event, text=money_event.summary or ""))
        store.upsert_bot_context_chunk(_mail_chunk(medical_event, text=medical_event.summary or ""))
        store._con.executescript(  # noqa: SLF001 - fixture creates the A2 facts side table slice.
            """
            CREATE TABLE a2v3_mail_event_facts (
              message_sha256 TEXT PRIMARY KEY,
              event_id TEXT NOT NULL,
              client_safe INTEGER NOT NULL,
              client_safe_reason TEXT NOT NULL DEFAULT 'no_sensitive_signals',
              sensitivity_tags_json TEXT NOT NULL DEFAULT '[]',
              bot_visible INTEGER NOT NULL DEFAULT 0
            );
            """
        )
        store._con.executemany(  # noqa: SLF001
            """
            INSERT INTO a2v3_mail_event_facts(
              message_sha256, event_id, client_safe, client_safe_reason, sensitivity_tags_json, bot_visible
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                ("unsafe", unsafe_event.event_id, 0, "sensitive_credentials", '["sensitive_credentials"]', 1),
                ("safe", safe_event.event_id, 1, "no_sensitive_signals", "[]", 1),
                ("money", money_event.event_id, 0, "sensitive_money", '["sensitive_money"]', 1),
                ("medical", medical_event.event_id, 0, "sensitive_money", '["sensitive_money", "sensitive_medical"]', 1),
            ],
        )
        store._con.commit()  # noqa: SLF001

    report = run_stage4b_bot_opening(
        Stage4BBotOpeningConfig(
            timeline_db_path=db_path,
            allowed_root=tmp_path,
            out_dir=tmp_path / "out",
            apply=True,
            allow_test_paths=True,
        )
    )

    assert report["client_unsafe_mail_chunks_indexed"] == 2
    assert report["client_safe_mail_chunks_indexed"] == 3
    assert report["plan"]["skipped"]["client_unsafe_mail_chunks"] == 2
    assert report["plan"]["skipped"]["mail_chunks_not_allowed_by_output_gate"] == 1
    assert report["apply"]["chunks_updated"] == 2
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        rows = {
            row["event_id"]: row
            for row in con.execute(
                "SELECT event_id, allowed_for_bot, requires_manager_review FROM bot_context_chunks"
            )
        }
    assert rows[unsafe_event.event_id]["allowed_for_bot"] == 0
    assert rows[unsafe_event.event_id]["requires_manager_review"] == 1
    assert rows[safe_event.event_id]["allowed_for_bot"] == 1
    assert rows[safe_event.event_id]["requires_manager_review"] == 0
    assert rows[money_event.event_id]["allowed_for_bot"] == 1
    assert rows[money_event.event_id]["requires_manager_review"] == 0
    assert rows[medical_event.event_id]["allowed_for_bot"] == 0
    assert rows[medical_event.event_id]["requires_manager_review"] == 1


def test_stage4b_does_not_open_mail_without_a2_bot_visible_flag(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        customer = CustomerIdentity(
            tenant_id="foton",
            identity_status="strong",
            customer_id="customer:bot-hidden-mail",
            primary_email="client@example.com",
            created_at=NOW,
            updated_at=NOW,
        )
        store.upsert_customer(customer)
        hidden_event = _mail_event(customer, "hiddenbot", "Фотон. Письмо безопасное, но A2 bot_visible=0.")
        store.upsert_event(hidden_event)
        store.upsert_bot_context_chunk(_mail_chunk(hidden_event, text=hidden_event.summary or ""))
        store._con.executescript(  # noqa: SLF001 - fixture creates the A2 facts side table slice.
            """
            CREATE TABLE a2v3_mail_event_facts (
              message_sha256 TEXT PRIMARY KEY,
              event_id TEXT NOT NULL,
              client_safe INTEGER NOT NULL,
              client_safe_reason TEXT NOT NULL DEFAULT 'no_sensitive_signals',
              sensitivity_tags_json TEXT NOT NULL DEFAULT '[]',
              bot_visible INTEGER NOT NULL DEFAULT 0
            );
            """
        )
        store._con.execute(  # noqa: SLF001
            """
            INSERT INTO a2v3_mail_event_facts(
              message_sha256, event_id, client_safe, client_safe_reason, sensitivity_tags_json, bot_visible
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("hiddenbot", hidden_event.event_id, 1, "no_sensitive_signals", "[]", 0),
        )
        store._con.execute(  # noqa: SLF001 - simulate a previous bad opening that must be retracted.
            """
            UPDATE bot_context_chunks
            SET allowed_for_bot = 1,
                requires_manager_review = 0,
                record_json = json_set(record_json, '$.metadata.e4b_bot_opening.opened', true)
            WHERE event_id = ?
            """,
            (hidden_event.event_id,),
        )
        store._con.commit()  # noqa: SLF001

    report = run_stage4b_bot_opening(
        Stage4BBotOpeningConfig(
            timeline_db_path=db_path,
            allowed_root=tmp_path,
            out_dir=tmp_path / "out",
            apply=True,
            allow_test_paths=True,
        )
    )

    assert report["client_safe_mail_chunks_indexed"] == 0
    assert report["plan"]["candidate_chunks"] == 0
    assert report["plan"]["skipped"]["mail_chunks_not_allowed_by_output_gate"] == 1
    assert report["apply"]["chunks_retracted_not_openable"] == 1
    with sqlite3.connect(db_path) as con:
        row = con.execute(
            "SELECT allowed_for_bot, requires_manager_review FROM bot_context_chunks WHERE event_id = ?",
            (hidden_event.event_id,),
        ).fetchone()
    assert row == (0, 1)


def test_stage4b_refuses_non_staging_path_without_test_override(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path).close()

    config = Stage4BBotOpeningConfig(
        timeline_db_path=db_path,
        allowed_root=tmp_path,
        out_dir=tmp_path / "out",
        apply=True,
    )

    try:
        run_stage4b_bot_opening(config)
    except ValueError as exc:
        assert ".codex_local/staging" in str(exc)
    else:  # pragma: no cover - defensive assertion.
        raise AssertionError("stage4b opening accepted a non-staging path")


def test_stage4b_refuses_nested_fake_staging_path(tmp_path: Path) -> None:
    db_path = tmp_path / ".codex_local" / "foo" / "staging" / "customer_timeline.sqlite"
    CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path).close()

    config = Stage4BBotOpeningConfig(
        timeline_db_path=db_path,
        allowed_root=tmp_path,
        out_dir=tmp_path / "out",
        apply=True,
    )

    try:
        run_stage4b_bot_opening(config)
    except ValueError as exc:
        assert ".codex_local/staging" in str(exc)
    else:  # pragma: no cover - defensive assertion.
        raise AssertionError("stage4b opening accepted a nested fake staging path")


def _mail_event(customer: CustomerIdentity, suffix: str, summary: str) -> TimelineEvent:
    return TimelineEvent(
        tenant_id=customer.tenant_id,
        customer_id=customer.customer_id,
        event_type="email_message",
        event_at=NOW,
        source_system="mail_archive_stage2",
        source_id=f"{suffix:0<64}"[:64],
        direction="inbound",
        summary=summary,
        text_preview=summary,
        match_status="strong_unique",
        created_at=NOW,
        record={"message_sha256": f"{suffix:0<64}"[:64]},
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
        allowed_for_bot=False,
        requires_manager_review=True,
        metadata={"sensitivity_tags": ["brand_unknown", "manager_review"]},
        created_at=event.created_at,
    )


def _channel_event(
    customer: CustomerIdentity,
    source_system: str,
    suffix: str,
    *,
    match_status: str,
    text: str,
) -> TimelineEvent:
    event_type = "max_message" if source_system == "wappi_max" else "telegram_message"
    return TimelineEvent(
        tenant_id=customer.tenant_id,
        customer_id=customer.customer_id,
        event_type=event_type,
        event_at=NOW,
        source_system=source_system,
        source_id=f"{suffix:0<64}"[:64],
        direction="inbound",
        summary=text,
        text_preview=text,
        match_status=match_status,
        created_at=NOW,
        record={"message_id": suffix},
    )


def _channel_chunk(event: TimelineEvent, *, text: str) -> BotContextChunk:
    return BotContextChunk(
        tenant_id=event.tenant_id,
        customer_id=event.customer_id or "",
        event_id=event.event_id,
        source_ref=event.source_ref,
        source_system=event.source_system,
        chunk_type="channel_message",
        text=text,
        summary=event.summary or "",
        event_at=event.event_at,
        allowed_for_bot=False,
        requires_manager_review=True,
        created_at=event.created_at,
    )


def _mango_call_event(
    customer: CustomerIdentity,
    suffix: str,
    *,
    match_status: str,
    text: str,
    brand: str = "unknown",
) -> TimelineEvent:
    return TimelineEvent(
        tenant_id=customer.tenant_id,
        customer_id=customer.customer_id,
        event_type="mango_call",
        event_at=NOW,
        source_system="mango_processed_summary",
        source_id=f"{suffix:0<64}"[:64],
        direction="inbound",
        summary=text,
        text_preview=text,
        match_status=match_status,
        created_at=NOW,
        metadata={"brand": brand},
        record={"call_id": suffix, "brand": brand},
    )


def _mango_call_chunk(
    event: TimelineEvent,
    *,
    text: str,
    chunk_type: str = "mango_call_summary",
    customer_id: str | None = None,
) -> BotContextChunk:
    return BotContextChunk(
        tenant_id=event.tenant_id,
        customer_id=customer_id or event.customer_id or "",
        event_id=event.event_id,
        source_ref=event.source_ref,
        source_system=event.source_system,
        chunk_type=chunk_type,
        text=text,
        summary=event.summary or "",
        event_at=event.event_at,
        allowed_for_bot=False,
        requires_manager_review=True,
        created_at=event.created_at,
    )
