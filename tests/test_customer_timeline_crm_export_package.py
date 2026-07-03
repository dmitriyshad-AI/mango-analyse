from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mango_mvp.customer_timeline.contracts import (
    CustomerIdentity,
    CustomerOpportunity,
    DerivedSignal,
    IdentityLink,
    OpportunityType,
    TimelineEvent,
)
from mango_mvp.customer_timeline.crm_export_package import (
    CrmExportPackageConfig,
    _inject_child_data_soft_warning,
    _row_blockers,
    build_crm_export_package,
)
from mango_mvp.customer_timeline.store import CustomerTimelineSQLiteStore
from mango_mvp.deal_aware.deal_text_builder import DEAL_AI_FIELDS


NOW = datetime(2026, 7, 2, 12, 0, tzinfo=timezone.utc)


def test_crm_export_package_builds_staging_only_deterministic_package(tmp_path: Path) -> None:
    db_path = _seed_db(tmp_path)
    out_dir = tmp_path / ".codex_local" / "staging" / "e5_crm_export"

    first = build_crm_export_package(
        CrmExportPackageConfig(
            timeline_db_path=db_path,
            allowed_root=tmp_path,
            out_dir=out_dir,
            pilot_size=1,
        )
    )
    repeat = build_crm_export_package(
        CrmExportPackageConfig(
            timeline_db_path=db_path,
            allowed_root=tmp_path,
            out_dir=tmp_path / ".codex_local" / "staging" / "e5_crm_export_repeat",
            pilot_size=1,
        )
    )

    assert first["candidate_rows"] == 1
    assert first["pilot_rows"] == 1
    assert first["ready_rows"] == 1
    assert first["safety"]["write_amo"] is False
    assert first["output_sha256"] == repeat["output_sha256"]
    csv_text = (out_dir / "pilot_20_crm_card_candidates.csv").read_text(encoding="utf-8-sig")
    preview = (out_dir / "pilot_20_preview.md").read_text(encoding="utf-8")
    row = json.loads((out_dir / "pilot_20_crm_card_candidates.jsonl").read_text(encoding="utf-8").splitlines()[0])

    assert "Покупки и оплаты" in row["contact_payload"]["Авто история общения"]
    assert "оплачено факт 12 000" in row["contact_payload"]["Авто история общения"]
    assert "план сделки 50 000" in row["contact_payload"]["Авто история общения"]
    assert "Возражения и бюджет" in row["contact_payload"]["Авто история общения"]
    assert "Сигналы" in row["contact_payload"]["Авто история общения"]
    assert "Клиент активно отвечает" in row["contact_payload"]["Авто история общения"]
    assert "hot_streak:" not in row["contact_payload"]["Авто история общения"]
    assert "Письмо: отправлено расписание" in row["contact_payload"]["Авто история общения"]
    assert "all_candidates_jsonl" in first["output_sha256"]
    assert (out_dir / "all_candidates_crm_card_candidates.jsonl").exists()
    assert row["CRM writeback policy"] == "live_update_ready"
    assert isinstance(row["active_signals_count"], int)
    for field in DEAL_AI_FIELDS:
        assert field in row
        assert field in csv_text
    assert row["AI-сводка по сделке"]
    assert row["AI-история по сделке"]
    assert row["AI-рекомендованный следующий шаг"]
    assert row["AI-дата обновления сделки"]
    assert "crm_card_contact_payload_json" in csv_text
    assert "AMO write=0" in preview


def test_crm_export_package_adds_interests_and_pains_from_client_call_text(tmp_path: Path) -> None:
    db_path = _seed_db(tmp_path)
    canonical_db = tmp_path / ".codex_local" / "staging" / "canonical_calls.sqlite"
    with sqlite3.connect(canonical_db) as con:
        con.execute("CREATE TABLE canonical_calls (canonical_call_id TEXT PRIMARY KEY, transcript_client TEXT)")
        con.execute(
            "INSERT INTO canonical_calls VALUES (?, ?)",
            (
                "call-1",
                "Хотим интенсив по математике. Сложно по времени, переживаем, что не успеваем к экзамену.",
            ),
        )
    out_dir = tmp_path / ".codex_local" / "staging" / "e5_crm_export"

    build_crm_export_package(
        CrmExportPackageConfig(
            timeline_db_path=db_path,
            allowed_root=tmp_path,
            out_dir=out_dir,
            pilot_size=1,
            canonical_calls_db_path=canonical_db,
        )
    )
    row = json.loads((out_dir / "pilot_20_crm_card_candidates.jsonl").read_text(encoding="utf-8").splitlines()[0])
    history = row["contact_payload"]["Авто история общения"]

    assert "Интересы:" in history
    assert "Интерес из звонка: Хотим интенсив по математике" in history
    assert "Боли:" in history
    assert "Боль из звонка: Сложно по времени" in history
    assert row["Интересы"]
    assert row["Боли"]
    assert "mango_call:" not in history


def test_crm_export_package_canonical_calls_path_is_fail_soft(tmp_path: Path) -> None:
    db_path = _seed_db(tmp_path)
    out_dir = tmp_path / ".codex_local" / "staging" / "e5_crm_export"

    manifest = build_crm_export_package(
        CrmExportPackageConfig(
            timeline_db_path=db_path,
            allowed_root=tmp_path,
            out_dir=out_dir,
            pilot_size=1,
            canonical_calls_db_path=tmp_path / ".codex_local" / "staging" / "missing_canonical.sqlite",
        )
    )

    assert manifest["candidate_rows"] == 1
    assert any(str(item).startswith("canonical_calls_db_missing:") for item in manifest["warnings"])


def test_crm_export_package_soft_warns_child_mentions_without_hard_family_block() -> None:
    projection = {
        "contact_card": {
            "fields": {
                "Последняя сводка": "Сводка:\nКлиент рассказал, что ребёнок хочет заниматься математикой.",
                "История общения": "Клиент ждёт расписание.",
            }
        },
        "deal_card": {"fields": {"Следующий шаг": "Позвонить клиенту."}},
    }

    _inject_child_data_soft_warning(projection, family_text="")
    deal_fields = projection["deal_card"]["fields"]

    assert "Семейные данные: есть упоминание вне проверенного блока" in deal_fields["Предупреждения"]
    blockers = _row_blockers(
        {
            "contact_card": {"ready_for_amo": True, "blockers": []},
            "deal_card": {"ready_for_amo": True, "blockers": []},
        },
        contact_payload=projection["contact_card"]["fields"],
        deal_payload=deal_fields,
    )
    assert "family_or_child_data_requires_review" not in blockers


def test_crm_export_package_refuses_non_staging_paths(tmp_path: Path) -> None:
    db_path = _seed_db(tmp_path)
    with pytest.raises(ValueError, match=".codex_local/staging"):
        build_crm_export_package(
            CrmExportPackageConfig(
                timeline_db_path=tmp_path / "customer_timeline.sqlite",
                allowed_root=tmp_path,
                out_dir=tmp_path / ".codex_local" / "staging" / "e5",
            )
        )
    with pytest.raises(ValueError, match=".codex_local/staging"):
        build_crm_export_package(
            CrmExportPackageConfig(
                timeline_db_path=db_path,
                allowed_root=tmp_path,
                out_dir=tmp_path / "e5",
            )
        )


def test_crm_export_package_blocks_raw_email_thread_payload(tmp_path: Path) -> None:
    db_path = _seed_db(tmp_path, raw_email_artifact=True)
    out_dir = tmp_path / ".codex_local" / "staging" / "e5_crm_export"

    manifest = build_crm_export_package(
        CrmExportPackageConfig(
            timeline_db_path=db_path,
            allowed_root=tmp_path,
            out_dir=out_dir,
            pilot_size=1,
        )
    )
    row = json.loads((out_dir / "pilot_20_crm_card_candidates.jsonl").read_text(encoding="utf-8").splitlines()[0])

    assert manifest["candidate_rows"] == 1
    assert manifest["ready_rows"] == 0
    assert row["Готово к записи в AMO"] == "нет"
    assert "crm_text_quality:raw_email_thread_artifact" in row["CRM writeback blockers"]
    assert (out_dir / "batch_ready_crm_card_candidates.jsonl").read_text(encoding="utf-8") == ""


def test_crm_export_package_blocks_weak_summary_and_manual_review_next_step() -> None:
    blockers = _row_blockers(
        {
            "contact_card": {"ready_for_amo": True, "blockers": []},
            "deal_card": {"ready_for_amo": True, "blockers": []},
        },
        contact_payload={
            "Последняя сводка": "Сводка:\nСтандартный",
            "История общения": "Сводка:\nСм. поле «Последняя сводка».",
        },
        deal_payload={
            "Следующий шаг": "Уточнить у менеджера: более позднее событие противоречит закрытию шага",
        },
    )

    assert "weak_or_empty_summary" in blockers
    assert "manual_review_next_step_not_live_ready" in blockers


def test_crm_export_package_blocks_raw_timeline_and_sensitive_payload() -> None:
    blockers = _row_blockers(
        {
            "contact_card": {"ready_for_amo": True, "blockers": []},
            "deal_card": {"ready_for_amo": True, "blockers": []},
        },
        contact_payload={
            "Последняя сводка": "Сводка:\nКлиент выбрал курс и попросил ссылку на оплату.",
            "История общения": "Клиент прислал паспортные данные для договора.",
        },
        deal_payload={
            "Следующий шаг": "Позвонить 10.07 и подтвердить способ оплаты.",
            "Tallanto": "Tallanto: out 2026-06-20",
        },
    )

    assert "raw_timeline_or_email_artifact" in blockers
    assert "sensitive_personal_data_requires_review" in blockers


def test_crm_export_package_blocks_masked_or_debug_placeholders() -> None:
    blockers = _row_blockers(
        {
            "contact_card": {"ready_for_amo": True, "blockers": []},
            "deal_card": {
                "ready_for_amo": True,
                "blockers": [],
                "fields": {"Возражения": "Клиент обсуждал скидку [сжато]"},
            },
        },
        contact_payload={
            "Последняя сводка": "Сводка:\nКлиент выбрал курс и ждёт ссылку.",
            "История общения": "Клиент попросил оформить запись.",
        },
        deal_payload={
            "Следующий шаг": "Позвонить клиенту, срок: <phone_masked>",
        },
    )

    assert "masked_or_debug_placeholder" in blockers

    non_mask_blockers = _row_blockers(
        {
            "contact_card": {"ready_for_amo": True, "blockers": []},
            "deal_card": {
                "ready_for_amo": True,
                "blockers": [],
                "fields": {"Возражения": "Клиент обсуждал скидку [сжато]"},
            },
        },
        contact_payload={
            "Последняя сводка": "Сводка:\nКлиент выбрал курс и ждёт ссылку.",
            "История общения": "Клиент попросил оформить запись.",
        },
        deal_payload={
            "Следующий шаг": "Позвонить клиенту и уточнить решение.",
        },
    )

    assert "masked_or_debug_placeholder" not in non_mask_blockers


def test_crm_export_package_blocks_foreign_brand_marker_for_active_brand() -> None:
    blockers = _row_blockers(
        {
            "contact_card": {
                "ready_for_amo": True,
                "blockers": [],
                "fields": {
                    "Последняя сводка": "Клиент обсуждал курсы УНПК.",
                    "История общения": "В истории есть исходящее письмо [Фотон] про оплату.",
                },
            },
            "deal_card": {
                "ready_for_amo": True,
                "blockers": [],
                "fields": {"Следующий шаг": "Связаться с клиентом по УНПК."},
            },
        },
        contact_payload={
            "Последняя сводка": "Сводка:\nКлиент обсуждал курсы УНПК.",
            "История общения": "В истории есть исходящее письмо [Фотон] про оплату.",
        },
        deal_payload={"Следующий шаг": "Связаться с клиентом по УНПК."},
        active_brand="unpk",
    )

    assert "foreign_brand_marker_in_payload:foton_inside_unpk_card" in blockers


def test_crm_export_package_allows_clean_family_block_in_live_ready_payload() -> None:
    family_text = "- Аня (класс: 8; предметы: математика)"
    blockers = _row_blockers(
        {
            "contact_card": {"ready_for_amo": True, "blockers": []},
            "deal_card": {"ready_for_amo": True, "blockers": []},
        },
        contact_payload={
            "Последняя сводка": "Сводка:\nКлиент обсуждает годовой курс.",
            "История общения": "Семья:\n" + family_text,
        },
        deal_payload={"Следующий шаг": "Позвонить взрослому клиенту."},
        family_text=family_text,
    )

    assert "family_or_child_data_requires_review" not in blockers


def test_crm_export_package_blocks_ambiguous_family_or_raw_child_mentions() -> None:
    clean_family_text = "- Аня (класс: 8; предметы: математика)"
    family_review_blockers = _row_blockers(
        {
            "contact_card": {"ready_for_amo": True, "blockers": []},
            "deal_card": {"ready_for_amo": True, "blockers": []},
        },
        contact_payload={
            "Последняя сводка": "Сводка:\nКлиент обсуждает годовой курс.",
            "История общения": "Семья:\n" + clean_family_text,
        },
        deal_payload={"Следующий шаг": "Позвонить взрослому клиенту."},
        family_text=clean_family_text,
        family_review_required=True,
    )
    raw_child_blockers = _row_blockers(
        {
            "contact_card": {"ready_for_amo": True, "blockers": []},
            "deal_card": {"ready_for_amo": True, "blockers": []},
        },
        contact_payload={
            "Последняя сводка": "Сводка:\nКлиент обсуждает годовой курс.",
            "История общения": "--- part ---\nВ письме клиент упомянул ребёнка и 8 класс, но в family graph этого нет.",
        },
        deal_payload={"Следующий шаг": "Позвонить взрослому клиенту."},
    )

    assert "family_or_child_data_requires_review" in family_review_blockers
    assert "family_or_child_data_requires_review" in raw_child_blockers


def test_crm_export_package_blocks_raw_email_separator_in_live_ready_payload() -> None:
    blockers = _row_blockers(
        {
            "contact_card": {"ready_for_amo": True, "blockers": []},
            "deal_card": {"ready_for_amo": True, "blockers": []},
        },
        contact_payload={
            "Последняя сводка": "Сводка:\nКлиент попросил варианты курса.",
            "История общения": "------------------------- Здравствуйте! Подскажите расписание.",
        },
        deal_payload={"Следующий шаг": "Отправить варианты курса."},
    )

    assert "raw_timeline_or_email_artifact" in blockers


def test_crm_export_package_blocks_closed_next_step_for_stalling_deal() -> None:
    blockers = _row_blockers(
        {
            "contact_card": {"ready_for_amo": True, "blockers": []},
            "deal_card": {"ready_for_amo": True, "blockers": []},
        },
        contact_payload={
            "Последняя сводка": "Сводка:\nСделка зависла, клиент пока не ответил.",
            "История общения": "Клиент получил предложение и не вернулся.",
        },
        deal_payload={"Следующий шаг": "Шаг закрыт: клиент оплатил."},
    )

    assert "closed_next_step_with_active_or_stalling_deal" in blockers


def test_crm_export_package_blocks_stale_next_step_date() -> None:
    blockers = _row_blockers(
        {
            "contact_card": {"ready_for_amo": True, "blockers": []},
            "deal_card": {"ready_for_amo": True, "blockers": []},
        },
        contact_payload={
            "Последняя сводка": "Сводка:\nКлиент ждёт повторного контакта.",
            "История общения": "Последний контакт был давно.",
        },
        deal_payload={"Следующий шаг": "Позвонить 21.10.2025 и уточнить решение."},
        as_of="2026-07-03T12:00:00+00:00",
    )

    assert "stale_next_step_date_requires_review" in blockers


def test_crm_export_package_adds_family_block_with_ambiguity_warning(tmp_path: Path) -> None:
    db_path = _seed_db(tmp_path)
    out_dir = tmp_path / ".codex_local" / "staging" / "e5_crm_export"
    with sqlite3.connect(db_path) as con:
        con.executescript(
            """
            CREATE TABLE family_links_v1 (
              tenant_id TEXT NOT NULL,
              family_id TEXT NOT NULL,
              customer_id TEXT NOT NULL,
              child_key TEXT NOT NULL,
              canonical_name TEXT NOT NULL,
              name_variants_json TEXT NOT NULL,
              grades_json TEXT NOT NULL,
              subjects_json TEXT NOT NULL,
              brand TEXT NOT NULL,
              status TEXT NOT NULL,
              confidence TEXT NOT NULL,
              reason TEXT NOT NULL,
              source_refs_json TEXT NOT NULL,
              evidence_count INTEGER NOT NULL,
              created_at TEXT NOT NULL,
              record_hash TEXT NOT NULL,
              record_json TEXT NOT NULL,
              PRIMARY KEY (tenant_id, customer_id, child_key)
            );
            """
        )
        con.execute(
            "INSERT INTO family_links_v1 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "foton",
                "family:1",
                "customer:e5",
                "child:1",
                "Аня",
                json.dumps(["Аня"], ensure_ascii=False),
                json.dumps(["8"], ensure_ascii=False),
                json.dumps(["математика"], ensure_ascii=False),
                "foton",
                "confident",
                "high",
                "single_child_family",
                "[]",
                2,
                NOW.isoformat(),
                "hash1",
                "{}",
            ),
        )
        con.execute(
            "INSERT INTO family_links_v1 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "foton",
                "family:1",
                "customer:e5",
                "child:2",
                "Миша",
                json.dumps(["Миша"], ensure_ascii=False),
                json.dumps([], ensure_ascii=False),
                json.dumps([], ensure_ascii=False),
                "unknown",
                "needs_review",
                "medium",
                "multiple_child_candidates",
                "[]",
                1,
                NOW.isoformat(),
                "hash2",
                "{}",
            ),
        )
        con.commit()

    build_crm_export_package(
        CrmExportPackageConfig(
            timeline_db_path=db_path,
            allowed_root=tmp_path,
            out_dir=out_dir,
            pilot_size=1,
        )
    )
    row = json.loads((out_dir / "pilot_20_crm_card_candidates.jsonl").read_text(encoding="utf-8").splitlines()[0])

    history = row["contact_payload"]["Авто история общения"]
    warnings = row["Предупреждения"]
    assert "Семья:" in history
    assert "Аня (класс: 8; предметы: математика)" in history
    assert "Миша — уточнить привязку" in history
    assert "Семья: есть неоднозначность" in warnings


def test_crm_export_package_filters_non_client_objections(tmp_path: Path) -> None:
    db_path = _seed_db(tmp_path)
    out_dir = tmp_path / ".codex_local" / "staging" / "e5_crm_export"
    with sqlite3.connect(db_path) as con:
        con.execute("UPDATE customer_objections_v1 SET speaker = 'manager', confidence = 'high'")
        con.execute(
            """
            CREATE TABLE customer_objection_extraction_runs_v1 (
              tenant_id TEXT NOT NULL,
              extractor_version TEXT NOT NULL,
              extracted_at TEXT NOT NULL,
              call_events_total INTEGER NOT NULL,
              call_events_matched INTEGER NOT NULL,
              call_events_with_client_transcript INTEGER NOT NULL,
              call_match_coverage REAL NOT NULL,
              crm_objections_enabled INTEGER NOT NULL,
              metrics_json TEXT NOT NULL,
              PRIMARY KEY (tenant_id, extractor_version)
            )
            """
        )
        con.execute(
            "INSERT INTO customer_objection_extraction_runs_v1 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("foton", "ob_v1", NOW.isoformat(), 10, 10, 10, 1.0, 1, "{}"),
        )
        con.commit()

    build_crm_export_package(
        CrmExportPackageConfig(
            timeline_db_path=db_path,
            allowed_root=tmp_path,
            out_dir=out_dir,
            pilot_size=1,
        )
    )
    row = json.loads((out_dir / "pilot_20_crm_card_candidates.jsonl").read_text(encoding="utf-8").splitlines()[0])

    assert "Возражения и бюджет" not in row["contact_payload"]["Авто история общения"]
    assert "дорого, просит скидку" not in row["contact_payload"]["Авто история общения"]


def test_crm_export_package_omits_objections_when_coverage_gate_failed(tmp_path: Path) -> None:
    db_path = _seed_db(tmp_path)
    out_dir = tmp_path / ".codex_local" / "staging" / "e5_crm_export"
    with sqlite3.connect(db_path) as con:
        con.execute(
            """
            CREATE TABLE customer_objection_extraction_runs_v1 (
              tenant_id TEXT NOT NULL,
              extractor_version TEXT NOT NULL,
              extracted_at TEXT NOT NULL,
              call_events_total INTEGER NOT NULL,
              call_events_matched INTEGER NOT NULL,
              call_events_with_client_transcript INTEGER NOT NULL,
              call_match_coverage REAL NOT NULL,
              crm_objections_enabled INTEGER NOT NULL,
              metrics_json TEXT NOT NULL,
              PRIMARY KEY (tenant_id, extractor_version)
            )
            """
        )
        con.execute(
            "INSERT INTO customer_objection_extraction_runs_v1 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("foton", "ob_v1", NOW.isoformat(), 10, 5, 5, 0.5, 0, "{}"),
        )
        con.commit()

    build_crm_export_package(
        CrmExportPackageConfig(
            timeline_db_path=db_path,
            allowed_root=tmp_path,
            out_dir=out_dir,
            pilot_size=1,
        )
    )
    row = json.loads((out_dir / "pilot_20_crm_card_candidates.jsonl").read_text(encoding="utf-8").splitlines()[0])

    assert "Возражения и бюджет" not in row["contact_payload"]["Авто история общения"]
    assert row["objections_count"] == "0"


def _seed_db(tmp_path: Path, *, raw_email_artifact: bool = False) -> Path:
    stage = tmp_path / ".codex_local" / "staging"
    stage.mkdir(parents=True)
    db_path = stage / "customer_timeline_staging.sqlite"
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        customer = CustomerIdentity(
            tenant_id="foton",
            customer_id="customer:e5",
            identity_status="strong",
            primary_phone="+79161234567",
            created_at=NOW,
            updated_at=NOW,
        )
        store.upsert_customer(customer)
        store.upsert_identity_link(
            IdentityLink(
                tenant_id="foton",
                customer_id=customer.customer_id,
                link_type="phone",
                link_value="+79161234567",
                source_system="fixture",
                source_ref="phone",
                match_class="strong_unique",
                first_seen_at=NOW,
                last_seen_at=NOW,
            )
        )
        store.upsert_identity_link(
            IdentityLink(
                tenant_id="foton",
                customer_id=customer.customer_id,
                link_type="amo_contact_id",
                link_value="777",
                source_system="amocrm_snapshot",
                source_ref="contact:777",
                match_class="strong_unique",
                first_seen_at=NOW,
                last_seen_at=NOW,
            )
        )
        opportunity = CustomerOpportunity(
            tenant_id="foton",
            customer_id=customer.customer_id,
            opportunity_type=OpportunityType.AMO_DEAL,
            source_system="amocrm_snapshot",
            source_id="888",
            title="Фотон математика",
            status="open",
            product_context={"brand": "foton"},
            opened_at=NOW,
            confidence=0.99,
        )
        store.upsert_opportunity(opportunity)
        call = TimelineEvent(
            tenant_id="foton",
            customer_id=customer.customer_id,
            opportunity_id=opportunity.opportunity_id,
            event_type="mango_call",
            event_at=NOW,
            source_system="mango",
            source_id="call-1",
            direction="inbound",
            summary="Клиент обсудил годовой курс по математике и попросил прислать расписание.",
            match_status="strong_unique",
            record={"call_analysis": {"history_summary": "Клиент обсудил годовой курс по математике.", "call_history_eligible": True}},
            created_at=NOW,
        )
        store.upsert_event(call)
        mail = TimelineEvent(
            tenant_id="foton",
            customer_id=customer.customer_id,
            opportunity_id=opportunity.opportunity_id,
            event_type="email_message",
            event_at=NOW.replace(hour=11),
            source_system="mail_archive_stage2",
            source_id="mail-1",
            direction="outbound",
            summary=(
                "Письмо: отправлено расписание и стоимость курса. --- part --- Links: https://example.invalid"
                if raw_email_artifact
                else "Письмо: отправлено расписание и стоимость курса."
            ),
            match_status="strong_unique",
            record={"full_clean_text": "Отправлено расписание и стоимость курса."},
            created_at=NOW,
        )
        store.upsert_event(mail)
        store.upsert_signal(
            DerivedSignal(
                tenant_id="foton",
                customer_id=customer.customer_id,
                opportunity_id=opportunity.opportunity_id,
                event_id=call.event_id,
                signal_type="hot_streak",
                severity="high",
                evidence_text="Клиент активно отвечает.",
                recommended_action="Позвонить сегодня.",
                requires_manager_review=True,
                created_at=NOW,
            )
        )
    with sqlite3.connect(db_path) as con:
        con.executescript(
            """
            CREATE TABLE customer_purchases_v1 (
              tenant_id TEXT NOT NULL,
              customer_id TEXT NOT NULL,
              period TEXT NOT NULL,
              money_kind TEXT NOT NULL DEFAULT 'plan'
                CHECK (money_kind IN ('plan', 'fact')),
              total_in REAL,
              total_out REAL,
              deals_cnt INTEGER NOT NULL DEFAULT 0,
              last_purchase_at TEXT,
              sources_json TEXT NOT NULL,
              computability TEXT NOT NULL,
              code_version TEXT,
              PRIMARY KEY (tenant_id, customer_id, period, money_kind)
            );
            CREATE TABLE customer_objections_v1 (
              tenant_id TEXT NOT NULL,
              customer_id TEXT NOT NULL,
              source_event_id TEXT NOT NULL,
              source_channel TEXT NOT NULL,
              objection_type TEXT NOT NULL,
              quote_preview TEXT NOT NULL,
              budget_hint_rub INTEGER,
              price_sensitivity TEXT NOT NULL,
              speaker TEXT NOT NULL DEFAULT 'unknown',
              direction TEXT NOT NULL DEFAULT 'unknown',
              confidence TEXT NOT NULL DEFAULT 'low',
              extracted_at TEXT NOT NULL,
              extractor_version TEXT NOT NULL,
              PRIMARY KEY (tenant_id, customer_id, source_event_id, objection_type)
            );
            """
        )
        con.execute(
            "INSERT INTO customer_purchases_v1 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("foton", "customer:e5", "all", "fact", 12000, 0, 1, NOW.isoformat(), "{}", "computed", "test"),
        )
        con.execute(
            "INSERT INTO customer_purchases_v1 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("foton", "customer:e5", "all", "plan", 50000, 0, 1, NOW.isoformat(), "{}", "computed", "test"),
        )
        con.execute(
            "INSERT INTO customer_objections_v1 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "foton",
                "customer:e5",
                "call-1",
                "call",
                "price",
                "дорого, просит скидку",
                12000,
                "medium",
                "client",
                "outbound",
                "high",
                NOW.isoformat(),
                "test",
            ),
        )
    return db_path
