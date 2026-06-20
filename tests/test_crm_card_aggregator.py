from __future__ import annotations

import json
from pathlib import Path

from openpyxl import load_workbook

from mango_mvp.crm_card_aggregator import (
    apply_deal_card_payload,
    build_crm_card_projection,
    compact_objection_explicit,
)
from mango_mvp.crm_card_workbook import CrmCardWorkbookConfig, build_crm_card_workbook
from mango_mvp.customer_timeline import CustomerTimelineReadApi, CustomerTimelineReadApiConfig
from tests.test_customer_timeline_read_api import seed_timeline_db


def _profile(tmp_path: Path) -> dict:
    db_path, customer_id = seed_timeline_db(tmp_path)
    with CustomerTimelineReadApi.open(CustomerTimelineReadApiConfig(timeline_db=db_path, allowed_root=tmp_path)) as api:
        return dict(api.customer_profile("foton", customer_id, event_limit=10, bot_context_limit=10))


def test_crm_card_aggregator_builds_two_projections_from_one_profile(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    profile["manager_projection"] = {
        "amo_contact_ids": ["123"],
        "amo_lead_ids": ["456"],
        "identity_links": [
            {"link_type": "amo_contact_id", "link_value": "123", "match_class": "strong_unique"},
            {"link_type": "amo_lead_id", "link_value": "456", "match_class": "strong_unique"},
        ],
        "opportunities": [
            {
                "opportunity_id": "opp-manager",
                "opportunity_type": "amo_deal",
                "source_system": "amocrm_snapshot",
                "source_id": "456",
                "title": "ЕГЭ математика",
                "status": "open",
            }
        ],
    }

    card = build_crm_card_projection(
        profile,
        manager_facts={
            "Следующий шаг": "Согласовать дату звонка",
            "Возражения": "цена",
            "Вероятность продажи, %": "70",
            "AI-бюджет диапазон": "50k_100k",
            "AMO contact IDs": "123",
            "selected_deal_id": "456",
        },
    )

    assert card["snapshot_as_of"] == profile["snapshot_as_of"]
    assert card["deal_card"]["fields"]["Следующий шаг"] == "Перезвонить (от 12.05.2026)"
    assert "AI-бюджет диапазон" not in card["deal_card"]["fields"]
    assert "AI-дата обновления сделки" not in card["deal_card"]["fields"]
    assert set(card["contact_card"]["fields"]).issubset({"Запрос", "Последняя сводка", "История общения"})
    assert set(card["deal_card"]["fields"]).issubset({"Статус сделки", "Возражения", "Следующий шаг", "Tallanto", "Предупреждения"})
    assert card["bot_safety"]["money_fields_manager_only"] is True


def test_crm_card_uses_full_call_analysis_and_filters_non_conversation() -> None:
    live_summary = "Полный разбор живого звонка. " + ("Клиент обсуждал курс и оплату. " * 20)
    older_summary = "Полный разбор предыдущего звонка: семья выбирала формат и предмет."
    profile = {
        "found": True,
        "customer_id": "customer:call-analysis",
        "snapshot_as_of": "2026-06-18T10:00:00+00:00",
        "last_event_at": "2026-06-18T10:00:00+00:00",
        "customer": {"customer_id": "customer:call-analysis", "identity_status": "strong", "display_name": "Татьяна Николаевна", "summary": {}},
        "customer_id_mappings": [],
        "identity_links": [{"match_class": "strong_unique"}],
        "manager_projection": {
            "amo_contact_ids": ["123"],
            "amo_lead_ids": ["456"],
            "opportunities": [{"opportunity_id": "opp1", "opportunity_type": "amo_deal", "source_system": "amocrm_snapshot", "source_id": "456"}],
        },
        "opportunities": [],
        "timeline": {
            "items": [
                {
                    "event_type": "amo_contact_snapshot",
                    "event_at": "2026-06-19T10:00:00+00:00",
                    "source_system": "amocrm_snapshot",
                    "summary": "Read-only AMO contact snapshot",
                    "text_preview": "Read-only AMO contact snapshot",
                },
                {
                    "event_type": "tallanto_student_snapshot",
                    "event_at": "2026-06-19T09:00:00+00:00",
                    "source_system": "tallanto_snapshot",
                    "summary": "exact_phone_single",
                    "text_preview": "Статус: exact_phone_single | Тип ученика: 7 класс | Филиал: Фотон",
                },
                {
                    "event_type": "amo_deal_stage",
                    "event_at": "2026-06-19T08:00:00+00:00",
                    "source_system": "amocrm_snapshot",
                    "summary": "Закрыто и не реализовано",
                    "stage_after": "Закрыто и не реализовано",
                },
                {
                    "event_type": "mango_call",
                    "event_at": "2026-06-18T10:00:00+00:00",
                    "source_system": "mango",
                    "summary": "Недозвон не должен попасть в историю",
                    "call_type": "non_conversation",
                    "call_history_eligible": False,
                    "call_analysis": {
                        "history_summary": "Недозвон не должен попасть в историю",
                        "call_type": "non_conversation",
                        "call_history_eligible": False,
                    },
                },
                {
                    "event_type": "mango_call",
                    "event_at": "2026-06-17T10:00:00+00:00",
                    "source_system": "mango",
                    "summary": live_summary,
                    "call_type": "sales_call",
                    "call_history_eligible": True,
                    "call_analysis": {
                        "history_summary": live_summary,
                        "call_type": "sales_call",
                        "call_history_eligible": True,
                        "objections": ["цена"],
                        "next_step": "Перезвонить завтра",
                        "pain_points": ["нет времени"],
                        "interests": ["математика", "математика"],
                        "target_product": "годовой курс",
                        "structured_fields": {
                            "people": {"child_fio": "Анна"},
                            "student": {"grade_current": "7"},
                            "interests": {
                                "products": ["годовой курс", "годовой курс"],
                                "subjects": ["математика"],
                            },
                        },
                    },
                },
                {
                    "event_type": "mango_call",
                    "event_at": "2026-06-16T10:00:00+00:00",
                    "source_system": "mango",
                    "summary": older_summary,
                    "call_type": "sales_call",
                    "call_history_eligible": True,
                    "call_analysis": {
                        "history_summary": older_summary,
                        "call_type": "sales_call",
                        "call_history_eligible": True,
                        "next_step": "Дождаться выбора формата",
                        "interests": ["физика"],
                    },
                },
            ]
        },
        "signals": [],
        "bot_context": {"items": []},
        "conflicts": {"items": [], "summary": {"open_conflicts": 0}},
        "readiness": {"open_conflicts": 0},
    }

    card = build_crm_card_projection(profile)
    history = card["contact_card"]["fields"]["История общения"]
    preview = card["workbook"]["what_goes_to_amo"]

    assert card["contact_card"]["fields"]["Запрос"] == "Анна, 7 класс; годовой курс; математика"
    assert "Татьяна Николаевна" not in card["contact_card"]["fields"]["Запрос"]
    assert card["contact_card"]["fields"]["Последняя сводка"] == "Сводка:\n" + live_summary.strip()
    assert "Полный разбор живого звонка" not in history
    assert preview.count("Полный разбор живого звонка") == 1
    assert older_summary in history
    assert preview.count(older_summary) == 1
    assert "Возражения: цена" not in history
    assert "Интересы: математика" not in history
    assert "Шаг: Перезвонить завтра (от 17.06.2026)" in history
    assert "Недозвон не должен попасть" not in history
    assert "Read-only AMO contact snapshot" not in history
    assert "exact_phone_single" not in history
    assert "Закрыто и не реализовано" not in history
    assert "Закрыто и не реализовано" in card["deal_card"]["fields"]["Статус сделки"]
    assert card["deal_card"]["fields"]["Следующий шаг"] == "Перезвонить завтра (от 17.06.2026)"
    assert card["deal_card"]["fields"]["Возражения"] == "цена"
    assert "[сжато]" not in history
    assert "Read-only AMO contact snapshot" not in preview
    assert "exact_phone_single" not in preview
    assert "Tallanto: найден один ученик по телефону." not in history
    assert card["deal_card"]["fields"]["Tallanto"] == "Один ученик по телефону\nТип ученика: 7 класс\nФилиал: Фотон"
    assert card["workbook"]["ready"] == "да"


def test_crm_card_empty_timeline_falls_back_to_analyze_and_is_idempotent() -> None:
    profile = {
        "found": True,
        "customer_id": "customer:empty",
        "snapshot_as_of": "2026-06-18T10:00:00+00:00",
        "last_event_at": None,
        "customer": {"customer_id": "customer:empty", "identity_status": "strong", "summary": {}},
        "customer_id_mappings": [{"old_customer_id": "customer:legacy", "new_customer_id": "customer:empty"}],
        "identity_links": [{"match_class": "strong_unique"}],
        "opportunities": [],
        "timeline": {"items": []},
        "signals": [],
        "bot_context": {"items": []},
        "conflicts": {"items": [], "summary": {"open_conflicts": 0}},
        "readiness": {"open_conflicts": 0},
    }
    facts = {
        "Краткая история общения": "Клиент интересовался курсом.",
        "Следующий шаг": "Перезвонить и уточнить предмет.",
        "AMO contact IDs": "123",
    }

    first = build_crm_card_projection(profile, manager_facts=facts)
    second = build_crm_card_projection(profile, manager_facts=facts)

    assert first == second
    assert first["contact_card"]["fields"]["Последняя сводка"] == "Сводка:\nКлиент интересовался курсом."
    assert first["deal_card"]["fields"]["Следующий шаг"] == "Перезвонить и уточнить предмет."


def test_crm_card_ambiguous_identity_blocks_ready_and_keeps_family_summary() -> None:
    profile = {
        "found": True,
        "customer_id": "customer:family",
        "snapshot_as_of": "2026-06-18T10:00:00+00:00",
        "customer": {"customer_id": "customer:family", "identity_status": "ambiguous", "summary": {}},
        "customer_id_mappings": [{"old_customer_id": "old", "new_customer_id": "customer:family"}],
        "identity_links": [{"match_class": "ambiguous", "link_type": "phone"}],
        "opportunities": [{"opportunity_id": "opp1", "opportunity_type": "amo_deal", "title": "ЛВШ", "status": "open"}],
        "timeline": {"items": [{"event_at": "2026-06-18T10:00:00+00:00", "summary": "Семейный контакт: несколько учеников."}]},
        "signals": [],
        "bot_context": {"items": []},
        "conflicts": {"items": [{"status": "open", "conflict_type": "ambiguous_identity"}], "summary": {"open_conflicts": 1}},
        "readiness": {"open_conflicts": 1},
    }

    card = build_crm_card_projection(profile, manager_facts={"AMO contact IDs": "123", "selected_deal_id": "456"})

    assert card["contact_card"]["fields"]["Последняя сводка"] == "Сводка:\nСемейный контакт: несколько учеников."
    assert card["workbook"]["ready"] == "нет"
    assert "На телефоне несколько человек — проверьте, к кому относится" in card["workbook"]["blockers"]
    assert card["deal_card"]["ready_for_amo"] is False


def test_deal_card_payload_is_replaced_only_when_flag_enabled(monkeypatch) -> None:
    base = {"Статус сделки": "old"}
    card = {"Статус сделки": "new"}
    monkeypatch.delenv("CRM_CARD_AGGREGATOR_ENABLED", raising=False)

    assert apply_deal_card_payload(base, card)["Статус сделки"] == "old"

    monkeypatch.setenv("CRM_CARD_AGGREGATOR_ENABLED", "1")
    assert apply_deal_card_payload(base, card)["Статус сделки"] == "new"


def test_explicit_objection_compaction_marker_keeps_text_visible() -> None:
    text = " ".join(["клиент сомневается из-за стоимости программы"] * 5)

    compacted = compact_objection_explicit(text, limit=90)

    assert len(compacted) <= 90
    assert compacted.endswith("[сжато]")
    assert "…" not in compacted


def test_crm_card_workbook_is_transient_read_only_xlsx(tmp_path: Path) -> None:
    db_path, _customer_id = seed_timeline_db(tmp_path)
    out_xlsx = tmp_path / "crm_cards.xlsx"

    summary = build_crm_card_workbook(
        CrmCardWorkbookConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            out_xlsx=out_xlsx,
            sample_size=2,
        )
    )

    assert summary["generated_at"] == "2026-05-12T12:01:00+00:00"
    assert summary["safety"]["write_amo"] is False
    assert summary["safety"]["new_storage_table"] is False
    assert out_xlsx.exists()
    assert out_xlsx.with_suffix(".csv").exists()
    wb = load_workbook(out_xlsx)
    assert wb.sheetnames == ["Сводка", "Клиенты"]
    assert [cell.value for cell in wb["Клиенты"][1]][:17] == [
        "Имя",
        "Телефон",
        "Бренд",
        "Открыть в AMO",
        "Запрос",
        "Статус сделки",
        "Возражения",
        "Следующий шаг",
        "Последняя сводка",
        "Tallanto",
        "Предупреждения",
        "История общения",
        "Готово",
        "Блокеры",
        "Вердикт",
        "Комментарий",
        "customer_id",
    ]
    phone_cell = wb["Клиенты"]["B2"].value
    assert phone_cell and "***" not in str(phone_cell)
    csv_rows = out_xlsx.with_suffix(".csv").read_text(encoding="utf-8-sig")
    assert "crm_card_contact_payload_json" in csv_rows
    saved_summary = json.loads(out_xlsx.with_suffix(".summary.json").read_text(encoding="utf-8"))
    assert saved_summary["generated_at"] == summary["generated_at"]
