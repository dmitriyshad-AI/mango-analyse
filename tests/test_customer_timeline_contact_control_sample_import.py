from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mango_mvp.customer_timeline.contact_control_sample_import import (
    ContactControlTimelineSampleConfig,
    audit_contact_control_timeline_sample,
    build_contact_control_timeline_sample,
    select_contact_control_rows,
)
from mango_mvp.customer_timeline.context_provider import get_customer_context_for_phone


NOW = datetime(2026, 5, 16, 12, 0, tzinfo=timezone.utc)


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _contact(
    phone: str,
    *,
    calls: str,
    last: str,
    tallanto: str,
    amo: str = "",
    text: str = "Клиент интересуется курсом. Оплатить счет позже.",
) -> dict[str, str]:
    return {
        "Телефон клиента": phone,
        "Всего звонков в истории": calls,
        "Содержательных звонков в истории": calls,
        "Первый звонок": "2026-01-10 10:00:00",
        "Последний звонок": last,
        "Краткая история общения": text,
        "Хронология общения (последние 5 касаний)": "2026-01-10: обсуждали курс",
        "ФИО родителя": "Иван Петров",
        "Email": "ivan@example.com",
        "Рекомендуемый продукт": "Годовой курс",
        "Продукты интереса": "математика",
        "Возражения": "дорого",
        "Следующий шаг": "Перезвонить",
        "Приоритет лида": "warm",
        "Вероятность продажи, %": "55",
        "Статус матчинга Tallanto": tallanto,
        "Количество кандидатов Tallanto": "2" if tallanto == "exact_phone_multiple" else ("1" if tallanto == "exact_phone_single" else "0"),
        "ID Tallanto": "student-1" if tallanto == "exact_phone_single" else "",
        "ФИО родителя Tallanto": "Иван Петров" if tallanto == "exact_phone_single" else "",
        "Контакт Tallanto": "Иван Петров" if tallanto == "exact_phone_single" else "",
        "Тип ученика Tallanto": "10 класс" if tallanto == "exact_phone_single" else "",
        "AMO contact IDs": "",
        "AMO lead IDs": amo,
        "Outcome source": "tallanto_match" if tallanto == "exact_phone_single" else "unknown",
        "Utility score": "10",
        "Нужна ручная проверка": "Нет",
    }


def _config(tmp_path: Path) -> ContactControlTimelineSampleConfig:
    contacts = tmp_path / "master_contacts_ru.csv"
    calls = tmp_path / "master_calls_ru.csv"
    exclude = tmp_path / "exclude.csv"
    _write_csv(
        contacts,
        [
            _contact("+79161230001", calls="2", last="2026-04-10 10:00:00", tallanto="no_exact_phone_match", text="Клиент недавно выбирал курс."),
            _contact("+79161230002", calls="1", last="2025-06-10 10:00:00", tallanto="exact_phone_single", text="Клиент раньше занимался математикой."),
            _contact("+79161230003", calls="1", last="2025-12-10 10:00:00", tallanto="no_exact_phone_match", text="Новый лид без сделки."),
            _contact("+79161230004", calls="1", last="2026-01-10 10:00:00", tallanto="exact_phone_single", text="Клиент уточнял оплату и договор."),
            _contact("+79161230005", calls="2", last="2026-04-10 10:00:00", tallanto="no_exact_phone_match", text="Эта строка исключается."),
        ],
    )
    _write_csv(
        calls,
        [
            {
                "ID звонка": f"call-{idx}",
                "Дата и время звонка": "2026-04-10 10:00:00",
                "Телефон клиента": phone,
                "Менеджер": "Настя",
                "Направление звонка": "inbound",
                "Содержательный звонок": "Да",
                "Краткое резюме разговора": "Клиент обсуждал курс и следующий шаг.",
                "Тип звонка": "sales_call",
            }
            for idx, phone in enumerate(["+79161230001", "+79161230002", "+79161230003", "+79161230004", "+79161230005"], start=1)
        ],
    )
    _write_csv(exclude, [{"phones": "+79161230005"}])
    return ContactControlTimelineSampleConfig(
        master_contacts_csv=contacts,
        master_calls_csv=calls,
        exclude_phones_csv=exclude,
        allowed_root=tmp_path / "product_data",
        out_root=tmp_path / "product_data" / "customer_timeline" / "control",
        timeline_db=tmp_path / "product_data" / "customer_timeline" / "control" / "customer_timeline.sqlite",
        generated_at=NOW,
        target_bucket_counts={
            "active_recent": 1,
            "former_tallanto": 1,
            "new_lead_no_deal": 1,
            "paid_or_success": 1,
        },
    )


def test_selector_builds_deterministic_control_buckets_and_excludes_dealaware_phones(tmp_path: Path) -> None:
    config = _config(tmp_path)
    rows = []
    with config.master_contacts_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]

    selected, pool_counts = select_contact_control_rows(
        rows,
        excluded_phones={"+79161230005"},
        target_bucket_counts=config.target_bucket_counts,
    )

    assert [row["control_bucket"] for row in selected] == [
        "active_recent",
        "former_tallanto",
        "new_lead_no_deal",
        "paid_or_success",
    ]
    assert "+79161230005" not in {row["primary_phone"] for row in selected}
    assert pool_counts["active_recent"] == 1


def test_hard_selector_builds_deterministic_risk_buckets(tmp_path: Path) -> None:
    rows = [
        _contact("+79161230101", calls="1", last="2026-01-10 10:00:00", tallanto="no_exact_phone_match", text="Клиент выбирает курс."),
        _contact("+79161230102", calls="1", last="2026-01-10 10:00:00", tallanto="exact_phone_multiple", text="Клиент выбирает курс."),
        _contact("+79161230103", calls="2", last="2026-01-10 10:00:00", tallanto="exact_phone_single", amo="lead-1 | lead-2", text="Клиент выбирает курс."),
        _contact("+79161230104", calls="8", last="2026-01-10 10:00:00", tallanto="exact_phone_single", text="Длинная история выбора курса."),
        _contact("+79161230105", calls="1", last="2026-01-10 10:00:00", tallanto="exact_phone_single", text="Клиент просил счет и договор."),
    ]

    selected, pool_counts = select_contact_control_rows(
        rows,
        excluded_phones=set(),
        target_bucket_counts={
            "no_reliable_tallanto": 1,
            "tallanto_multiple": 1,
            "multi_phone_or_multi_deal": 1,
            "long_history": 1,
            "payment_or_documents_risk": 1,
        },
        sample_profile="hard",
    )

    assert [row["control_bucket"] for row in selected] == [
        "no_reliable_tallanto",
        "tallanto_multiple",
        "multi_phone_or_multi_deal",
        "long_history",
        "payment_or_documents_risk",
    ]
    assert pool_counts["tallanto_multiple"] == 1
    assert {row["sample_kind"] for row in selected} == {"hard_contact_control"}


def test_builds_local_timeline_from_ordinary_contacts_without_required_deal(tmp_path: Path) -> None:
    config = _config(tmp_path)

    report = build_contact_control_timeline_sample(config)
    context = get_customer_context_for_phone("+79161230001", timeline_db=config.timeline_db)

    assert report["summary"]["selected_contacts"] == 4
    assert report["summary"]["selected_unique_phones"] == 4
    assert report["summary"]["store_counts"]["customer_identities"] == 4
    assert context["source"] == "customer_timeline"
    assert context["fallback_used"] is False
    assert context["readiness"]["safe_for_automatic_bot"] is False
    assert context["readiness"]["bot_review_required_chunks"] > 0
    assert context["bot_context"]["summary"]["allowed_chunks"] == 0


def test_audit_separates_timeline_found_from_valid_identity_and_preview(tmp_path: Path) -> None:
    config = _config(tmp_path)
    build_contact_control_timeline_sample(config)

    audit = audit_contact_control_timeline_sample(config)

    assert audit["summary"]["timeline_matched_contacts"] == 4
    assert audit["summary"]["valid_customer_identity"] == 4
    assert audit["summary"]["ready_for_preview"] == 4
    assert audit["rows"][0]["timeline_found"] == "Да"
    assert audit["rows"][0]["valid_customer_identity"] == "Да"
    assert audit["rows"][0]["identity_validation_status"] == "auto_valid"


def test_hard_audit_keeps_risky_control_rows_in_manual_review(tmp_path: Path) -> None:
    contacts = tmp_path / "hard_contacts.csv"
    calls = tmp_path / "hard_calls.csv"
    _write_csv(
        contacts,
        [
            _contact("+79161230201", calls="1", last="2026-01-10 10:00:00", tallanto="no_exact_phone_match", text="Клиент выбирает курс."),
            _contact("+79161230202", calls="1", last="2026-01-10 10:00:00", tallanto="exact_phone_multiple", text="Клиент выбирает курс."),
            _contact("+79161230203", calls="8", last="2026-01-10 10:00:00", tallanto="exact_phone_single", text="Длинная история выбора курса."),
        ],
    )
    _write_csv(
        calls,
        [
            {
                "ID звонка": f"hard-call-{idx}",
                "Дата и время звонка": "2026-01-10 10:00:00",
                "Телефон клиента": phone,
                "Менеджер": "Настя",
                "Направление звонка": "inbound",
                "Содержательный звонок": "Да",
                "Краткое резюме разговора": "Клиент обсуждал курс.",
                "Тип звонка": "sales_call",
            }
            for idx, phone in enumerate(["+79161230201", "+79161230202", "+79161230203"], start=1)
        ],
    )
    config = ContactControlTimelineSampleConfig(
        master_contacts_csv=contacts,
        master_calls_csv=calls,
        allowed_root=tmp_path / "product_data",
        out_root=tmp_path / "product_data" / "customer_timeline" / "hard",
        timeline_db=tmp_path / "product_data" / "customer_timeline" / "hard" / "customer_timeline.sqlite",
        generated_at=NOW,
        sample_profile="hard",
        target_bucket_counts={
            "no_reliable_tallanto": 1,
            "tallanto_multiple": 1,
            "long_history": 1,
        },
    )

    build_contact_control_timeline_sample(config)
    audit = audit_contact_control_timeline_sample(config)

    assert audit["summary"]["timeline_matched_contacts"] == 3
    assert audit["summary"]["valid_customer_identity"] == 3
    assert audit["summary"]["ready_for_preview"] == 0
    assert audit["summary"]["needs_manual_review"] == 3
    assert audit["summary"]["manual_review_by_category"] == {"hard_control_risk": 3}
    assert all("hard_" in row["not_ready_reasons"] for row in audit["rows"])


def test_output_root_must_not_be_under_stable_runtime(tmp_path: Path) -> None:
    config = _config(tmp_path)
    unsafe = ContactControlTimelineSampleConfig(
        master_contacts_csv=config.master_contacts_csv,
        master_calls_csv=config.master_calls_csv,
        exclude_phones_csv=config.exclude_phones_csv,
        allowed_root=tmp_path,
        out_root=tmp_path / "stable_runtime" / "customer_timeline",
        timeline_db=tmp_path / "stable_runtime" / "customer_timeline" / "customer_timeline.sqlite",
        generated_at=NOW,
    )

    with pytest.raises(ValueError, match="stable_runtime"):
        build_contact_control_timeline_sample(unsafe)
