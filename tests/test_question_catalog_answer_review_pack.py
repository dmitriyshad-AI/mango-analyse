import csv
import json
from pathlib import Path

from mango_mvp.question_catalog.answer_review_pack import (
    audit_review_rows,
    build_pack,
    build_review_rows,
    proposed_answer,
    select_review_classes,
)


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def _catalog(tmp_path: Path) -> Path:
    root = tmp_path / "question_catalog"
    classes = [
        {
            "question_class_id": "class:price",
            "parent_question_class": "стоимость",
            "question_subclass": "базовая стоимость",
            "canonical_question": "стоимость / базовая стоимость",
            "examples_for_rop": "Сколько стоит ЕГЭ по математике?",
            "examples_redacted": "Сколько стоит ЕГЭ по математике?",
            "count_total": "120",
            "count_calls": "5",
            "count_telegram": "4",
            "count_email": "111",
            "answer_status": "template_ready_needs_current_fact",
            "required_fact_keys": "price.current",
            "bot_permission": "allowed_after_fact_check",
        },
        {
            "question_class_id": "class:matcap",
            "parent_question_class": "оплата / возврат / чек",
            "question_subclass": "оплата материнским капиталом",
            "canonical_question": "оплата / возврат / чек / оплата материнским капиталом",
            "examples_for_rop": "Можно оплатить мат капиталом?",
            "examples_redacted": "Можно оплатить мат капиталом?",
            "count_total": "12",
            "count_calls": "1",
            "count_telegram": "1",
            "count_email": "10",
            "answer_status": "manager_only",
            "required_fact_keys": "documents.current",
            "bot_permission": "manager_only",
        },
        {
            "question_class_id": "class:receipt",
            "parent_question_class": "оплата / возврат / чек",
            "question_subclass": "чек, квитанция или счет",
            "canonical_question": "оплата / возврат / чек / чек, квитанция или счет",
            "examples_for_rop": "Пришлите квитанцию для оплаты.",
            "examples_redacted": "Пришлите квитанцию для оплаты.",
            "count_total": "90",
            "count_calls": "2",
            "count_telegram": "8",
            "count_email": "80",
            "answer_status": "manager_only",
            "required_fact_keys": "documents.current",
            "bot_permission": "manager_only",
        },
        {
            "question_class_id": "class:blocked",
            "parent_question_class": "общий вопрос",
            "question_subclass": "без уточненного подкласса",
            "canonical_question": "общий вопрос / без уточненного подкласса",
            "examples_for_rop": "Расскажите подробнее.",
            "examples_redacted": "Расскажите подробнее.",
            "count_total": "55",
            "count_calls": "0",
            "count_telegram": "2",
            "count_email": "53",
            "answer_status": "draft_answer_exists_needs_review",
            "required_fact_keys": "",
            "bot_permission": "draft_only_needs_review",
        },
        {
            "question_class_id": "class:noise",
            "parent_question_class": "общий вопрос",
            "question_subclass": "короткий обрывок внутри темы",
            "canonical_question": "общий вопрос / короткий обрывок внутри темы",
            "examples_for_rop": "угу",
            "examples_redacted": "угу",
            "count_total": "999",
            "count_calls": "0",
            "count_telegram": "0",
            "count_email": "999",
            "answer_status": "not_enough_context",
            "required_fact_keys": "",
            "bot_permission": "not_allowed",
        },
    ]
    _write_csv(root / "customer_question_classes.csv", classes)
    _write_csv(
        root / "rop_review_priority_top100.csv",
        [
            {
                "Место": "1",
                "Приоритетный балл": "500",
                "Класс вопроса": "стоимость / базовая стоимость",
            }
        ],
    )
    (root / "answer_quality_check_report.json").write_text(
        json.dumps(
            {
                "findings": [
                    {
                        "severity": "p1",
                        "code": "wide_class_block_until_split",
                        "question_class_id": "class:blocked",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (root / "current_fact_source_registry.json").write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "source_id": "price_1",
                        "fact_types": ["price"],
                        "path": "/tmp/prices.xlsx",
                        "approval_status": "manual_review_required",
                    },
                    {
                        "source_id": "doc_1",
                        "fact_types": ["documents"],
                        "path": "/tmp/docs.docx",
                        "approval_status": "manual_review_required",
                    },
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        root / "customer_question_items.jsonl",
        [
            {
                "question_class_id": "class:matcap",
                "source_channel": "telegram",
                "occurred_at": "2026-01-01T00:00:00+00:00",
                "customer_text_redacted": "Можно оплатить мат капиталом?",
                "manager_text_redacted": "Нет.",
                "metadata": {"customer_text_for_rop": "Можно оплатить мат капиталом?"},
            },
            {
                "question_class_id": "class:receipt",
                "source_channel": "telegram",
                "occurred_at": "2026-01-02T00:00:00+00:00",
                "customer_text_redacted": "Пришлите квитанцию для оплаты.",
                "manager_text_redacted": "Отправим актуальное окно записи.",
                "metadata": {"customer_text_for_rop": "Пришлите квитанцию для оплаты."},
            },
        ],
    )
    return root


def test_select_review_classes_excludes_noise_and_force_includes_matcap(tmp_path: Path) -> None:
    root = _catalog(tmp_path)

    rows = select_review_classes(root, row_limit=2)
    class_ids = {row["question_class_id"] for row in rows}

    assert "class:matcap" in class_ids
    assert "class:noise" not in class_ids


def test_build_review_rows_keeps_historical_answer_non_approved_and_safe(tmp_path: Path) -> None:
    root = _catalog(tmp_path)

    rows = build_review_rows(root, row_limit=4)
    by_id = {row["ID класса"]: row for row in rows}

    matcap = by_id["class:matcap"]
    assert not matcap["Предлагаемый ответ"].lower().startswith("нет")
    assert "материнским капиталом" in matcap["Предлагаемый ответ"]
    assert matcap["Исторический ответ менеджера (не утверждено)"] == "Нет."
    assert matcap["Почему исторический ответ не готов"] == "слишком короткий ответ; нельзя утверждать как шаблон"

    receipt = by_id["class:receipt"]
    assert "возврат" not in receipt["Предлагаемый ответ"].casefold()
    assert "[актуальная дата или окно записи]" in receipt["Исторический ответ менеджера (не утверждено)"]


def test_blocked_class_gets_split_action_not_ready_answer(tmp_path: Path) -> None:
    root = _catalog(tmp_path)

    rows = build_review_rows(root, row_limit=4)
    blocked = {row["ID класса"]: row for row in rows}["class:blocked"]

    assert blocked["Что бот может делать"] == "нельзя утверждать, сначала дробить класс"
    assert blocked["Много тем в одном вопросе"] == "да"


def test_refund_logic_ignores_broad_payment_parent_without_refund_intent() -> None:
    answer, _, action, _, _, multi_topic = proposed_answer(
        {
            "parent_question_class": "оплата / возврат / чек",
            "question_subclass": "расписание занятий",
            "canonical_question": "оплата / возврат / чек / расписание занятий",
            "examples_for_rop": "Когда будет занятие по математике?",
            "examples_redacted": "Когда будет занятие по математике?",
            "required_fact_keys": "schedule.current",
        },
        blocker_code="",
        fact_keys=["schedule"],
        groups=[],
    )

    assert "возврат" not in answer.casefold()
    assert "расписание" in answer.casefold()
    assert action == "после проверки актуального факта и утверждения РОПом"
    assert multi_topic == "нет"


def test_single_refund_example_inside_other_class_is_blocked_for_split() -> None:
    answer, why, action, _, rop_check, multi_topic = proposed_answer(
        {
            "parent_question_class": "оплата / возврат / чек",
            "question_subclass": "способ оплаты и сроки платежа",
            "canonical_question": "оплата / возврат / чек / способ оплаты и сроки платежа",
            "examples_for_rop": "Как оплатить курс? | Если деньги придут позже, вы нам вернете эти пятьдесят семь тысяч?",
            "examples_redacted": "Как оплатить курс? | Если деньги придут позже, вы нам вернете эти пятьдесят семь тысяч?",
            "required_fact_keys": "documents.current",
        },
        blocker_code="",
        fact_keys=["documents"],
        groups=["возврат / перерасчет"],
    )

    assert "единый ответ" in answer
    assert "смешаны возврат/перерасчет" in why
    assert action == "нельзя утверждать, сначала дробить класс"
    assert "Разбить класс" in rop_check
    assert multi_topic == "да"


def test_tax_deduction_documents_are_not_payment_refund() -> None:
    answer, _, action, _, _, _ = proposed_answer(
        {
            "parent_question_class": "письма / справки / подтверждающие документы",
            "question_subclass": "документы для налоговой",
            "canonical_question": "письма / справки / подтверждающие документы / документы для налоговой",
            "examples_for_rop": "Хочу получить документы на возврат НДФЛ. Нужна справка для налогового вычета.",
            "examples_redacted": "Хочу получить документы на возврат НДФЛ. Нужна справка для налогового вычета.",
            "required_fact_keys": "documents.current",
        },
        blocker_code="",
        fact_keys=["documents"],
        groups=[],
    )

    assert "возврату или перерасчету" not in answer
    assert action != "только менеджер"


def test_single_tax_example_does_not_override_receipt_class() -> None:
    answer, _, action, _, _, _ = proposed_answer(
        {
            "parent_question_class": "оплата / возврат / чек",
            "question_subclass": "чек, квитанция или счет",
            "canonical_question": "оплата / возврат / чек / чек, квитанция или счет",
            "examples_for_rop": "Пришлите чек об оплате. | Нужен чек для налогового вычета.",
            "examples_redacted": "Пришлите чек об оплате. | Нужен чек для налогового вычета.",
            "required_fact_keys": "documents.current",
        },
        blocker_code="",
        fact_keys=["documents"],
        groups=[],
    )

    assert "платежный документ" in answer
    assert "налогового вычета" not in answer
    assert action == "после проверки актуального факта и утверждения РОПом"


def test_audit_catches_forbidden_tokens_and_missing_placeholders() -> None:
    rows = [
        {
            "Номер": "1",
            "ID класса": "class:bad",
            "Класс вопроса": "оплата / материнский капитал",
            "Узкий класс": "оплата материнским капиталом",
            "Реальные примеры вопросов": "Можно оплатить мат капиталом?",
            "Предлагаемый ответ": "Нет. актуальное окно записи",
            "Что бот может делать": "готовый шаблон только после утверждения РОПом",
            "Нужные актуальные факты": "documents",
            "Источник факта": "",
            "Блокер качества": "",
            "Источники": json.dumps({"звонки": 1, "telegram": 1, "почта": 0}, ensure_ascii=False),
        }
    ]

    audit = audit_review_rows(rows, min_rows=1)

    assert audit["verdict"] == "blocked"
    codes = {finding["code"] for finding in audit["findings"]}
    assert "forbidden_token_in_proposed_answer" in codes
    assert "mat_capital_short_no" in codes
    assert "placeholder_missing_for_fact_dependent_class" in codes


def test_audit_catches_answer_topic_mismatch() -> None:
    rows = [
        {
            "Номер": "1",
            "ID класса": "class:wrong_topic",
            "Класс вопроса": "адрес / очная площадка / адрес площадки",
            "Узкий класс": "адрес площадки",
            "Реальные примеры вопросов": "Где проходят занятия?",
            "Предлагаемый ответ": "Да, подготовим и отправим платежный документ для оплаты.",
            "Что бот может делать": "после проверки актуального факта и утверждения РОПом",
            "Нужные актуальные факты": "location",
            "Источник факта": "location_source",
            "Блокер качества": "",
            "Источники": json.dumps({"звонки": 1, "telegram": 1, "почта": 0}, ensure_ascii=False),
        }
    ]

    audit = audit_review_rows(rows, min_rows=1)

    assert audit["verdict"] == "blocked"
    assert {finding["code"] for finding in audit["findings"]} >= {"location_topic_mismatch"}


def test_build_pack_writes_csv_and_summary(tmp_path: Path) -> None:
    root = _catalog(tmp_path)
    output_csv = tmp_path / "pack.csv"
    output_summary = tmp_path / "summary.json"

    result = build_pack(root, output_csv, output_summary, row_limit=4, iteration="iter_test")

    assert output_csv.exists()
    assert output_summary.exists()
    assert result["totals"]["rows"] >= 4
    assert result["audit"]["verdict"] == "pass"
