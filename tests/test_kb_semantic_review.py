from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.run_kb_semantic_review import run_kb_semantic_review


def test_semantic_review_blocks_implausible_client_price(tmp_path: Path) -> None:
    release = _write_release(
        tmp_path,
        facts=[
            {
                "fact_id": "fact:bad-price",
                "fact_key": "academic_year.total_lessons",
                "fact_type": "price",
                "brand": "foton",
                "allowed_for_client_answer": True,
                "client_safe_text": "Фотон: занятий за год — 35 ₽.",
                "structured_value": {"amount": 35, "currency": "RUB", "path": "academic_year.total_lessons"},
            }
        ],
    )

    report = run_kb_semantic_review(release)

    assert report["semantic_pass"] is False
    assert any(item["check_id"] == "implausible_low_client_price" for item in report["findings"])
    assert any(item["check_id"] == "non_money_path_became_price" for item in report["findings"])


def test_semantic_review_blocks_cross_brand_client_text(tmp_path: Path) -> None:
    release = _write_release(
        tmp_path,
        facts=[
            {
                "fact_id": "fact:cross-brand",
                "fact_key": "contacts.telegram",
                "fact_type": "contact",
                "brand": "foton",
                "allowed_for_client_answer": True,
                "client_safe_text": "Фотон: пишите в @unpkmfti.",
                "structured_value": {"path": "contacts.telegram"},
            }
        ],
    )

    report = run_kb_semantic_review(release)

    assert report["semantic_pass"] is False
    assert any(item["check_id"] == "cross_brand_foton_client_text" for item in report["findings"])


def test_semantic_review_blocks_technical_english_client_text(tmp_path: Path) -> None:
    release = _write_release(
        tmp_path,
        facts=[
            {
                "fact_id": "fact:machine-english",
                "fact_key": "city.prices.base",
                "fact_type": "price",
                "brand": "foton",
                "allowed_for_client_answer": True,
                "client_safe_text": "Фотон: prices , base — 34 300 ₽.",
                "structured_value": {"amount": 34300, "currency": "RUB", "path": "city.prices.base"},
            }
        ],
    )

    report = run_kb_semantic_review(release)

    assert report["semantic_pass"] is False
    assert any(item["check_id"] == "technical_english_in_client_fact" for item in report["findings"])


def test_semantic_review_blocks_bad_rop_queue_priority(tmp_path: Path) -> None:
    release = _write_release(
        tmp_path,
        facts=[
            {
                "fact_id": "fact:internal",
                "fact_key": "internal.price",
                "fact_type": "price",
                "brand": "internal",
                "allowed_for_client_answer": False,
                "client_safe_text": "",
                "structured_value": {"amount": 50000, "currency": "RUB", "path": "internal.price"},
            }
        ],
        approval_rows=[
            {
                "priority": "P0",
                "approval_item_id": "approve:internal",
                "suggested_decision": "keep_internal_only",
                "rop_question": "Можно ли использовать этот факт в ответе клиенту текущего бренда?",
            }
        ],
    )

    report = run_kb_semantic_review(release)

    assert report["semantic_pass"] is False
    assert any(item["check_id"] == "p0_keep_internal_only" for item in report["findings"])


def test_semantic_review_accepts_manager_only_wording_for_internal_items(tmp_path: Path) -> None:
    release = _write_release(
        tmp_path,
        facts=[
            {
                "fact_id": "fact:internal",
                "fact_key": "internal.note",
                "fact_type": "policy",
                "brand": "internal",
                "allowed_for_client_answer": False,
                "client_safe_text": "",
                "structured_value": {"path": "internal.note"},
            }
        ],
        approval_rows=[
            {
                "priority": "P2",
                "approval_item_id": "approve:internal",
                "suggested_decision": "keep_internal_only",
                "rop_question": "Оставляем только для менеджера: подтвердите, что клиентская версия не нужна.",
            }
        ],
    )

    report = run_kb_semantic_review(release)

    assert report["semantic_pass"] is True
    assert report["findings"] == []


def test_semantic_review_passes_minimal_good_release(tmp_path: Path) -> None:
    release = _write_release(
        tmp_path,
        facts=[
            {
                "fact_id": "fact:good-price",
                "fact_key": "prices.offline_5_11.year",
                "fact_type": "price",
                "brand": "foton",
                "allowed_for_client_answer": True,
                "client_safe_text": "Фотон: цена за год для 5-11 класса — 74 500 ₽.",
                "structured_value": {"amount": 74500, "currency": "RUB", "path": "prices.offline_5_11.year"},
            },
            {
                "fact_id": "fact:good-lessons",
                "fact_key": "academic_year.total_lessons",
                "fact_type": "course_parameter",
                "brand": "foton",
                "allowed_for_client_answer": True,
                "client_safe_text": "Фотон: за год проходит 35 занятий.",
                "structured_value": {"count": 35, "unit": "lessons", "path": "academic_year.total_lessons"},
            },
        ],
        approval_rows=[
            {
                "priority": "P0",
                "approval_item_id": "approve:price",
                "suggested_decision": "approve_for_client_answer_after_rop_review",
                "rop_question": "Подтверждаете цену 74 500 ₽ для Фотона, 5-11 класс, год?",
            }
        ],
    )

    report = run_kb_semantic_review(release)

    assert report["semantic_pass"] is True
    assert report["findings"] == []


def _write_release(
    tmp_path: Path,
    *,
    facts: list[dict],
    approval_rows: list[dict] | None = None,
) -> Path:
    release = tmp_path / "release"
    release.mkdir()
    snapshot = {
        "quality_summary": {"quality_passed": True},
        "facts": facts,
        "summary": {"facts_total": len(facts)},
    }
    (release / "kb_release_v3_snapshot.json").write_text(json.dumps(snapshot, ensure_ascii=False), encoding="utf-8")
    if approval_rows is not None:
        fieldnames = sorted({key for row in approval_rows for key in row})
        with (release / "approval_queue_for_rop_v3.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(approval_rows)
    return release
