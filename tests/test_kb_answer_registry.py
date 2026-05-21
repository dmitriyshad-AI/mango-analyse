from __future__ import annotations

from mango_mvp.knowledge_base.answer_registry import (
    AnswerRegistryEntry,
    contains_debug_leak,
    select_answer_blocks,
    semantic_passed,
    validate_answer_registry,
    validate_draft_semantics,
)


def test_answer_registry_has_brand_for_each_answer() -> None:
    entries = [
        _entry(answer_id="foton_price", brand="foton", source_ids=("src:foton-price",)),
        _entry(answer_id="unpk_price", brand="unpk", source_ids=("src:unpk-price",)),
    ]

    issues = validate_answer_registry(entries, known_source_ids={"src:foton-price", "src:unpk-price"})

    assert semantic_passed(issues)


def test_answer_registry_rejects_missing_source_for_precise_answer() -> None:
    entries = [_entry(answer_id="foton_price", brand="foton", route="draft_for_manager", source_ids=())]

    issues = validate_answer_registry(entries)

    assert any(issue.code == "missing_source_for_precise_answer" for issue in issues)
    assert not semantic_passed(issues)


def test_answer_registry_rejects_cross_brand_templates() -> None:
    entries = [
        _entry(
            answer_id="foton_cross_brand",
            brand="foton",
            source_ids=("src:foton",),
            template="В Фотоне условия такие, а в УНПК другие.",
        )
    ]

    issues = validate_answer_registry(entries, known_source_ids={"src:foton"})

    assert any(issue.code == "cross_brand_template" for issue in issues)


def test_manager_only_missing_fact_uses_safe_handoff() -> None:
    entries = [
        _entry(
            answer_id="foton_unknown_price",
            brand="foton",
            route="manager_only",
            source_ids=(),
            template="Передам вопрос менеджеру, он проверит актуальную стоимость.",
        )
    ]

    issues = validate_answer_registry(entries)

    assert semantic_passed(issues)


def test_combined_question_can_select_multiple_answer_blocks() -> None:
    entries = [
        _entry(answer_id="foton_price", brand="foton", topic="pricing", source_ids=("src:price",)),
        _entry(answer_id="foton_discount", brand="foton", topic="discount", source_ids=("src:discount",)),
        _entry(answer_id="unpk_price", brand="unpk", topic="pricing", source_ids=("src:unpk",)),
    ]

    selected = select_answer_blocks(entries, brand="foton", topics=("pricing", "discount"))

    assert [entry.answer_id for entry in selected] == ["foton_price", "foton_discount"]


def test_draft_semantic_gate_catches_debug_leak_and_identity() -> None:
    issues = validate_draft_semantics(
        draft_text="Как ИИ, вижу source_id=fact:123 и отвечаю клиенту.",
        brand="foton",
        route="draft_for_manager",
    )

    assert {issue.code for issue in issues} >= {"debug_leak", "bot_identity_leak"}


def test_draft_semantic_gate_catches_cross_brand_leak() -> None:
    issues = validate_draft_semantics(
        draft_text="В Фотоне условия такие, а в УНПК можно иначе.",
        brand="foton",
        route="draft_for_manager",
    )

    assert any(issue.code == "cross_brand_leak" for issue in issues)


def test_draft_semantic_gate_catches_implausible_price() -> None:
    issues = validate_draft_semantics(
        draft_text="Годовой курс стоит 35 рублей.",
        brand="unpk",
        route="draft_for_manager",
    )

    assert any(issue.code == "implausible_course_price" for issue in issues)


def test_draft_semantic_gate_catches_pii_collection_on_refund() -> None:
    issues = validate_draft_semantics(
        draft_text="Пришлите ФИО ребёнка, номер договора, телефон и сумму возврата.",
        brand="foton",
        route="manager_only",
        priority="P0",
        category="high_risk",
        subcategory="refund",
    )

    assert any(issue.code == "pii_collection_in_high_risk" for issue in issues)


def test_debug_leak_detector_accepts_normal_customer_text() -> None:
    assert not contains_debug_leak("Менеджер проверит актуальную стоимость и свяжется с вами.")


def _entry(
    *,
    answer_id: str,
    brand: str,
    topic: str = "pricing",
    route: str = "draft_for_manager",
    source_ids: tuple[str, ...] = ("src:default",),
    template: str = "Менеджер подскажет актуальную стоимость по выбранной программе.",
) -> AnswerRegistryEntry:
    return AnswerRegistryEntry(
        answer_id=answer_id,
        brand=brand,
        topic=topic,
        route=route,
        source_ids=source_ids,
        template=template,
    )
