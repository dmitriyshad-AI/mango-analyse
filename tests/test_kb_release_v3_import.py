from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE_DIR = PROJECT_ROOT / "product_data" / "knowledge_base" / "kb_release_20260610_v6_7_staging_r2"

REQUIRED_APPROVAL_QUEUE_COLUMNS = {
    "priority",
    "approval_item_id",
    "item_type",
    "topic",
    "fact_id_ref",
    "brand",
    "product",
    "manager_text",
    "suggested_decision",
    "rop_question",
    "source_id",
    "linked_open_question",
    "risk_notes",
}

NON_MONEY_NUMERIC_PATH_MARKERS = (
    "total_lessons",
    "weekly_lessons",
    "daily_hours",
    "semester_1_weeks",
    "semester_2_weeks",
    "daily_pairs",
    "pair_duration_minutes",
    "classes",
    "start",
    "retroactive_years",
)

EXPECTED_APPROVAL_ITEM_TYPES = {
    "price",
    "discount",
    "promocode",
    "deadline",
    "lvsh",
    "program",
    "installment",
    "tax",
    "matkap",
}

EXPECTED_NUMERIC_FACTS: tuple[dict[str, Any], ...] = (
    {"amount": 44600, "brand": "foton", "tokens": ("offline", "5", "11")},
    {"amount": 74500, "brand": "foton", "tokens": ("offline", "5", "11")},
    {"amount": 29750, "brand": "foton", "tokens": ("online", "5", "11")},
    {"amount": 47250, "brand": "foton", "tokens": ("online",)},
    {"amount": 98000, "brand": "foton", "tokens": ("lvsh", "mendeleevo")},
    {"amount": 3900, "brand": "foton", "tokens": ("individual", "lesson")},
    {"amount": 6900, "brand": "foton", "tokens": ("individual", "lesson")},
    {"amount": 23000, "brand": "foton", "tokens": ("individual", "lesson")},
    {"amount": 49000, "brand": "unpk", "tokens": ("offline", "5", "11")},
    {"amount": 82000, "brand": "unpk", "tokens": ("offline", "5", "11")},
    {"amount": 41800, "brand": "unpk", "tokens": ("online", "5", "11")},
    {"amount": 69900, "brand": "unpk", "tokens": ("online", "5", "11")},
    {"amount": 114000, "brand": "unpk", "tokens": ("lvsh", "mendeleevo")},
    {"amount": 33000, "brand": "unpk", "tokens": ("fiztech", "olympiad")},
    {"amount": 50000, "brand": "unpk", "tokens": ("fiztech", "olympiad")},
    {"amount": 18800, "brand": "unpk", "tokens": ("intensive", "ege")},
    {"amount": 34400, "brand": "unpk", "tokens": ("intensive", "ege")},
)

FORBIDDEN_TO_SAY_TERMS = (
    "в УНПК рассрочки нет",
    "у наших партнёров",
    "Фотон даёт, а другие нет",
    "Т-Банк",
    "Долями",
    "Раньше были одно",
    "Это наш партнёр",
    "У них есть",
)

ALWAYS_FORBIDDEN_TO_SAY_CLIENT_TERMS = (
    "в УНПК рассрочки нет",
    "у наших партнёров",
    "Фотон даёт, а другие нет",
    "Раньше были одно",
    "Это наш партнёр",
)

LICENSE_AND_INTERNAL_MARKERS = (
    "Л035",
    "50Л01",
    "№77753",
    "70369",
    "АНО ДПО",
    "НОУ УНПК",
    "internal_only_for_number",
    "_internal",
)

FOTON_CLIENT_FORBIDDEN_MARKERS = (
    "УНПК",
    "АНО ДПО",
    "НОУ УНПК",
    "kmipt.ru",
    "@unpk_mipt",
    "лицензия 70369",
    "лицензия № 70369",
    "Сретенка",
    "Сретенка, 20",
    "edu@kmipt.ru",
)

UNPK_CLIENT_FORBIDDEN_MARKERS = (
    "Фотон",
    "ЦДПО",
    "ЦРДО",
    "cdpofoton.ru",
    "edu@cdpofoton.ru",
    "Т-Банк",
    "Долями",
)


@dataclass(frozen=True)
class KbReleaseV3:
    root: Path
    snapshot: Mapping[str, Any]
    facts: list[Mapping[str, Any]]
    sources: list[Mapping[str, Any]]
    source_ids: set[str]
    approval_queue: list[Mapping[str, str]]
    quality_report: Mapping[str, Any]
    brand_rules: Mapping[str, Any]
    bot_policy: Mapping[str, Any]


@pytest.fixture(scope="module")
def kb_v3() -> KbReleaseV3:
    root = Path(os.environ.get("KB_RELEASE_V3_DIR", str(DEFAULT_RELEASE_DIR)))
    if not root.exists():
        pytest.fail(
            "Expected KB v3 release outputs at "
            f"{root}. Run the v3 builder before this import test layer."
        )

    snapshot = _load_json_required(root / "kb_release_v3_snapshot.json")
    facts = _load_facts(root, snapshot)
    sources = _load_sources(root, snapshot)
    source_ids = {str(source.get("source_id") or "") for source in sources if source.get("source_id")}
    approval_queue = _load_csv_required(root / "approval_queue_for_rop_v3.csv")
    quality_report = _load_json_required(root / "quality_report.json")
    brand_rules = _load_mapping_artifact(root, "brand_rules")
    bot_policy = _load_mapping_artifact(root, "bot_policy")

    if not facts:
        pytest.fail(f"KB v3 facts registry is empty in {root}")
    if not sources:
        pytest.fail(f"KB v3 source registry is empty in {root}")

    return KbReleaseV3(
        root=root,
        snapshot=snapshot,
        facts=facts,
        sources=sources,
        source_ids=source_ids,
        approval_queue=approval_queue,
        quality_report=quality_report,
        brand_rules=brand_rules,
        bot_policy=bot_policy,
    )


def test_v3_expands_nested_numeric_blocks_into_atomic_facts(kb_v3: KbReleaseV3) -> None:
    for expectation in EXPECTED_NUMERIC_FACTS:
        matching = _facts_for_amount(
            kb_v3.facts,
            amount=expectation["amount"],
            brand=expectation["brand"],
            tokens=expectation["tokens"],
        )
        assert matching, f"Missing atomic fact for {expectation}"

        for fact in matching:
            assert fact.get("fact_id"), fact
            assert fact.get("fact_key"), fact
            assert fact.get("fact_type"), fact
            assert fact.get("brand") == expectation["brand"], fact
            assert str(fact.get("fact_text") or "").strip(), fact
            assert fact.get("source_id"), fact
            assert str(fact.get("source_id")) in kb_v3.source_ids, fact
            assert _structured_value_text(fact), f"Atomic numeric fact has no structured_value: {fact}"
            if _is_true(fact.get("allowed_for_client_answer")):
                assert str(fact.get("client_safe_text") or "").strip(), fact


def test_v3_keeps_forbidden_to_say_out_of_client_facts_and_post_filter_blocks_it(
    kb_v3: KbReleaseV3,
) -> None:
    allowed_client_facts = _allowed_client_facts(kb_v3.facts)
    assert allowed_client_facts

    for fact in allowed_client_facts:
        client_blob = _lower_blob(
            fact,
            fields=("fact_id", "fact_key", "fact_text", "client_safe_text", "title"),
        )
        assert "forbidden_to_say" not in client_blob, fact
        for term in ALWAYS_FORBIDDEN_TO_SAY_CLIENT_TERMS:
            assert term.casefold() not in client_blob, fact

    post_filter_terms = _post_filter_terms(kb_v3)
    missing = [
        term
        for term in FORBIDDEN_TO_SAY_TERMS
        if term.casefold() not in post_filter_terms
    ]
    assert not missing, f"forbidden_to_say terms are not represented in post-filter outputs: {missing}"


def test_v3_internal_only_for_number_keeps_license_numbers_out_of_client_safe_text(
    kb_v3: KbReleaseV3,
) -> None:
    allowed_client_facts = _allowed_client_facts(kb_v3.facts)
    assert allowed_client_facts

    for fact in allowed_client_facts:
        client_text = str(fact.get("client_safe_text") or "")
        for marker in LICENSE_AND_INTERNAL_MARKERS:
            assert marker.casefold() not in client_text.casefold(), fact

    license_summary_facts = [
        fact
        for fact in allowed_client_facts
        if "лиценз" in str(fact.get("client_safe_text") or "").casefold()
    ]
    assert license_summary_facts, "Expected a client-safe generic license summary fact"
    assert any("есть лицензия" in str(fact.get("client_safe_text") or "").casefold() for fact in license_summary_facts)


def test_v3_all_fact_source_ids_exist_in_source_registry(kb_v3: KbReleaseV3) -> None:
    orphan_facts = [
        fact
        for fact in kb_v3.facts
        if not fact.get("source_id") or str(fact.get("source_id")) not in kb_v3.source_ids
    ]
    assert not orphan_facts, f"Found orphan source_id facts: {_fact_ids(orphan_facts[:20])}"

    claude_sources_without_hash = [
        source
        for source in kb_v3.sources
        if str(source.get("source_id") or "").startswith("claude_layer_v3:")
        and not (source.get("sha256") or source.get("source_sha256"))
    ]
    assert not claude_sources_without_hash, claude_sources_without_hash


def test_v3_q14_q15_closed_with_correct_scope(kb_v3: KbReleaseV3) -> None:
    q14_semester = _single_fact_for_amount(kb_v3.facts, amount=49000, brand="unpk", tokens=("offline", "5", "11"))
    q14_year = _single_fact_for_amount(kb_v3.facts, amount=82000, brand="unpk", tokens=("offline", "5", "11"))

    for fact in (q14_semester, q14_year):
        assert _is_true(fact.get("allowed_for_client_answer")), fact
        assert _is_true(fact.get("usable_for_precise_answer")), fact
        q14_scope = _fact_scope_blob(fact)
        assert _has_class_scope(q14_scope, first="5", last="11"), fact
        assert not _has_any(q14_scope, ("1-4", "1_4", "1 4", "1–4", "1—4")), fact

    q15_semester = _single_fact_for_amount(
        kb_v3.facts,
        amount=41800,
        brand="unpk",
        tokens=("online", "olympiad", "phystech"),
    )
    q15_year = _single_fact_for_amount(
        kb_v3.facts,
        amount=69900,
        brand="unpk",
        tokens=("online", "olympiad", "phystech"),
    )

    for fact in (q15_semester, q15_year):
        assert not _is_true(fact.get("allowed_for_client_answer")), fact
        assert not _is_true(fact.get("usable_for_precise_answer")), fact
        assert _is_true(fact.get("internal_only")) or "stale_previous_year_not_current" in _fact_blob(fact), fact
        q15_scope = _fact_scope_blob(fact)
        assert _has_any(q15_scope, ("superseded", "старый отдельный продукт")), fact
        assert _has_class_scope(q15_scope, first="9", last="11"), fact
        assert not _has_any(q15_scope, ("5-11", "5_11", "1-4", "1_4")), fact

    by_key = {str(fact.get("fact_key") or ""): fact for fact in kb_v3.facts if fact.get("brand") == "unpk"}
    online_regular_handoff = by_key["prices_regular_2026_27.online_5_11_class_regular.bot_behavior_when_asked"]
    handoff_text = str(
        online_regular_handoff.get("client_safe_text") or online_regular_handoff.get("fact_text") or ""
    )
    assert "онлайн-направление есть" in handoff_text
    assert "цену менеджер проверит" in handoff_text
    handoff_scope = _fact_scope_blob(online_regular_handoff)
    assert _has_class_scope(handoff_scope, first="5", last="11"), online_regular_handoff
    assert "online" in handoff_scope, online_regular_handoff

    unexpected_unpk_online_precise_prices = [
        fact
        for fact in kb_v3.facts
        if fact.get("brand") == "unpk"
        and _is_price_fact(fact)
        and "online" in _fact_blob(fact)
        and _has_money_amount(fact)
        and _is_true(fact.get("allowed_for_client_answer"))
        and _is_true(fact.get("usable_for_precise_answer"))
        and not str(fact.get("fact_key") or "").startswith("prices_regular_2026_27.online_5_11_class_regular.")
    ]
    assert not unexpected_unpk_online_precise_prices, (
        "UNPK online precise prices outside confirmed 2x/week regular online scope must stay manager-handoff only: "
        f"{_fact_ids(unexpected_unpk_online_precise_prices[:20])}"
    )


def test_v3_non_money_numbers_are_not_ruble_prices(kb_v3: KbReleaseV3) -> None:
    bad_small_price_facts = []
    bad_marker_facts = []
    for fact in kb_v3.facts:
        structured = _jsonish(fact.get("structured_value"))
        amount = structured.get("amount")
        if (
            fact.get("fact_type") == "price"
            and _is_true(fact.get("allowed_for_client_answer"))
            and isinstance(amount, (int, float))
            and amount < 3000
        ):
            bad_small_price_facts.append(fact)

        path_blob = f"{fact.get('fact_key') or ''} {_jsonish(fact.get('structured_value')).get('path') or ''}".casefold()
        if _is_true(fact.get("allowed_for_client_answer")) and any(
            marker in path_blob for marker in NON_MONEY_NUMERIC_PATH_MARKERS
        ):
            if fact.get("fact_type") == "price" or structured.get("currency") == "RUB":
                bad_marker_facts.append(fact)

    assert not bad_small_price_facts, _fact_ids(bad_small_price_facts[:30])
    assert not bad_marker_facts, _fact_ids(bad_marker_facts[:30])


def test_v3_range_facts_are_linked(kb_v3: KbReleaseV3) -> None:
    range_facts = [
        fact
        for fact in kb_v3.facts
        if _jsonish(fact.get("structured_value")).get("amount_min") is not None
        or _jsonish(fact.get("structured_value")).get("amount_max") is not None
    ]
    assert range_facts

    bad = []
    for fact in range_facts:
        structured = _jsonish(fact.get("structured_value"))
        if structured.get("do_not_use_as_current_price"):
            continue
        amount_min = structured.get("amount_min")
        amount_max = structured.get("amount_max")
        text_blob = _fact_blob(fact)
        if not isinstance(amount_min, (int, float)) or not isinstance(amount_max, (int, float)):
            bad.append(fact)
        elif amount_min > amount_max:
            bad.append(fact)
        elif "диапазон" not in text_blob and "от " not in text_blob:
            bad.append(fact)

    split_range_price_facts = [
        fact
        for fact in kb_v3.facts
        if (
            ".range.min" in str(fact.get("fact_key") or "")
            or ".range.max" in str(fact.get("fact_key") or "")
            or "_range_min" in str(fact.get("fact_id") or "")
            or "_range_max" in str(fact.get("fact_id") or "")
        )
        and not _jsonish(fact.get("structured_value")).get("do_not_use_as_current_price")
    ]

    assert not bad, _fact_ids(bad[:30])
    assert not split_range_price_facts, _fact_ids(split_range_price_facts[:30])


def test_v3_phystech_products_are_not_collapsed(kb_v3: KbReleaseV3) -> None:
    q15_facts = [
        fact
        for fact in kb_v3.facts
        if "online_olympiad_phystech_9_and_11" in str(fact.get("fact_key") or "")
    ]
    old_general_facts = [
        fact for fact in kb_v3.facts if str(fact.get("fact_key") or "").startswith("fiztech_olympiad.")
    ]
    assert q15_facts
    assert old_general_facts
    assert {fact.get("product") for fact in q15_facts} == {"online_olympiad_phystech_classes_9_11"}
    assert {fact.get("product") for fact in old_general_facts} == {"fiztech_olympiad_general"}


def test_v3_rc2a_client_safe_discount_and_olympiad_facts(kb_v3: KbReleaseV3) -> None:
    allowed = list(_allowed_client_facts(kb_v3.facts))
    by_key_brand = {
        (str(fact.get("brand") or ""), str(fact.get("fact_key") or "")): fact
        for fact in allowed
    }

    for brand in ("foton", "unpk"):
        fact = by_key_brand[(brand, "discounts.second_subject.offline.client_safe_text")]
        text = str(fact.get("client_safe_text") or fact.get("fact_text") or "")
        assert "Скидка на второй предмет очно" in text
        assert "20%" in text
        assert "не суммируются" in text
        assert fact.get("usable_for_precise_answer") is True

    olymp = by_key_brand[("unpk", "prices_regular_2026_27.online_olympiad_phystech_classes.client_safe_text")]
    olymp_text = str(olymp.get("client_safe_text") or olymp.get("fact_text") or "")
    assert "Олимпиадная подготовка онлайн" in olymp_text
    assert "обычных онлайн-курсов" in olymp_text
    assert "отдельного продукта нет" in olymp_text
    assert "41 800" not in olymp_text and "69 900" not in olymp_text
    assert olymp.get("usable_for_precise_answer") is True

    weekday = by_key_brand[
        ("unpk", "kb_v6_6_client_safe_facts_2026_06_08.annual_online_courses_math_physics_informatics_9_11_weekday_2026_27.client_safe_text")
    ]
    weekday_text = str(weekday.get("client_safe_text") or weekday.get("fact_text") or "")
    assert "9 и 11 классов" in weekday_text
    assert "будням" in weekday_text
    assert "41 800" in weekday_text and "69 900" in weekday_text
    weekday_applies_to = (_jsonish(weekday.get("structured_value")).get("applies_to") or {})
    assert weekday_applies_to.get("grades") == [9, 11]
    assert weekday_applies_to.get("formats") == ["online"]


def test_v3_refund_post_payment_is_client_safe_but_limited(kb_v3: KbReleaseV3) -> None:
    allowed = list(_allowed_client_facts(kb_v3.facts))
    by_key_brand = {
        (str(fact.get("brand") or ""), str(fact.get("fact_key") or "")): fact
        for fact in allowed
    }

    expected_text = (
        "По возврату средств после оплаты — здесь нужен расчёт от менеджера: он посмотрит, "
        "какая часть курса уже пройдена, и пришлёт точную сумму к возврату. Я уже передал "
        "ему ваш запрос — он свяжется с вами в рабочее время."
    )
    for brand in ("foton", "unpk"):
        fact = by_key_brand[
            (brand, "presentation_format_facts_2026_05_21.client_safe_facts.refund_post_payment.client_safe_text")
            if brand == "foton"
            else (brand, "tg_unpk_verified_2026_05_21.client_safe_facts.refund_post_payment.client_safe_text")
        ]
        text = str(fact.get("client_safe_text") or "")
        assert text == expected_text
        assert "все деньги" not in text.casefold()
        assert "полный возврат" not in text.casefold()
        assert fact.get("usable_for_precise_answer") is True


def test_v6_1_manifest_owns_v64_business_overrides() -> None:
    import yaml

    from scripts import build_kb_release_v3_from_claude_handoff as direct_builder

    manifest_path = (
        PROJECT_ROOT
        / "product_data"
        / "knowledge_base"
        / "kb_release_20260520_v6_3_team_answers_sources"
        / "release_manifest.yaml"
    )
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))

    assert "refund_post_payment" in manifest["client_safe_path_markers"]
    assert "refund_post_payment" not in direct_builder.CLIENT_SAFE_PATH_MARKERS

    manual_overrides = {
        item["fact_key"]: item for item in manifest["manual_decision_fact_overrides"]
    }
    q15 = manual_overrides["team_answers.q15.unpk_online_other_classes.manager_handoff"]
    assert q15["fact_type"] == "program"
    assert q15["route_policy"] == "bot_answer_self_for_pilot"
    assert "2 раза в неделю" in q15["fact_text"]

    structured_rules = manifest["structured_metadata_rules"]
    assert any(
        rule.get("fact_key_prefix") == "prices_regular_2026_27.online_5_11_class_regular."
        and (rule.get("applies_to") or {}).get("frequency") == "2 раза в неделю"
        for rule in structured_rules
    )
    assert any(
        rule.get("fact_key_prefix") == "prices_regular_2026_27.online_olympiad_phystech_classes."
        and (rule.get("applies_to") or {}).get("grades") == [9, 11]
        for rule in structured_rules
    )


def test_v3_confirmed_manager_only_candidates_move_to_client_safe(kb_v3: KbReleaseV3) -> None:
    allowed = list(_allowed_client_facts(kb_v3.facts))
    by_key_brand = {
        (str(fact.get("brand") or ""), str(fact.get("fact_key") or "")): fact
        for fact in allowed
    }

    installment = by_key_brand[("foton", "installment.client_confirmed_terms.client_safe_text")]
    installment_text = str(installment.get("client_safe_text") or "")
    assert "6, 10 или 12 месяцев" in installment_text
    assert "Долями" in installment_text
    assert "УНПК" not in installment_text
    assert installment.get("usable_for_precise_answer") is True

    q15_handoff = by_key_brand[("unpk", "team_answers.q15.unpk_online_other_classes.manager_handoff")]
    q15_text = str(q15_handoff.get("client_safe_text") or "")
    assert "Вне двух подтверждённых онлайн-тарифов УНПК" in q15_text
    assert "точные условия проверяет менеджер" in q15_text
    assert "41 800" not in q15_text and "69 900" not in q15_text


def test_v3_weekly_lessons_do_not_parse_academic_year_as_frequency(kb_v3: KbReleaseV3) -> None:
    allowed = list(_allowed_client_facts(kb_v3.facts))
    weekly = [
        fact
        for fact in allowed
        if str(fact.get("fact_key") or "") == "academic_year_2026_27.weekly_lessons"
    ]
    assert {fact.get("brand") for fact in weekly} == {"foton", "unpk"}
    for fact in weekly:
        text = " ".join(
            str(fact.get(field) or "")
            for field in ("fact_text", "client_safe_text", "manager_check_text", "manager_display_text")
        )
        structured = _jsonish(fact.get("structured_value"))
        assert "1 раз в неделю" in text
        assert "2 026 раз в неделю" not in text
        assert structured.get("weeks") == 1
        assert structured.get("count") != 2026
        assert not (structured.get("unit") == "lessons" and structured.get("count") == 2026)


def test_v3_quality_report_includes_general_integrity_gates(kb_v3: KbReleaseV3) -> None:
    checks = kb_v3.quality_report.get("checks") or {}
    assert checks.get("text_number_grounded") is True
    assert checks.get("field_ranges_ok") is True
    assert checks.get("weekly_frequency_is_plausible") is True
    details = kb_v3.quality_report.get("details") or {}
    assert details.get("text_number_grounding_findings") == []
    assert details.get("field_range_findings") == []


def test_v3_integrity_helpers_do_not_block_dates_classes_urls_or_multivalue_raw() -> None:
    from scripts import build_kb_release_v3_from_claude_handoff as builder

    safe_fact = {
        "fact_id": "safe",
        "fact_key": "safe.multivalue",
        "brand": "foton",
        "allowed_for_client_answer": True,
        "fact_text": "Фотон: в 2026/27 для 9 и 11 классов доступна рассрочка на 6, 10 или 12 месяцев; ссылка https://example.ru/course/2018411; телефон +7 999 123-45-67.",
        "client_safe_text": "Фотон: в 2026/27 для 9 и 11 классов доступна рассрочка на 6, 10 или 12 месяцев.",
        "structured_value": {"raw_value": "Рассрочка на 6, 10 или 12 месяцев для 9 и 11 классов в 2026/27."},
    }
    assert builder.text_number_grounding_findings_for_fact(safe_fact) == []


def test_v3_integrity_helpers_block_contradictory_business_numbers() -> None:
    from scripts import build_kb_release_v3_from_claude_handoff as builder

    fact = {
        "fact_id": "bad_price",
        "fact_key": "bad.price",
        "brand": "foton",
        "allowed_for_client_answer": True,
        "client_safe_text": "Фотон: занятие стоит 9 900 ₽.",
        "structured_value": {"amount": 6900, "currency": "RUB", "raw_value": "6900 ₽"},
    }
    findings = builder.text_number_grounding_findings_for_fact(fact)
    assert findings
    assert findings[0]["kind"] == "money"
    assert findings[0]["value"] == 9900


def test_v3_integrity_helpers_block_bad_field_ranges() -> None:
    from scripts import build_kb_release_v3_from_claude_handoff as builder

    fact = {
        "fact_id": "bad_ranges",
        "fact_key": "academic_year_2026_27.weekly_lessons",
        "structured_value": {"path": "academic_year_2026_27.weekly_lessons", "weeks": 2026, "percentage": 101, "amount": 0},
    }
    findings = builder.field_range_findings_for_fact(fact)
    reasons = {item["reason"] for item in findings}
    assert "expected_1_to_7" in reasons
    assert "expected_0_to_100" in reasons
    assert "money_expected_positive" in reasons


def test_v3_structured_value_does_not_parse_dates_as_business_values() -> None:
    from scripts import build_kb_release_v3_from_claude_handoff as builder

    discount_note = builder.build_structured_value(
        ("discounts", "multichild", "note"),
        "Поправка Дмитрия 2026-05-18: один ребёнок из многодетной семьи получает скидку 10%.",
        fact_type="discount",
    )
    assert discount_note.get("percentage") == 10

    installment_note = builder.build_structured_value(
        ("installment", "term_months", "internal_note"),
        "Старая широкая формулировка срока рассрочки не используется после подтверждения Дмитрия 2026-05-22.",
        fact_type="installment",
    )
    assert "months" not in installment_note


def test_v3_b2_separates_global_match_suspect_from_blocking() -> None:
    from scripts import build_kb_release_v3_from_claude_handoff as builder

    facts = [
        {
            "fact_id": "grounded",
            "fact_key": "discount.grounded",
            "brand": "foton",
            "allowed_for_client_answer": True,
            "client_safe_text": "Фотон: скидка 20%.",
            "structured_value": {"percentage": 20, "raw_value": "20%"},
        },
        {
            "fact_id": "suspect",
            "fact_key": "discount.summary",
            "brand": "foton",
            "allowed_for_client_answer": True,
            "client_safe_text": "Фотон: в кратком описании тоже указана скидка 20%.",
            "structured_value": {},
        },
        {
            "fact_id": "blocking",
            "fact_key": "discount.bad",
            "brand": "foton",
            "allowed_for_client_answer": True,
            "client_safe_text": "Фотон: неподтверждённая скидка 33%.",
            "structured_value": {},
        },
    ]
    raw_findings = [
        finding
        for fact in facts
        for finding in builder.text_number_grounding_findings_for_fact(fact)
    ]
    index = builder.grounded_number_index_for_facts(facts)
    by_key = {finding["fact_key"]: builder.same_brand_global_fact_matches(finding, index) for finding in raw_findings}

    assert by_key["discount.summary"] == ["discount.grounded"]
    assert by_key["discount.bad"] == []


def test_v3_client_facts_do_not_use_machine_slug_text(kb_v3: KbReleaseV3) -> None:
    bad = []
    technical_english_re = re.compile(
        r"\b(?:prices?|lesson|session|package|base|plus|one\s+block|one\s+subject|two\s+subjects|"
        r"after\s+20\d{2}|before\s+20\d{2}|moscow|dolgoprudny|location|start\s+date|"
        r"online\s+platform|free\s+morning\s+club|factultative)\b",
        re.I,
    )
    for fact in _allowed_client_facts(kb_v3.facts):
        text = str(fact.get("client_safe_text") or fact.get("fact_text") or "")
        text_without_handles = re.sub(r"@[A-Za-z0-9_]+", "", text)
        text_without_handles = re.sub(
            r"\b(?:https?://|www\.|[a-z0-9-]+\.(?:ru|com|org|net))/[A-Za-z0-9_./-]+",
            "",
            text_without_handles,
            flags=re.I,
        )
        if " / " in text or re.search(r"\b[a-z]+_[a-z0-9_]+\b", text_without_handles) or technical_english_re.search(text):
            bad.append(fact)
    assert not bad, _fact_ids(bad[:30])


def test_v3_pilot_client_facts_do_not_have_machine_short_tails(kb_v3: KbReleaseV3) -> None:
    bad = []
    for fact in _allowed_client_facts(kb_v3.facts):
        if fact.get("route_policy") != "bot_answer_self_for_pilot":
            continue
        text = str(fact.get("client_safe_text") or "")
        if re.search(r"—\s*([0-9]+(?:[.,][0-9]+)?%?|да|нет)\.$", text, re.I) and not fact.get(
            "bot_template_required"
        ):
            bad.append(fact)
    assert not bad, _fact_ids(bad[:30])


def test_v3_pilot_discounts_include_condition_in_client_text(kb_v3: KbReleaseV3) -> None:
    bad = []
    for fact in _allowed_client_facts(kb_v3.facts):
        if fact.get("route_policy") != "bot_answer_self_for_pilot" or fact.get("fact_type") != "discount":
            continue
        text = str(fact.get("client_safe_text") or "").casefold().replace("ё", "е")
        if not any(
            marker in text
            for marker in (
                "для ",
                "при ",
                "если",
                "после",
                "услов",
                "многодет",
                "сотрудник",
                "второй предмет",
                "помесяч",
                "оплатив",
                "постоянн",
                "участник",
                "друга",
            )
        ):
            bad.append(fact)
    assert not bad, _fact_ids(bad[:30])


def test_v3_client_facts_do_not_claim_discounts_stack(kb_v3: KbReleaseV3) -> None:
    bad = [
        fact
        for fact in _allowed_client_facts(kb_v3.facts)
        if re.search(r"(?<!не\s)скидки\s+суммируются", str(fact.get("client_safe_text") or "").casefold())
    ]
    assert not bad, _fact_ids(bad[:30])


def test_v3_client_text_does_not_expose_internal_approval_notes(kb_v3: KbReleaseV3) -> None:
    forbidden = ("q7 закрыт", "ответу бухгалтерии", "цифра 25% устарела", "dynamic needs check")
    bad = [
        fact
        for fact in _allowed_client_facts(kb_v3.facts)
        if any(marker in str(fact.get("client_safe_text") or "").casefold() for marker in forbidden)
    ]
    assert not bad, _fact_ids(bad[:30])


def test_v3_time_sensitive_allowed_facts_have_valid_until(kb_v3: KbReleaseV3) -> None:
    sensitive = {"price", "discount", "deadline", "program", "camp_lvsh", "camp_city", "intensive", "installment"}
    bad = [
        fact
        for fact in _allowed_client_facts(kb_v3.facts)
        if fact.get("fact_type") in sensitive
        and not (fact.get("valid_until") or _jsonish(fact.get("structured_value")).get("valid_until"))
    ]
    assert not bad, _fact_ids(bad[:30])


def test_v3_all_facts_have_refresh_date(kb_v3: KbReleaseV3) -> None:
    bad = [fact for fact in kb_v3.facts if not (fact.get("valid_until") or _jsonish(fact.get("structured_value")).get("valid_until"))]
    assert not bad, _fact_ids(bad[:30])


def test_v3_foton_client_facts_do_not_expose_unpk_telegram_variants(kb_v3: KbReleaseV3) -> None:
    forbidden = ("@unpk_mipt", "@unpkmfti", "@unpk mipt", "unpkmfti")
    bad = [
        fact
        for fact in _allowed_client_facts(kb_v3.facts)
        if fact.get("brand") == "foton"
        and any(marker in str(fact.get("client_safe_text") or "").casefold() for marker in forbidden)
    ]
    assert not bad, _fact_ids(bad[:30])


def test_v3_unpk_telegram_handle_keeps_underscore(kb_v3: KbReleaseV3) -> None:
    fact = next(fact for fact in kb_v3.facts if fact.get("fact_key") == "contacts_unpk.telegram")
    assert "@unpk_mipt" in str(fact.get("client_safe_text") or fact.get("manager_check_text") or "")


def test_v3_post_filter_has_discount_stacking_regex(kb_v3: KbReleaseV3) -> None:
    post_filter = kb_v3.snapshot.get("post_filter_registry") or kb_v3.snapshot.get("post_filter") or {}
    blob = json.dumps(post_filter, ensure_ascii=False)
    assert "скидки" in blob and "суммируются" in blob
    assert "regex_patterns" in post_filter.get("matcher_fields", [])
    assert "phrases" in post_filter.get("matcher_fields", [])
    assert "pattern_descriptions" in post_filter.get("human_only_fields", [])
    assert not set(post_filter.get("pattern_descriptions", [])) & set(post_filter.get("phrases", []))
    assert post_filter.get("regex_patterns_total", 0) >= 3


def test_v3_refer_a_friend_cashback_text_includes_condition(kb_v3: KbReleaseV3) -> None:
    bad = [
        fact
        for fact in _allowed_client_facts(kb_v3.facts)
        if (text := str(fact.get("client_safe_text") or "").casefold())
        and "приведи друга" in text
        and re.search(r"\b[0-9][0-9\s]{2,}\b", str(fact.get("client_safe_text") or ""))
        and not any(marker in text for marker in ("условие:", "после", "ранее не посещав"))
    ]
    assert not bad, _fact_ids(bad[:30])


def test_v3_price_and_discount_client_facts_require_templates(kb_v3: KbReleaseV3) -> None:
    bad = [
        fact
        for fact in _allowed_client_facts(kb_v3.facts)
        if fact.get("fact_type") in {"price", "discount"}
        and re.search(r"(?:\d[\d\s]{2,}\s*(?:₽|руб)?|\d{1,2}\s*%)", str(fact.get("client_safe_text") or ""))
        and not fact.get("bot_template_required")
    ]
    assert not bad, _fact_ids(bad[:30])


def test_v3_manager_display_text_has_no_client_blocked_debug_tags(kb_v3: KbReleaseV3) -> None:
    bad = [
        fact
        for fact in kb_v3.facts
        if "[client_blocked:" in str(fact.get("manager_display_text") or "")
    ]
    assert not bad, _fact_ids(bad[:30])


def test_v3_client_safe_text_has_no_known_machine_fragments(kb_v3: KbReleaseV3) -> None:
    fragments = (
        "рассрочка и оплата — т-банк.",
        "рассрочка и оплата — 1-2 минуты.",
        "материнский капитал — договор.",
        "онлайн-платформа, онлайн — да.",
        "скидка, правило —",
        "..",
    )
    bad = [
        fact
        for fact in _allowed_client_facts(kb_v3.facts)
        if any(fragment in str(fact.get("client_safe_text") or "").casefold() for fragment in fragments)
    ]
    assert not bad, _fact_ids(bad[:30])


def test_v3_short_machine_client_text_requires_template(kb_v3: KbReleaseV3) -> None:
    bad = []
    for fact in _allowed_client_facts(kb_v3.facts):
        text = str(fact.get("client_safe_text") or "")
        tail_match = re.search(r"—\s*([^—.]{1,42})\.$", text)
        if not tail_match:
            continue
        tail = tail_match.group(1).strip()
        short_label = bool(re.fullmatch(r"[А-Яа-яA-Za-z-]+(?:\s+[А-Яа-яA-Za-z-]+){0,2}", tail))
        machine_tail = bool(
            re.fullmatch(r"[0-9]+(?:[.,][0-9]+)?%?|[0-9]+[–-][0-9]+|да|нет", tail, re.I)
            or short_label
        )
        if machine_tail:
            if fact.get("fact_type") != "contact" and not fact.get("bot_template_required"):
                bad.append(fact)
    assert not bad, _fact_ids(bad[:30])


def test_v3_forbidden_brand_relationship_phrase_is_not_client_allowed(kb_v3: KbReleaseV3) -> None:
    bad = [
        fact
        for fact in _allowed_client_facts(kb_v3.facts)
        if "раньше сотрудничали" in str(fact.get("client_safe_text") or "").casefold()
    ]
    assert not bad, _fact_ids(bad)


def test_v3_safe_key_unification_removes_known_duplicate_sources(kb_v3: KbReleaseV3) -> None:
    fact_keys = {str(fact.get("fact_key") or "") for fact in kb_v3.facts}
    stacking_facts = [
        fact
        for fact in _allowed_client_facts(kb_v3.facts)
        if str(fact.get("fact_key") or "") in {"discounts.stacking_rule", "discounts.stacking_rule_text"}
    ]
    assert {fact.get("brand") for fact in stacking_facts} == {"foton", "unpk"}
    assert all("не суммируются" in str(fact.get("client_safe_text") or "").casefold() for fact in stacking_facts)
    assert "brand_rules.approved_brand_relationship_answer.foton" in fact_keys
    assert "brand_rules.approved_brand_relationship_answer.unpk" in fact_keys
    brand_link_responses = [
        fact
        for fact in _allowed_client_facts(kb_v3.facts)
        if str(fact.get("fact_key") or "") == "objection_responses.brand_link_question.approved_response"
    ]
    canonical_relationship_texts = {
        str(fact.get("client_safe_text") or "")
        for fact in _allowed_client_facts(kb_v3.facts)
        if str(fact.get("fact_key") or "").startswith("brand_rules.approved_brand_relationship_answer.")
    }
    assert {fact.get("brand") for fact in brand_link_responses} == {"foton", "unpk"}
    assert all(str(fact.get("client_safe_text") or "") in canonical_relationship_texts for fact in brand_link_responses)
    assert "objection_responses.too_expensive_course.3" in fact_keys


def test_v3_certificate_phrase_does_not_collect_unconfirmed_fields(kb_v3: KbReleaseV3) -> None:
    bad = [
        fact
        for fact in _allowed_client_facts(kb_v3.facts)
        if "theme_12_certificate" in str(fact.get("fact_key") or "")
        if "ФИО плательщика" in str(fact.get("client_safe_text") or "")
        or "ФИО ребёнка" in str(fact.get("client_safe_text") or "")
        or "за какой период" in str(fact.get("client_safe_text") or "")
    ]
    assert not bad, _fact_ids(bad)


def test_v3_approval_queue_questions_are_specific(kb_v3: KbReleaseV3) -> None:
    questions = [str(item.get("rop_question") or "") for item in kb_v3.approval_queue]
    assert len(set(questions)) >= 100
    assert not [
        item
        for item in kb_v3.approval_queue
        if item.get("priority") == "P0" and item.get("suggested_decision") == "keep_internal_only"
    ]
    too_generic = [
        question
        for question in questions
        if question in {
            "Можно ли использовать этот факт в ответе клиенту текущего бренда?",
            "Подтверждаете эту цену и область применения для бота?",
        }
    ]
    assert not too_generic


def test_v3_brand_scope_blocks_cross_brand_prices(kb_v3: KbReleaseV3) -> None:
    for fact in _allowed_client_facts(kb_v3.facts):
        client_text = str(fact.get("client_safe_text") or "")
        if fact.get("brand") == "foton":
            for marker in FOTON_CLIENT_FORBIDDEN_MARKERS:
                assert marker.casefold() not in client_text.casefold(), fact
        if fact.get("brand") == "unpk":
            for marker in UNPK_CLIENT_FORBIDDEN_MARKERS:
                assert marker.casefold() not in client_text.casefold(), fact

    foton_amounts = _allowed_price_amounts(kb_v3.facts, "foton")
    unpk_amounts = _allowed_price_amounts(kb_v3.facts, "unpk")

    assert foton_amounts
    assert unpk_amounts
    assert 82000 not in foton_amounts
    assert 41800 not in foton_amounts and 69900 not in foton_amounts
    assert 44600 not in unpk_amounts and 74500 not in unpk_amounts
    assert 29750 not in unpk_amounts and 47250 not in unpk_amounts


def test_v3_unpk_bank_installment_absence_is_client_safe_and_brand_clean(kb_v3: KbReleaseV3) -> None:
    matches = [
        fact
        for fact in _allowed_client_facts(kb_v3.facts)
        if fact.get("brand") == "unpk"
        and "payment_options.bank_installment.absent.client_safe_text" == str(fact.get("fact_key") or "")
    ]
    assert len(matches) == 1
    text = str(matches[0].get("client_safe_text") or "")
    lowered = text.casefold()
    assert "отдельной банковской рассрочки нет" in lowered
    assert "помесяч" in lowered
    assert "семестр" in lowered
    assert "год" in lowered
    assert "фотон" not in lowered
    assert "т-банк" not in lowered
    assert "долями" not in lowered
    assert "в унпк рассрочки нет" not in lowered


def test_v3_presale_refund_policy_is_client_safe_for_both_brands(kb_v3: KbReleaseV3) -> None:
    matches = [
        fact
        for fact in _allowed_client_facts(kb_v3.facts)
        if "refund_presale_policy.client_safe_text" in str(fact.get("fact_key") or "")
    ]
    assert {fact.get("brand") for fact in matches} == {"foton", "unpk"}
    for fact in matches:
        text = str(fact.get("client_safe_text") or "")
        lowered = text.casefold()
        assert "остаток неистраченных средств" in lowered
        assert "все деньги" not in lowered
        assert fact.get("allowed_for_client_answer") is True
        assert fact.get("usable_for_precise_answer") is True
        assert fact.get("route_policy") == "bot_answer_self_for_pilot"
        if fact.get("brand") == "foton":
            assert "унпк" not in lowered
        if fact.get("brand") == "unpk":
            assert "фотон" not in lowered


def test_v3_tax_deduction_facts_are_restored_and_client_safe(kb_v3: KbReleaseV3) -> None:
    tax_facts = [
        fact
        for fact in _allowed_client_facts(kb_v3.facts)
        if str(fact.get("fact_key") or "").startswith("tax_deduction.")
    ]
    assert tax_facts
    text = " ".join(str(fact.get("client_safe_text") or "") for fact in tax_facts)
    for marker in ("14 300", "110 000", "50 000", "6 500", "3 года", "3 месяц", "1 месяц", "10 рабочих дней"):
        assert marker in text


def test_v3_matkap_over_18_phrase_is_explicit_for_both_brands(kb_v3: KbReleaseV3) -> None:
    for brand in ("foton", "unpk"):
        fact = next(
            fact
            for fact in _allowed_client_facts(kb_v3.facts)
            if fact.get("brand") == brand and fact.get("fact_key") == "matkap.client_safe_text.when_age_over_18"
        )
        text = str(fact.get("client_safe_text") or "")
        assert "18 лет" in text
        assert "менеджер" in text.casefold()


def test_v3_foton_contacts_and_address_are_current(kb_v3: KbReleaseV3) -> None:
    foton_client_text = " ".join(
        str(fact.get("client_safe_text") or "")
        for fact in _allowed_client_facts(kb_v3.facts)
        if fact.get("brand") == "foton"
    )
    assert "vk.ru/foton_edu" in foton_client_text
    assert "@cdpoFoton" in foton_client_text
    assert "Сретенка" not in foton_client_text
    assert "Скорняжный" not in foton_client_text

    blocked_terms = _recursive_text(
        ((kb_v3.brand_rules.get("forbidden_client_mentions") or {}).get("when_active_brand_is_foton") or {}).get(
            "blocked_terms",
            [],
        )
    )
    assert "Сретенка" in blocked_terms
    assert "Сретенка, 20" in blocked_terms


def test_v3_future_and_expired_prices_are_not_client_safe(kb_v3: KbReleaseV3) -> None:
    foton_client_price_text = " ".join(
        str(fact.get("client_safe_text") or "")
        for fact in _allowed_client_facts(kb_v3.facts)
        if fact.get("brand") == "foton" and fact.get("fact_type") == "price"
    )
    for marker in ("16 900", "27 720", "до 07.04.2026", "после 01.07.2026", "после 01.08.2026"):
        assert marker not in foton_client_price_text

    for fact in _allowed_client_facts(kb_v3.facts):
        if fact.get("fact_type") == "price":
            text = str(fact.get("client_safe_text") or "")
            assert not re.search(r"до 0?1\.0[78]\.2026", text), fact


def test_v3_confirmed_social_proof_is_client_safe(kb_v3: KbReleaseV3) -> None:
    unpk_social = [
        fact
        for fact in _allowed_client_facts(kb_v3.facts)
        if fact.get("brand") == "unpk" and fact.get("fact_key") == "results_social_proof.total_alumni"
    ]
    foton_social = [
        fact
        for fact in _allowed_client_facts(kb_v3.facts)
        if fact.get("brand") == "foton" and fact.get("fact_key") == "results_social_proof.industry_rating_2025"
    ]
    assert unpk_social and "100 000" in str(unpk_social[0].get("client_safe_text") or "")
    assert foton_social and "Лидер отрасли" in str(foton_social[0].get("client_safe_text") or "")


def test_v3_offline_group_size_is_client_safe_for_both_brands(kb_v3: KbReleaseV3) -> None:
    for brand in ("foton", "unpk"):
        fact = next(
            fact
            for fact in _allowed_client_facts(kb_v3.facts)
            if fact.get("brand") == brand and fact.get("fact_key") == "online_platform.offline_group_size"
        )
        assert "6-12 человек" in str(fact.get("client_safe_text") or "")


def test_v3_matkap_required_docs_are_manager_only(kb_v3: KbReleaseV3) -> None:
    client_matkap_docs = [
        fact
        for fact in _allowed_client_facts(kb_v3.facts)
        if str(fact.get("fact_key") or "").startswith("matkap.required_docs")
    ]
    assert not client_matkap_docs

    manager_docs = [
        fact
        for fact in kb_v3.facts
        if str(fact.get("fact_key") or "").startswith("matkap.required_docs")
    ]
    assert manager_docs
    assert all(fact.get("internal_only") for fact in manager_docs)


def test_v3_bot_policy_uses_latest_dmitry_decisions(kb_v3: KbReleaseV3) -> None:
    installment = (((kb_v3.bot_policy.get("theme_routes") or {}).get("installment") or {}).get("unpk_specific") or {})
    fallback = str(installment.get("fallback_phrase") or "")
    assert "помесячно" in fallback.casefold()
    assert "10%" in fallback
    assert "14%" in fallback

    complaint = ((kb_v3.bot_policy.get("theme_routes") or {}).get("complaint") or {})
    p0_phrase = str(complaint.get("bot_phrase_p0") or "")
    assert "передам" in p0_phrase.casefold()
    assert "ответственному" in p0_phrase.casefold()
    assert "зафиксировано" not in p0_phrase.casefold()
    assert "автоматический" not in p0_phrase.casefold()


def test_v3_approval_queue_contains_atomic_business_items(kb_v3: KbReleaseV3) -> None:
    rows = kb_v3.approval_queue
    assert rows, "approval_queue_for_rop_v3.csv is empty"

    fieldnames = set(rows[0].keys())
    missing_columns = REQUIRED_APPROVAL_QUEUE_COLUMNS - fieldnames
    assert not missing_columns, f"approval queue misses required columns: {sorted(missing_columns)}"

    fact_ids = {str(fact.get("fact_id")) for fact in kb_v3.facts if fact.get("fact_id")}
    item_types = {_normalize_item_type(str(row.get("item_type") or "").strip()) for row in rows}
    missing_types = EXPECTED_APPROVAL_ITEM_TYPES - item_types
    assert not missing_types, f"approval queue misses business item types: {sorted(missing_types)}"

    bad_fact_refs = [
        row
        for row in rows
        if str(row.get("fact_id_ref") or "").strip()
        and str(row.get("fact_id_ref") or "").strip() not in fact_ids
    ]
    bad_source_refs = [
        row
        for row in rows
        if str(row.get("source_id") or "").strip()
        and str(row.get("source_id") or "").strip() not in kb_v3.source_ids
    ]
    assert not bad_fact_refs, f"approval queue has unknown fact_id_ref: {bad_fact_refs[:10]}"
    assert not bad_source_refs, f"approval queue has unknown source_id: {bad_source_refs[:10]}"

    if len(rows) < 400:
        assert kb_v3.quality_report.get("quality_passed") is False
        quality_blob = _recursive_text(kb_v3.quality_report).casefold()
        assert "approval" in quality_blob or "400" in quality_blob or "очеред" in quality_blob


def _load_json_required(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        pytest.fail(f"Missing required KB v3 artifact: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        pytest.fail(f"Expected mapping JSON artifact at {path}, got {type(payload)!r}")
    return payload


def _load_json_or_empty(path: Path) -> Any:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv_required(path: Path) -> list[Mapping[str, str]]:
    if not path.exists():
        pytest.fail(f"Missing required KB v3 artifact: {path}")
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _load_facts(root: Path, snapshot: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    jsonl_path = root / "facts_registry.jsonl"
    if jsonl_path.exists():
        facts = [
            json.loads(line)
            for line in jsonl_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return [fact for fact in facts if isinstance(fact, Mapping)]
    for key in ("facts", "facts_registry", "fact_records"):
        value = snapshot.get(key)
        if isinstance(value, list):
            return [fact for fact in value if isinstance(fact, Mapping)]
    csv_path = root / "facts_registry.csv"
    if csv_path.exists():
        return list(_load_csv_required(csv_path))
    pytest.fail(f"Missing facts registry in {root}")


def _load_sources(root: Path, snapshot: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    source_json = _load_json_or_empty(root / "source_registry.json")
    if isinstance(source_json, Mapping) and isinstance(source_json.get("items"), list):
        return [source for source in source_json["items"] if isinstance(source, Mapping)]
    if isinstance(source_json, list):
        return [source for source in source_json if isinstance(source, Mapping)]
    if isinstance(snapshot.get("sources"), list):
        return [source for source in snapshot["sources"] if isinstance(source, Mapping)]
    csv_path = root / "source_registry.csv"
    if csv_path.exists():
        return list(_load_csv_required(csv_path))
    pytest.fail(f"Missing source registry in {root}")


def _load_mapping_artifact(root: Path, stem: str) -> Mapping[str, Any]:
    json_payload = _load_json_or_empty(root / f"{stem}.json")
    if isinstance(json_payload, Mapping) and json_payload:
        return json_payload
    snapshot = _load_json_or_empty(root / "kb_release_v3_snapshot.json")
    if isinstance(snapshot, Mapping) and isinstance(snapshot.get(stem), Mapping):
        return snapshot[stem]
    yaml_path = root / f"{stem}.yaml"
    if yaml_path.exists():
        try:
            import yaml
        except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
            pytest.fail(f"PyYAML is required to read {yaml_path}: {exc}")
        payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        if isinstance(payload, Mapping):
            return payload
    return {}


def _facts_for_amount(
    facts: Iterable[Mapping[str, Any]],
    *,
    amount: int,
    brand: str,
    tokens: Iterable[str],
) -> list[Mapping[str, Any]]:
    return [
        fact
        for fact in facts
        if fact.get("brand") == brand
        and _has_amount(fact, amount)
        and all(_token_present(_fact_blob(fact), token) for token in tokens)
    ]


def _single_fact_for_amount(
    facts: Iterable[Mapping[str, Any]],
    *,
    amount: int,
    brand: str,
    tokens: Iterable[str],
) -> Mapping[str, Any]:
    matching = _facts_for_amount(facts, amount=amount, brand=brand, tokens=tokens)
    assert matching, f"Missing fact for amount={amount}, brand={brand}, tokens={tuple(tokens)}"
    return matching[0]


def _has_amount(fact: Mapping[str, Any], amount: int) -> bool:
    normalized = re.sub(r"\D+", "", _fact_blob(fact))
    return str(amount) in normalized


def _jsonish(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            loaded = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return dict(loaded) if isinstance(loaded, Mapping) else {}
    return {}


def _has_money_amount(fact: Mapping[str, Any]) -> bool:
    blob = _lower_blob(
        fact,
        fields=("fact_text", "client_safe_text", "manager_check_text", "structured_value", "structured_values"),
    )
    for number in re.findall(r"\d+", blob):
        value = int(number)
        if value >= 3000 and value not in {2026, 2027}:
            return True
    return False


def _token_present(blob: str, token: str) -> bool:
    folded = blob.casefold()
    aliases = {
        "offline": ("offline", "очно", "очная", "очный"),
        "online": ("online", "онлайн"),
        "olympiad": ("olympiad", "олимпиад"),
        "phystech": ("phystech", "физтех"),
        "fiztech": ("fiztech", "физтех"),
        "lvsh": ("lvsh", "лвш", "летняя выездная"),
        "mendeleevo": ("mendeleevo", "менделеево"),
        "individual": ("individual", "индивидуаль"),
        "lesson": ("lesson", "занят"),
        "intensive": ("intensive", "интенсив"),
        "oge": ("oge", "огэ"),
        "ege": ("ege", "егэ"),
        "preschool": ("preschool", "дошколь"),
        "patsayeva": ("patsayeva", "пацаева"),
    }
    candidates = aliases.get(token, (token,))
    return any(candidate.casefold() in folded for candidate in candidates)


def _allowed_client_facts(facts: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [
        fact
        for fact in facts
        if _is_true(fact.get("allowed_for_client_answer"))
        and str(fact.get("client_safe_text") or "").strip()
    ]


def _is_price_fact(fact: Mapping[str, Any]) -> bool:
    fact_type = str(fact.get("fact_type") or "").casefold()
    fact_types = _recursive_text(fact.get("fact_types", "")).casefold()
    return fact_type == "price" or "price" in fact_types or "стоим" in _fact_blob(fact) or "цен" in _fact_blob(fact)


def _allowed_price_text(facts: Iterable[Mapping[str, Any]], brand: str) -> str:
    text = " ".join(
        str(fact.get("client_safe_text") or "")
        for fact in facts
        if fact.get("brand") == brand
        and _is_true(fact.get("allowed_for_client_answer"))
        and _is_price_fact(fact)
    )
    return re.sub(r"\D+", "", text)


def _allowed_price_amounts(facts: Iterable[Mapping[str, Any]], brand: str) -> set[int]:
    amounts: set[int] = set()
    for fact in facts:
        if (
            fact.get("brand") == brand
            and _is_true(fact.get("allowed_for_client_answer"))
            and _is_price_fact(fact)
        ):
            structured = _jsonish(fact.get("structured_value"))
            for key in ("amount", "amount_min", "amount_max"):
                value = structured.get(key)
                if isinstance(value, (int, float)):
                    amounts.add(int(value))
    return amounts


def _post_filter_terms(kb_v3: KbReleaseV3) -> str:
    policy_payload: dict[str, Any] = {
        "brand_rules": kb_v3.brand_rules or kb_v3.snapshot.get("brand_rules", {}),
        "bot_policy": kb_v3.bot_policy or kb_v3.snapshot.get("bot_policy", {}),
        "safety": kb_v3.snapshot.get("safety", {}),
        "post_filter": kb_v3.snapshot.get("post_filter", {}),
        "post_filter_terms": kb_v3.snapshot.get("post_filter_terms", {}),
        "forbidden_to_say_registry": kb_v3.snapshot.get("forbidden_to_say_registry", {}),
    }
    return _recursive_text(policy_payload).casefold()


def _structured_value_text(fact: Mapping[str, Any]) -> str:
    for key in ("structured_value", "structured_values", "value", "amount"):
        value = fact.get(key)
        if value not in (None, "", {}, []):
            return _recursive_text(value)
    return ""


def _fact_scope_blob(fact: Mapping[str, Any]) -> str:
    fields = (
        "fact_id",
        "fact_key",
        "product",
        "title",
        "fact_text",
        "client_safe_text",
        "manager_check_text",
        "structured_value",
        "structured_values",
        "route_policy",
        "linked_open_question",
    )
    return " ".join(_recursive_text(fact.get(field, "")) for field in fields).casefold()


def _fact_blob(fact: Mapping[str, Any]) -> str:
    return _lower_blob(
        fact,
        fields=(
            "fact_id",
            "fact_key",
            "fact_type",
            "fact_types",
            "title",
            "fact_text",
            "client_safe_text",
            "manager_check_text",
            "product",
            "structured_value",
            "structured_values",
            "route_policy",
            "linked_open_question",
        ),
    )


def _lower_blob(fact: Mapping[str, Any], *, fields: Iterable[str]) -> str:
    return " ".join(_recursive_text(fact.get(field, "")) for field in fields).casefold()


def _recursive_text(value: Any) -> str:
    if isinstance(value, Mapping):
        return " ".join(f"{key} {_recursive_text(item)}" for key, item in value.items())
    if isinstance(value, (list, tuple, set)):
        return " ".join(_recursive_text(item) for item in value)
    return str(value)


def _has_any(haystack: str, needles: Iterable[str]) -> bool:
    folded = haystack.casefold()
    return any(needle.casefold() in folded for needle in needles)


def _has_class_scope(scope: str, *, first: str, last: str) -> bool:
    if _has_any(scope, (f"{first}-{last}", f"{first}_{last}", f"{first}–{last}", f"{first}—{last}")):
        return True
    tokens = re.findall(r"\d+", scope)
    return first in tokens and last in tokens


def _normalize_item_type(value: str) -> str:
    normalized = value.casefold().replace("-", "_")
    aliases = {
        "promo_code": "promocode",
        "promo_codes": "promocode",
        "promocode": "promocode",
        "promocodes": "promocode",
        "tax_deduction": "tax",
        "matkapital": "matkap",
        "camp_lvsh": "lvsh",
        "lvsh_mendeleevo": "lvsh",
    }
    return aliases.get(normalized, normalized)


def _is_true(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, str):
        return value.strip().casefold() in {"true", "1", "yes", "y"}
    return False


def _fact_ids(facts: Iterable[Mapping[str, Any]]) -> list[str]:
    return [str(fact.get("fact_id") or fact.get("fact_key") or fact) for fact in facts]
