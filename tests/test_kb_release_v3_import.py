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
DEFAULT_RELEASE_DIR = PROJECT_ROOT / "product_data" / "knowledge_base" / "kb_release_20260518_v3"

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
    {"amount": 75000, "brand": "foton", "tokens": ("lvsh", "mendeleevo")},
    {"amount": 3900, "brand": "foton", "tokens": ("individual", "lesson")},
    {"amount": 6900, "brand": "foton", "tokens": ("individual", "lesson")},
    {"amount": 23000, "brand": "foton", "tokens": ("individual", "lesson")},
    {"amount": 16900, "brand": "foton", "tokens": ("intensive", "oge")},
    {"amount": 27720, "brand": "foton", "tokens": ("intensive", "oge")},
    {"amount": 49000, "brand": "unpk", "tokens": ("offline", "5", "11")},
    {"amount": 82000, "brand": "unpk", "tokens": ("offline", "5", "11")},
    {"amount": 41800, "brand": "unpk", "tokens": ("online", "olympiad", "phystech")},
    {"amount": 69900, "brand": "unpk", "tokens": ("online", "olympiad", "phystech")},
    {"amount": 120000, "brand": "unpk", "tokens": ("lvsh", "mendeleevo")},
    {"amount": 89900, "brand": "unpk", "tokens": ("lvsh", "mendeleevo")},
    {"amount": 83800, "brand": "unpk", "tokens": ("lvsh", "mendeleevo")},
    {"amount": 33000, "brand": "unpk", "tokens": ("fiztech", "olympiad")},
    {"amount": 50000, "brand": "unpk", "tokens": ("fiztech", "olympiad")},
    {"amount": 11900, "brand": "unpk", "tokens": ("preschool", "patsayeva")},
    {"amount": 56500, "brand": "unpk", "tokens": ("preschool", "patsayeva")},
    {"amount": 94000, "brand": "unpk", "tokens": ("preschool", "patsayeva")},
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
        assert _is_true(fact.get("allowed_for_client_answer")), fact
        assert _is_true(fact.get("usable_for_precise_answer")), fact
        q15_scope = _fact_scope_blob(fact)
        assert _has_any(q15_scope, ("физтех", "phystech")), fact
        assert _has_any(q15_scope, ("олимпиад", "olympiad")), fact
        assert _has_class_scope(q15_scope, first="9", last="11"), fact
        assert not _has_any(q15_scope, ("5-11", "5_11", "1-4", "1_4")), fact

    other_unpk_online_precise_prices = [
        fact
        for fact in kb_v3.facts
        if fact.get("brand") == "unpk"
        and _is_price_fact(fact)
        and "online" in _fact_blob(fact)
        and _has_money_amount(fact)
        and not _has_amount(fact, 41800)
        and not _has_amount(fact, 69900)
        and _is_true(fact.get("allowed_for_client_answer"))
        and _is_true(fact.get("usable_for_precise_answer"))
    ]
    assert not other_unpk_online_precise_prices, (
        "UNPK online precise prices outside q15 must stay manager-handoff only: "
        f"{_fact_ids(other_unpk_online_precise_prices[:20])}"
    )


def test_v3_brand_scope_blocks_cross_brand_prices(kb_v3: KbReleaseV3) -> None:
    for fact in _allowed_client_facts(kb_v3.facts):
        client_text = str(fact.get("client_safe_text") or "")
        if fact.get("brand") == "foton":
            for marker in FOTON_CLIENT_FORBIDDEN_MARKERS:
                assert marker.casefold() not in client_text.casefold(), fact
        if fact.get("brand") == "unpk":
            for marker in UNPK_CLIENT_FORBIDDEN_MARKERS:
                assert marker.casefold() not in client_text.casefold(), fact

    foton_prices = _allowed_price_text(kb_v3.facts, "foton")
    unpk_prices = _allowed_price_text(kb_v3.facts, "unpk")

    assert foton_prices
    assert unpk_prices
    assert "82000" not in foton_prices
    assert "41800" not in foton_prices and "69900" not in foton_prices
    assert "44600" not in unpk_prices and "74500" not in unpk_prices
    assert "29750" not in unpk_prices and "47250" not in unpk_prices


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
    if isinstance(json_payload, Mapping):
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
