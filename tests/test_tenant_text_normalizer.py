import re
import json
from pathlib import Path

from mango_mvp.quality.tenant_text_normalizer import (
    DETECTOR_KNOWN_BRAND_VARIANTS,
    detect_product_list_artifacts,
    detect_residual_manager_text_artifacts,
    format_objection_list,
    format_product_list,
    normalize_manager_text,
    normalize_objection_label,
)


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "tenant_text_normalizer_frozen_corpus.jsonl"


def test_normalizes_unpk_mfti_brand_aliases() -> None:
    text = "Менеджер сказал, что занятия проходят в МПК МФТИ, а клиент переспросил про НПК МФТИ, ОМПК МФТИ, ВНПК МФТИ и МНПК МФТИ."

    normalized = normalize_manager_text(text)

    assert "МПК МФТИ" not in normalized
    assert re.search(r"(?<!У)НПК МФТИ", normalized) is None
    assert "ОМПК МФТИ" not in normalized
    assert "ВНПК МФТИ" not in normalized
    assert "МНПК МФТИ" not in normalized
    assert normalized.count("УНПК МФТИ") == 5


def test_normalizes_mfti_tail_variants() -> None:
    for value in ("УНПК МФТИШ", "УНПК МФТИК", "УНПК МФТИЙ", "УНПК МФТИВ", "УНПК МФТИНГ"):
        normalized = normalize_manager_text(f"{value} подготовка")

        assert normalized == "УНПК МФТИ подготовка"
        assert not detect_residual_manager_text_artifacts(normalized)


def test_detector_flags_known_brand_variants_with_real_artifact_fields() -> None:
    for variant in DETECTOR_KNOWN_BRAND_VARIANTS:
        findings = detect_residual_manager_text_artifacts(f"Клиент спросил про {variant} подготовка")

        assert findings, variant
        assert findings[0].class_id == "known_brand_variant_residual"
        assert findings[0].matched_text == variant
        assert findings[0].reason


def test_detector_uses_word_boundaries_for_known_variants() -> None:
    assert not detect_residual_manager_text_artifacts("УНПК МФТИ подготовка")
    assert not detect_residual_manager_text_artifacts("СверхУНПК МФТИШовый маркер не является отдельным брендом")


def test_detector_does_not_flag_canonical_unpk_mfti() -> None:
    assert not detect_residual_manager_text_artifacts("УНПК МФТИ подготовка")


def test_normalizes_summer_night_school_asr_artifact() -> None:
    text = "Клиент интересовался летними ночными школами и летней ночной школой."

    normalized = normalize_manager_text(text)

    assert "ночн" not in normalized.casefold()
    assert "летними очными школами" in normalized
    assert "летней очной школой" in normalized


def test_product_list_collapses_counts_and_synonyms() -> None:
    value = "летний лагерь (8 касаний) | летняя очная школа (1 касаний) | летняя школа (1 касаний) | индивидуальные занятия (1 касаний)"

    assert format_product_list(value) == "летний лагерь | летняя очная школа | индивидуальные занятия"


def test_objection_dedupe_collapses_family_contract_variants() -> None:
    assert normalize_objection_label("нужно, чтобы муж прочитал договор") == "нужно согласовать договор с мужем"
    assert normalize_objection_label("не обсудили с мужем") == "нужно согласовать договор с мужем"
    assert format_objection_list("нужно, чтобы муж прочитал договор | не обсудили с мужем | нужен июньский заезд") == (
        "нужно согласовать договор с мужем | нужен июньский заезд"
    )


def test_frozen_corpus_cases_pass() -> None:
    for line in FIXTURE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        normalized_text = normalize_manager_text(case["input"])
        normalized_products = format_product_list(case["input"])
        normalized_objections = format_objection_list(case["input"])
        if case.get("expected_text"):
            assert normalized_text == case["expected_text"], case["case_id"]
            assert not detect_residual_manager_text_artifacts(normalized_text), case["case_id"]
            for forbidden in case.get("not_contains", []):
                assert forbidden.casefold() not in normalized_text.casefold(), case["case_id"]
            for forbidden_pattern in case.get("not_regex", []):
                assert re.search(forbidden_pattern, normalized_text, flags=re.IGNORECASE) is None, case["case_id"]
        if case.get("expected_products"):
            assert normalized_products == case["expected_products"], case["case_id"]
            assert not detect_product_list_artifacts(normalized_products), case["case_id"]
            for forbidden in case.get("not_contains", []):
                assert forbidden.casefold() not in normalized_products.casefold(), case["case_id"]
            for forbidden_pattern in case.get("not_regex", []):
                assert re.search(forbidden_pattern, normalized_products, flags=re.IGNORECASE) is None, case["case_id"]
        if case.get("expected_objections"):
            assert normalized_objections == case["expected_objections"], case["case_id"]
            for forbidden in case.get("not_contains", []):
                assert forbidden.casefold() not in normalized_objections.casefold(), case["case_id"]
            for forbidden_pattern in case.get("not_regex", []):
                assert re.search(forbidden_pattern, normalized_objections, flags=re.IGNORECASE) is None, case["case_id"]
