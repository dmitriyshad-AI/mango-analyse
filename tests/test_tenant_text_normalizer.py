import re

from mango_mvp.quality.tenant_text_normalizer import (
    format_objection_list,
    format_product_list,
    normalize_manager_text,
    normalize_objection_label,
)


def test_normalizes_unpk_mfti_brand_aliases() -> None:
    text = "Менеджер сказал, что занятия проходят в МПК МФТИ, а клиент переспросил про НПК МФТИ, ОМПК МФТИ, ВНПК МФТИ и МНПК МФТИ."

    normalized = normalize_manager_text(text)

    assert "МПК МФТИ" not in normalized
    assert re.search(r"(?<!У)НПК МФТИ", normalized) is None
    assert "ОМПК МФТИ" not in normalized
    assert "ВНПК МФТИ" not in normalized
    assert "МНПК МФТИ" not in normalized
    assert normalized.count("УНПК МФТИ") == 5


def test_normalizes_summer_night_school_asr_artifact() -> None:
    text = "Клиент интересовался летними ночными школами и летней ночной школой."

    normalized = normalize_manager_text(text)

    assert "ночн" not in normalized.casefold()
    assert "летние очные школы" in normalized
    assert "летняя очная школа" in normalized


def test_product_list_collapses_counts_and_synonyms() -> None:
    value = "летний лагерь (8 касаний) | летняя очная школа (1 касаний) | летняя школа (1 касаний) | индивидуальные занятия (1 касаний)"

    assert format_product_list(value) == "летний лагерь | летняя очная школа | индивидуальные занятия"


def test_objection_dedupe_collapses_family_contract_variants() -> None:
    assert normalize_objection_label("нужно, чтобы муж прочитал договор") == "нужно согласовать договор с мужем"
    assert normalize_objection_label("не обсудили с мужем") == "нужно согласовать договор с мужем"
    assert format_objection_list("нужно, чтобы муж прочитал договор | не обсудили с мужем | нужен июньский заезд") == (
        "нужно согласовать договор с мужем | нужен июньский заезд"
    )
