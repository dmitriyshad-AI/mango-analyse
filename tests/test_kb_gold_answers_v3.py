from __future__ import annotations

from scripts.build_kb_release_v6_1_team_answers import DEFAULT_SOURCE_OUT, gold_answers_v3_payload, load_yaml


def test_gold_answers_v3_contains_confirmed_core_rules() -> None:
    payload = gold_answers_v3_payload()

    assert payload["status"] == "approved_by_dmitry_for_bot_quality"
    assert payload["use_as"] == "answer_quality_tone_and_confirmed_business_rules_not_raw_script"
    confirmed = payload["confirmed_rules"]
    assert "6, 10 или 12 месяцев" in confirmed["foton_installment"]
    assert "Долями" in confirmed["foton_installment"]
    assert "14% за год" in confirmed["unpk_installment"]
    assert "Верхняя Красносельская, 30" in confirmed["foton_moscow_address"]
    assert "Сретенка, 20" in confirmed["unpk_moscow_regular"]
    assert "уточнять класс ребёнка" in confirmed["camp_question_key"]
    assert "цифровой помощник активного бренда" in " ".join(payload["global_rules"])
    assert "handoff-флаг" in " ".join(payload["global_rules"])


def test_gold_answers_v3_forbids_foton_old_installment_term() -> None:
    payload = gold_answers_v3_payload()
    foton_installment = payload["topics"]["installment"]["foton"]

    assert "6, 10 или 12 месяцев" in foton_installment["gold_answer_example"]
    assert "Долями" in foton_installment["gold_answer_example"]
    assert "до 36 месяцев" in foton_installment["must_not_include"]


def test_gold_answers_v3_contains_identity_policy_c_examples() -> None:
    payload = gold_answers_v3_payload()
    identity = payload["topics"]["identity"]

    foton = identity["foton"]
    assert "цифровой помощник Фотона" in foton["gold_answer_example"]
    assert "не живой оператор" in foton["gold_answer_example"]
    assert "GPT" in foton["must_not_include"]
    assert "я человек" in foton["must_not_include"]

    unpk = identity["unpk"]
    assert "цифровой помощник УНПК МФТИ" in unpk["gold_answer_example"]
    assert "не живой оператор" in unpk["gold_answer_example"]
    assert "Фотон" in unpk["must_not_include"]


def test_foton_installment_source_yaml_contains_confirmed_client_terms() -> None:
    facts = load_yaml(DEFAULT_SOURCE_OUT / "facts" / "facts_for_bot_FOTON.yaml")
    client_text = facts["installment"]["client_safe_text"]["when_asked"]

    assert "6, 10 или 12 месяцев" in client_text
    assert "Долями" in client_text
    assert "до 36 месяцев" not in client_text


def test_foton_online_year_source_omits_unconfirmed_upper_bound() -> None:
    facts = load_yaml(DEFAULT_SOURCE_OUT / "facts" / "facts_for_bot_FOTON.yaml")
    online_price = facts["prices_regular_2026_27"]["online_5_11_class"]["before_2026_08_01"]

    assert online_price["year"] == 47250
    assert "year_range" not in online_price
    assert "52500" not in str(online_price)
