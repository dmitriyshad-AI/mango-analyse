from __future__ import annotations

from scripts.build_kb_release_v6_1_team_answers import gold_answers_v3_payload, patch_foton_installment_client_terms


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


def test_gold_answers_v3_forbids_foton_old_installment_term() -> None:
    payload = gold_answers_v3_payload()
    foton_installment = payload["topics"]["installment"]["foton"]

    assert "6, 10 или 12 месяцев" in foton_installment["gold_answer_example"]
    assert "Долями" in foton_installment["gold_answer_example"]
    assert "до 36 месяцев" in foton_installment["must_not_include"]


def test_patch_foton_installment_replaces_client_facing_36_months() -> None:
    facts = {
        "installment": {
            "products": {"regular": {"term_months": "3-36"}},
            "client_safe_text": {"when_asked": "Рассрочка до 36 месяцев."},
        }
    }

    patch_foton_installment_client_terms(facts)

    installment = facts["installment"]
    assert installment["products"]["regular"]["term_months"] == "6, 10 или 12"
    assert "6, 10 или 12 месяцев" in installment["client_safe_text"]["when_asked"]
    assert "Долями" in installment["client_safe_text"]["when_asked"]
    assert "до 36 месяцев" not in installment["client_safe_text"]["when_asked"]
    assert "forbidden_client_claims" not in installment
