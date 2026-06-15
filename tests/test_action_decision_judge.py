import json
from pathlib import Path

import pytest

from mango_mvp.channels import action_decision_judge as judge
from scripts.calibrate_action_decision_judge import load_gold_rows


GOLD_PATH = Path("product_data/telegram_dynamic_test_sets/action_decision_judge_gold_20260614.json")


def test_normalize_action_maps_action_spellings() -> None:
    assert judge.normalize_action("send materials") == "send_materials"
    assert judge.normalize_action("not_in_dictionary") == "unknown"


def test_payment_reward_requires_action_preconditions_facts_and_text() -> None:
    persona = {
        "brand": "unpk",
        "expected_action": {"action": "send_payment_link", "manual_label": True},
        "deal_card": {
            "brand": "unpk",
            "preconditions": {
                "product_selected": True,
                "price_confirmed": True,
                "client_ready_to_pay": True,
            },
        },
    }
    turn = {
        "client_message": "Готовы оплатить курс.",
        "bot_text": "Отправлю ссылку на оплату 50 000 рублей.",
        "bot_action_decision": {"enabled": True, "action": "send_payment_link", "active_brand": "unpk"},
        "bot_direct_path": {"retrieved_facts": {"price": "УНПК: подтверждённая сумма 50 000 рублей."}},
    }

    result = judge.evaluate_action_turn(turn, persona=persona)

    assert result["reward_eligible"] is True
    assert result["hard_barriers"] == []
    assert result["soft_flags"] == []


def test_hard_barriers_use_retrieved_facts_not_bot_text() -> None:
    persona = {
        "brand": "unpk",
        "expected_action": {"action": "send_payment_link", "manual_label": True},
        "deal_card": {
            "brand": "unpk",
            "preconditions": {
                "product_selected": True,
                "price_confirmed": True,
                "client_ready_to_pay": True,
            },
        },
    }
    turn = {
        "client_message": "Готовы оплатить.",
        "bot_text": "Отправлю ссылку на оплату 60 000 рублей.",
        "bot_action_decision": {"enabled": True, "action": "send_payment_link", "active_brand": "unpk"},
        "bot_direct_path": {"retrieved_facts": {"price": "УНПК: подтверждённая сумма 50 000 рублей."}},
    }

    result = judge.evaluate_action_turn(turn, persona=persona)

    assert result["reward_eligible"] is False
    assert result["hard_barriers"] == ["fabricated_amount"]
    assert result["fact_amounts"] == ["50000"]
    assert result["text_amounts"] == ["60000"]


def test_cross_brand_action_is_deterministic_from_retrieved_facts() -> None:
    persona = {
        "brand": "unpk",
        "expected_action": {"action": "send_materials", "manual_label": True},
        "deal_card": {"brand": "unpk", "preconditions": {"product_selected": True, "wants_trial": True}},
    }
    turn = {
        "client_message": "Можно пробный фрагмент?",
        "bot_text": "Отправлю фрагмент занятия и материалы.",
        "bot_action_decision": {"enabled": True, "action": "send_materials", "active_brand": "unpk"},
        "bot_direct_path": {"retrieved_facts": {"materials": "Фотон: онлайн-фрагмент занятия и материалы."}},
    }

    result = judge.evaluate_action_turn(turn, persona=persona)

    assert result["hard_barriers"] == ["cross_brand_action"]
    assert result["fact_brands"] == ["foton"]


def test_expected_action_without_action_decision_is_hard_signal_missing() -> None:
    persona = {
        "brand": "unpk",
        "expected_action": {"action": "send_materials", "manual_label": True},
        "deal_card": {"brand": "unpk", "preconditions": {"product_selected": True, "wants_trial": True}},
    }
    turn = {"client_message": "Можно пробный фрагмент?", "bot_text": "Да, подскажу порядок."}

    result = judge.evaluate_action_turn(turn, persona=persona)

    assert result["hard_barriers"] == ["action_signal_missing"]
    assert result["soft_flags"] == ["expected_action_missing"]


def test_action_judge_input_validation_forbids_protection_flags_and_requires_env(monkeypatch) -> None:
    judge_spec = {"action_judge_enabled": True}
    persona = {"dialog_id": "p1", "brand": "unpk", "expected_action": {"action": "answer_only"}}
    monkeypatch.delenv(judge.ACTION_JUDGE_FLAG_ENV, raising=False)

    with pytest.raises(ValueError, match=judge.ACTION_JUDGE_FLAG_ENV):
        judge.validate_action_judge_inputs([persona], judge_spec=judge_spec)

    monkeypatch.setenv(judge.ACTION_JUDGE_FLAG_ENV, "1")
    forbidden = {**persona, "context": {"TELEGRAM_OUTPUT_SANITIZER": "0"}}
    with pytest.raises(ValueError, match="protection flag keys"):
        judge.validate_action_judge_inputs([forbidden], judge_spec=judge_spec)


def test_manual_gold_calibration_accepts_26_rows() -> None:
    rows = load_gold_rows(GOLD_PATH)
    report = judge.evaluate_action_gold_rows(rows)

    assert len(rows) == 26
    assert report["accepted"] is True
    assert report["unsafe_false_passes"] == 0
    assert report["hard_false_negatives"] == 0
    assert report["hard_false_positives"] == 0
    assert report["soft_false_positives"] <= 1
    assert all(row.get("manual_label") is True for row in rows)
    assert "срочно оплатить" in json.dumps(rows, ensure_ascii=False)
    assert "send_crm_data" in json.dumps(rows, ensure_ascii=False)
