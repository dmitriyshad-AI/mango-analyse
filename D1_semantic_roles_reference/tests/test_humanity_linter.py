"""Тесты линтера человечности. Запуск: PYTHONPATH=../reference python3 -m pytest test_humanity_linter.py"""

from humanity_linter import (
    detect_meta_leak,
    detect_over_handoff,
    detect_repeat,
    detect_stock_opener,
    is_p0,
    lint_turn,
)


def test_meta_leak_service_phrase():
    assert detect_meta_leak("Передам менеджеру, он ответит без служебных пометок.")
    assert detect_meta_leak("fact_id: FP-2025-0099 цена")  # внутренний id
    assert not detect_meta_leak("Здравствуйте! Цена семестра 29 750 ₽.")


def test_is_p0():
    assert is_p0("manager_only", "high_risk_manager_only")
    assert is_p0("draft_for_manager", "conversation_intent_plan_p0")
    assert not is_p0("bot_answer_self_for_pilot", "autonomy_matrix_passed")


def test_over_handoff_only_non_p0():
    # не-P0 хендофф без факта -> флаг
    assert detect_over_handoff("Передам менеджеру, он подберёт.", "draft_for_manager", "autonomy_matrix_passed")
    # законный P0 хендофф -> НЕ флаг
    assert not detect_over_handoff("Приняли обращение. Передам менеджеру.", "manager_only", "high_risk_manager_only")
    # нет фразы хендоффа -> НЕ флаг
    assert not detect_over_handoff("Цена семестра 29 750 ₽.", "bot_answer_self_for_pilot", "autonomy_matrix_passed")


def test_near_repeat_excludes_p0():
    a = "Да, это можно уточнить заранее по программе. По смыслу всё штатно."
    assert detect_repeat(a, a, "draft_for_manager", "autonomy_matrix_passed") is not None  # дословный повтор не-P0
    # сухой P0-повтор допустим
    p0 = "Приняли обращение. Передам менеджеру."
    assert detect_repeat(p0, p0, "manager_only", "high_risk_manager_only") is None


def test_stock_opener():
    assert detect_stock_opener("Сориентирую по проверенным данным: цена 29 750 ₽.", []) == ("crutch", "сориентирую по проверенным данным")
    # повтор зачина в диалоге
    assert detect_stock_opener("Фотон цены на", ["фотон цены на"])[0] == "reused_in_dialog"
    # уникальный первый зачин -> не флаг
    assert detect_stock_opener("Конечно, помогу с выбором группы.", []) is None
    # P0 зачин не трогаем
    assert detect_stock_opener("Приняли обращение. Передам.", [], route="manager_only", safety_flags="p0_refund") is None


def test_lint_turn_aggregates():
    turn = {
        "bot_text": "Передам менеджеру, он ответит без служебных пометок.",
        "bot_route": "draft_for_manager",
        "bot_safety_flags": "autonomy_matrix_passed",
    }
    flags = lint_turn(turn, prev_bot_text="", prior_openers=[])
    assert "meta_leak" in flags and "over_handoff" in flags
