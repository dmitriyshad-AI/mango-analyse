"""Тесты каркаса X2 (рерайт + безопасность). Модель замокана — живых вызовов нет.
Запуск: PYTHONPATH=../reference python3 -m pytest test_humanity_rewriter.py"""

from humanity_rewriter import (
    apply_rewrite,
    brand_leak,
    fact_drift,
    should_rewrite,
)

FACTS = {"price": "семестр — 29 750 ₽, год — 47 250 ₽", "format": "онлайн для 5-11 классов"}


def test_should_rewrite_gates():
    p0 = {"bot_route": "manager_only", "bot_safety_flags": "high_risk_manager_only"}
    assert should_rewrite(p0, {"meta_leak": ["x"]}) is False  # P0 никогда
    nofact = {"bot_route": "draft_for_manager", "bot_safety_flags": "autonomy_default_cautious_missing_facts"}
    assert should_rewrite(nofact, {"over_handoff": True}) is False  # нет факта
    ok = {"bot_route": "bot_answer_self_for_pilot", "bot_safety_flags": "autonomy_matrix_passed"}
    assert should_rewrite(ok, {"stock_opener": ("crutch", "x")}) is True  # есть флаг
    assert should_rewrite(ok, {}) is False  # режим linter: без флага не зовём
    assert should_rewrite(ok, {}, mode="all_eligible") is True  # режим «на всех подходящих»


def test_fact_drift_detects_new_number():
    # рерайт ввёл цену 31 000, которой нет ни в черновике, ни в фактах -> дрейф
    assert fact_drift("теперь стоит 31 000 ₽", "стоит 29 750 ₽", FACTS) == ["31000"]
    # число из фактов (с пробелом) -> не дрейф
    assert fact_drift("год — 47 250 ₽", "семестр 29 750 ₽", FACTS) == []


def test_brand_leak():
    assert brand_leak("Это есть и в УНПК МФТИ", "foton") is True
    assert brand_leak("Программа МФТИ для 9 класса", "unpk") is False  # для УНПК «мфти» нормально
    assert brand_leak("Фотон предлагает рассрочку", "unpk") is True


def _turn(text):
    return {"bot_text": text, "bot_route": "bot_answer_self_for_pilot", "bot_safety_flags": "autonomy_matrix_passed"}


def test_apply_rewrite_disabled():
    out = apply_rewrite(_turn("исходный"), rewrite_fn=None, linter_flags={"stock_opener": 1})
    assert out["rewritten"] is False and out["draft_text"] == "исходный" and out["fallback_reason"] == "rewriter_disabled"


def test_apply_rewrite_p0_not_triggered():
    p0 = {"bot_text": "Приняли обращение.", "bot_route": "manager_only", "bot_safety_flags": "high_risk_manager_only"}
    out = apply_rewrite(p0, rewrite_fn=lambda p: "живее", linter_flags={"meta_leak": ["x"]})
    assert out["rewritten"] is False and out["draft_text"] == "Приняли обращение."


def test_apply_rewrite_good_candidate():
    orig = "Стоимость: семестр 29 750 ₽. Передам менеджеру для записи."
    better = "По онлайн-математике семестр — 29 750 ₽. Подскажу с записью или подберу удобную группу — что ближе?"
    out = apply_rewrite(
        _turn(orig), rewrite_fn=lambda p: better, confirmed_facts=FACTS,
        active_brand="foton", linter_flags={"over_handoff": True},
    )
    assert out["rewritten"] is True and out["draft_text"] == better


def test_apply_rewrite_fallback_on_fact_drift():
    orig = "Семестр 29 750 ₽."
    bad = "Семестр всего 19 900 ₽, успейте!"  # выдуманная цена
    out = apply_rewrite(
        _turn(orig), rewrite_fn=lambda p: bad, confirmed_facts=FACTS,
        active_brand="foton", linter_flags={"stock_opener": ("crutch", "x")},
    )
    assert out["rewritten"] is False and out["draft_text"] == orig and out["fallback_reason"].startswith("fact_drift")


def test_apply_rewrite_fallback_on_brand_leak():
    orig = "Семестр 29 750 ₽."
    bad = "А ещё в УНПК МФТИ дешевле."
    out = apply_rewrite(
        _turn(orig), rewrite_fn=lambda p: bad, confirmed_facts=FACTS,
        active_brand="foton", linter_flags={"over_handoff": True},
    )
    assert out["rewritten"] is False and out["draft_text"] == orig and out["fallback_reason"] == "brand_leak"


def test_apply_rewrite_fallback_on_meta_leak():
    orig = "Семестр 29 750 ₽."
    bad = "Отвечу без служебных пометок: семестр 29 750 ₽."
    out = apply_rewrite(
        _turn(orig), rewrite_fn=lambda p: bad, confirmed_facts=FACTS,
        active_brand="foton", linter_flags={"stock_opener": ("crutch", "x")},
    )
    assert out["rewritten"] is False and out["draft_text"] == orig and out["fallback_reason"] == "meta_leak"


def test_apply_rewrite_fallback_on_error():
    def boom(p):
        raise RuntimeError("model down")
    out = apply_rewrite(
        _turn("Семестр 29 750 ₽."), rewrite_fn=boom, confirmed_facts=FACTS,
        active_brand="foton", linter_flags={"over_handoff": True},
    )
    assert out["rewritten"] is False and out["fallback_reason"] == "rewriter_error"
