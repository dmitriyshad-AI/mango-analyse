from __future__ import annotations

from mango_mvp.channels.humanity_rewriter import (
    apply_rewrite,
    brand_leak,
    build_rewrite_prompt,
    fact_drift,
    should_rewrite,
)


FACTS = {"price": "семестр — 29 750 ₽, год — 47 250 ₽", "format": "онлайн для 5-11 классов"}


def _turn(text: str, *, route: str = "bot_answer_self_for_pilot", flags: str = "autonomy_matrix_passed") -> dict[str, str]:
    return {"bot_text": text, "bot_route": route, "bot_safety_flags": flags}


def test_should_rewrite_gates() -> None:
    p0 = {"bot_route": "manager_only", "bot_safety_flags": "high_risk_manager_only"}
    assert should_rewrite(p0, {"meta_leak": ["x"]}) is False
    nofact = {"bot_route": "draft_for_manager", "bot_safety_flags": "autonomy_default_cautious_missing_facts"}
    assert should_rewrite(nofact, {"over_handoff": True}) is False
    draft = {"bot_route": "draft_for_manager", "bot_safety_flags": "autonomy_matrix_passed"}
    assert should_rewrite(draft, {"over_handoff": True}) is True
    ok = {"bot_route": "bot_answer_self_for_pilot", "bot_safety_flags": "autonomy_matrix_passed"}
    assert should_rewrite(ok, {"stock_opener": ("crutch", "x")}) is True
    assert should_rewrite(ok, {}) is False
    assert should_rewrite(ok, {}, mode="all_eligible") is True


def test_build_rewrite_prompt_contains_manager_playbook_and_downgrade_rules() -> None:
    prompt = build_rewrite_prompt(
        "Сориентирую по проверенным данным: семестр 29 750 ₽. Передам менеджеру.",
        client_message="Дорого, есть варианты?",
        confirmed_facts=FACTS,
        active_brand="foton",
        linter_flags={"stock_opener": True},
    )

    assert "Playbook менеджера" in prompt
    assert "Сначала прямой ответ" in prompt
    assert "Срочность допустима только честная" in prompt
    assert "Никаких новых чисел" in prompt
    assert "Не меняй маршрут" in prompt
    assert "Иначе не создавай" in prompt


def test_fact_drift_detects_new_number() -> None:
    assert fact_drift("теперь стоит 31 000 ₽", "стоит 29 750 ₽", FACTS) == ["31000"]
    assert fact_drift("год — 47 250 ₽", "семестр 29 750 ₽", FACTS) == []


def test_brand_leak() -> None:
    assert brand_leak("Это есть и в УНПК МФТИ", "foton") is True
    assert brand_leak("Программа МФТИ для 9 класса", "unpk") is False
    assert brand_leak("Фотон предлагает рассрочку", "unpk") is True


def test_apply_rewrite_disabled() -> None:
    out = apply_rewrite(_turn("исходный"), rewrite_fn=None, linter_flags={"stock_opener": 1})
    assert out["rewritten"] is False
    assert out["draft_text"] == "исходный"
    assert out["fallback_reason"] == "rewriter_disabled"


def test_apply_rewrite_p0_not_triggered() -> None:
    p0 = _turn("Приняли обращение.", route="manager_only", flags="high_risk_manager_only")
    out = apply_rewrite(p0, rewrite_fn=lambda prompt: "живее", linter_flags={"meta_leak": ["x"]})
    assert out["rewritten"] is False
    assert out["draft_text"] == "Приняли обращение."


def test_apply_rewrite_good_candidate() -> None:
    original = "Стоимость: семестр 29 750 ₽. Передам менеджеру для записи."
    better = "По онлайн-математике семестр — 29 750 ₽. Подскажу с записью или подберу удобную группу — что ближе?"
    out = apply_rewrite(
        _turn(original),
        rewrite_fn=lambda prompt: better,
        confirmed_facts=FACTS,
        active_brand="foton",
        linter_flags={"over_handoff": True},
    )
    assert out["rewritten"] is True
    assert out["draft_text"] == better


def test_apply_rewrite_fallback_on_fact_drift() -> None:
    original = "Семестр 29 750 ₽."
    bad = "Семестр всего 19 900 ₽, успейте!"
    out = apply_rewrite(
        _turn(original),
        rewrite_fn=lambda prompt: bad,
        confirmed_facts=FACTS,
        active_brand="foton",
        linter_flags={"stock_opener": ("crutch", "x")},
    )
    assert out["rewritten"] is False
    assert out["draft_text"] == original
    assert str(out["fallback_reason"]).startswith("fact_drift")


def test_apply_rewrite_fallback_on_brand_leak() -> None:
    original = "Семестр 29 750 ₽."
    bad = "А ещё в УНПК МФТИ дешевле."
    out = apply_rewrite(
        _turn(original),
        rewrite_fn=lambda prompt: bad,
        confirmed_facts=FACTS,
        active_brand="foton",
        linter_flags={"over_handoff": True},
    )
    assert out["rewritten"] is False
    assert out["draft_text"] == original
    assert out["fallback_reason"] == "brand_leak"


def test_apply_rewrite_fallback_on_meta_leak() -> None:
    original = "Семестр 29 750 ₽."
    bad = "Отвечу без служебных пометок: семестр 29 750 ₽."
    out = apply_rewrite(
        _turn(original),
        rewrite_fn=lambda prompt: bad,
        confirmed_facts=FACTS,
        active_brand="foton",
        linter_flags={"stock_opener": ("crutch", "x")},
    )
    assert out["rewritten"] is False
    assert out["draft_text"] == original
    assert out["fallback_reason"] == "meta_leak"


def test_apply_rewrite_fallback_on_error() -> None:
    def boom(prompt: str) -> str:
        raise RuntimeError("model down")

    out = apply_rewrite(
        _turn("Семестр 29 750 ₽."),
        rewrite_fn=boom,
        confirmed_facts=FACTS,
        active_brand="foton",
        linter_flags={"over_handoff": True},
    )
    assert out["rewritten"] is False
    assert out["fallback_reason"] == "rewriter_error"


def test_apply_rewrite_runs_extra_repo_gate() -> None:
    original = "Семестр 29 750 ₽."
    out = apply_rewrite(
        _turn(original),
        rewrite_fn=lambda prompt: "Семестр 29 750 ₽, но внутренний фильтр против.",
        confirmed_facts=FACTS,
        active_brand="foton",
        linter_flags={"stock_opener": ("crutch", "x")},
        validate_fn=lambda candidate: "repo_gate:blocked" if "внутренний фильтр" in candidate else None,
    )
    assert out["rewritten"] is False
    assert out["fallback_reason"] == "repo_gate:blocked"
