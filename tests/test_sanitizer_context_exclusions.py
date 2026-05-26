from __future__ import annotations

from mango_mvp.insights.sanitizers import has_any_safety_risk, has_money_or_terms_risk, sanitize_answer


def test_sanitize_answer_keeps_non_money_counts() -> None:
    for text in ("5 000 человек", "за 5000 человек", "2 500 баллов"):
        result = sanitize_answer(text, mode="bot")

        assert "[CURRENT_PRICE]" not in result.text
        assert "актуальную стоимость" not in result.text
        assert "price_redacted" not in result.flags
        assert has_money_or_terms_risk(text) is False


def test_sanitize_answer_preserves_money_amount_regressions() -> None:
    cases = (
        "50к",
        "100 т.р.",
        "пятьдесят тысяч рублей",
        "стоимость 50000",
        "50 000 рублей",
        "7900 за 4 занятия",
    )

    for text in cases:
        result = sanitize_answer(text, mode="bot")

        assert ("[CURRENT_PRICE]" in result.text) or ("актуальную стоимость" in result.text)
        assert "price_redacted" in result.flags
        assert has_money_or_terms_risk(text) is True


def test_sanitize_answer_keeps_non_discount_percent_context() -> None:
    for text in ("100% результат", "100 процентов результат", "95% посещаемость", "98% сдача"):
        result = sanitize_answer(text, mode="bot")

        assert "[PAYMENT_OPTIONS]" not in result.text
        assert "актуальные варианты" not in result.text
        assert "percent_redacted" not in result.flags
        assert has_money_or_terms_risk(text) is False

    guarantee_result = sanitize_answer("100% гарантия результата", mode="bot")
    assert "[PAYMENT_OPTIONS]" not in guarantee_result.text
    assert "актуальные варианты" not in guarantee_result.text
    assert "percent_redacted" not in guarantee_result.flags


def test_sanitize_answer_preserves_discount_percent_forms() -> None:
    for text in ("10 процентов", "десять процентов", "15% скидка", "скидка 10%"):
        result = sanitize_answer(text, mode="bot")

        assert ("[PAYMENT_OPTIONS]" in result.text) or ("актуальные варианты" in result.text)
        assert has_money_or_terms_risk(text) is True


def test_sanitize_answer_removes_internal_fact_metadata() -> None:
    result = sanitize_answer(
        "Ответ клиенту. fact_id:abc fact:v3:foton:price source_id=fact:v3:price trace_id=run-1",
        mode="bot",
    )

    assert "fact_id" not in result.text
    assert "fact:v3" not in result.text
    assert "source_id" not in result.text
    assert "trace_id" not in result.text
    assert "internal_metadata_redacted" in result.flags
    assert has_any_safety_risk("Ответ fact_id:abc") is True


def test_sanitize_answer_flags_raw_json_leak() -> None:
    raw = '{"route":"draft_for_manager","draft_text":"Ответ","trace_id":"run-1"}'
    result = sanitize_answer(raw, mode="bot")

    assert "raw_json_redacted" in result.flags
    assert has_any_safety_risk(raw) is True
