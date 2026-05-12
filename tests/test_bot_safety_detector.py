from __future__ import annotations

from pathlib import Path

from mango_mvp.quality.bot_safety_detector import detect_bot_safety_risks


def test_detector_is_independent_from_sanitizer_regex_module() -> None:
    source = Path("src/mango_mvp/quality/bot_safety_detector.py").read_text(encoding="utf-8")

    assert "mango_mvp.insights.sanitizers" not in source


def test_detector_finds_core_bot_safety_risks() -> None:
    text = (
        "Первый семестр за 88000. Преподаватель Гамзяков ждет на улице Майская, кабинет 324. "
        "До 17 числа напишем @manager."
    )

    risk_types = {finding.risk_type for finding in detect_bot_safety_risks(text, min_severity="P2")}

    assert "money_or_terms" in risk_types
    assert "contact_data" in risk_types
    assert "role_name" in risk_types
    assert "address_or_room" in risk_types
    assert "deadline" in risk_types


def test_detector_treats_safe_generic_answer_as_clean() -> None:
    text = (
        "Актуальные условия менеджер подтвердит по текущим правилам. "
        "Адрес, который подтвердит менеджер, будет отправлен после проверки."
    )

    assert detect_bot_safety_risks(text, min_severity="P2") == []


def test_detector_marks_repeated_placeholder_cluster_as_p3_only() -> None:
    text = "Подойдите к вахте в адрес, который подтвердит менеджер, адрес, который подтвердит менеджер."

    assert detect_bot_safety_risks(text, min_severity="P2") == []
    p3_findings = detect_bot_safety_risks(text, min_severity="P3")
    assert {finding.risk_type for finding in p3_findings} == {"over_sanitization_cluster_repeat"}
