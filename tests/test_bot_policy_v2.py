from __future__ import annotations

from typing import Any, Callable, Mapping

import pytest

from mango_mvp.channels import subscription_llm
from mango_mvp.channels.subscription_llm import (
    SubscriptionDraftResult,
    detect_high_risk_input_markers,
)


def _draft(
    text: str = "Здравствуйте! Менеджер сверит детали и подскажет следующий шаг.",
    *,
    route: str = "draft_for_manager",
    topic_id: str = "theme:001_pricing",
) -> SubscriptionDraftResult:
    return SubscriptionDraftResult(
        route=route,
        draft_text=text,
        message_type="question",
        topic_id=topic_id,
        topic_confidence=0.95,
        confidence_group=0.95,
        risk_level="low",
    )


def _require_guard(name: str) -> Callable[..., SubscriptionDraftResult]:
    guard = getattr(subscription_llm, name, None)
    if not callable(guard):
        pytest.fail(f"subscription_llm.{name} must exist for bot policy v2")
    return guard


def _assert_input_is_not_high_risk(client_message: str, forbidden_marker: str) -> None:
    markers = detect_high_risk_input_markers(client_message)
    assert forbidden_marker not in markers


def _payment_context(
    *,
    amo_status: str | None = None,
    tallanto_status: str | None = None,
    conflict: bool = False,
) -> Mapping[str, Any]:
    context: dict[str, Any] = {
        "payment_last_seen_at": "2026-05-17T12:00:00+03:00",
        "payment_source_confidence": "high",
    }
    if amo_status is not None:
        context["amo_payment_status"] = amo_status
    if tallanto_status is not None:
        context["tallanto_payment_status"] = tallanto_status
    if conflict:
        context["payment_conflict"] = True
    return context


def test_matkap_information_question_is_not_forced_manager_only() -> None:
    _assert_input_is_not_high_risk("Можно оплатить обучение материнским капиталом?", "matkap")
    _assert_input_is_not_high_risk("Какие документы нужны для маткапитала?", "matkap")


def test_tax_deduction_information_question_is_not_forced_manager_only() -> None:
    _assert_input_is_not_high_risk("Можно оформить налоговый вычет за обучение?", "tax")
    _assert_input_is_not_high_risk("Вы даете справку для налоговой?", "tax")


def test_refund_request_is_classified_high_risk() -> None:
    message = "Хочу вернуть деньги и расторгнуть договор."

    markers = detect_high_risk_input_markers(message)
    assert "refund" in markers


def test_legal_threat_is_classified_high_risk() -> None:
    message = "Если не решите вопрос сегодня, подам иск в суд и обращусь в Роспотребнадзор."

    markers = detect_high_risk_input_markers(message)
    assert "legal" in markers


def test_general_legal_reference_does_not_force_manager_only() -> None:
    _assert_input_is_not_high_risk("Можно ли по закону получить справку об обучении?", "legal")


def test_payment_report_does_not_force_manager_only_without_confirmation() -> None:
    _assert_input_is_not_high_risk("Я оплатил курс, чек пришлю чуть позже.", "payment_status")


def test_payment_confirmation_requires_matching_amo_tallanto() -> None:
    guard = _require_guard("apply_payment_confirmation_guard")
    confirming_draft = _draft(
        "Вижу, что оплата отмечена.",
        topic_id="theme:003_payment_status",
    )
    client_message = "Проверьте, прошла ли оплата?"

    without_sources = guard(confirming_draft, client_message=client_message, context={})
    only_amo = guard(
        confirming_draft,
        client_message=client_message,
        context=_payment_context(amo_status="paid"),
    )
    matched_sources = guard(
        confirming_draft,
        client_message=client_message,
        context=_payment_context(amo_status="paid", tallanto_status="paid"),
    )

    assert without_sources.route == "manager_only"
    assert only_amo.route == "manager_only"
    assert any("payment" in flag and ("source" in flag or "confirmation" in flag) for flag in without_sources.safety_flags)
    assert "оплата отмечена" not in without_sources.draft_text.casefold()
    assert "оплата отмечена" not in only_amo.draft_text.casefold()
    assert matched_sources.route == "draft_for_manager"
    assert "оплата отмечена" in matched_sources.draft_text.casefold()
    assert "payment_source_conflict" not in matched_sources.safety_flags


def test_payment_conflict_forces_manager_only() -> None:
    guard = _require_guard("apply_payment_confirmation_guard")

    result = guard(
        _draft("Вижу, что оплата отмечена.", topic_id="theme:003_payment_status"),
        client_message="Прошла ли оплата?",
        context=_payment_context(amo_status="paid", tallanto_status="not_paid", conflict=True),
    )

    assert result.route == "manager_only"
    assert "payment_source_conflict" in result.safety_flags
    assert any("Сверить AMO и Tallanto" in item for item in result.manager_checklist)
    assert "оплата отмечена" not in result.draft_text.casefold()
