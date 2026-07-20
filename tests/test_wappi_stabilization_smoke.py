from __future__ import annotations

from mango_mvp.channels.draft_prompt_builder import build_draft_prompt
from mango_mvp.channels.subscription_llm import SubscriptionDraftResult, apply_payment_confirmation_guard


def _draft(text: str, *, topic_id: str = "theme:003_payment_status") -> SubscriptionDraftResult:
    return SubscriptionDraftResult(
        route="draft_for_manager",
        draft_text=text,
        message_type="question",
        topic_id=topic_id,
        topic_confidence=0.95,
        confidence_group=0.95,
        risk_level="low",
    )


def test_payment_status_is_not_autoconfirmed_without_two_sources() -> None:
    result = apply_payment_confirmation_guard(
        _draft("Вижу, что оплата отмечена."),
        client_message="Проверьте, прошла ли оплата?",
        context={"amo_payment_status": "paid"},
    )

    assert result.route == "manager_only"
    assert "оплата отмечена" not in result.draft_text.casefold()
    assert any("payment" in flag for flag in result.safety_flags)


def test_prompt_contract_covers_brand_docs_family_phone_and_live_places() -> None:
    prompt = build_draft_prompt(
        "Есть места? И какие документы нужны?",
        context={
            "active_brand": "foton",
            "known_slots": {"grade": "8", "subject": "физика"},
            "read_only_customer_context": {
                "summary": "Семейный телефон: два ученика, требуется уточнение перед поученическим ответом.",
            },
        },
    )

    assert "Если active_brand=foton, не консультируй по УНПК МФТИ" in prompt
    assert "документов" in prompt
    assert "не обещай решение, скидку, возврат, место в группе или запись в CRM" in prompt
    assert "Не говори «места есть» без проверки" in prompt
    assert "Если в контексте несколько учеников, семейный телефон или конфликт данных" in prompt
