from __future__ import annotations

from mango_mvp.channels.draft_prompt_builder import build_draft_prompt
from mango_mvp.channels.rules_engine import apply_rule, load_rules_registry
from mango_mvp.channels.subscription_llm import SubscriptionDraftResult, apply_input_policy_guards, apply_payment_confirmation_guard


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


def test_place_singular_address_is_not_live_seats_and_places_require_manager_check() -> None:
    registry = load_rules_registry()

    address = apply_rule(
        registry["contact_address"],
        plan={"primary_intent": "address", "direct_question": "Где место занятий Фотона?", "active_brand": "foton"},
        facts={},
        context={"active_brand": "foton"},
    )
    seats = apply_rule(
        registry["camp_lvsh"],
        plan={"primary_intent": "live_availability", "direct_question": "Есть места на ЛВШ?", "active_brand": "foton"},
        facts={},
        context={"active_brand": "foton"},
    )

    assert address is not None
    assert address.route == "bot_answer_self_for_pilot"
    assert "Верхняя Красносельская" in address.text
    assert "места есть" not in address.text.casefold()
    assert seats is not None
    assert seats.route == "draft_for_manager"
    assert "rules_engine_camp_live_availability_handoff" in seats.flags
    assert "места есть" not in seats.text.casefold()
    assert "провер" in seats.text.casefold()


def test_payment_status_is_not_autoconfirmed_without_two_sources() -> None:
    result = apply_payment_confirmation_guard(
        _draft("Вижу, что оплата отмечена."),
        client_message="Проверьте, прошла ли оплата?",
        context={"amo_payment_status": "paid"},
    )

    assert result.route == "manager_only"
    assert "оплата отмечена" not in result.draft_text.casefold()
    assert any("payment" in flag for flag in result.safety_flags)


def test_real_refund_is_manager_only_p0_without_llm() -> None:
    result = apply_input_policy_guards(
        _draft("Передам менеджеру.", topic_id="theme:009_refund"),
        client_message="Хочу вернуть деньги за курс и расторгнуть договор.",
    )

    assert result.route == "manager_only"
    assert "high_risk_input_manager_only" in result.safety_flags


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
