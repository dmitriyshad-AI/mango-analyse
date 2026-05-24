from __future__ import annotations

from mango_mvp.channels.conversation_intent_plan import build_conversation_intent_plan


def test_intent_plan_treats_place_booking_as_live_availability_not_price_fix() -> None:
    plan = build_conversation_intent_plan(
        current_message="Можно закрепить место на ЛВШ для 8 класса?",
        active_brand="foton",
        recent_messages=["Клиент: интересует лагерь в Менделеево", "Ответ: Подберём смену."],
    )

    assert plan.primary_intent == "live_availability"
    assert plan.topic_id == "theme:026_camp_general"
    assert plan.answer_policy == "answer_safe_parts_then_manager_live_check"
    assert plan.route_bias == "draft_for_manager"
    assert "availability.current" in plan.required_fact_keys
    assert "seat_or_booking_words_do_not_mean_price_fix" in plan.decision_notes


def test_intent_plan_continues_online_price_context_from_memory() -> None:
    plan = build_conversation_intent_plan(
        current_message="А это цена на сейчас?",
        active_brand="foton",
        known_slots={"grade": "8", "subject": "физика", "format": "онлайн"},
        dialogue_memory_view={
            "known_slots": {"grade": "8", "subject": "физика", "format": "онлайн"},
            "open_question": {"kind": "price", "text": "Сколько стоит онлайн?"},
            "topic_focus": {"product_family": "regular_course", "product": "онлайн"},
        },
    )

    assert plan.primary_intent == "pricing"
    assert plan.product_family == "regular_course"
    assert plan.known_slots["grade"] == "8"
    assert plan.known_slots["subject"] == "физика"
    assert plan.known_slots["format"] == "онлайн"
    assert plan.topic_switch_decision == "continue"
    assert plan.route_bias == "bot_answer_self_for_pilot"


def test_intent_plan_detects_real_topic_switch_only_with_context() -> None:
    plan = build_conversation_intent_plan(
        current_message="А вместо лагеря можно онлайн курс по физике?",
        active_brand="unpk",
        dialogue_memory_view={
            "open_question": {"kind": "camp", "text": "Есть места в лагере?"},
            "topic_focus": {"product_family": "camp", "product": "lvsh_mendeleevo"},
        },
    )

    assert plan.primary_intent in {"format", "pricing", "general_consultation"}
    assert plan.topic_switch_decision == "confirmed_switch"
    assert plan.product_family == "regular_course"


def test_intent_plan_does_not_treat_obsudit_as_court() -> None:
    plan = build_conversation_intent_plan(
        current_message="А чтобы записаться или с менеджером обсудить, надо приезжать или можно дистанционно?",
        active_brand="unpk",
        known_slots={"grade": "9", "subject": "информатика", "format": "онлайн"},
    )

    assert plan.primary_intent in {"format", "trial", "general_consultation"}
    assert "legal" not in plan.risk_signals
    assert plan.route_bias != "manager_only"


def test_intent_plan_keeps_real_legal_threat_p0() -> None:
    plan = build_conversation_intent_plan(
        current_message="Если не решите вопрос, пойду в суд и Роспотребнадзор.",
        active_brand="unpk",
    )

    assert plan.primary_intent == "legal_threat"
    assert "legal" in plan.risk_signals
    assert plan.route_bias == "manager_only"


def test_intent_plan_treats_vo_skolko_as_schedule_not_price() -> None:
    plan = build_conversation_intent_plan(
        current_message="Во сколько проходят занятия по физике для 9 класса?",
        active_brand="foton",
        known_slots={"grade": "9", "subject": "физика"},
    )

    assert plan.primary_intent == "schedule"
    assert plan.topic_id == "theme:013_schedule"
    assert "prices.current" not in plan.required_fact_keys


def test_intent_plan_treats_certificate_request_as_documents_not_address() -> None:
    plan = build_conversation_intent_plan(
        current_message="Где запросить справку для налогового вычета?",
        active_brand="unpk",
    )

    assert plan.primary_intent == "tax"
    assert plan.topic_id == "theme:008_tax_deduction"
    assert "locations.current" not in plan.required_fact_keys


def test_intent_plan_treats_plain_document_request_as_documents() -> None:
    plan = build_conversation_intent_plan(
        current_message="Где запросить справку об обучении?",
        active_brand="foton",
    )

    assert plan.primary_intent == "document"
    assert plan.topic_id == "theme:012_certificates"
    assert "documents.current" in plan.required_fact_keys


def test_intent_plan_does_not_force_individual_when_client_asks_group_vs_individual() -> None:
    plan = build_conversation_intent_plan(
        current_message="Есть группы по физике или только индивидуально?",
        active_brand="foton",
        known_slots={"subject": "физика"},
    )

    assert plan.primary_intent in {"format", "general_consultation"}
    assert plan.route_bias != "manager_only"
