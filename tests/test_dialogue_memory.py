from __future__ import annotations

from mango_mvp.channels.dialogue_memory import (
    build_dialogue_memory,
    update_dialogue_memory_after_answer,
)


def test_dialogue_memory_extracts_slots_and_open_question_from_multiturn_context() -> None:
    memory = build_dialogue_memory(
        current_message="Это цена прямо сейчас? Можно оформить по текущей?",
        active_brand="foton",
        recent_messages=[
            "Клиент: 9 класс, физика, онлайн",
            "Ответ: Да, подберём вариант.",
        ],
        known_slots={},
        session_id="s1",
    )
    view = memory.to_prompt_view()

    assert view["active_brand"] == "foton"
    assert view["known_slots"]["grade"] == "9"
    assert view["known_slots"]["subject"] == "физика"
    assert view["known_slots"]["format"] == "онлайн"
    assert view["open_question"]["kind"] == "price_fix"
    assert "grade" in view["do_not_ask_again"]
    assert view["next_best_action_hint"].startswith("answer_open_question")


def test_dialogue_memory_preserves_known_slots_across_price_fix_followup() -> None:
    initial = build_dialogue_memory(
        current_message="8 класс, физика, онлайн",
        active_brand="foton",
        recent_messages=[],
        known_slots={},
        session_id="s-known",
    )

    followup = build_dialogue_memory(
        current_message="Как оформить по текущей цене?",
        active_brand="foton",
        recent_messages=[],
        known_slots={},
        previous_memory=initial,
        session_id="s-known",
    )
    view = followup.to_prompt_view()

    assert view["known_slots"]["grade"] == "8"
    assert view["known_slots"]["subject"] == "физика"
    assert view["known_slots"]["format"] == "онлайн"
    assert view["open_question"]["kind"] == "price_fix"
    assert set(view["do_not_ask_again"]) >= {"grade", "subject", "format"}
    assert view["next_best_action_hint"] == "answer_open_question:price_fix"


def test_dialogue_memory_never_changes_active_brand_from_client_text() -> None:
    memory = build_dialogue_memory(
        current_message="А в Фотоне есть рассрочка?",
        active_brand="unpk",
        recent_messages=["Клиент: интересует УНПК МФТИ"],
        session_id="s2",
    )

    assert memory.active_brand == "unpk"


def test_dialogue_memory_tracks_commitment_and_closes_open_question() -> None:
    memory = build_dialogue_memory(
        current_message="Вы бот?",
        active_brand="unpk",
        recent_messages=[],
        session_id="s3",
    )

    updated = update_dialogue_memory_after_answer(
        memory,
        answer_text="Да, я цифровой помощник УНПК МФТИ, не живой оператор. Менеджер проверит сложные вопросы.",
        route="draft_for_manager",
    )

    assert updated.open_question.answered is True
    assert updated.answered_questions == ("Вы бот?",)
    assert "manager_handoff" in updated.last_bot_commitments


def test_dialogue_memory_marks_p0_as_handoff_required() -> None:
    memory = build_dialogue_memory(
        current_message="Хочу вернуть деньги, иначе пойду в суд.",
        active_brand="foton",
        recent_messages=["Клиент: 8 класс математика"],
        session_id="s4",
    )

    assert "refund" in memory.risk_flags
    assert "legal_threat" in memory.risk_flags
    assert memory.handoff_state == "required"
    assert memory.sales_stage == "handoff_required"
    assert memory.p0_latch.active is True
    assert memory.p0_latch.primary_risk == "legal_threat"


def test_dialogue_memory_does_not_latch_presale_refund_policy_question() -> None:
    memory = build_dialogue_memory(
        current_message="До оплаты хочу понять: если ребёнку не понравится, деньги вернёте?",
        active_brand="foton",
        recent_messages=["Клиент: 6 класс математика онлайн"],
        session_id="s-presale-refund",
    )

    assert "refund" not in memory.risk_flags
    assert memory.handoff_state != "required"
    assert memory.p0_latch.active is False


def test_dialogue_memory_does_not_latch_presale_refund_after_manager_check_draft() -> None:
    memory = build_dialogue_memory(
        current_message="До оплаты хочу понять условия возврата.",
        active_brand="foton",
        recent_messages=["Клиент: 6 класс математика онлайн"],
        session_id="s-presale-refund-draft",
    )

    updated = update_dialogue_memory_after_answer(
        memory,
        answer_text="Условия возврата подтвердит менеджер до оплаты.",
        route="draft_for_manager",
        safety_flags=(
            "presale_refund_policy_manager_check",
            "manager_approval_required",
            "no_auto_send",
            "high_risk_manager_only",
        ),
    )

    assert updated.p0_latch.active is False
    assert "p0" not in updated.risk_flags


def test_dialogue_memory_latches_active_paid_refund_request() -> None:
    memory = build_dialogue_memory(
        current_message="Мы уже оплатили курс, ребёнку не понравилось, верните деньги.",
        active_brand="foton",
        recent_messages=["Клиент: 6 класс математика онлайн"],
        session_id="s-active-refund",
    )

    assert "refund" in memory.risk_flags
    assert memory.handoff_state == "required"
    assert memory.p0_latch.active is True


def test_dialogue_memory_current_terms_safe_next_action_asks_one_missing_slot() -> None:
    memory = build_dialogue_memory(
        current_message="Что нужно для записи по текущей цене?",
        active_brand="foton",
        recent_messages=["Клиент: 8 класс, физика"],
        session_id="s5",
    )

    action = memory.to_prompt_view()["safe_next_action"]

    assert action["type"] == "ask_one_slot_for_current_terms_request"
    assert action["missing_slot"] == "format"
    assert "оформление по текущим условиям" in action["client_safe_text"]
    assert "бронь" in action["do_not_promise"]
    assert "место" in action["do_not_promise"]


def test_dialogue_memory_current_terms_action_is_suppressed_for_p0() -> None:
    memory = build_dialogue_memory(
        current_message="Хочу вернуть деньги и зафиксировать цену на другой курс.",
        active_brand="foton",
        recent_messages=["Клиент: 8 класс, физика, онлайн"],
        session_id="s6",
    )

    assert memory.risk_flags
    assert memory.to_prompt_view()["safe_next_action"] == {}


def test_dialogue_memory_current_client_format_overrides_previous_format() -> None:
    initial = build_dialogue_memory(
        current_message="8 класс, физика, онлайн",
        active_brand="foton",
        session_id="s-format",
    )

    followup = build_dialogue_memory(
        current_message="Нет, всё-таки хотим очно. Это цена на сейчас?",
        active_brand="foton",
        previous_memory=initial,
        session_id="s-format",
    )

    view = followup.to_prompt_view()
    assert view["known_slots"]["format"] == "очно"
    assert view["topic_focus"]["format"] == "очно"
    assert "format" in view["do_not_ask_again"]


def test_dialogue_memory_current_client_subject_and_grade_override_previous_slots() -> None:
    initial = build_dialogue_memory(
        current_message="8 класс, математика, онлайн",
        active_brand="foton",
        session_id="s-slot-correction",
    )

    followup = build_dialogue_memory(
        current_message="Нет, не математика, а физика. И не 8, а 9 класс.",
        active_brand="foton",
        previous_memory=initial,
        session_id="s-slot-correction",
    )

    view = followup.to_prompt_view()
    assert view["known_slots"]["grade"] == "9"
    assert view["known_slots"]["subject"] == "физика"
    assert "математика" not in view["conversation_summary_short"]


def test_dialogue_memory_prompt_view_keeps_recent_history_cap_twenty_turns() -> None:
    memory = build_dialogue_memory(
        current_message="Последний вопрос: сколько стоит?",
        active_brand="foton",
        recent_messages=[f"Клиент: реплика {index}" for index in range(25)],
        session_id="s-long-history",
    )

    turns = memory.to_prompt_view()["recent_turns"]
    assert len(turns) == 20
    assert turns[-1]["text"] == "Последний вопрос: сколько стоит?"


def test_dialogue_memory_exposes_memory_2_fields_and_open_loop() -> None:
    memory = build_dialogue_memory(
        current_message="А трансфер из Москвы есть? И места есть?",
        active_brand="foton",
        recent_messages=["Клиент: 8 класс, физика, ЛВШ"],
        session_id="s-memory2",
    )
    view = memory.to_prompt_view()

    assert view["topic_focus"]["product_family"] == "camp"
    assert view["unanswered_questions"] == ["А трансфер из Москвы есть? И места есть?"]
    assert view["client_confirmed_slots"]["grade"] == "8"
    assert "Нужно сначала закрыть прямой вопрос" in view["open_loop_summary"]

    updated = update_dialogue_memory_after_answer(
        memory,
        answer_text="Да, трансфер из Москвы включён. По местам не буду обещать без проверки: менеджер проверит наличие.",
        route="draft_for_manager",
        safety_flags=(),
    )

    updated_view = updated.to_prompt_view()
    assert "transport" in updated_view["safe_answered_parts"]
    assert "availability_handoff" in updated_view["safe_answered_parts"]
    assert "check_availability" in updated_view["pending_manager_actions"]


def test_dialogue_memory_does_not_extract_slots_from_bot_answers() -> None:
    memory = build_dialogue_memory(
        current_message="Здравствуйте, что есть для 6 класса?",
        active_brand="foton",
        recent_messages=[
            "Клиент: Здравствуйте, что есть для 6 класса?",
            "Ответ: Можно посмотреть информатику и программирование.",
        ],
        session_id="s-self-pollution",
    )

    assert memory.to_prompt_view()["known_slots"]["grade"] == "6"
    assert "subject" not in memory.to_prompt_view()["known_slots"]


def test_dialogue_memory_latches_payment_dispute_until_manager_event() -> None:
    memory = build_dialogue_memory(
        current_message="С меня дважды списали деньги за оплату, верните одну.",
        active_brand="foton",
        recent_messages=["Клиент: 8 класс, физика"],
        session_id="s-p0-latch",
    )

    assert memory.p0_latch.active is True
    assert "payment_dispute" in memory.p0_latch.codes
    assert memory.handoff_state == "required"

    followup = build_dialogue_memory(
        current_message="А вообще сколько стоит год?",
        active_brand="foton",
        previous_memory=memory,
        session_id="s-p0-latch",
    )

    assert followup.p0_latch.active is True
    assert followup.handoff_state == "required"
    assert followup.to_prompt_view()["next_best_action_hint"] == "handoff_required"

    cleared = build_dialogue_memory(
        current_message="Спасибо, теперь можно вернуться к цене?",
        active_brand="foton",
        previous_memory=followup,
        context={"manager_clear_p0_latch": "manager_took_over_case"},
        session_id="s-p0-latch",
    )

    assert cleared.p0_latch.active is False
    assert cleared.p0_latch.release_event_id == "manager_took_over_case"


def test_dialogue_memory_does_not_latch_benign_hypothetical_refund() -> None:
    memory = build_dialogue_memory(
        current_message="У знакомых был возврат, а у вас как с такими ситуациями?",
        active_brand="foton",
        session_id="s-benign-refund",
    )

    assert memory.p0_latch.active is False
    assert "refund" not in memory.risk_flags
    assert memory.handoff_state != "required"
