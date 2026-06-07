from __future__ import annotations

import json

from mango_mvp.channels.dialogue_memory import (
    build_memory_llm_prompt,
    build_dialogue_memory,
    update_dialogue_memory_after_answer,
)


def _trace_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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


def test_dialogue_memory_llm_enriches_paraphrased_slots_without_changing_brand() -> None:
    memory = build_dialogue_memory(
        current_message="Сыну нужно по айти-ЕГЭ, десятый, хотим дистанционно. Сколько стоит?",
        active_brand="foton",
        recent_messages=[],
        session_id="s-memory-llm",
    )
    prompts: list[str] = []

    def memory_llm_fn(prompt: str):
        prompts.append(prompt)
        return {
            "slots": {
                "subject": "информатика",
                "grade": "10",
                "format": "онлайн",
                "active_brand": "unpk",
            },
            "topic": {
                "brand": "unpk",
                "subject": "информатика",
                "grade": "10",
                "format": "онлайн",
                "product_family": "regular_course",
            },
            "open_question": {"text": "Сколько стоит?", "kind": "price", "answered": False},
            "commitments": ["manager_handoff"],
            "summary": "Клиент интересуется онлайн-информатикой для 10 класса и ценой.",
        }

    updated = update_dialogue_memory_after_answer(
        memory,
        answer_text="Передам менеджеру, он уточнит стоимость.",
        route="bot_answer_self",
        memory_llm_fn=memory_llm_fn,
    )
    view = updated.to_prompt_view()

    assert prompts
    assert "low reasoning" in prompts[0]
    assert "active_brand менять нельзя" in prompts[0]
    assert updated.active_brand == "foton"
    assert view["topic_focus"]["brand"] == "foton"
    assert view["known_slots"]["subject"] == "информатика"
    assert view["known_slots"]["grade"] == "10"
    assert view["known_slots"]["format"] == "онлайн"
    assert updated.known_slots["subject"].source == "memory_llm"
    assert view["topic_focus"]["product_family"] == "regular_course"
    assert "manager_handoff" in updated.last_bot_commitments
    assert view["conversation_summary_short"].startswith("Клиент интересуется онлайн-информатикой")


def test_dialogue_memory_llm_does_not_override_prior_client_slot_without_current_support() -> None:
    initial = build_dialogue_memory(
        current_message="9 класс, физика, онлайн",
        active_brand="foton",
        session_id="s-memory-llm-prior-slot",
    )
    followup = build_dialogue_memory(
        current_message="А сколько стоит?",
        active_brand="foton",
        previous_memory=initial,
        session_id="s-memory-llm-prior-slot",
    )

    def memory_llm_fn(_prompt: str):
        return {
            "slots": {"grade": "10", "subject": "информатика"},
            "topic": {"grade": "10", "subject": "информатика"},
            "open_question": {"text": "А сколько стоит?", "kind": "price", "answered": False},
            "commitments": [],
            "summary": "Клиент уточняет цену.",
        }

    updated = update_dialogue_memory_after_answer(
        followup,
        answer_text="Стоимость уточнит менеджер.",
        route="bot_answer_self",
        memory_llm_fn=memory_llm_fn,
    )
    view = updated.to_prompt_view()

    assert view["known_slots"]["grade"] == "9"
    assert view["known_slots"]["subject"] == "физика"
    assert updated.known_slots["grade"].source == "dialogue_memory"


def test_dialogue_memory_llm_summary_keeps_multiturn_meaning_and_commitment() -> None:
    memory = build_dialogue_memory(
        current_message="8 класс, физика, онлайн, хотим понять места.",
        active_brand="foton",
        recent_messages=[
            "Клиент: Добрый вечер, выбираем курс.",
            "Ответ: Подскажу по проверенным данным.",
            "Клиент: Ребёнок сейчас в 8 классе.",
            "Ответ: Зафиксировал класс.",
            "Клиент: Интересует физика онлайн.",
            "Ответ: Сориентирую по формату.",
        ],
        session_id="s-memory-llm-summary",
    )

    def memory_llm_fn(prompt: str):
        assert "1-2 смысловые фразы" in prompt
        assert "на чём остановились" in prompt
        return {
            "slots": {"grade": "8", "subject": "физика", "format": "онлайн"},
            "topic": {"grade": "8", "subject": "физика", "format": "онлайн"},
            "open_question": {"text": "Хотим понять места.", "kind": "live_availability", "answered": False},
            "commitments": ["check_availability"],
            "summary": "Семья выбирает онлайн-физику для 8 класса; бот обещал передать менеджеру проверку мест.",
        }

    updated = update_dialogue_memory_after_answer(
        memory,
        answer_text="По местам без проверки не обещаю: менеджер проверит наличие.",
        route="bot_answer_self",
        safety_flags=(),
        memory_llm_fn=memory_llm_fn,
    )

    assert updated.conversation_summary_short == (
        "Семья выбирает онлайн-физику для 8 класса; бот обещал передать менеджеру проверку мест."
    )
    assert "check_availability" in updated.last_bot_commitments
    assert updated.to_prompt_view()["conversation_summary_short"] == updated.conversation_summary_short


def test_dialogue_memory_summary_falls_back_to_slot_glue_without_model() -> None:
    memory = build_dialogue_memory(
        current_message="8 класс, физика, онлайн, сколько стоит?",
        active_brand="foton",
        recent_messages=[],
        session_id="s-memory-summary-fallback",
    )

    updated = update_dialogue_memory_after_answer(
        memory,
        answer_text="Стоимость уточнит менеджер.",
        route="bot_answer_self",
        memory_llm_fn=None,
    )

    assert updated.conversation_summary_short == memory.conversation_summary_short
    assert "класс: 8" in updated.conversation_summary_short
    assert "предмет: физика" in updated.conversation_summary_short
    assert "формат: онлайн" in updated.conversation_summary_short


def test_dialogue_memory_llm_is_optional_and_regex_fallback_stays_unchanged() -> None:
    memory = build_dialogue_memory(
        current_message="Айти-ЕГЭ дистанционно, сколько стоит?",
        active_brand="foton",
        recent_messages=[],
        session_id="s-memory-llm-none",
    )

    without_model = update_dialogue_memory_after_answer(
        memory,
        answer_text="Сейчас точно ответить не могу, передаю менеджеру.",
        route="bot_answer_self",
        memory_llm_fn=None,
    )

    def failing_memory_llm(_prompt: str):
        raise RuntimeError("memory model unavailable")

    failed_model = update_dialogue_memory_after_answer(
        memory,
        answer_text="Сейчас точно ответить не могу, передаю менеджеру.",
        route="bot_answer_self",
        memory_llm_fn=failing_memory_llm,
    )

    assert failed_model.known_slots == without_model.known_slots
    assert failed_model.topic_focus == without_model.topic_focus
    assert failed_model.open_question == without_model.open_question
    assert failed_model.conversation_summary_short == without_model.conversation_summary_short


def test_build_memory_llm_prompt_requests_strict_json_and_keeps_brand_rule() -> None:
    memory = build_dialogue_memory(
        current_message="9 класс, физика онлайн",
        active_brand="unpk",
        session_id="s-memory-llm-prompt",
    )

    prompt = build_memory_llm_prompt(memory.turns, memory)

    assert "строгий JSON" in prompt
    assert "low reasoning" in prompt
    assert "мелкую/быструю модель" in prompt
    assert "active_brand менять нельзя" in prompt


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


def test_dialogue_memory_debug_trace_records_p0_latch_transition(tmp_path) -> None:
    memory = build_dialogue_memory(
        current_message="Хочу вернуть деньги, иначе пойду в суд.",
        active_brand="foton",
        recent_messages=["Клиент: 8 класс математика"],
        context={
            "dialogue_contract_debug_trace": {
                "enabled": True,
                "run_dir": str(tmp_path),
                "dialog_id": "memory_trace",
                "turn": 1,
            }
        },
        session_id="s4-trace",
    )

    assert memory.p0_latch.active is True
    rows = _trace_rows(tmp_path / "debug_trace.jsonl")
    latch = next(row for row in rows if row["node"] == "_next_p0_latch")
    assert latch["dialog_id"] == "memory_trace"
    assert latch["values"]["next_active"] is True
    assert "legal_threat" in latch["values"]["next_codes"]


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
        current_message="Деньги списали, а платежа в системе нет.",
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
    assert "p0" not in cleared.risk_flags
    assert "payment_dispute" not in cleared.risk_flags


def test_dialogue_memory_does_not_latch_benign_hypothetical_refund() -> None:
    memory = build_dialogue_memory(
        current_message="У знакомых был возврат, а у вас как с такими ситуациями?",
        active_brand="foton",
        session_id="s-benign-refund",
    )

    assert memory.p0_latch.active is False
    assert "refund" not in memory.risk_flags
    assert memory.handoff_state != "required"


def _build_memory_sequence(messages: list[str], *, session_id: str = "s-p0-auto-release"):
    memory = None
    for message in messages:
        memory = build_dialogue_memory(
            current_message=message,
            active_brand="foton",
            previous_memory=memory,
            session_id=session_id,
        )
    assert memory is not None
    return memory


def test_dialogue_memory_autonomously_releases_refund_latch_after_five_neutral_turns() -> None:
    memory = _build_memory_sequence(
        [
            "Верните деньги за курс.",
            "А по каким дням занятия?",
            "Сколько длится урок?",
            "Можно онлайн?",
            "Где смотреть записи?",
            "Какой адрес очной площадки?",
        ]
    )

    assert memory.p0_latch.active is False
    assert memory.p0_latch.release_event_id == "autonomous_neutral_p0_latch_release_5_turns"
    assert memory.p0_latch.had_hard_p0_claim is True
    assert "p0" not in memory.risk_flags
    assert "refund" not in memory.risk_flags
    assert memory.handoff_state != "required"
    assert memory.held_state.p0_latched is False


def test_dialogue_memory_released_refund_latch_does_not_mute_next_benign_turn() -> None:
    released = _build_memory_sequence(
        [
            "Верните деньги за курс.",
            "А по каким дням занятия?",
            "Сколько длится урок?",
            "Можно онлайн?",
            "Где смотреть записи?",
            "Какой адрес очной площадки?",
        ],
        session_id="s-refund-release-next",
    )

    followup = build_dialogue_memory(
        current_message="И какая цена за семестр?",
        active_brand="foton",
        previous_memory=released,
        session_id="s-refund-release-next",
    )

    assert released.p0_latch.release_event_id == "autonomous_neutral_p0_latch_release_5_turns"
    assert followup.p0_latch.active is False
    assert followup.p0_latch.had_hard_p0_claim is True
    assert followup.handoff_state != "required"
    assert "p0" not in followup.risk_flags


def test_dialogue_memory_complaint_latch_autonomously_releases_after_five_neutral_turns() -> None:
    memory = _build_memory_sequence(
        [
            "Преподаватель ужасно ведёт занятия, я недовольна.",
            "А по каким дням занятия?",
            "Сколько длится урок?",
            "Можно онлайн?",
            "Где смотреть записи?",
            "Какой адрес очной площадки?",
        ],
        session_id="s-complaint-release",
    )

    assert memory.p0_latch.active is False
    assert memory.p0_latch.release_event_id == "autonomous_neutral_p0_latch_release_5_turns"
    assert memory.handoff_state != "required"


def test_dialogue_memory_soft_negative_does_not_start_p0_latch() -> None:
    memory = _build_memory_sequence(
        [
            "Вы не отвечаете нормально, но вопрос по расписанию.",
            "А по каким дням занятия?",
            "Сколько длится урок?",
            "Можно онлайн?",
            "Где смотреть записи?",
        ],
        session_id="s-soft-negative",
    )

    assert memory.p0_latch.active is False
    assert "p0" not in memory.risk_flags
    assert memory.handoff_state != "required"


def test_dialogue_memory_does_not_release_latch_when_new_p0_appears_inside_window() -> None:
    memory = _build_memory_sequence(
        [
            "Верните деньги за курс.",
            "А по каким дням занятия?",
            "Сколько длится урок?",
            "Можно онлайн?",
            "Где смотреть записи?",
            "Преподаватель ужасно ведёт занятия, я недовольна.",
        ],
        session_id="s-refund-new-p0-window",
    )

    assert memory.p0_latch.active is True
    assert "refund" in memory.p0_latch.codes
    assert memory.handoff_state == "required"


def test_dialogue_memory_does_not_autonomously_release_legal_latch() -> None:
    memory = _build_memory_sequence(
        [
            "Пойду в суд, если вопрос не решите.",
            "А по каким дням занятия?",
            "Сколько длится урок?",
            "Можно онлайн?",
            "Где смотреть записи?",
            "Какой адрес очной площадки?",
            "А цена какая?",
        ],
        session_id="s-legal-latch",
    )

    assert memory.p0_latch.active is True
    assert "legal_threat" in memory.p0_latch.codes
    assert memory.handoff_state == "required"


def test_dialogue_memory_does_not_autonomously_release_payment_dispute_latch() -> None:
    memory = _build_memory_sequence(
        [
            "Деньги списали, а платежа в системе нет.",
            "А по каким дням занятия?",
            "Сколько длится урок?",
            "Можно онлайн?",
            "Где смотреть записи?",
            "Какой адрес очной площадки?",
            "А цена какая?",
        ],
        session_id="s-payment-latch",
    )

    assert memory.p0_latch.active is True
    assert "payment_dispute" in memory.p0_latch.codes
    assert memory.handoff_state == "required"


def test_dialogue_memory_autonomous_release_treats_bot_frustration_as_neutral() -> None:
    memory = _build_memory_sequence(
        [
            "Верните деньги за курс.",
            "Вы не отвечаете нормально.",
            "А по каким дням занятия?",
            "Сколько длится урок?",
            "Можно онлайн?",
            "Где смотреть записи?",
        ],
        session_id="s-refund-plus-frustration",
    )

    assert memory.p0_latch.active is False
    assert memory.p0_latch.release_event_id == "autonomous_neutral_p0_latch_release_5_turns"
    assert "p0" not in memory.risk_flags
    assert memory.handoff_state != "required"
