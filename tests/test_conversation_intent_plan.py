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
    assert plan.fact_scope == "tax_deduction"
    assert "matkap_process" in plan.blocked_neighbor_scopes


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


def test_intent_plan_keeps_payment_question_in_installment_not_camp_from_recent_bot_text() -> None:
    plan = build_conversation_intent_plan(
        current_message="А какие именно условия, можно помесячно или за семестр? И Долями это на сколько частей?",
        active_brand="foton",
        known_slots={"grade": "3", "subject": "математика", "format": "онлайн"},
        recent_messages=[
            "Ответ: доступны варианты на 6, 10 или 12 месяцев, а также сервис Долями. Это относится к очным и онлайн-курсам, ЛВШ, ЛШ.",
        ],
    )

    assert plan.primary_intent == "installment"
    assert plan.product_family == "regular_course"
    assert plan.product_scope == "онлайн"
    assert plan.topic_id == "theme:006_installment"


def test_intent_plan_understands_not_about_seats_but_payment_terms() -> None:
    plan = build_conversation_intent_plan(
        current_message="Вы опять про места... мне нужны условия оплаты, можно помесячно или Долями?",
        active_brand="foton",
        known_slots={"grade": "3", "subject": "математика", "format": "онлайн"},
        dialogue_memory_view={"topic_focus": {"product_family": "camp", "product": "lvsh_mendeleevo"}},
    )

    assert plan.primary_intent == "installment"
    assert plan.topic_id == "theme:006_installment"
    assert plan.answer_policy == "answer_directly_if_fact_verified"


def test_intent_plan_scopes_matkap_followup_away_from_tax() -> None:
    plan = build_conversation_intent_plan(
        current_message="Маткапиталом можно оплатить? Какие документы и сколько СФР смотрит?",
        active_brand="foton",
    )

    assert plan.primary_intent == "matkap"
    assert plan.fact_scope == "matkap_process"
    assert "tax_deduction" in plan.blocked_neighbor_scopes


def test_intent_plan_scopes_class_schedule_away_from_office_hours() -> None:
    plan = build_conversation_intent_plan(
        current_message="По каким дням проходят занятия по физике?",
        active_brand="unpk",
        known_slots={"subject": "физика", "format": "очно"},
    )

    assert plan.primary_intent == "schedule"
    assert plan.fact_scope == "class_schedule"
    assert "office_hours" in plan.blocked_neighbor_scopes


def test_intent_plan_scopes_day_camp_away_from_residential_lvsh() -> None:
    plan = build_conversation_intent_plan(
        current_message="Есть дневной летний лагерь без проживания?",
        active_brand="foton",
    )

    assert plan.product_family == "camp"
    assert plan.product_scope == "city_camp"
    assert plan.fact_scope == "city_day_camp"
    assert "residential_lvsh" in plan.blocked_neighbor_scopes


def test_intent_plan_current_residential_camp_signal_wins_over_city_neighbor() -> None:
    plan = build_conversation_intent_plan(
        current_message="У нас 11 класс, физика. А выездной лагерь есть или только городская школа? И сколько смена стоит?",
        active_brand="unpk",
        known_slots={"grade": "11", "subject": "физика"},
        dialogue_memory_view={"topic_focus": {"product_family": "camp", "product": "city_camp"}},
    )

    assert plan.product_family == "camp"
    assert plan.product_scope == "lvsh_mendeleevo"
    assert plan.fact_scope == "residential_lvsh"
    assert "city_day_camp" in plan.blocked_neighbor_scopes


def test_intent_plan_client_correction_to_residential_wins_over_city_memory() -> None:
    plan = build_conversation_intent_plan(
        current_message="Я как раз про выездной, а не городской. Что там по смене?",
        active_brand="foton",
        dialogue_memory_view={"topic_focus": {"product_family": "camp", "product": "city_camp"}},
    )

    assert plan.product_family == "camp"
    assert plan.product_scope == "lvsh_mendeleevo"
    assert plan.fact_scope == "residential_lvsh"
    assert "city_day_camp" in plan.blocked_neighbor_scopes


def test_intent_plan_scopes_second_subject_discount_away_from_multichild() -> None:
    plan = build_conversation_intent_plan(
        current_message="Если взять второй предмет онлайн одному ребёнку, какая скидка?",
        active_brand="foton",
    )

    assert plan.primary_intent == "discount"
    assert plan.fact_scope == "discount_second_subject"
    assert "discount_multichild" in plan.blocked_neighbor_scopes


def test_intent_plan_scopes_second_subject_discount_with_instrumental_wording() -> None:
    plan = build_conversation_intent_plan(
        current_message="Если взять вторым предметом физику онлайн для 7 класса, какая скидка?",
        active_brand="foton",
    )

    assert plan.primary_intent == "discount"
    assert plan.fact_scope == "discount_second_subject"
    assert "discount_multichild" in plan.blocked_neighbor_scopes


def test_intent_plan_scopes_discount_stacking_as_delta_question() -> None:
    plan = build_conversation_intent_plan(
        current_message="А с многодетной скидкой она суммируется или выбирается одна?",
        active_brand="foton",
        known_slots={"grade": "7", "subject": "физика", "format": "онлайн"},
        dialogue_memory_view={"topic_focus": {"product_family": "regular_course", "product": "онлайн"}},
    )

    assert plan.primary_intent == "discount"
    assert plan.fact_scope == "discount_stacking"
    assert "discount_referral" in plan.blocked_neighbor_scopes


def test_intent_plan_scopes_multichild_discount_away_from_second_subject() -> None:
    plan = build_conversation_intent_plan(
        current_message="Для многодетной семьи какая скидка, если учится один ребёнок?",
        active_brand="unpk",
    )

    assert plan.primary_intent == "discount"
    assert plan.fact_scope == "discount_multichild"
    assert "discount_second_subject" in plan.blocked_neighbor_scopes


def test_intent_plan_keeps_trial_fragment_followup_from_previous_context() -> None:
    plan = build_conversation_intent_plan(
        current_message="А как его получить — ссылку пришлёте, запись или надо где-то регистрироваться?",
        active_brand="unpk",
        known_slots={"grade": "9", "subject": "физика", "format": "онлайн"},
        dialogue_memory_view={
            "known_slots": {"grade": "9", "subject": "физика", "format": "онлайн"},
            "open_question": {"kind": "trial", "text": "Фрагмент занятия можно посмотреть?"},
            "topic_focus": {"product_family": "regular_course", "product": "онлайн"},
        },
    )

    assert plan.primary_intent == "trial"
    assert plan.fact_scope == "trial_online_fragment"
    assert "trial_offline" in plan.blocked_neighbor_scopes


def test_intent_plan_scopes_offline_trial_away_from_online_fragment() -> None:
    plan = build_conversation_intent_plan(
        current_message="Бесплатное пробное очно на Пацаева есть?",
        active_brand="unpk",
    )

    assert plan.primary_intent == "trial"
    assert plan.fact_scope == "trial_offline"
    assert "trial_online_fragment" in plan.blocked_neighbor_scopes


def test_intent_plan_scopes_offline_recordings_away_from_camp_facts() -> None:
    plan = build_conversation_intent_plan(
        current_message="У очных занятий записи есть, если пропустим?",
        active_brand="unpk",
        known_slots={"format": "очно"},
    )

    assert plan.fact_scope == "offline_recordings"
    assert "camp_extra_facts" in plan.blocked_neighbor_scopes


def test_intent_plan_scopes_offline_recording_question_with_word_zapis_uroka() -> None:
    plan = build_conversation_intent_plan(
        current_message="8 класс, математика очно. Если пропустим занятие, запись урока будет?",
        active_brand="unpk",
        known_slots={"format": "очно"},
    )

    assert plan.primary_intent == "schedule"
    assert plan.fact_scope == "offline_recordings"
    assert "online_recordings" in plan.blocked_neighbor_scopes


def test_intent_plan_does_not_treat_signup_recording_word_as_lesson_recording() -> None:
    plan = build_conversation_intent_plan(
        current_message="Я очно смотрю, надо к вам приезжать для записи или можно записаться дистанционно?",
        active_brand="unpk",
        known_slots={"format": "очно"},
    )

    assert plan.fact_scope != "offline_recordings"
    assert "offline_recordings" not in plan.blocked_neighbor_scopes
    assert plan.primary_intent in {"format", "trial", "general_consultation"}


def test_intent_plan_scopes_online_recording_followup_from_known_format() -> None:
    plan = build_conversation_intent_plan(
        current_message="А записи занятий будут, если ребёнок пропустит?",
        active_brand="foton",
        known_slots={"grade": "7", "subject": "информатика", "format": "онлайн"},
    )

    assert plan.primary_intent == "schedule"
    assert plan.fact_scope == "online_recordings"
    assert "offline_recordings" in plan.blocked_neighbor_scopes


def test_intent_plan_treats_skolko_raz_v_nedelyu_as_schedule_not_price() -> None:
    plan = build_conversation_intent_plan(
        current_message="7 класс, информатика онлайн. По каким дням занятия и сколько раз в неделю?",
        active_brand="foton",
    )

    assert plan.primary_intent == "schedule"
    assert plan.topic_id == "theme:013_schedule"


def test_intent_plan_scopes_dolyami_away_from_bank_installment_terms() -> None:
    plan = build_conversation_intent_plan(
        current_message="Долями это на сколько частей?",
        active_brand="foton",
    )

    assert plan.primary_intent == "installment"
    assert plan.fact_scope == "dolyami_parts"
    assert "installment_bank" in plan.blocked_neighbor_scopes


def test_intent_plan_scopes_regular_online_away_from_olympiad_online() -> None:
    plan = build_conversation_intent_plan(
        current_message="Обычный онлайн 10 класс физика, сколько стоит и какое расписание?",
        active_brand="unpk",
    )

    assert plan.fact_scope == "regular_online"
    assert "olympiad_online" in plan.blocked_neighbor_scopes


def test_intent_plan_presale_refund_policy_is_not_full_p0() -> None:
    plan = build_conversation_intent_plan(
        current_message="До оплаты хочу понять: если ребёнку не понравится, деньги вернёте?",
        active_brand="foton",
    )

    assert plan.primary_intent != "refund"
    assert "refund" not in plan.risk_signals
    assert plan.route_bias != "manager_only"


def test_intent_plan_token_traps_do_not_trigger_neighbor_intents() -> None:
    tochnoy = build_conversation_intent_plan(
        current_message="Цена на сейчас, но без точной даты повышения?",
        active_brand="foton",
        known_slots={"grade": "8", "subject": "физика", "format": "онлайн"},
    )
    assert tochnoy.primary_intent in {"pricing", "price_fix"}
    assert tochnoy.known_slots["format"] == "онлайн"

    multichild = build_conversation_intent_plan(
        current_message="Для многодетной семьи какой процент скидки?",
        active_brand="foton",
    )
    assert multichild.primary_intent == "discount"
    assert multichild.fact_scope == "discount_multichild"
    assert "schedule" not in multichild.keyword_signals

    percent = build_conversation_intent_plan(
        current_message="На второй предмет какой процент скидки?",
        active_brand="foton",
    )
    assert percent.primary_intent == "discount"
    assert percent.fact_scope == "discount_second_subject"
