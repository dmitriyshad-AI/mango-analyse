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


def test_intent_plan_does_not_treat_arrival_place_as_live_availability() -> None:
    plan = build_conversation_intent_plan(
        current_message="Я ее привезу сразу на место, мы живем неподалёку",
        active_brand="foton",
        dialogue_memory_view={
            "open_question": {"kind": "camp", "text": "Ждём смену."},
            "topic_focus": {"product_family": "camp", "product": "city_day_camp"},
        },
    )

    assert plan.primary_intent != "live_availability"
    assert "availability.current" not in plan.required_fact_keys
    assert "availability" not in plan.requested_slots


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


def test_intent_plan_presale_refund_wins_over_schedule_word() -> None:
    plan = build_conversation_intent_plan(
        current_message="А если передумаю до начала занятий, деньги вернут?",
        active_brand="foton",
        known_slots={"grade": "9", "subject": "физика", "format": "очно"},
    )

    assert plan.primary_intent == "refund_policy"
    assert plan.topic_id == "theme:009_refund"
    assert plan.route_bias == "bot_answer_self_for_pilot"
    assert plan.refund_frame == "presale_policy"
    assert "refund_policy.current" in plan.required_fact_keys
    assert "schedule.current" not in plan.required_fact_keys


def test_intent_plan_refund_process_followup_does_not_become_schedule() -> None:
    plan = build_conversation_intent_plan(
        current_message="А это оформляется по заявлению? Какой порядок возврата?",
        active_brand="foton",
        known_slots={"grade": "9", "subject": "физика", "format": "очно"},
        dialogue_memory_view={
            "held_state": {
                "active_fact_scope": "refund_policy",
                "active_topics": ["refund_presale"],
                "required_fact_keys": ["refund_policy.current"],
            },
            "open_question": {"kind": "refund_policy", "text": "Если передумаю до старта, деньги вернут?"},
        },
    )

    assert plan.primary_intent == "refund_policy"
    assert plan.topic_id == "theme:009_refund"
    assert plan.fact_scope == "refund_policy"
    assert "office_hours" in plan.blocked_neighbor_scopes
    assert "refund_policy.current" in plan.required_fact_keys
    assert "schedule.current" not in plan.required_fact_keys


def test_intent_plan_negated_refund_keeps_recording_scope() -> None:
    plan = build_conversation_intent_plan(
        current_message="Я не про возврат, где смотреть запись в личном кабинете?",
        active_brand="foton",
        dialogue_memory_view={
            "held_state": {
                "active_fact_scope": "online_recordings",
                "active_topics": ["recording"],
                "required_fact_keys": ["online_recordings.current"],
            },
            "open_question": {"kind": "recording", "text": "Записи уроков будут?"},
        },
    )

    assert plan.primary_intent == "recording"
    assert plan.topic_id == "theme:018_materials_homework"
    assert plan.fact_scope == "online_recordings"
    assert "online_recordings.current" in plan.required_fact_keys
    assert "refund" not in plan.risk_signals


def test_intent_plan_olympiad_online_uses_specific_fact_key() -> None:
    plan = build_conversation_intent_plan(
        current_message="Есть олимпиадная подготовка Физтех онлайн для 10 класса?",
        active_brand="unpk",
    )

    assert plan.primary_intent == "olympiad_online"
    assert plan.fact_scope == "olympiad_online"
    assert "olympiad_online.current" in plan.required_fact_keys


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


def test_intent_plan_keeps_recording_followup_from_held_scope() -> None:
    plan = build_conversation_intent_plan(
        current_message="А где её смотреть потом?",
        active_brand="foton",
        known_slots={"grade": "8", "subject": "физика", "format": "онлайн"},
        dialogue_memory_view={
            "held_state": {
                "active_fact_scope": "online_recordings",
                "active_topics": ["recording"],
                "required_fact_keys": ["online_recordings.current"],
            },
            "topic_focus": {"product_family": "regular_course", "product": "онлайн"},
        },
    )

    assert plan.primary_intent == "recording"
    assert plan.topic_id == "theme:018_materials_homework"
    assert plan.fact_scope == "online_recordings"
    assert "online_recordings.current" in plan.required_fact_keys
    assert "locations.current" not in plan.required_fact_keys


def test_intent_plan_treats_bank_transfer_as_payment_method_not_installment() -> None:
    plan = build_conversation_intent_plan(
        current_message="Можно оплатить банковским переводом на счёт?",
        active_brand="foton",
        known_slots={"grade": "8", "subject": "физика", "format": "очно"},
    )

    assert plan.primary_intent == "payment_method"
    assert plan.topic_id == "theme:002_payment_method"
    assert "payment_methods.current" in plan.required_fact_keys
    assert "installment_terms.current" not in plan.required_fact_keys


def test_intent_plan_treats_invoice_monthly_as_payment_method_not_installment() -> None:
    plan = build_conversation_intent_plan(
        current_message="Я про счёт каждый месяц, не рассрочку через банк",
        active_brand="foton",
        known_slots={"grade": "8", "subject": "физика", "format": "очно"},
        dialogue_memory_view={
            "held_state": {
                "active_topics": ["installment"],
                "required_fact_keys": ["installment_terms.current"],
            },
            "open_question": {"kind": "installment", "text": "Можно помесячно?"},
        },
    )

    assert plan.primary_intent == "payment_by_invoice_monthly"
    assert plan.topic_id == "theme:002_payment_method"
    assert "payment_methods.current" in plan.required_fact_keys
    assert "installment_terms.current" not in plan.required_fact_keys
    assert "installment" not in plan.answer_topics


def test_intent_plan_preserves_explicit_both_formats_without_single_slot() -> None:
    plan = build_conversation_intent_plan(
        current_message="Я просила оба формата: и очно, и онлайн. По каким дням занятия?",
        active_brand="unpk",
        known_slots={"grade": "8", "subject": "физика", "format": "очно"},
        dialogue_memory_view={"held_state": {"training_format": "ochno", "training_formats": ["ochno"]}},
    )

    assert plan.primary_intent == "schedule"
    assert set(plan.training_formats) == {"online", "ochno"}
    assert "format" not in plan.known_slots


def test_intent_plan_does_not_force_individual_when_client_asks_group_vs_individual() -> None:
    plan = build_conversation_intent_plan(
        current_message="Есть группы по физике или только индивидуально?",
        active_brand="foton",
        known_slots={"subject": "физика"},
    )

    assert plan.primary_intent in {"format", "general_consultation"}
    assert plan.route_bias != "manager_only"


def test_intent_plan_multitopic_price_installment_requires_both_fact_families() -> None:
    plan = build_conversation_intent_plan(
        current_message="Сколько стоит год онлайн 11 класс физика и можно ли в рассрочку?",
        active_brand="foton",
        known_slots={"grade": "11", "subject": "физика", "format": "онлайн"},
    )

    assert "price" in plan.answer_topics
    assert "installment" in plan.answer_topics
    assert "prices.current" in plan.required_fact_keys
    assert "installment_terms.current" in plan.required_fact_keys


def test_intent_plan_price_followup_can_supersede_installment_context() -> None:
    plan = build_conversation_intent_plan(
        current_message="Я не про рассрочку уже, мне нужна цена за год онлайн по физике 11 класс",
        active_brand="foton",
        known_slots={"grade": "11", "subject": "физика", "format": "онлайн"},
        dialogue_memory_view={
            "held_state": {
                "active_topics": ["installment"],
                "required_fact_keys": ["installment_terms.current"],
            },
            "open_question": {"kind": "installment", "text": "Можно в рассрочку?"},
        },
    )

    assert plan.primary_intent == "pricing"
    assert "prices.current" in plan.required_fact_keys
    assert plan.topic_id == "theme:001_pricing"


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


def test_intent_plan_keeps_city_camp_scope_on_program_followup() -> None:
    plan = build_conversation_intent_plan(
        current_message="а что по программе там для 6 класса?",
        active_brand="foton",
        known_slots={"grade": "6"},
        dialogue_memory_view={
            "held_state": {
                "active_fact_scope": "city_day_camp",
                "active_topics": ["camp"],
                "required_fact_keys": ["programs.current"],
            },
            "topic_focus": {"product_family": "camp", "product": "city_camp"},
        },
    )

    assert plan.primary_intent == "camp"
    assert plan.product_family == "camp"
    assert plan.product_scope == "city_camp"
    assert plan.fact_scope == "city_day_camp"
    assert "programs.current" in plan.required_fact_keys


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

    assert plan.primary_intent == "recording"
    assert "offline_recordings.current" in plan.required_fact_keys
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

    assert plan.primary_intent == "recording"
    assert "online_recordings.current" in plan.required_fact_keys
    assert plan.fact_scope == "online_recordings"
    assert "offline_recordings" in plan.blocked_neighbor_scopes


def test_intent_plan_uses_matkap_timeline_key_for_sfr_timing() -> None:
    plan = build_conversation_intent_plan(
        current_message="Маткапиталом можно оплатить? Сколько СФР рассматривает заявление?",
        active_brand="unpk",
    )

    assert plan.primary_intent == "matkap"
    assert "matkap_timeline.current" in plan.required_fact_keys
    assert "matkap_documents.current" in plan.required_fact_keys


def test_intent_plan_uses_specific_year_discount_key() -> None:
    plan = build_conversation_intent_plan(
        current_message="Если оплатить сразу за год, какая скидка?",
        active_brand="unpk",
    )

    assert plan.primary_intent == "discount"
    assert plan.required_fact_keys[0] == "discounts_year.current"
    assert "discounts.current" in plan.required_fact_keys


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


def test_intent_plan_scopes_payment_method_away_from_office_and_camp_facts() -> None:
    plan = build_conversation_intent_plan(
        current_message="Можно оплатить по счёту банковским переводом?",
        active_brand="foton",
    )

    assert plan.primary_intent == "payment_method"
    assert plan.fact_scope == "payment_methods"
    assert "office_hours" in plan.blocked_neighbor_scopes
    assert "camp_extra_facts" in plan.blocked_neighbor_scopes


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
    assert plan.route_bias == "bot_answer_self_for_pilot"


def test_intent_plan_emits_selling_price_objection_and_exit_without_extra_call() -> None:
    objection = build_conversation_intent_plan(
        current_message="Дороговато, надо подумать. Можно как-то подешевле или частями?",
        active_brand="foton",
        known_slots={"grade": "9", "subject": "физика", "format": "онлайн"},
    )
    plain_price = build_conversation_intent_plan(
        current_message="Сколько стоит онлайн для 9 класса?",
        active_brand="foton",
        known_slots={"grade": "9", "subject": "физика", "format": "онлайн"},
    )

    assert objection.selling["objection"] == "price"
    assert objection.selling["exit_signal"] is True
    assert objection.selling["anxiety"] is False
    assert objection.selling["unmet_need"] == ""
    assert objection.selling["readiness"] == "exploring"
    assert objection.to_prompt_view()["selling"]["objection"] == "price"

    assert plain_price.primary_intent == "pricing"
    assert plain_price.selling["objection"] == "none"
    assert plain_price.selling["exit_signal"] is False


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
