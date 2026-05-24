from __future__ import annotations

from mango_mvp.channels.answer_quality_rewriter import (
    assess_answer_quality,
    apply_answer_quality_rewriter,
    build_answer_quality_llm_rewrite_prompt,
)
from mango_mvp.channels.subscription_llm import SubscriptionDraftResult


def _context(**extra):
    payload = {
        "active_brand": "foton",
        "client_safe_fact_verified": True,
        "confirmed_facts": {"fact:price": "Фотон 8 класс физика онлайн: текущая цена сейчас 74 500 ₽ за год."},
        "known_slots": {"grade": "8", "subject": "физика", "format": "онлайн"},
        "funnel_state": {"filled_slots": {"grade": "8", "subject": "физика", "format": "онлайн"}},
    }
    payload.update(extra)
    return payload


def test_assess_flags_price_fixation_followup_ignored_question() -> None:
    result = SubscriptionDraftResult(
        route="draft_for_manager",
        topic_id="theme:001_pricing",
        topic_confidence=0.94,
        draft_text="Стоимость зависит от класса, формата и периода оплаты. Менеджер проверит актуальную стоимость.",
    )

    assessment = assess_answer_quality(
        result,
        client_message="Это цена прямо на сейчас? Можно зафиксировать?",
        context=_context(),
    )

    codes = {finding.code for finding in assessment.findings}
    assert "ignored_direct_question" in codes
    assert "generic_price_template_after_slots_known" in codes
    assert "over_handoff_with_verified_fact" in codes
    assert assessment.needs_rewrite is True


def test_assess_uses_dialogue_memory_slots_when_top_level_slots_missing() -> None:
    result = SubscriptionDraftResult(
        route="draft_for_manager",
        topic_id="theme:001_pricing",
        topic_confidence=0.94,
        draft_text="Стоимость зависит от класса, формата и периода оплаты. Менеджер проверит актуальную стоимость.",
    )

    assessment = assess_answer_quality(
        result,
        client_message="Это цена прямо сейчас?",
        context={
            "active_brand": "foton",
            "client_safe_fact_verified": True,
            "confirmed_facts": {"fact:price": "Фотон 8 класс физика онлайн: текущая цена сейчас 74 500 ₽ за год."},
            "dialogue_memory_view": {
                "known_slots": {"grade": "8", "subject": "физика", "format": "онлайн"},
                "open_question": {"text": "Это цена прямо сейчас?", "kind": "price_fix", "answered": False},
            },
        },
    )

    assert assessment.known_slots["grade"] == "8"
    assert assessment.direct_question == "Это цена прямо сейчас?"
    assert "generic_price_template_after_slots_known" in {finding.code for finding in assessment.findings}


def test_rewrite_price_fixation_uses_verified_current_price_without_promise() -> None:
    result = SubscriptionDraftResult(
        route="draft_for_manager",
        topic_id="theme:001_pricing",
        topic_confidence=0.94,
        draft_text="Стоимость зависит от класса, формата и периода оплаты. Менеджер проверит актуальную стоимость.",
    )

    rewritten = apply_answer_quality_rewriter(
        result,
        client_message="Это цена прямо на сейчас? Можно зафиксировать?",
        context=_context(),
    )

    assert "текущая подтверждённая цена" in rewritten.draft_text
    assert "74 500 ₽" in rewritten.draft_text
    assert "как оформить по текущим условиям" in rewritten.draft_text
    assert "закрепил" not in rewritten.draft_text.casefold()
    assert "22 мая" not in rewritten.draft_text
    assert "answer_quality_rewritten" in rewritten.safety_flags


def test_rewrite_current_price_answers_direct_question_first() -> None:
    result = SubscriptionDraftResult(
        route="draft_for_manager",
        topic_id="theme:001_pricing",
        topic_confidence=0.94,
        draft_text="Стоимость зависит от класса, формата и периода оплаты. Менеджер проверит актуальную стоимость.",
    )

    rewritten = apply_answer_quality_rewriter(
        result,
        client_message="Это цена прямо на сейчас?",
        context=_context(),
    )

    first_sentence = rewritten.draft_text.split(".", 1)[0]
    first_two_sentences = ".".join(rewritten.draft_text.split(".")[:2])
    assert first_sentence == "Да, это текущая подтверждённая цена на сейчас"
    assert "74 500 ₽" in first_two_sentences
    assert not rewritten.draft_text.startswith("Стоимость зависит")


def test_rewrite_price_fixation_process_is_safe_without_booking_or_seat_promise() -> None:
    result = SubscriptionDraftResult(
        route="draft_for_manager",
        topic_id="theme:001_pricing",
        topic_confidence=0.94,
        draft_text="Менеджер подскажет, как оформить.",
    )

    rewritten = apply_answer_quality_rewriter(
        result,
        client_message="Как зафиксировать цену и оформить по текущей?",
        context=_context(),
    )

    lowered = rewritten.draft_text.casefold()
    assert rewritten.draft_text.startswith("Чтобы оформить по текущим условиям")
    assert "передать заявку" in lowered
    assert "менеджер проверит группу" in lowered
    assert "price_fixation_process_needs_manager_confirmation" in rewritten.missing_facts
    assert "заброниру" not in lowered
    assert "бронь" not in lowered
    assert "места есть" not in lowered
    assert "закрепил" not in lowered
    assert "закреплю" not in lowered
    assert "записал" not in lowered
    assert "запишу" not in lowered
    assert "гарантир" not in lowered


def test_rewrite_price_fixation_process_understands_what_do_i_need() -> None:
    result = SubscriptionDraftResult(
        route="draft_for_manager",
        topic_id="theme:001_pricing",
        topic_confidence=0.94,
        draft_text="Да, это текущая цена. Если хотите, передам менеджеру.",
    )

    rewritten = apply_answer_quality_rewriter(
        result,
        client_message="Что от меня нужно, чтобы зафиксировать год за 74 500?",
        context=_context(),
    )

    lowered = rewritten.draft_text.casefold()
    assert rewritten.draft_text.startswith("Чтобы оформить по текущим условиям")
    assert "передать заявку" in lowered
    assert "74 500" in rewritten.draft_text
    assert "заброниру" not in lowered
    assert "закреплю" not in lowered


def test_rewrite_price_fact_cleans_source_labels_and_normalizes_online_format() -> None:
    result = SubscriptionDraftResult(
        route="draft_for_manager",
        topic_id="theme:001_pricing",
        topic_confidence=0.94,
        draft_text="Стоимость зависит от класса, формата и периода оплаты. Менеджер проверит актуальную стоимость.",
    )

    rewritten = apply_answer_quality_rewriter(
        result,
        client_message="Это цена на сейчас? и она точно не поменяется, если сегодня решу?",
        context=_context(
            confirmed_facts={},
            known_slots={"grade": "8", "subject": "физика", "format": "online"},
            knowledge_snippets=[
                "prices regular 2026 27 / early booking 2026 27 / online 5 11 semester: Фотон: цены на 2026/27 учебный год, 5-11 класс, онлайн — 29 750 ₽.",
                "prices regular 2026 27 / early booking 2026 27 / online 5 11 year: Фотон: цены на 2026/27 учебный год, 5-11 класс, онлайн — 47 250 ₽.",
                "prices regular 2026 27 / offline 5 11 year: Фотон: цены на 2026/27 учебный год, 5-11 класс, очно — 74 500 ₽.",
            ],
        ),
    )

    assert "prices regular" not in rewritten.draft_text
    assert "47 250 ₽" in rewritten.draft_text
    assert "29 750 ₽" in rewritten.draft_text
    assert "offline" not in rewritten.draft_text
    assert "(8 класс, физика, онлайн)" in rewritten.draft_text
    assert "74 500" not in rewritten.draft_text


def test_assess_flags_offline_price_when_context_is_online() -> None:
    result = SubscriptionDraftResult(
        route="draft_for_manager",
        topic_id="theme:001_pricing",
        topic_confidence=0.94,
        draft_text="Да, это текущая подтверждённая цена на сейчас. очно, семестр — 44 600 ₽. очно, год — 74 500 ₽.",
    )

    assessment = assess_answer_quality(
        result,
        client_message="Это цена на сейчас? Я же онлайн спрашиваю.",
        context=_context(
            known_slots={"grade": "8", "subject": "физика", "format": "онлайн"},
            knowledge_snippets=[
                "prices regular 2026 27 / online 5 11 semester: Фотон: цены на 2026/27 учебный год, 5-11 класс, онлайн — 29 750 ₽.",
                "prices regular 2026 27 / online 5 11 year: Фотон: цены на 2026/27 учебный год, 5-11 класс, онлайн — 47 250 ₽.",
            ],
        ),
    )

    assert "wrong_scope_fact_selected" in {finding.code for finding in assessment.findings}


def test_rewrite_installment_does_not_repeat_availability_when_client_corrects_topic() -> None:
    result = SubscriptionDraftResult(
        route="draft_for_manager",
        topic_id="theme:006_installment",
        topic_confidence=0.94,
        draft_text="По местам не буду обещать без проверки. Передам менеджеру, чтобы он проверил наличие по конкретной группе.",
    )

    rewritten = apply_answer_quality_rewriter(
        result,
        client_message="Я про оплату спрашивала, не про места. Можно помесячно или за семестр? Долями можно?",
        context=_context(confirmed_facts={"fact:installment": "Фотон: доступны 6, 10 или 12 месяцев и сервис Долями."}),
    )

    lowered = rewritten.draft_text.casefold()
    assert "6, 10 или 12 месяцев" in rewritten.draft_text
    assert "Долями" in rewritten.draft_text
    assert "помесячную оплату" in lowered
    assert "по местам" not in lowered


def test_rewrite_booking_without_payment_answers_directly_before_price() -> None:
    result = SubscriptionDraftResult(
        route="draft_for_manager",
        topic_id="theme:004_payment_schedule",
        topic_confidence=0.82,
        draft_text="Менеджер подскажет условия оформления.",
    )

    rewritten = apply_answer_quality_rewriter(
        result,
        client_message="А что значит закрепить, надо сразу платить или можно бронь без оплаты?",
        context=_context(
            confirmed_facts={},
            known_slots={"grade": "8", "subject": "физика", "format": "online"},
            knowledge_snippets=[
                "prices regular 2026 27 / early booking 2026 27 / online 5 11 year: Фотон: цены на 2026/27 учебный год, 5-11 класс, онлайн — 47 250 ₽.",
            ],
        ),
    )

    assert rewritten.draft_text.startswith("Бронь или фиксацию условий без оформления я не буду обещать")
    assert "без созвона" in rewritten.draft_text
    assert "prices regular" not in rewritten.draft_text
    assert "booking_without_payment_policy" in rewritten.missing_facts


def test_rewrite_foton_installment_answers_bank_interest_and_dolyami_parts() -> None:
    result = SubscriptionDraftResult(
        route="draft_for_manager",
        topic_id="theme:006_installment",
        topic_confidence=0.94,
        draft_text="Менеджер подскажет условия оплаты.",
    )

    rewritten = apply_answer_quality_rewriter(
        result,
        client_message="Это через банк? Там проценты есть? Долями можно?",
        context=_context(confirmed_facts={"fact:installment": "Фотон: доступны 6, 10 или 12 месяцев и сервис Долями."}),
    )

    assert "6, 10 или 12 месяцев" in rewritten.draft_text
    assert "Долями" in rewritten.draft_text
    assert "одобрение я не обещаю" in rewritten.draft_text
    assert "лагер" not in rewritten.draft_text.casefold()


def test_rewrite_unpk_installment_second_turn_not_repeated_verbatim() -> None:
    result = SubscriptionDraftResult(
        route="draft_for_manager",
        topic_id="theme:006_installment",
        topic_confidence=0.94,
        draft_text="В УНПК можно платить помесячно, за семестр или за год.",
    )

    rewritten = apply_answer_quality_rewriter(
        result,
        client_message="То есть банк не нужен и одобрения нет?",
        context=_context(
            active_brand="unpk",
            confirmed_facts={"fact:installment": "УНПК: рассрочки нет; оплата помесячно, за семестр или за год."},
        ),
    )

    assert "банк не нужен" in rewritten.draft_text
    assert "не банковская рассрочка" in rewritten.draft_text
    assert "Долями" not in rewritten.draft_text
    assert "Фотон" not in rewritten.draft_text


def test_rewrite_unpk_installment_monthly_discount_delta() -> None:
    result = SubscriptionDraftResult(
        route="draft_for_manager",
        topic_id="theme:006_installment",
        topic_confidence=0.94,
        draft_text="В УНПК рассрочки нет, зато можно платить помесячно, за семестр или за год. "
        "При оплате за семестр действует скидка 10%, за год — 14%. Менеджер поможет выбрать удобный вариант оплаты.",
    )

    rewritten = apply_answer_quality_rewriter(
        result,
        client_message="А если помесячно, скидка сохраняется?",
        context=_context(
            active_brand="unpk",
            confirmed_facts={"fact:installment": "УНПК: помесячно без скидки, семестр 10%, год 14%."},
            dialogue_memory_view={
                "recent_turns": [
                    {
                        "role": "bot",
                        "text": "В УНПК рассрочки нет, зато можно платить помесячно, за семестр или за год. "
                        "При оплате за семестр действует скидка 10%, за год — 14%. Менеджер поможет выбрать удобный вариант оплаты.",
                    }
                ]
            },
        ),
    )

    assert rewritten.draft_text.startswith("Да, помесячно платить можно")
    assert "Скидка при этом не применяется" in rewritten.draft_text
    assert rewritten.draft_text.count("В УНПК рассрочки нет") == 0
    assert "safe_template_repeated_across_turns" in rewritten.metadata["answer_quality"]["finding_codes"]


def test_rewrite_foton_installment_answers_no_interest_followup_with_delta() -> None:
    result = SubscriptionDraftResult(
        route="draft_for_manager",
        topic_id="theme:006_installment",
        topic_confidence=0.94,
        draft_text="Да, в Фотоне можно оплатить обучение частями: есть варианты на 6, 10 или 12 месяцев и сервис Долями. "
        "Менеджер поможет подобрать удобный вариант и оформить его дистанционно.",
    )

    rewritten = apply_answer_quality_rewriter(
        result,
        client_message="То есть без процентов для клиента?",
        context=_context(
            active_brand="foton",
            confirmed_facts={
                "fact:installment": "Фотон: рассрочка без переплаты для клиента; варианты оплаты частями 6, 10 или 12 месяцев и Долями."
            },
            dialogue_memory_view={
                "recent_turns": [
                    {
                        "role": "bot",
                        "text": "Да, в Фотоне можно оплатить обучение частями: есть варианты на 6, 10 или 12 месяцев и сервис Долями. "
                        "Менеджер поможет подобрать удобный вариант и оформить его дистанционно.",
                    }
                ]
            },
        ),
    )

    assert "без переплаты для клиента" in rewritten.draft_text
    assert "6, 10 или 12 месяцев" in rewritten.draft_text
    assert "Долями" in rewritten.draft_text
    assert rewritten.draft_text.count("Да, в Фотоне можно оплатить обучение частями") == 0


def test_rewrite_foton_dolyami_parts_does_not_reintroduce_old_four_parts() -> None:
    result = SubscriptionDraftResult(
        route="draft_for_manager",
        topic_id="theme:006_installment",
        topic_confidence=0.94,
        draft_text="Да, в Фотоне можно оплатить обучение частями: есть варианты на 6, 10 или 12 месяцев и сервис Долями.",
    )

    rewritten = apply_answer_quality_rewriter(
        result,
        client_message="А Долями на сколько частей?",
        context=_context(
            active_brand="foton",
            confirmed_facts={
                "fact:installment": "Фотон: доступны варианты оплаты частями на 6, 10 или 12 месяцев и сервис Долями."
            },
        ),
    )

    assert "Долями в Фотоне доступен" in rewritten.draft_text
    assert "4 части" not in rewritten.draft_text
    assert "6, 10 или 12 месяцев" in rewritten.draft_text
    assert "платёжный сервис" in rewritten.draft_text


def test_rewrite_repeated_template_to_short_ack_when_client_is_thinking() -> None:
    repeated_text = (
        "Да, в Фотоне можно оплатить обучение частями: есть варианты на 6, 10 или 12 месяцев и сервис Долями. "
        "По обычным курсам также можно обсудить помесячную оплату или оплату за семестр. "
        "Менеджер поможет подобрать удобный вариант и оформить его дистанционно."
    )
    result = SubscriptionDraftResult(
        route="draft_for_manager",
        topic_id="theme:006_installment",
        topic_confidence=0.94,
        draft_text=repeated_text,
    )

    rewritten = apply_answer_quality_rewriter(
        result,
        client_message="Спасибо, подумаю",
        context=_context(
            active_brand="foton",
            known_slots={"grade": "8", "subject": "физика", "format": "онлайн"},
            dialogue_memory_view={"recent_turns": [{"role": "bot", "text": repeated_text}]},
        ),
    )

    assert "подумайте спокойно" in rewritten.draft_text
    assert "Повторять условия заново не буду" in rewritten.draft_text
    assert rewritten.draft_text.count("есть варианты на 6, 10 или 12 месяцев") == 0
    assert "repeated_template_replaced_with_delta" in rewritten.missing_facts


def test_assess_flags_wrong_scope_when_online_summer_gets_lvsh_fact() -> None:
    result = SubscriptionDraftResult(
        route="draft_for_manager",
        topic_id="theme:026_camp_general",
        topic_confidence=0.9,
        draft_text="ЛВШ Менделеево УНПК стоит 114 000 ₽, в стоимость входит проживание и питание.",
    )

    assessment = assess_answer_quality(
        result,
        client_message="А есть онлайн по физике на лето без проживания?",
        context=_context(active_brand="unpk"),
    )

    codes = {finding.code for finding in assessment.findings}
    assert "wrong_scope_fact_selected" in codes
    assert "ignored_direct_question" in codes


def test_rewrite_online_summer_not_residential_does_not_repeat_lvsh_price() -> None:
    result = SubscriptionDraftResult(
        route="draft_for_manager",
        topic_id="theme:026_camp_general",
        topic_confidence=0.9,
        draft_text="ЛВШ Менделеево УНПК стоит 114 000 ₽, в стоимость входит проживание и питание.",
    )

    rewritten = apply_answer_quality_rewriter(
        result,
        client_message="А есть онлайн по физике на лето без проживания?",
        context=_context(
            active_brand="unpk",
            known_slots={"grade": "11", "subject": "физика", "format": "онлайн"},
            confirmed_facts={"fact:camp": "УНПК: ЛВШ Менделеево очная с проживанием, текущая цена 114 000 ₽."},
        ),
    )

    assert "именно про онлайн-занятия" in rewritten.draft_text
    assert "114 000" not in rewritten.draft_text
    assert "Менделеево" not in rewritten.draft_text
    assert "11 класс" in rewritten.draft_text
    assert "физика" in rewritten.draft_text
    assert "online_summer_program_needs_manager_check" in rewritten.missing_facts


def test_rewrite_foton_online_trial_answers_remote_no_visit_no_free_promise() -> None:
    result = SubscriptionDraftResult(
        route="draft_for_manager",
        topic_id="theme:023_trial_class",
        topic_confidence=0.91,
        draft_text="Менеджер уточнит формат пробного занятия.",
    )

    rewritten = apply_answer_quality_rewriter(
        result,
        client_message="Онлайн точно, никуда приезжать не надо? Пробное есть?",
        context=_context(confirmed_facts={"fact:trial": "Фотон: пробное занятие есть, оформляется дистанционно при записи."}),
    )

    assert "пробное занятие есть" in rewritten.draft_text
    assert "дистанционно" in rewritten.draft_text
    assert "приезжать не нужно" in rewritten.draft_text
    assert "Напишите класс и предмет" not in rewritten.draft_text
    assert "бесплат" not in rewritten.draft_text.casefold()


def test_rewrite_unpk_online_trial_fragment_keeps_online_context() -> None:
    result = SubscriptionDraftResult(
        route="draft_for_manager",
        topic_id="theme:023_trial_class",
        topic_confidence=0.91,
        draft_text="По очному формату сейчас не начинаем с бесплатного пробного занятия. По онлайн-формату можно прислать фрагмент занятия.",
    )

    rewritten = apply_answer_quality_rewriter(
        result,
        client_message="9 класс информатика онлайн, пришлите фрагмент занятия",
        context=_context(
            active_brand="unpk",
            known_slots={"grade": "9", "subject": "информатика", "format": "онлайн"},
            confirmed_facts={"fact:trial": "УНПК: по онлайн-формату можно прислать фрагмент занятия."},
        ),
    )

    assert "онлайн-формату УНПК" in rewritten.draft_text
    assert "фрагмент занятия" in rewritten.draft_text
    assert "приезжать для этого не нужно" in rewritten.draft_text
    assert "По очному формату" not in rewritten.draft_text


def test_rewrite_seats_question_never_promises_availability_or_booking() -> None:
    result = SubscriptionDraftResult(
        route="draft_for_manager",
        topic_id="theme:026_camp_general",
        topic_confidence=0.91,
        draft_text="Да, подберём смену.",
    )

    rewritten = apply_answer_quality_rewriter(
        result,
        client_message="Места есть? Можно забронировать?",
        context=_context(confirmed_facts={"fact:camp": "Фотон: ЛВШ Менделеево текущая цена 93 100 ₽."}),
    )

    assert "не буду обещать без проверки" in rewritten.draft_text
    assert "места есть" not in rewritten.draft_text.casefold()
    assert "забронирую" not in rewritten.draft_text.casefold()
    assert "забронировать можно" not in rewritten.draft_text.casefold()
    assert "availability_by_group_or_shift" in rewritten.missing_facts


def test_llm_polish_does_not_override_live_availability_answer() -> None:
    result = SubscriptionDraftResult(
        route="draft_for_manager",
        topic_id="theme:026_camp_general",
        topic_confidence=0.91,
        draft_text="Да, сориентирую по проверенной информации. Фотон: ЛВШ Менделеево — 5-10 класс.",
    )
    calls = []

    rewritten = apply_answer_quality_rewriter(
        result,
        client_message="А места на физику для 8 класса ещё есть?",
        context=_context(
            known_slots={"grade": "8", "subject": "физика", "format": "очно"},
            confirmed_facts={"fact:camp": "Фотон: ЛВШ Менделеево рассчитана на 5-10 класс, есть физика."},
        ),
        rewrite_runner=lambda **kwargs: calls.append(kwargs) or {"draft_text": "Если напишете класс ребёнка и задачу, поможем подобрать подходящий вариант."},
        force_llm_polish=True,
    )

    assert calls == []
    assert "8 класс" in rewritten.draft_text
    assert "физика" in rewritten.draft_text
    assert "не буду обещать без проверки" in rewritten.draft_text
    assert "Если напишете класс" not in rewritten.draft_text
    assert "availability_by_group_or_shift" in rewritten.missing_facts


def test_multitopic_transport_and_booking_answers_safe_part_first() -> None:
    result = SubscriptionDraftResult(
        route="draft_for_manager",
        topic_id="theme:028_transport_logistics",
        topic_confidence=0.91,
        draft_text="Передам менеджеру, он подскажет.",
    )

    rewritten = apply_answer_quality_rewriter(
        result,
        client_message="а трансфер из Москвы есть? и можно как-то закрепить место пока менеджер проверит?",
        context=_context(
            known_slots={"grade": "8", "subject": "физика", "product": "ЛВШ"},
            funnel_state={"filled_slots": {"grade": "8", "subject": "физика", "product": "ЛВШ"}},
            confirmed_facts={
                "fact:transfer": "Трансфер до ЛВШ Фотона включён в стоимость; ориентир места сбора — метро Ховрино, точные детали отправляем перед сменой."
            },
        ),
        rewrite_runner=lambda **kwargs: {"draft_text": "bad"},
        force_llm_polish=True,
    )

    assert "Трансфер" in rewritten.draft_text
    assert "включён" in rewritten.draft_text or "включен" in rewritten.draft_text
    assert "не буду обещать" in rewritten.draft_text
    assert "мест" in rewritten.draft_text
    assert "8 класс" in rewritten.draft_text
    assert "физика" in rewritten.draft_text
    assert "онлайн" not in rewritten.draft_text
    assert "Подскажите класс" not in rewritten.draft_text
    assert "booking_without_payment_policy" in rewritten.missing_facts


def test_assess_multitopic_flags_single_topic_answer() -> None:
    result = SubscriptionDraftResult(
        route="draft_for_manager",
        topic_id="theme:006_installment",
        topic_confidence=0.91,
        draft_text="В Фотоне можно оплатить частями на 6, 10 или 12 месяцев.",
    )

    assessment = assess_answer_quality(
        result,
        client_message="Сколько стоит онлайн на год? Есть скидка на второй предмет? Можно частями? Занятия в прямом эфире или записи?",
        context=_context(),
    )

    assert "single_topic_answer_to_multitopic_question" in {finding.code for finding in assessment.findings}


def test_assess_flags_templated_opening_as_rewrite() -> None:
    result = SubscriptionDraftResult(
        route="draft_for_manager",
        topic_id="service:S5_general_consultation",
        topic_confidence=0.8,
        draft_text="Здравствуйте! Помогу подобрать оптимальный образовательный продукт под ваш запрос.",
    )

    assessment = assess_answer_quality(
        result,
        client_message="Что есть для 6 класса?",
        context=_context(known_slots={"grade": "6"}),
    )

    assert "templated_opening" in {finding.code for finding in assessment.findings}
    assert assessment.needs_rewrite is True


def test_assess_flags_missing_next_step_after_short_fact_answer() -> None:
    result = SubscriptionDraftResult(
        route="draft_for_manager",
        topic_id="theme:014_format",
        topic_confidence=0.86,
        draft_text="Занятия проходят онлайн на платформе МТС Линк.",
    )

    assessment = assess_answer_quality(
        result,
        client_message="Где проходят онлайн-занятия?",
        context=_context(confirmed_facts={"fact:platform": "Фотон: онлайн-занятия проходят на МТС Линк."}),
    )

    assert "missing_next_step" in {finding.code for finding in assessment.findings}


def test_assess_flags_answered_nearby_topic_for_camp_contents() -> None:
    result = SubscriptionDraftResult(
        route="draft_for_manager",
        topic_id="theme:027_camp_living_conditions",
        topic_confidence=0.9,
        draft_text="ЛВШ Менделеево у Фотона сейчас стоит 93 100 ₽.",
    )

    assessment = assess_answer_quality(
        result,
        client_message="Что входит в лагерь?",
        context=_context(confirmed_facts={"fact:camp": "Фотон: в ЛВШ есть проживание, питание и трансфер."}),
    )

    assert "ignored_direct_question" in {finding.code for finding in assessment.findings}


def test_llm_rewrite_is_not_called_when_assessment_passed_even_with_polish_mode() -> None:
    result = SubscriptionDraftResult(
        route="draft_for_manager",
        topic_id="theme:023_trial_class",
        topic_confidence=0.91,
        draft_text="Да, в онлайн-формате Фотона пробное занятие есть по умолчанию. Если хотите, передам менеджеру запрос на онлайн-пробное.",
    )
    calls = []

    rewritten = apply_answer_quality_rewriter(
        result,
        client_message="Пробное онлайн есть?",
        context=_context(known_slots={"format": "онлайн"}, confirmed_facts={"fact:trial": "Фотон: онлайн-пробное есть."}),
        rewrite_runner=lambda **kwargs: calls.append(kwargs) or {"draft_text": "bad"},
        force_llm_polish=True,
    )

    assert calls == []
    assert rewritten.draft_text == result.draft_text


def test_llm_rewrite_is_rejected_when_post_rewrite_reasks_known_data() -> None:
    result = SubscriptionDraftResult(
        route="draft_for_manager",
        topic_id="theme:001_pricing",
        topic_confidence=0.94,
        draft_text="Стоимость зависит от класса, формата и периода оплаты. Менеджер проверит актуальную стоимость.",
    )

    rewritten = apply_answer_quality_rewriter(
        result,
        client_message="Это цена прямо на сейчас?",
        context=_context(),
        rewrite_runner=lambda **_: {
            "draft_text": "Подскажите, пожалуйста, какой класс, предмет и формат: очно или онлайн?"
        },
        force_llm_polish=True,
    )

    aq = rewritten.metadata["answer_quality"]
    assert aq["rewrite_provider"] == "llm_runner"
    assert aq["rewritten"] is False
    assert aq["rewrite_rejected"] is True
    assert "reasked_known_grade" in aq["rewrite_rejection_reasons"]


def test_assess_flags_reasking_known_class_subject_and_format() -> None:
    result = SubscriptionDraftResult(
        route="draft_for_manager",
        topic_id="theme:001_pricing",
        topic_confidence=0.91,
        draft_text="Подскажите, какой класс, какой предмет интересует и какой формат: очно или онлайн?",
    )

    assessment = assess_answer_quality(
        result,
        client_message="Сколько стоит обучение?",
        context=_context(
            known_slots={},
            dialogue_memory_view={
                "known_slots": {"grade": "8", "subject": "физика", "format": "онлайн"},
                "open_question": {"text": "Сколько стоит обучение?", "kind": "price", "answered": False},
            },
        ),
    )

    codes = {finding.code for finding in assessment.findings}
    assert "reasked_known_grade" in codes
    assert "reasked_known_subject" in codes
    assert "reasked_known_format" in codes
    assert assessment.needs_rewrite is True


def test_rewriter_does_not_touch_p0_manager_only() -> None:
    result = SubscriptionDraftResult(
        route="manager_only",
        topic_id="theme:009_refund",
        topic_confidence=0.95,
        draft_text="Приняли обращение. Передам ответственному сотруднику, он вернется с ответом.",
        safety_flags=("high_risk_manager_only",),
    )

    rewritten = apply_answer_quality_rewriter(
        result,
        client_message="Хочу вернуть деньги и пойду в суд.",
        context=_context(),
    )

    assert rewritten.draft_text == result.draft_text
    assert rewritten.route == "manager_only"
    assert "answer_quality_rewritten" not in rewritten.safety_flags
    assert rewritten.metadata["answer_quality"]["rewritten"] is False


def test_p0_client_text_does_not_call_llm_rewriter_even_when_enabled() -> None:
    result = SubscriptionDraftResult(
        route="draft_for_manager",
        topic_id="theme:001_pricing",
        topic_confidence=0.92,
        draft_text="Менеджер подскажет актуальные условия.",
    )
    calls = []

    rewritten = apply_answer_quality_rewriter(
        result,
        client_message="Хочу вернуть деньги, иначе пойду в суд.",
        context=_context(),
        rewrite_runner=lambda **kwargs: calls.append(kwargs) or {"draft_text": "Опасный переписанный ответ."},
        force_llm_polish=True,
    )

    assert calls == []
    assert rewritten.draft_text == result.draft_text
    assert rewritten.route == result.route
    assert "answer_quality_rewritten" not in rewritten.safety_flags
    assert rewritten.metadata["answer_quality"]["rewritten"] is False


def test_assess_flags_assumed_unstated_need_as_rewrite() -> None:
    result = SubscriptionDraftResult(
        route="draft_for_manager",
        topic_id="service:S5_general_consultation",
        topic_confidence=0.7,
        draft_text="Вам подойдёт информатика: можно начать с онлайн-группы и потом перейти в очный формат.",
    )

    assessment = assess_answer_quality(
        result,
        client_message="Здравствуйте, хочу понять, что у вас есть для 6 класса.",
        context=_context(known_slots={"grade": "6"}),
    )

    codes = {finding.code for finding in assessment.findings}
    assert "assumed_unstated_need" in codes
    assert assessment.needs_rewrite is True


def test_llm_rewrite_runner_is_used_only_when_deterministic_cannot_fix() -> None:
    result = SubscriptionDraftResult(
        route="draft_for_manager",
        topic_id="service:S5_general_consultation",
        topic_confidence=0.7,
        draft_text="Вам подойдёт информатика: можно начать с онлайн-группы и потом перейти в очный формат.",
    )
    calls = []

    def runner(**kwargs):
        calls.append(kwargs)
        return {
            "draft_text": "Для 6 класса можем подобрать подходящее направление, но предмет лучше выбрать от вашей цели. Напишите, что важнее: подтянуть школу или попробовать олимпиадный уровень?"
        }

    rewritten = apply_answer_quality_rewriter(
        result,
        client_message="Здравствуйте, хочу понять, что у вас есть для 6 класса.",
        context=_context(known_slots={"grade": "6"}),
        rewrite_runner=runner,
    )

    assert calls
    assert rewritten.metadata["answer_quality"]["rewrite_provider"] == "llm_runner"
    assert rewritten.metadata["answer_quality"]["rewritten"] is True
    assert "answer_quality_rewritten" in rewritten.safety_flags
    assert "post_finding_codes" in rewritten.metadata["answer_quality"]


def test_llm_rewrite_prompt_hides_identity_fields_and_keeps_brand_facts() -> None:
    result = SubscriptionDraftResult(
        route="draft_for_manager",
        topic_id="theme:001_pricing",
        topic_confidence=0.9,
        draft_text="Менеджер подскажет стоимость.",
    )
    assessment = assess_answer_quality(
        result,
        client_message="Сколько стоит онлайн 8 класс физика?",
        context=_context(
            known_slots={"grade": "8", "subject": "физика", "format": "онлайн", "phone": "79092009933"},
            client_identity={"parent_name": "Иванова Анна", "phone": "79092009933"},
        ),
    )

    prompt = build_answer_quality_llm_rewrite_prompt(
        result=result,
        client_message="Сколько стоит онлайн 8 класс физика?",
        context=_context(
            known_slots={"grade": "8", "subject": "физика", "format": "онлайн", "phone": "79092009933"},
            client_identity={"parent_name": "Иванова Анна", "phone": "79092009933"},
        ),
        assessment=assessment,
    )

    assert "74 500 ₽" in prompt
    assert "79092009933" not in prompt
    assert "Иванова" not in prompt


def test_llm_rewrite_is_rejected_when_it_adds_unsupported_precise_claim() -> None:
    result = SubscriptionDraftResult(
        route="draft_for_manager",
        topic_id="service:S5_general_consultation",
        topic_confidence=0.7,
        draft_text="Вам подойдёт информатика: можно начать с онлайн-группы.",
    )

    rewritten = apply_answer_quality_rewriter(
        result,
        client_message="Здравствуйте, хочу понять, что у вас есть для 6 класса.",
        context=_context(known_slots={"grade": "6"}, confirmed_facts={"fact:general": "Фотон: есть курсы для школьников."}),
        rewrite_runner=lambda **_: {"draft_text": "Для 6 класса есть курс за 99 999 ₽, можно начать завтра."},
    )

    aq = rewritten.metadata["answer_quality"]
    assert aq["rewrite_provider"] == "llm_runner"
    assert aq["rewritten"] is False
    assert aq["rewrite_rejected"] is True
    assert "99 999 ₽" not in rewritten.draft_text


def test_rewrite_price_fixation_without_verified_price_gives_safe_partial_answer() -> None:
    result = SubscriptionDraftResult(
        route="draft_for_manager",
        topic_id="theme:001_pricing",
        topic_confidence=0.81,
        draft_text="Менеджер подскажет условия.",
    )

    rewritten = apply_answer_quality_rewriter(
        result,
        client_message="Можно оформить по текущей цене, пока она не выросла?",
        context=_context(
            confirmed_facts={},
            knowledge_snippets=[],
            known_slots={"grade": "8", "subject": "физика"},
            funnel_state={"filled_slots": {"grade": "8", "subject": "физика"}},
        ),
    )

    assert "не буду обещать цену, место или бронь без проверки" in rewritten.draft_text
    assert "формат: очно или онлайн" in rewritten.draft_text
    assert "price_fixation_needs_verified_price_or_manager_check" in rewritten.missing_facts


def test_rewrite_ignored_direct_question_can_use_relevant_verified_fact() -> None:
    result = SubscriptionDraftResult(
        route="draft_for_manager",
        topic_id="theme:008_tax_deduction",
        topic_confidence=0.88,
        draft_text="Менеджер подскажет, какие документы нужны.",
    )

    rewritten = apply_answer_quality_rewriter(
        result,
        client_message="А налоговый вычет у вас можно оформить?",
        context=_context(
            confirmed_facts={
                "fact:tax": "Фотон: можно оформить налоговый вычет до 13% с расходов на обучение; решение принимает ФНС."
            },
            known_slots={"grade": "8", "subject": "физика", "format": "онлайн"},
        ),
    )

    assert rewritten.draft_text.startswith("Сориентирую по проверенным данным")
    assert "налоговый вычет" in rewritten.draft_text
    assert "ФНС" in rewritten.draft_text
    assert "передам менеджеру" in rewritten.draft_text
