from __future__ import annotations

import csv
from pathlib import Path

import pytest

from mango_mvp.quality.non_conversation import (
    LABEL_CONTENTFUL_PROTECTED_LIVE_DIALOGUE,
    LABEL_MANUAL_REVIEW_BORDERLINE_LIVE_CONTEXT,
    LABEL_MANUAL_REVIEW_PROBABLE_NO_LIVE,
    LABEL_NON_CONVERSATION_HIGH_CONFIDENCE,
    blocks_email_from_voice_mail,
    blocks_system_next_step,
    detect_non_conversation_signals,
)


def test_detects_high_confidence_voicemail_artifact() -> None:
    text = (
        "MANAGER:\n"
        "Здравствуйте.\n\n"
        "CLIENT:\n"
        "Звонок был перенаправлен на голосовой почтовый ящик. "
        "Оставьте сообщение после звукового сигнала. Продолжение следует."
    )

    result = detect_non_conversation_signals(text)

    assert result.label == LABEL_NON_CONVERSATION_HIGH_CONFIDENCE
    assert result.should_force_non_conversation is True
    assert result.recommended_call_type == "non_conversation"
    assert result.recommended_contentful is False
    assert "asr_artifact_marker" in result.reason_codes
    assert "no_live_marker" in result.reason_codes


def test_protects_live_service_call_with_email_and_callback_words() -> None:
    text = (
        "MANAGER:\n"
        "Добрый день, я отправлю чек и расписание на почту, а завтра перезвоню, "
        "если ссылка не откроется.\n\n"
        "CLIENT:\n"
        "Да, чек нужен на почту. Оплату внесли, но ссылка на занятие не работает, "
        "пожалуйста помогите с доступом и расписанием на следующую неделю."
    )

    result = detect_non_conversation_signals(text, call_type="service_call", duration_sec=180)

    assert result.label == LABEL_CONTENTFUL_PROTECTED_LIVE_DIALOGUE
    assert result.protected_live_dialogue is True
    assert result.should_force_non_conversation is False
    assert result.requires_manual_review is False


def test_voice_mail_blocks_false_email_channel_and_system_next_step() -> None:
    text = "Голосовая почта: оставьте сообщение после звукового сигнала. Попробуйте перезвонить позднее."

    assert blocks_email_from_voice_mail(text) is True
    assert blocks_system_next_step(text) is True


def test_live_client_callback_phrase_is_not_voicemail_by_itself() -> None:
    text = (
        "MANAGER:\n"
        "Добрый день, удобно сейчас поговорить?\n\n"
        "CLIENT:\n"
        "Алло, да, здравствуйте. Сейчас неудобно, перезвоните позже."
    )

    result = detect_non_conversation_signals(text, call_type="service_call", duration_sec=18)

    assert result.should_force_non_conversation is False
    assert result.system_no_dialogue_phrase is False


def test_abonent_system_phrase_is_not_stripped_as_speaker_label() -> None:
    text = (
        "MANAGER:\n"
        "Здравствуйте! Спасибо, что были с нами.\n\n"
        "CLIENT:\n"
        "Абонент сейчас не может ответить на ваш звонок. Попробуйте перезвонить позднее. "
        "Если вы хотите оставить ему голосовое сообщение, пожалуйста."
    )

    result = detect_non_conversation_signals(text, call_type="sales_call", duration_sec=20)

    assert result.label == LABEL_NON_CONVERSATION_HIGH_CONFIDENCE
    assert result.should_force_non_conversation is True
    assert "system_no_dialogue_phrase" in result.reason_codes


def test_outbound_voicemail_pitch_is_non_conversation_high_confidence() -> None:
    text = (
        "MANAGER:\n"
        "Добрый день, это учебный центр Фотон. Хотели предложить курс подготовки к ЕГЭ по математике, "
        "оставляю информацию, перезвоните нам, пожалуйста.\n\n"
        "CLIENT:\n"
        "Абонент сейчас не может ответить на ваш звонок. Оставьте сообщение после звукового сигнала."
    )

    result = detect_non_conversation_signals(text, call_type="sales_call", duration_sec=44)

    assert result.label == LABEL_NON_CONVERSATION_HIGH_CONFIDENCE
    assert result.should_force_non_conversation is True
    assert result.outbound_voicemail_marker is True
    assert result.recommended_call_type == "non_conversation"
    assert result.recommended_contact_subtype == "outbound_voicemail"
    assert "outbound_voicemail" in result.reason_codes


def test_detects_virtual_secretary_and_third_party_ivr_markers() -> None:
    examples = [
        "CLIENT: На связи я секретарь, временно попросили отвечать на звонки.",
        "CLIENT: Сбербанк, я ваш голосовой помощник. Все разговоры записываются.",
        "CLIENT: Вас приветствует компания Сервис Резерв. Для соединения нажмите 1.",
        "CLIENT: Абонент вне зоны действия сети. Если хотите отправить бесплатное смс, нажмите 1.",
    ]

    for text in examples:
        result = detect_non_conversation_signals(f"MANAGER: Добрый день.\n{text}", duration_sec=20)
        assert result.strong_no_live_marker is True
        assert result.system_no_dialogue_phrase is True
        assert result.label == LABEL_NON_CONVERSATION_HIGH_CONFIDENCE
        assert result.should_force_non_conversation is True
        assert result.recommended_call_type == "non_conversation"


def test_compliance_preamble_with_live_sales_dialogue_is_not_forced_non_conversation() -> None:
    text = (
        "MANAGER:\n"
        "Вас приветствует компания Фотон, все разговоры записываются. "
        "Подскажите, какой курс вам интересен?\n\n"
        "CLIENT:\n"
        "Да, здравствуйте. Интересует курс по математике для 9 класса, "
        "какая стоимость, расписание и можно ли оплатить частями?"
    )

    result = detect_non_conversation_signals(text, call_type="sales_call", duration_sec=120)

    assert result.should_force_non_conversation is False
    assert result.system_no_dialogue_phrase is False
    assert result.protected_live_dialogue is True


def test_pure_ivr_still_forces_non_conversation() -> None:
    text = (
        "MANAGER:\n"
        "Добрый день.\n\n"
        "CLIENT:\n"
        "Вас приветствует компания Сервис Резерв. Для соединения нажмите 1."
    )

    result = detect_non_conversation_signals(text, duration_sec=20)

    assert result.label == LABEL_NON_CONVERSATION_HIGH_CONFIDENCE
    assert result.should_force_non_conversation is True
    assert result.recommended_call_type == "non_conversation"


def test_detects_bank_collector_and_mts_ivr_even_with_business_words() -> None:
    examples = [
        (
            "MANAGER: Динамичная музыка.\n"
            "CLIENT: Здравствуйте, уважаемый клиент, это Сбербанк. Если вы испытываете трудности "
            "с оплатой кредита, нажмите 1."
        ),
        (
            "MANAGER: Алло.\n"
            "CLIENT: Вы позвонили в ООО ПКО АБК, действующее в интересах ООО Хартия. "
            "Пожалуйста, перезвоните нам по телефону."
        ),
        (
            "MANAGER: Абазаева нет такого?\n"
            "CLIENT: Здравствуйте! Вас приветствует группа компаний МТС. Если вас интересует услуга, нажмите 1."
        ),
    ]

    third_party_hits = 0
    for text in examples:
        result = detect_non_conversation_signals(text, call_type="sales_call", duration_sec=30)
        assert result.label == LABEL_NON_CONVERSATION_HIGH_CONFIDENCE
        assert result.should_force_non_conversation is True
        third_party_hits += int("third_party_ivr" in result.reason_codes)
    assert third_party_hits >= 2


def test_third_party_business_dialogue_is_not_forced_as_ivr() -> None:
    text = (
        "MANAGER:\n"
        "Добрый день. На наш учебный центр несколько раз поступают обратные звонки, "
        "хотим понять, почему ваш номер отображается в пропущенных и можно ли убрать его из базы. "
        "Я перечислю номера, которые видим у себя, а вы проверьте, пожалуйста.\n\n"
        "CLIENT:\n"
        "Здравствуйте, ООО ПКО Актив Бизнес Консалт, я вас слышу. По указанным номерам данных в базе нет. "
        "Назовите еще раз последние цифры, мы проверим обращение и передадим ответственному сотруднику. "
        "Если звонки повторятся, попросите клиента обратиться с того номера, на который они приходят."
    )

    result = detect_non_conversation_signals(text, call_type="service_call", duration_sec=150)

    assert result.should_force_non_conversation is False
    assert result.label != LABEL_NON_CONVERSATION_HIGH_CONFIDENCE
    assert "third_party_ivr" not in result.reason_codes
    assert "safeguard_third_party_business_dialogue" in result.reason_codes


def test_bank_payment_mentions_inside_live_dialogue_are_not_third_party_ivr() -> None:
    text = (
        "MANAGER:\n"
        "Здравствуйте, учебный центр МФТИ. Ссылку на оплату пришлем от Альфа-банка, "
        "можно оплатить через мобильный банк или по реквизитам.\n\n"
        "CLIENT:\n"
        "Здравствуйте. Мне Сбербанк не принимает QR-код, пришлите, пожалуйста, КПП или реквизиты. "
        "Ребенок продолжает курс по физике, оплату внесу в воскресенье."
    )

    result = detect_non_conversation_signals(text, call_type="sales_call", duration_sec=180)

    assert result.should_force_non_conversation is False
    assert "third_party_ivr" not in result.reason_codes
    assert result.protected_live_dialogue is True


def test_bridge_system_phrase_inside_live_dialogue_does_not_force_non_conversation() -> None:
    text = (
        "MANAGER:\n"
        "Добрый день, учебный центр МФТИ. Хотели уточнить, удобно ли продолжить обучение "
        "и когда можно обсудить расписание.\n\n"
        "CLIENT:\n"
        "Продолжаем дозваниваться. Оставайтесь на линии. Алло, да, слушаю. "
        "Сейчас неудобно, перезвоните завтра, но курс по математике нам актуален."
    )

    result = detect_non_conversation_signals(text, call_type="sales_call", duration_sec=90)

    assert result.should_force_non_conversation is False
    assert result.protected_live_dialogue is True
    assert "bridge_live_dialogue" in result.reason_codes


def test_detects_repeated_asr_loop_without_client_dialogue() -> None:
    text = (
        "MANAGER:\n"
        "Алло, подскажите, пожалуйста, вам курсы актуальны?\n\n"
        "CLIENT:\n"
        "Папочка попал на меня. Папочка попал на меня. Папочка попал на меня. "
        "Папочка попал на меня. Папочка попал на меня."
    )

    result = detect_non_conversation_signals(text, history_summary="Контакт не состоялся.", call_type="sales_call")

    assert result.label == LABEL_NON_CONVERSATION_HIGH_CONFIDENCE
    assert result.should_force_non_conversation is True
    assert "asr_loop_marker" in result.reason_codes


def test_does_not_force_live_human_response_because_of_audio_artifact() -> None:
    examples = [
        "MANAGER: ЗВОНОК В ДВЕРЬ\nCLIENT: Ольга Сергеевна, слушаю вас.",
        "MANAGER: Алло.\nCLIENT: ЗВОНОК В ДВЕРЬ Говорите, пожалуйста. Здравствуйте.",
        (
            "MANAGER: Здравствуйте, это учебный центр Фотон. У вас будет пара минут?\n"
            "CLIENT: Алло, алло. Да, здравствуйте. Нет, нет, нет, нет."
        ),
    ]

    for text in examples:
        result = detect_non_conversation_signals(text, call_type="unknown", duration_sec=15)
        assert result.should_force_non_conversation is False


def test_manual_review_for_borderline_no_live_with_possible_live_context() -> None:
    text = (
        "MANAGER:\n"
        "Здравствуйте, соединяю с администратором по расписанию.\n\n"
        "CLIENT:\n"
        "Секретарь ответила, что абонент сейчас не может ответить, но клиент ранее "
        "интересовался курсом по математике и просил перезвонить по стоимости."
    )

    result = detect_non_conversation_signals(
        text,
        history_summary="Клиент ранее спрашивал стоимость курса по математике.",
        call_type="sales_call",
    )

    assert result.label in {
        LABEL_MANUAL_REVIEW_PROBABLE_NO_LIVE,
        LABEL_MANUAL_REVIEW_BORDERLINE_LIVE_CONTEXT,
    }
    assert result.should_force_non_conversation is False
    assert result.requires_manual_review is True


def test_negative_non_contentful_history_does_not_create_live_protection() -> None:
    text = (
        "MANAGER:\n"
        "Подскажите, пожалуйста, по поводу курсов.\n\n"
        "CLIENT:\n"
        "Здравствуйте. Я выставляю смартфон, ничего не понимаю."
    )

    result = detect_non_conversation_signals(
        text,
        history_summary=(
            "Разговор не состоялся как содержательная консультация: тема обращения не была раскрыта. "
            "Интерес к конкретному продукту не подтвержден."
        ),
        call_type="non_conversation",
    )

    assert result.label != LABEL_CONTENTFUL_PROTECTED_LIVE_DIALOGUE
    assert result.protected_live_dialogue is False
    assert "negative_non_contentful_context" in result.reason_codes


def test_artifact_tail_after_real_sales_dialogue_is_protected() -> None:
    text = (
        "MANAGER:\n"
        "Предлагаю онлайн-курс по информатике ОГЭ, занятия вторник и четверг, стоимость 41800 за семестр.\n\n"
        "CLIENT:\n"
        "Ребенок в 9 классе, информатика ОГЭ интересует. По времени мешает волейбол, но онлайн можем обсудить. "
        "После 16 часов поговорю с сыном, перезвоните мне, пожалуйста. Абонент временно недоступен. "
        "Продолжение следует."
    )

    result = detect_non_conversation_signals(
        text,
        history_summary=(
            "Клиентка обсуждала обучение ребенка в 9 классе по информатике ОГЭ, стоимость курса, "
            "онлайн-формат и попросила перезвонить после 16:00."
        ),
        call_type="sales_call",
        products=["годовые курсы", "онлайн"],
        subjects=["информатика"],
        objections=["время"],
        duration_sec=240,
    )

    assert result.label == LABEL_CONTENTFUL_PROTECTED_LIVE_DIALOGUE
    assert result.protected_live_dialogue is True
    assert result.should_force_non_conversation is False


def test_transfer_to_voicemail_after_live_customer_consent_requires_manual_review() -> None:
    text = (
        "MANAGER:\n"
        "Алексей, здравствуйте. С вами ранее общались по поводу онлайн-обучения. "
        "Удобно говорить сейчас? Хотела соединить вас с коллегами по онлайн-обучению, "
        "чтобы они вас проконсультировали. Оставайтесь на линии, благодарю.\n\n"
        "CLIENT:\n"
        "Алло. Да, здравствуйте. Да-да, я слушаю вас. Хорошо, давайте. "
        "Абонент не отвечает или временно недоступен. Попробуйте перезвонить позднее."
    )

    result = detect_non_conversation_signals(
        text,
        history_summary=(
            "Клиент подтвердил, что удобно говорить, и согласился на переключение "
            "к коллегам по онлайн-обучению."
        ),
        call_type="sales_call",
        duration_sec=64,
    )

    assert result.should_force_non_conversation is False
    assert result.requires_manual_review is True
    assert "safeguard_transfer_after_live_dialogue" in result.reason_codes


def test_third_party_ivr_tail_after_live_service_dialogue_requires_manual_review() -> None:
    text = (
        "MANAGER:\n"
        "Добрый день, это учебный центр НПК МФТИ. Недавно нам звонили по поводу чека. "
        "Документ нужен хотя бы в электронном виде, можно передать через менеджера.\n\n"
        "CLIENT:\n"
        "Вас приветствует компания Almin Provision Service. Для соединения с оператором нажмите 0. "
        "Наталья, добрый день. Я могу скан скинуть, просто не знаю, на какой номер или почту. "
        "Сейчас через менеджера попробую узнать, с кем она общается, и тогда напрямую скину."
    )

    result = detect_non_conversation_signals(
        text,
        history_summary=(
            "Обсуждали передачу недостающего чека/скана. Клиент сообщил, что может "
            "отправить скан, но не знает, на какой номер или почту."
        ),
        call_type="service_call",
        duration_sec=130,
    )

    assert result.should_force_non_conversation is False
    assert result.requires_manual_review is True
    assert "safeguard_third_party_ivr_after_live" in result.reason_codes


def test_short_live_opt_out_from_site_application_is_not_auto_non_conversation() -> None:
    text = (
        "MANAGER:\n"
        "Здравствуйте, Арсений. Это учебный центр. Вы у нас на сайте регистрацию прошли, "
        "заявку оставили. Хотели уточнить, чем можем помочь. Не надо. Ошибочно? "
        "Да. Спасибо за обратную связь. До свидания.\n\n"
        "CLIENT:\n"
        "Мам... Мам... Мам..."
    )

    result = detect_non_conversation_signals(
        text,
        history_summary=(
            "Связались не с целевым абонентом: на звонок ответил ребенок, "
            "контакт подтвердили как ошибочный."
        ),
        call_type="existing_client_progress",
        duration_sec=28,
    )

    assert result.should_force_non_conversation is False
    assert result.requires_manual_review is True
    assert "safeguard_live_opt_out" in result.reason_codes


def test_ambiguous_service_callback_with_asr_junk_is_manual_not_auto_apply() -> None:
    text = (
        "MANAGER:\n"
        "Добрый день, учебный центр, от вас пропущенный вызов. Скажите, чем могу помочь? "
        "Ничего страшного, удачного дня, до свидания.\n\n"
        "CLIENT:\n"
        "Алик, а? А, я не чай, но... Я не чай, но..."
    )

    result = detect_non_conversation_signals(
        text,
        history_summary="Обработан пропущенный входящий вызов. Клиент ответил неразборчиво.",
        call_type="service_call",
        duration_sec=14,
    )

    assert result.should_force_non_conversation is False
    assert result.requires_manual_review is True
    assert "safeguard_ambiguous_service_attempt" in result.reason_codes


def test_short_live_answer_under_a_manager_no_contact_phrase_is_not_auto_closed() -> None:
    """ТЗ 2026-08-22: brevity plus a manager-side no-contact phrase is not proof.

    The client really answered and carries no IVR/voicemail/secretary marker, so
    the row goes to review instead of being rewritten to non_conversation.
    """
    text = (
        "MANAGER:\n"
        "Здравствуйте, вчера не удалось дозвониться до вас.\n\n"
        "CLIENT:\n"
        "Да, здравствуйте, слушаю."
    )

    result = detect_non_conversation_signals(text, call_type="sales_call", duration_sec=18)

    assert result.client_human_response is True
    assert result.should_force_non_conversation is False
    assert result.requires_manual_review is True
    assert result.recommended_call_type is None


@pytest.mark.parametrize("reply", ["Перезвоните завтра", "Я сейчас занята"])
def test_plain_multiword_client_reply_is_not_auto_closed(reply: str) -> None:
    text = f"MANAGER:\nЗдравствуйте.\n\nCLIENT:\n{reply}."

    result = detect_non_conversation_signals(text, duration_sec=12)

    assert result.client_human_response is True
    assert result.should_force_non_conversation is False
    assert result.requires_manual_review is True
    assert result.recommended_contact_subtype == "live_contact_without_business_content"


def test_unattributed_tracks_with_short_action_reply_are_not_auto_closed() -> None:
    result = detect_non_conversation_signals(
        "CHANNEL_LEFT:\nАлло, здравствуйте.\n\nCHANNEL_RIGHT:\nПерезвоните завтра."
    )

    assert result.should_force_non_conversation is False
    assert "explicit_short_action_response" in result.reason_codes


def test_manager_only_short_phrase_is_not_invented_as_a_client_reply() -> None:
    result = detect_non_conversation_signals(
        "CHANNEL_LEFT:\nХорошо.\n\nCHANNEL_RIGHT:\n"
    )

    assert result.client_human_response is False
    assert "explicit_short_action_response" not in result.reason_codes


def test_long_client_dialogue_does_not_get_the_short_reply_reason() -> None:
    result = detect_non_conversation_signals(
        "MANAGER:\nРасскажите, пожалуйста.\n\nCLIENT:\n"
        "Хорошо, ребёнок уже учится у вас, и сейчас я уточняю расписание, "
        "доступ к материалам и время следующего занятия с преподавателем."
    )

    assert "explicit_short_action_response" not in result.reason_codes


def test_a_voicemail_answering_for_the_client_is_still_closed_automatically() -> None:
    """NEG: the relaxation must not reach a call the client never joined."""
    text = (
        "MANAGER:\n"
        "Здравствуйте, вчера не удалось дозвониться до вас.\n\n"
        "CLIENT:\n"
        "Абонент сейчас не может ответить на ваш звонок. "
        "Оставьте сообщение после звукового сигнала."
    )

    result = detect_non_conversation_signals(text, call_type="sales_call", duration_sec=18)

    assert result.client_human_response is False
    assert result.label == LABEL_NON_CONVERSATION_HIGH_CONFIDENCE
    assert result.should_force_non_conversation is True


def test_multiword_asr_artifact_is_not_treated_as_a_live_client_reply() -> None:
    result = detect_non_conversation_signals(
        "MANAGER:\nАлло.\n\nCLIENT:\nПродолжение следует. Спасибо за просмотр.",
        duration_sec=12,
    )

    assert result.client_human_response is False
    assert result.should_force_non_conversation is True


@pytest.mark.parametrize(
    "system_reply",
    [
        "Алло, с вами говорит виртуальный ассистент.",
        "Алло, абонент сейчас никак не сможет взять трубку.",
        "Телефон звонит. Соединение установлено, пожалуйста подождите. "
        "Здравствуйте, с вами говорит автоответчик.",
    ],
)
def test_virtual_assistant_and_nikak_ne_smozhet_are_auto_closed(system_reply: str) -> None:
    result = detect_non_conversation_signals(
        f"MANAGER:\nАлло.\n\nCLIENT:\n{system_reply}",
        duration_sec=8,
    )

    assert result.client_human_response is False
    assert result.should_force_non_conversation is True


def test_split_virtual_secretary_script_is_not_a_live_client_reply() -> None:
    result = detect_non_conversation_signals(
        "MANAGER:\nЗдравствуйте, это секретарь Оля. Передам, что вы звонили. "
        "Если хотите, добавьте что-то и положите трубку.\n\n"
        "CLIENT:\nАлло. Чем могу помочь? Хорошо, все запомнила. Еще что-то?",
        duration_sec=24,
    )

    assert result.should_force_non_conversation is True
    assert result.requires_manual_review is False


@pytest.mark.parametrize(
    "system_reply",
    [
        "Это Мия. У абонента нет возможности взять трубку.",
        "Я электронный помощник. Абоненту пока неудобно разговаривать.",
        "Меня попросили принять ваше сообщение, я передам это абоненту.",
    ],
)
def test_high_precision_secretary_scripts_are_auto_closed_across_tracks(
    system_reply: str,
) -> None:
    result = detect_non_conversation_signals(
        f"MANAGER:\n{system_reply}\n\nCLIENT:\nАлло, меня слышно?",
        history_summary="Содержательного разговора не было.",
        duration_sec=20,
    )

    assert result.should_force_non_conversation is True
    assert result.requires_manual_review is False


@pytest.mark.parametrize(
    "human_reply",
    [
        "Мне сейчас не очень удобно говорить, перезвоните вечером.",
        "Здравствуйте, меня слышно? Я вас слышу.",
    ],
)
def test_live_first_person_reply_is_not_confused_with_a_secretary_script(
    human_reply: str,
) -> None:
    result = detect_non_conversation_signals(
        f"MANAGER:\nДобрый день.\n\nCLIENT:\n{human_reply}",
        history_summary="Содержательного разговора не было.",
        duration_sec=20,
    )

    assert result.should_force_non_conversation is False
    assert result.label != LABEL_NON_CONVERSATION_HIGH_CONFIDENCE


def test_live_person_named_miya_is_not_a_virtual_secretary() -> None:
    result = detect_non_conversation_signals(
        "MANAGER:\nЭто Мия, учебный центр Фотон. Расскажу про курс и оплату.\n\n"
        "CLIENT:\nЗдравствуйте, хочу записать ребёнка на математику.",
        call_type="sales_call",
        duration_sec=80,
    )

    assert result.should_force_non_conversation is False
    assert result.label != LABEL_NON_CONVERSATION_HIGH_CONFIDENCE


def test_live_person_named_miya_may_offer_to_take_a_message() -> None:
    result = detect_non_conversation_signals(
        "MANAGER:\nЭто Мия, учебный центр Фотон. Оставьте сообщение куратору.\n\n"
        "CLIENT:\nХочу записать ребёнка на курс и уточнить стоимость.",
        call_type="sales_call",
        duration_sec=70,
    )

    assert result.should_force_non_conversation is False
    assert result.label != LABEL_NON_CONVERSATION_HIGH_CONFIDENCE


def test_generated_history_cannot_turn_a_live_call_into_a_virtual_secretary() -> None:
    result = detect_non_conversation_signals(
        "MANAGER:\nРасскажу про курс и договор.\n\n"
        "CLIENT:\nХочу записать ребёнка на математику и уточнить стоимость.",
        history_summary="Я передам это абоненту.",
        call_type="sales_call",
        duration_sec=80,
    )

    assert result.should_force_non_conversation is False
    assert result.label != LABEL_NON_CONVERSATION_HIGH_CONFIDENCE


@pytest.mark.parametrize(
    "system_reply",
    [
        "Абонент не берет трубку. Попробуйте перезвонить.",
        "Тот, кому вы звоните, не отвечает. Попросите его перезвонить.",
    ],
)
def test_provider_no_answer_phrases_are_not_short_live_replies(system_reply: str) -> None:
    result = detect_non_conversation_signals(
        f"MANAGER:\nПродолжение следует.\n\nCLIENT:\n{system_reply}",
        duration_sec=15,
    )

    assert result.client_human_response is False
    assert result.should_force_non_conversation is True


def test_manager_business_language_is_not_a_virtual_secretary() -> None:
    result = detect_non_conversation_signals(
        "MANAGER:\nПередам ваш вопрос руководителю курса по искусственному интеллекту.\n\n"
        "CLIENT:\nХорошо, я перезвоню завтра."
    )

    assert result.should_force_non_conversation is False


def test_adversarial_regression_dataset_keeps_expected_safety_boundaries() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "stable_runtime"
        / "transcript_quality_adversarial_audit_20260509"
        / "regression_dataset.csv"
    )
    if not path.exists():
        pytest.skip("adversarial regression dataset is generated outside the unit fixture")

    counts = {
        "high_confidence": 0,
        "protected": 0,
        "manual_review": 0,
        "manual_review_upgraded_to_v2_high_confidence": 0,
    }
    failures: list[str] = []
    with path.open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            expected = row["expected_label"]
            result = detect_non_conversation_signals(
                history_summary=row.get("history_summary", ""),
                transcript_text=row.get("transcript_excerpt", ""),
                call_type=row.get("call_type_current", ""),
            )
            source = row.get("source_filename", "<unknown>")
            if expected == "non_conversation_high_confidence":
                counts["high_confidence"] += 1
                if result.label != LABEL_NON_CONVERSATION_HIGH_CONFIDENCE:
                    failures.append(f"{source}: expected high confidence, got {result.label} {result.reason_codes}")
            elif expected.startswith("contentful_"):
                counts["protected"] += 1
                if result.label != LABEL_CONTENTFUL_PROTECTED_LIVE_DIALOGUE:
                    failures.append(f"{source}: expected protected live, got {result.label} {result.reason_codes}")
            elif expected.startswith("manual_review_"):
                counts["manual_review"] += 1
                if result.should_force_non_conversation:
                    v2_safe_upgrade = (
                        result.label == LABEL_NON_CONVERSATION_HIGH_CONFIDENCE
                        and result.recommended_contact_subtype in {"no_live_or_voicemail", "outbound_voicemail"}
                        and "system_no_dialogue_phrase" in result.reason_codes
                        and "no_live_marker" in result.reason_codes
                    )
                    if v2_safe_upgrade:
                        counts["manual_review_upgraded_to_v2_high_confidence"] += 1
                    else:
                        failures.append(f"{source}: expected no automatic rewrite, got {result.label} {result.reason_codes}")

    assert counts == {
        "high_confidence": 25,
        "protected": 40,
        "manual_review": 25,
        "manual_review_upgraded_to_v2_high_confidence": 23,
    }
    assert not failures
