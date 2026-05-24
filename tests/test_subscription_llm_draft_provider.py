from __future__ import annotations

import subprocess
from pathlib import Path

from mango_mvp.channels.subscription_llm import (
    ADDRESS_FOTON_MOSCOW_SAFE_TEXT,
    ADDRESS_UNPK_MOSCOW_REGULAR_SAFE_TEXT,
    ADDRESS_UNPK_SAFE_TEXT,
    CodexExecConfig,
    CodexExecDraftProvider,
    COMPLAINT_SAFE_TEXT,
    DraftGenerationResult,
    FakeDraftProvider,
    LEGAL_THREAT_SAFE_TEXT,
    KNOWN_CONTEXT_REPAIR_TEXT,
    MATKAP_FEDERAL_TIMING_SAFE_TEXT,
    OFF_TOPIC_FOTON_SAFE_TEXT,
    OFF_TOPIC_UNPK_SAFE_TEXT,
    SubscriptionDraftResult,
    UNPK_EGE_INTENSIVE_PRICE_SAFE_TEXT,
    UNPK_FOUR_WEEKS_NEW_PRICE_SAFE_TEXT,
    UNPK_INSTALLMENT_APPROVED_FALLBACK_TEXT,
    apply_conversation_intent_plan_guard,
    contains_bot_identity_disclosure,
    draft_has_internal_service_markers,
    detect_high_risk_input_markers,
    find_redundant_questions_for_known_context,
    parse_llm_json,
    strip_internal_service_markers,
)
from mango_mvp.channels.subscription_llm import apply_high_risk_content_guards


def test_codex_exec_provider_builds_command_without_openai_key(tmp_path: Path) -> None:
    command = CodexExecConfig(model="gpt-5.5", reasoning_effort="medium").build_command(tmp_path / "out.txt")

    assert "OPENAI_API_KEY" not in " ".join(command)
    assert command[:2] == ["codex", "exec"]
    assert "--sandbox" in command
    assert "read-only" in command


def test_provider_parses_valid_json() -> None:
    result = parse_llm_json(
        '{"route":"draft_for_manager","draft_text":"Здравствуйте! Уточним детали.",'
        '"message_type":"question","broad_group":"commercial","topic_id":"theme:001_pricing",'
        '"confidence_theme":0.8,"confidence_group":0.9,"alternative_themes":["theme:002_payment_method"],'
        '"risk_level":"low","context_used":["recent_messages"],"context_warnings":[]}'
    )

    assert result.route == "draft_for_manager"
    assert result.topic_id == "theme:001_pricing"
    assert result.message_type == "question"
    assert result.broad_group == "commercial"
    assert result.topic_confidence == 0.8
    assert result.confidence_group == 0.9
    assert result.alternative_themes == ("theme:002_payment_method",)
    assert result.to_json_dict()["confidence_theme"] == 0.8


def test_provider_normalizes_unknown_topic_ids_to_unclear_manager_only() -> None:
    result = parse_llm_json(
        '{"route":"draft_for_manager","draft_text":"Здравствуйте! Уточним.",'
        '"message_type":"question","topic_id":"theme:refund_payment","confidence_theme":0.91,'
        '"alternative_themes":["theme:013_schedule","theme:made_up"]}'
    )

    assert result.topic_id == "service:S2_unclear"
    assert result.alternative_themes == ("theme:013_schedule",)
    assert result.route == "manager_only"
    assert "invalid_topic_id_normalized" in result.safety_flags
    assert "invalid_alternative_themes_removed" in result.safety_flags
    assert result.metadata["original_invalid_topic_id"] == "theme:refund_payment"


def test_provider_falls_back_on_invalid_json() -> None:
    result = parse_llm_json("not json")

    assert result.route == "manager_only"
    assert "llm_fallback" in result.safety_flags


def test_provider_timeout_returns_safe_fallback() -> None:
    def runner(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=1)

    result = CodexExecDraftProvider(runner=runner).generate("prompt")

    assert result.route == "manager_only"
    assert "codex_exec_timeout" in result.safety_flags


def test_codex_provider_llm_rewriter_is_feature_flagged(monkeypatch, tmp_path: Path) -> None:
    calls = []

    def runner(cmd, input, capture_output, text, check, timeout, env):
        calls.append(input)
        output_path = Path(cmd[cmd.index("--output-last-message") + 1])
        if len(calls) == 1:
            output_path.write_text(
                '{"route":"draft_for_manager","draft_text":"Вам подойдёт информатика: можно начать с онлайн-группы.",'
                '"message_type":"question","topic_id":"service:S5_general_consultation","confidence_theme":0.8}',
                encoding="utf-8",
            )
        else:
            output_path.write_text(
                '{"draft_text":"Для 6 класса можем подобрать направление без догадок по предмету. '
                'Напишите, что важнее: подтянуть школьную программу или попробовать олимпиадный уровень?"}',
                encoding="utf-8",
            )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setenv("TELEGRAM_ANSWER_QUALITY_LLM_REWRITE", "1")
    provider = CodexExecDraftProvider(runner=runner, cache_dir=None, timeout_sec=20)

    result = provider.build_draft(
        "Здравствуйте, хочу понять, что у вас есть для 6 класса.",
        context={"active_brand": "foton", "known_slots": {"grade": "6"}, "confirmed_facts": {"fact:general": "Фотон: есть курсы для школьников."}},
    )

    assert len(calls) == 2
    assert result.metadata["answer_quality"]["rewrite_provider"] == "llm_runner"
    assert "информатика" not in result.draft_text
    assert "answer_quality_rewritten" in result.safety_flags


def test_codex_provider_supports_rewriter_feature_flag_alias(monkeypatch, tmp_path: Path) -> None:
    calls = []

    def runner(cmd, input, capture_output, text, check, timeout, env):
        calls.append(input)
        output_path = Path(cmd[cmd.index("--output-last-message") + 1])
        if len(calls) == 1:
            output_path.write_text(
                '{"route":"draft_for_manager","draft_text":"Вам подойдёт информатика: можно начать с онлайн-группы.",'
                '"message_type":"question","topic_id":"service:S5_general_consultation","confidence_theme":0.8}',
                encoding="utf-8",
            )
        else:
            output_path.write_text(
                '{"draft_text":"Для 6 класса можем подобрать направление без догадок по предмету. '
                'Напишите, что важнее: подтянуть школьную программу или попробовать олимпиадный уровень?"}',
                encoding="utf-8",
            )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.delenv("TELEGRAM_ANSWER_QUALITY_LLM_REWRITE", raising=False)
    monkeypatch.setenv("TELEGRAM_ANSWER_QUALITY_LLM_REWRITER", "1")
    provider = CodexExecDraftProvider(runner=runner, cache_dir=None, timeout_sec=20)

    result = provider.build_draft(
        "Здравствуйте, хочу понять, что у вас есть для 6 класса.",
        context={"active_brand": "foton", "known_slots": {"grade": "6"}, "confirmed_facts": {"fact:general": "Фотон: есть курсы для школьников."}},
    )

    assert len(calls) == 2
    assert result.metadata["answer_quality"]["rewrite_provider"] == "llm_runner"
    assert "информатика" not in result.draft_text


def test_draft_text_blocks_vendor_prompt_or_identity_lies() -> None:
    result = parse_llm_json('{"route":"draft_for_manager","draft_text":"Как ИИ я могу подсказать."}')

    assert result.route == "manager_only"
    assert "bot_identity_disclosure" in result.safety_flags
    assert not contains_bot_identity_disclosure("Да, я цифровой помощник Фотона, не живой оператор.")
    for phrase in ("как ИИ", "я нейросеть", "GPT", "Claude", "Codex", "OpenAI", "я человек", "я не бот", "system prompt"):
        assert contains_bot_identity_disclosure(f"Тест: {phrase}")


def test_direct_identity_question_gets_brand_safe_policy_c_answer() -> None:
    base = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        topic_id="service:S5_general_consultation",
        topic_confidence=0.95,
        draft_text="Здравствуйте! Чем помочь?",
    )

    foton = apply_high_risk_content_guards(base, client_message="Вы бот или человек?", context={"active_brand": "foton"})
    assert foton.route == "draft_for_manager"
    assert "цифровой помощник Фотона" in foton.draft_text
    assert "GPT" not in foton.draft_text

    unpk = apply_high_risk_content_guards(base, client_message="Ты GPT?", context={"active_brand": "unpk"})
    assert unpk.route == "draft_for_manager"
    assert "цифровой помощник" in unpk.draft_text
    assert "GPT" not in unpk.draft_text


def test_conversation_intent_plan_guard_uses_context_not_keyword_branch() -> None:
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        topic_id="theme:001_pricing",
        topic_confidence=0.84,
        draft_text="Да, текущая цена такая-то, можно закрепить условия.",
    )

    guarded = apply_conversation_intent_plan_guard(
        result,
        client_message="Можно закрепить место на ЛВШ?",
        context={
            "active_brand": "foton",
            "conversation_intent_plan": {
                "primary_intent": "live_availability",
                "topic_id": "theme:026_camp_general",
                "answer_policy": "answer_safe_parts_then_manager_live_check",
                "route_bias": "draft_for_manager",
                "product_family": "camp",
            },
        },
    )

    assert guarded.topic_id == "theme:026_camp_general"
    assert guarded.route == "draft_for_manager"
    assert "conversation_intent_plan_live_availability" in guarded.safety_flags
    assert "conversation_intent_plan_topic_applied" in guarded.safety_flags


def test_conversation_intent_plan_guard_keeps_p0_manager_only() -> None:
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        topic_id="theme:001_pricing",
        topic_confidence=0.84,
        draft_text="Стоимость зависит от курса.",
    )

    guarded = apply_conversation_intent_plan_guard(
        result,
        client_message="Хочу вернуть деньги.",
        context={
            "active_brand": "unpk",
            "conversation_intent_plan": {
                "primary_intent": "refund",
                "topic_id": "theme:009_refund",
                "answer_policy": "manager_only_p0",
                "route_bias": "manager_only",
            },
        },
    )

    assert guarded.topic_id == "theme:009_refund"
    assert guarded.route == "manager_only"
    assert "conversation_intent_plan_p0" in guarded.safety_flags


def test_conversation_intent_plan_repairs_false_legal_from_model_when_current_message_is_process_question() -> None:
    result = SubscriptionDraftResult(
        route="manager_only",
        topic_id="theme:029_legal_question",
        topic_confidence=0.84,
        draft_text="Приняли обращение. Передам его ответственному сотруднику, он вернется с ответом.",
        safety_flags=("high_risk_manager_only", "legal_threat_topic_overrode_refund"),
    )

    guarded = apply_conversation_intent_plan_guard(
        result,
        client_message="А чтобы записаться или с менеджером обсудить, надо приезжать или можно дистанционно?",
        context={
            "active_brand": "unpk",
            "conversation_intent_plan": {
                "primary_intent": "format",
                "topic_id": "theme:014_format",
                "answer_policy": "answer_directly_if_fact_verified",
                "route_bias": "bot_answer_self_for_pilot",
                "risk_signals": [],
            },
        },
    )

    assert guarded.topic_id == "theme:014_format"
    assert guarded.route == "draft_for_manager"
    assert "conversation_intent_plan_false_p0_repaired" in guarded.metadata
    assert "high_risk_manager_only" not in guarded.safety_flags


def test_high_risk_guards_do_not_recreate_false_legal_when_plan_is_semantic_non_p0() -> None:
    result = apply_high_risk_content_guards(
        SubscriptionDraftResult(
            route="draft_for_manager",
            topic_id="theme:029_legal_question",
            topic_confidence=0.84,
            draft_text="Можно оформить дистанционно: приезжать не нужно. Передам менеджеру запрос на запись.",
        ),
        client_message="А чтобы записаться или с менеджером обсудить, надо приезжать или можно дистанционно?",
        context={
            "active_brand": "unpk",
            "conversation_intent_plan": {
                "primary_intent": "format",
                "topic_id": "theme:014_format",
                "answer_policy": "answer_directly_if_fact_verified",
                "route_bias": "bot_answer_self_for_pilot",
                "risk_signals": [],
            },
        },
    )

    assert result.route == "draft_for_manager"
    assert "zero_collect_legal_guarded" not in result.safety_flags
    assert "Приняли обращение" not in result.draft_text


def test_soft_negative_feedback_is_not_treated_as_complaint_p0() -> None:
    result = apply_high_risk_content_guards(
        SubscriptionDraftResult(
            route="draft_for_manager",
            topic_id="theme:023_trial_class",
            topic_confidence=0.9,
            draft_text="Передам менеджеру контекст.",
        ),
        client_message="Я же про очный курс спрашиваю. Похоже, вы не можете ответить, подумаю тогда.",
        context={
            "active_brand": "unpk",
            "conversation_intent_plan": {
                "primary_intent": "format",
                "topic_id": "theme:014_format",
                "answer_policy": "answer_directly_if_fact_verified",
                "route_bias": "bot_answer_self_for_pilot",
                "risk_signals": [],
            },
        },
    )

    assert result.route == "draft_for_manager"
    assert "complaint_apology_guarded" not in result.safety_flags
    assert "high_risk_manager_only" not in result.safety_flags
    assert result.draft_text.startswith("Понял, давайте не буду повторять общий ответ")


def test_quality_rewrite_is_not_overwritten_by_unpk_installment_fallback() -> None:
    rewritten = SubscriptionDraftResult(
        route="draft_for_manager",
        topic_id="theme:006_installment",
        topic_confidence=0.94,
        draft_text="Да, помесячно платить можно. Скидка при этом не применяется: 10% действует при оплате за семестр, 14% — при оплате за год.",
        safety_flags=("answer_quality_rewritten",),
        metadata={"answer_quality": {"rewritten": True}},
    )

    guarded = apply_high_risk_content_guards(
        rewritten,
        client_message="А если помесячно, скидка сохраняется?",
        context={"active_brand": "unpk"},
    )

    assert guarded.draft_text == rewritten.draft_text
    assert "unpk_installment_approved_fallback_applied" not in guarded.safety_flags


def test_price_fix_process_answers_directly_without_reasking_known_slots() -> None:
    result = apply_high_risk_content_guards(
        SubscriptionDraftResult(
            route="draft_for_manager",
            topic_id="theme:001_pricing",
            topic_confidence=0.9,
            draft_text="Онлайн-обучение в Фотоне: есть варианты оплаты за семестр и год.",
        ),
        client_message="Как закрепить 47 250 — заявка или оплата?",
        context={
            "active_brand": "foton",
            "known_slots": {"grade": "8", "subject": "физика", "format": "онлайн"},
            "conversation_intent_plan": {
                "primary_intent": "price_fix",
                "topic_id": "theme:001_pricing",
                "answer_policy": "answer_directly_if_fact_verified",
                "route_bias": "bot_answer_self_for_pilot",
                "known_slots": {"grade": "8", "subject": "физика", "format": "онлайн"},
            },
        },
    )

    assert "достаточно ли одной заявки или нужна оплата" in result.draft_text
    assert "8 класс" in result.draft_text
    assert "физика" in result.draft_text
    assert "direct_process_safe_template_applied" in result.safety_flags


def test_installment_question_with_negated_places_is_not_overwritten_by_camp_template() -> None:
    result = apply_high_risk_content_guards(
        SubscriptionDraftResult(
            route="draft_for_manager",
            topic_id="theme:006_installment",
            topic_confidence=0.9,
            draft_text="По местам не буду обещать без проверки.",
        ),
        client_message="Я не про места спрашиваю, а про оплату. Можно помесячно или за семестр?",
        context={
            "active_brand": "foton",
            "recent_messages": ["Ответ: По местам не буду обещать без проверки."],
            "known_slots": {"grade": "4", "subject": "математика", "format": "очно"},
            "conversation_intent_plan": {
                "primary_intent": "installment",
                "topic_id": "theme:006_installment",
                "answer_policy": "answer_directly_if_fact_verified",
                "route_bias": "bot_answer_self_for_pilot",
            },
        },
    )

    assert result.topic_id == "theme:006_installment"
    assert "местам" not in result.draft_text.casefold()
    assert "оплат" in result.draft_text.casefold() or "рассроч" in result.draft_text.casefold()


def test_manager_handoff_request_is_acknowledged_as_action() -> None:
    result = apply_high_risk_content_guards(
        SubscriptionDraftResult(
            route="draft_for_manager",
            topic_id="theme:023_trial_class",
            topic_confidence=0.9,
            draft_text="Если хотите, передам менеджеру.",
        ),
        client_message="Да, передайте менеджеру, пожалуйста.",
        context={"active_brand": "unpk", "known_slots": {"grade": "7", "subject": "математика"}},
    )

    assert result.route == "draft_for_manager"
    assert result.draft_text.startswith("Да, передам менеджеру:")
    assert "7 класс" in result.draft_text
    assert "математика" in result.draft_text

    followup = apply_high_risk_content_guards(
        SubscriptionDraftResult(
            route="bot_answer_self_for_pilot",
            topic_id="theme:023_trial_class",
            topic_confidence=0.9,
            draft_text="По онлайн-формату можно прислать фрагмент занятия.",
        ),
        client_message="Пусть менеджер тогда напишет и скажет по фрагменту точно.",
        context={
            "active_brand": "unpk",
            "known_slots": {"grade": "9", "subject": "информатика", "format": "онлайн"},
        },
    )

    assert followup.draft_text.startswith("Да, передам менеджеру:")
    assert "9 класс" in followup.draft_text
    assert "информатика" in followup.draft_text
    assert "онлайн" in followup.draft_text


def test_unpk_online_trial_context_beats_address_trigger_words() -> None:
    result = apply_high_risk_content_guards(
        SubscriptionDraftResult(
            route="bot_answer_self_for_pilot",
            topic_id="theme:014_format",
            topic_confidence=0.9,
            draft_text="Старшая школа проходит в Главном корпусе МФТИ по адресу Институтский пер., 9.",
        ),
        client_message="9 класс, информатика. мы онлайн хотим, приезжать не надо будет?",
        context={
            "active_brand": "unpk",
            "recent_messages": ["Клиент: а пробное занятие у вас есть?", "Ответ: По онлайну можно прислать фрагмент занятия."],
            "known_slots": {"grade": "9", "subject": "информатика", "format": "онлайн"},
            "conversation_intent_plan": {
                "primary_intent": "format",
                "topic_id": "theme:014_format",
                "known_slots": {"grade": "9", "subject": "информатика", "format": "онлайн"},
            },
        },
    )

    assert "фрагмент занятия" in result.draft_text
    assert "приезжать" in result.draft_text or "приезжать для этого не нужно" in result.draft_text
    assert "Институтский" not in result.draft_text
    assert "Сретенка" not in result.draft_text
    assert "trial_safe_template_applied" in result.safety_flags


def test_trial_fragment_data_question_answers_directly_without_reasking_known_slots() -> None:
    result = apply_high_risk_content_guards(
        SubscriptionDraftResult(
            route="bot_answer_self_for_pilot",
            topic_id="theme:023_trial_class",
            topic_confidence=0.9,
            draft_text="По онлайн-формату можно прислать фрагмент занятия.",
        ),
        client_message="Я уже написала: 9 класс, информатика, онлайн. Какие данные нужны, чтобы мне прислали фрагмент?",
        context={
            "active_brand": "unpk",
            "recent_messages": ["Клиент: а пробное занятие есть?"],
            "known_slots": {"grade": "9", "subject": "информатика", "format": "онлайн"},
        },
    )

    assert "9 класс" in result.draft_text
    assert "информатика" in result.draft_text
    assert "онлайн" in result.draft_text
    assert "повторять" in result.draft_text.casefold() or "повторно" in result.draft_text.casefold()
    assert "личные документы" in result.draft_text
    assert "Передам менеджеру запрос на фрагмент" in result.draft_text
    assert "trial_safe_template_applied" in result.safety_flags


def test_trial_fragment_process_and_ack_do_not_repeat_generic_trial_template() -> None:
    base_context = {
        "active_brand": "unpk",
        "recent_messages": ["Клиент: а пробное занятие есть?", "Клиент: 9 класс, информатика, онлайн"],
        "known_slots": {"grade": "9", "subject": "информатика", "format": "онлайн"},
    }
    process = apply_high_risk_content_guards(
        SubscriptionDraftResult(
            route="bot_answer_self_for_pilot",
            topic_id="theme:023_trial_class",
            topic_confidence=0.9,
            draft_text="По онлайн-формату можно прислать фрагмент занятия.",
        ),
        client_message="Как получить этот фрагмент?",
        context=base_context,
    )
    ack = apply_high_risk_content_guards(
        SubscriptionDraftResult(
            route="manager_only",
            topic_id="theme:023_trial_class",
            topic_confidence=0.9,
            draft_text="По онлайн-формату можно прислать фрагмент занятия.",
        ),
        client_message="Хорошо, жду фрагмент, посмотрю и потом решу.",
        context=base_context,
    )

    assert "Вижу уже: 9 класс, информатика, онлайн." in process.draft_text
    assert "Передам менеджеру запрос на фрагмент" in process.draft_text
    assert "Вижу уже: 9 класс, информатика, онлайн." in ack.draft_text
    assert "передам менеджеру запрос на онлайн-фрагмент" in ack.draft_text.casefold()
    assert "Бесплатность отдельно не обещаю" not in process.draft_text
    assert "Бесплатность отдельно не обещаю" not in ack.draft_text


def test_foton_trial_process_does_not_use_unpk_fragment_wording() -> None:
    result = apply_high_risk_content_guards(
        SubscriptionDraftResult(
            route="bot_answer_self_for_pilot",
            topic_id="theme:023_trial_class",
            topic_confidence=0.9,
            draft_text="В онлайн-формате Фотона пробное занятие есть.",
        ),
        client_message="5 класс математика онлайн. Как тогда записаться на пробное?",
        context={
            "active_brand": "foton",
            "recent_messages": ["Клиент: а пробное занятие есть?"],
            "known_slots": {"grade": "5", "subject": "математика", "format": "онлайн"},
        },
    )

    assert "онлайн-пробное" in result.draft_text
    assert "фрагмент" not in result.draft_text.casefold()
    assert "5 класс" in result.draft_text
    assert "математика" in result.draft_text
    assert "trial_safe_template_applied" in result.safety_flags


def test_foton_trial_live_or_recording_question_does_not_switch_to_teacher_template() -> None:
    result = apply_high_risk_content_guards(
        SubscriptionDraftResult(
            route="bot_answer_self_for_pilot",
            topic_id="theme:023_trial_class",
            topic_confidence=0.9,
            draft_text="У нас преподают специалисты из МФТИ.",
        ),
        client_message="Пробное будет живое онлайн занятие с преподавателем или запись?",
        context={
            "active_brand": "foton",
            "recent_messages": ["Клиент: 5 класс математика онлайн, хочу пробное"],
            "known_slots": {"grade": "5", "subject": "математика", "format": "онлайн"},
        },
    )

    assert "формат пробного" in result.draft_text
    assert "живые вебинары" in result.draft_text
    assert "записи уроков" in result.draft_text
    assert "У нас преподают специалисты" not in result.draft_text
    assert "teacher_safe_template_applied" not in result.safety_flags


def test_negated_address_question_does_not_return_address_template() -> None:
    result = apply_high_risk_content_guards(
        SubscriptionDraftResult(
            route="bot_answer_self_for_pilot",
            topic_id="theme:023_trial_class",
            topic_confidence=0.9,
            draft_text="Площадки УНПК: Москва — Сретенка, 20; Долгопрудный — МФТИ.",
        ),
        client_message="Адрес не нужен, я спрашиваю про онлайн-фрагмент бесплатно или нет.",
        context={"active_brand": "unpk", "recent_messages": ["Клиент: пробное занятие есть?"]},
    )

    assert "Сретенка" not in result.draft_text
    assert "фрагмент" in result.draft_text


def test_unconfirmed_followup_deadline_blocks_within_day_wording() -> None:
    provider = FakeDraftProvider(
        {
            "route": "draft_for_manager",
            "draft_text": "Менеджер ответит в течение суток и пришлёт фрагмент.",
            "message_type": "question",
            "topic_id": "theme:013_schedule",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "Когда мне ответят?",
        context={"active_brand": "unpk", "rop_policy": {"bot_permission": "draft_for_manager"}},
    )

    assert result.route == "manager_only"
    assert "в течение суток" not in result.draft_text
    assert "unsupported_followup_deadline_detected" in result.safety_flags


def test_draft_text_strips_kb_source_and_freshness_metadata() -> None:
    result = parse_llm_json(
        '{"route":"draft_for_manager","draft_text":"Здравствуйте! '
        '[Стоимость; source=source:price; freshness=fresh_verified] Менеджер сверит условия.",'
        '"message_type":"question","topic_id":"theme:001_pricing","confidence_theme":0.91}'
    )

    assert "source=" not in result.draft_text
    assert "freshness=" not in result.draft_text
    assert "source:" not in result.draft_text
    assert "internal_metadata_removed_from_draft" in result.safety_flags
    assert draft_has_internal_service_markers("[x; source=source:price; freshness=fresh]")
    assert strip_internal_service_markers("[x; source=source:price; freshness=fresh] Ответ") == "Ответ"
    assert strip_internal_service_markers("[source_id=fact:v3:price; kb_release_20260520_v6_3] Ответ") == "Ответ"
    assert "product_data" not in strip_internal_service_markers("Ответ source_id=fact:v3:price product_data/knowledge_base/kb_release_20260520_v6_3")
    assert "/Users/" not in strip_internal_service_markers("Ответ /Users/dmitrijfabarisov/Projects/Mango")


def test_final_p0_override_replaces_non_p0_draft_text() -> None:
    base = parse_llm_json(
        '{"route":"bot_answer_self_for_pilot","draft_text":"Стоимость зависит от класса, подскажите детали.",'
        '"message_type":"question","topic_id":"theme:001_pricing","confidence_theme":0.91}'
    )

    result = apply_high_risk_content_guards(
        base,
        client_message="Если не вернете деньги, пойду в суд.",
        context={"active_brand": "unpk"},
    )

    assert result.route == "manager_only"
    assert result.topic_id == "theme:029_legal_question"
    assert result.draft_text == LEGAL_THREAT_SAFE_TEXT
    assert "final_p0_text_override" in result.safety_flags
    assert "high_risk_manager_only" in result.safety_flags


def test_answer_contract_prevents_green_installment_fallback_lock_in() -> None:
    base = parse_llm_json(
        '{"route":"bot_answer_self_for_pilot","draft_text":"В УНПК можно платить помесячно, за семестр или за год.",'
        '"message_type":"question","topic_id":"theme:006_installment","confidence_theme":0.91}'
    )

    result = apply_high_risk_content_guards(
        base,
        client_message="А банк не участвует? Можно помесячно?",
        context={
            "active_brand": "unpk",
            "answer_contract": {
                "primary_intent": "installment",
                "direct_question": "А банк не участвует? Можно помесячно?",
                "must_answer_first": True,
                "p0_required": False,
            },
        },
    )

    assert result.draft_text != UNPK_INSTALLMENT_APPROVED_FALLBACK_TEXT
    assert "unpk_installment_approved_fallback_applied" not in result.safety_flags
    assert result.metadata["answer_contract_controls_green_templates"] is True


def test_group_vs_individual_question_does_not_force_individual_handoff() -> None:
    base = parse_llm_json(
        '{"route":"bot_answer_self_for_pilot","draft_text":"Есть групповые форматы, менеджер поможет выбрать по уровню.",'
        '"message_type":"question","topic_id":"theme:014_format","confidence_theme":0.91}'
    )

    result = apply_high_risk_content_guards(
        base,
        client_message="Есть группы по физике или только индивидуально?",
        context={"active_brand": "foton"},
    )

    assert result.draft_text != "Менеджер свяжется и подскажет варианты индивидуальных занятий."
    assert "terminal_safe_template_applied" not in result.safety_flags


def test_draft_text_blocks_manager_placeholder() -> None:
    result = parse_llm_json(
        '{"route":"draft_for_manager","draft_text":"Актуальный канал: [вставить проверенную ссылку на канал]",'
        '"message_type":"question","topic_id":"service:S5_general_consultation","confidence_theme":0.91}'
    )

    assert result.route == "manager_only"
    assert result.draft_text != "Актуальный канал: [вставить проверенную ссылку на канал]"
    assert "placeholder_in_draft" in result.safety_flags
    assert "placeholder_in_draft" in result.forbidden_promises_detected


def test_draft_text_blocks_known_promocode_leak() -> None:
    result = parse_llm_json(
        '{"route":"draft_for_manager","draft_text":"Проверю, действует ли промокод LVSH-KF-10 для вашей программы.",'
        '"message_type":"question","topic_id":"theme:005_discounts","confidence_theme":0.91}'
    )

    assert result.route == "manager_only"
    assert "LVSH-KF-10" not in result.draft_text
    assert "promocode_in_draft_guarded" in result.safety_flags
    assert "promocode_in_draft" in result.forbidden_promises_detected


def test_low_confidence_forces_manager_only() -> None:
    result = parse_llm_json(
        '{"route":"draft_for_manager","draft_text":"Здравствуйте!","message_type":"question",'
        '"topic_id":"theme:001_pricing","confidence_theme":0.55}'
    )

    assert result.route == "manager_only"
    assert "low_confidence_manager_only" in result.safety_flags


def test_high_risk_theme_forces_manager_only() -> None:
    result = parse_llm_json(
        '{"route":"draft_for_manager","draft_text":"Вернем деньги.","message_type":"question",'
        '"topic_id":"theme:009_refund","confidence_theme":0.91}'
    )

    assert result.route == "manager_only"
    assert "high_risk_manager_only" in result.safety_flags
    assert any("Высокорисковая" in item for item in result.manager_checklist)


def test_high_risk_client_message_forces_manager_only_even_when_topic_is_wrong() -> None:
    provider = FakeDraftProvider(
        {
            "route": "draft_for_manager",
            "draft_text": "Здравствуйте! Уточним условия.",
            "message_type": "question",
            "topic_id": "theme:001_pricing",
            "confidence_theme": 0.86,
        }
    )

    result = provider.build_draft(
        "В случае невозможности замены класса, как можно получить возврат платежа?",
        context={"rop_policy": {"bot_permission": "draft_for_manager"}},
    )

    assert result.topic_id == "theme:009_refund"
    assert result.route == "manager_only"
    assert result.draft_text != "Здравствуйте! Уточним условия."
    assert "zero_collect_refund_guarded" in result.safety_flags
    assert "final_p0_text_override" in result.safety_flags


def test_refund_zero_collect_removes_pii_request_from_draft() -> None:
    provider = FakeDraftProvider(
        {
            "route": "draft_for_manager",
            "draft_text": "Пришлите, пожалуйста, ФИО ученика и номер договора, если он есть.",
            "message_type": "question",
            "topic_id": "theme:009_refund",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "В случае невозможности замены класса, как можно получить возврат платежа?",
        context={"active_brand": "unpk", "rop_policy": {"bot_permission": "draft_for_manager"}},
    )

    assert result.route == "manager_only"
    assert "zero_collect_refund_guarded" in result.safety_flags
    lowered = result.draft_text.casefold()
    for forbidden in ("фио", "номер договора", "телефон", "email", "сумм", "причин"):
        assert forbidden not in lowered
    assert "Пока ничего дополнительно присылать не нужно." in result.draft_text


def test_refund_zero_collect_removes_dogovor_or_amount_mentions_even_without_direct_ask() -> None:
    provider = FakeDraftProvider(
        {
            "route": "manager_only",
            "draft_text": "Уточню информацию по оплате и договору и вернусь с дальнейшими шагами.",
            "message_type": "question",
            "topic_id": "theme:009_refund",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "Как вернуть деньги за оплату?",
        context={"active_brand": "unpk", "rop_policy": {"bot_permission": "draft_for_manager"}},
    )

    assert "zero_collect_refund_guarded" in result.safety_flags
    assert "договор" not in result.draft_text.casefold()
    assert "оплат" not in result.draft_text.casefold()


def test_known_context_guard_blocks_reasking_known_student_and_grade() -> None:
    provider = FakeDraftProvider(
        {
            "route": "bot_answer_self_for_pilot",
            "draft_text": "Напишите, пожалуйста, ФИО ребёнка, какой класс и какой предмет интересует.",
            "message_type": "question",
            "topic_id": "theme:016_program",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "Подберите курс",
        context={
            "active_brand": "unpk",
            "autonomy_enabled": True,
            "client_safe_fact_verified": True,
            "known_client_fields": {"student_name": "Колосов Даниил Максимович"},
            "known_dialog_fields": {"grade": "9", "subject": "физика"},
            "facts_context": {"client_safe_fact_verified": True, "fresh": True},
        },
    )

    assert result.route == "draft_for_manager"
    assert result.draft_text == KNOWN_CONTEXT_REPAIR_TEXT
    assert "asked_known_data_again" in result.safety_flags
    assert result.metadata["asked_known_data_again_fields"] == ["student_name", "grade", "subject"]


def test_find_redundant_questions_ignores_unknown_fields() -> None:
    repeated = find_redundant_questions_for_known_context(
        "Напишите класс ребёнка и предмет.",
        context={"active_brand": "foton"},
    )

    assert repeated == ()


def test_legal_threat_zero_collect_removes_pii_request_from_draft() -> None:
    provider = FakeDraftProvider(
        {
            "route": "draft_for_manager",
            "draft_text": "Пришлите ФИО, телефон и номер договора, разберёмся.",
            "message_type": "question",
            "topic_id": "theme:029_legal_question",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "Если не вернёте деньги, обращусь в прокуратуру и суд.",
        context={"active_brand": "foton", "rop_policy": {"bot_permission": "draft_for_manager"}},
    )

    assert result.route == "manager_only"
    assert "zero_collect_legal_guarded" in result.safety_flags
    assert result.draft_text == LEGAL_THREAT_SAFE_TEXT
    assert "автоматический" not in result.draft_text.casefold()
    assert "зафикс" not in result.draft_text.casefold()


def test_legal_threat_overrides_refund_topic_and_text() -> None:
    provider = FakeDraftProvider(
        {
            "route": "manager_only",
            "draft_text": "Здравствуйте! Сначала проверим, можно ли оперативно решить ситуацию без возврата.",
            "message_type": "question",
            "topic_id": "theme:009_refund",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "Если не вернёте деньги, оставлю отзыв и пойду в прокуратуру.",
        context={"active_brand": "unpk", "rop_policy": {"bot_permission": "draft_for_manager"}},
    )

    assert result.topic_id == "theme:029_legal_question"
    assert result.route == "manager_only"
    assert "legal_threat_topic_overrode_refund" in result.safety_flags
    assert result.draft_text == LEGAL_THREAT_SAFE_TEXT
    assert "автоматический" not in result.draft_text.casefold()
    assert "зафикс" not in result.draft_text.casefold()


def test_combined_refund_and_price_sets_combined_high_risk_flag() -> None:
    provider = FakeDraftProvider(
        {
            "route": "draft_for_manager",
            "draft_text": "Онлайн для 7 класса стоит 37 125 рублей, по возврату пришлите договор.",
            "message_type": "question",
            "topic_id": "theme:001_pricing",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "Сколько стоит онлайн 7 класс? И ещё хочу вернуть деньги за прошлый курс",
        context={"active_brand": "foton", "rop_policy": {"bot_permission": "draft_for_manager"}},
    )

    assert result.route == "manager_only"
    assert "combined_high_risk_manager_only" in result.safety_flags
    assert "zero_collect_refund_guarded" in result.safety_flags
    assert "37 125" not in result.draft_text
    assert "договор" not in result.draft_text.casefold()


def test_combined_legal_and_price_sets_combined_high_risk_flag() -> None:
    provider = FakeDraftProvider(
        {
            "route": "draft_for_manager",
            "draft_text": "Цена за год 74 500 рублей.",
            "message_type": "question",
            "topic_id": "theme:001_pricing",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "Какая цена очного курса на год? И если что — пойду в суд",
        context={"active_brand": "foton", "rop_policy": {"bot_permission": "draft_for_manager"}},
    )

    assert result.route == "manager_only"
    assert "combined_high_risk_manager_only" in result.safety_flags
    assert "zero_collect_legal_guarded" in result.safety_flags
    assert "74 500" not in result.draft_text


def test_legal_threat_with_complaint_word_keeps_legal_template() -> None:
    provider = FakeDraftProvider(
        {
            "route": "manager_only",
            "draft_text": "Спасибо, что обратились. Передам вопрос менеджеру: он свяжется с вами в ближайшее время.",
            "message_type": "manager_only",
            "topic_id": "theme:029_legal_question",
            "confidence_theme": 0.93,
        }
    )

    result = provider.build_draft(
        "Подам жалобу в Роспотребнадзор!",
        context={"active_brand": "unpk", "rop_policy": {"bot_permission": "draft_for_manager"}},
    )

    assert result.topic_id == "theme:029_legal_question"
    assert result.route == "manager_only"
    assert "zero_collect_legal_guarded" in result.safety_flags
    assert "complaint_apology_guarded" not in result.safety_flags
    assert result.draft_text == LEGAL_THREAT_SAFE_TEXT


def test_reputation_threat_uses_complaint_template_not_legal_template() -> None:
    provider = FakeDraftProvider(
        {
            "route": "draft_for_manager",
            "draft_text": "Передам менеджеру.",
            "message_type": "question",
            "topic_id": "theme:019b_negative_feedback",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "Напишу отзыв в интернете о вас, всех предупрежу.",
        context={"active_brand": "unpk", "rop_policy": {"bot_permission": "draft_for_manager"}},
    )

    assert result.route == "manager_only"
    assert "complaint_apology_guarded" in result.safety_flags
    assert "zero_collect_legal_guarded" not in result.safety_flags
    assert result.draft_text == COMPLAINT_SAFE_TEXT
    assert "извин" not in result.draft_text.casefold()
    assert "неприятно" not in result.draft_text.casefold()


def test_reputation_threat_overrides_llm_legal_topic_to_complaint_template() -> None:
    provider = FakeDraftProvider(
        {
            "route": "manager_only",
            "draft_text": "Ваше обращение принято.",
            "message_type": "manager_only",
            "topic_id": "theme:029_legal_question",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "Напишу отзыв в интернете о вас, всех предупрежу.",
        context={"active_brand": "unpk", "rop_policy": {"bot_permission": "draft_for_manager"}},
    )

    assert result.topic_id == "theme:019b_negative_feedback"
    assert result.route == "manager_only"
    assert "complaint_apology_guarded" in result.safety_flags
    assert "zero_collect_legal_guarded" not in result.safety_flags
    assert result.draft_text == COMPLAINT_SAFE_TEXT


def test_result_guarantee_uses_safe_template() -> None:
    provider = FakeDraftProvider(
        {
            "route": "draft_for_manager",
            "draft_text": "Гарантировать конкретный результат нельзя.",
            "message_type": "question",
            "topic_id": "service:S2_unclear",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "Гарантируете, что ребёнок сдаст ЕГЭ на 90 баллов?",
        context={"active_brand": "unpk", "rop_policy": {"bot_permission": "draft_for_manager"}},
    )

    assert result.route == "draft_for_manager"
    assert "result_guarantee_safe_template_applied" in result.safety_flags
    assert "не даём" in result.draft_text
    assert "не гарантируем" in result.draft_text
    assert "зависит от ученика" in result.draft_text
    assert "статистика" in result.draft_text
    assert "менеджер" in result.draft_text.casefold()


def test_admission_guarantee_uses_statistic_without_guarantee() -> None:
    provider = FakeDraftProvider(
        {
            "route": "draft_for_manager",
            "draft_text": "Поступление зависит от подготовки.",
            "message_type": "question",
            "topic_id": "service:S2_unclear",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "Гарантируете поступление в МФТИ после ваших курсов?",
        context={"active_brand": "unpk", "rop_policy": {"bot_permission": "draft_for_manager"}},
    )

    assert result.route == "draft_for_manager"
    assert "admission_guarantee_safe_template_applied" in result.safety_flags
    assert "не даём" in result.draft_text
    assert "не гарантируем" in result.draft_text
    assert "97%" in result.draft_text
    assert "статистика" in result.draft_text


def test_forced_discount_uses_manager_template() -> None:
    provider = FakeDraftProvider(
        {
            "route": "draft_for_manager",
            "draft_text": "Предлагаем скидку 50%.",
            "message_type": "question",
            "topic_id": "theme:005_discounts",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "Дайте скидку 50%, иначе уйду к конкурентам",
        context={"active_brand": "unpk", "rop_policy": {"bot_permission": "draft_for_manager"}},
    )

    assert result.route == "manager_only"
    assert "discount_safe_template_applied" in result.safety_flags
    assert "менеджеру" in result.draft_text
    assert "свяжется" in result.draft_text
    assert "50%" not in result.draft_text


def test_second_subject_discount_uses_verified_brand_template() -> None:
    provider = FakeDraftProvider(
        {
            "route": "manager_only",
            "draft_text": "Скидка есть.",
            "message_type": "question",
            "topic_id": "theme:005_discounts",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "Если ребёнок будет ходить на два предмета — скидка есть?",
        context={"active_brand": "foton", "rop_policy": {"bot_permission": "draft_for_manager"}},
    )

    assert result.route == "draft_for_manager"
    assert "20%" in result.draft_text
    assert "30%" in result.draft_text
    assert "одного и того же ребёнка" in result.draft_text
    assert "очно" in result.draft_text
    assert "онлайн" in result.draft_text
    assert "не суммируются" in result.draft_text


def test_multichild_discount_mentions_one_child_and_certificate() -> None:
    provider = FakeDraftProvider(
        {
            "route": "manager_only",
            "draft_text": "Скидка есть.",
            "message_type": "question",
            "topic_id": "theme:005_discounts",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "У меня трое детей, но учится только один сейчас. Можно скидку?",
        context={"active_brand": "foton", "rop_policy": {"bot_permission": "draft_for_manager"}},
    )

    assert result.route == "draft_for_manager"
    assert "10%" in result.draft_text
    assert "удостоверение" in result.draft_text
    assert "даже если учится один" in result.draft_text


def test_discount_stacking_uses_non_stacking_template() -> None:
    provider = FakeDraftProvider(
        {
            "route": "manager_only",
            "draft_text": "Скидки сложатся.",
            "message_type": "question",
            "topic_id": "theme:005_discounts",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "Я многодетная, и второй ребёнок будет на двух предметах. Все скидки сложатся?",
        context={"active_brand": "foton", "rop_policy": {"bot_permission": "draft_for_manager"}},
    )

    assert result.route == "draft_for_manager"
    assert "не суммируются" in result.draft_text
    assert "наибольшая" in result.draft_text
    assert "применяется" in result.draft_text


def test_foton_installment_uses_verified_template() -> None:
    provider = FakeDraftProvider(
        {
            "route": "manager_only",
            "draft_text": "Рассрочка есть.",
            "message_type": "question",
            "topic_id": "theme:006_installment",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "Есть рассрочка?",
        context={"active_brand": "foton", "rop_policy": {"bot_permission": "draft_for_manager"}},
    )

    assert result.route == "draft_for_manager"
    assert "Долями" in result.draft_text
    assert "6, 10 или 12 месяцев" in result.draft_text
    assert "4 части" not in result.draft_text
    assert "3, 6 или 10 месяцев" not in result.draft_text
    assert "16,9%" not in result.draft_text
    assert "36" not in result.draft_text


def test_foton_regular_installment_ignores_llm_camp_terms_when_client_asks_course() -> None:
    provider = FakeDraftProvider(
        {
            "route": "bot_answer_self_for_pilot",
            "draft_text": "Для ЛВШ и лагерей доступна рассрочка Т-Банка на 3, 6 или 10 месяцев с комиссией до 16,9%.",
            "message_type": "question",
            "topic_id": "theme:006_installment",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "4 класс математика очно. Можно платить частями?",
        context={
            "active_brand": "foton",
            "autonomy_policy": {"allow_autonomous": True, "allowed_topic_ids": ["theme:006_installment"]},
            "facts_context": {"client_safe": True, "fresh": True, "facts_missing": False},
        },
    )

    assert "6, 10 или 12 месяцев" in result.draft_text
    assert "Долями" in result.draft_text
    assert "4 части" not in result.draft_text
    assert "3, 6 или 10 месяцев" not in result.draft_text
    assert "16,9%" not in result.draft_text


def test_foton_installment_followup_answers_bank_and_monthly_without_camp_self_pollution() -> None:
    provider = FakeDraftProvider(
        {
            "route": "draft_for_manager",
            "draft_text": "Менеджер подскажет оплату частями.",
            "message_type": "question",
            "topic_id": "theme:006_installment",
            "confidence_theme": 0.92,
        }
    )

    result = provider.build_draft(
        "Я про обычные занятия по математике 4 класс очно, не лагеря. Помесячно можно или только через банк? Банк точно одобрит?",
        context={
            "active_brand": "foton",
            "recent_messages": [
                "Клиент: 4 класс математика очно, можно платить частями?",
                "Бот: Да, в Фотоне можно оплатить обучение частями: доступны варианты на 6, 10 или 12 месяцев, а также сервис Долями. Это относится к очным и онлайн-курсам, ЛВШ, ЛШ и другим программам Фотона.",
            ],
            "autonomy_policy": {"allow_autonomous": True, "allowed_topic_ids": ["theme:006_installment"]},
            "facts_context": {"client_safe": True, "fresh": True, "facts_missing": False},
            "dialogue_memory_view": {"known_slots": {"grade": "4", "subject": "математика", "format": "очно"}},
        },
    )

    assert "решение принимает" in result.draft_text
    assert "помесяч" in result.draft_text.casefold()
    assert "одобрение заранее" in result.draft_text
    assert "ЛВШ" not in result.draft_text
    assert "лагер" not in result.draft_text.casefold()


def test_complaint_draft_does_not_apologize_from_company() -> None:
    provider = FakeDraftProvider(
        {
            "route": "draft_for_manager",
            "draft_text": "Понимаю, что вам неприятно. Извините за ситуацию, всё исправим.",
            "message_type": "question",
            "topic_id": "theme:019b_negative_feedback",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "Я возмущена качеством занятия, преподаватель плохо объяснял.",
        context={"active_brand": "foton", "rop_policy": {"bot_permission": "draft_for_manager"}},
    )

    assert result.route == "manager_only"
    assert "complaint_apology_guarded" in result.safety_flags
    lowered = result.draft_text.casefold()
    for forbidden in ("понимаю", "извините", "неприятно", "сожале", "жаль", "уточните"):
        assert forbidden not in lowered


def test_complaint_draft_does_not_collect_lesson_details() -> None:
    provider = FakeDraftProvider(
        {
            "route": "draft_for_manager",
            "draft_text": "Уточните, пожалуйста, дату занятия, предмет, имя ученика и имя преподавателя.",
            "message_type": "question",
            "topic_id": "theme:019b_negative_feedback",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "Хочу оставить жалобу на качество занятия.",
        context={"active_brand": "unpk", "rop_policy": {"bot_permission": "draft_for_manager"}},
    )

    assert "complaint_apology_guarded" in result.safety_flags
    lowered = result.draft_text.casefold()
    for forbidden in ("дат", "предмет", "имя", "преподав"):
        assert forbidden not in lowered


def test_negative_feedback_non_question_is_not_off_topic_fallback() -> None:
    provider = FakeDraftProvider(
        {
            "route": "manager_only",
            "draft_text": "Я помогаю с вопросами об обучении.",
            "message_type": "non_question",
            "topic_id": "service:S3_out_of_scope",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "Вижу, что вы не отвечаете нормально. Тогда не буду оставлять заявку.",
        context={"active_brand": "unpk", "rop_policy": {"bot_permission": "draft_for_manager"}},
    )

    assert result.route == "draft_for_manager"
    assert result.draft_text == "Понял, давайте не буду повторять общий ответ. Передам менеджеру контекст переписки, чтобы он ответил по вашему вопросу точнее."
    assert "complaint_apology_guarded" not in result.safety_flags
    assert "По другим темам" not in result.draft_text


def test_complaint_draft_does_not_collect_details_with_podskazhite() -> None:
    provider = FakeDraftProvider(
        {
            "route": "manager_only",
            "draft_text": "Подскажите, пожалуйста, дату занятия, имя ученика и что именно было непонятно.",
            "message_type": "question",
            "topic_id": "theme:019b_negative_feedback",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "Преподаватель плохо объяснял, ребёнок ничего не понял.",
        context={"active_brand": "foton", "rop_policy": {"bot_permission": "draft_for_manager"}},
    )

    assert result.route == "manager_only"
    assert "complaint_apology_guarded" in result.safety_flags
    lowered = result.draft_text.casefold()
    assert "подскажите" not in lowered
    assert "имя" not in lowered


def test_unpk_installment_uses_approved_fallback_instead_of_bank_deflect() -> None:
    provider = FakeDraftProvider(
        {
            "route": "manager_only",
            "draft_text": "Уточню, через какой банк сейчас доступно оформление.",
            "message_type": "question",
            "topic_id": "theme:006_installment",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "Рассрочку можно оформить только в Т-банке? На каких условиях?",
        context={"active_brand": "unpk", "rop_policy": {"bot_permission": "draft_for_manager"}},
    )

    assert result.route == "draft_for_manager"
    assert result.draft_text == UNPK_INSTALLMENT_APPROVED_FALLBACK_TEXT
    assert "unpk_installment_approved_fallback_applied" in result.safety_flags
    assert "т-банк" not in result.draft_text.casefold()
    assert "рассрочки нет" in result.draft_text.casefold()
    assert "10%" in result.draft_text
    assert "14%" in result.draft_text
    assert "помесячно" in result.draft_text.casefold()
    assert "семестр" in result.draft_text.casefold()


def test_unpk_installment_intent_beats_wrong_teacher_draft() -> None:
    provider = FakeDraftProvider(
        {
            "route": "draft_for_manager",
            "draft_text": "У нас преподают специалисты из МФТИ и МГУ, менеджер подскажет преподавателя.",
            "message_type": "question",
            "topic_id": "theme:016_program",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "Не хочу платить всё сразу, можно как-то частями?",
        context={"active_brand": "unpk", "rop_policy": {"bot_permission": "draft_for_manager"}},
    )

    assert result.route == "draft_for_manager"
    assert result.draft_text == UNPK_INSTALLMENT_APPROVED_FALLBACK_TEXT
    assert "unpk_installment_approved_fallback_applied" in result.safety_flags
    assert "рассрочки нет" in result.draft_text.casefold()
    assert "teacher_safe_template_applied" not in result.safety_flags
    assert "преподают" not in result.draft_text.casefold()


def test_unpk_installment_bank_clarification_is_not_overwritten_by_discount_template() -> None:
    provider = FakeDraftProvider(
        {
            "route": "bot_answer_self_for_pilot",
            "draft_text": "В УНПК можно платить помесячно, за семестр или за год.",
            "message_type": "question",
            "topic_id": "theme:006_installment",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "Помесячно — это без банка и одобрений?",
        context={
            "active_brand": "unpk",
            "autonomy_policy": {"allow_autonomous": True, "allowed_topic_ids": ["theme:006_installment"]},
            "facts_context": {"client_safe": True, "fresh": True, "facts_missing": False},
            "confirmed_facts": {
                "fact:installment": "УНПК: оплата помесячно, за семестр или за год; это не банковская рассрочка.",
            },
        },
    )

    assert result.route == "bot_answer_self_for_pilot"
    assert "unpk_installment_approved_fallback_applied" in result.safety_flags
    assert "discount_safe_template_applied" not in result.safety_flags
    assert "не банковская" in result.draft_text.casefold()
    assert "одобрение банка не требуется" in result.draft_text.casefold()


def test_unpk_installment_thanks_context_update_does_not_repeat_template() -> None:
    provider = FakeDraftProvider(
        {
            "route": "draft_for_manager",
            "draft_text": "Отлично, тогда будем ориентироваться на помесячную оплату.",
            "message_type": "context_update",
            "topic_id": "theme:006_installment",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "Поняла, помесячно удобнее. Спасибо!",
        context={"active_brand": "unpk", "rop_policy": {"bot_permission": "draft_for_manager"}},
    )

    assert "unpk_installment_approved_fallback_applied" not in result.safety_flags
    assert result.draft_text == "Отлично, тогда будем ориентироваться на помесячную оплату."
    assert "Здравствуйте!" not in result.draft_text


def test_unpk_zvsh_uses_waitlist_template_without_collecting_contacts() -> None:
    provider = FakeDraftProvider(
        {
            "route": "manager_only",
            "draft_text": (
                "Подскажите имя ученика, класс и контактный номер, чтобы записать в лист ожидания "
                "на зимнюю выездную школу в Менделеево."
            ),
            "message_type": "question",
            "topic_id": "theme:026_camp_general",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "Когда будет зимняя выездная школа в Менделеево?",
        context={"active_brand": "unpk", "rop_policy": {"bot_permission": "draft_for_manager"}},
    )

    assert result.route == "manager_only"
    assert "unpk_zvsh_waitlist_safe_template_applied" in result.safety_flags
    assert "лист ожидания" in result.draft_text.casefold()
    assert "дат" in result.draft_text.casefold()
    lowered = result.draft_text.casefold()
    for forbidden in ("имя", "класс", "телефон", "номер"):
        assert forbidden not in lowered


def test_unpk_winter_camp_uses_zvsh_waitlist_template_without_mendeleevo_word() -> None:
    provider = FakeDraftProvider(
        {
            "route": "draft_for_manager",
            "draft_text": "Уточню актуальные даты зимнего лагеря по УНПК и вернусь к вам с ответом.",
            "message_type": "question",
            "topic_id": "theme:026_camp_general",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "Когда зимний лагерь?",
        context={"active_brand": "unpk", "rop_policy": {"bot_permission": "draft_for_manager"}},
    )

    assert result.route == "draft_for_manager"
    assert "unpk_zvsh_waitlist_safe_template_applied" in result.safety_flags
    lowered = result.draft_text.casefold()
    assert "лист ожидания" in lowered
    assert "уточню актуальные даты" not in lowered


def test_unpk_installment_approved_fallback_is_not_removed_as_unsupported_promise() -> None:
    provider = FakeDraftProvider(
        {
            "route": "manager_only",
            "draft_text": "Уточню условия.",
            "message_type": "question",
            "topic_id": "theme:006_installment",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "Можно ли оплатить в рассрочку?",
        context={"active_brand": "unpk", "rop_policy": {"bot_permission": "draft_for_manager"}},
    )

    assert result.draft_text == UNPK_INSTALLMENT_APPROVED_FALLBACK_TEXT
    assert result.route == "draft_for_manager"
    assert "10%" in result.draft_text
    assert "14%" in result.draft_text
    assert "unsupported_promise_detected" not in result.safety_flags


def test_high_risk_input_marker_coverage_for_russian_forms() -> None:
    cases = {
        "refund": [
            "Возврат",
            "вернуть деньги",
            "верните мне деньги",
            "верните нам деньги",
            "возвращу оплату",
            "верните оплату",
            "возвратить платеж",
            "расторгнуть договор",
            "отказаться от обучения",
            "забрать деньги",
            "возрат денег",
            "ВОЗВРАТ платежа",
        ],
        "legal": [
            "подам в суд",
            "иск",
            "претензия",
            "роспотребнадзор",
            "по закону обязаны",
            "нарушили права",
            "расторжение договора",
        ],
        "complaint": [
            "жалоба",
            "жалуюсь",
            "возмущена",
            "недовольны",
            "плохо учит",
            "некомпетентный преподаватель",
            "преподаватель ужасный",
        ],
    }

    for marker, texts in cases.items():
        for text in texts:
            assert marker in detect_high_risk_input_markers(text), text


def test_high_risk_input_marker_false_positives_are_not_forced() -> None:
    provider = FakeDraftProvider(
        {
            "route": "draft_for_manager",
            "draft_text": "Здравствуйте! Уточним.",
            "message_type": "question",
            "topic_id": "theme:001_pricing",
            "confidence_theme": 0.86,
        }
    )

    for text in (
        "Можете прислать материалы для возврата к теме?",
        "Жалоба на сайт работает?",
        "Какая скидка действует?",
    ):
        result = provider.build_draft(text, context={"rop_policy": {"bot_permission": "draft_for_manager"}})
        assert result.route == "draft_for_manager", text
        assert "high_risk_input_manager_only" not in result.safety_flags


def test_neutral_discount_theme_is_allowed_as_manager_draft_without_auto_send() -> None:
    result = parse_llm_json(
        '{"route":"draft_for_manager","draft_text":"Здравствуйте! Уточним актуальную скидку.",'
        '"message_type":"question","topic_id":"theme:005_discounts","confidence_theme":0.91}'
    )

    assert result.route == "draft_for_manager"
    assert "manager_approval_required" in result.safety_flags
    assert "high_risk_manager_only" not in result.safety_flags


def test_neutral_discount_question_without_numeric_promise_is_allowed() -> None:
    provider = FakeDraftProvider(
        {
            "route": "draft_for_manager",
            "draft_text": "Здравствуйте! Уточним, какая скидка сейчас действует.",
            "message_type": "question",
            "topic_id": "theme:005_discounts",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "Какая скидка действует?",
        context={"rop_policy": {"bot_permission": "draft_for_manager"}},
    )

    assert result.route == "draft_for_manager"
    assert "unsupported_promise_detected" not in result.safety_flags


def test_cross_brand_online_discount_uses_generic_brand_template_not_platform_answer() -> None:
    provider = FakeDraftProvider(
        {
            "route": "draft_for_manager",
            "draft_text": "В нашем учебном центре онлайн-занятия проходят в МТС Линк / Webinar.",
            "message_type": "question",
            "topic_id": "theme:005_discounts",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "В Фотоне за второй предмет онлайн 30% скидка. У вас тоже 30%?",
        context={"active_brand": "unpk", "rop_policy": {"bot_permission": "draft_for_manager"}},
    )

    assert result.route == "manager_only"
    assert "cross_brand_safe_template_applied" in result.safety_flags
    assert "отдельные организации" in result.draft_text.casefold()
    assert "мтс линк" not in result.draft_text.casefold()


def test_cross_brand_discount_does_not_get_overwritten_by_discount_fallback() -> None:
    provider = FakeDraftProvider(
        {
            "route": "draft_for_manager",
            "draft_text": "Здравствуйте! Уточним, какая скидка сейчас действует.",
            "message_type": "question",
            "topic_id": "theme:005_discounts",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "У УНПК многодетным дают 20%, а у вас сколько?",
        context={"active_brand": "foton", "rop_policy": {"bot_permission": "draft_for_manager"}},
    )

    assert result.route == "manager_only"
    assert "cross_brand_safe_template_applied" in result.safety_flags
    assert "отдельные организации" in result.draft_text.casefold()
    assert "процен" not in result.draft_text.casefold()


def test_draft_with_numeric_discount_without_fresh_fact_is_forced_to_manager_only() -> None:
    provider = FakeDraftProvider(
        {
            "route": "draft_for_manager",
            "draft_text": "Здравствуйте! Для вас действует скидка 10% до 31 мая.",
            "message_type": "question",
            "topic_id": "theme:005_discounts",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "Какая скидка действует?",
        context={"rop_policy": {"bot_permission": "draft_for_manager"}},
    )

    assert result.route == "manager_only"
    assert "unsupported_promise_detected" in result.safety_flags
    assert "10%" in result.forbidden_promises_detected
    assert "до 31 мая" in result.forbidden_promises_detected
    assert result.metadata["unsupported_promises"] == ["10%", "до 31 мая"]


def test_draft_with_numeric_discount_from_fresh_fact_is_allowed() -> None:
    provider = FakeDraftProvider(
        {
            "route": "draft_for_manager",
            "draft_text": "Здравствуйте! Для вас действует скидка 10% до 31 мая.",
            "message_type": "question",
            "topic_id": "theme:005_discounts",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "Какая скидка действует?",
        context={
            "active_brand": "foton",
            "rop_policy": {"bot_permission": "draft_for_manager"},
            "facts_context": {"fresh": True, "discount": "Скидка 10% действует до 31 мая."},
        },
    )

    assert result.route == "draft_for_manager"
    assert "unsupported_promise_detected" not in result.safety_flags


def test_neutral_price_question_is_not_forced_by_input_guard() -> None:
    provider = FakeDraftProvider(
        {
            "route": "draft_for_manager",
            "draft_text": "Здравствуйте! Уточним цену.",
            "message_type": "question",
            "topic_id": "theme:001_pricing",
            "confidence_theme": 0.86,
        }
    )

    result = provider.build_draft(
        "Сколько стоит подготовка по математике?",
        context={"rop_policy": {"bot_permission": "draft_for_manager"}},
    )

    assert result.route == "draft_for_manager"
    assert "high_risk_input_manager_only" not in result.safety_flags


def test_unpk_draft_does_not_allow_bank_installment_wording() -> None:
    provider = FakeDraftProvider(
        {
            "route": "draft_for_manager",
            "draft_text": "Здравствуйте! Возможность рассрочки через банк нужно проверить по конкретному курсу.",
            "message_type": "question",
            "topic_id": "theme:006_installment",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "Можно ли оплатить курс в рассрочку через банк?",
        context={"active_brand": "unpk", "rop_policy": {"bot_permission": "draft_for_manager"}},
    )

    assert result.route == "draft_for_manager"
    assert "unpk_installment_approved_fallback_applied" in result.safety_flags
    assert "рассрочка через банк" not in result.draft_text.casefold()
    assert "помесячно" in result.draft_text.casefold()
    assert "10%" in result.draft_text
    assert "14%" in result.draft_text


def test_unpk_cross_brand_installment_question_answers_only_for_unpk() -> None:
    provider = FakeDraftProvider(
        {
            "route": "draft_for_manager",
            "draft_text": "В Фотоне есть Долями через Т-Банк.",
            "message_type": "question",
            "topic_id": "theme:006_installment",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "А у Фотона есть Долями через Т-Банк — а у вас?",
        context={"active_brand": "unpk", "rop_policy": {"bot_permission": "draft_for_manager"}},
    )

    assert result.route == "draft_for_manager"
    assert result.draft_text == UNPK_INSTALLMENT_APPROVED_FALLBACK_TEXT
    lowered = result.draft_text.casefold()
    assert "фотон" not in lowered
    assert "т-банк" not in lowered
    assert "долями" not in lowered
    assert "10%" in result.draft_text
    assert "14%" in result.draft_text


def test_matkap_federal_question_uses_timing_template_without_document_list() -> None:
    provider = FakeDraftProvider(
        {
            "route": "bot_answer_self_for_pilot",
            "draft_text": "По документам менеджер подскажет.",
            "message_type": "question",
            "topic_id": "theme:007_matkap_payment",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "Маткапиталом оплатить можно и сколько ждать СФР?",
        context={
            "active_brand": "foton",
            "autonomy_policy": {"allow_autonomous": True},
            "client_safe_fact_verified": True,
        },
    )

    assert result.route == "bot_answer_self_for_pilot"
    assert result.draft_text == MATKAP_FEDERAL_TIMING_SAFE_TEXT
    assert "до 10 рабочих дней" in result.draft_text
    assert "до 5 рабочих дней" in result.draft_text
    assert "до 15 рабочих дней" in result.draft_text
    lowered = result.draft_text.casefold()
    assert "паспорт" not in lowered
    assert "снилс" not in lowered


def test_unpk_address_question_is_manager_draft_without_lobnya() -> None:
    provider = FakeDraftProvider(
        {
            "route": "draft_for_manager",
            "draft_text": "Уточню площадки.",
            "message_type": "question",
            "topic_id": "theme:015_location",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "Какие у вас площадки?",
        context={"active_brand": "unpk", "rop_policy": {"bot_permission": "draft_for_manager"}},
    )

    assert result.route == "draft_for_manager"
    assert result.draft_text == ADDRESS_UNPK_SAFE_TEXT
    assert "Лобня" not in result.draft_text
    assert "placeholder_in_draft" not in result.safety_flags


def test_unpk_regular_moscow_address_uses_clean_regular_template() -> None:
    provider = FakeDraftProvider(
        {
            "route": "bot_answer_self_for_pilot",
            "draft_text": "регулярные занятия в Москве — Сретенка, 20 Здравствуйте! Регулярные занятия в Москве проходят по адресу Сретенка, 20.",
            "message_type": "question",
            "topic_id": "theme:015_address",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "По какому адресу обычные занятия в Москве?",
        context={
            "active_brand": "unpk",
            "autonomy_policy": {"allow_autonomous": True},
            "client_safe_fact_verified": True,
        },
    )

    assert result.route == "bot_answer_self_for_pilot"
    assert result.draft_text == ADDRESS_UNPK_MOSCOW_REGULAR_SAFE_TEXT
    assert "Верхняя Красносельская" not in result.draft_text
    assert "Фотон" not in result.draft_text


def test_bot_answer_self_is_demoted_when_topic_not_in_autonomy_matrix() -> None:
    provider = FakeDraftProvider(
        {
            "route": "bot_answer_self_for_pilot",
            "draft_text": "Здравствуйте! Обсудим партнерство.",
            "message_type": "question",
            "topic_id": "theme:030_partnership_b2b",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "Хотим обсудить партнерство с вашей компанией.",
        context={
            "active_brand": "foton",
            "autonomy_policy": {"allow_autonomous": True},
            "client_safe_fact_verified": True,
        },
    )

    assert result.route == "draft_for_manager"
    assert "autonomy_default_cautious_topic_not_allowed" in result.safety_flags


def test_bot_answer_self_requires_client_safe_verified_fact() -> None:
    provider = FakeDraftProvider(
        {
            "route": "bot_answer_self_for_pilot",
            "draft_text": "Здравствуйте! Занятия проходят очно и онлайн.",
            "message_type": "question",
            "topic_id": "theme:014_format",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "Можно учиться онлайн?",
        context={"active_brand": "foton", "autonomy_policy": {"allow_autonomous": True}},
    )

    assert result.route == "draft_for_manager"
    assert "autonomy_default_cautious_unverified_fact" in result.safety_flags


def test_bot_answer_self_allowed_for_safe_topic_with_verified_fact() -> None:
    provider = FakeDraftProvider(
        {
            "route": "bot_answer_self_for_pilot",
            "draft_text": "Здравствуйте! Занятия проходят очно и онлайн.",
            "message_type": "question",
            "topic_id": "theme:014_format",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "Можно учиться онлайн?",
        context={
            "active_brand": "foton",
            "autonomy_policy": {"allow_autonomous": True, "allowed_topic_ids": ["theme:014_format"]},
            "facts_context": {
                "format": "Фотон проводит очные и онлайн-занятия.",
                "client_safe": True,
                "fresh": True,
            },
        },
    )

    assert result.route == "bot_answer_self_for_pilot"
    assert "autonomy_matrix_passed" in result.safety_flags


def test_safe_green_draft_is_promoted_to_autonomous_when_fact_verified() -> None:
    provider = FakeDraftProvider(
        {
            "route": "draft_for_manager",
            "draft_text": "Да, можно учиться онлайн. Занятия проходят на платформе, доступна запись.",
            "message_type": "question",
            "topic_id": "theme:014_format",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "Можно учиться онлайн?",
        context={
            "active_brand": "foton",
            "autonomy_policy": {"allow_autonomous": True, "allowed_topic_ids": ["theme:014_format"]},
            "facts_context": {
                "format": "Фотон проводит онлайн-занятия, доступна запись.",
                "client_safe": True,
                "fresh": True,
            },
        },
    )

    assert result.route == "bot_answer_self_for_pilot"
    assert "autonomy_matrix_promoted_safe_draft" in result.safety_flags
    assert "autonomy_matrix_passed" in result.safety_flags


def test_promoted_autonomous_answer_uses_confirmed_facts_when_original_draft_is_weak() -> None:
    provider = FakeDraftProvider(
        {
            "route": "draft_for_manager",
            "draft_text": "Стоимость зависит от класса и формата.",
            "message_type": "question",
            "topic_id": "theme:001_pricing",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "Сколько стоит очный курс для 5 класса?",
        context={
            "active_brand": "foton",
            "autonomy_policy": {"allow_autonomous": True, "allowed_topic_ids": ["theme:001_pricing"]},
            "facts_context": {"client_safe": True, "fresh": True},
            "confirmed_facts": {
                "fact:semester": "Фотон: 5-11 класс, очно, семестр — 44 600 ₽.",
                "fact:year": "Фотон: 5-11 класс, очно, год — 74 500 ₽.",
            },
        },
    )

    assert result.route == "bot_answer_self_for_pilot"
    assert "autonomy_verified_fact_answer_template_applied" in result.safety_flags
    assert "44 600" in result.draft_text
    assert "74 500" in result.draft_text
    assert "подходящий вариант оплаты" in result.draft_text


def test_foton_online_price_answers_with_confirmed_price_immediately() -> None:
    provider = FakeDraftProvider(
        {
            "route": "draft_for_manager",
            "draft_text": "Онлайн-обучение в Фотоне: есть варианты оплаты за семестр и за год. Менеджер подскажет актуальную стоимость.",
            "message_type": "question",
            "topic_id": "theme:001_pricing",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "Сколько стоит онлайн 8 класс физика?",
        context={
            "active_brand": "foton",
            "autonomy_policy": {"allow_autonomous": True, "allowed_topic_ids": ["theme:001_pricing"]},
            "facts_context": {"client_safe": True, "fresh": True, "facts_missing": False},
            "confirmed_facts": {
                "fact:semester": "Фотон: онлайн, 5-11 классы, семестр — 29 750 ₽.",
                "fact:year": "Фотон: онлайн, 5-11 классы, год — 47 250 ₽.",
            },
        },
    )

    assert result.route == "bot_answer_self_for_pilot"
    assert "pricing_safe_template_applied" in result.safety_flags
    assert "29 750" in result.draft_text
    assert "47 250" in result.draft_text
    assert "Менеджер подскажет актуальную стоимость" not in result.draft_text
    assert "скоро подрастёт" in result.draft_text


def test_foton_followup_price_uses_known_online_format_from_memory() -> None:
    provider = FakeDraftProvider(
        {
            "route": "draft_for_manager",
            "draft_text": "Фотон: 5-11 класс, очно, семестр — 44 600 ₽, год — 74 500 ₽.",
            "message_type": "question",
            "topic_id": "theme:001_pricing",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "А это цена на сейчас?",
        context={
            "active_brand": "foton",
            "autonomy_policy": {"allow_autonomous": True, "allowed_topic_ids": ["theme:001_pricing"]},
            "facts_context": {"client_safe": True, "fresh": True, "facts_missing": False},
            "confirmed_facts": {
                "fact:online_semester": "Фотон: цены на 2026/27 учебный год, 5-11 класс, онлайн, семестр — 47 250 ₽.",
                "fact:online_year": "Фотон: цены на 2026/27 учебный год, 5-11 класс, онлайн, год — 78 750 ₽.",
                "fact:offline_semester": "Фотон: цены на 2026/27 учебный год, 5-11 класс, очно, семестр — 44 600 ₽.",
            },
            "dialogue_memory_view": {"known_slots": {"grade": "8", "subject": "физика", "format": "онлайн"}},
        },
    )

    assert "онлайн" in result.draft_text.casefold()
    assert "47 250" in result.draft_text
    assert "78 750" in result.draft_text
    assert "44 600" not in result.draft_text


def test_foton_online_trial_question_uses_approved_safe_text() -> None:
    provider = FakeDraftProvider(
        {
            "route": "draft_for_manager",
            "draft_text": "Пробное занятие лучше проверить у менеджера.",
            "message_type": "question",
            "topic_id": "theme:020_enrollment",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "Как закрепить год за 47 250 и есть ли пробное онлайн?",
        context={
            "active_brand": "foton",
            "autonomy_policy": {"allow_autonomous": True, "allowed_topic_ids": ["theme:023_trial_class"]},
            "facts_context": {"client_safe": True, "fresh": True, "facts_missing": False},
            "confirmed_facts": {"fact:trial": "Фотон: пробное онлайн есть по умолчанию и оформляется дистанционно."},
        },
    )

    assert result.route == "bot_answer_self_for_pilot"
    assert result.topic_id == "theme:023_trial_class"
    assert "trial_safe_template_applied" in result.safety_flags
    assert "пробное занятие есть по умолчанию" in result.draft_text
    assert "дистанционно" in result.draft_text
    assert "приезжать не нужно" in result.draft_text
    assert "фрагмент занятия" not in result.draft_text
    assert "бесплат" not in result.draft_text.casefold()


def test_foton_trial_remote_question_does_not_invite_visit() -> None:
    provider = FakeDraftProvider(
        {
            "route": "manager_only",
            "draft_text": "Запись и оформление проходят дистанционно. Если нужна личная встреча, её можно согласовать.",
            "message_type": "question",
            "topic_id": "theme:014_format",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "То есть приезжать точно не надо, всё онлайн оформляется?",
        context={
            "active_brand": "foton",
            "autonomy_policy": {"allow_autonomous": True, "allowed_topic_ids": ["theme:014_format", "theme:023_trial_class"]},
            "facts_context": {"client_safe": True, "fresh": True, "facts_missing": False},
            "dialogue_memory_view": {"known_slots": {"format": "онлайн", "product": "пробное"}},
        },
    )

    assert "приезжать не нужно" in result.draft_text
    assert "личная встреча" not in result.draft_text.casefold()
    assert "согласовать" not in result.draft_text.casefold()


def test_unpk_trial_question_does_not_promise_offline_free_trial() -> None:
    provider = FakeDraftProvider(
        {
            "route": "draft_for_manager",
            "draft_text": "Бесплатное пробное на Пацаева есть.",
            "message_type": "question",
            "topic_id": "theme:023_trial_class",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "Можно бесплатное пробное на Пацаева или в МФТИ?",
        context={
            "active_brand": "unpk",
            "autonomy_policy": {"allow_autonomous": True, "allowed_topic_ids": ["theme:023_trial_class"]},
            "facts_context": {"client_safe": True, "fresh": True, "facts_missing": False},
        },
    )

    assert result.route == "bot_answer_self_for_pilot"
    assert "trial_safe_template_applied" in result.safety_flags
    assert "очному формату" in result.draft_text.casefold()
    assert "не начинаем" in result.draft_text
    assert "фрагмент занятия" in result.draft_text
    assert "бесплат" not in result.draft_text.casefold()


def test_unpk_online_fragment_followup_answers_free_and_no_visit_directly() -> None:
    provider = FakeDraftProvider(
        {
            "route": "draft_for_manager",
            "draft_text": "По очному формату сейчас обычно не начинаем с отдельного пробного занятия.",
            "message_type": "question",
            "topic_id": "theme:023_trial_class",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "Я про онлайн, фрагмент этот бесплатно пришлёте? И приезжать никуда не надо?",
        context={
            "active_brand": "unpk",
            "autonomy_policy": {"allow_autonomous": True, "allowed_topic_ids": ["theme:023_trial_class"]},
            "facts_context": {"client_safe": True, "fresh": True, "facts_missing": False},
            "confirmed_facts": {"fact:fragment": "По онлайн-формату можно прислать фрагмент занятия для знакомства с подачей и уровнем."},
            "dialogue_memory_view": {"known_slots": {"format": "онлайн"}},
        },
    )

    assert "онлайн-формату" in result.draft_text
    assert "фрагмент занятия" in result.draft_text
    assert "приезжать" in result.draft_text
    assert "не обещаю" in result.draft_text
    assert "очному формату" not in result.draft_text.casefold()


def test_llm_missing_facts_do_not_block_autonomy_when_context_fact_is_verified() -> None:
    provider = FakeDraftProvider(
        {
            "route": "draft_for_manager",
            "draft_text": "Да, можно учиться онлайн, доступна запись занятий.",
            "message_type": "question",
            "topic_id": "theme:014_format",
            "confidence_theme": 0.91,
            "missing_facts": ["класс ребёнка"],
        }
    )

    result = provider.build_draft(
        "Можно учиться онлайн?",
        context={
            "active_brand": "foton",
            "autonomy_policy": {"allow_autonomous": True, "allowed_topic_ids": ["theme:014_format"]},
            "facts_context": {"client_safe": True, "fresh": True, "facts_missing": False},
            "confirmed_facts": {"fact:format": "Фотон: онлайн-занятия проходят на платформе МТС Линк, записи доступны."},
        },
    )

    assert result.route == "bot_answer_self_for_pilot"
    assert "autonomy_default_cautious_missing_facts" not in result.safety_flags


def test_live_availability_missing_fact_blocks_autonomy_even_with_verified_program_fact() -> None:
    provider = FakeDraftProvider(
        {
            "route": "draft_for_manager",
            "draft_text": "Да, сориентирую по проверенной информации. Фотон: ЛВШ Менделеево — 5-10 класс.",
            "message_type": "question",
            "topic_id": "theme:026_camp_general",
            "confidence_theme": 0.91,
            "missing_facts": ["availability_by_group_or_shift"],
        }
    )

    result = provider.build_draft(
        "А места на физику для 8 класса ещё есть?",
        context={
            "active_brand": "foton",
            "autonomy_policy": {"allow_autonomous": True, "allowed_topic_ids": ["theme:026_camp_general"]},
            "facts_context": {"client_safe": True, "fresh": True, "facts_missing": False},
            "confirmed_facts": {"fact:camp": "Фотон: ЛВШ Менделеево рассчитана на 5-10 класс, есть физика."},
            "dialogue_memory_view": {"known_slots": {"grade": "8", "subject": "физика", "product": "ЛВШ"}},
        },
    )

    assert result.route == "draft_for_manager"
    assert "autonomy_default_cautious_live_status_missing" in result.safety_flags
    assert "не буду обещать без проверки" in result.draft_text
    assert "8 класс" in result.draft_text
    assert "физика" in result.draft_text
    assert "Если напишете класс" not in result.draft_text


def test_live_availability_fixation_question_answers_process_not_repeated_handoff() -> None:
    provider = FakeDraftProvider(
        {
            "route": "draft_for_manager",
            "draft_text": "Передам менеджеру, он проверит наличие мест.",
            "message_type": "question",
            "topic_id": "theme:026_camp_general",
            "confidence_theme": 0.91,
            "missing_facts": ["availability_by_group_or_shift"],
        }
    )

    result = provider.build_draft(
        "Хорошо, а как можно закрепить место?",
        context={
            "active_brand": "foton",
            "autonomy_policy": {"allow_autonomous": True, "allowed_topic_ids": ["theme:026_camp_general"]},
            "facts_context": {"client_safe": True, "fresh": True, "facts_missing": False},
            "confirmed_facts": {"fact:camp": "Фотон: ЛВШ Менделеево рассчитана на 5-10 класс, есть физика."},
            "dialogue_memory_view": {"known_slots": {"grade": "8", "subject": "физика", "product": "ЛВШ"}},
        },
    )

    assert result.route == "draft_for_manager"
    assert "Сначала менеджер проверит наличие" in result.draft_text
    assert "Если место есть" in result.draft_text
    assert "оформление заявки" in result.draft_text
    assert "место точно доступно" in result.draft_text


def test_live_availability_data_needed_question_uses_known_slots() -> None:
    provider = FakeDraftProvider(
        {
            "route": "draft_for_manager",
            "draft_text": "Передам менеджеру, он проверит наличие мест.",
            "message_type": "question",
            "topic_id": "theme:026_camp_general",
            "confidence_theme": 0.91,
            "missing_facts": ["availability_by_group_or_shift"],
        }
    )

    result = provider.build_draft(
        "Давайте, что от меня нужно для проверки мест?",
        context={
            "active_brand": "foton",
            "autonomy_policy": {"allow_autonomous": True, "allowed_topic_ids": ["theme:026_camp_general"]},
            "facts_context": {"client_safe": True, "fresh": True, "facts_missing": False},
            "confirmed_facts": {"fact:camp": "Фотон: ЛВШ Менделеево рассчитана на 5-10 класс, есть физика."},
            "dialogue_memory_view": {"known_slots": {"grade": "8", "subject": "физика", "product": "ЛВШ"}},
        },
    )

    assert "уже вижу: 8 класс, физика" in result.draft_text
    assert "Повторно присылать это не нужно" in result.draft_text
    assert "предпочтение по датам смены" in result.draft_text
    assert "Напишите класс" not in result.draft_text


def test_missing_price_fact_uses_helpful_template_without_autonomy() -> None:
    provider = FakeDraftProvider(
        {
            "route": "draft_for_manager",
            "draft_text": "Передам вопрос менеджеру, он свяжется.",
            "message_type": "question",
            "topic_id": "theme:001_pricing",
            "confidence_theme": 0.91,
            "missing_facts": ["prices.current"],
        }
    )

    result = provider.build_draft(
        "Сколько стоит обучение?",
        context={
            "active_brand": "foton",
            "autonomy_policy": {"allow_autonomous": True, "allowed_topic_ids": ["theme:001_pricing"]},
            "missing_facts": ["prices.current"],
            "facts_context": {"facts_missing": True, "client_safe": False, "fresh": False},
        },
    )

    assert result.route == "draft_for_manager"
    assert "missing_fact_helpful_template_applied" in result.safety_flags
    assert "класс ребёнка" in result.draft_text
    assert "очно или онлайн" in result.draft_text
    assert "₽" not in result.draft_text
    assert "autonomy_matrix_promoted_safe_draft" not in result.safety_flags


def test_missing_intensive_price_uses_specific_helpful_template_without_numbers() -> None:
    provider = FakeDraftProvider(
        {
            "route": "draft_for_manager",
            "draft_text": "Менеджер свяжется и подскажет актуальную программу.",
            "message_type": "question",
            "topic_id": "theme:001_pricing",
            "confidence_theme": 0.91,
            "missing_facts": ["prices.current"],
        }
    )

    result = provider.build_draft(
        "Сколько сейчас стоит интенсив ОГЭ?",
        context={
            "active_brand": "foton",
            "autonomy_policy": {"allow_autonomous": True, "allowed_topic_ids": ["theme:001_pricing"]},
            "missing_facts": ["prices.current"],
            "facts_context": {"facts_missing": True, "client_safe": False, "fresh": False},
        },
    )

    assert result.route == "draft_for_manager"
    assert "missing_fact_helpful_template_applied" in result.safety_flags
    assert "интенсив" in result.draft_text.casefold()
    assert "класс" in result.draft_text.casefold()
    assert "предмет" in result.draft_text.casefold()
    assert "₽" not in result.draft_text
    assert "руб" not in result.draft_text.casefold()


def test_missing_camp_date_asks_safe_followup_without_numbers() -> None:
    provider = FakeDraftProvider(
        {
            "route": "draft_for_manager",
            "draft_text": "Менеджер проверит дату и время заезда и напишет Вам точные детали.",
            "message_type": "question",
            "topic_id": "theme:027_camp_living_conditions",
            "confidence_theme": 0.91,
            "missing_facts": ["точная дата заезда"],
        }
    )

    result = provider.build_draft(
        "Когда точно заезд в лагерь Менделеево?",
        context={
            "active_brand": "foton",
            "facts_context": {"missing": True, "facts_missing": True},
        },
    )

    assert result.route == "draft_for_manager"
    assert "missing_fact_helpful_template_applied" in result.safety_flags
    assert "Напишите" in result.draft_text
    assert "класс" in result.draft_text.casefold()
    assert "направление" in result.draft_text.casefold()
    assert "₽" not in result.draft_text


def test_tax_amount_question_uses_amount_formula_template() -> None:
    provider = FakeDraftProvider(
        {
            "route": "draft_for_manager",
            "draft_text": "ФНС рассматривает заявление и принимает решение.",
            "message_type": "question",
            "topic_id": "theme:008_tax_deduction",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "Какую сумму вернёт налоговый вычет за год?",
        context={
            "active_brand": "foton",
            "autonomy_policy": {"allow_autonomous": True, "allowed_topic_ids": ["theme:008_tax_deduction"]},
            "facts_context": {"client_safe": True, "fresh": True, "facts_missing": False},
            "confirmed_facts": {"fact:tax": "Налоговый вычет: до 14 300 ₽, 13% с расходов до 110 000 ₽."},
        },
    )

    assert result.route == "bot_answer_self_for_pilot"
    assert "tax_safe_template_applied" in result.safety_flags
    assert "14 300" in result.draft_text
    assert "13%" in result.draft_text
    assert "110 000" in result.draft_text
    assert "2023" in result.draft_text
    assert "50 000" in result.draft_text
    assert "6 500" in result.draft_text
    assert "28 600" in result.draft_text
    assert "ФНС" in result.draft_text


def test_foton_moscow_address_uses_verified_autonomous_template() -> None:
    provider = FakeDraftProvider(
        {
            "route": "draft_for_manager",
            "draft_text": "Подскажу адрес после проверки.",
            "message_type": "question",
            "topic_id": "theme:015_address",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "Где вы находитесь в Москве?",
        context={
            "active_brand": "foton",
            "autonomy_policy": {"allow_autonomous": True, "allowed_topic_ids": ["theme:015_address"]},
        },
    )

    assert result.route == "bot_answer_self_for_pilot"
    assert result.draft_text == ADDRESS_FOTON_MOSCOW_SAFE_TEXT
    assert "Верхняя Красносельская" in result.draft_text
    assert "УНПК" not in result.draft_text


def test_current_price_with_deadline_is_not_blocked_by_future_context() -> None:
    provider = FakeDraftProvider(
        {
            "route": "draft_for_manager",
            "draft_text": "При раннем бронировании до 1 июля очный год стоит 74 500 ₽, семестр — 44 600 ₽.",
            "message_type": "question",
            "topic_id": "theme:001_pricing",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "Очный год 7 класс сколько стоит сейчас?",
        context={
            "active_brand": "foton",
            "autonomy_policy": {"allow_autonomous": True, "allowed_topic_ids": ["theme:001_pricing"]},
            "facts_context": {"client_safe": True, "fresh": True, "facts_missing": False},
            "confirmed_facts": {
                "fact:early": "При раннем бронировании до 1 июля очный год — 74 500 ₽.",
                "fact:semester": "При раннем бронировании до 1 июля семестр — 44 600 ₽.",
                "fact:future": "После 1 июля цена меняется.",
            },
        },
    )

    assert result.route == "bot_answer_self_for_pilot"
    assert "future_price_handoff_applied" not in result.safety_flags
    assert "74 500" in result.draft_text
    assert "до 1 июля" not in result.draft_text
    assert "текущ" in result.draft_text.casefold()
    assert "сейчас" in result.draft_text.casefold()


def test_current_price_numeric_deadline_is_softened() -> None:
    provider = FakeDraftProvider(
        {
            "route": "bot_answer_self_for_pilot",
            "draft_text": "Для 3 класса очно: семестр — 31 000 ₽, год — 51 700 ₽. Это цена сейчас, до 01.07.2026.",
            "message_type": "question",
            "topic_id": "theme:001_pricing",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "3 класс, математика, очно. Это цена на сейчас или скоро поменяется?",
        context={
            "active_brand": "unpk",
            "known_dialog_fields": {"grade": "3", "subject": "математика", "format": "очно"},
            "autonomy_policy": {"allow_autonomous": True, "allowed_topic_ids": ["theme:001_pricing"]},
            "facts_context": {"client_safe": True, "fresh": True, "facts_missing": False},
            "confirmed_facts": {
                "fact:semester": "УНПК: 1-4 класс, очно, семестр — 31 000 ₽.",
                "fact:year": "УНПК: 1-4 класс, очно, год — 51 700 ₽.",
            },
        },
    )

    assert "31 000" in result.draft_text
    assert "51 700" in result.draft_text
    assert "01.07" not in result.draft_text
    assert "до 1 июля" not in result.draft_text
    assert "текущ" in result.draft_text.casefold()
    assert "сейчас" in result.draft_text.casefold()


def test_current_online_price_august_deadline_is_softened() -> None:
    provider = FakeDraftProvider(
        {
            "route": "bot_answer_self_for_pilot",
            "draft_text": "Онлайн 8 класс: семестр — 29 750 ₽, год — 47 250 ₽. По текущим данным такие условия указаны для периода до 1 августа 2026 года; после этого стоимость может измениться.",
            "message_type": "question",
            "topic_id": "theme:001_pricing",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "Это цена прямо на сейчас? Потом поменяется?",
        context={
            "active_brand": "foton",
            "known_slots": {"grade": "8", "subject": "физика", "format": "онлайн"},
            "autonomy_policy": {"allow_autonomous": True, "allowed_topic_ids": ["theme:001_pricing"]},
            "facts_context": {"client_safe": True, "fresh": True, "facts_missing": False},
            "confirmed_facts": {
                "fact:semester": "Фотон: онлайн 5-11, семестр — 29 750 ₽.",
                "fact:year": "Фотон: онлайн 5-11, год — 47 250 ₽.",
            },
        },
    )

    assert "29 750" in result.draft_text
    assert "47 250" in result.draft_text
    assert "1 августа" not in result.draft_text
    assert "01.08" not in result.draft_text
    assert "текущ" in result.draft_text.casefold()
    assert "сейчас" in result.draft_text.casefold()


def test_current_price_direct_date_answer_is_softened_without_fixation_promise() -> None:
    provider = FakeDraftProvider(
        {
            "route": "bot_answer_self_for_pilot",
            "draft_text": (
                "Да, уточняю: дата — 1 июля 2026. Цена 74 500 ₽ за год для очного формата "
                "5–11 классов в Фотоне указана как действующая. Сейчас по дате вы укладываетесь; "
                "отдельно нужно проверить место в группе и передать оформление по текущим условиям."
            ),
            "message_type": "question",
            "topic_id": "theme:001_pricing",
            "confidence_theme": 0.91,
            "manager_checklist": [
                "Проверить наличие места в очной группе 8 класса по информатике",
                "Подтвердить, что оформление возможно по цене 74 500 ₽ до 01.07.2026",
            ],
        }
    )

    result = provider.build_draft(
        "После какой даты? Мне надо понять, успеваю ли по 74500 за год",
        context={
            "active_brand": "foton",
            "known_slots": {"grade": "8", "subject": "информатика", "format": "очно"},
            "autonomy_policy": {"allow_autonomous": True, "allowed_topic_ids": ["theme:001_pricing"]},
            "facts_context": {"client_safe": True, "fresh": True, "facts_missing": False},
            "confirmed_facts": {
                "fact:year": "Фотон: цены на 2026/27 учебный год, 5-11 класс, очно, год — 74 500 ₽.",
            },
        },
    )

    assert "74 500" in result.draft_text
    assert "1 июля" not in result.draft_text
    assert "01.07" not in result.draft_text
    assert "укладываетесь" not in result.draft_text.casefold()
    assert "успева" not in result.draft_text.casefold()
    assert "передать оформление" not in result.draft_text.casefold()
    assert "Точную дату изменения цены менеджер подтвердит" in result.draft_text
    assert all("1 июля" not in item and "01.07" not in item for item in result.manager_checklist)
    assert "current_price_deadline_softened" in result.safety_flags


def test_price_deadline_softener_does_not_leave_broken_fixation_fragment() -> None:
    provider = FakeDraftProvider(
        {
            "route": "bot_answer_self_for_pilot",
            "draft_text": (
                "Да, это цена на сейчас: очно 8 класс, информатика — семестр 44 600 ₽, год 74 500 ₽. "
                "Раннее бронирование позволяет зафиксировать цену + выбрать день недели до 1 июля. "
                "Могу передать менеджеру, чтобы проверил группу и подсказал, как зафиксировать текущие условия до 1 июля."
            ),
            "message_type": "question",
            "topic_id": "theme:001_pricing",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "Это цена на сейчас или скоро поменяется?",
        context={
            "active_brand": "foton",
            "known_slots": {"grade": "8", "subject": "информатика", "format": "очно"},
            "autonomy_policy": {"allow_autonomous": True, "allowed_topic_ids": ["theme:001_pricing"]},
            "facts_context": {"client_safe": True, "fresh": True, "facts_missing": False},
            "confirmed_facts": {
                "fact:semester": "Фотон: цены на 2026/27 учебный год, 5-11 класс, очно, семестр — 44 600 ₽.",
                "fact:year": "Фотон: цены на 2026/27 учебный год, 5-11 класс, очно, год — 74 500 ₽.",
            },
        },
    )

    assert "1 июля" not in result.draft_text
    assert "позволяет Оформление" not in result.draft_text
    assert "как Оформление" not in result.draft_text
    assert "как оформить по текущим условиям" in result.draft_text
    assert "Оформление по текущим условиям проверит менеджер" not in result.draft_text


def test_price_fixation_phrase_is_softened_even_without_explicit_deadline() -> None:
    provider = FakeDraftProvider(
        {
            "route": "bot_answer_self_for_pilot",
            "draft_text": (
                "Да, это цена на сейчас: очно 8 класс, информатика — семестр 44 600 ₽, год 74 500 ₽. "
                "Раннее бронирование позволяет зафиксировать цену + выбрать день недели. "
                "Могу передать менеджеру, чтобы проверил группу и подсказал, как зафиксировать текущие условия."
            ),
            "message_type": "question",
            "topic_id": "theme:001_pricing",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "Это цена на сейчас или скоро поменяется?",
        context={
            "active_brand": "foton",
            "known_slots": {"grade": "8", "subject": "информатика", "format": "очно"},
            "autonomy_policy": {"allow_autonomous": True, "allowed_topic_ids": ["theme:001_pricing"]},
            "facts_context": {"client_safe": True, "fresh": True, "facts_missing": False},
            "confirmed_facts": {
                "fact:semester": "Фотон: цены на 2026/27 учебный год, 5-11 класс, очно, семестр — 44 600 ₽.",
                "fact:year": "Фотон: цены на 2026/27 учебный год, 5-11 класс, очно, год — 74 500 ₽.",
            },
        },
    )

    assert "зафиксировать" not in result.draft_text.casefold()
    assert "позволяет Оформление" not in result.draft_text
    assert "как Оформление" not in result.draft_text
    assert "Чтобы Оформление" not in result.draft_text
    assert "как оформить по текущим условиям" in result.draft_text


def test_unpk_four_weeks_new_price_uses_verified_fact_template() -> None:
    provider = FakeDraftProvider(
        {
            "route": "draft_for_manager",
            "draft_text": "Для нового ученика 4 недели стоят 9 900 ₽.",
            "message_type": "question",
            "topic_id": "theme:001_pricing",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "Сколько стоит 4 недели для нового ученика?",
        context={
            "active_brand": "unpk",
            "autonomy_policy": {"allow_autonomous": True, "allowed_topic_ids": ["theme:001_pricing"]},
            "facts_context": {"client_safe": True, "fresh": True, "facts_fresh": True},
            "confirmed_facts": {"fact:4weeks": "4 недели: ориентир 10 900 ₽, для новых учеников 9 900 ₽."},
        },
    )

    assert result.route == "bot_answer_self_for_pilot"
    assert result.draft_text == UNPK_FOUR_WEEKS_NEW_PRICE_SAFE_TEXT
    assert "10 900" in result.draft_text
    assert "9 900" in result.draft_text


def test_unpk_ege_intensive_price_normalizes_to_program_topic_for_autonomy() -> None:
    provider = FakeDraftProvider(
        {
            "route": "draft_for_manager",
            "draft_text": "Для 11 класса перед ЕГЭ есть онлайн-интенсив на 8 недель. Один предмет — 18 800 ₽, два предмета — 34 400 ₽.",
            "message_type": "question",
            "topic_id": "theme:001_pricing",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "Что есть для 11 класса перед ЕГЭ и сколько стоит?",
        context={
            "active_brand": "unpk",
            "autonomy_policy": {"allow_autonomous": True, "allowed_topic_ids": ["theme:016_program"]},
            "facts_context": {"client_safe": True, "fresh": True, "facts_fresh": True},
            "confirmed_facts": {
                "fact:ege_intensive": "ЕГЭ-интенсив 10-11 класс: онлайн, 8 недель; 1 предмет 18 800 ₽, 2 предмета 34 400 ₽."
            },
        },
    )

    assert result.route == "bot_answer_self_for_pilot"
    assert result.topic_id == "theme:016_program"
    assert result.draft_text == UNPK_EGE_INTENSIVE_PRICE_SAFE_TEXT
    assert "program_topic_normalized" in result.safety_flags


def test_future_price_handoff_is_not_overwritten_by_discount_template() -> None:
    provider = FakeDraftProvider(
        {
            "route": "draft_for_manager",
            "draft_text": "Стоимость уточнит менеджер.",
            "message_type": "question",
            "topic_id": "theme:001_pricing",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "А если платить в августе, очно семестр почём?",
        context={
            "active_brand": "unpk",
            "autonomy_policy": {"allow_autonomous": True, "allowed_topic_ids": ["theme:001_pricing"]},
            "facts_context": {"client_safe": True, "fresh": True, "facts_missing": False},
            "confirmed_facts": {"fact:future": "После 1 июля очный семестр — от 36 900 до 49 000 ₽."},
        },
    )

    assert result.route == "manager_only"
    assert "future_price_handoff_applied" in result.safety_flags
    assert "unpk_installment_approved_fallback_applied" not in result.safety_flags
    assert "10%" not in result.draft_text
    assert "14%" not in result.draft_text


def test_unconfirmed_followup_deadline_is_blocked() -> None:
    provider = FakeDraftProvider(
        {
            "route": "bot_answer_self_for_pilot",
            "draft_text": "Менеджер свяжется не позднее завтра, 22 мая, и всё подскажет.",
            "message_type": "question",
            "topic_id": "theme:013_schedule",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "Когда мне ответят?",
        context={"active_brand": "unpk", "rop_policy": {"bot_permission": "draft_for_manager"}},
    )

    assert result.route == "manager_only"
    assert "unsupported_followup_deadline_detected" in result.safety_flags
    assert "22 мая" not in result.draft_text
    assert "завтра" not in result.draft_text.casefold()
    assert "рабочее время" in result.draft_text


def test_unconfirmed_schedule_assumption_is_blocked() -> None:
    provider = FakeDraftProvider(
        {
            "route": "bot_answer_self_for_pilot",
            "draft_text": "По физике очные группы чаще подбирают на выходных, можно смотреть такой вариант.",
            "message_type": "question",
            "topic_id": "theme:013_schedule",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "9 класс, физика, какие дни занятий?",
        context={"active_brand": "foton", "rop_policy": {"bot_permission": "draft_for_manager"}},
    )

    assert result.route == "manager_only"
    assert "unsupported_schedule_assumption_detected" in result.safety_flags
    assert "чаще" not in result.draft_text.casefold()
    assert "выходных" not in result.draft_text.casefold()
    assert "Точное расписание зависит" in result.draft_text


def test_confirmed_schedule_assumption_is_allowed_when_fact_matches() -> None:
    provider = FakeDraftProvider(
        {
            "route": "draft_for_manager",
            "draft_text": "По физике очные группы чаще подбирают на выходных, можно смотреть такой вариант.",
            "message_type": "question",
            "topic_id": "theme:013_schedule",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "9 класс, физика, какие дни занятий?",
        context={
            "active_brand": "foton",
            "rop_policy": {"bot_permission": "draft_for_manager"},
            "facts_context": {"fresh": True, "facts_missing": False},
            "confirmed_facts": {"fact:schedule": "По физике очные группы чаще подбирают на выходных."},
        },
    )

    assert "unsupported_schedule_assumption_detected" not in result.safety_flags
    assert result.draft_text.startswith("По физике")


def test_foton_lvsh_price_template_names_current_verified_price() -> None:
    provider = FakeDraftProvider(
        {
            "route": "draft_for_manager",
            "draft_text": "Летняя выездная школа стоит от 75 000 ₽, полный тариф — 98 000 ₽, взнос — 15 000 ₽.",
            "message_type": "question",
            "topic_id": "theme:026_camp_general",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "Сколько стоит выездной лагерь в Менделеево?",
        context={
            "active_brand": "foton",
            "autonomy_policy": {"allow_autonomous": True, "allowed_topic_ids": ["theme:026_camp_general"]},
            "facts_context": {"client_safe": True, "fresh": True, "facts_missing": False},
            "confirmed_facts": {
                "fact:price": "ЛВШ Менделеево Фотон: текущая цена 93 100 ₽ до 1 июня, полная цена 98 000 ₽.",
            },
            "knowledge_snippets": ["ЛВШ Менделеево: взнос 15 000 ₽."],
        },
    )

    assert result.route == "bot_answer_self_for_pilot"
    assert "unsupported_promise_detected" not in result.safety_flags
    assert "93 100" in result.draft_text
    assert "98 000" in result.draft_text
    assert "1 июня" not in result.draft_text
    assert "75 000" not in result.draft_text
    assert "15 000" not in result.draft_text
    assert "класс" in result.draft_text.casefold()
    assert "возраст" not in result.draft_text.casefold()


def test_foton_camp_general_question_mentions_both_main_formats() -> None:
    provider = FakeDraftProvider(
        {
            "route": "draft_for_manager",
            "draft_text": "Есть летний лагерь в Москве, менеджер подскажет детали.",
            "message_type": "question",
            "topic_id": "theme:026_camp_general",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "Расскажите про летний лагерь",
        context={
            "active_brand": "foton",
            "autonomy_policy": {"allow_autonomous": True, "allowed_topic_ids": ["theme:026_camp_general"]},
            "facts_context": {"client_safe": True, "fresh": True, "facts_missing": False},
            "confirmed_facts": {
                "fact:camp": "У Фотона есть выездная школа в Менделеево и городская летняя школа в Москве.",
            },
        },
    )

    assert result.route == "bot_answer_self_for_pilot"
    assert "camp_safe_template_applied" in result.safety_flags
    assert "Менделеево" in result.draft_text
    assert "городская летняя школа" in result.draft_text
    assert "наличие мест" in result.draft_text.casefold()


def test_foton_outbound_camp_living_question_answers_living_meals_transfer() -> None:
    provider = FakeDraftProvider(
        {
            "route": "draft_for_manager",
            "draft_text": "Менеджер уточнит условия лагеря.",
            "message_type": "question",
            "topic_id": "theme:027_camp_living_conditions",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "8 класс, физика. Выездной лагерь: проживание, питание и трансфер входят?",
        context={
            "active_brand": "foton",
            "known_dialog_fields": {"grade": "8", "subject": "физика"},
            "autonomy_policy": {"allow_autonomous": True, "allowed_topic_ids": ["theme:027_camp_living_conditions"]},
            "facts_context": {"client_safe": True, "fresh": True, "facts_missing": False},
            "confirmed_facts": {
                "fact:lvsh_living": "Фотон: в ЛВШ Менделеево есть проживание, 5-разовое питание и трансфер из Москвы.",
            },
        },
    )

    assert result.route == "bot_answer_self_for_pilot"
    assert "проживание" in result.draft_text.casefold()
    assert "5-разовое питание" in result.draft_text
    assert "трансфер" in result.draft_text.casefold()
    assert "городской летний лагерь" not in result.draft_text.casefold()
    assert "онлайн" not in result.draft_text.casefold()


def test_foton_camp_living_question_is_not_overwritten_by_missing_shift_fact() -> None:
    provider = FakeDraftProvider(
        {
            "route": "draft_for_manager",
            "draft_text": (
                "Поняла: 8 класс, физика. У Фотона есть выездная школа в Менделеево "
                "и городская летняя школа в Москве. Менеджер проверит подходящую смену."
            ),
            "message_type": "question",
            "topic_id": "theme:027_camp_living_conditions",
            "confidence_theme": 0.91,
            "missing_facts": ["availability_by_shift"],
        }
    )

    result = provider.build_draft(
        "А в Менделеево что входит в смену, проживание и питание отдельно?",
        context={
            "active_brand": "foton",
            "known_dialog_fields": {"grade": "8", "subject": "физика"},
            "autonomy_policy": {"allow_autonomous": True, "allowed_topic_ids": ["theme:027_camp_living_conditions"]},
            "facts_context": {"client_safe": True, "fresh": True, "facts_missing": True},
            "confirmed_facts": {
                "fact:lvsh_living": "Фотон: в ЛВШ Менделеево есть проживание, 5-разовое питание и трансфер из Москвы.",
                "fact:lvsh_transfer": "Фотон: трансфер из Москвы включён в стоимость ЛВШ Менделеево.",
            },
            "dialogue_memory_view": {"known_slots": {"grade": "8", "subject": "физика", "product": "ЛВШ"}},
        },
    )

    assert "проживание" in result.draft_text.casefold()
    assert "5-разовое питание" in result.draft_text
    assert "трансфер" in result.draft_text.casefold()
    assert "Напишите, пожалуйста, класс" not in result.draft_text
    assert "missing_fact_helpful_template_applied" not in result.safety_flags


def test_unpk_lvsh_price_uses_camp_price_template_not_generic_price() -> None:
    provider = FakeDraftProvider(
        {
            "route": "draft_for_manager",
            "draft_text": "Стоимость зависит от класса и периода оплаты.",
            "message_type": "question",
            "topic_id": "theme:026_camp_general",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "Сколько стоит ЛВШ Менделеево?",
        context={
            "active_brand": "unpk",
            "autonomy_policy": {"allow_autonomous": True, "allowed_topic_ids": ["theme:026_camp_general"]},
            "facts_context": {"client_safe": True, "fresh": True, "facts_missing": False},
            "confirmed_facts": {
                "fact:price": "ЛВШ Менделеево УНПК: текущая цена 114 000 ₽, места почти распроданы.",
            },
        },
    )

    assert result.route == "bot_answer_self_for_pilot"
    assert "camp_safe_template_applied" in result.safety_flags
    assert "114 000" in result.draft_text
    assert "почти распроданы" in result.draft_text
    assert "класс" in result.draft_text.casefold()
    assert "возраст" not in result.draft_text.casefold()
    assert "очно или онлайн" not in result.draft_text.casefold()


def test_unpk_lvsh_price_uses_previous_mendeleevo_context() -> None:
    provider = FakeDraftProvider(
        {
            "route": "draft_for_manager",
            "draft_text": "Стоимость лагеря уточнит менеджер.",
            "message_type": "question",
            "topic_id": "theme:001_pricing",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "А полная цена какая примерно сейчас?",
        context={
            "active_brand": "unpk",
            "recent_messages": [
                "Клиент: Расскажите про летнюю выездную школу в Менделеево.",
                "Бот: Есть ЛВШ Менделеево.",
            ],
            "autonomy_policy": {"allow_autonomous": True, "allowed_topic_ids": ["theme:026_camp_general"]},
            "facts_context": {"client_safe": True, "fresh": True, "facts_missing": False},
            "confirmed_facts": {
                "fact:price": "ЛВШ Менделеево УНПК: текущая цена 114 000 ₽, места почти распроданы.",
            },
        },
    )

    assert result.route == "bot_answer_self_for_pilot"
    assert result.topic_id == "theme:026_camp_general"
    assert "camp_safe_template_applied" in result.safety_flags
    assert "114 000" in result.draft_text
    assert "почти распроданы" in result.draft_text
    assert "15 000" not in result.draft_text


def test_unpk_lvsh_price_does_not_reask_known_grade() -> None:
    provider = FakeDraftProvider(
        {
            "route": "draft_for_manager",
            "draft_text": "Напишите класс ребёнка, чтобы подобрать смену.",
            "message_type": "question",
            "topic_id": "theme:026_camp_general",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "А цену по Менделеево можете сказать?",
        context={
            "active_brand": "unpk",
            "known_dialog_fields": {"grade": "11", "subject": "физика"},
            "recent_messages": ["Клиент: 11 класс, физика. Выездной лагерь с проживанием есть?"],
            "autonomy_policy": {"allow_autonomous": True, "allowed_topic_ids": ["theme:026_camp_general"]},
            "facts_context": {"client_safe": True, "fresh": True, "facts_missing": False},
            "confirmed_facts": {
                "fact:price": "ЛВШ Менделеево УНПК: текущая цена 114 000 ₽, места почти распроданы.",
            },
        },
    )

    assert result.route == "bot_answer_self_for_pilot"
    assert "114 000" in result.draft_text
    assert "5-10 класс" in result.draft_text
    assert "применимость" in result.draft_text
    assert "информатике" not in result.draft_text.casefold()
    assert "под ваш предмет" in result.draft_text
    assert "Напишите класс" not in result.draft_text
    assert "asked_known_data_again" not in result.safety_flags


def test_unpk_lvsh_price_is_not_overwritten_by_missing_seats_fact() -> None:
    provider = FakeDraftProvider(
        {
            "route": "draft_for_manager",
            "draft_text": "Напишите класс ребёнка, чтобы подобрать смену.",
            "message_type": "question",
            "topic_id": "theme:026_camp_general",
            "confidence_theme": 0.91,
            "missing_facts": ["наличие мест"],
        }
    )

    result = provider.build_draft(
        "11 класс, физика. ЛВШ Менделеево сколько стоит?",
        context={
            "active_brand": "unpk",
            "known_dialog_fields": {"grade": "11", "subject": "физика"},
            "autonomy_policy": {"allow_autonomous": True, "allowed_topic_ids": ["theme:026_camp_general"]},
            "facts_context": {"client_safe": True, "fresh": True, "facts_missing": True},
            "confirmed_facts": {
                "fact:price": "ЛВШ Менделеево УНПК: текущая цена 114 000 ₽, места почти распроданы.",
            },
        },
    )

    assert "114 000" in result.draft_text
    assert "5-10 класс" in result.draft_text
    assert "применимость" in result.draft_text
    assert "информатике" not in result.draft_text.casefold()
    assert "под ваш предмет" in result.draft_text
    assert "missing_fact_helpful_template_applied" not in result.safety_flags
    assert "Напишите класс" not in result.draft_text


def test_unpk_camp_general_question_uses_overview_not_city_price_dump() -> None:
    provider = FakeDraftProvider(
        {
            "route": "bot_answer_self_for_pilot",
            "draft_text": "Есть городская смена 20-31 июля за 37 500 ₽.",
            "message_type": "question",
            "topic_id": "theme:026_camp_general",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "Здравствуйте, расскажите про летний лагерь",
        context={
            "active_brand": "unpk",
            "autonomy_policy": {"allow_autonomous": True, "allowed_topic_ids": ["theme:026_camp_general"]},
            "facts_context": {"client_safe": True, "fresh": True, "facts_missing": False},
        },
    )

    assert result.route == "bot_answer_self_for_pilot"
    assert "выездная ЛВШ" in result.draft_text
    assert "городская летняя школа" in result.draft_text
    assert "37 500" not in result.draft_text
    assert "20-31 июля" not in result.draft_text


def test_unpk_camp_online_question_does_not_overstate_online_program() -> None:
    provider = FakeDraftProvider(
        {
            "route": "bot_answer_self_for_pilot",
            "draft_text": "У УНПК есть онлайн-курсы по физике для 11 класса.",
            "message_type": "question",
            "topic_id": "theme:014_format",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "А онлайн формат есть или лагерь только очно?",
        context={
            "active_brand": "unpk",
            "recent_messages": ["Клиент: расскажите про ЛВШ Менделеево"],
            "known_dialog_fields": {"grade": "11", "subject": "физика"},
            "autonomy_policy": {"allow_autonomous": True, "allowed_topic_ids": ["theme:014_format", "theme:026_camp_general"]},
            "facts_context": {"client_safe": True, "fresh": True, "facts_missing": False},
        },
    )

    assert result.route == "bot_answer_self_for_pilot"
    assert "Летние лагеря и ЛВШ УНПК — очные форматы" in result.draft_text
    assert "менеджер проверит актуальные варианты" in result.draft_text
    assert "есть онлайн-курсы" not in result.draft_text.casefold()


def test_unpk_lvsh_grade_11_detailed_price_answers_full_and_current_price() -> None:
    provider = FakeDraftProvider(
        {
            "route": "draft_for_manager",
            "draft_text": "Менеджер уточнит цену.",
            "message_type": "question",
            "topic_id": "theme:026_camp_general",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "11 класс, физика. 114 это скидочная цена или полная? Есть вариант дешевле?",
        context={
            "active_brand": "unpk",
            "known_dialog_fields": {"grade": "11", "subject": "физика"},
            "recent_messages": ["Клиент: интересует ЛВШ Менделеево с проживанием"],
            "autonomy_policy": {"allow_autonomous": True, "allowed_topic_ids": ["theme:026_camp_general"]},
            "facts_context": {"client_safe": True, "fresh": True, "facts_missing": False},
        },
    )

    assert "120 000" in result.draft_text
    assert "114 000" in result.draft_text
    assert "83 800" not in result.draft_text
    assert "11 класса" in result.draft_text
    assert "под ваш предмет" in result.draft_text


def test_unpk_lvsh_dates_do_not_offer_closed_august_shift() -> None:
    provider = FakeDraftProvider(
        {
            "route": "draft_for_manager",
            "draft_text": "УНПК: 18-26 июля и 15-25 августа.",
            "message_type": "question",
            "topic_id": "theme:026_camp_general",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "Какие смены ЛВШ Менделеево у УНПК?",
        context={
            "active_brand": "unpk",
            "autonomy_policy": {"allow_autonomous": True, "allowed_topic_ids": ["theme:026_camp_general"]},
            "facts_context": {"client_safe": True, "fresh": True, "facts_missing": False},
        },
    )

    assert "18-26 июля" in result.draft_text
    assert "августовская смена закрыта" in result.draft_text.casefold()
    assert "15-25 августа" not in result.draft_text
    assert "места почти распроданы" in result.draft_text.casefold()


def test_known_grade_camp_followup_does_not_use_dry_service_repair() -> None:
    provider = FakeDraftProvider(
        {
            "route": "bot_answer_self_for_pilot",
            "draft_text": "Напишите, пожалуйста, класс ребёнка, чтобы подобрать летний лагерь.",
            "message_type": "question",
            "topic_id": "theme:026_camp_general",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "А онлайн-смена есть?",
        context={
            "active_brand": "foton",
            "known_dialog_fields": {"grade": "8"},
            "autonomy_policy": {"allow_autonomous": True, "allowed_topic_ids": ["theme:026_camp_general"]},
            "facts_context": {"client_safe": True, "fresh": True, "facts_missing": False},
        },
    )

    assert result.route == "draft_for_manager"
    assert "asked_known_data_again" in result.safety_flags
    assert "8 класс" in result.draft_text
    assert "онлайн-формату" in result.draft_text
    assert "Сориентирую по безопасной части" not in result.draft_text
    assert "напишите" not in result.draft_text.casefold()


def test_offline_visit_invitation_without_confirmation_is_blocked() -> None:
    provider = FakeDraftProvider(
        {
            "route": "bot_answer_self_for_pilot",
            "draft_text": "Приезжайте к нам на площадку познакомиться и оформить запись.",
            "message_type": "question",
            "topic_id": "theme:020_enrollment",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "Можно записаться?",
        context={"active_brand": "foton", "rop_policy": {"bot_permission": "draft_for_manager"}},
    )

    assert result.route == "manager_only"
    assert "unsupported_offline_visit_invitation_detected" in result.safety_flags
    assert "Приезжайте" not in result.draft_text
    assert "дистанционно" in result.draft_text


def test_multi_topic_with_refund_demotes_autonomous_answer_to_manager_only() -> None:
    provider = FakeDraftProvider(
        {
            "route": "bot_answer_self_for_pilot",
            "draft_text": "Стоимость уточню, а по возврату передам менеджеру.",
            "message_type": "question",
            "topic_id": "theme:013_schedule",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "Сколько стоит курс и как вернуть деньги за прошлый месяц?",
        context={
            "active_brand": "foton",
            "autonomy_policy": {"allow_autonomous": True},
            "client_safe_fact_verified": True,
        },
    )

    assert result.route == "manager_only"
    assert "autonomy_blocked_high_risk" in result.safety_flags
    assert "combined_high_risk_manager_only" in result.safety_flags


def test_out_of_scope_fallback_is_brand_specific_foton() -> None:
    provider = FakeDraftProvider(
        {
            "route": "draft_for_manager",
            "draft_text": "Расскажу про айфоны.",
            "message_type": "question",
            "topic_id": "service:S3_out_of_scope",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "Чем 17 айфон лучше 13?",
        context={"active_brand": "foton", "rop_policy": {"bot_permission": "draft_for_manager"}},
    )

    assert result.route == "draft_for_manager"
    assert result.draft_text == OFF_TOPIC_FOTON_SAFE_TEXT
    assert "УНПК" not in result.draft_text


def test_out_of_scope_fallback_is_brand_specific_unpk() -> None:
    provider = FakeDraftProvider(
        {
            "route": "draft_for_manager",
            "draft_text": "Расскажу про айфоны.",
            "message_type": "question",
            "topic_id": "service:S3_out_of_scope",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "Чем 17 айфон лучше 13?",
        context={"active_brand": "unpk", "rop_policy": {"bot_permission": "draft_for_manager"}},
    )

    assert result.route == "draft_for_manager"
    assert result.draft_text == OFF_TOPIC_UNPK_SAFE_TEXT
    assert "Фотон" not in result.draft_text


def test_non_question_message_type_forces_manager_only() -> None:
    result = parse_llm_json(
        '{"route":"draft_for_manager","draft_text":"Спасибо!","message_type":"context_update",'
        '"topic_id":"service:S1_non_question","confidence_theme":0.9}'
    )

    assert result.route == "manager_only"
    assert "message_type_context_update" in result.safety_flags


def test_fake_provider_records_prompt() -> None:
    provider = FakeDraftProvider(DraftGenerationResult(route="draft_for_manager", draft_text="Здравствуйте!"))

    result = provider.generate("prompt")

    assert result.draft_text == "Здравствуйте!"
    assert provider.prompts == ["prompt"]


def test_funnel_known_slots_block_reasking_even_without_known_dialog_fields() -> None:
    provider = FakeDraftProvider(
        {
            "route": "bot_answer_self_for_pilot",
            "draft_text": "Напишите, пожалуйста, класс ребёнка и предмет, тогда подберём курс.",
            "message_type": "question",
            "topic_id": "theme:016_program",
            "confidence_theme": 0.92,
        }
    )

    result = provider.build_draft(
        "Что дальше?",
        context={
            "active_brand": "unpk",
            "autonomy_policy": {"allow_autonomous": True, "allowed_topic_ids": ["theme:016_program"]},
            "client_safe_fact_verified": True,
            "funnel_state": {
                "lead_stage": "next_step_offered",
                "filled_slots": {"grade": "9", "subject": "физика"},
            },
            "known_slots": {"grade": "9", "subject": "физика"},
        },
    )

    assert result.route == "draft_for_manager"
    assert "asked_known_data_again" in result.safety_flags
    assert "autonomy_blocked_asked_known_data_again" in result.safety_flags
    assert result.metadata["asked_known_data_again_fields"] == ["grade", "subject"]


def test_funnel_p0_route_recommendation_overrides_autonomous_llm() -> None:
    provider = FakeDraftProvider(
        {
            "route": "bot_answer_self_for_pilot",
            "draft_text": "Стоимость такая-то, а по возврату менеджер подскажет.",
            "message_type": "question",
            "topic_id": "theme:001_pricing",
            "confidence_theme": 0.91,
        }
    )

    result = provider.build_draft(
        "Сколько стоит и как вернуть деньги?",
        context={
            "active_brand": "foton",
            "autonomy_policy": {"allow_autonomous": True, "allowed_topic_ids": ["theme:001_pricing"]},
            "client_safe_fact_verified": True,
            "funnel_state": {"lead_stage": "p0_manager_only", "next_step_type": "manager_only_p0"},
        },
    )

    assert result.route == "manager_only"
    assert "autonomy_blocked_funnel_p0" in result.safety_flags


def test_provider_runs_answer_quality_rewriter_after_first_guards_before_autonomy() -> None:
    provider = FakeDraftProvider(
        {
            "route": "draft_for_manager",
            "draft_text": "Стоимость зависит от класса, формата и периода оплаты. Менеджер проверит актуальную стоимость.",
            "message_type": "question",
            "topic_id": "theme:001_pricing",
            "confidence_theme": 0.94,
        }
    )

    result = provider.build_draft(
        "Это цена прямо на сейчас? Можно зафиксировать?",
        context={
            "active_brand": "foton",
            "autonomy_policy": {"allow_autonomous": True, "allowed_topic_ids": ["theme:001_pricing"]},
            "client_safe_fact_verified": True,
            "facts_context": {"fresh": True, "client_safe": True},
            "confirmed_facts": {"fact:price": "Фотон 8 класс физика онлайн: текущая цена сейчас 74 500 ₽ за год."},
            "known_slots": {"grade": "8", "subject": "физика", "format": "онлайн"},
            "funnel_state": {"filled_slots": {"grade": "8", "subject": "физика", "format": "онлайн"}},
        },
    )

    assert result.metadata["answer_quality"]["rewritten"] is True
    assert "answer_quality_rewritten" in result.safety_flags
    assert "74 500 ₽" in result.draft_text
    assert "autonomy_matrix_passed" in result.safety_flags


def test_provider_rewriter_cannot_promote_manager_only_to_autonomous() -> None:
    provider = FakeDraftProvider(
        {
            "route": "manager_only",
            "draft_text": "Приняли обращение. Передам ответственному сотруднику, он вернется с ответом.",
            "message_type": "question",
            "topic_id": "theme:009_refund",
            "confidence_theme": 0.95,
            "safety_flags": ["high_risk_manager_only"],
        }
    )

    result = provider.build_draft(
        "Сколько стоит и как вернуть деньги?",
        context={
            "active_brand": "foton",
            "autonomy_policy": {"allow_autonomous": True, "allowed_topic_ids": ["theme:001_pricing"]},
            "client_safe_fact_verified": True,
            "confirmed_facts": {"fact:price": "Фотон: текущая цена 74 500 ₽."},
        },
    )

    assert result.route == "manager_only"
    assert "answer_quality_rewritten" not in result.safety_flags
    assert result.metadata["answer_quality"]["rewritten"] is False
