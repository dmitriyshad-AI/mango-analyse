from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from mango_mvp.channels.dialogue_contract_pipeline import (
    FactStore,
    Toggles,
    build_conversation,
    build_draft_prompt,
    check_claim_faithfulness,
    parse_contract,
    pipeline_enabled,
    run_pipeline,
    verify_output,
)
from mango_mvp.channels.subscription_llm import SubscriptionLlmDraftProvider


def _conv(text: str):
    return ({"role": "client", "text": text},)


def _understanding(payload: Mapping[str, Any]):
    return lambda _prompt: payload


def test_flag_default_off() -> None:
    assert pipeline_enabled({}) is False


def test_parse_contract_allows_real_fact_keys_with_digits() -> None:
    contract = parse_contract(
        {
            "current_question": "для каких классов олимпиадный онлайн",
            "needed_fact_keys": ["prices_regular_2026_27.online_olympiad_phystech_classes"],
            "answerability": "answer_self",
            "confidence": 0.9,
        },
        active_brand="unpk",
        fact_key_catalog=("prices_regular_2026_27.online_olympiad_phystech_classes",),
    )
    assert contract.needed_fact_keys == ("prices_regular_2026_27.online_olympiad_phystech_classes",)


def test_parse_contract_rejects_fact_values_in_needed_keys() -> None:
    contract = parse_contract(
        {
            "current_question": "цена",
            "needed_fact_keys": ["29 750 ₽", "prices.current"],
            "answerability": "answer_self",
        },
        active_brand="foton",
        fact_key_catalog=("prices.current",),
    )
    assert contract.needed_fact_keys == ("prices.current",)


def test_pipeline_happy_path_with_key_retrieval() -> None:
    store = FactStore(
        catalog=("price.online",),
        store={"foton": {"price.online": "Онлайн: семестр — 29 750 ₽, год — 47 250 ₽."}},
    )
    result = run_pipeline(
        conversation=_conv("сколько стоит онлайн?"),
        active_brand="foton",
        fact_store=store,
        understand_fn=_understanding(
            {"current_question": "цена онлайн", "needed_fact_keys": ["price.online"], "answerability": "answer_self"}
        ),
        draft_fn=lambda _prompt: "По онлайну: семестр — 29 750 ₽, год — 47 250 ₽. Подобрать группу?",
    )
    assert result.route == "bot_answer_self"
    assert not result.findings
    assert "29 750" in result.draft_text


def test_pipeline_p0_pregate_overrides_llm_contract() -> None:
    store = FactStore(catalog=("price.online",), store={"foton": {"price.online": "цена 29 750 ₽"}})
    result = run_pipeline(
        conversation=_conv("я оплатил, доступа нет, верните деньги"),
        active_brand="foton",
        fact_store=store,
        understand_fn=_understanding(
            {"current_question": "цена", "needed_fact_keys": ["price.online"], "answerability": "answer_self"}
        ),
        draft_fn=lambda _prompt: "цена 29 750 ₽",
    )
    assert result.route == "manager_only"
    assert result.contract.is_p0
    assert "Приняли обращение" in result.draft_text


def test_refund_followup_uses_refund_handoff_not_dry_repeat() -> None:
    store = FactStore(catalog=("price.online",), store={"foton": {"price.online": "цена 29 750 ₽"}})
    conversation = (
        {"role": "client", "text": "а если передумаю, вернуть деньги можно?"},
        {
            "role": "bot",
            "text": "Порядок возврата подтвердит менеджер по договору; передам именно про возврат.",
        },
        {"role": "client", "text": "а порядок возврата по заявлению оформляется?"},
    )
    result = run_pipeline(
        conversation=conversation,
        active_brand="foton",
        fact_store=store,
        understand_fn=_understanding(
            {
                "current_question": "порядок возврата по заявлению",
                "answerability": "manager_only",
                "confidence": 0.9,
            }
        ),
        draft_fn=lambda _prompt: "цена 29 750 ₽",
    )
    assert result.route == "draft_for_manager"
    assert result.fallback_reason == "refund_policy_manager_only"
    assert "возврат" in result.draft_text.casefold()
    assert "приняли обращение" not in result.draft_text.casefold()
    assert "обращение принято" not in result.draft_text.casefold()


def test_refund_p0_pregate_followup_uses_refund_handoff_not_dry_repeat() -> None:
    store = FactStore(catalog=("price.online",), store={"foton": {"price.online": "цена 29 750 ₽"}})
    conversation = (
        {"role": "client", "text": "а если передумаю, вернуть деньги можно?"},
        {
            "role": "bot",
            "text": "Порядок возврата подтвердит менеджер по договору; передам именно про возврат.",
        },
        {"role": "client", "text": "тогда верните деньги по возврату"},
    )
    result = run_pipeline(
        conversation=conversation,
        active_brand="foton",
        fact_store=store,
        understand_fn=_understanding(
            {
                "current_question": "верните деньги по возврату",
                "answerability": "manager_only",
                "confidence": 0.9,
            }
        ),
        draft_fn=lambda _prompt: "цена 29 750 ₽",
    )
    assert result.route == "manager_only"
    assert result.contract.is_p0
    assert result.fallback_reason == "p0_refund_policy"
    assert "возврат" in result.draft_text.casefold()
    assert "приняли обращение" not in result.draft_text.casefold()
    assert "обращение принято" not in result.draft_text.casefold()


def test_pipeline_verifier_blocks_brand_leak() -> None:
    store = FactStore(
        catalog=("price.online",),
        store={"foton": {"price.online": "Онлайн: семестр — 29 750 ₽, год — 47 250 ₽."}},
    )
    result = run_pipeline(
        conversation=_conv("сколько стоит онлайн?"),
        active_brand="foton",
        fact_store=store,
        understand_fn=_understanding(
            {"current_question": "цена онлайн", "needed_fact_keys": ["price.online"], "answerability": "answer_self"}
        ),
        draft_fn=lambda _prompt: "У Фотона 29 750 ₽, а в УНПК бывает иначе.",
    )
    assert result.route == "draft_for_manager"
    assert result.fallback_reason == "hard_verification_failed"
    assert any(finding.code == "brand_leak" for finding in result.findings)


def test_verify_output_blocks_named_entity_not_in_current_facts() -> None:
    findings = verify_output(
        "Запись будет в Zoom.",
        facts={"recordings": "Записи доступны в личном кабинете."},
        active_brand="foton",
        client_message="где будет запись?",
    )

    assert any(finding.code == "unsupported_entity" for finding in findings)


def test_verify_output_allows_named_entity_from_current_facts_and_neutral_words() -> None:
    findings = verify_output(
        "В Москве онлайн-вебинары проходят на МТС Линк, физика для 9 класса по выходным.",
        facts={"platform.webinars": "В Москве онлайн-вебинары проходят на платформе МТС Линк; физика для 9 класса по выходным."},
        active_brand="unpk",
        client_message="",
    )

    assert not [finding for finding in findings if finding.code == "unsupported_entity"]


def test_pipeline_unsupported_entity_falls_back_to_manager_draft() -> None:
    store = FactStore(catalog=("recordings",), store={"foton": {"recordings": "Записи доступны в личном кабинете."}})
    result = run_pipeline(
        conversation=_conv("где смотреть запись?"),
        active_brand="foton",
        fact_store=store,
        understand_fn=_understanding(
            {"current_question": "где запись", "needed_fact_keys": ["recordings"], "answerability": "answer_self"}
        ),
        draft_fn=lambda _prompt: "Запись будет доступна в Zoom.",
        faithfulness_fn=lambda _prompt: {"unsupported": []},
    )

    assert result.route == "draft_for_manager"
    assert result.fallback_reason == "hard_verification_failed"
    assert any(finding.code == "unsupported_entity" for finding in result.findings)


def test_pipeline_missing_fact_uses_narrow_handoff() -> None:
    store = FactStore(catalog=("schedule.exact_day",), store={"unpk": {}})
    result = run_pipeline(
        conversation=_conv("в какой именно день будет группа?"),
        active_brand="unpk",
        fact_store=store,
        understand_fn=_understanding(
            {"current_question": "точный день группы", "needed_fact_keys": ["schedule.exact_day"], "answerability": "answer_self"}
        ),
        draft_fn=lambda _prompt: "Точный день группы подтвердит менеджер по выбранной группе.",
    )
    assert result.missing == ("schedule.exact_day",)
    assert result.route == "bot_answer_self"


def test_t5_single_missing_slot_asks_one_question_without_manager_handoff() -> None:
    store = FactStore(catalog=("student.grade",), store={"foton": {}})
    result = run_pipeline(
        conversation=_conv("подберите курс"),
        active_brand="foton",
        fact_store=store,
        understand_fn=_understanding(
            {
                "current_question": "подобрать курс",
                "needed_fact_keys": ["student.grade"],
                "answerability": "manager_only",
            }
        ),
        draft_fn=lambda _prompt: "Передам менеджеру.",
    )
    assert result.route == "bot_answer_self"
    assert result.fallback_reason == "single_missing_slot_question"
    assert "класс" in result.draft_text.casefold()
    assert "менеджер" not in result.draft_text.casefold()


def test_manager_only_contract_is_not_promoted_by_retrieved_facts() -> None:
    store = FactStore(catalog=("price.online",), store={"foton": {"price.online": "Онлайн: семестр — 29 750 ₽."}})
    result = run_pipeline(
        conversation=_conv("сколько онлайн?"),
        active_brand="foton",
        fact_store=store,
        understand_fn=_understanding(
            {"current_question": "цена", "needed_fact_keys": ["price.online"], "answerability": "manager_only"}
        ),
        draft_fn=lambda _prompt: "Онлайн: семестр — 29 750 ₽.",
        faithfulness_fn=lambda _prompt: {"unsupported": []},
    )
    assert result.route == "draft_for_manager"
    assert result.fallback_reason == ""
    assert "29 750" in result.draft_text


def test_p9_self_subquestion_with_fact_is_autonomous_without_global_override() -> None:
    store = FactStore(
        catalog=("price.online", "schedule.exact_day"),
        store={"foton": {"price.online": "Онлайн: семестр — 29 750 ₽."}},
    )
    result = run_pipeline(
        conversation=_conv("сколько стоит онлайн и в какой день группа?"),
        active_brand="foton",
        fact_store=store,
        understand_fn=_understanding(
            {
                "current_question": "цена онлайн и точный день группы",
                "subquestions": [
                    {"text": "цена онлайн", "answerable": "self", "needed_fact_keys": ["price.online"]},
                    {"text": "точный день группы", "answerable": "manager", "needed_fact_keys": ["schedule.exact_day"]},
                ],
                "answerability": "manager_only",
            }
        ),
        draft_fn=lambda prompt: (
            "Онлайн: семестр — 29 750 ₽. Точный день группы подтвердит менеджер по выбранной группе."
            if "schedule.exact_day" in prompt
            else ""
        ),
        faithfulness_fn=lambda _prompt: {"unsupported": []},
    )

    assert result.route == "bot_answer_self"
    assert "29 750" in result.draft_text
    assert result.missing == ("schedule.exact_day",)


def test_manager_only_refund_does_not_substitute_course_rules_fact() -> None:
    store = FactStore(
        catalog=("course_rules_safe",),
        store={"foton": {"course_rules_safe": "На занятиях есть правила поведения и цифровой этикет."}},
    )
    result = run_pipeline(
        conversation=_conv("это не обращение, просто хочу заранее понимать правила возврата"),
        active_brand="foton",
        fact_store=store,
        understand_fn=_understanding(
            {
                "current_question": "Какие правила возврата действуют до начала занятий?",
                "needed_fact_keys": ["course_rules_safe"],
                "answerability": "manager_only",
                "confidence": 0.9,
            }
        ),
        draft_fn=lambda _prompt: "На занятиях есть правила поведения и цифровой этикет.",
        faithfulness_fn=lambda _prompt: {"unsupported": []},
    )

    assert result.route == "draft_for_manager"
    assert result.fallback_reason in {"contract_manager_only", "refund_policy_manager_only"}
    assert "возврат" in result.draft_text.casefold()
    assert "правила поведения" not in result.draft_text.casefold()
    assert "цифров" not in result.draft_text.casefold()


def test_c8_known_absence_allows_no_but_not_yes_for_unpk_bank_installment() -> None:
    absence = (
        "В УНПК отдельной банковской рассрочки нет. Оплату можно обсудить с менеджером; "
        "доступны помесячная оплата, оплата за семестр или за год."
    )
    store = FactStore(
        catalog=("payment_options.bank_installment.absent.client_safe_text",),
        store={"unpk": {"payment_options.bank_installment.absent.client_safe_text": absence}},
    )
    good = run_pipeline(
        conversation=_conv("есть банковская рассрочка?"),
        active_brand="unpk",
        fact_store=store,
        understand_fn=_understanding(
            {
                "current_question": "есть банковская рассрочка?",
                "question_type": "existence_yes_no",
                "existence_target": "банковская рассрочка",
                "needed_fact_keys": ["payment_options.bank_installment.absent.client_safe_text"],
                "answerability": "answer_self",
            }
        ),
        draft_fn=lambda _prompt: "Нет, отдельной банковской рассрочки нет. Можно обсудить помесячную оплату с менеджером.",
        faithfulness_fn=lambda _prompt: {"unsupported": []},
    )
    assert good.route == "bot_answer_self"
    assert good.fallback_reason == ""
    assert good.draft_text.startswith("Нет")

    bad = run_pipeline(
        conversation=_conv("есть банковская рассрочка?"),
        active_brand="unpk",
        fact_store=store,
        understand_fn=_understanding(
            {
                "current_question": "есть банковская рассрочка?",
                "question_type": "existence_yes_no",
                "existence_target": "банковская рассрочка",
                "needed_fact_keys": ["payment_options.bank_installment.absent.client_safe_text"],
                "answerability": "answer_self",
            }
        ),
        draft_fn=lambda _prompt: "Да, можно платить помесячно.",
        faithfulness_fn=lambda _prompt: {"unsupported": []},
    )
    assert bad.route == "draft_for_manager"
    assert bad.fallback_reason == "hard_verification_failed"
    assert "отдельной банковской рассрочки нет" in bad.draft_text.casefold()
    assert not bad.draft_text.casefold().startswith("да")


def test_c8_neighbor_payment_fact_does_not_allow_yes_or_closed_world_no() -> None:
    store = FactStore(
        catalog=("payment_options.current",),
        store={"unpk": {"payment_options.current": "У нас оплата возможна помесячно, за семестр или за год."}},
    )
    contract_payload = {
        "current_question": "есть банковская рассрочка?",
        "question_type": "existence_yes_no",
        "existence_target": "банковская рассрочка",
        "needed_fact_keys": ["payment_options.current"],
        "answerability": "answer_self",
    }
    yes_result = run_pipeline(
        conversation=_conv("есть банковская рассрочка?"),
        active_brand="unpk",
        fact_store=store,
        understand_fn=_understanding(contract_payload),
        draft_fn=lambda _prompt: "Да, можно платить помесячно.",
        faithfulness_fn=lambda _prompt: {"unsupported": []},
    )
    assert yes_result.route == "draft_for_manager"
    assert any(finding.code == "unsupported_existence_affirmation" for finding in yes_result.findings)

    no_result = run_pipeline(
        conversation=_conv("есть банковская рассрочка?"),
        active_brand="unpk",
        fact_store=store,
        understand_fn=_understanding(contract_payload),
        draft_fn=lambda _prompt: "Нет, банковской рассрочки нет.",
        faithfulness_fn=lambda _prompt: {"unsupported": []},
    )
    assert no_result.route == "draft_for_manager"
    assert any(finding.code == "unsupported_existence_negative" for finding in no_result.findings)


def test_c8_direct_invoice_question_is_not_answered_with_bank_installment_neighbor() -> None:
    store = FactStore(
        catalog=("installment.tbank",),
        store={
            "foton": {
                "installment.tbank": "Фотон: есть рассрочка через Т-Банк на 6, 10 или 12 месяцев."
            }
        },
    )
    result = run_pipeline(
        conversation=_conv("а помесячно прямым переводом на счёт можно?"),
        active_brand="foton",
        fact_store=store,
        understand_fn=_understanding(
            {
                "current_question": "а помесячно прямым переводом на счёт можно?",
                "needed_fact_keys": ["installment.tbank"],
                "answerability": "answer_self",
            }
        ),
        draft_fn=lambda _prompt: "Да, можно оформить рассрочку через Т-Банк на 6, 10 или 12 месяцев.",
        faithfulness_fn=lambda _prompt: {"unsupported": []},
    )
    assert result.route == "draft_for_manager"
    assert result.fallback_reason == "hard_verification_failed"
    assert any(finding.code == "neighbor_payment_method_as_answer" for finding in result.findings)
    assert any(finding.code == "unsupported_payment_method_affirmation" for finding in result.findings)
    assert "т-банк" not in result.draft_text.casefold()
    assert "рассроч" not in result.draft_text.casefold()
    assert "прямым переводом" in result.draft_text.casefold()


def test_c8_direct_invoice_fact_allows_direct_invoice_answer() -> None:
    store = FactStore(
        catalog=("payment.direct_invoice_monthly",),
        store={
            "foton": {
                "payment.direct_invoice_monthly": "Оплату можно вносить помесячно по счёту; прямой перевод на счёт подтверждает менеджер при оформлении."
            }
        },
    )
    result = run_pipeline(
        conversation=_conv("а помесячно прямым переводом на счёт можно?"),
        active_brand="foton",
        fact_store=store,
        understand_fn=_understanding(
            {
                "current_question": "а помесячно прямым переводом на счёт можно?",
                "needed_fact_keys": ["payment.direct_invoice_monthly"],
                "answerability": "answer_self",
            }
        ),
        draft_fn=lambda _prompt: "Да, помесячно по счёту можно; прямой перевод на счёт менеджер подтвердит при оформлении.",
        faithfulness_fn=lambda _prompt: {"unsupported": []},
    )
    assert result.route == "bot_answer_self"
    assert not result.findings


def test_soft_weekend_guidance_is_partial_not_exact_schedule() -> None:
    store = FactStore(
        catalog=("regular_schedule_publication", "objection_responses.inconvenient_time"),
        store={
            "unpk": {
                "regular_schedule_publication": "Расписание и подробная информация появятся в июне.",
                "objection_responses.inconvenient_time": "УНПК: черновик для ситуации «возражение о неудобном времени»: Разные слоты по выходным.",
            }
        },
    )
    result = run_pipeline(
        conversation=_conv("бывают занятия по субботам или воскресеньям?"),
        active_brand="unpk",
        fact_store=store,
        understand_fn=_understanding(
            {
                "current_question": "Бывают ли группы по выходным?",
                "needed_fact_keys": ["regular_schedule_publication"],
                "answerability": "manager_only",
                "confidence": 0.9,
            }
        ),
        draft_fn=lambda _prompt: "По субботам и воскресеньям точно есть занятия.",
        faithfulness_fn=lambda _prompt: {"unsupported": []},
    )

    assert result.route == "draft_for_manager"
    assert "разные варианты слотов" in result.draft_text.casefold()
    assert "точное расписание" in result.draft_text.casefold()
    assert "точно есть занятия" not in result.draft_text.casefold()
    assert "objection_responses.inconvenient_time" in result.facts


def test_build_conversation_keeps_recent_messages_and_last_client() -> None:
    conversation = build_conversation("а записи будут?", context={"recent_messages": ["9 класс", "Да, вижу 9 класс"]})
    assert conversation[-1] == {"role": "client", "text": "а записи будут?"}
    assert conversation[0]["text"] == "9 класс"


class _ContractProvider(SubscriptionLlmDraftProvider):
    def _dialogue_contract_understanding_runner(self, prompt: str) -> Mapping[str, Any]:
        assert "Каталог ключей фактов" in prompt
        return {
            "current_question": "цена онлайн",
            "needed_fact_keys": ["price.online"],
            "answerability": "answer_self",
            "confidence": 0.95,
        }

    def _dialogue_contract_draft_runner(self, prompt: str) -> str:
        assert "price.online" in prompt
        return "По онлайну: семестр — 29 750 ₽. Подобрать группу?"

    def _dialogue_contract_faithfulness_runner(self, prompt: str) -> Mapping[str, Any]:
        return {"unsupported": []}


def test_subscription_provider_parallel_path_is_opt_in() -> None:
    provider = _ContractProvider(max_attempts=1)
    result = provider.build_draft(
        "сколько онлайн?",
        context={
            "TELEGRAM_DIALOGUE_CONTRACT_PIPELINE": "1",
            "active_brand": "foton",
            "confirmed_facts": {"price.online": "Онлайн: семестр — 29 750 ₽."},
            "facts_context": {"client_safe_fact_verified": True, "required_fact_keys": ["price.online"]},
            "conversation_intent_plan": {"topic_id": "theme:001_pricing"},
        },
    )
    assert "dialogue_contract_pipeline" in result.safety_flags
    assert "manager_approval_required" in result.safety_flags
    assert "no_auto_send" in result.safety_flags
    assert "29 750" in result.draft_text


def test_contract_subquestions_slots_and_client_state() -> None:
    contract = parse_contract(
        {
            "current_question": "цена и записи",
            "client_state": "сравнивает цену",
            "subquestions": [
                {"text": "цена онлайн", "answerable": "self", "needed_fact_keys": ["price.online"]},
                {"text": "записи уроков", "answerable": "self", "needed_fact_keys": ["recording.access"]},
            ],
            "known_slots": {
                "class": {"value": "9", "source": "client_turn_1"},
                "subject": {"value": "физика"},
            },
            "answerability": "answer_self",
        },
        active_brand="foton",
        fact_key_catalog=("price.online", "recording.access"),
    )
    assert contract.all_needed_fact_keys() == ("price.online", "recording.access")
    assert contract.assertable_slots() == {"class": "9"}
    assert contract.unsourced_slots() == ("subject",)
    prompt = build_draft_prompt(conversation=_conv("цена?"), contract=contract, facts={}, missing=())
    assert "'class': '9'" in prompt
    assert "'subject':" not in prompt


def test_semantic_faithfulness_exception_fail_closed() -> None:
    store = FactStore(catalog=("price.online",), store={"foton": {"price.online": "Онлайн: семестр — 29 750 ₽."}})

    def boom(_prompt: str) -> Mapping[str, Any]:
        raise RuntimeError("faithfulness down")

    result = run_pipeline(
        conversation=_conv("сколько онлайн?"),
        active_brand="foton",
        fact_store=store,
        understand_fn=_understanding(
            {"current_question": "цена", "needed_fact_keys": ["price.online"], "answerability": "answer_self"}
        ),
        draft_fn=lambda _prompt: "Онлайн: семестр — 29 750 ₽.",
        faithfulness_fn=boom,
    )
    assert result.route == "draft_for_manager"
    assert result.fallback_reason == "semantic_check_unavailable"


def test_semantic_faithfulness_bad_json_fail_closed() -> None:
    store = FactStore(catalog=("price.online",), store={"foton": {"price.online": "Онлайн: семестр — 29 750 ₽."}})
    result = run_pipeline(
        conversation=_conv("сколько онлайн?"),
        active_brand="foton",
        fact_store=store,
        understand_fn=_understanding(
            {"current_question": "цена", "needed_fact_keys": ["price.online"], "answerability": "answer_self"}
        ),
        draft_fn=lambda _prompt: "Онлайн: семестр — 29 750 ₽.",
        faithfulness_fn=lambda _prompt: "not json",
    )
    assert result.route == "draft_for_manager"
    assert result.fallback_reason == "semantic_check_unavailable"


def test_structured_faithfulness_requires_fact_key_from_current_turn() -> None:
    result = check_claim_faithfulness(
        "Онлайн стоит 29 750 ₽.",
        facts={"price.online": "Онлайн стоит 29 750 ₽."},
        client_words="",
        faithfulness_fn=lambda _prompt: {
            "claims": [
                {
                    "claim": "Онлайн стоит 29 750 ₽",
                    "evidence_fact_key": "price.other",
                    "verdict": "supported",
                }
            ],
            "unsupported": [],
        },
    )

    assert result.available
    assert result.unsupported == ("Онлайн стоит 29 750 ₽",)
    assert result.claims[0].evidence_fact_key == "price.other"


def test_structured_faithfulness_checks_claim_anchors_against_one_fact() -> None:
    result = check_claim_faithfulness(
        "Записи уроков будут доступны в личном кабинете на МТС Линк.",
        facts={
            "recordings.cabinet": "Записи уроков доступны для пересмотра в личном кабинете.",
            "platform.webinars": "Онлайн-вебинары проходят на платформе МТС Линк.",
        },
        client_words="",
        faithfulness_fn=lambda _prompt: {
            "claims": [
                {
                    "claim": "Записи уроков доступны в личном кабинете на МТС Линк",
                    "evidence_fact_key": "recordings.cabinet",
                    "verdict": "supported",
                }
            ],
            "unsupported": [],
        },
    )

    assert result.available
    assert result.unsupported == ("Записи уроков доступны в личном кабинете на МТС Линк",)


def test_structured_faithfulness_fixture_covers_gluing_regressions() -> None:
    fixture = Path(__file__).parent / "fixtures" / "dialogue_contract_gluing_regressions.jsonl"
    rows = [json.loads(line) for line in fixture.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows
    for row in rows:
        result = check_claim_faithfulness(
            row["draft"],
            facts=row["facts"],
            client_words="",
            faithfulness_fn=lambda _prompt, payload=row["faithfulness"]: payload,
        )
        assert bool(result.unsupported) is bool(row["expected_unsupported"]), row["case_id"]


def test_composite_missing_fact_is_narrow_not_neighbor_substitution() -> None:
    store = FactStore(
        catalog=("price.online", "schedule.exact_day"),
        store={"foton": {"price.online": "Онлайн: семестр — 29 750 ₽."}},
    )
    result = run_pipeline(
        conversation=_conv("сколько онлайн и в какой день группа?"),
        active_brand="foton",
        fact_store=store,
        understand_fn=_understanding(
            {
                "current_question": "цена и точный день",
                "subquestions": [
                    {"text": "цена онлайн", "answerable": "self", "needed_fact_keys": ["price.online"]},
                    {"text": "точный день", "answerable": "self", "needed_fact_keys": ["schedule.exact_day"]},
                ],
                "answerability": "answer_self",
            }
        ),
        draft_fn=lambda prompt: (
            "Онлайн: семестр — 29 750 ₽. Точный день группы подтвердит менеджер по выбранной группе."
            if "schedule.exact_day" in prompt
            else ""
        ),
        faithfulness_fn=lambda _prompt: {"unsupported": []},
    )
    assert result.route == "bot_answer_self"
    assert result.missing == ("schedule.exact_day",)
    assert "29 750" in result.draft_text


def test_warmth_rewrite_reverified_and_rejected_on_new_price() -> None:
    store = FactStore(catalog=("price.online",), store={"foton": {"price.online": "Онлайн: семестр — 29 750 ₽."}})
    result = run_pipeline(
        conversation=_conv("сколько онлайн?"),
        active_brand="foton",
        fact_store=store,
        understand_fn=_understanding(
            {"current_question": "цена", "needed_fact_keys": ["price.online"], "answerability": "answer_self"}
        ),
        draft_fn=lambda _prompt: "По проверенным данным онлайн: семестр — 29 750 ₽.",
        faithfulness_fn=lambda _prompt: {"unsupported": []},
        warmth_fn=lambda _prompt: "Отлично, сейчас онлайн всего 19 900 ₽!",
    )
    assert not result.warmed
    assert "29 750" in result.draft_text
    assert "19 900" not in result.draft_text
    assert result.warmth_attempted
    assert result.warmth_rejected_reason == "new_concrete_anchor"


def test_warmth_rewrite_rejected_on_new_platform_anchor() -> None:
    store = FactStore(catalog=("recordings",), store={"foton": {"recordings": "Записи доступны в личном кабинете."}})
    result = run_pipeline(
        conversation=_conv("где смотреть запись?"),
        active_brand="foton",
        fact_store=store,
        understand_fn=_understanding(
            {"current_question": "где запись", "needed_fact_keys": ["recordings"], "answerability": "answer_self"}
        ),
        draft_fn=lambda _prompt: "Записи доступны в личном кабинете.",
        faithfulness_fn=lambda prompt: {"unsupported": ["запись в Zoom"]} if "Zoom" in prompt else {"unsupported": []},
        warmth_fn=lambda _prompt: "Да, запись будет доступна в Zoom.",
        toggles=Toggles(warmth_mode="all_eligible"),
    )

    assert result.warmth_attempted
    assert not result.warmed
    assert result.warmth_rejected_reason == "new_concrete_anchor"
    assert "Zoom" not in result.draft_text


def test_warmth_rewrite_accepts_form_only_paraphrase_even_if_semantic_is_overstrict() -> None:
    store = FactStore(catalog=("price.online",), store={"foton": {"price.online": "Онлайн стоит 29 750 ₽."}})
    calls = 0

    def _faithfulness(_prompt: str):
        nonlocal calls
        calls += 1
        if calls == 2:
            return {"unsupported": ["живой зачин не является фактом"]}
        return {"unsupported": []}

    result = run_pipeline(
        conversation=_conv("сколько онлайн?"),
        active_brand="foton",
        fact_store=store,
        understand_fn=_understanding(
            {"current_question": "цена", "needed_fact_keys": ["price.online"], "answerability": "answer_self"}
        ),
        draft_fn=lambda _prompt: "Онлайн стоит 29 750 ₽.",
        faithfulness_fn=_faithfulness,
        warmth_fn=lambda _prompt: "Да, онлайн стоит 29 750 ₽. Подобрать группу?",
        toggles=Toggles(warmth_mode="all_eligible"),
    )

    assert result.warmth_attempted
    assert result.warmed
    assert result.draft_text.startswith("Да,")


def test_warmth_all_eligible_rewrites_without_form_findings() -> None:
    store = FactStore(catalog=("price.online",), store={"foton": {"price.online": "Онлайн: семестр — 29 750 ₽."}})
    result = run_pipeline(
        conversation=_conv("сколько онлайн?"),
        active_brand="foton",
        fact_store=store,
        understand_fn=_understanding(
            {"current_question": "цена", "needed_fact_keys": ["price.online"], "answerability": "answer_self"}
        ),
        draft_fn=lambda _prompt: "Онлайн стоит 29 750 ₽. Могу подсказать группы?",
        faithfulness_fn=lambda _prompt: {"unsupported": []},
        warmth_fn=lambda _prompt: "Да, онлайн стоит 29 750 ₽. Подберём группу под класс?",
        toggles=Toggles(warmth_mode="all_eligible"),
    )
    assert not result.form_findings
    assert result.warmth_attempted
    assert result.warmed
    assert result.warmth_mode == "all_eligible"
    assert result.draft_text.startswith("Да, онлайн")


def test_warmth_linter_mode_does_not_call_rewriter_without_form_findings() -> None:
    store = FactStore(catalog=("price.online",), store={"foton": {"price.online": "Онлайн: семестр — 29 750 ₽."}})
    calls: list[str] = []

    def _warmth(prompt: str) -> str:
        calls.append(prompt)
        return "Да, онлайн стоит 29 750 ₽. Подберём группу под класс?"

    result = run_pipeline(
        conversation=_conv("сколько онлайн?"),
        active_brand="foton",
        fact_store=store,
        understand_fn=_understanding(
            {"current_question": "цена", "needed_fact_keys": ["price.online"], "answerability": "answer_self"}
        ),
        draft_fn=lambda _prompt: "Онлайн стоит 29 750 ₽. Могу подсказать группы?",
        faithfulness_fn=lambda _prompt: {"unsupported": []},
        warmth_fn=_warmth,
        toggles=Toggles(warmth_mode="linter"),
    )
    assert not result.form_findings
    assert not result.warmth_attempted
    assert not result.warmed
    assert calls == []


def test_warmth_all_eligible_skips_draft_for_manager() -> None:
    store = FactStore(catalog=("refund.policy",), store={"foton": {"refund.policy": "Возврат подтверждает менеджер."}})
    calls: list[str] = []
    result = run_pipeline(
        conversation=_conv("а как оформить возврат?"),
        active_brand="foton",
        fact_store=store,
        understand_fn=_understanding(
            {"current_question": "возврат", "needed_fact_keys": ["refund.policy"], "answerability": "manager_only"}
        ),
        draft_fn=lambda _prompt: "Передам менеджеру вопрос про возврат.",
        faithfulness_fn=lambda _prompt: {"unsupported": []},
        warmth_fn=lambda prompt: calls.append(prompt) or "Тёплая версия",
        toggles=Toggles(warmth_mode="all_eligible"),
    )
    assert result.route == "draft_for_manager"
    assert not result.warmth_attempted
    assert calls == []


def test_p0_dry_text_rotates_by_previous_bot_turns() -> None:
    store = FactStore(catalog=(), store={"foton": {}})
    first = run_pipeline(
        conversation=({"role": "client", "text": "верните деньги, я оплатил"},),
        active_brand="foton",
        fact_store=store,
        understand_fn=_understanding({}),
        draft_fn=lambda _prompt: "",
    )
    second = run_pipeline(
        conversation=(
            {"role": "client", "text": "верните деньги, я оплатил"},
            {"role": "bot", "text": first.draft_text},
            {"role": "client", "text": "и что дальше по возврату?"},
        ),
        active_brand="foton",
        fact_store=store,
        understand_fn=_understanding({}),
        draft_fn=lambda _prompt: "",
    )
    assert first.route == second.route == "manager_only"
    assert first.draft_text != second.draft_text
    assert "ответственному сотруднику" in first.draft_text
    assert "ответственному сотруднику" in second.draft_text


def test_p0_pregate_does_not_call_warmth() -> None:
    store = FactStore(catalog=("price.online",), store={"foton": {"price.online": "цена 29 750 ₽"}})

    def _warmth(_prompt: str) -> str:
        raise AssertionError("P0 must not call warmth rewriter")

    result = run_pipeline(
        conversation=_conv("я оплатил, доступа нет, верните деньги"),
        active_brand="foton",
        fact_store=store,
        understand_fn=_understanding(
            {"current_question": "цена", "needed_fact_keys": ["price.online"], "answerability": "answer_self"}
        ),
        draft_fn=lambda _prompt: "цена 29 750 ₽",
        warmth_fn=_warmth,
        toggles=Toggles(warmth_mode="all_eligible"),
    )
    assert result.route == "manager_only"
    assert result.fallback_reason == "p0"
    assert not result.warmth_attempted
    assert not result.warmed


def test_refund_policy_question_answers_from_refund_fact_not_contract_fact() -> None:
    refund_fact = (
        "Если клиент заранее спрашивает про возврат до оплаты, можно спокойно ответить, "
        "что при досрочном отказе возвращается остаток неистраченных средств; можно не переживать."
    )
    store = FactStore(
        catalog=("contract.email", "presentation.client_safe_facts.refund_presale_policy.client_safe_text"),
        store={
            "foton": {
                "contract.email": "Договор пришлёт менеджер на email.",
                "presentation.client_safe_facts.refund_presale_policy.client_safe_text": refund_fact,
            }
        },
    )
    result = run_pipeline(
        conversation=_conv("а у вас вообще возвраты бывают, если что? просто заранее интересно"),
        active_brand="foton",
        fact_store=store,
        understand_fn=_understanding(
            {
                "current_question": "а у вас вообще возвраты бывают, если что? просто заранее интересно",
                "needed_fact_keys": ["contract.email"],
                "answerability": "manager_only",
            }
        ),
        draft_fn=lambda _prompt: (
            "Да, можно не переживать: при досрочном отказе возвращается остаток неистраченных средств. "
            "Порядок оформления менеджер подтвердит по договору."
        ),
        faithfulness_fn=lambda _prompt: {"unsupported": []},
        toggles=Toggles(warmth_mode="all_eligible"),
    )
    assert result.route == "bot_answer_self"
    assert result.fallback_reason == ""
    assert "остаток неистраченных средств" in result.draft_text.casefold()
    assert "email" not in result.draft_text.casefold()


def test_refund_policy_without_fact_stays_narrow_handoff() -> None:
    store = FactStore(
        catalog=("contract.email",),
        store={"foton": {"contract.email": "Договор пришлёт менеджер на email."}},
    )
    result = run_pipeline(
        conversation=_conv("а у вас вообще возвраты бывают, если что? просто заранее интересно"),
        active_brand="foton",
        fact_store=store,
        understand_fn=_understanding(
            {
                "current_question": "а у вас вообще возвраты бывают, если что? просто заранее интересно",
                "needed_fact_keys": ["contract.email"],
                "answerability": "answer_self",
            }
        ),
        draft_fn=lambda _prompt: "Договор пришлёт менеджер на email. Подскажите email.",
        faithfulness_fn=lambda _prompt: {"unsupported": []},
        toggles=Toggles(warmth_mode="all_eligible"),
    )
    assert result.route == "draft_for_manager"
    assert result.fallback_reason == "refund_policy_manager_only"
    assert "возврат" in result.draft_text.casefold()
    assert "email" not in result.draft_text.casefold()


def test_tax_return_question_is_not_treated_as_refund_policy() -> None:
    store = FactStore(catalog=("tax.max_return",), store={"foton": {"tax.max_return": "Можно вернуть до 14 300 ₽."}})
    result = run_pipeline(
        conversation=_conv("сколько можно вернуть по налоговому вычету?"),
        active_brand="foton",
        fact_store=store,
        understand_fn=_understanding(
            {
                "current_question": "сколько можно вернуть по налоговому вычету?",
                "needed_fact_keys": ["tax.max_return"],
                "answerability": "answer_self",
            }
        ),
        draft_fn=lambda _prompt: "По налоговому вычету можно вернуть до 14 300 ₽.",
        faithfulness_fn=lambda _prompt: {"unsupported": []},
    )
    assert result.route == "bot_answer_self"
    assert result.fallback_reason == ""
    assert "14 300" in result.draft_text


def test_bare_mfti_is_brand_leak_for_foton() -> None:
    store = FactStore(catalog=("price.online",), store={"foton": {"price.online": "Онлайн: семестр — 29 750 ₽."}})
    result = run_pipeline(
        conversation=_conv("сколько онлайн?"),
        active_brand="foton",
        fact_store=store,
        understand_fn=_understanding(
            {"current_question": "цена", "needed_fact_keys": ["price.online"], "answerability": "answer_self"}
        ),
        draft_fn=lambda _prompt: "В МФТИ формат другой, а онлайн — 29 750 ₽.",
        faithfulness_fn=lambda _prompt: {"unsupported": []},
    )
    assert result.route == "draft_for_manager"
    assert any(finding.code == "brand_leak" for finding in result.findings)
