from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from mango_mvp.channels.dialogue_contract_pipeline import (
    FactStore,
    Toggles,
    build_conversation,
    build_draft_prompt,
    build_understanding_prompt,
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


def _refund_fact() -> str:
    return (
        "Если клиент заранее спрашивает про возврат до оплаты, можно спокойно ответить, "
        "что при досрочном отказе возвращается остаток неистраченных средств; можно не переживать. "
        "Конкретный порядок оформления менеджер подтвердит по выбранному курсу и договору."
    )


def _refund_store() -> FactStore:
    return FactStore(
        catalog=("presentation.client_safe_facts.refund_presale_policy.client_safe_text",),
        store={
            "foton": {
                "presentation.client_safe_facts.refund_presale_policy.client_safe_text": _refund_fact(),
            }
        },
    )


def _trace_rows(path: Path) -> list[Mapping[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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


def test_dialogue_contract_debug_trace_off_does_not_write_or_change_result(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("DIALOGUE_CONTRACT_DEBUG_TRACE", raising=False)
    store = FactStore(
        catalog=("price.online",),
        store={"foton": {"price.online": "Онлайн: семестр — 29 750 ₽, год — 47 250 ₽."}},
    )
    context = {
        "dialogue_contract_debug_trace": {
            "enabled": False,
            "run_dir": str(tmp_path),
            "dialog_id": "trace_off",
            "turn": 1,
        }
    }

    result = run_pipeline(
        conversation=_conv("сколько стоит онлайн?"),
        active_brand="foton",
        fact_store=store,
        understand_fn=_understanding(
            {"current_question": "цена онлайн", "needed_fact_keys": ["price.online"], "answerability": "answer_self"}
        ),
        draft_fn=lambda _prompt: "По онлайну: семестр — 29 750 ₽, год — 47 250 ₽. Подобрать группу?",
        context=context,
    )

    assert result.route == "bot_answer_self"
    assert "29 750" in result.draft_text
    assert not (tmp_path / "debug_trace.jsonl").exists()


def test_dialogue_contract_debug_trace_on_writes_expected_nodes(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("DIALOGUE_CONTRACT_DEBUG_TRACE", raising=False)
    long_question = "сколько стоит онлайн? " + ("очень длинное уточнение " * 20)
    store = FactStore(
        catalog=("price.online",),
        store={"foton": {"price.online": "Онлайн: семестр — 29 750 ₽, год — 47 250 ₽."}},
    )
    context = {
        "dialogue_contract_debug_trace": {
            "enabled": True,
            "run_dir": str(tmp_path),
            "dialog_id": "trace_on",
            "turn": 2,
        }
    }

    result = run_pipeline(
        conversation=_conv(long_question),
        active_brand="foton",
        fact_store=store,
        understand_fn=_understanding(
            {"current_question": "цена онлайн", "needed_fact_keys": ["price.online"], "answerability": "answer_self"}
        ),
        draft_fn=lambda _prompt: "По онлайну: семестр — 29 750 ₽, год — 47 250 ₽. Подобрать группу?",
        context=context,
    )

    assert result.route == "bot_answer_self"
    rows = _trace_rows(tmp_path / "debug_trace.jsonl")
    nodes = {row["node"] for row in rows}
    assert {"p0_pre_gate", "understand", "retrieve_facts", "build_draft", "_hard_check"} <= nodes
    assert all(row["dialog_id"] == "trace_on" for row in rows)
    assert all(row["turn"] == 2 for row in rows)
    understand_row = next(row for row in rows if row["node"] == "understand")
    assert len(understand_row["values"]["client_message"]) <= 200


def test_dialogue_contract_debug_trace_synthetic_paths_cover_fallback_and_p0(tmp_path) -> None:
    store = FactStore(catalog=("price.online",), store={"foton": {"price.online": "цена 29 750 ₽"}})
    trace_file = tmp_path / "debug_trace.jsonl"
    common_trace = {"enabled": True, "run_dir": str(tmp_path), "dialog_id": "synthetic", "turn": 1}

    p0_result = run_pipeline(
        conversation=_conv("я оплатил, доступа нет, верните деньги"),
        active_brand="foton",
        fact_store=store,
        understand_fn=_understanding(
            {"current_question": "цена", "needed_fact_keys": ["price.online"], "answerability": "answer_self"}
        ),
        draft_fn=lambda _prompt: "цена 29 750 ₽",
        context={"dialogue_contract_debug_trace": {**common_trace, "dialog_id": "synthetic_p0"}},
    )
    fallback_result = run_pipeline(
        conversation=_conv("какая цена?"),
        active_brand="foton",
        fact_store=store,
        understand_fn=_understanding(
            {"current_question": "цена", "needed_fact_keys": ["price.online"], "answerability": "manager_only"}
        ),
        draft_fn=None,
        context={"dialogue_contract_debug_trace": {**common_trace, "dialog_id": "synthetic_fallback", "turn": 2}},
    )

    assert p0_result.route == "manager_only"
    assert fallback_result.route == "draft_for_manager"
    rows = _trace_rows(trace_file)
    nodes = {row["node"] for row in rows}
    assert {"understand", "retrieve_facts", "build_draft", "_safe_fallback_text", "p0_pre_gate"} <= nodes


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
    assert result.fallback_reason == "p0"
    assert "ответственному сотруднику" in result.draft_text.casefold()


def test_presale_refund_multiturn_followups_keep_answering_from_fact() -> None:
    store = _refund_store()
    prior_bot_text = (
        "Да, если передумаете до начала занятий, возвращается остаток неистраченных средств. "
        "Точный порядок оформления менеджер подтвердит по выбранному курсу и договору."
    )
    followups = (
        "а это в договоре прописано?",
        "то есть можно правило понять до оплаты?",
        "я не спорю, просто хочу заранее понимать без звонков",
    )
    conversation: tuple[dict[str, str], ...] = (
        {"role": "client", "text": "а если передумаю до начала занятий, деньги вернут?"},
        {"role": "bot", "text": prior_bot_text},
    )
    for followup in followups:
        current_conversation = (*conversation, {"role": "client", "text": followup})
        result = run_pipeline(
            conversation=current_conversation,
            active_brand="foton",
            fact_store=store,
            understand_fn=_understanding(
                {
                    "current_question": followup,
                    "continued_topics": ["refund_policy"],
                    "needed_fact_keys": ["refund_policy.current"],
                    "answerability": "manager_only",
                    "is_p0": True,
                    "p0_reason": "llm_false_positive_refund",
                }
            ),
            draft_fn=lambda _prompt: (
                "Можно спокойно ориентироваться на это правило: при досрочном отказе возвращается "
                "остаток неистраченных средств. Точные пункты договора менеджер подтвердит по выбранному курсу."
            ),
            faithfulness_fn=lambda _prompt: {"unsupported": []},
            toggles=Toggles(warmth_mode="all_eligible"),
        )

        assert result.route == "bot_answer_self"
        assert not result.manager_only
        assert result.fallback_reason == ""
        assert "остаток неистраченных средств" in result.draft_text.casefold()
        assert "если клиент заранее спрашивает" not in result.draft_text.casefold()
        assert "можно спокойно ответить" not in result.draft_text.casefold()
        assert "передам менеджеру" not in result.draft_text.casefold()
        assert "не буду подменять" not in result.draft_text.casefold()
        conversation = (*current_conversation, {"role": "bot", "text": result.draft_text})


def test_presale_refund_thread_escalates_on_real_refund_claim_turn() -> None:
    conversation = (
        {"role": "client", "text": "если передумаю до начала занятий, деньги вернут?"},
        {
            "role": "bot",
            "text": "Да, возвращается остаток неистраченных средств. Точный порядок менеджер подтвердит.",
        },
        {"role": "client", "text": "занятий нет, верните деньги, я недоволен"},
    )
    result = run_pipeline(
        conversation=conversation,
        active_brand="foton",
        fact_store=_refund_store(),
        understand_fn=_understanding(
            {
                "current_question": "занятий нет, верните деньги, я недоволен",
                "continued_topics": ["refund_policy"],
                "needed_fact_keys": ["refund_policy.current"],
                "answerability": "answer_self",
                "is_p0": False,
            }
        ),
        draft_fn=lambda _prompt: "Возвращается остаток неистраченных средств.",
        faithfulness_fn=lambda _prompt: {"unsupported": []},
        toggles=Toggles(warmth_mode="all_eligible"),
    )

    assert result.route == "manager_only"
    assert result.manager_only
    assert result.contract.is_p0
    assert result.fallback_reason in {"p0", "p0_refund_policy"}
    assert "остаток неистраченных средств" not in result.draft_text.casefold()


def test_refund_waiting_month_followup_stays_p0_manager_only() -> None:
    conversation = (
        {"role": "client", "text": "я оплатил месяц назад, а занятий нет. верните деньги"},
        {"role": "bot", "text": "Обращение принято. Передам ответственному сотруднику."},
        {"role": "client", "text": "ну так вернёте или нет? я уже месяц жду"},
    )
    result = run_pipeline(
        conversation=conversation,
        active_brand="foton",
        fact_store=_refund_store(),
        understand_fn=_understanding(
            {
                "current_question": "ну так вернёте или нет? я уже месяц жду",
                "continued_topics": ["refund_policy"],
                "needed_fact_keys": ["refund_policy.current"],
                "answerability": "answer_self",
                "is_p0": True,
            }
        ),
        draft_fn=lambda _prompt: "Возвращается остаток неистраченных средств.",
        faithfulness_fn=lambda _prompt: {"unsupported": []},
    )

    assert result.route == "manager_only"
    assert result.fallback_reason == "p0"
    assert "остаток неистраченных средств" not in result.draft_text.casefold()


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


def test_pricing_hard_check_fallback_answers_from_verified_price_facts() -> None:
    store = FactStore(
        catalog=(
            "prices_regular_2026_27.online_5_11_class.before_2026_08_01.semester",
            "prices_regular_2026_27.online_5_11_class.before_2026_08_01.year",
        ),
        store={
            "foton": {
                "prices_regular_2026_27.online_5_11_class.before_2026_08_01.semester": (
                    "Фотон: цены на 2026/27 учебный год, 5-11 класс, онлайн, семестр — 29 750 ₽."
                ),
                "prices_regular_2026_27.online_5_11_class.before_2026_08_01.year": (
                    "Фотон: цены на 2026/27 учебный год, 5-11 класс, онлайн, год — 47 250 ₽."
                ),
            }
        },
    )

    def faithfulness(prompt: str) -> dict[str, list[str]]:
        if "до 1 августа" in prompt:
            return {"unsupported": ["до 1 августа 2026"]}
        return {"unsupported": []}

    result = run_pipeline(
        conversation=_conv("подскажите цену на онлайн математику для 7 класса?"),
        active_brand="foton",
        fact_store=store,
        understand_fn=_understanding(
            {
                "current_question": "цена на онлайн математику для 7 класса",
                "needed_fact_keys": [
                    "prices_regular_2026_27.online_5_11_class.before_2026_08_01.semester",
                    "prices_regular_2026_27.online_5_11_class.before_2026_08_01.year",
                ],
                "answerability": "answer_self",
            }
        ),
        draft_fn=lambda _prompt: "До 1 августа семестр стоит 29 750 ₽, год — 47 250 ₽.",
        faithfulness_fn=faithfulness,
    )

    assert result.route == "bot_answer_self"
    assert result.fallback_reason == "verified_fact_fallback_after_hard_check"
    assert "29 750 ₽" in result.draft_text
    assert "47 250 ₽" in result.draft_text
    assert "до 1 августа" not in result.draft_text.casefold()


def test_recording_hard_check_fallback_answers_from_recording_facts() -> None:
    store = FactStore(
        catalog=("online_platform.recording_client_safe_text", "online_platform.name"),
        store={
            "foton": {
                "online_platform.recording_client_safe_text": "Онлайн-записи занятий сохраняются и доступны для пересмотра.",
                "online_platform.name": "Онлайн-платформа Фотон: МТС Линк (бывший Webinar).",
            }
        },
    )

    result = run_pipeline(
        conversation=_conv("значит потом в мтс линке можно будет пересмотреть?"),
        active_brand="foton",
        fact_store=store,
        understand_fn=_understanding(
            {
                "current_question": "значит потом в МТС Линк можно будет пересмотреть запись урока",
                "needed_fact_keys": ["online_platform.recording_client_safe_text", "online_platform.name"],
                "answerability": "answer_self",
            }
        ),
        draft_fn=lambda _prompt: "Подтверждаю: это подтверждено целиком.",
        faithfulness_fn=lambda prompt: {"unsupported": ["слишком общий ответ"]} if "Подтверждаю" in prompt else {"unsupported": []},
    )

    assert result.route == "bot_answer_self"
    assert result.fallback_reason == "verified_fact_fallback_after_hard_check"
    assert "доступны для пересмотра" in result.draft_text.casefold()
    assert "мтс линк" in result.draft_text.casefold()


def test_format_hard_check_fallback_answers_from_online_format_fact() -> None:
    store = FactStore(
        catalog=("tg_unpk_verified_2026_05_21.client_facts.online_courses_format.client_safe_text",),
        store={
            "unpk": {
                "tg_unpk_verified_2026_05_21.client_facts.online_courses_format.client_safe_text": (
                    "Онлайн-курсы УНПК проходят на платформе МТС-Link. После каждого урока доступны записи."
                )
            }
        },
    )

    result = run_pipeline(
        conversation=_conv("а онлайн по математике для 9 класса тоже есть или только очно?"),
        active_brand="unpk",
        fact_store=store,
        understand_fn=_understanding(
            {
                "current_question": "Есть ли онлайн-курс по математике для 9 класса в УНПК или обучение только очное?",
                "needed_fact_keys": ["tg_unpk_verified_2026_05_21.client_facts.online_courses_format.client_safe_text"],
                "answerability": "answer_self",
            }
        ),
        draft_fn=lambda _prompt: "Не знаю, уточнит менеджер.",
        faithfulness_fn=lambda prompt: {"unsupported": ["нет ответа из факта"]} if "Не знаю" in prompt else {"unsupported": []},
    )

    assert result.route == "bot_answer_self"
    assert result.fallback_reason == "verified_fact_fallback_after_hard_check"
    assert "онлайн-формату подтверждено" in result.draft_text.casefold()
    assert "конкретную группу по предмету и классу" in result.draft_text.casefold()


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


def test_handoff_paraphrase_without_claim_does_not_fail_hard_check() -> None:
    store = FactStore(catalog=(), store={"unpk": {}})
    handoffs = (
        "Передам менеджеру уточнить именно этот вопрос.",
        "Сейчас точно ответить не могу. Передам вопрос менеджеру — он уточнит и свяжется с вами.",
        "Менеджер сверит эту деталь и вернётся с ответом.",
        "По этому пункту нужна проверка менеджера.",
        "Передам менеджеру именно этот пункт, чтобы он подтвердил точную информацию.",
    )

    for handoff in handoffs:
        result = run_pipeline(
            conversation=_conv("сколько стоит онлайн для 9 класса?"),
            active_brand="unpk",
            fact_store=store,
            understand_fn=_understanding(
                {
                    "current_question": "цена онлайн для 9 класса",
                    "needed_fact_keys": [],
                    "answerability": "answer_self",
                }
            ),
            draft_fn=lambda _prompt, handoff=handoff: handoff,
            faithfulness_fn=lambda _prompt: {"unsupported": ["нет факта"]},
        )

        assert result.draft_text == handoff
        assert result.fallback_reason == ""
        assert not result.findings
        assert not result.unsupported_claims


def test_handoff_with_unsupported_price_claim_fails_hard_check() -> None:
    store = FactStore(catalog=(), store={"unpk": {}})
    result = run_pipeline(
        conversation=_conv("сколько стоит онлайн для 9 класса?"),
        active_brand="unpk",
        fact_store=store,
        understand_fn=_understanding(
            {
                "current_question": "цена онлайн для 9 класса",
                "needed_fact_keys": [],
                "answerability": "answer_self",
            }
        ),
        draft_fn=lambda _prompt: "Передам менеджеру; цена 99 999 ₽ за семестр.",
        faithfulness_fn=lambda _prompt: {"unsupported": []},
    )

    assert result.route == "draft_for_manager"
    assert result.fallback_reason == "hard_verification_failed"
    assert any(finding.code == "fact_grounding" for finding in result.findings)
    assert "99 999" not in result.draft_text


def test_handoff_with_unsupported_date_claim_fails_hard_check() -> None:
    store = FactStore(catalog=(), store={"unpk": {}})
    result = run_pipeline(
        conversation=_conv("когда смена?"),
        active_brand="unpk",
        fact_store=store,
        understand_fn=_understanding(
            {
                "current_question": "дата смены",
                "needed_fact_keys": [],
                "answerability": "answer_self",
            }
        ),
        draft_fn=lambda _prompt: "Передам менеджеру; смена начнётся 1 февраля 2030 года.",
        faithfulness_fn=lambda _prompt: {"unsupported": []},
    )

    assert result.route == "draft_for_manager"
    assert result.fallback_reason == "hard_verification_failed"
    assert any(finding.code == "fact_grounding" for finding in result.findings)
    assert "2030" not in result.draft_text


def test_handoff_with_supported_price_claim_passes_hard_check() -> None:
    store = FactStore(catalog=("price.online",), store={"unpk": {"price.online": "Онлайн: семестр — 69 900 ₽."}})
    result = run_pipeline(
        conversation=_conv("сколько стоит онлайн для 9 класса?"),
        active_brand="unpk",
        fact_store=store,
        understand_fn=_understanding(
            {
                "current_question": "цена онлайн для 9 класса",
                "needed_fact_keys": ["price.online"],
                "answerability": "answer_self",
            }
        ),
        draft_fn=lambda _prompt: "Передам менеджеру; семестр — 69 900 ₽.",
        faithfulness_fn=lambda _prompt: {"unsupported": []},
    )

    assert result.route == "bot_answer_self"
    assert result.fallback_reason == ""
    assert "69 900" in result.draft_text


def test_phase1_coverage_repairs_self_answer_that_omits_retrieved_date() -> None:
    store = FactStore(
        catalog=("course.start.dates",),
        store={"foton": {"course.start.dates": "Старт ближайшего курса Фотон — 3-14 августа."}},
    )
    repair_prompts: list[str] = []
    result = run_pipeline(
        conversation=_conv("когда стартует ближайший курс?"),
        active_brand="foton",
        fact_store=store,
        understand_fn=_understanding(
            {
                "current_question": "даты ближайшего курса",
                "subquestions": [
                    {"text": "даты ближайшего курса", "answerable": "self", "needed_fact_keys": ["course.start.dates"]}
                ],
                "answerability": "answer_self",
            }
        ),
        draft_fn=lambda _prompt: "Ближайший курс подберём под возраст ребёнка.",
        repair_fn=lambda prompt: repair_prompts.append(prompt) or "Старт ближайшего курса Фотон — 3-14 августа.",
        faithfulness_fn=lambda _prompt: {"unsupported": []},
    )

    assert result.route == "bot_answer_self"
    assert result.repaired is True
    assert "3-14 августа" in result.draft_text
    assert repair_prompts and "course.start.dates" in repair_prompts[0]


def test_phase1_coverage_repair_requires_price_when_draft_only_mentions_format() -> None:
    store = FactStore(
        catalog=("price.online",),
        store={"unpk": {"price.online": "УНПК онлайн для 9 класса: семестр — 69 900 ₽."}},
    )
    result = run_pipeline(
        conversation=_conv("онлайн для 9 класса сколько стоит?"),
        active_brand="unpk",
        fact_store=store,
        understand_fn=_understanding(
            {
                "current_question": "цена онлайн для 9 класса",
                "subquestions": [
                    {"text": "цена онлайн для 9 класса", "answerable": "self", "needed_fact_keys": ["price.online"]}
                ],
                "answerability": "answer_self",
            }
        ),
        draft_fn=lambda _prompt: "Да, онлайн-формат для 9 класса есть.",
        repair_fn=lambda _prompt: "Онлайн для 9 класса: семестр — 69 900 ₽.",
        faithfulness_fn=lambda _prompt: {"unsupported": []},
    )

    assert result.route == "bot_answer_self"
    assert result.repaired is True
    assert "69 900" in result.draft_text


def test_phase1_coverage_cite_only_fallback_uses_retrieved_fact_without_repair_fn() -> None:
    store = FactStore(
        catalog=("price.year",),
        store={"foton": {"price.year": "Фотон очно за год — 82 000 ₽."}},
    )
    result = run_pipeline(
        conversation=_conv("сколько стоит год очно?"),
        active_brand="foton",
        fact_store=store,
        understand_fn=_understanding(
            {
                "current_question": "цена очно за год",
                "subquestions": [
                    {"text": "цена очно за год", "answerable": "self", "needed_fact_keys": ["price.year"]}
                ],
                "answerability": "answer_self",
            }
        ),
        draft_fn=lambda _prompt: "Очный формат доступен.",
        faithfulness_fn=lambda _prompt: {"unsupported": []},
    )

    assert result.route == "bot_answer_self"
    assert result.repaired is True
    assert "82 000" in result.draft_text


def test_phase1_coverage_does_not_force_manager_subquestion() -> None:
    store = FactStore(catalog=("schedule.exact_day",), store={"unpk": {}})
    result = run_pipeline(
        conversation=_conv("в какой день группа?"),
        active_brand="unpk",
        fact_store=store,
        understand_fn=_understanding(
            {
                "current_question": "точный день группы",
                "subquestions": [
                    {"text": "точный день группы", "answerable": "manager", "needed_fact_keys": ["schedule.exact_day"]}
                ],
                "answerability": "manager_only",
            }
        ),
        draft_fn=lambda _prompt: "Точный день группы подтвердит менеджер.",
        repair_fn=lambda _prompt: "Не должно вызываться",
        faithfulness_fn=lambda _prompt: {"unsupported": []},
    )

    assert result.route == "draft_for_manager"
    assert result.fallback_reason == "contract_manager_only"
    assert "менеджер" in result.draft_text.casefold()


def test_phase1_coverage_noop_when_no_rfk() -> None:
    store = FactStore(catalog=("price.online",), store={"unpk": {}})
    result = run_pipeline(
        conversation=_conv("сколько стоит онлайн?"),
        active_brand="unpk",
        fact_store=store,
        understand_fn=_understanding(
            {
                "current_question": "цена онлайн",
                "subquestions": [
                    {"text": "цена онлайн", "answerable": "self", "needed_fact_keys": ["price.online"]}
                ],
                "answerability": "answer_self",
            }
        ),
        draft_fn=lambda _prompt: "Точную цену подтвердит менеджер.",
        faithfulness_fn=lambda _prompt: {"unsupported": []},
    )

    assert result.missing == ("price.online",)
    assert result.route == "bot_answer_self"
    assert "менеджер" in result.draft_text.casefold()


def test_phase1_coverage_noop_when_retrieved_fact_is_already_cited() -> None:
    store = FactStore(catalog=("price.online",), store={"foton": {"price.online": "Онлайн: семестр — 29 750 ₽."}})
    calls: list[str] = []
    result = run_pipeline(
        conversation=_conv("сколько онлайн?"),
        active_brand="foton",
        fact_store=store,
        understand_fn=_understanding(
            {
                "current_question": "цена онлайн",
                "subquestions": [
                    {"text": "цена онлайн", "answerable": "self", "needed_fact_keys": ["price.online"]}
                ],
                "answerability": "answer_self",
            }
        ),
        draft_fn=lambda _prompt: "Онлайн: семестр — 29 750 ₽.",
        repair_fn=lambda prompt: calls.append(prompt) or "Не должно вызываться",
        faithfulness_fn=lambda _prompt: {"unsupported": []},
    )

    assert result.route == "bot_answer_self"
    assert result.repaired is False
    assert calls == []
    assert "29 750" in result.draft_text


def test_phase1_composition_counts_two_offline_subjects_total() -> None:
    store = FactStore(
        catalog=("price.offline.year", "discounts.second_subject.offline.pct", "discounts.stacking_rule"),
        store={
            "foton": {
                "price.offline.year": "Фотон очно за год — 74 500 ₽.",
                "discounts.second_subject.offline.pct": "Фотон: на второй очный предмет действует скидка 20%.",
                "discounts.stacking_rule": "Скидки не суммируются.",
            }
        },
    )
    result = run_pipeline(
        conversation=_conv("два предмета очно на год сколько выйдет?"),
        active_brand="foton",
        fact_store=store,
        understand_fn=_understanding(
            {
                "current_question": "два предмета очно год",
                "subquestions": [
                    {
                        "text": "два предмета очно год",
                        "answerable": "self",
                        "needed_fact_keys": [
                            "price.offline.year",
                            "discounts.second_subject.offline.pct",
                            "discounts.stacking_rule",
                        ],
                    }
                ],
                "answerability": "answer_self",
            }
        ),
        draft_fn=lambda _prompt: "Первый предмет 74 500 ₽, второй со скидкой 20%.",
        faithfulness_fn=lambda _prompt: {"unsupported": []},
    )

    assert result.route == "bot_answer_self"
    assert "134 100 ₽" in result.draft_text
    assert "59 600 ₽" in result.draft_text


def test_phase1_composition_counts_two_online_subjects_total() -> None:
    store = FactStore(
        catalog=("price.online.year", "discounts.second_subject.online.pct"),
        store={
            "foton": {
                "price.online.year": "Фотон онлайн за год — 47 250 ₽.",
                "discounts.second_subject.online.pct": "Фотон: на второй онлайн-предмет действует скидка 30%.",
            }
        },
    )
    result = run_pipeline(
        conversation=_conv("два предмета онлайн на год сколько получится?"),
        active_brand="foton",
        fact_store=store,
        understand_fn=_understanding(
            {
                "current_question": "два предмета онлайн год",
                "subquestions": [
                    {
                        "text": "два предмета онлайн год",
                        "answerable": "self",
                        "needed_fact_keys": ["price.online.year", "discounts.second_subject.online.pct"],
                    }
                ],
                "answerability": "answer_self",
            }
        ),
        draft_fn=lambda _prompt: "47 250 ₽ + скидка 30% на второй предмет.",
        faithfulness_fn=lambda _prompt: {"unsupported": []},
    )

    assert "80 325 ₽" in result.draft_text
    assert "33 075 ₽" in result.draft_text


def test_phase1_composition_counts_three_subjects_without_stacking_discounts() -> None:
    store = FactStore(
        catalog=("price.offline.year", "discounts.second_subject.offline.pct", "discounts.stacking_rule"),
        store={
            "unpk": {
                "price.offline.year": "УНПК очно за год — 82 000 ₽.",
                "discounts.second_subject.offline.pct": "УНПК: на второй очный предмет действует скидка 20%.",
                "discounts.stacking_rule": "Скидки не суммируются; применяется наибольшая.",
            }
        },
    )
    result = run_pipeline(
        conversation=_conv("три предмета очно за год, какая сумма?"),
        active_brand="unpk",
        fact_store=store,
        understand_fn=_understanding(
            {
                "current_question": "три предмета очно год",
                "subquestions": [
                    {
                        "text": "три предмета очно год",
                        "answerable": "self",
                        "needed_fact_keys": [
                            "price.offline.year",
                            "discounts.second_subject.offline.pct",
                            "discounts.stacking_rule",
                        ],
                    }
                ],
                "answerability": "answer_self",
            }
        ),
        draft_fn=lambda _prompt: "82 000 ₽ за первый, скидка 20% на следующие.",
        faithfulness_fn=lambda _prompt: {"unsupported": []},
    )

    assert "213 200 ₽" in result.draft_text
    assert result.draft_text.count("65 600 ₽") == 2
    assert "не суммируются" in result.draft_text.casefold()


def test_phase1_composition_counts_ordinal_third_subject_as_discounted_subject() -> None:
    store = FactStore(
        catalog=("price.offline.year", "discounts.second_subject.offline.pct", "discounts.stacking_rule"),
        store={
            "foton": {
                "price.offline.year": "Фотон очно за год — 74 500 ₽.",
                "discounts.second_subject.offline.pct": "Фотон: на второй и последующий очный предмет одного ребёнка действует скидка 20%.",
                "discounts.stacking_rule": "Скидки не суммируются; применяется наибольшая.",
            }
        },
    )
    result = run_pipeline(
        conversation=_conv("а третий предмет очно тому же ребенку тоже со скидкой?"),
        active_brand="foton",
        fact_store=store,
        understand_fn=_understanding(
            {
                "current_question": "третий предмет очно тому же ребенку",
                "subquestions": [
                    {
                        "text": "третий предмет очно тому же ребенку",
                        "answerable": "self",
                        "needed_fact_keys": [
                            "price.offline.year",
                            "discounts.second_subject.offline.pct",
                            "discounts.stacking_rule",
                        ],
                    }
                ],
                "answerability": "answer_self",
            }
        ),
        draft_fn=lambda _prompt: "Третий предмет пока без скидки, менеджер уточнит.",
        faithfulness_fn=lambda _prompt: {"unsupported": []},
    )

    assert result.route == "bot_answer_self"
    assert "3-й предмет со скидкой 20%" in result.draft_text
    assert result.draft_text.count("59 600 ₽") == 2
    assert "третий предмет пока без скидки" not in result.draft_text.casefold()
    assert "Итого — 193 700 ₽" in result.draft_text


def test_phase1_composition_does_not_apply_second_subject_discount_to_first_subject() -> None:
    store = FactStore(
        catalog=("price.offline.year", "discounts.second_subject.offline.pct"),
        store={
            "foton": {
                "price.offline.year": "Фотон очно за год — 74 500 ₽.",
                "discounts.second_subject.offline.pct": "Фотон: на второй и последующий очный предмет действует скидка 20%.",
            }
        },
    )
    result = run_pipeline(
        conversation=_conv("первый предмет очно на год сколько стоит?"),
        active_brand="foton",
        fact_store=store,
        understand_fn=_understanding(
            {
                "current_question": "первый предмет очно год",
                "subquestions": [
                    {
                        "text": "первый предмет очно год",
                        "answerable": "self",
                        "needed_fact_keys": ["price.offline.year", "discounts.second_subject.offline.pct"],
                    }
                ],
                "answerability": "answer_self",
            }
        ),
        draft_fn=lambda _prompt: "Очно за год — 74 500 ₽.",
        faithfulness_fn=lambda _prompt: {"unsupported": []},
    )

    assert "59 600 ₽" not in result.draft_text
    assert "Итого" not in result.draft_text
    assert "74 500 ₽" in result.draft_text


def test_phase1_composition_camp_date_price_and_included() -> None:
    store = FactStore(
        catalog=("camp.dates", "camp.price", "camp.included"),
        store={
            "foton": {
                "camp.dates": "ЛВШ Менделеево — смена 3-14 августа.",
                "camp.price": "ЛВШ Менделеево — цена 89 900 ₽.",
                "camp.included": "ЛВШ Менделеево — в стоимость входит обучение и проживание.",
            }
        },
    )
    result = run_pipeline(
        conversation=_conv("когда ближайшая ЛВШ и что входит?"),
        active_brand="foton",
        fact_store=store,
        understand_fn=_understanding(
            {
                "current_question": "даты ближайшей ЛВШ и что входит",
                "subquestions": [
                    {
                        "text": "даты ближайшей ЛВШ и что входит",
                        "answerable": "self",
                        "needed_fact_keys": ["camp.dates", "camp.price", "camp.included"],
                    }
                ],
                "answerability": "answer_self",
            }
        ),
        draft_fn=lambda _prompt: "По ЛВШ сориентирует менеджер.",
        faithfulness_fn=lambda _prompt: {"unsupported": []},
    )

    assert "3-14 августа" in result.draft_text
    assert "89 900 ₽" in result.draft_text
    assert "обучение и проживание" in result.draft_text


def test_phase1_camp_scope_does_not_substitute_regular_online_price_or_format() -> None:
    store = FactStore(
        catalog=("ls_city_2026_foton_format", "regular_online.price.year", "online_courses_format"),
        store={
            "foton": {
                "ls_city_2026_foton_format": "Фотон: городской летний лагерь — очная городская школа, без проживания.",
                "regular_online.price.year": "Фотон онлайн-курс 2026/27: год — 47 250 ₽.",
                "online_courses_format": "Фотон: онлайн-курсы проходят дистанционно.",
            }
        },
    )
    result = run_pipeline(
        conversation=_conv("по смене это онлайн или очно, с проживанием или дневная?"),
        active_brand="foton",
        fact_store=store,
        understand_fn=_understanding(
            {
                "current_question": "формат лагерной смены с проживанием или дневная",
                "subquestions": [
                    {
                        "text": "формат лагерной смены с проживанием или дневная",
                        "answerable": "self",
                        "needed_fact_keys": [
                            "ls_city_2026_foton_format",
                            "regular_online.price.year",
                            "online_courses_format",
                        ],
                    }
                ],
                "answerability": "answer_self",
            }
        ),
        draft_fn=lambda _prompt: "Онлайн-курс стоит 47 250 ₽ за год и проходит дистанционно.",
        faithfulness_fn=lambda _prompt: {"unsupported": []},
    )

    assert result.route == "bot_answer_self"
    assert "без проживания" in result.draft_text.casefold()
    assert "очная городская школа" in result.draft_text.casefold()
    assert "47 250" not in result.draft_text
    assert "онлайн-курс" not in result.draft_text.casefold()
    assert "дистанционно" not in result.draft_text.casefold()


def test_phase1_regular_online_price_still_answers_outside_camp_scope() -> None:
    store = FactStore(
        catalog=("regular_online.price.year", "online_courses_format"),
        store={
            "foton": {
                "regular_online.price.year": "Фотон онлайн-курс 2026/27: год — 47 250 ₽.",
                "online_courses_format": "Фотон: онлайн-курсы проходят дистанционно.",
            }
        },
    )
    result = run_pipeline(
        conversation=_conv("онлайн или очно и сколько стоит онлайн?"),
        active_brand="foton",
        fact_store=store,
        understand_fn=_understanding(
            {
                "current_question": "формат и цена онлайн-курса",
                "subquestions": [
                    {
                        "text": "формат и цена онлайн-курса",
                        "answerable": "self",
                        "needed_fact_keys": ["regular_online.price.year", "online_courses_format"],
                    }
                ],
                "answerability": "answer_self",
            }
        ),
        draft_fn=lambda _prompt: "Онлайн-курс стоит 47 250 ₽ за год и проходит дистанционно.",
        faithfulness_fn=lambda _prompt: {"unsupported": []},
    )

    assert result.route == "bot_answer_self"
    assert "47 250 ₽" in result.draft_text
    assert "онлайн" in result.draft_text.casefold()


def test_phase1_composition_does_not_apply_subject_discount_without_subject_count() -> None:
    store = FactStore(
        catalog=("price.offline.year", "discounts.second_subject.offline.pct"),
        store={
            "foton": {
                "price.offline.year": "Фотон очно за год — 74 500 ₽.",
                "discounts.second_subject.offline.pct": "Фотон: на второй очный предмет действует скидка 20%.",
            }
        },
    )
    result = run_pipeline(
        conversation=_conv("сколько стоит очно за год?"),
        active_brand="foton",
        fact_store=store,
        understand_fn=_understanding(
            {
                "current_question": "цена очно год",
                "subquestions": [
                    {
                        "text": "цена очно год",
                        "answerable": "self",
                        "needed_fact_keys": ["price.offline.year", "discounts.second_subject.offline.pct"],
                    }
                ],
                "answerability": "answer_self",
            }
        ),
        draft_fn=lambda _prompt: "Очно за год — 74 500 ₽.",
        faithfulness_fn=lambda _prompt: {"unsupported": []},
    )

    assert "134 100" not in result.draft_text
    assert "74 500 ₽" in result.draft_text


def test_phase1_composition_does_not_calculate_when_discount_fact_missing() -> None:
    store = FactStore(
        catalog=("price.offline.year", "discounts.second_subject.offline.pct"),
        store={"foton": {"price.offline.year": "Фотон очно за год — 74 500 ₽."}},
    )
    result = run_pipeline(
        conversation=_conv("два предмета очно на год сколько выйдет?"),
        active_brand="foton",
        fact_store=store,
        understand_fn=_understanding(
            {
                "current_question": "два предмета очно год",
                "subquestions": [
                    {
                        "text": "два предмета очно год",
                        "answerable": "self",
                        "needed_fact_keys": ["price.offline.year", "discounts.second_subject.offline.pct"],
                    }
                ],
                "answerability": "answer_self",
            }
        ),
        draft_fn=lambda _prompt: "Очно за год — 74 500 ₽. Скидку на второй предмет уточнит менеджер.",
        faithfulness_fn=lambda _prompt: {"unsupported": []},
    )

    assert "134 100" not in result.draft_text
    assert "74 500 ₽" in result.draft_text


def test_phase1_composition_does_not_emit_monthly_orientir_before_math_tolerance() -> None:
    store = FactStore(catalog=("price.year",), store={"unpk": {"price.year": "УНПК очно за год — 82 000 ₽."}})
    result = run_pipeline(
        conversation=_conv("сколько примерно в месяц?"),
        active_brand="unpk",
        fact_store=store,
        understand_fn=_understanding(
            {
                "current_question": "примерная сумма в месяц",
                "subquestions": [
                    {"text": "примерная сумма в месяц", "answerable": "self", "needed_fact_keys": ["price.year"]}
                ],
                "answerability": "answer_self",
            }
        ),
        draft_fn=lambda _prompt: "Годовая цена — 82 000 ₽.",
        faithfulness_fn=lambda _prompt: {"unsupported": []},
    )

    assert "9 100" not in result.draft_text
    assert "82 000 ₽" in result.draft_text


def test_phase1_empty_handoff_replaced_when_answer_self_has_rfk() -> None:
    store = FactStore(catalog=("price.online",), store={"unpk": {"price.online": "УНПК онлайн: семестр — 69 900 ₽."}})
    result = run_pipeline(
        conversation=_conv("сколько стоит онлайн?"),
        active_brand="unpk",
        fact_store=store,
        understand_fn=_understanding(
            {
                "current_question": "цена онлайн",
                "subquestions": [
                    {"text": "цена онлайн", "answerable": "self", "needed_fact_keys": ["price.online"]}
                ],
                "answerability": "answer_self",
            }
        ),
        draft_fn=lambda _prompt: "Передам менеджеру уточнить именно это.",
        faithfulness_fn=lambda _prompt: {"unsupported": []},
    )

    assert result.route == "bot_answer_self"
    assert result.repaired is True
    assert "69 900 ₽" in result.draft_text
    assert "Передам менеджеру уточнить именно это" not in result.draft_text


def test_phase1_empty_handoff_replaced_after_no_draft_fn() -> None:
    store = FactStore(catalog=("price.online",), store={"foton": {"price.online": "Фотон онлайн: семестр — 29 750 ₽."}})
    result = run_pipeline(
        conversation=_conv("сколько стоит онлайн?"),
        active_brand="foton",
        fact_store=store,
        understand_fn=_understanding(
            {
                "current_question": "цена онлайн",
                "subquestions": [
                    {"text": "цена онлайн", "answerable": "self", "needed_fact_keys": ["price.online"]}
                ],
                "answerability": "answer_self",
            }
        ),
        draft_fn=None,
        faithfulness_fn=lambda _prompt: {"unsupported": []},
    )

    assert result.route == "bot_answer_self"
    assert result.fallback_reason == "fact_composer_after_no_draft_fn"
    assert "29 750 ₽" in result.draft_text


def test_phase1_empty_handoff_replaced_after_draft_error() -> None:
    store = FactStore(catalog=("price.online",), store={"foton": {"price.online": "Фотон онлайн: семестр — 29 750 ₽."}})
    result = run_pipeline(
        conversation=_conv("сколько стоит онлайн?"),
        active_brand="foton",
        fact_store=store,
        understand_fn=_understanding(
            {
                "current_question": "цена онлайн",
                "subquestions": [
                    {"text": "цена онлайн", "answerable": "self", "needed_fact_keys": ["price.online"]}
                ],
                "answerability": "answer_self",
            }
        ),
        draft_fn=lambda _prompt: "",
        faithfulness_fn=lambda _prompt: {"unsupported": []},
    )

    assert result.route == "bot_answer_self"
    assert result.fallback_reason == "fact_composer_after_draft_error"
    assert "29 750 ₽" in result.draft_text


def test_phase1_empty_handoff_does_not_override_manager_no_fact() -> None:
    store = FactStore(catalog=("schedule.exact_day",), store={"unpk": {}})
    result = run_pipeline(
        conversation=_conv("в какой день группа?"),
        active_brand="unpk",
        fact_store=store,
        understand_fn=_understanding(
            {
                "current_question": "точный день группы",
                "subquestions": [
                    {"text": "точный день группы", "answerable": "manager", "needed_fact_keys": ["schedule.exact_day"]}
                ],
                "answerability": "manager_only",
            }
        ),
        draft_fn=lambda _prompt: "Передам менеджеру уточнить именно это.",
        faithfulness_fn=lambda _prompt: {"unsupported": []},
    )

    assert result.route == "draft_for_manager"
    assert result.fallback_reason == "contract_manager_only"
    assert "менеджер" in result.draft_text.casefold()


def test_phase1_empty_handoff_does_not_override_p0() -> None:
    store = FactStore(catalog=("price.online",), store={"foton": {"price.online": "Фотон онлайн: семестр — 29 750 ₽."}})
    result = run_pipeline(
        conversation=_conv("я оплатил, занятий нет, верните деньги"),
        active_brand="foton",
        fact_store=store,
        understand_fn=_understanding(
            {
                "current_question": "цена онлайн",
                "subquestions": [
                    {"text": "цена онлайн", "answerable": "self", "needed_fact_keys": ["price.online"]}
                ],
                "answerability": "answer_self",
            }
        ),
        draft_fn=lambda _prompt: "Онлайн стоит 29 750 ₽.",
    )

    assert result.route == "manager_only"
    assert result.fallback_reason == "p0"
    assert "29 750" not in result.draft_text


def test_empty_facts_guard_blocks_answer_self_fact_question_without_rfk() -> None:
    store = FactStore(catalog=("schedule.exact_day",), store={"unpk": {}})

    def unexpected_draft(_prompt: str) -> str:
        raise AssertionError("empty-facts guard must stop before draft generation")

    result = run_pipeline(
        conversation=_conv("в какой день группа?"),
        active_brand="unpk",
        fact_store=store,
        understand_fn=_understanding(
            {
                "current_question": "точный день группы",
                "subquestions": [
                    {"text": "точный день группы", "answerable": "self", "needed_fact_keys": ["schedule.exact_day"]}
                ],
                "answerability": "answer_self",
            }
        ),
        draft_fn=unexpected_draft,
        faithfulness_fn=lambda _prompt: {"unsupported": []},
    )

    assert result.route == "bot_answer_self"
    assert result.fallback_reason == "empty_facts_no_fabrication"
    assert "менеджер" in result.draft_text.casefold()
    assert "вторник" not in result.draft_text.casefold()
    assert result.missing == ("schedule.exact_day",)


def test_empty_facts_guard_does_not_intercept_answer_self_with_retrieved_fact() -> None:
    store = FactStore(catalog=("price.online",), store={"foton": {"price.online": "Фотон онлайн: семестр — 29 750 ₽."}})
    result = run_pipeline(
        conversation=_conv("сколько стоит онлайн?"),
        active_brand="foton",
        fact_store=store,
        understand_fn=_understanding(
            {
                "current_question": "цена онлайн",
                "subquestions": [
                    {"text": "цена онлайн", "answerable": "self", "needed_fact_keys": ["price.online"]}
                ],
                "answerability": "answer_self",
            }
        ),
        draft_fn=lambda _prompt: "Фотон онлайн: семестр — 29 750 ₽.",
        faithfulness_fn=lambda _prompt: {"unsupported": []},
    )

    assert result.route == "bot_answer_self"
    assert result.fallback_reason == ""
    assert "29 750 ₽" in result.draft_text


def test_empty_facts_guard_does_not_intercept_answer_self_without_needed_facts() -> None:
    store = FactStore(catalog=(), store={"foton": {}})
    result = run_pipeline(
        conversation=_conv("здравствуйте"),
        active_brand="foton",
        fact_store=store,
        understand_fn=_understanding(
            {
                "current_question": "приветствие",
                "subquestions": [],
                "needed_fact_keys": [],
                "answerability": "answer_self",
            }
        ),
        draft_fn=lambda _prompt: "Здравствуйте! Подскажите, пожалуйста, какой курс интересует?",
        faithfulness_fn=lambda _prompt: {"unsupported": []},
    )

    assert result.route == "bot_answer_self"
    assert result.fallback_reason == ""
    assert "менеджер" not in result.draft_text.casefold()


def test_narrow_discount_subquestion_ignores_unrelated_rfk_facts() -> None:
    store = FactStore(
        catalog=(
            "discounts.multichild.pct",
            "discounts.second_subject_offline.pct",
            "discounts.refer_a_friend.pct",
            "discounts.year_payment.pct",
            "discounts.early_booking.pct",
        ),
        store={
            "unpk": {
                "discounts.multichild.pct": "УНПК: для семьи с двумя детьми действует скидка 10%.",
                "discounts.second_subject_offline.pct": "УНПК: на второй очный предмет действует скидка 20%.",
                "discounts.refer_a_friend.pct": "УНПК: по акции «приведи друга» скидка 5%.",
                "discounts.year_payment.pct": "УНПК: при оплате за год действует скидка 7%.",
                "discounts.early_booking.pct": "УНПК: раннее бронирование даёт скидку 3%.",
            }
        },
    )
    result = run_pipeline(
        conversation=_conv("у нас двое детей, скидка есть?"),
        active_brand="unpk",
        fact_store=store,
        understand_fn=_understanding(
            {
                "current_question": "скидка для семьи с двумя детьми",
                "subquestions": [
                    {
                        "text": "двое детей — скидка?",
                        "answerable": "self",
                        "needed_fact_keys": [
                            "discounts.multichild.pct",
                            "discounts.second_subject_offline.pct",
                            "discounts.refer_a_friend.pct",
                            "discounts.year_payment.pct",
                            "discounts.early_booking.pct",
                        ],
                    }
                ],
                "answerability": "answer_self",
            }
        ),
        draft_fn=lambda _prompt: "Для семьи с двумя детьми действует скидка 10%.",
        faithfulness_fn=lambda _prompt: {"unsupported": ["для семьи с двумя детьми действует скидка 10%"]},
    )

    assert result.route == "bot_answer_self"
    assert result.fallback_reason == ""
    assert not result.unsupported_claims


def test_narrow_grade_price_subquestion_ignores_other_price_facts() -> None:
    store = FactStore(
        catalog=("prices.grade6.semester", "prices.grade9.semester", "prices.grade11.year"),
        store={
            "unpk": {
                "prices.grade6.semester": "УНПК: для 6 класса семестр стоит 49 000 ₽.",
                "prices.grade9.semester": "УНПК: для 9 класса семестр стоит 69 900 ₽.",
                "prices.grade11.year": "УНПК: для 11 класса год стоит 119 000 ₽.",
            }
        },
    )
    result = run_pipeline(
        conversation=_conv("6 класс цена за семестр?"),
        active_brand="unpk",
        fact_store=store,
        understand_fn=_understanding(
            {
                "current_question": "6 класс цена",
                "subquestions": [
                    {
                        "text": "6 класс цена за семестр",
                        "answerable": "self",
                        "needed_fact_keys": ["prices.grade6.semester", "prices.grade9.semester", "prices.grade11.year"],
                    }
                ],
                "answerability": "answer_self",
            }
        ),
        draft_fn=lambda _prompt: "Для 6 класса семестр стоит 49 000 ₽.",
        faithfulness_fn=lambda _prompt: {"unsupported": ["для 6 класса семестр стоит 49 000 ₽"]},
    )

    assert result.route == "bot_answer_self"
    assert result.fallback_reason == ""
    assert not result.unsupported_claims


def test_narrow_supported_claim_siblings_pass_when_claim_matches_current_subquestion_fact() -> None:
    rows = (
        {
            "message": "на второй предмет скидка есть?",
            "question": "скидка на второй предмет",
            "keys": ("discounts.second_subject.online.pct", "discounts.multichild.pct"),
            "facts": {
                "discounts.second_subject.online.pct": "УНПК: на второй онлайн-предмет действует скидка 20%.",
                "discounts.multichild.pct": "УНПК: семейная скидка составляет 10%.",
            },
            "draft": "На второй онлайн-предмет действует скидка 20%.",
            "claim": "на второй онлайн-предмет действует скидка 20%",
        },
        {
            "message": "у нас двое детей, семейная скидка есть?",
            "question": "семейная скидка для двух детей",
            "keys": ("discounts.family.pct", "discounts.second_subject.online.pct"),
            "facts": {
                "discounts.family.pct": "УНПК: семейная скидка для двух детей составляет 10%.",
                "discounts.second_subject.online.pct": "УНПК: на второй онлайн-предмет действует скидка 20%.",
            },
            "draft": "Семейная скидка для двух детей составляет 10%.",
            "claim": "семейная скидка для двух детей составляет 10%",
        },
        {
            "message": "по акции с другом есть скидка?",
            "question": "скидка приведи друга",
            "keys": ("discounts.referral.pct", "discounts.multichild.pct"),
            "facts": {
                "discounts.referral.pct": "УНПК: по акции «приведи друга» действует скидка 5%.",
                "discounts.multichild.pct": "УНПК: для семьи с двумя детьми действует скидка 10%.",
            },
            "draft": "По акции «приведи друга» действует скидка 5%.",
            "claim": "по акции приведи друга действует скидка 5%",
        },
        {
            "message": "если платить за год, скидка есть?",
            "question": "скидка при оплате за год",
            "keys": ("discounts.year_payment.pct", "discounts.referral.pct"),
            "facts": {
                "discounts.year_payment.pct": "УНПК: при оплате за год действует скидка 7%.",
                "discounts.referral.pct": "УНПК: по акции «приведи друга» действует скидка 5%.",
            },
            "draft": "При оплате за год действует скидка 7%.",
            "claim": "при оплате за год действует скидка 7%",
        },
        {
            "message": "ранняя скидка бывает?",
            "question": "раннее бронирование скидка",
            "keys": ("discounts.early_booking.pct", "discounts.year_payment.pct"),
            "facts": {
                "discounts.early_booking.pct": "УНПК: раннее бронирование даёт скидку 3%.",
                "discounts.year_payment.pct": "УНПК: при оплате за год действует скидка 7%.",
            },
            "draft": "Раннее бронирование даёт скидку 3%.",
            "claim": "раннее бронирование даёт скидку 3%",
        },
    )
    for row in rows:
        store = FactStore(catalog=tuple(row["keys"]), store={"unpk": row["facts"]})
        result = run_pipeline(
            conversation=_conv(row["message"]),
            active_brand="unpk",
            fact_store=store,
            understand_fn=_understanding(
                {
                    "current_question": row["question"],
                    "subquestions": [
                        {
                            "text": row["question"],
                            "answerable": "self",
                            "needed_fact_keys": list(row["keys"]),
                        }
                    ],
                    "answerability": "answer_self",
                }
            ),
            draft_fn=lambda _prompt, draft=row["draft"]: draft,
            faithfulness_fn=lambda _prompt, claim=row["claim"]: {"unsupported": [claim]},
        )

        assert result.route == "bot_answer_self", row["message"]
        assert result.fallback_reason == "", row["message"]
        assert not result.unsupported_claims, row["message"]


def test_narrow_discount_wrong_value_stays_unsupported_even_if_other_rfk_fact_has_number() -> None:
    store = FactStore(
        catalog=("discounts.multichild.pct", "discounts.refer_a_friend.pct"),
        store={
            "unpk": {
                "discounts.multichild.pct": "УНПК: для семьи с двумя детьми действует скидка 10%.",
                "discounts.refer_a_friend.pct": "УНПК: по акции «приведи друга» действует скидка 15%.",
            }
        },
    )
    result = run_pipeline(
        conversation=_conv("у нас двое детей, скидка есть?"),
        active_brand="unpk",
        fact_store=store,
        understand_fn=_understanding(
            {
                "current_question": "скидка для семьи с двумя детьми",
                "subquestions": [
                    {
                        "text": "двое детей — скидка?",
                        "answerable": "self",
                        "needed_fact_keys": ["discounts.multichild.pct", "discounts.refer_a_friend.pct"],
                    }
                ],
                "answerability": "answer_self",
            }
        ),
        draft_fn=lambda _prompt: "Для семьи с двумя детьми действует скидка 15%.",
        faithfulness_fn=lambda _prompt: {"unsupported": ["для семьи с двумя детьми действует скидка 15%"]},
    )

    assert result.route == "draft_for_manager"
    assert result.fallback_reason == "hard_verification_failed"
    assert result.unsupported_claims == ("для семьи с двумя детьми действует скидка 15%",)


def test_narrow_claim_outside_rfk_stays_blocked() -> None:
    store = FactStore(
        catalog=("discounts.multichild.pct",),
        store={"unpk": {"discounts.multichild.pct": "УНПК: для семьи с двумя детьми действует скидка 10%."}},
    )
    result = run_pipeline(
        conversation=_conv("какие результаты будут?"),
        active_brand="unpk",
        fact_store=store,
        understand_fn=_understanding(
            {
                "current_question": "результаты обучения",
                "subquestions": [
                    {
                        "text": "результаты обучения",
                        "answerable": "self",
                        "needed_fact_keys": ["discounts.multichild.pct"],
                    }
                ],
                "answerability": "answer_self",
            }
        ),
        draft_fn=lambda _prompt: "По итогам курса ученик наберёт 100 баллов.",
        faithfulness_fn=lambda _prompt: {"unsupported": ["ученик наберёт 100 баллов"]},
    )

    assert result.route == "draft_for_manager"
    assert result.fallback_reason == "hard_verification_failed"
    assert result.unsupported_claims == ("ученик наберёт 100 баллов",)


def test_narrow_claim_with_empty_rfk_stays_blocked() -> None:
    store = FactStore(catalog=(), store={"unpk": {}})
    result = run_pipeline(
        conversation=_conv("сколько стоит?"),
        active_brand="unpk",
        fact_store=store,
        understand_fn=_understanding(
            {
                "current_question": "цена",
                "subquestions": [{"text": "цена", "answerable": "self", "needed_fact_keys": []}],
                "answerability": "answer_self",
            }
        ),
        draft_fn=lambda _prompt: "Семестр стоит 49 000 ₽.",
        faithfulness_fn=lambda _prompt: {"unsupported": ["семестр стоит 49 000 ₽"]},
    )

    assert result.route == "draft_for_manager"
    assert result.fallback_reason == "hard_verification_failed"


def test_safe_fallback_does_not_leak_third_person_question() -> None:
    store = FactStore(catalog=("price.online",), store={"unpk": {}})
    questions = (
        "Клиент спрашивает точный порядок оформления",
        "Клиент уточняет индивидуальное условие",
        "клиент хочет понять порядок согласования",
        "Клиент спрашивает про оплату переводом",
        "Клиент интересуется датой старта группы",
    )

    for question in questions:
        result = run_pipeline(
            conversation=_conv("сколько стоит?"),
            active_brand="unpk",
            fact_store=store,
            understand_fn=_understanding(
                {
                    "current_question": question,
                    "needed_fact_keys": ["price.online"],
                    "answerability": "manager_only",
                }
            ),
            draft_fn=None,
            faithfulness_fn=lambda _prompt: {"unsupported": []},
        )

        assert result.route == "draft_for_manager"
        assert result.fallback_reason == "contract_manager_only"
        assert "клиент" not in result.draft_text.casefold()
        assert "спрашивает" not in result.draft_text.casefold()
        assert "уточняет" not in result.draft_text.casefold()
        assert "интересуется" not in result.draft_text.casefold()
        assert "менеджер" in result.draft_text.casefold()


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


def test_r1_manager_only_contract_with_exact_retrieved_fact_is_promoted() -> None:
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
    assert result.route == "bot_answer_self"
    assert result.fallback_reason == ""
    assert "29 750" in result.draft_text


def test_r1_neighbor_payment_fact_does_not_promote_manager_only_to_autonomous_answer() -> None:
    store = FactStore(
        catalog=("installment.tbank",),
        store={"foton": {"installment.tbank": "Фотон: есть рассрочка через Т-Банк на 6, 10 или 12 месяцев."}},
    )
    result = run_pipeline(
        conversation=_conv("а помесячно прямым переводом на счёт можно?"),
        active_brand="foton",
        fact_store=store,
        understand_fn=_understanding(
            {
                "current_question": "а помесячно прямым переводом на счёт можно?",
                "needed_fact_keys": ["installment.tbank"],
                "answerability": "manager_only",
            }
        ),
        draft_fn=None,
        faithfulness_fn=lambda _prompt: {"unsupported": []},
    )
    assert result.route == "draft_for_manager"
    assert result.fallback_reason == "no_draft_fn"
    assert "точного подтверждения нет" in result.draft_text.casefold()
    assert result.draft_text.casefold().index("точного подтверждения нет") < result.draft_text.casefold().index("из подтверждённого")
    assert "т-банк" in result.draft_text.casefold()


def test_secondary_payment_fact_skips_lvsh_when_client_did_not_ask_camp() -> None:
    store = FactStore(
        catalog=("lvsh_mendeleevo_2026.payment_options_2026.client_safe_text",),
        store={
            "foton": {
                "lvsh_mendeleevo_2026.payment_options_2026.client_safe_text": "Для ЛВШ Фотона доступны варианты оплаты частями на 6, 10 или 12 месяцев, а также сервис Долями."
            }
        },
    )
    result = run_pipeline(
        conversation=_conv("можно оплатить банковским переводом на счёт?"),
        active_brand="foton",
        fact_store=store,
        understand_fn=_understanding(
            {
                "current_question": "Можно ли оплатить банковским переводом на счёт?",
                "needed_fact_keys": ["lvsh_mendeleevo_2026.payment_options_2026.client_safe_text"],
                "answerability": "manager_only",
            }
        ),
        draft_fn=lambda _prompt: "Для ЛВШ Фотона доступны варианты оплаты частями на 6, 10 или 12 месяцев.",
        faithfulness_fn=lambda _prompt: {"unsupported": []},
    )
    assert result.route == "draft_for_manager"
    assert result.fallback_reason == "hard_verification_failed"
    assert "лвш" not in result.draft_text.casefold()
    assert "менделеев" not in result.draft_text.casefold()
    assert "долями" not in result.draft_text.casefold()


def test_r1_address_exact_fact_answers_without_manager_handoff() -> None:
    store = FactStore(
        catalog=(
            "locations_unpk.addresses.1.address",
            "locations_unpk.addresses.1.city",
            "locations_unpk.addresses.1.metro",
        ),
        store={
            "unpk": {
                "locations_unpk.addresses.1.address": "УНПК: адрес и место занятий — Сретенка, 20.",
                "locations_unpk.addresses.1.city": "УНПК: адрес и место занятий — Москва.",
                "locations_unpk.addresses.1.metro": "УНПК: адрес и место занятий — Чистые Пруды.",
            }
        },
    )
    result = run_pipeline(
        conversation=_conv("где вы находитесь в Москве?"),
        active_brand="unpk",
        fact_store=store,
        understand_fn=_understanding(
            {
                "current_question": "Где вы находитесь в Москве?",
                "needed_fact_keys": [
                    "locations_unpk.addresses.1.address",
                    "locations_unpk.addresses.1.city",
                    "locations_unpk.addresses.1.metro",
                ],
                "answerability": "manager_only",
            }
        ),
        draft_fn=lambda _prompt: "Передам менеджеру.",
        faithfulness_fn=lambda _prompt: {"unsupported": ["адрес"]},
    )
    assert result.route == "bot_answer_self"
    assert result.fallback_reason == "direct_exact_fact_answer"
    assert "Сретенка, 20" in result.draft_text
    assert "менеджер уточнит" not in result.draft_text.casefold()


def test_address_fact_does_not_answer_non_address_slot_fill() -> None:
    store = FactStore(
        catalog=(
            "locations_unpk.addresses.1.address",
            "locations_unpk.addresses.1.city",
            "locations_unpk.addresses.1.metro",
        ),
        store={
            "unpk": {
                "locations_unpk.addresses.1.address": "УНПК: адрес и место занятий — Сретенка, 20.",
                "locations_unpk.addresses.1.city": "УНПК: адрес и место занятий — Москва.",
                "locations_unpk.addresses.1.metro": "УНПК: адрес и место занятий — Чистые Пруды.",
            }
        },
    )
    result = run_pipeline(
        conversation=_conv("понятно, тогда интересует 9 класс информатика очно"),
        active_brand="unpk",
        fact_store=store,
        understand_fn=_understanding(
            {
                "current_question": "интересует 9 класс информатика очно",
                "needed_fact_keys": [
                    "locations_unpk.addresses.1.address",
                    "locations_unpk.addresses.1.city",
                    "locations_unpk.addresses.1.metro",
                ],
                "answerability": "answer_self",
            }
        ),
        draft_fn=lambda _prompt: "В Москва: Сретенка, 20; метро Чистые Пруды.",
        faithfulness_fn=lambda _prompt: {"unsupported": []},
    )
    assert result.route == "draft_for_manager"
    assert result.fallback_reason == "hard_verification_failed"
    assert "Сретенка, 20" not in result.draft_text


def test_r1_direct_payment_fact_answers_without_neighbor_scope() -> None:
    store = FactStore(
        catalog=("payment.methods",),
        store={
            "unpk": {
                "payment.methods": "Оплатить можно по QR-коду или по квитанции/реквизитам в банке. Ссылку формирует бухгалтерия и присылает на email или по СМС."
            }
        },
    )
    result = run_pipeline(
        conversation=_conv("а помесячно это без банка просто напрямую вам платить?"),
        active_brand="unpk",
        fact_store=store,
        understand_fn=_understanding(
            {
                "current_question": "помесячно это без банка просто напрямую вам платить?",
                "needed_fact_keys": ["payment.methods"],
                "answerability": "answer_self",
                "question_type": "existence_yes_no",
                "existence_target": "оплата напрямую УНПК без банка",
            }
        ),
        draft_fn=lambda _prompt: "Да, можно платить помесячно, за семестр или за год.",
        faithfulness_fn=lambda _prompt: {"unsupported": []},
    )
    assert result.route == "bot_answer_self"
    assert result.fallback_reason == "direct_exact_fact_answer"
    assert "qr" in result.draft_text.casefold() or "реквизит" in result.draft_text.casefold()
    assert "за год" not in result.draft_text.casefold()


def test_monthly_no_bank_question_answers_from_two_exact_payment_facts() -> None:
    store = FactStore(
        catalog=(
            "payment_options.bank_installment.absent.client_safe_text",
            "payment_options.available_schedules.1.monthly.discount_extra",
        ),
        store={
            "unpk": {
                "payment_options.bank_installment.absent.client_safe_text": "В УНПК отдельной банковской рассрочки нет.",
                "payment_options.available_schedules.1.monthly.discount_extra": "У нас оплата возможна помесячно, за семестр или за год.",
            }
        },
    )
    result = run_pipeline(
        conversation=(
            {"role": "client", "text": "у вас есть рассрочка через банк?"},
            {"role": "bot", "text": "В УНПК отдельной банковской рассрочки нет. Оплата возможна помесячно."},
            {"role": "client", "text": "а помесячно это без банка, просто каждый месяц платить?"},
        ),
        active_brand="unpk",
        fact_store=store,
        understand_fn=_understanding(
            {
                "current_question": "помесячно это без банка, просто каждый месяц платить?",
                "needed_fact_keys": [
                    "payment_options.bank_installment.absent.client_safe_text",
                    "payment_options.available_schedules.1.monthly.discount_extra",
                ],
                "answerability": "answer_self",
                "question_type": "existence_yes_no",
                "existence_target": "помесячная оплата без банка",
            }
        ),
        draft_fn=lambda _prompt: "Передам менеджеру уточнить.",
        faithfulness_fn=lambda _prompt: {"unsupported": []},
    )
    assert result.route == "bot_answer_self"
    assert result.fallback_reason == "direct_exact_fact_answer"
    assert "банковской рассрочки нет" in result.draft_text.casefold()
    assert "помесячная оплата доступна" in result.draft_text.casefold()
    assert "уже отметил" not in result.draft_text.casefold()


def test_r1_semester_price_is_not_exact_monthly_amount_fact() -> None:
    store = FactStore(
        catalog=("price.semester", "payment.monthly"),
        store={
            "unpk": {
                "price.semester": "Семестр — 51 700 ₽.",
                "payment.monthly": "Доступна помесячная оплата, но сумма зависит от выбранной программы.",
            }
        },
    )
    result = run_pipeline(
        conversation=_conv("а помесячно какая сумма выходит?"),
        active_brand="unpk",
        fact_store=store,
        understand_fn=_understanding(
            {
                "current_question": "помесячно какая сумма выходит?",
                "needed_fact_keys": ["price.semester", "payment.monthly"],
                "answerability": "manager_only",
            }
        ),
        draft_fn=lambda _prompt: "Семестр — 51 700 ₽.",
        faithfulness_fn=lambda _prompt: {"unsupported": []},
    )
    assert result.route == "draft_for_manager"
    assert result.fallback_reason in {"", "hard_verification_failed"}


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
    assert result.draft_text.casefold().index("точного подтверждения нет") < result.draft_text.casefold().index("т-банк")
    assert "прямым переводом" in result.draft_text.casefold()


def test_t6_specific_grade_replaces_supported_range_in_answer() -> None:
    store = FactStore(
        catalog=("price.online_5_11"),
        store={"foton": {"price.online_5_11": "Онлайн для 5–11 классов: семестр — 29 750 ₽, год — 47 250 ₽."}},
    )
    result = run_pipeline(
        conversation=_conv("сколько стоит онлайн для 7 класса?"),
        active_brand="foton",
        fact_store=store,
        understand_fn=_understanding(
            {
                "current_question": "цена онлайн для 7 класса",
                "needed_fact_keys": ["price.online_5_11"],
                "answerability": "answer_self",
            }
        ),
        draft_fn=lambda _prompt: "Для онлайн-формата 5–11 классов: семестр — 29 750 ₽, год — 47 250 ₽.",
        faithfulness_fn=lambda _prompt: {"unsupported": []},
    )
    assert result.route == "bot_answer_self"
    assert "7 класса" in result.draft_text
    assert "5–11 классов" not in result.draft_text


def test_t3_repeat_handoff_changes_tactic_without_new_fact() -> None:
    store = FactStore(catalog=("installment.tbank",), store={"foton": {"installment.tbank": "Рассрочка через Т-Банк на 6, 10 или 12 месяцев."}})
    prior = "Передам менеджеру уточнить именно это: прямой перевод на счёт. Он подтвердит точную информацию."
    result = run_pipeline(
        conversation=(
            {"role": "client", "text": "а переводом на счёт можно?"},
            {"role": "bot", "text": prior},
            {"role": "client", "text": "так переводом на счёт всё-таки можно?"},
        ),
        active_brand="foton",
        fact_store=store,
        understand_fn=_understanding(
            {
                "current_question": "прямой перевод на счёт",
                "needed_fact_keys": ["installment.tbank"],
                "answerability": "manager_only",
            }
        ),
        draft_fn=None,
        faithfulness_fn=lambda _prompt: {"unsupported": []},
    )
    assert result.route == "draft_for_manager"
    assert result.draft_text != prior
    assert "уже отметил" not in result.draft_text.casefold()
    assert "клиент" not in result.draft_text.casefold()
    assert "точную деталь" in result.draft_text.casefold()
    assert "6, 10 или 12" not in result.draft_text


def test_contact_hours_are_not_class_schedule_days() -> None:
    store = FactStore(
        catalog=("contacts_unpk.schedule",),
        store={"unpk": {"contacts_unpk.schedule": "УНПК на связи ежедневно Пн–Вс 10:00–18:00."}},
    )
    result = run_pipeline(
        conversation=_conv("а по каким дням там занятия?"),
        active_brand="unpk",
        fact_store=store,
        understand_fn=_understanding(
            {
                "current_question": "по каким дням там занятия?",
                "needed_fact_keys": ["contacts_unpk.schedule"],
                "answerability": "answer_self",
            }
        ),
        draft_fn=lambda _prompt: "УНПК: расписание — Пн–Вс 10:00–18:00.",
        faithfulness_fn=lambda _prompt: {"unsupported": []},
    )
    assert result.route == "draft_for_manager"
    assert result.fallback_reason == "hard_verification_failed"
    assert "10:00" not in result.draft_text


def test_schedule_publication_fact_answers_without_contact_hours() -> None:
    store = FactStore(
        catalog=("contacts_unpk.schedule", "regular_courses_schedule_publication"),
        store={
            "unpk": {
                "contacts_unpk.schedule": "УНПК: контакты, расписание — Пн-Вс 10:00-18:00.",
                "regular_courses_schedule_publication": "Очные курсы 2026/27 стартуют в середине сентября; расписание и подробная информация появятся в июне.",
            }
        },
    )
    result = run_pipeline(
        conversation=_conv("подскажите расписание очных занятий по физике для 9 класса"),
        active_brand="unpk",
        fact_store=store,
        understand_fn=_understanding(
            {
                "current_question": "расписание очных занятий по физике для 9 класса",
                "needed_fact_keys": ["regular_courses_schedule_publication", "contacts_unpk.schedule"],
                "answerability": "answer_self",
            }
        ),
        draft_fn=lambda _prompt: "УНПК: расписание — Пн–Вс 10:00–18:00.",
        faithfulness_fn=lambda _prompt: {"unsupported": []},
    )
    assert result.route == "bot_answer_self"
    assert result.fallback_reason == "schedule_publication_answer"
    assert "появятся в июне" in result.draft_text.casefold()
    assert "10:00" not in result.draft_text


def test_multitopic_format_and_schedule_answers_both_parts() -> None:
    store = FactStore(
        catalog=("online.format", "regular_courses_schedule_publication", "contacts_unpk.schedule"),
        store={
            "unpk": {
                "online.format": "Онлайн-курсы УНПК проходят на платформе МТС-Link.",
                "regular_courses_schedule_publication": "Очные курсы 2026/27 стартуют в середине сентября; расписание и подробная информация появятся в июне.",
                "contacts_unpk.schedule": "УНПК: контакты, расписание — Пн-Вс 10:00-18:00.",
            }
        },
    )
    result = run_pipeline(
        conversation=_conv("это онлайн или очно? и по каким дням занятия?"),
        active_brand="unpk",
        fact_store=store,
        understand_fn=_understanding(
            {
                "current_question": "это онлайн или очно? и по каким дням занятия?",
                "needed_fact_keys": ["online.format", "regular_courses_schedule_publication", "contacts_unpk.schedule"],
                "answerability": "answer_self",
            }
        ),
        draft_fn=lambda _prompt: "УНПК: расписание — Пн–Вс 10:00–18:00.",
        faithfulness_fn=lambda _prompt: {"unsupported": []},
    )
    assert result.route == "bot_answer_self"
    assert result.fallback_reason == "schedule_publication_answer"
    assert "онлайн-формат" in result.draft_text.casefold()
    assert "очные курсы" in result.draft_text.casefold()
    assert "появятся в июне" in result.draft_text.casefold()
    assert "10:00" not in result.draft_text


def test_weekend_question_prefers_soft_guidance_over_schedule_publication() -> None:
    store = FactStore(
        catalog=("regular_courses_schedule_publication", "objection_responses.inconvenient_time"),
        store={
            "unpk": {
                "regular_courses_schedule_publication": "Очные курсы 2026/27 стартуют в середине сентября; расписание и подробная информация появятся в июне.",
                "objection_responses.inconvenient_time": "УНПК: черновик для ситуации «возражение о неудобном времени»: Разные слоты по выходным.",
            }
        },
    )
    result = run_pipeline(
        conversation=_conv("по выходным группы обычно бывают?"),
        active_brand="unpk",
        fact_store=store,
        understand_fn=_understanding(
            {
                "current_question": "по выходным группы обычно бывают?",
                "needed_fact_keys": ["regular_courses_schedule_publication"],
                "answerability": "answer_self",
            }
        ),
        draft_fn=lambda _prompt: "Расписание появится в июне.",
        faithfulness_fn=lambda _prompt: {"unsupported": []},
    )
    assert result.route == "bot_answer_self"
    assert result.fallback_reason == "soft_weekend_guidance"
    assert "разные варианты слотов" in result.draft_text.casefold()
    assert "в том числе по выходным" in result.draft_text.casefold()


def test_schedule_publication_repeat_changes_wording() -> None:
    store = FactStore(
        catalog=("regular_courses_schedule_publication",),
        store={
            "unpk": {
                "regular_courses_schedule_publication": "Очные курсы 2026/27 стартуют в середине сентября; расписание и подробная информация появятся в июне."
            }
        },
    )
    prior = "Очные курсы 2026/27 стартуют в середине сентября; расписание и подробная информация появятся в июне. Точные дни конкретной группы сейчас не подтверждаю."
    result = run_pipeline(
        conversation=(
            {"role": "client", "text": "по каким дням занятия?"},
            {"role": "bot", "text": prior},
            {"role": "client", "text": "вы уже это написали, а дни какие?"},
        ),
        active_brand="unpk",
        fact_store=store,
        understand_fn=_understanding(
            {
                "current_question": "дни занятий",
                "needed_fact_keys": ["regular_courses_schedule_publication"],
                "answerability": "answer_self",
            }
        ),
        draft_fn=lambda _prompt: prior,
        faithfulness_fn=lambda _prompt: {"unsupported": []},
    )
    assert result.route == "bot_answer_self"
    assert result.draft_text != prior
    assert "опубликуют в июне" in result.draft_text.casefold()


def test_m6_real_p0_composite_does_not_answer_sales_part() -> None:
    store = FactStore(catalog=("price.online",), store={"foton": {"price.online": "Онлайн: семестр — 29 750 ₽."}})
    result = run_pipeline(
        conversation=_conv("занятий нет, верните деньги, и сколько стоит продление?"),
        active_brand="foton",
        fact_store=store,
        understand_fn=_understanding(
            {
                "current_question": "верните деньги и цена продления",
                "needed_fact_keys": ["price.online"],
                "answerability": "answer_self",
                "is_p0": False,
            }
        ),
        draft_fn=lambda _prompt: "Продление стоит 29 750 ₽.",
        faithfulness_fn=lambda _prompt: {"unsupported": []},
    )
    assert result.route == "manager_only"
    assert "29 750" not in result.draft_text
    assert result.contract.is_p0


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

    assert result.route == "bot_answer_self"
    assert result.fallback_reason == "soft_weekend_guidance"
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


def test_build_draft_prompt_includes_dialogue_memory_view_before_history() -> None:
    contract = parse_contract(
        {
            "current_question": "а онлайн для 10?",
            "needed_fact_keys": ["price.online.grade10"],
            "answerability": "answer_self",
        },
        active_brand="foton",
        fact_key_catalog=("price.online.grade10",),
    )
    prompt = build_draft_prompt(
        conversation=_conv("а онлайн для 10?"),
        contract=contract,
        facts={"price.online.grade10": "Фотон онлайн для 10 класса: семестр — 29 750 ₽."},
        missing=(),
        dialogue_memory_view={
            "conversation_summary_short": "обсуждали курс по информатике",
            "topic_focus": {"subject": "информатика", "grade": "10", "format": "онлайн"},
            "open_question": {"text": "сколько стоит онлайн для 10 класса"},
            "known_slots": {"subject": "информатика", "grade": "10"},
            "do_not_ask_again": ["subject", "grade"],
            "last_bot_commitments": ["сориентировать по цене"],
        },
    )

    assert "Рабочая память переписки" in prompt
    assert "P0/бренд/факт-гарды важнее памяти" in prompt
    assert "обсуждали курс по информатике" in prompt
    assert '"subject": "информатика"' in prompt
    assert "сколько стоит онлайн для 10 класса" in prompt
    assert '"grade": "10"' in prompt
    assert "subject, grade" in prompt
    assert "сориентировать по цене" in prompt
    assert prompt.index("Рабочая память переписки") < prompt.index("История диалога:")


def test_build_draft_prompt_without_dialogue_memory_keeps_memory_block_empty() -> None:
    contract = parse_contract(
        {
            "current_question": "цена?",
            "needed_fact_keys": ["price.online"],
            "answerability": "answer_self",
        },
        active_brand="foton",
        fact_key_catalog=("price.online",),
    )
    base_kwargs = {
        "conversation": _conv("цена?"),
        "contract": contract,
        "facts": {"price.online": "Фотон онлайн: семестр — 29 750 ₽."},
        "missing": (),
    }

    prompt_without_memory = build_draft_prompt(**base_kwargs)
    prompt_with_none = build_draft_prompt(**base_kwargs, dialogue_memory_view=None)
    prompt_with_empty = build_draft_prompt(**base_kwargs, dialogue_memory_view={})

    assert prompt_without_memory == prompt_with_none == prompt_with_empty
    assert "Рабочая память переписки" not in prompt_without_memory


def test_build_understanding_prompt_includes_topic_focus_for_ellipsis() -> None:
    prompt = build_understanding_prompt(
        conversation=(
            {"role": "client", "text": "интересует информатика для 10 класса"},
            {"role": "client", "text": "а онлайн?"},
        ),
        active_brand="foton",
        fact_key_catalog=("regular_course.informatics.grade10.online.price",),
        context={
            "dialogue_memory_view": {
                "known_slots": {"subject": "информатика", "grade": "10"},
                "topic_focus": {
                    "subject": "информатика",
                    "grade": "10",
                    "format": "онлайн",
                    "product_family": "regular_course",
                },
            }
        },
    )

    assert "Фокус темы из памяти" in prompt
    assert '"subject": "информатика"' in prompt
    assert "ВОССТАНОВИ тему" in prompt
    assert "product_family" in prompt
    assert "switched_topics" in prompt


def test_memory_topic_augment_recovers_elliptic_online_question_fact() -> None:
    fact_key = "regular_course.informatics.grade10.online.price"
    store = FactStore(
        catalog=(fact_key,),
        store={"foton": {fact_key: "Фотон: информатика 10 класс онлайн — семестр 29 750 ₽."}},
    )
    result = run_pipeline(
        conversation=(
            {"role": "client", "text": "интересует информатика для 10 класса"},
            {"role": "client", "text": "а онлайн?"},
        ),
        active_brand="foton",
        fact_store=store,
        understand_fn=_understanding(
            {
                "current_question": "а онлайн?",
                "needed_fact_keys": [],
                "answerability": "answer_self",
            }
        ),
        draft_fn=lambda _prompt: "Информатика 10 класс онлайн — семестр 29 750 ₽.",
        faithfulness_fn=lambda _prompt: {"unsupported": []},
        context={
            "dialogue_memory_view": {
                "topic_focus": {
                    "subject": "информатика",
                    "grade": "10",
                    "format": "онлайн",
                    "product_family": "regular_course",
                }
            }
        },
    )

    assert result.route == "bot_answer_self"
    assert fact_key in result.facts
    assert "29 750" in result.draft_text


def test_memory_topic_augment_does_not_glue_explicit_subject_switch() -> None:
    informatics_key = "regular_course.informatics.grade10.online.price"
    physics_key = "regular_course.physics.grade10.online.price"
    store = FactStore(
        catalog=(informatics_key, physics_key),
        store={
            "foton": {
                informatics_key: "Фотон: информатика 10 класс онлайн — семестр 29 750 ₽.",
                physics_key: "Фотон: физика 10 класс онлайн — семестр 31 000 ₽.",
            }
        },
    )
    result = run_pipeline(
        conversation=(
            {"role": "client", "text": "интересует информатика для 10 класса"},
            {"role": "client", "text": "а по физике онлайн?"},
        ),
        active_brand="foton",
        fact_store=store,
        understand_fn=_understanding(
            {
                "current_question": "а по физике онлайн?",
                "needed_fact_keys": [physics_key],
                "answerability": "answer_self",
            }
        ),
        draft_fn=lambda _prompt: "Физика 10 класс онлайн — семестр 31 000 ₽.",
        faithfulness_fn=lambda _prompt: {"unsupported": []},
        context={
            "dialogue_memory_view": {
                "topic_focus": {
                    "subject": "информатика",
                    "grade": "10",
                    "format": "онлайн",
                    "product_family": "regular_course",
                }
            }
        },
    )

    assert physics_key in result.facts
    assert informatics_key not in result.facts
    assert "31 000" in result.draft_text


def test_memory_topic_augment_keeps_camp_family_from_regular_course() -> None:
    regular_key = "regular_course.informatics.grade10.online.price"
    camp_key = "lvsh_mendeleevo_2026.online_format"
    store = FactStore(
        catalog=(regular_key, camp_key),
        store={
            "foton": {
                regular_key: "Фотон: регулярный курс информатики 10 класс онлайн — семестр 29 750 ₽.",
                camp_key: "ЛВШ Менделеево: по онлайн-формату смены менеджер сориентирует отдельно.",
            }
        },
    )
    result = run_pipeline(
        conversation=(
            {"role": "client", "text": "интересует ЛВШ по информатике"},
            {"role": "client", "text": "а онлайн?"},
        ),
        active_brand="foton",
        fact_store=store,
        understand_fn=_understanding(
            {
                "current_question": "а онлайн?",
                "needed_fact_keys": [],
                "answerability": "answer_self",
            }
        ),
        draft_fn=lambda _prompt: "По ЛВШ Менделеево онлайн-формат смены менеджер сориентирует отдельно.",
        faithfulness_fn=lambda _prompt: {"unsupported": []},
        context={
            "dialogue_memory_view": {
                "topic_focus": {
                    "subject": "информатика",
                    "product": "ЛВШ Менделеево",
                    "product_family": "camp",
                }
            }
        },
    )

    assert camp_key in result.facts
    assert regular_key not in result.facts
    assert "ЛВШ" in result.draft_text


def test_memory_topic_augment_without_memory_keeps_single_turn_unchanged() -> None:
    fact_key = "regular_course.informatics.grade10.online.price"
    store = FactStore(
        catalog=(fact_key,),
        store={"foton": {fact_key: "Фотон: информатика 10 класс онлайн — семестр 29 750 ₽."}},
    )
    result = run_pipeline(
        conversation=_conv("а онлайн?"),
        active_brand="foton",
        fact_store=store,
        understand_fn=_understanding(
            {
                "current_question": "а онлайн?",
                "needed_fact_keys": [],
                "answerability": "answer_self",
            }
        ),
        draft_fn=lambda _prompt: "Подскажите, пожалуйста, какой предмет и класс интересуют?",
        faithfulness_fn=lambda _prompt: {"unsupported": []},
    )

    assert result.route == "bot_answer_self"
    assert result.facts == {}
    assert "предмет" in result.draft_text.casefold()


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


def test_non_refund_complaint_p0_does_not_use_refund_template() -> None:
    store = FactStore(catalog=(), store={"unpk": {}})
    result = run_pipeline(
        conversation=(
            {"role": "client", "text": "преподаватель ужасно ведёт физику, это возмутительно"},
            {"role": "bot", "text": "Обращение принято. Передам ответственному сотруднику."},
            {"role": "client", "text": "ну хорошо, тогда жду ответа от ответственного"},
        ),
        active_brand="unpk",
        fact_store=store,
        understand_fn=_understanding(
            {
                "current_question": "клиент ждёт ответа по жалобе на преподавателя",
                "continued_topics": ["refund_policy"],
                "client_state": "жалоба на преподавателя, недовольство",
                "answerability": "manager_only",
                "is_p0": True,
                "p0_reason": "complaint",
            }
        ),
        draft_fn=lambda _prompt: "",
    )
    assert result.route == "manager_only"
    assert result.fallback_reason == "p0"
    assert "возврат" not in result.draft_text.casefold()
    assert "отмен" not in result.draft_text.casefold()


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
