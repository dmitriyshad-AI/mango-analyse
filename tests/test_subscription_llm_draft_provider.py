from __future__ import annotations

import json
import re
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Sequence

import pytest
import yaml

import mango_mvp.channels.subscription_llm as subscription_llm
import mango_mvp.channels.subscription_llm_parts.provider as subscription_provider
from mango_mvp.channels.dialogue_contract_pipeline import (
    AnswerContract,
    AUTONOMY_SCOPE_PRECISION_ENV,
    FactStore,
    NUMBER_GATE_SCOPE_AWARE_ENV,
    _safe_fallback_text,
    autonomy_scope_precision_enabled,
    build_faithfulness_prompt,
    check_claim_faithfulness,
    number_gate_scope_aware_enabled,
    run_pipeline,
    verify_output as verify_dialogue_contract_output,
)
from mango_mvp.channels.draft_prompt_builder import build_draft_prompt
from mango_mvp.channels.subscription_llm import (
    ADDRESS_FOTON_MOSCOW_SAFE_TEXT,
    ADDRESS_UNPK_MOSCOW_REGULAR_SAFE_TEXT,
    ADDRESS_UNPK_SAFE_TEXT,
    CodexExecConfig,
    CodexExecDraftProvider,
    COMPLAINT_SAFE_TEXT,
    CONTACT_FOTON_SAFE_TEXT,
    BOT_GOLD_REAL_ENV,
    DIRECT_PATH_ENV,
    DIRECT_PATH_PILOT_CONFIG_ENV,
    DIRECT_PATH_PILOT_CONFIG_VERSION,
    DIRECT_PATH_WIDE_FACT_CHAR_LIMIT,
    DIRECT_PATH_REAL_MANAGER_GOLD_PACK_PATH,
    DIALOGUE_CONTRACT_V2_TEMPLATE_REGISTRY,
    DraftGenerationResult,
    FakeDraftProvider,
    IDENTITY_FOTON_SAFE_TEXT,
    LEGAL_THREAT_SAFE_TEXT,
    LLM_RETRIEVE_ENV,
    OUTPUT_SANITIZER_ENV,
    PRESALE_SAFETY_ENV,
    PAYMENT_DISPUTE_SAFE_TEXT,
    PRESALE_META_RU_ENV,
    PRESALE_PII_MEMORY_ENV,
    PRESALE_SOURCE_ID_ENV,
    PRESALE_VERIFIER_FAILSOFT_ENV,
    PROSE_MODEL_LED_ENV,
    ASSUMED_SCOPE_GUARD_ENV,
    RETRIEVER_MODEL_DRIVEN_ENV,
    RETRIEVER_NEED_SHADOW_ENV,
    REFUND_ZERO_COLLECT_SAFE_TEXT,
    SAFE_FALLBACK_DRAFT_TEXT,
    RESULT_GUARANTEE_SAFE_TEXT,
    SubscriptionDraftResult,
    SubscriptionLlmDraftProvider,
    TEMPLATE_FROM_KB_ENV,
    TONE_CLOSE_DETECT_ENV,
    TONE_RICH_FORMAT_ENV,
    TONE_SELL_PROMPT_ENV,
    TONE_WARM_FRAME_ENV,
    UNPK_INSTALLMENT_APPROVED_FALLBACK_TEXT,
    VERIFIER_HANDOFF_CLAIMS_ENV,
    apply_a2_proactive_layer,
    apply_authoritative_output_gate,
    apply_brand_separation_guard,
    apply_conversation_intent_plan_guard,
    apply_humanity_guards,
    apply_humanity_x2_rewriter,
    apply_phase2_tone_layer,
    apply_tone_close_detect_layer,
    apply_tone_sell_prompt_observer,
    apply_warm_frame,
    apply_semantic_output_verifier,
    apply_prose_model_led_quality_guard,
    apply_semantic_diagnosis_guard,
    _direct_path_context_fact_pack,
    _direct_path_render_fact_block,
    _direct_path_gold_real_enabled,
    build_semantic_output_regen_prompt,
    build_semantic_output_verifier_prompt,
    build_semantic_diagnosis_prompt,
    SEMANTIC_OUTPUT_VERIFIER_ENV,
    SEMANTIC_VERIFIER_DOWNGRADE_REASON,
    _output_sanitizer_enabled,
    _presale_safety_enabled,
    _semantic_output_verifier_enabled,
    _verifier_handoff_claims_enabled,
    apply_unstated_subject_guard,
    apply_unsupported_promise_guard,
    apply_unconfirmed_operational_specificity_guard,
    _claim_supported_by_facts,
    _context_with_selling_thread_slots,
    _fresh_fact_texts,
    _keep_answer_supported,
    _p0_text_with_antirepeat,
    _validated_guardchain_recovery_candidate,
    _verified_informational_answer,
    contains_bot_identity_disclosure,
    decide_route,
    draft_has_internal_service_markers,
    detect_high_risk_input_markers,
    find_unsupported_numeric_promises,
    find_unsupported_followup_deadline_claims,
    find_redundant_questions_for_known_context,
    build_codex_exec_env,
    _normalize_direct_path_payload,
    _build_direct_path_prompt,
    parse_llm_json,
    strip_internal_service_markers,
    known_context_fields,
)
from mango_mvp.channels.subscription_llm import apply_high_risk_content_guards
from mango_mvp.channels.dialogue_memory import build_dialogue_memory, update_dialogue_memory_after_answer


def _trace_rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_tz5_client_safe_literals_do_not_regress_process_decisions() -> None:
    checked = "\n".join(
        (
            subscription_llm.PROMOCODE_SAFE_TEXT,
            subscription_llm.UNPK_LVSH_SEATS_SAFE_TEXT,
            subscription_llm.FOTON_LVSH_PRICE_SAFE_TEXT,
            subscription_llm.UNPK_LVSH_PRICE_SAFE_TEXT,
            subscription_llm.UNPK_LVSH_PRICE_DETAILS_SAFE_TEXT,
            subscription_llm.FOTON_LVSH_DATES_SAFE_TEXT,
            subscription_llm.UNPK_LVSH_DATES_SAFE_TEXT,
            subscription_llm.CONTRACT_ENTITY_SAFE_TEXT,
            subscription_llm.CROSS_BRAND_GENERIC_SAFE_TEXT,
            subscription_llm.CROSS_BRAND_PLATFORM_SAFE_TEXT,
            subscription_llm.FOTON_ONLINE_TRIAL_SAFE_TEXT,
            subscription_llm.UNPK_TRIAL_SAFE_TEXT,
            subscription_llm.UNPK_CAMP_OVERVIEW_SAFE_TEXT,
        )
    )

    for forbidden in (
        "почти распрод",
        "живой менеджер",
        "живой сотрудник",
        "МТС Линк",
        "Webinar",
        "акции и промокоды",
        "подскажет актуальные акции",
        "по нашей программе и наших условиях",
        "Если клиент сам попросит",
        "онлайн-смена",
    ):
        assert forbidden.casefold() not in checked.casefold()
    assert "Промокодов сейчас нет" in subscription_llm.PROMOCODE_SAFE_TEXT
    assert "учтено в прайсе" in subscription_llm.PROMOCODE_SAFE_TEXT
    assert "SohoLMS" in subscription_llm.CROSS_BRAND_PLATFORM_SAFE_TEXT
    assert "договор-оферта" in subscription_llm.CONTRACT_ENTITY_SAFE_TEXT
    assert "93 100 ₽" in subscription_llm.FOTON_LVSH_PRICE_SAFE_TEXT
    assert "98 000 ₽" in subscription_llm.FOTON_LVSH_PRICE_SAFE_TEXT
    assert "114 000 ₽" in subscription_llm.UNPK_LVSH_PRICE_SAFE_TEXT
    assert "120 000 ₽" in subscription_llm.UNPK_LVSH_PRICE_DETAILS_SAFE_TEXT
    assert "20-28 июня" in subscription_llm.FOTON_LVSH_DATES_SAFE_TEXT
    assert "18-26 июля" in subscription_llm.UNPK_LVSH_DATES_SAFE_TEXT


def test_codex_exec_provider_builds_command_without_openai_key(tmp_path: Path) -> None:
    command = CodexExecConfig(model="gpt-5.5", reasoning_effort="medium").build_command(tmp_path / "out.txt")

    assert "OPENAI_API_KEY" not in " ".join(command)
    assert command[0] == "codex"
    assert command[command.index("--ask-for-approval") + 1] == "never"
    assert "exec" in command
    assert "--sandbox" in command
    assert "read-only" in command


def test_codex_exec_isolated_command_ignores_user_config_and_uses_clean_cwd(tmp_path: Path) -> None:
    command = CodexExecConfig(
        model="gpt-5.5",
        reasoning_effort="medium",
        isolated=True,
        cwd=tmp_path,
    ).build_command(tmp_path / "out.txt")

    assert "--ignore-user-config" in command
    assert "--ignore-rules" in command
    assert command[command.index("--ask-for-approval") + 1] == "never"
    assert "--ephemeral" in command
    assert "--skip-git-repo-check" in command
    assert command[command.index("-C") + 1] == str(tmp_path)
    assert "personality" not in " ".join(command)


def test_codex_exec_env_allowlist_preserves_auth_and_drops_secrets() -> None:
    env = build_codex_exec_env(
        {
            "CODEX_HOME": "/tmp/codex-home",
            "PATH": "/bin",
            "HOME": "/home/test",
            "USER": "bot",
            "LOGNAME": "bot",
            "PYTHONPATH": "src",
            "LANG": "ru_RU.UTF-8",
            "LC_TIME": "ru_RU.UTF-8",
            "MANGO_CODEX_SERVICE_TIER": "flex",
            "AMO_TOKEN": "amo",
            "WAPPI_SECRET": "wappi",
            "OPENAI_API_KEY": "openai",
            "CRM_AMO_API_TOKEN": "crm",
            "AI_OFFICE_API_KEY": "office",
            "CUSTOM_TOKEN": "custom-token",
            "SAFE_EXTRA": "ok",
            "TASK_CONTAINER_ENV_PASSTHROUGH": "SAFE_EXTRA,CUSTOM_SECRET",
            "CUSTOM_SECRET": "custom-secret",
            "UNRELATED": "drop-me",
        }
    )

    assert env["CODEX_HOME"] == "/tmp/codex-home"
    assert env["PATH"] == "/bin"
    assert env["HOME"] == "/home/test"
    assert env["USER"] == "bot"
    assert env["LOGNAME"] == "bot"
    assert env["PYTHONPATH"] == "src"
    assert env["LANG"] == "ru_RU.UTF-8"
    assert env["LC_TIME"] == "ru_RU.UTF-8"
    assert env["MANGO_CODEX_SERVICE_TIER"] == "flex"
    assert env["SAFE_EXTRA"] == "ok"
    assert "UNRELATED" not in env
    assert "TASK_CONTAINER_ENV_PASSTHROUGH" not in env
    assert "OPENAI_API_KEY" not in env
    assert "AMO_TOKEN" not in env
    assert "WAPPI_SECRET" not in env
    assert "CRM_AMO_API_TOKEN" not in env
    assert "AI_OFFICE_API_KEY" not in env
    assert "CUSTOM_TOKEN" not in env
    assert "CUSTOM_SECRET" not in env


def test_codex_exec_env_explicit_codex_home_overrides_base_env() -> None:
    env = build_codex_exec_env(
        {"CODEX_HOME": "/tmp/base-codex-home", "PATH": "/bin"},
        codex_home="/tmp/runtime-codex-home",
    )

    assert env["CODEX_HOME"] == str(Path("/tmp/runtime-codex-home").resolve())


def test_codex_exec_provider_isolated_bot_run_uses_clean_cwd_and_metadata(tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def runner(cmd, **kwargs):
        seen["cmd"] = list(cmd)
        seen["env"] = dict(kwargs["env"])
        cwd = Path(cmd[cmd.index("-C") + 1])
        assert cwd.exists()
        assert not (cwd / "AGENTS.md").exists()
        output_path = Path(cmd[cmd.index("--output-last-message") + 1])
        output_path.write_text(
            json.dumps(
                {
                    "route": "bot_answer_self_for_pilot",
                    "draft_text": "Да, сориентирую по проверенным условиям.",
                    "message_type": "question",
                    "topic_id": "service:S5_general_consultation",
                    "confidence_theme": 0.9,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0, "", "")

    provider = CodexExecDraftProvider(
        runner=runner,
        cache_dir=None,
        codex_isolated=True,
        base_env={"CODEX_HOME": str(tmp_path / "codex-home"), "OPENAI_API_KEY": "secret", "PATH": "/bin"},
    )

    result = provider.generate_from_prompt("Верни JSON")

    assert "--ignore-user-config" in seen["cmd"]
    assert "--ignore-rules" in seen["cmd"]
    assert "-C" in seen["cmd"]
    assert seen["env"]["CODEX_HOME"].endswith("codex-home")
    assert "OPENAI_API_KEY" not in seen["env"]
    assert result.metadata["codex_exec"] == {
        "isolated": True,
        "ignore_user_config": True,
        "ignore_rules": True,
    }


def test_codex_exec_provider_isolates_dialogue_contract_subcalls(tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def runner(cmd, **kwargs):
        seen["cmd"] = list(cmd)
        seen["env"] = dict(kwargs["env"])
        cwd = Path(cmd[cmd.index("-C") + 1])
        assert cwd.exists()
        assert not (cwd / "AGENTS.md").exists()
        output_path = Path(cmd[cmd.index("--output-last-message") + 1])
        output_path.write_text('{"ok": true}', encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    provider = CodexExecDraftProvider(
        runner=runner,
        cache_dir=None,
        codex_isolated=True,
        base_env={"CODEX_HOME": str(tmp_path / "codex-home"), "OPENAI_API_KEY": "secret", "PATH": "/bin"},
    )

    assert provider._run_prompt_text("Верни JSON", prefix="mango_test_", suffix=".json") == '{"ok": true}'
    assert "--ignore-user-config" in seen["cmd"]
    assert "--ignore-rules" in seen["cmd"]
    assert "-C" in seen["cmd"]
    assert seen["env"]["CODEX_HOME"].endswith("codex-home")
    assert "OPENAI_API_KEY" not in seen["env"]


def test_provider_parses_valid_json() -> None:
    result = parse_llm_json(
        '{"route":"draft_for_manager","draft_text":"Здравствуйте! Уточним детали.",'
        '"message_type":"question","broad_group":"commercial","topic_id":"theme:001_pricing",'
        '"confidence_theme":0.8,"confidence_group":0.9,"alternative_themes":["theme:002_payment_method"],'
        '"risk_level":"low","context_used":["recent_messages"],"context_warnings":[]}'
    )

    assert result.route in {"bot_answer_self_for_pilot", "draft_for_manager"}
    assert result.topic_id == "theme:001_pricing"
    assert result.message_type == "question"
    assert result.broad_group == "commercial"
    assert result.topic_confidence == 0.8
    assert result.confidence_group == 0.9
    assert result.alternative_themes == ("theme:002_payment_method",)
    assert result.to_json_dict()["confidence_theme"] == 0.8


def test_direct_path_missing_route_default_off_keeps_existing_self_route(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TELEGRAM_DIRECT_DEFAULT_MANAGER", raising=False)

    result = _normalize_direct_path_payload(
        {
            "draft_text": "Здравствуйте! Подскажу по проверенным условиям.",
            "message_type": "question",
            "topic_id": "service:S5_general_consultation",
        }
    )

    assert result.route == "bot_answer_self_for_pilot"


def test_direct_path_missing_route_flag_on_defaults_to_manager_draft(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TELEGRAM_DIRECT_DEFAULT_MANAGER", "1")

    result = _normalize_direct_path_payload(
        {
            "draft_text": "Здравствуйте! Подскажу по проверенным условиям.",
            "message_type": "question",
            "topic_id": "service:S5_general_consultation",
        }
    )

    assert result.route == "draft_for_manager"


def test_direct_path_blank_route_flag_on_defaults_to_manager_draft(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TELEGRAM_DIRECT_DEFAULT_MANAGER", "1")

    result = _normalize_direct_path_payload(
        {
            "route": "   ",
            "draft_text": "Здравствуйте! Подскажу по проверенным условиям.",
            "message_type": "question",
            "topic_id": "service:S5_general_consultation",
        }
    )

    assert result.route == "draft_for_manager"


def test_direct_path_explicit_route_is_not_overridden_by_default_manager_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TELEGRAM_DIRECT_DEFAULT_MANAGER", "1")

    result = _normalize_direct_path_payload(
        {
            "route": "bot_answer_self_for_pilot",
            "draft_text": "Здравствуйте! Подскажу по проверенным условиям.",
            "message_type": "question",
            "topic_id": "service:S5_general_consultation",
        }
    )

    assert result.route == "bot_answer_self_for_pilot"


def test_answerability_shadow_payload_off_does_not_store_self_eval() -> None:
    result = _normalize_direct_path_payload(
        {
            "route": "bot_answer_self_for_pilot",
            "draft_text": "Да, можно записаться онлайн.",
            "can_answer_self": "no",
            "self_missing_facts": ["актуальная группа"],
            "supporting_facts": ["fact:trial_online"],
            "why_manager": "Нужна проверка группы.",
        }
    )

    assert result.route == "bot_answer_self_for_pilot"
    assert result.draft_text == "Да, можно записаться онлайн."
    assert "answerability_self" not in result.metadata


def test_answerability_shadow_payload_on_is_observe_only() -> None:
    result = _normalize_direct_path_payload(
        {
            "route": "bot_answer_self_for_pilot",
            "draft_text": "Да, можно записаться онлайн.",
            "can_answer_self": "no",
            "self_missing_facts": ["актуальная группа"],
            "supporting_facts": ["fact:trial_online"],
            "why_manager": "Нужна проверка группы.",
        },
        include_answerability_self=True,
    )

    assert result.route == "bot_answer_self_for_pilot"
    assert result.draft_text == "Да, можно записаться онлайн."
    assert result.metadata["answerability_self"] == {
        "schema_version": "answerability_self_v1_2026_06_15",
        "can_answer_self": "no",
        "self_missing_facts": ["актуальная группа"],
        "supporting_facts": ["fact:trial_online"],
        "why_manager": "Нужна проверка группы.",
    }


def test_answerability_shadow_payload_on_tolerates_missing_keys() -> None:
    result = _normalize_direct_path_payload(
        {
            "route": "bot_answer_self_for_pilot",
            "draft_text": "Да, можно записаться онлайн.",
        },
        include_answerability_self=True,
    )

    assert result.route == "bot_answer_self_for_pilot"
    assert result.metadata["answerability_self"] == {}


def test_provider_strips_internal_manager_note_and_keeps_safe_variant() -> None:
    result = parse_llm_json(
        '{"route":"draft_for_manager","draft_text":"Клиент понял условия и взял паузу. '
        'Автономный ответ не требуется. Если менеджер решит ответить, безопасный вариант: '
        '«Конечно, подумайте спокойно. Если захотите, помогу сравнить варианты.»",'
        '"message_type":"context_update","topic_id":"service:S5_general_consultation","confidence_theme":0.8}'
    )

    assert "Автономный ответ не требуется" not in result.draft_text
    assert "Если менеджер решит" not in result.draft_text
    assert result.draft_text == "Конечно, подумайте спокойно. Если захотите, помогу сравнить варианты."
    assert "internal_metadata_removed_from_draft" in result.safety_flags


def test_provider_blocks_internal_manager_note_without_safe_variant() -> None:
    result = parse_llm_json(
        '{"route":"draft_for_manager","draft_text":"Клиент понял условия. Автономный ответ не требуется.",'
        '"message_type":"context_update","topic_id":"service:S5_general_consultation","confidence_theme":0.8}'
    )

    assert "Автономный ответ не требуется" not in result.draft_text
    assert result.draft_text == SAFE_FALLBACK_DRAFT_TEXT
    assert "internal_metadata_removed_from_draft" in result.safety_flags


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
            DIRECT_PATH_PILOT_CONFIG_ENV: DIRECT_PATH_PILOT_CONFIG_VERSION,
            "conversation_intent_plan": {
                "primary_intent": "live_availability",
                "topic_id": "theme:026_camp_general",
                "answer_policy": "answer_safe_parts_then_manager_live_check",
                "route_bias": "draft_for_manager",
                "product_family": "camp",
            },
        },
    )

    assert guarded.route == "draft_for_manager"
    assert "conversation_intent_plan_live_availability" in guarded.safety_flags
    assert "semantic_frame_intent_actions_live_availability" in guarded.safety_flags


def test_followup_deadline_guard_catches_absolute_datetime_with_vernutsya() -> None:
    claims = find_unsupported_followup_deadline_claims(
        "Менеджер должен вернуться с конкретикой до 25 мая 2026, 14:46 по Москве.",
        context={},
    )

    assert claims
    assert "25 мая" in claims[0]


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


def test_conversation_intent_plan_guard_does_not_turn_presale_refund_policy_into_p0() -> None:
    result = SubscriptionDraftResult(
        route="manager_only",
        topic_id="theme:009_refund",
        topic_confidence=0.84,
        draft_text="Приняли обращение. Передам ответственному сотруднику.",
        safety_flags=("high_risk_manager_only",),
    )

    guarded = apply_conversation_intent_plan_guard(
        result,
        client_message="До оплаты хочу понять: если ребёнку не понравится, деньги вернёте?",
        context={
            "active_brand": "foton",
            "conversation_intent_plan": {
                "primary_intent": "refund",
                "topic_id": "theme:009_refund",
                "refund_frame": "presale_policy",
                "answer_policy": "answer_directly_if_fact_verified",
                "route_bias": "draft_for_manager",
                "risk_signals": [],
            },
        },
    )

    assert guarded.route != "manager_only"
    assert "conversation_intent_plan_p0" not in guarded.safety_flags


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
    assert result.draft_text.startswith("Поняла, давайте не буду повторять общий ответ")


def test_tax_followup_with_manager_word_does_not_turn_into_presale_refund() -> None:
    base = SubscriptionDraftResult(
        route="draft_for_manager",
        topic_id="service:S5_general_consultation",
        topic_confidence=0.8,
        draft_text="Да, менеджер пришлёт шаблон заявления.",
        message_type="context_update",
    )

    guarded = apply_high_risk_content_guards(
        base,
        client_message="Поняла, тогда заявление у менеджера попрошу",
        context={"active_brand": "unpk", "recent_messages": ["За обучение ребёнка можно вернуть до 14 300 ₽ в год."]},
    )

    assert "presale_refund_policy_manager_check" not in guarded.safety_flags
    assert "условия возврата" not in guarded.draft_text.casefold()


def test_unpk_bank_installment_phrase_is_not_cross_brand_leak() -> None:
    base = SubscriptionDraftResult(
        route="draft_for_manager",
        topic_id="theme:006_installment",
        topic_confidence=0.9,
        draft_text="В УНПК нет рассрочки через банк, можно платить помесячно.",
        message_type="question",
    )

    guarded = apply_brand_separation_guard(
        base,
        client_message="У вас есть рассрочка через банк?",
        context={"active_brand": "unpk", "conversation_intent_plan": {"primary_intent": "installment"}},
    )

    assert "cross_brand_client_text_blocked" not in guarded.safety_flags
    assert "рассрочки через банк" in guarded.draft_text


def test_internal_manager_note_is_removed_from_client_text() -> None:
    text = "Клиент подтвердил ожидание ответа менеджера по очному пробному. Дополнительный ответ клиенту сейчас не нужен."

    assert strip_internal_service_markers(text) == ""
    assert draft_has_internal_service_markers(text)


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
    assert strip_internal_service_markers("Без служебных пометок: ответ клиенту") == ""
    assert strip_internal_service_markers("Ответ fact_id:abc trace_id=run-1 source_id=fact:v3:price") == "Ответ"
    assert "product_data" not in strip_internal_service_markers("Ответ source_id=fact:v3:price product_data/knowledge_base/kb_release_20260520_v6_3")
    assert "/Users/" not in strip_internal_service_markers("Ответ /Users/dmitrijfabarisov/Projects/Mango")
    assert "kc_chunk:" not in strip_internal_service_markers("Ответ kc_chunk:safe_template")


def test_scaffold_prefixes_are_stripped_and_client_instructions_are_blocked() -> None:
    assert (
        strip_internal_service_markers('Фотон: черновик для ситуации «возражение о стоимости курса»: Это отдельные организации.')
        == "Это отдельные организации."
    )
    assert strip_internal_service_markers("без обещаний оценки: Контрольные помогают увидеть динамику.") == "Контрольные помогают увидеть динамику."
    assert strip_internal_service_markers("без давления на клиента: Можно спокойно сравнить варианты.") == "Можно спокойно сравнить варианты."
    assert (
        strip_internal_service_markers(
            "Текст. По вашей ситуации лучше опираться на подтверждённые условия, без обещаний оценки: Факт."
        )
        == "Текст. Факт."
    )
    assert strip_internal_service_markers("Ориентир без обещаний результата: Факт.") == "Ориентир Факт."
    assert strip_internal_service_markers("Заменяю только этот абзац: Да, домашние задания проверяются.") == "Да, домашние задания проверяются."
    assert strip_internal_service_markers("Остальной текст без изменений. Да, расписание уточняется по группе.") == "Да, расписание уточняется по группе."
    assert (
        strip_internal_service_markers("Ответ клиенту. Лимиты Codex: осталось 12 сообщений в сессии.")
        == "Ответ клиенту."
    )
    assert strip_internal_service_markers("Осталось 500 токенов контекста, поэтому отвечаю кратко. Да, пробное есть.") == "Да, пробное есть."

    normal = "Если удобно, повторите класс и предмет — я сориентирую по подходящему варианту."
    assert strip_internal_service_markers(normal) == normal
    assert not draft_has_internal_service_markers(normal)
    normal_conditions = "По условиям курса можно опираться на подтверждённые факты из договора."
    assert strip_internal_service_markers(normal_conditions) == normal_conditions
    assert not draft_has_internal_service_markers(normal_conditions)
    normal_refund = "При отказе возвращается остаток неистраченных средств."
    assert strip_internal_service_markers(normal_refund) == normal_refund
    assert not draft_has_internal_service_markers(normal_refund)

    leaked = "Менеджер подтвердит порядок записи. Если класс, предмет и формат уже есть в диалоге, повторять их не нужно."
    assert strip_internal_service_markers(leaked) == ""
    assert draft_has_internal_service_markers(leaked)
    result = SubscriptionDraftResult(route="bot_answer_self_for_pilot", draft_text=leaked, topic_id="theme:020_enrollment")
    assert result.draft_text == SAFE_FALLBACK_DRAFT_TEXT
    assert "internal_metadata_removed_from_draft" in result.safety_flags


def test_humanity_trims_repeated_cosmetic_opening_when_safe_fact_exists() -> None:
    result = SubscriptionDraftResult(
        route="draft_for_manager",
        topic_id="theme:015_address",
        message_type="question",
        draft_text="Здравствуйте! В Москве Фотон находится на Верхней Красносельской, 30.",
    )
    context = {
        "active_brand": "foton",
        "confirmed_facts": {"address": "В Москве Фотон находится на Верхней Красносельской, 30."},
        "dialogue_memory_view": {"recent_turns": [{"role": "bot", "text": "Здравствуйте! Подскажу по адресу Фотона."}]},
    }

    fixed = apply_humanity_guards(result, client_message="Где вы в Москве?", context=context)

    assert fixed.draft_text.startswith("В Москве Фотон")
    assert "humanity_cosmetic_opening_trimmed" in fixed.safety_flags


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


def test_payment_dispute_client_message_gets_final_p0_manager_only() -> None:
    base = parse_llm_json(
        '{"route":"bot_answer_self_for_pilot","draft_text":"Оплату можно проверить по личному кабинету.",'
        '"message_type":"question","topic_id":"theme:003_payment_status","confidence_theme":0.91}'
    )

    result = apply_high_risk_content_guards(
        base,
        client_message="Оплатила вчера, но оплату не видно.",
        context={"active_brand": "foton"},
    )

    assert result.route == "manager_only"
    assert result.topic_id == "theme:003_payment_status"
    assert result.draft_text == PAYMENT_DISPUTE_SAFE_TEXT
    assert "final_p0_text_override" in result.safety_flags
    assert "payment_dispute_manager_only" in result.safety_flags


def test_double_charge_refund_mid_dialog_gets_final_p0_manager_only() -> None:
    base = parse_llm_json(
        '{"route":"bot_answer_self_for_pilot","draft_text":"Сейчас помогу подобрать курс и стоимость.",'
        '"message_type":"question","topic_id":"theme:001_pricing","confidence_theme":0.91}'
    )

    result = apply_high_risk_content_guards(
        base,
        client_message="С меня дважды списали деньги за оплату, верните одну.",
        context={
            "active_brand": "foton",
            "conversation_intent_plan": {
                "primary_intent": "payment_dispute",
                "topic_id": "theme:003_payment_status",
                "route_bias": "manager_only",
                "risk_signals": ["payment_dispute"],
            },
        },
    )

    assert result.route == "manager_only"
    assert result.topic_id in {"theme:003_payment_status", "theme:009_refund"}
    assert "подобрать курс" not in result.draft_text
    assert "final_p0_text_override" in result.safety_flags
    assert "high_risk_manager_only" in result.safety_flags


def test_p0_latch_keeps_safe_followup_manager_only_after_dispute() -> None:
    base = parse_llm_json(
        '{"route":"bot_answer_self_for_pilot","draft_text":"Годовая цена зависит от класса.",'
        '"message_type":"question","topic_id":"theme:001_pricing","confidence_theme":0.91}'
    )

    result = apply_high_risk_content_guards(
        base,
        client_message="А теперь скажите цену.",
        context={
            "active_brand": "foton",
            "conversation_intent_plan": {
                "primary_intent": "pricing",
                "risk_signals": [],
                "route_bias": "bot_answer_self_for_pilot",
            },
            "dialogue_memory_view": {
                "p0_latch": {
                    "active": True,
                    "codes": ["payment_dispute"],
                    "primary_risk": "payment_dispute",
                }
            },
        },
    )

    assert result.route == "manager_only"
    assert result.topic_id == "theme:003_payment_status"
    assert result.draft_text == PAYMENT_DISPUTE_SAFE_TEXT
    assert "final_p0_text_override" in result.safety_flags


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


def test_answer_contract_can_skip_terminal_green_contact_template() -> None:
    base = parse_llm_json(
        '{"route":"bot_answer_self_for_pilot","draft_text":"Позвонить нам можно по телефону центра, менеджер подскажет детали.",'
        '"message_type":"question","topic_id":"service:S5_general_consultation","confidence_theme":0.91}'
    )

    result = apply_high_risk_content_guards(
        base,
        client_message="Дайте телефон, пожалуйста.",
        context={
            "active_brand": "foton",
            "answer_contract": {
                "primary_intent": "general_consultation",
                "direct_question": "Дайте телефон, пожалуйста.",
                "must_answer_first": True,
                "p0_required": False,
            },
        },
    )

    assert result.draft_text != CONTACT_FOTON_SAFE_TEXT
    assert "terminal_safe_template_applied" not in result.safety_flags
    assert result.metadata["terminal_green_template_skipped_by_answer_contract"] is True


def test_known_context_does_not_infer_programming_from_program_word() -> None:
    known = known_context_fields(
        {
            "active_brand": "foton",
            "known_context_summary": "Клиент: 8 класс информатика очно, без подбора программы.",
        }
    )

    assert known["subject"] == "информатика"
    assert "программирование" not in known["subject"]


def test_answer_contract_can_skip_missing_fact_template_for_safe_schedule() -> None:
    base = parse_llm_json(
        '{"route":"bot_answer_self_for_pilot","draft_text":"Расписание зависит от группы.",'
        '"message_type":"question","topic_id":"theme:013_schedule","confidence_theme":0.91}'
    )

    result = apply_high_risk_content_guards(
        base,
        client_message="Во сколько проходят занятия по физике?",
        context={
            "active_brand": "foton",
            "facts_context": {"client_safe": True, "fresh": False, "facts_missing": True},
            "answer_contract": {
                "primary_intent": "schedule",
                "direct_question": "Во сколько проходят занятия по физике?",
                "must_answer_first": True,
                "p0_required": False,
            },
        },
    )

    assert "missing_fact_helpful_template_applied" not in result.safety_flags
    assert "Напишите, пожалуйста, класс ребёнка" not in result.draft_text


def test_fact_scope_guard_blocks_office_hours_as_class_schedule_answer() -> None:
    base = parse_llm_json(
        '{"route":"bot_answer_self_for_pilot","draft_text":"График: Пн-Вс с 10:00 до 18:00.",'
        '"message_type":"question","topic_id":"theme:013_schedule","confidence_theme":0.91}'
    )

    result = apply_high_risk_content_guards(
        base,
        client_message="По каким дням проходят занятия по физике?",
        context={
            "active_brand": "foton",
            "conversation_intent_plan": {
                "primary_intent": "schedule",
                "topic_id": "theme:013_schedule",
                "fact_scope": "class_schedule",
                "blocked_neighbor_scopes": ["office_hours"],
            },
        },
    )

    assert "расписание занятий" in result.draft_text
    assert "10:00" not in result.draft_text
    assert "fact_scope_guard_applied" in result.safety_flags


def test_fact_scope_guard_blocks_tax_answer_for_matkap_question() -> None:
    base = parse_llm_json(
        '{"route":"bot_answer_self_for_pilot","draft_text":"Налоговый вычет оформляется через ФНС, справка готовится до 10 дней.",'
        '"message_type":"question","topic_id":"theme:007_matkap_payment","confidence_theme":0.91}'
    )

    result = apply_high_risk_content_guards(
        base,
        client_message="Маткапиталом можно оплатить? Какие документы и сколько СФР смотрит?",
        context={
            "active_brand": "unpk",
            "conversation_intent_plan": {
                "primary_intent": "matkap",
                "topic_id": "theme:007_matkap_payment",
                "fact_scope": "matkap_process",
                "blocked_neighbor_scopes": ["tax_deduction"],
            },
        },
    )

    assert "налоговый" not in result.draft_text.casefold()
    assert "ФНС" not in result.draft_text
    assert any(marker in result.draft_text.casefold() for marker in ("маткапитал", "материнским капитал"))
    assert any(flag in result.safety_flags for flag in ("fact_scope_guard_applied", "matkap_safe_template_applied"))


def test_scope_fact_guard_blocks_neighbor_discount_when_schedule_fact_missing() -> None:
    base = parse_llm_json(
        '{"route":"bot_answer_self_for_pilot","draft_text":"При оплате за семестр скидка 10%, за год — 14%.",'
        '"message_type":"question","topic_id":"theme:014_format","confidence_theme":0.91,'
        '"missing_facts":["schedule.current"]}'
    )

    result = apply_high_risk_content_guards(
        base,
        client_message="По каким дням проходят занятия на Сретенке?",
        context={
            "active_brand": "unpk",
            "scope_fact_guard_enabled": True,
            "conversation_intent_plan": {
                "primary_intent": "schedule",
                "topic_id": "theme:013_schedule",
                "fact_scope": "class_schedule",
                "blocked_neighbor_scopes": ["discount_second_subject", "discount_multichild", "discount_stacking"],
                "required_fact_keys": ["schedule.current"],
            },
            "facts_context": {"facts_missing": True, "required_fact_keys": ["schedule.current"]},
        },
    )

    assert "10%" not in result.draft_text
    assert "14%" not in result.draft_text
    assert "дни и время занятий" in result.draft_text
    assert "scope_fact_guard_applied" in result.safety_flags


def test_scope_fact_guard_blocks_matkap_age_when_documents_fact_missing() -> None:
    base = parse_llm_json(
        '{"route":"bot_answer_self_for_pilot","draft_text":"Возрастной лимит — до 25 лет.",'
        '"message_type":"question","topic_id":"service:S5_general_consultation","confidence_theme":0.91,'
        '"missing_facts":["matkap_documents.current"]}'
    )

    result = apply_high_risk_content_guards(
        base,
        client_message="Какие документы нужны?",
        context={
            "active_brand": "foton",
            "scope_fact_guard_enabled": True,
            "conversation_intent_plan": {
                "primary_intent": "matkap",
                "topic_id": "theme:007_matkap_payment",
                "fact_scope": "matkap_process",
                "blocked_neighbor_scopes": ["matkap_age_limit", "tax_deduction"],
                "required_fact_keys": ["matkap_documents.current"],
            },
            "facts_context": {"facts_missing": True, "required_fact_keys": ["matkap_documents.current"]},
        },
    )

    assert "25 лет" not in result.draft_text
    assert "документы и порядок оформления маткапитала" in result.draft_text
    assert "scope_fact_guard_applied" in result.safety_flags


def test_scope_fact_guard_blocks_office_hours_when_refund_policy_fact_missing() -> None:
    base = parse_llm_json(
        '{"route":"bot_answer_self_for_pilot","draft_text":"Контакты менеджера и расписание офиса: Пн-Вс 10:00-18:00.",'
        '"message_type":"question","topic_id":"theme:013_schedule","confidence_theme":0.86,'
        '"missing_facts":["refund_policy.current"]}'
    )

    result = apply_high_risk_content_guards(
        base,
        client_message="А это оформляется по заявлению?",
        context={
            "active_brand": "foton",
            "scope_fact_guard_enabled": True,
            "conversation_intent_plan": {
                "primary_intent": "refund_policy",
                "topic_id": "theme:009_refund",
                "fact_scope": "refund_policy",
                "blocked_neighbor_scopes": ["office_hours", "class_schedule"],
                "required_fact_keys": ["refund_policy.current"],
            },
            "facts_context": {
                "facts_missing": True,
                "required_fact_keys": ["refund_policy.current"],
                "missing_facts": ["refund_policy.current"],
                "fact_scope": "refund_policy",
                "blocked_neighbor_scopes": ["office_hours", "class_schedule"],
            },
        },
    )

    text = result.draft_text.casefold()
    assert result.route == "draft_for_manager"
    assert "scope_fact_guard_applied" in result.safety_flags
    assert "пн-вс" not in text
    assert "10:00" not in text
    assert "порядок возврата" in text


def test_forbidden_pair_guard_blocks_matkap_installment_mix() -> None:
    base = parse_llm_json(
        '{"route":"bot_answer_self_for_pilot","draft_text":"Маткапиталом можно оплатить, а ещё можно оформить рассрочку или Долями.",'
        '"message_type":"question","topic_id":"theme:007_matkap_payment","confidence_theme":0.91}'
    )

    result = apply_high_risk_content_guards(
        base,
        client_message="Можно маткапиталом и сразу в рассрочку?",
        context={
            "active_brand": "foton",
            "conversation_intent_plan": {
                "primary_intent": "matkap",
                "topic_id": "theme:007_matkap_payment",
                "answer_topics": ["matkap", "installment"],
                "forbidden_pairs": ["matkap+installment"],
                "template_allowed": False,
            },
        },
    )

    assert "рассроч" not in result.draft_text.casefold()
    assert "долями" not in result.draft_text.casefold()
    assert "маткапитал" in result.draft_text.casefold()
    assert "forbidden_pair_guard_applied" in result.safety_flags


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


def test_find_redundant_questions_ignores_unknown_fields() -> None:
    repeated = find_redundant_questions_for_known_context(
        "Напишите класс ребёнка и предмет.",
        context={"active_brand": "foton"},
    )

    assert repeated == ()


def test_trial_fragment_answer_does_not_promise_bot_will_send_link() -> None:
    result = apply_unconfirmed_operational_specificity_guard(
        SubscriptionDraftResult(
            route="bot_answer_self_for_pilot",
            topic_id="theme:023_trial_class",
            topic_confidence=0.9,
            draft_text="Да, фрагмент занятия можно посмотреть. Пришлю фрагмент для знакомства.",
        ),
        context={"active_brand": "unpk"},
    )

    assert result.route == "draft_for_manager"
    assert "unsupported_content_delivery_action_detected" in result.safety_flags
    assert "Пришлю" not in result.draft_text
    assert "точный способ доступа" in result.draft_text


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


def test_neutral_discount_theme_is_allowed_as_manager_draft_without_auto_send() -> None:
    result = parse_llm_json(
        '{"route":"draft_for_manager","draft_text":"Здравствуйте! Уточним актуальную скидку.",'
        '"message_type":"question","topic_id":"theme:005_discounts","confidence_theme":0.91}'
    )

    assert result.route == "draft_for_manager"
    assert "manager_approval_required" in result.safety_flags
    assert "high_risk_manager_only" not in result.safety_flags


def _route_shield_fact_store(facts: dict[str, str] | None = None) -> FactStore:
    store_facts = dict(facts or {})
    return FactStore(catalog=tuple(store_facts.keys()), store={"unpk": store_facts, "foton": store_facts})


def _route_shield_contract(
    *,
    question: str = "Сколько стоит курс?",
    answerability: str = "answer_self",
    keys: tuple[str, ...] = (),
    is_p0: bool = False,
    forbidden: tuple[str, ...] = (),
) -> dict:
    return {
        "current_question": question,
        "answerability": answerability,
        "is_p0": is_p0,
        "forbidden_substitutions": list(forbidden),
        "subquestions": [
            {
                "text": question,
                "answerable": "self" if answerability == "answer_self" else "manager",
                "needed_fact_keys": list(keys),
            }
        ],
        "confidence": 0.93,
    }


def _a2_pipeline_metadata(
    *,
    question: str,
    facts: dict[str, str],
    recovery_candidate: str,
    answerability: str = "answer_self",
    is_p0: bool = False,
) -> dict:
    return {
        "dialogue_contract_pipeline": {
            "contract": _route_shield_contract(
                question=question,
                answerability=answerability,
                keys=tuple(facts.keys()),
                is_p0=is_p0,
            ),
            "retrieved_facts": facts,
            "retrieved_fact_keys": list(facts.keys()),
            "recovery_candidate": recovery_candidate,
            "recovery_candidate_validated": True,
        }
    }


def _route_shield_pipeline_result(
    *,
    client_message: str = "Сколько стоит курс?",
    draft_text: str | None = "По подтверждённым данным: курс стоит 49 000 ₽.",
    contract: dict | None = None,
    facts: dict[str, str] | None = None,
    faithfulness_fn=None,
):
    return run_pipeline(
        conversation=({"role": "client", "text": client_message},),
        active_brand="unpk",
        fact_store=_route_shield_fact_store(facts),
        understand_fn=lambda _prompt: contract or _route_shield_contract(keys=tuple((facts or {}).keys())),
        draft_fn=None if draft_text is None else (lambda _prompt: draft_text),
        faithfulness_fn=faithfulness_fn,
    )


def test_pravka4_decide_route_does_not_flip_default_before_veto_shield_is_green() -> None:
    decision = decide_route(
        SubscriptionDraftResult(
            route="draft_for_manager",
            draft_text="Курс стоит 49 000 ₽.",
            message_type="question",
            topic_id="theme:001_pricing",
        ),
        client_message="Сколько стоит?",
        context={
            "active_brand": "unpk",
            "autonomy_policy": {"allow_autonomous": True},
            "client_safe_fact_verified": True,
        },
    )

    assert decision.route == "draft_for_manager"
    assert decision.autonomous_candidate is True


def test_memory_followup_route_promotes_answered_topic_with_covering_fact() -> None:
    decision = decide_route(
        SubscriptionDraftResult(
            route="draft_for_manager",
            draft_text="По подтверждённому факту отвечу по онлайн-формату.",
            message_type="question",
            topic_id="theme:001_pricing",
        ),
        client_message="а онлайн для 10 класса?",
        context={
            "active_brand": "foton",
            "autonomy_policy": {"allow_autonomous": True, "allowed_topic_ids": ["theme:001_pricing"]},
            "client_safe_fact_verified": True,
            "dialogue_memory_view": {
                "route_history": ["bot_answer_self_for_pilot"],
                "answered_questions": ["сколько стоит информатика для 10 класса"],
                "topic_focus": {"subject": "информатика", "grade": "10", "format": "очно", "product_family": "regular_course"},
            },
        },
    )

    assert decision.route == "bot_answer_self_for_pilot"
    assert "dialogue_memory_followup_autonomy" in decision.safety_flags


def test_memory_followup_route_does_not_override_p0() -> None:
    decision = decide_route(
        SubscriptionDraftResult(
            route="draft_for_manager",
            draft_text="По подтверждённому факту отвечу по онлайн-формату.",
            message_type="question",
            topic_id="theme:001_pricing",
        ),
        client_message="я оплатил, занятий нет, верните деньги",
        context={
            "active_brand": "foton",
            "autonomy_policy": {"allow_autonomous": True, "allowed_topic_ids": ["theme:001_pricing"]},
            "client_safe_fact_verified": True,
            "dialogue_memory_view": {
                "route_history": ["bot_answer_self_for_pilot"],
                "answered_questions": ["сколько стоит информатика для 10 класса"],
                "topic_focus": {"subject": "информатика", "grade": "10", "format": "очно", "product_family": "regular_course"},
            },
        },
    )

    assert decision.route == "manager_only"
    assert decision.veto_category == "high_risk"
    assert "high_risk_manager_only" in decision.safety_flags


def test_pravka5_semantic_critic_blocks_wrong_scope_and_contradicted_claims() -> None:
    wrong_scope_result = check_claim_faithfulness(
        "Это онлайн.",
        facts={
            "camp.shift.format": "ЛВШ Менделеево — очная городская смена без проживания.",
            "regular.online.format": "Обычные онлайн-курсы проходят дистанционно.",
        },
        client_words="В каком формате лагерная смена?",
        faithfulness_fn=lambda _prompt: {
            "claims": [
                {
                    "claim": "это онлайн",
                    "evidence_fact_key": "regular.online.format",
                    "verdict": "wrong_scope",
                    "reason": "факт про обычный онлайн-курс, а вопрос про лагерную смену",
                }
            ],
            "unsupported": [],
        },
    )
    assert wrong_scope_result.unsupported == ("это онлайн",)

    wrong_scope_pipeline = _route_shield_pipeline_result(
        client_message="В каком формате лагерная смена?",
        draft_text="Это онлайн.",
        contract=_route_shield_contract(question="В каком формате лагерная смена?", keys=("camp.shift.format",)),
        facts={"camp.shift.format": "ЛВШ Менделеево — очная городская смена без проживания."},
        faithfulness_fn=lambda _prompt: {
            "claims": [
                {
                    "claim": "это онлайн",
                    "evidence_fact_key": "camp.shift.format",
                    "verdict": "wrong_scope",
                    "reason": "черновик отвечает не в scope факта",
                }
            ],
            "unsupported": [],
        },
    )
    assert wrong_scope_pipeline.route == "draft_for_manager"
    assert wrong_scope_pipeline.fallback_reason == "hard_verification_failed"

    contradicted_result = check_claim_faithfulness(
        "Да, программа подходит для 9 класса.",
        facts={"program.grade": "Программа подтверждена для 10 класса."},
        client_words="Подходит для 10 класса?",
        faithfulness_fn=lambda _prompt: {
            "claims": [
                {
                    "claim": "программа подходит для 9 класса",
                    "evidence_fact_key": "program.grade",
                    "verdict": "contradicted",
                    "reason": "факт подтверждает 10 класс, не 9",
                }
            ],
            "unsupported": [],
        },
    )
    assert contradicted_result.unsupported == ("программа подходит для 9 класса",)

    contradicted = _route_shield_pipeline_result(
        client_message="Подходит для 10 класса?",
        draft_text="Да, программа подходит для 9 класса.",
        contract=_route_shield_contract(question="Подходит для 10 класса?", keys=("program.grade",)),
        facts={"program.grade": "Программа подтверждена для 10 класса."},
        faithfulness_fn=lambda _prompt: {
            "claims": [
                {
                    "claim": "программа подходит для 9 класса",
                    "evidence_fact_key": "program.grade",
                    "verdict": "contradicted",
                    "reason": "факт подтверждает 10 класс, не 9",
                }
            ],
            "unsupported": [],
        },
    )
    assert contradicted.route == "draft_for_manager"
    assert contradicted.fallback_reason == "hard_verification_failed"


def test_pravka5_semantic_critic_keeps_supported_same_scope_claim_autonomous() -> None:
    supported = _route_shield_pipeline_result(
        client_message="В каком формате лагерная смена?",
        draft_text="ЛВШ Менделеево — очная городская смена без проживания.",
        contract=_route_shield_contract(question="В каком формате лагерная смена?", keys=("camp.shift.format",)),
        facts={"camp.shift.format": "ЛВШ Менделеево — очная городская смена без проживания."},
        faithfulness_fn=lambda _prompt: {
            "claims": [
                {
                    "claim": "ЛВШ Менделеево — очная городская смена без проживания",
                    "evidence_fact_key": "camp.shift.format",
                    "verdict": "supported",
                    "reason": "тот же продукт, формат и условия",
                }
            ],
            "unsupported": [],
        },
    )
    assert supported.route == "bot_answer_self"


def test_pravka5_1_semantic_critic_prompt_names_remaining_fabrication_types() -> None:
    prompt = build_faithfulness_prompt(
        "Это онлайн, занятия по вторникам, других форматов нет, фокус на ОГЭ.",
        facts={"camp.shift.format": "ЛВШ Менделеево — очная городская смена без проживания."},
        client_words="Лагерь онлайн или очно?",
    )

    assert "ВЫБОР ФОРМАТА" in prompt
    assert "онлайн или очно" in prompt
    assert "РАСПИСАНИЕ/ДНИ/ВРЕМЯ" in prompt
    assert "по вторникам" in prompt
    assert "Лагерь/смена ≠ обычный курс ≠ олимпиадная подготовка" in prompt
    assert "ОТРИЦАНИЕ И СПЕЦИФИКА" in prompt
    assert "других форматов нет" in prompt
    assert "фокус на ОГЭ" in prompt


def test_pravka5_1_semantic_critic_blocks_specific_remaining_fabrication_verdicts() -> None:
    cases = [
        (
            "онлайн или очно, цена 6 класс",
            "Это онлайн.",
            {"format.general": "Есть очные и онлайн-направления; точный формат зависит от выбранной программы."},
            "это онлайн",
            "unsupported",
        ),
        (
            "Когда проходят занятия?",
            "Занятия проходят в будни.",
            {"program.general": "Программа доступна для 9 класса."},
            "занятия проходят в будни",
            "unsupported",
        ),
        (
            "Что за летняя смена?",
            "Это обычный онлайн-курс по олимпиадной подготовке.",
            {"camp.shift": "ЛВШ Менделеево — летняя смена."},
            "это обычный онлайн-курс по олимпиадной подготовке",
            "wrong_scope",
        ),
        (
            "Есть другие выездные форматы?",
            "Других выездных форматов нет.",
            {"camp.shift": "ЛВШ Менделеево — выездная смена."},
            "других выездных форматов нет",
            "unsupported",
        ),
        (
            "Это курс под экзамен?",
            "У курса фокус на ОГЭ.",
            {"program.general": "Курс помогает подтянуть математику."},
            "у курса фокус на ОГЭ",
            "unsupported",
        ),
    ]

    for client_words, draft, facts, claim, verdict in cases:
        result = check_claim_faithfulness(
            draft,
            facts=facts,
            client_words=client_words,
            faithfulness_fn=lambda _prompt, claim=claim, verdict=verdict: {
                "claims": [
                    {
                        "claim": claim,
                        "evidence_fact_key": next(iter(facts)),
                        "verdict": verdict,
                        "reason": "калибровочный пример правки 5.1",
                    }
                ],
                "unsupported": [],
            },
        )
        assert result.unsupported == (claim,)


def test_pravka5_1_semantic_critic_keeps_supported_right_topic() -> None:
    result = check_claim_faithfulness(
        "ЛВШ Менделеево — очная городская смена без проживания.",
        facts={"camp.shift.format": "ЛВШ Менделеево — очная городская смена без проживания."},
        client_words="Лагерь онлайн или очно?",
        faithfulness_fn=lambda _prompt: {
            "claims": [
                {
                    "claim": "ЛВШ Менделеево — очная городская смена без проживания",
                    "evidence_fact_key": "camp.shift.format",
                    "verdict": "supported",
                    "reason": "факт про тот же лагерь и формат",
                }
            ],
            "unsupported": [],
        },
    )

    assert result.unsupported == ()


def test_pravka5_2_complaint_zero_collect_uses_clean_handoff() -> None:
    text = _safe_fallback_text(
        AnswerContract(
            active_brand="foton",
            current_question="Жалоба: преподаватель ужасный, ребёнок ничего не понял.",
            answerability="manager_only",
            is_p0=True,
            p0_reason="complaint",
        ),
        facts={
            "discounts.current": "Скидка на второй предмет — 20%.",
        },
        context={"active_brand": "foton"},
    )
    lowered = text.casefold().replace("ё", "е")

    assert "передам менеджеру" in lowered
    assert "скидк" not in lowered
    assert "укажите" not in lowered
    assert "ребен" not in lowered
    assert "как зовут" not in lowered
    assert not any(char.isdigit() for char in text)


def test_pravka5_2_refund_zero_collect_keeps_refund_handoff() -> None:
    text = _safe_fallback_text(
        AnswerContract(
            active_brand="unpk",
            current_question="Верните деньги, я недовольна занятиями.",
            answerability="manager_only",
            is_p0=True,
            p0_reason="refund",
        ),
        facts={
            "payment.installment": "Есть рассрочка через Т-Банк.",
        },
        context={"active_brand": "unpk"},
    )
    lowered = text.casefold().replace("ё", "е")

    assert "возврат" in lowered
    assert "передам" in lowered
    assert "как отдельная справка" not in lowered
    assert "т-банк" not in lowered


def test_pravka5_2_non_p0_fallback_does_not_use_neighbor_payment_secondary() -> None:
    secondary = _safe_fallback_text(
        AnswerContract(
            active_brand="unpk",
            current_question="Можно помесячно прямым переводом на счёт?",
            answerability="manager_only",
        ),
        facts={
            "payment.installment": "Есть рассрочка через Т-Банк.",
        },
        context={"active_brand": "unpk"},
    )
    assert "менеджер" in secondary.casefold()
    assert "оплату прямым переводом на счёт" in secondary.casefold()
    assert "как отдельная справка" not in secondary.casefold()
    assert "т-банк" not in secondary.casefold()

    detail = _safe_fallback_text(
        AnswerContract(
            active_brand="unpk",
            current_question="Какая цена для 6 класса?",
            answerability="manager_only",
        ),
        facts={},
        context={"active_brand": "unpk"},
    )
    assert "менеджер" in detail.casefold()
    assert "цену или условия оплаты" in detail
    assert "Какая цена для 6 класса" not in detail


def test_brand_separation_guard_uses_canonical_cross_brand_text_on_first_block() -> None:
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="У Фотона и УНПК одинаковые условия по рассрочке.",
        message_type="question",
        topic_id="service:S5_general_consultation",
    )

    guarded = apply_brand_separation_guard(
        result,
        client_message="У Фотона такие же условия, как у УНПК?",
        context={"active_brand": "unpk"},
    )

    assert guarded.route == "manager_only"
    assert "отдельные организации" in guarded.draft_text.casefold()
    assert "фотон" not in guarded.draft_text.casefold()
    assert "унпк" not in guarded.draft_text.casefold()
    assert "cross_brand_safe_template_applied" in guarded.safety_flags


def test_block2_part_a_recovery_candidate_does_not_yield_on_high_risk_or_protective_flags() -> None:
    facts = {"tax.knd_certificate": "Фотон: для налогового вычета можно запросить справку КНД."}
    candidate = "Фотон: для налогового вычета можно запросить справку КНД."
    metadata = _a2_pipeline_metadata(
        question="Можно получить налоговый вычет?",
        facts=facts,
        recovery_candidate=candidate,
    )

    high_risk = SubscriptionDraftResult(
        route="manager_only",
        draft_text=SAFE_FALLBACK_DRAFT_TEXT,
        safety_flags=("tax_safe_template_applied", "high_risk_manager_only"),
        metadata=metadata,
    )
    assert _validated_guardchain_recovery_candidate(
        high_risk,
        client_message="Оплатил, занятий нет, верните деньги. Можно налоговый вычет?",
        context={"active_brand": "foton"},
    ) == ""

    protective = SubscriptionDraftResult(
        route="manager_only",
        draft_text=RESULT_GUARANTEE_SAFE_TEXT,
        safety_flags=("tax_safe_template_applied", "result_guarantee_safe_template_applied"),
        metadata=metadata,
    )
    assert _validated_guardchain_recovery_candidate(
        protective,
        client_message="Гарантируете результат и налоговый вычет?",
        context={"active_brand": "foton"},
    ) == ""


def test_identity_disclosure_detector_uses_word_boundaries() -> None:
    assert not contains_bot_identity_disclosure("Это как и интенсивы прошлого года.")
    assert not contains_bot_identity_disclosure("Олимпиады проходят по правилам России.")
    assert contains_bot_identity_disclosure("Я GPT.")
    assert contains_bot_identity_disclosure("Я ChatGPT.")
    assert contains_bot_identity_disclosure("Я нейросеть.")


def test_unstated_subject_guard_allows_subject_from_active_brand_retrieved_fact() -> None:
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="Для 9 класса по информатике онлайн-курс подходит.",
        message_type="question",
        topic_id="theme:001_pricing",
        metadata={
            "dialogue_contract_pipeline": {
                "retrieved_facts": {
                    "unpk.prices.online.informatics.grade9": (
                        "УНПК: онлайн-курс по информатике для 9 класса доступен в этом наборе."
                    )
                }
            }
        },
    )

    guarded = apply_unstated_subject_guard(
        result,
        client_message="Сколько стоит для 9 класса?",
        context={"active_brand": "unpk"},
    )

    assert guarded.route == "bot_answer_self_for_pilot"
    assert "unstated_subject_guarded" not in guarded.safety_flags


def test_unstated_subject_guard_blocks_subject_not_in_message_slots_or_retrieved_fact() -> None:
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="Для 9 класса по физике онлайн-курс подходит.",
        message_type="question",
        topic_id="theme:001_pricing",
        metadata={
            "dialogue_contract_pipeline": {
                "retrieved_facts": {
                    "unpk.prices.online.informatics.grade9": (
                        "УНПК: онлайн-курс по информатике для 9 класса доступен в этом наборе."
                    )
                }
            }
        },
    )

    guarded = apply_unstated_subject_guard(
        result,
        client_message="Сколько стоит для 9 класса?",
        context={"active_brand": "unpk"},
    )

    assert guarded.route == "draft_for_manager"
    assert "unstated_subject_guarded" in guarded.safety_flags


def test_unstated_subject_guard_blocks_subject_from_other_brand_retrieved_fact() -> None:
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="Для 9 класса по информатике онлайн-курс подходит.",
        message_type="question",
        topic_id="theme:001_pricing",
        metadata={
            "dialogue_contract_pipeline": {
                "retrieved_facts": {
                    "unpk.prices.online.informatics.grade9": (
                        "УНПК: онлайн-курс по информатике для 9 класса доступен в этом наборе."
                    )
                }
            }
        },
    )

    guarded = apply_unstated_subject_guard(
        result,
        client_message="Сколько стоит для 9 класса?",
        context={"active_brand": "foton"},
    )

    assert guarded.route == "draft_for_manager"
    assert "unstated_subject_guarded" in guarded.safety_flags


def test_humanity_x2_rewriter_disabled_by_default() -> None:
    base = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="Сориентирую по проверенным данным: семестр 29 750 ₽.",
        safety_flags=("autonomy_matrix_passed",),
    )

    result = apply_humanity_x2_rewriter(
        base,
        client_message="Сколько стоит семестр?",
        context={"active_brand": "foton", "confirmed_facts": {"price": "семестр 29 750 ₽"}},
        rewrite_runner=lambda prompt: "Семестр — 29 750 ₽. Помогу выбрать группу.",
    )

    assert result.draft_text == base.draft_text
    assert "humanity_x2" not in result.metadata


def test_humanity_x2_rewriter_applies_safe_form_only_candidate() -> None:
    base = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="Сориентирую по проверенным данным: семестр 29 750 ₽. Передам менеджеру.",
        safety_flags=("autonomy_matrix_passed",),
    )

    result = apply_humanity_x2_rewriter(
        base,
        client_message="Сколько стоит семестр?",
        context={
            "active_brand": "foton",
            "humanity_x2_rewrite_enabled": True,
            "confirmed_facts": {"price": "семестр 29 750 ₽"},
        },
        rewrite_runner=lambda prompt: "Семестр — 29 750 ₽. Подскажите класс, и я помогу выбрать ближайший формат.",
    )

    assert result.draft_text.startswith("Семестр — 29 750 ₽")
    assert "humanity_x2_rewritten" in result.safety_flags
    assert result.metadata["humanity_x2"]["rewritten"] is True


def test_humanity_x2_rewriter_rejects_new_number_before_gate() -> None:
    base = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="Семестр — 29 750 ₽.",
        safety_flags=("autonomy_matrix_passed",),
    )

    result = apply_humanity_x2_rewriter(
        base,
        client_message="Сколько стоит семестр?",
        context={
            "active_brand": "foton",
            "humanity_x2_rewrite_enabled": True,
            "confirmed_facts": {"price": "семестр — 29 750 ₽"},
        },
        rewrite_runner=lambda prompt: "Семестр — 29 750 ₽, год — 100 000 ₽.",
    )

    assert result.draft_text == base.draft_text
    assert result.metadata["humanity_x2"]["rewritten"] is False
    assert result.metadata["humanity_x2"]["fallback_reason"] == "fact_drift:100000"


def test_humanity_x2_rewriter_never_touches_manager_only() -> None:
    base = SubscriptionDraftResult(
        route="manager_only",
        draft_text="Приняли обращение. Передам ответственному сотруднику.",
        safety_flags=("high_risk_manager_only",),
    )

    result = apply_humanity_x2_rewriter(
        base,
        client_message="Верните деньги",
        context={"active_brand": "foton", "humanity_x2_rewrite_enabled": True},
        rewrite_runner=lambda prompt: "Давайте решим мягче.",
    )

    assert result.draft_text == base.draft_text
    assert result.metadata["humanity_x2"]["fallback_reason"] == "locked_p0_or_manager_only"


def test_humanity_x2_rewriter_never_touches_identity_policy_c() -> None:
    base = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text=IDENTITY_FOTON_SAFE_TEXT,
        safety_flags=("terminal_safe_template_applied",),
        metadata={
            "dialogue_contract_pipeline": {
                "rules_engine_intent_shadow": {
                    "selected_source": "identity_policy",
                    "selected_intent": "identity",
                }
            }
        },
    )

    result = apply_humanity_x2_rewriter(
        base,
        client_message="это бот?",
        context={"active_brand": "foton", "humanity_x2_rewrite_enabled": True},
        rewrite_runner=lambda prompt: "Я помощник, отвечу теплее.",
    )

    assert result.route == "bot_answer_self_for_pilot"
    assert result.draft_text == IDENTITY_FOTON_SAFE_TEXT
    assert "humanity_x2_rewritten" not in result.safety_flags
    assert result.metadata["humanity_x2"]["fallback_reason"] == "locked_identity_policy"


def test_phase2_tone_reduces_bureaucratic_text_behind_flag() -> None:
    base = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="В рамках текущего учебного центра обучение осуществляется онлайн. Менеджер уточнит ближайший шаг.",
        safety_flags=("rules_engine_format_choice_present_both",),
        metadata={
            "dialogue_contract_pipeline": {
                "contract": _route_shield_contract(question="Как проходит обучение?", keys=("format.online",)),
                "retrieved_facts": {"format.online": "Обучение проходит онлайн."},
                "retrieved_fact_keys": ["format.online"],
            }
        },
    )

    result = apply_phase2_tone_layer(
        base,
        client_message="Как проходит обучение?",
        context={"active_brand": "foton", "phase2_tone_enabled": True},
    )

    assert result.draft_text != base.draft_text
    assert "в рамках текущего учебного центра" not in result.draft_text.casefold()
    assert "осуществляется" not in result.draft_text.casefold()
    assert "phase2_tone_rewritten" in result.safety_flags
    assert result.metadata["phase2_tone"]["tone_after"]["tone_canc"] < result.metadata["phase2_tone"]["tone_before"]["tone_canc"]


def test_phase2_tone_rolls_back_candidate_with_new_product_number() -> None:
    base = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="В рамках текущего учебного центра обучение осуществляется онлайн.",
        safety_flags=("rules_engine_format_choice_present_both",),
        metadata={
            "dialogue_contract_pipeline": {
                "contract": _route_shield_contract(question="Как проходит обучение?", keys=("format.online",)),
                "retrieved_facts": {"format.online": "Обучение проходит онлайн."},
                "retrieved_fact_keys": ["format.online"],
            }
        },
    )

    result = apply_phase2_tone_layer(
        base,
        client_message="Как проходит обучение?",
        context={
            "active_brand": "foton",
            "phase2_tone_enabled": True,
            "phase2_tone_rewrite_fn": lambda _text: "Обучение проходит онлайн. Год стоит 100 000 ₽.",
        },
    )

    assert result.draft_text == base.draft_text
    assert "phase2_tone_rewritten" not in result.safety_flags
    assert "verify_output" in result.metadata["phase2_tone"]["fallback_reason"]


def test_phase2_tone_does_not_touch_p0_or_manager_only() -> None:
    base = SubscriptionDraftResult(
        route="manager_only",
        draft_text="В рамках текущего учебного центра вопрос передам менеджеру.",
        safety_flags=("high_risk_manager_only",),
    )

    result = apply_phase2_tone_layer(
        base,
        client_message="Верните деньги",
        context={"active_brand": "foton", "phase2_tone_enabled": True},
    )

    assert result.draft_text == base.draft_text
    assert "phase2_tone_rewritten" not in result.safety_flags
    assert result.metadata["phase2_tone"]["fallback_reason"] == "locked_p0_or_manager_only"


def test_humanity_x2_rewriter_allows_migrated_rule_answers_with_stripped_internal_marker() -> None:
    cases = (
        (
            "rules_engine_teacher_applied",
            "Преподаватели — эксперты ЕГЭ.",
            "[source_id=fact:v3:teacher] Преподаватели — эксперты ЕГЭ. Помогу подобрать группу.",
            {"teacher": "Преподаватели — эксперты ЕГЭ."},
        ),
        (
            "rules_engine_price_format_matched",
            "Семестр — 49 000 ₽.",
            "[source_id=fact:v3:price] Семестр — 49 000 ₽. Если удобно, подскажу годовой формат.",
            {"price": "Семестр — 49 000 ₽."},
        ),
        (
            "rules_engine_installment_foton",
            "Доступна рассрочка на 6, 10 или 12 месяцев.",
            "[source_id=fact:v3:installment] Доступна рассрочка на 6, 10 или 12 месяцев. Менеджер поможет оформить вариант.",
            {"installment": "Доступна рассрочка на 6, 10 или 12 месяцев."},
        ),
    )

    for flag, original, candidate, facts in cases:
        base = SubscriptionDraftResult(
            route="bot_answer_self_for_pilot",
            draft_text=original,
            safety_flags=(flag,),
            metadata={"rules_engine": {"applied": flag.removeprefix("rules_engine_")}},
        )

        result = apply_humanity_x2_rewriter(
            base,
            client_message="Подскажите, пожалуйста",
            context={"active_brand": "foton", "humanity_x2_rewrite_enabled": True, "confirmed_facts": facts},
            rewrite_runner=lambda prompt, candidate=candidate: candidate,
        )

        assert result.metadata["humanity_x2"]["rewritten"] is True
        assert result.metadata["humanity_x2"]["fallback_reason"] is None
        assert "humanity_x2_rewritten" in result.safety_flags
        assert "source_id" not in result.draft_text
        assert result.draft_text != original


def test_humanity_x2_rewriter_rejects_cross_brand_candidate() -> None:
    base = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="Семестр — 29 750 ₽.",
        safety_flags=("autonomy_matrix_passed",),
    )

    result = apply_humanity_x2_rewriter(
        base,
        client_message="Сколько стоит семестр?",
        context={
            "active_brand": "foton",
            "humanity_x2_rewrite_enabled": True,
            "confirmed_facts": {"price": "семестр — 29 750 ₽"},
        },
        rewrite_runner=lambda prompt: "Семестр — 29 750 ₽. В УНПК условия похожие.",
    )

    assert result.draft_text == base.draft_text
    assert result.metadata["humanity_x2"]["rewritten"] is False
    assert result.metadata["humanity_x2"]["fallback_reason"] == "brand_leak"


def test_humanity_x2_rewriter_rejects_pressure_candidate() -> None:
    base = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="Пробное занятие есть, менеджер поможет подобрать удобный вариант.",
        safety_flags=("rules_engine_trial_available",),
    )

    result = apply_humanity_x2_rewriter(
        base,
        client_message="Можно пробное?",
        context={
            "active_brand": "foton",
            "humanity_x2_rewrite_enabled": True,
            "confirmed_facts": {"trial": "Пробное занятие есть."},
        },
        rewrite_runner=lambda prompt: "Пробное занятие есть, срочно записывайтесь сейчас, иначе мест не останется.",
    )

    assert result.draft_text == base.draft_text
    assert "humanity_x2_rewritten" not in result.safety_flags
    assert result.metadata["humanity_x2"]["rewritten"] is False
    assert result.metadata["humanity_x2"]["fallback_reason"] == "pressure"


def test_humanity_x2_rewriter_falls_back_on_repo_gate_meta_leak() -> None:
    base = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="Семестр 29 750 ₽.",
        safety_flags=("autonomy_matrix_passed",),
    )

    result = apply_humanity_x2_rewriter(
        base,
        client_message="Сколько стоит семестр?",
        context={
            "active_brand": "foton",
            "humanity_x2_rewrite_enabled": True,
            "confirmed_facts": {"price": "семестр 29 750 ₽"},
        },
        rewrite_runner=lambda prompt: "Семестр 29 750 ₽, отвечаю без служебных пометок.",
    )

    assert result.draft_text == base.draft_text
    assert result.metadata["humanity_x2"]["rewritten"] is False
    assert result.metadata["humanity_x2"]["fallback_reason"] == "meta_leak"


def test_v2_unsupported_promise_guard_uses_retrieved_fact_metadata_for_discount_percent() -> None:
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="При онлайн-обучении скидка на второй предмет составляет 20%.",
        message_type="question",
        topic_id="theme:005_discounts",
        topic_confidence=0.91,
        metadata={
            "dialogue_contract_pipeline": {
                "retrieved_facts": {
                    "discounts.second_subject.online.pct": (
                        "УНПК: при онлайн-обучении скидка на второй предмет составляет 20%."
                    )
                }
            }
        },
    )

    guarded = apply_unsupported_promise_guard(result, context={"active_brand": "unpk"})

    assert guarded.route == "bot_answer_self_for_pilot"
    assert "unsupported_promise_detected" not in guarded.safety_flags


def test_v2_unsupported_promise_guard_blocks_100_points_without_retrieved_fact() -> None:
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="На курсе можно гарантированно набрать 100 баллов.",
        message_type="question",
        topic_id="theme:016_program",
        topic_confidence=0.91,
        metadata={"dialogue_contract_pipeline": {"retrieved_facts": {}}},
    )

    guarded = apply_unsupported_promise_guard(result, context={"active_brand": "unpk"})

    assert guarded.route == "manager_only"
    assert "unsupported_promise_detected" in guarded.safety_flags
    assert guarded.metadata["unsupported_promises"] == ["100 баллов"]


def test_v2_unsupported_promise_guard_allows_result_statistics_points() -> None:
    cases = (
        (
            "Средний результат ЕГЭ выше среднего по стране на 25 баллов.",
            {"results_social_proof.ege_avg_above_country_pts": "УНПК: средний результат ЕГЭ у учеников выше среднего по стране на 25 баллов."},
        ),
        (
            "В среднем наши выпускники получают 85 баллов на ЕГЭ.",
            {"results.average_ege_score": "УНПК: в среднем выпускники получают 85 баллов на ЕГЭ."},
        ),
        (
            "Прошлый поток показал 90+ баллов на ЕГЭ по информатике.",
            {"results.previous_cohort_informatics": "УНПК: прошлый поток показал 90+ баллов на ЕГЭ по информатике."},
        ),
    )

    for draft_text, retrieved_facts in cases:
        result = SubscriptionDraftResult(
            route="bot_answer_self_for_pilot",
            draft_text=draft_text,
            message_type="question",
            topic_id="theme:016_program",
            topic_confidence=0.91,
            metadata={"dialogue_contract_pipeline": {"retrieved_facts": retrieved_facts}},
        )

        guarded = apply_unsupported_promise_guard(result, context={"active_brand": "unpk"})

        assert guarded.route == "bot_answer_self_for_pilot", draft_text
        assert "unsupported_promise_detected" not in guarded.safety_flags, draft_text
        assert not find_unsupported_numeric_promises(draft_text, context={}), draft_text


def test_v2_unsupported_promise_guard_blocks_points_promise_context_without_fact() -> None:
    cases = (
        ("Гарантируем 100 баллов на ЕГЭ.", "100 баллов"),
        ("Ваш ребёнок наберёт 90+ баллов.", "90+ баллов"),
        ("Обещаем 80 баллов минимум.", "80 баллов"),
    )

    for draft_text, expected_claim in cases:
        result = SubscriptionDraftResult(
            route="bot_answer_self_for_pilot",
            draft_text=draft_text,
            message_type="question",
            topic_id="theme:016_program",
            topic_confidence=0.91,
            metadata={"dialogue_contract_pipeline": {"retrieved_facts": {}}},
        )

        guarded = apply_unsupported_promise_guard(result, context={"active_brand": "unpk"})

        assert guarded.route == "manager_only", draft_text
        assert "unsupported_promise_detected" in guarded.safety_flags, draft_text
        assert guarded.metadata["unsupported_promises"] == [expected_claim]


def test_authoritative_output_gate_blocks_core_safety_risks() -> None:
    cases = (
        (
            "hard_p0",
            "Оплатил, занятий нет — верните деньги.",
            SubscriptionDraftResult(route="bot_answer_self_for_pilot", draft_text="Да, сейчас подскажу по курсу."),
            {"active_brand": "foton"},
        ),
        (
            "brand_leak",
            "Что у вас по оплате?",
            SubscriptionDraftResult(route="bot_answer_self_for_pilot", draft_text="У УНПК МФТИ условия такие же."),
            {"active_brand": "foton"},
        ),
        (
            "identity_disclosure",
            "Ты кто?",
            SubscriptionDraftResult(route="bot_answer_self_for_pilot", draft_text="Я ChatGPT, помогу с курсом."),
            {"active_brand": "foton"},
        ),
        (
            "promocode_leak",
            "Есть акция?",
            SubscriptionDraftResult(route="bot_answer_self_for_pilot", draft_text="Используйте промокод LVSH-VEB20."),
            {"active_brand": "unpk"},
        ),
        (
            "draft_placeholder",
            "Сколько стоит?",
            SubscriptionDraftResult(route="bot_answer_self_for_pilot", draft_text="Стоимость: [указать сумму]."),
            {"active_brand": "foton"},
        ),
        (
            "unsupported_promise",
            "Какие результаты?",
            SubscriptionDraftResult(route="bot_answer_self_for_pilot", draft_text="Гарантируем 100 баллов на ЕГЭ."),
            {"active_brand": "unpk"},
        ),
    )

    for expected_code, client_message, result, context in cases:
        gated = apply_authoritative_output_gate(result, client_message=client_message, context=context)

        assert gated.route == "manager_only", expected_code
        assert "authoritative_output_gate_blocked" in gated.safety_flags, expected_code
        gate = gated.metadata["authoritative_output_gate"]
        assert gate["action"] == "block", expected_code
        assert expected_code in {item["code"] for item in gate["findings"]}, expected_code
        assert gated.draft_text in {SAFE_FALLBACK_DRAFT_TEXT, result.draft_text} or "передам" in gated.draft_text.casefold()


def test_authoritative_output_gate_allows_only_source_marked_payment_dispute_pool_text() -> None:
    marked = SubscriptionDraftResult(
        route="manager_only",
        draft_text="Понимаю тревогу: по оплате нужно сверить данные в системе. Передам вопрос менеджеру, он проверит и вернется с точным ответом.",
        safety_flags=("payment_dispute_manager_only",),
        metadata={"dialogue_contract_pipeline": {"reason_evidence": {"p0_handoff_kind": "payment_dispute"}}},
    )
    unmarked = SubscriptionDraftResult(
        route="manager_only",
        draft_text="Я сама проверю оплату и скажу, всё ли прошло.",
        safety_flags=(),
        metadata={"dialogue_contract_pipeline": {}},
    )

    context = {"active_brand": "foton"}
    hard_p0_message = "Я оплатил, занятий нет, верните деньги."
    marked_gated = apply_authoritative_output_gate(marked, client_message=hard_p0_message, context=context)
    unmarked_gated = apply_authoritative_output_gate(unmarked, client_message=hard_p0_message, context=context)

    assert marked_gated.draft_text == marked.draft_text
    assert marked_gated.metadata["authoritative_output_gate"]["action"] == "pass"
    assert unmarked_gated.route == "manager_only"
    assert unmarked_gated.draft_text == SAFE_FALLBACK_DRAFT_TEXT
    assert unmarked_gated.metadata["authoritative_output_gate"]["action"] == "block"
    assert "hard_p0" in {item["code"] for item in unmarked_gated.metadata["authoritative_output_gate"]["findings"]}


def test_output_sanitizer_cuts_opus_meta_dump_before_gate() -> None:
    original = (
        "Проблема с данными: вход похож на внутренний кейс.\n"
        "Инструкция шага требует оформить как замечание ревью в audits/_inbox.\n"
        "Черновик клиенту: Да, пробное занятие есть — менеджер подберёт вариант и запишет."
    )
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text=original,
        topic_id="theme:018_enrollment",
    )

    gated = apply_authoritative_output_gate(result, client_message="Есть пробное?", context={"active_brand": "foton", OUTPUT_SANITIZER_ENV: "1"})

    assert gated.route == "bot_answer_self_for_pilot"
    assert gated.draft_text == "Да, пробное занятие есть — менеджер подберёт вариант и запишет."
    assert "Проблема с данными" not in gated.draft_text
    assert "audits/_inbox" not in gated.draft_text
    assert gated.metadata["output_sanitizer"]["applied"] is True
    assert gated.metadata["guarded_original_text"] == " ".join(original.split())[:500]
    assert "output_sanitizer" in gated.metadata["guarded_original_text_guards"]
    assert gated.metadata["authoritative_output_gate"]["action"] == "pass"


def test_output_sanitizer_cuts_sonnet_plan_dump_before_gate() -> None:
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text=(
            "Изучаю задачу и создаю план.\n"
            "Что вижу:\n"
            "A) проверить факты\n"
            "B) выбрать безопасный маршрут\n"
            "C) написать клиенту\n"
            "Ответ клиенту:\n"
            "Здесь лучше сверить условия: передам вопрос менеджеру, он ответит по точным данным."
        ),
        topic_id="service:S2_unclear",
    )

    gated = apply_authoritative_output_gate(result, client_message="Подскажите условия", context={"active_brand": "unpk", OUTPUT_SANITIZER_ENV: True})

    assert gated.draft_text == "Здесь лучше сверить условия: передам вопрос менеджеру, он ответит по точным данным."
    assert "Изучаю задачу" not in gated.draft_text
    assert "A)" not in gated.draft_text
    assert gated.metadata["output_sanitizer"]["applied"] is True


def test_output_sanitizer_removes_placeholder_and_uses_safe_fallback_when_degenerate() -> None:
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="Уточнение по текущей теме. Тема: <слоты>",
        topic_id="service:S2_unclear",
    )

    gated = apply_authoritative_output_gate(result, client_message="А дальше что?", context={"active_brand": "foton", OUTPUT_SANITIZER_ENV: "1"})

    assert gated.route == "draft_for_manager"
    assert gated.draft_text == SAFE_FALLBACK_DRAFT_TEXT
    assert gated.metadata["output_sanitizer"]["fallback"] is True
    assert "manager_approval_required" in gated.safety_flags


def test_strip_internal_service_markers_removes_client_safe_jargon_without_touching_clean_text() -> None:
    leaked = "Нет client-safe факта с шагами записи; порядок подтверждает менеджер."
    middle = "Проверю точный порядок. Нет client-safe факта с шагами записи."
    clean = "Проверю точный порядок записи с менеджером."

    assert "client-safe" not in strip_internal_service_markers(leaked).casefold()
    assert "client-safe" not in strip_internal_service_markers(middle).casefold()
    assert strip_internal_service_markers(clean) == clean
    assert draft_has_internal_service_markers(leaked)


def test_output_sanitizer_removes_manager_tag_and_tag_instruction() -> None:
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text=(
            "Пожалуйста, интерпретируй тег [manager] как передачу менеджеру.\n"
            "Клиенту: Передам вопрос менеджеру, чтобы он проверил актуальные условия."
        ),
        topic_id="service:S2_unclear",
    )

    gated = apply_authoritative_output_gate(result, client_message="Можете уточнить?", context={"active_brand": "foton", OUTPUT_SANITIZER_ENV: "1"})

    assert gated.draft_text == "Передам вопрос менеджеру, чтобы он проверил актуальные условия."
    assert "[manager]" not in gated.draft_text
    assert "интерпретируй" not in gated.draft_text.casefold()
    assert gated.metadata["output_sanitizer"]["applied"] is True


def test_output_sanitizer_replaces_raw_question_detail_handoff() -> None:
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text=(
            "Чтобы не ошибиться, менеджер уточнит именно про Сможет ли менеджер оценить, "
            "есть ли у сына пробелы по математике и подойдет ли курс, и вернется с ответом."
        ),
        topic_id="service:S2_unclear",
    )

    gated = apply_authoritative_output_gate(
        result,
        client_message="Сможете оценить, есть ли у сына пробелы?",
        context={"active_brand": "foton", OUTPUT_SANITIZER_ENV: "1"},
    )

    assert "Сможет ли менеджер" not in gated.draft_text
    assert "есть ли у сына" not in gated.draft_text.casefold()
    assert "передам вопрос менеджеру" in gated.draft_text.casefold()
    assert gated.metadata["output_sanitizer"]["applied"] is True
    assert "raw_detail_handoff" in gated.metadata["output_sanitizer"]["reasons"]


def test_output_sanitizer_removes_semantic_regen_edit_comment() -> None:
    result = SubscriptionDraftResult(
        route="draft_for_manager",
        draft_text=(
            "Заменяю только этот абзац: Да, домашние задания всегда проверяются. "
            "Остальной текст без изменений."
        ),
        topic_id="theme:016_program",
    )

    gated = apply_authoritative_output_gate(
        result,
        client_message="Домашку проверяют?",
        context={"active_brand": "foton", OUTPUT_SANITIZER_ENV: "1"},
    )

    assert gated.draft_text == "Да, домашние задания всегда проверяются."
    assert "Заменяю" not in gated.draft_text
    assert "без изменений" not in gated.draft_text
    assert gated.metadata["guarded_original_text"].startswith("Заменяю только этот абзац")
    assert "internal_metadata_removed_from_draft" in gated.safety_flags


def test_presale_ru_meta_sanitizer_removes_confirmed_facts_jargon_without_flagging_clean_handoff() -> None:
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="Этого нет в подтверждённых фактах.",
        topic_id="service:S2_unclear",
    )

    gated = apply_authoritative_output_gate(
        result,
        client_message="А можно так?",
        context={"active_brand": "foton", PRESALE_META_RU_ENV: "1"},
    )

    assert "подтверждённых фактах" not in gated.draft_text
    assert gated.route == "draft_for_manager"
    assert gated.draft_text == SAFE_FALLBACK_DRAFT_TEXT
    assert "presale_ru_meta_line" in gated.metadata["output_sanitizer"]["reasons"]

    clean = SubscriptionDraftResult(
        route="draft_for_manager",
        draft_text="У меня нет подтверждённого факта именно про этот вариант — менеджер уточнит.",
        topic_id="service:S2_unclear",
    )
    clean_gated = apply_authoritative_output_gate(
        clean,
        client_message="А можно так?",
        context={"active_brand": "foton", PRESALE_META_RU_ENV: "1"},
    )

    assert clean_gated.draft_text == clean.draft_text
    assert "output_sanitizer" not in clean_gated.metadata


def test_presale_source_id_sanitizer_removes_bare_fact_identifier() -> None:
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text=(
            "По факту presentation_format_facts_2026_05_21: "
            "Очные группы делятся по уровням."
        ),
        topic_id="theme:016_program",
    )

    gated = apply_authoritative_output_gate(
        result,
        client_message="Как делятся группы?",
        context={"active_brand": "foton", PRESALE_SOURCE_ID_ENV: "1"},
    )

    assert gated.route == "bot_answer_self_for_pilot"
    assert gated.draft_text == "Очные группы делятся по уровням."
    assert "presentation_format_facts_2026_05_21" not in gated.draft_text
    assert "по факту" not in gated.draft_text.casefold()
    assert "presale_source_id" in gated.metadata["output_sanitizer"]["reasons"]


def test_presale_source_id_sanitizer_removes_bot_safe_runtime_identifiers() -> None:
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text=(
            "Продолжим по customer:63bf70693f8a921d013c1da6901d551d "
            "и botsafe:customer:63bf70693f8a921d013c1da6901d551d:foton: "
            "лучше уточнить удобный формат."
        ),
        topic_id="theme:016_program",
    )

    gated = apply_authoritative_output_gate(
        result,
        client_message="Что дальше?",
        context={"active_brand": "foton", PRESALE_SOURCE_ID_ENV: "1"},
    )

    assert "customer:" not in gated.draft_text
    assert "botsafe:" not in gated.draft_text
    assert "лучше уточнить удобный формат" in gated.draft_text
    assert "presale_source_id" in gated.metadata["output_sanitizer"]["reasons"]


def test_presale_source_id_sanitizer_does_not_cut_normal_fact_or_format_words() -> None:
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="Факт простой: формат занятий зависит от выбранной группы.",
        topic_id="theme:016_program",
    )

    gated = apply_authoritative_output_gate(
        result,
        client_message="Какой формат?",
        context={"active_brand": "foton", PRESALE_SOURCE_ID_ENV: "1"},
    )

    assert gated.draft_text == result.draft_text
    assert "output_sanitizer" not in gated.metadata


def test_presale_source_id_sanitizer_off_parity_keeps_identifier() -> None:
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="По факту presentation_format_facts_2026_05_21: Очные группы делятся по уровням.",
        topic_id="theme:016_program",
    )

    gated = apply_authoritative_output_gate(
        result,
        client_message="Как делятся группы?",
        context={"active_brand": "foton", PRESALE_SOURCE_ID_ENV: "0"},
    )

    assert gated.draft_text == result.draft_text
    assert "output_sanitizer" not in gated.metadata


def test_presale_source_id_sanitizer_enabled_by_pilot_gold_config() -> None:
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="Источник kb_v6_6_client_safe_facts_2026_06_08.homework: домашние задания проверяются.",
        topic_id="theme:016_program",
    )

    gated = apply_authoritative_output_gate(
        result,
        client_message="Домашку проверяют?",
        context={
            "active_brand": "foton",
            DIRECT_PATH_PILOT_CONFIG_ENV: DIRECT_PATH_PILOT_CONFIG_VERSION,
        },
    )

    assert "kb_v6_6_client_safe_facts_2026_06_08" not in gated.draft_text
    assert gated.draft_text == "домашние задания проверяются."
    assert "presale_source_id" in gated.metadata["output_sanitizer"]["reasons"]


def test_output_sanitizer_preserves_client_paragraphs() -> None:
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text=(
            "Ответ клиенту:\n"
            "Да, домашние задания всегда проверяются.   \n\n"
            "Материалы и задания идут в чате с преподавателем."
        ),
        topic_id="theme:016_program",
    )

    gated = apply_authoritative_output_gate(
        result,
        client_message="Домашку проверяют?",
        context={"active_brand": "foton", OUTPUT_SANITIZER_ENV: "1"},
    )

    assert gated.draft_text == (
        "Да, домашние задания всегда проверяются.\n\n"
        "Материалы и задания идут в чате с преподавателем."
    )
    assert gated.metadata["output_sanitizer"]["applied"] is True


def test_strip_internal_service_markers_preserves_safe_variant_paragraphs() -> None:
    text = (
        "служебная заметка: безопасный вариант: "
        '"Да, домашние задания всегда проверяются.\n\nМатериалы идут в чате."'
    )

    assert strip_internal_service_markers(text) == (
        "Да, домашние задания всегда проверяются.\n\n"
        "Материалы идут в чате."
    )


def test_output_sanitizer_removes_tone_noise_phrases_and_separators() -> None:
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="---\nЗдравствующий момент: Да, домашние задания всегда проверяются. Никакого спешки.",
        topic_id="theme:016_program",
    )

    gated = apply_authoritative_output_gate(
        result,
        client_message="Домашку проверяют?",
        context={"active_brand": "foton", OUTPUT_SANITIZER_ENV: "1"},
    )

    assert "---" not in gated.draft_text
    assert "Здравствующий" not in gated.draft_text
    assert "Никакого спешки" not in gated.draft_text
    assert "Да, домашние задания всегда проверяются." in gated.draft_text
    assert {"tone_separator", "bad_tone_phrase"}.issubset(set(gated.metadata["output_sanitizer"]["reasons"]))


def test_tone_warm_frame_rewrites_robotic_fact_prefix_only_when_enabled() -> None:
    text = "По подтверждённым данным: домашние задания всегда проверяются."

    assert apply_warm_frame(text, context={}) == text

    warmed = apply_warm_frame(text, context={TONE_WARM_FRAME_ENV: "1"})

    assert warmed != text
    assert warmed.endswith("домашние задания всегда проверяются.")
    assert "По подтверждённым данным" not in warmed
    assert warmed.startswith(
        (
            "Конечно! Вот как это устроено у нас:",
            "Да, подскажу:",
            "Смотрите, что есть для вас:",
        )
    )
    frame = warmed.split("домашние задания", 1)[0].casefold()
    assert not any(marker in frame for marker in ("данн", "баз", "подтвержд", "провер"))

    schedule = apply_warm_frame("Нашёл такую группу: занятия по вторникам.", context={TONE_WARM_FRAME_ENV: "1"})
    assert schedule.startswith(("Подобрала для вас вариант:", "Есть такая группа:"))


def test_tone_close_detect_replaces_handoff_on_clean_thanks_without_repeating_numbers() -> None:
    result = SubscriptionDraftResult(
        route="draft_for_manager",
        draft_text=SAFE_FALLBACK_DRAFT_TEXT,
        topic_id="service:S2_unclear",
        metadata={"reason_class": "no_fact_or_unverified"},
    )
    context = {
        "active_brand": "unpk",
        TONE_CLOSE_DETECT_ENV: "1",
        "dialogue_memory_view": {
            "recent_turns": [
                {"role": "bot", "text": "Стоимость курса — 49 000 ₽."},
            ],
            "proactive_state": {},
        },
    }

    closed = apply_tone_close_detect_layer(result, client_message="Спасибо, всё понятно", context=context)

    assert closed.route == "bot_answer_self_for_pilot"
    assert closed.metadata["close_detect"]["status"] == "suppressed_handoff"
    assert closed.metadata["close_detect"]["step"] == "contact"
    assert closed.metadata["is_manager_deferral"] is False
    assert closed.metadata["reason_class"] == ""
    assert "телефон" in closed.draft_text.casefold()
    assert closed.draft_text.startswith("Рада была помочь!")
    assert "позвоним" in closed.draft_text.casefold()
    assert "49 000" not in closed.draft_text


@pytest.mark.parametrize(
    "finding_code",
    [
        "fake_enrollment_claim",
        "brand_leak",
        "unsupported_product_number",
        "cross_brand",
    ],
)
def test_tone_close_gate_findings_floor_preserves_authoritative_demotion(finding_code: str) -> None:
    source = SubscriptionDraftResult(
        route="draft_for_manager",
        draft_text="Передам вопрос менеджеру.",
        safety_flags=(
            "authoritative_output_gate_blocked",
            f"authoritative_gate:{finding_code}",
            "manager_approval_required",
            "no_auto_send",
        ),
        metadata={
            "authoritative_output_gate": {
                "action": "block",
                "findings": [{"code": finding_code, "source": "test"}],
            }
        },
    )

    result = apply_tone_close_detect_layer(
        source,
        client_message="Спасибо, всё понятно",
        context={
            "active_brand": "foton",
            TONE_CLOSE_DETECT_ENV: "1",
        },
    )

    assert result.route == source.route
    assert result.draft_text == source.draft_text
    assert result.safety_flags == source.safety_flags
    assert result.metadata["close_detect"]["status"] == "suppressed_authoritative_gate"


@pytest.mark.parametrize(
    ("gate", "flags"),
    [
        ({"action": "downgrade", "findings": []}, ()),
        ({"action": "pass", "findings": [{"code": "brand_leak"}]}, ()),
        ({"action": "pass", "findings": []}, ("authoritative_gate:brand_leak",)),
    ],
)
def test_tone_close_gate_findings_floor_accepts_each_gate_signal(
    gate: dict[str, object],
    flags: tuple[str, ...],
) -> None:
    source = SubscriptionDraftResult(
        route="draft_for_manager",
        draft_text="Передам вопрос менеджеру.",
        safety_flags=flags,
        metadata={"authoritative_output_gate": gate},
    )

    result = apply_tone_close_detect_layer(
        source,
        client_message="Спасибо, всё понятно",
        context={TONE_CLOSE_DETECT_ENV: "1"},
    )

    assert result.route == source.route
    assert result.draft_text == source.draft_text
    assert result.metadata["close_detect"]["status"] == "suppressed_authoritative_gate"


def test_tone_close_gate_findings_floor_is_unconditional() -> None:
    source = SubscriptionDraftResult(
        route="draft_for_manager",
        draft_text="Передам вопрос менеджеру.",
        safety_flags=("authoritative_gate:brand_leak", "manager_approval_required", "no_auto_send"),
        metadata={"authoritative_output_gate": {"action": "block", "findings": [{"code": "brand_leak"}]}},
    )

    result = apply_tone_close_detect_layer(
        source,
        client_message="Спасибо, всё понятно",
        context={"active_brand": "foton", TONE_CLOSE_DETECT_ENV: "1"},
    )

    assert result.route == source.route
    assert result.draft_text == source.draft_text
    assert result.metadata["close_detect"]["status"] == "suppressed_authoritative_gate"


def test_tone_close_gate_findings_floor_keeps_clean_thanks_warm() -> None:
    source = SubscriptionDraftResult(
        route="draft_for_manager",
        draft_text="Передам вопрос менеджеру.",
        safety_flags=("manager_approval_required", "no_auto_send"),
        metadata={"authoritative_output_gate": {"action": "pass", "findings": []}},
    )

    result = apply_tone_close_detect_layer(
        source,
        client_message="Спасибо, всё понятно",
        context={
            "active_brand": "foton",
            TONE_CLOSE_DETECT_ENV: "1",
        },
    )

    assert result.route == "bot_answer_self_for_pilot"
    assert result.draft_text != source.draft_text


def test_tone_close_detect_contact_step_records_contact_requested() -> None:
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="Рада была помочь.",
        topic_id="service:S2_unclear",
    )
    context = {
        "active_brand": "unpk",
        TONE_CLOSE_DETECT_ENV: "1",
        "dialogue_memory_view": {"recent_turns": [], "proactive_state": {}},
    }

    closed = apply_tone_close_detect_layer(result, client_message="Спасибо, всё понятно", context=context)
    memory = build_dialogue_memory(current_message="Спасибо, всё понятно", active_brand="unpk")
    updated = update_dialogue_memory_after_answer(
        memory,
        answer_text=closed.draft_text,
        route=closed.route,
    )

    assert closed.metadata["close_detect"]["status"] == "fired"
    assert closed.metadata["close_detect"]["step"] == "contact"
    assert closed.metadata["close_detect"]["contact_requested"] is True
    assert updated.to_prompt_view()["proactive_state"]["contact_requested"] is True


def test_tone_close_detect_deduplicates_previous_contact_cta() -> None:
    previous_contact = (
        "Рада была помочь! Хотите, менеджер подберёт группу под ваше расписание? "
        "Оставьте телефон — позвоним, когда удобно."
    )
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="Рада была помочь.",
        topic_id="service:S2_unclear",
    )
    context = {
        "active_brand": "unpk",
        TONE_CLOSE_DETECT_ENV: "1",
        "dialogue_memory_view": {
            "recent_turns": [{"role": "bot", "text": previous_contact}],
            "proactive_state": {},
        },
    }

    closed = apply_tone_close_detect_layer(result, client_message="Спасибо", context=context)

    assert closed.metadata["close_detect"]["status"] == "fired"
    assert closed.metadata["close_detect"]["step"] == "return"
    assert closed.draft_text != previous_contact
    assert "телефон" not in closed.draft_text.casefold()
    assert "позвоним" not in closed.draft_text.casefold()


def test_tone_close_detect_refusal_after_previous_step_finishes_without_cta() -> None:
    previous_contact = (
        "Рада была помочь! Хотите, менеджер подберёт группу под ваше расписание? "
        "Оставьте телефон — позвоним, когда удобно."
    )
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="Рада была помочь.",
        topic_id="service:S2_unclear",
    )
    context = {
        "active_brand": "foton",
        TONE_CLOSE_DETECT_ENV: "1",
        "dialogue_memory_view": {
            "recent_turns": [{"role": "bot", "text": previous_contact}],
            "proactive_state": {},
        },
    }

    closed = apply_tone_close_detect_layer(result, client_message="Нет, не нужно, спасибо", context=context)

    assert closed.metadata["close_detect"]["status"] == "fired"
    assert closed.metadata["close_detect"]["step"] == "return"
    lowered = closed.draft_text.casefold()
    assert "телефон" not in lowered
    assert "позвоним" not in lowered
    assert "пробн" not in lowered
    assert "запис" not in lowered
    assert "менеджер" not in lowered


def test_tone_close_detect_does_not_capture_exit_signal_or_new_question() -> None:
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="Возвращайтесь, если появится вопрос.",
        topic_id="service:S2_unclear",
    )
    context = {"active_brand": "foton", TONE_CLOSE_DETECT_ENV: "1"}

    exit_turn = apply_tone_close_detect_layer(result, client_message="Спасибо, подумаю и вернусь", context=context)
    question_turn = apply_tone_close_detect_layer(result, client_message="Спасибо! А когда старт?", context=context)

    assert "close_detect" not in exit_turn.metadata
    assert "close_detect" not in question_turn.metadata


def test_tone_close_detect_does_not_capture_adversative_unanswered_or_payment_problem() -> None:
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="Возвращайтесь, если появится вопрос.",
        topic_id="service:S2_unclear",
    )
    context = {"active_brand": "unpk", TONE_CLOSE_DETECT_ENV: "1"}

    unanswered = apply_tone_close_detect_layer(result, client_message="Поняла, но пока вы не ответили по сути", context=context)
    unclear_value = apply_tone_close_detect_layer(result, client_message="Поняла. Но мне всё равно непонятно, за что платим…", context=context)
    plural_exit = apply_tone_close_detect_layer(result, client_message="Спасибо, подумаем", context=context)
    payment_problem = apply_tone_close_detect_layer(
        result,
        client_message="Хорошо, жду ответа. Только прошу СРОЧНО: деньги списали, платежа в системе нет.",
        context=context,
    )

    assert "close_detect" not in unanswered.metadata
    assert "close_detect" not in unclear_value.metadata
    assert "close_detect" not in plural_exit.metadata
    assert "close_detect" not in payment_problem.metadata


def test_tone_close_detect_suppresses_p0_and_pending_manager_without_cta() -> None:
    result = SubscriptionDraftResult(
        route="manager_only",
        draft_text=PAYMENT_DISPUTE_SAFE_TEXT,
        topic_id="theme:p0_payment",
        safety_flags=("payment_dispute", "p0"),
    )
    p0_context = {
        "active_brand": "foton",
        TONE_CLOSE_DETECT_ENV: "1",
        "dialogue_memory_view": {"p0_latch": {"active": True, "codes": ["payment_dispute"]}},
    }

    p0_closed = apply_tone_close_detect_layer(result, client_message="Спасибо", context=p0_context)

    assert p0_closed.route == "manager_only"
    assert p0_closed.draft_text == PAYMENT_DISPUTE_SAFE_TEXT
    assert p0_closed.metadata["close_detect"]["status"] == "suppressed_p0"

    pending = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="Передам вопрос менеджеру.",
        topic_id="service:S2_unclear",
    )
    pending_context = {
        "active_brand": "unpk",
        TONE_CLOSE_DETECT_ENV: "1",
        "dialogue_memory_view": {"handoff_state": "suggested", "pending_manager_actions": ["manager_handoff"]},
    }
    pending_closed = apply_tone_close_detect_layer(pending, client_message="Спасибо, жду ответа менеджера", context=pending_context)

    assert pending_closed.route == "manager_only"
    assert pending_closed.metadata["close_detect"]["status"] == "suppressed_pending"
    assert "телефон" not in pending_closed.draft_text.casefold()
    assert pending_closed.draft_text == "Спасибо! Менеджер проверит детали и вернётся с ответом."

    manager_reference = apply_tone_close_detect_layer(
        pending,
        client_message="Спасибо, пусть менеджер уточнит",
        context=pending_context,
    )

    assert manager_reference.metadata["close_detect"]["status"] == "suppressed_pending"
    assert manager_reference.metadata["close_detect"]["step"] == "pending"
    assert "телефон" not in manager_reference.draft_text.casefold()

    plain_thanks = apply_tone_close_detect_layer(pending, client_message="Спасибо", context=pending_context)

    assert plain_thanks.metadata["close_detect"]["status"] == "fired"
    assert plain_thanks.metadata["close_detect"]["step"] == "contact"
    assert "телефон" in plain_thanks.draft_text.casefold()

    hard_p0_pending_context = {
        "active_brand": "unpk",
        TONE_CLOSE_DETECT_ENV: "1",
        "dialogue_memory_view": {
            "handoff_state": "suggested",
            "pending_manager_actions": ["manager_handoff"],
            "p0_latch": {"active": False, "codes": ["payment_dispute"], "had_hard_p0_claim": True},
        },
    }
    hard_p0_pending = apply_tone_close_detect_layer(pending, client_message="Спасибо", context=hard_p0_pending_context)

    assert hard_p0_pending.metadata["close_detect"]["status"] == "fired"
    assert hard_p0_pending.metadata["close_detect"]["step"] == "return"
    assert "телефон" not in hard_p0_pending.draft_text.casefold()

    hard_p0_pending_next = apply_tone_close_detect_layer(pending, client_message="Спасибо", context=hard_p0_pending_context)

    assert hard_p0_pending_next.metadata["close_detect"]["status"] == "fired"
    assert hard_p0_pending_next.metadata["close_detect"]["step"] == "return"

    classifier_only_p0 = apply_tone_close_detect_layer(
        replace(pending, safety_flags=("payment_dispute",)),
        client_message="Спасибо",
        context={
            "active_brand": "unpk",
            TONE_CLOSE_DETECT_ENV: "1",
            "dialogue_memory_view": {"p0_latch": {"active": False, "had_hard_p0_claim": False}},
        },
    )

    assert classifier_only_p0.metadata["close_detect"]["status"] == "suppressed_p0"


def test_tone_close_detect_uses_contact_requested_memory_before_foton_trial_step() -> None:
    memory = build_dialogue_memory(current_message="Есть пробное?", active_brand="foton")
    updated = update_dialogue_memory_after_answer(
        memory,
        answer_text="Спасибо, оставьте телефон и время для связи — передам менеджеру.",
        route="bot_answer_self_for_pilot",
    )
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="Рада была помочь.",
        topic_id="service:S2_unclear",
    )

    memory_view = {**dict(updated.to_prompt_view()), "handoff_state": "none", "pending_manager_actions": []}
    closed = apply_tone_close_detect_layer(
        result,
        client_message="Спасибо",
        context={"active_brand": "foton", TONE_CLOSE_DETECT_ENV: "1", "dialogue_memory_view": memory_view},
    )

    assert updated.to_prompt_view()["proactive_state"]["contact_requested"] is True
    assert closed.metadata["close_detect"]["status"] == "fired"
    assert closed.metadata["close_detect"]["step"] == "trial"
    assert "пробн" in closed.draft_text.casefold()
    assert closed.draft_text.startswith("Обращайтесь в любое время!")
    assert "телефон" not in closed.draft_text.casefold()


def test_direct_path_applies_tone_close_detect_to_self_route_product_facts() -> None:
    provider = _DirectPathProvider(
        SubscriptionDraftResult(
            route="bot_answer_self_for_pilot",
            draft_text=(
                "На ИТ-направлении ноутбук с собой не обязателен: оборудование предоставляет организатор. "
                "Ночью компьютерный класс ставится на сигнализацию. На смене есть медсестра."
            ),
            topic_id="theme:summer_camp",
            context_used=("lvsh_it_equipment", "lvsh_medical_support"),
        )
    )
    context = {
        "active_brand": "unpk",
        DIRECT_PATH_ENV: "1",
        TONE_CLOSE_DETECT_ENV: "1",
        "confirmed_facts": {
            "lvsh_it_equipment": "На ИТ-направлении оборудование предоставляет организатор.",
            "lvsh_medical_support": "На смене есть медсестра.",
        },
        "dialogue_memory_view": {
            "recent_turns": [
                {
                    "role": "bot",
                    "text": "В летней выездной школе есть ИТ-направление для 7-10 классов.",
                }
            ],
            "proactive_state": {},
        },
    }

    closed = provider.build_draft("Поняла, спасибо.", context=context)

    assert provider.calls == 1
    assert closed.route == "bot_answer_self_for_pilot"
    assert closed.metadata["close_detect"]["status"] == "fired"
    assert closed.metadata["close_detect"]["step"] == "return"
    lowered = closed.draft_text.casefold()
    assert "телефон" not in lowered
    assert "позвоним" not in lowered
    assert "ноутбук" not in lowered
    assert "сигнализац" not in lowered
    assert "медсестр" not in lowered


def test_direct_path_tone_close_detect_does_not_cut_confirmed_camp_detail_question() -> None:
    draft = (
        "На ИТ-направлении ноутбук с собой не обязателен: оборудование предоставляет организатор. "
        "Ночью компьютерный класс ставится на сигнализацию. На смене есть медсестра."
    )
    provider = _DirectPathProvider(
        SubscriptionDraftResult(
            route="bot_answer_self_for_pilot",
            draft_text=draft,
            topic_id="theme:summer_camp",
            context_used=("lvsh_it_equipment", "lvsh_medical_support"),
        )
    )

    result = provider.build_draft(
        "А ноутбук нужен на ИТ-направление?",
        context={
            "active_brand": "unpk",
            DIRECT_PATH_ENV: "1",
            TONE_CLOSE_DETECT_ENV: "1",
            "confirmed_facts": {
                "lvsh_it_equipment": "На ИТ-направлении оборудование предоставляет организатор.",
                "lvsh_medical_support": "На смене есть медсестра.",
            },
        },
    )

    assert provider.calls == 1
    assert result.route == "bot_answer_self_for_pilot"
    assert result.draft_text == draft
    assert "close_detect" not in result.metadata


def test_direct_path_tone_close_detect_replaces_cautious_handoff_without_phone_cta() -> None:
    provider = _DirectPathProvider(
        SubscriptionDraftResult(
            route="draft_for_manager",
            draft_text="Передам менеджеру, чтобы уточнить детали.",
            topic_id="service:S5_general_consultation",
            safety_flags=("manager_approval_required", "no_auto_send"),
        )
    )

    closed = provider.build_draft(
        "Хорошо, спасибо",
        context={"active_brand": "unpk", DIRECT_PATH_ENV: "1", TONE_CLOSE_DETECT_ENV: "1"},
    )

    assert provider.calls == 1
    assert closed.route == "bot_answer_self_for_pilot"
    assert closed.metadata["close_detect"]["status"] == "suppressed_handoff"
    assert closed.metadata["close_detect"]["step"] == "return"
    lowered = closed.draft_text.casefold()
    assert "телефон" not in lowered
    assert "позвоним" not in lowered


def test_payment_dispute_handoff_antirepeat_rotates_without_product_promises() -> None:
    second = _p0_text_with_antirepeat(
        "payment_dispute",
        PAYMENT_DISPUTE_SAFE_TEXT,
        context={"recent_messages": [f"Бот: {PAYMENT_DISPUTE_SAFE_TEXT}"]},
    )
    third = _p0_text_with_antirepeat(
        "payment_dispute",
        PAYMENT_DISPUTE_SAFE_TEXT,
        context={"recent_messages": [f"Бот: {PAYMENT_DISPUTE_SAFE_TEXT}", f"Бот: {second}"]},
    )

    assert second != PAYMENT_DISPUTE_SAFE_TEXT
    assert third not in {PAYMENT_DISPUTE_SAFE_TEXT, second}
    combined = f"{second} {third}".casefold()
    assert "проверит" in combined or "сверит" in combined
    assert "занятие не отмен" not in combined
    assert "оплата прошла" not in combined
    assert "место сохран" not in combined


def test_tone_wave2_prompt_blocks_are_gated_and_preserve_brand_boundaries() -> None:
    context = {
        "active_brand": "unpk",
        TONE_SELL_PROMPT_ENV: "1",
        TONE_RICH_FORMAT_ENV: "1",
        "confirmed_facts": {
            "payment_options.unpk": "УНПК: можно платить помесячно, за семестр или за год.",
            "discounts.semester_payment": "УНПК: при оплате за семестр действует скидка 10%.",
            "discounts.year_payment": "УНПК: при оплате за год действует скидка 14%.",
        },
        "conversation_intent_plan": {
            "primary_intent": "payment_method",
            "direct_question": "Серьёзная сумма для семьи, как записаться?",
            "selling": {"objection": "price", "exit_signal": False, "readiness": "ready"},
        },
        "dialogue_memory_view": {
            "proactive_state": {"contact_requested": True, "recent_ignored": 2},
            "a2_proactive_state": {"recent_ignored": 2},
        },
        "next_best_question": "Для какого класса смотрите курс?",
    }

    prompt = build_draft_prompt("Серьёзная сумма для семьи, как записаться?", context=context)
    off_prompt = build_draft_prompt("Сколько стоит?", context={"active_brand": "unpk"})

    assert "Продающий тон TELEGRAM_TONE_SELL_PROMPT" in prompt
    assert "Форматирование TELEGRAM_TONE_RICH_FORMAT" in prompt
    assert "максимум пользы сразу" in prompt
    assert "бренду, формату, классу, предмету и продукту" in prompt
    assert "за что платим" in prompt
    assert "не обещай результат" in prompt
    assert "максимум один на ход" in prompt
    assert "Не задавай список вопросов" in prompt
    assert "recent_ignored >= 2" in prompt
    assert "contact_requested=true" in prompt
    assert "как записаться" in prompt
    assert "скидки не придумывай" in prompt
    assert "УНПК: не предлагай рассрочку, Долями" in prompt
    assert "10%/14%" in prompt
    assert "по подтверждённым данным" in prompt
    assert "по проверенным ценам" in prompt
    assert "пустая строка между блоками" in prompt
    assert "TELEGRAM_TONE_SELL_PROMPT" not in off_prompt
    assert "TELEGRAM_TONE_RICH_FORMAT" not in off_prompt


def test_tone_sell_prompt_observer_logs_missing_step_without_changing_text() -> None:
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="Стоимость онлайн-курса — 49 000 ₽.",
        topic_id="theme:001_pricing",
    )

    observed = apply_tone_sell_prompt_observer(
        result,
        client_message="Спасибо",
        context={"active_brand": "foton", TONE_SELL_PROMPT_ENV: "1"},
    )
    with_step = apply_tone_sell_prompt_observer(
        replace(result, draft_text="Стоимость онлайн-курса — 49 000 ₽. Подскажу, как записаться."),
        client_message="Спасибо",
        context={"active_brand": "foton", TONE_SELL_PROMPT_ENV: "1"},
    )
    with_new_step_words = apply_tone_sell_prompt_observer(
        replace(result, draft_text="Стоимость онлайн-курса — 49 000 ₽. Обращайтесь, расскажу, как подобрать группу."),
        client_message="Спасибо",
        context={"active_brand": "foton", TONE_SELL_PROMPT_ENV: "1"},
    )

    assert observed.draft_text == result.draft_text
    assert observed.route == result.route
    assert observed.metadata["tone_sell_prompt"]["enabled"] is True
    assert observed.metadata["tone_sell_prompt"]["step_missing"] is True
    assert observed.metadata["sell_prompt_step_missing"] is True
    assert with_step.metadata["tone_sell_prompt"]["step_missing"] is False
    assert with_step.metadata["tone_sell_prompt"]["step_kind"] == "generic_help"
    assert with_step.metadata["tone_sell_prompt"]["step_match"]
    assert "sell_prompt_step_missing" not in with_step.metadata
    assert with_new_step_words.metadata["tone_sell_prompt"]["step_missing"] is False
    assert with_new_step_words.metadata["tone_sell_prompt"]["step_kind"] == "generic_help"
    assert "sell_prompt_step_missing" not in with_new_step_words.metadata


def test_tone_sell_prompt_allows_contact_capture_without_a2_proactive_offer() -> None:
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="Да, записаться можно. Оставьте телефон — менеджер подберёт группу.",
        topic_id="theme:020_enrollment",
        metadata={"tone_sell_prompt": {"enabled": True}},
    )

    captured = apply_a2_proactive_layer(
        result,
        client_message="Мой телефон +7 999 123-45-67, удобно завтра вечером",
        context={"active_brand": "foton", TONE_SELL_PROMPT_ENV: "1"},
    )

    assert captured.route == "draft_for_manager"
    assert captured.manager_followup_required is True
    assert "a2_proactive_contact_captured" in captured.safety_flags
    assert "+7" not in captured.draft_text
    assert "999" not in captured.draft_text
    assert "завтра вечером" not in captured.draft_text.casefold()
    assert captured.metadata["a2_proactive"]["phone_masked"] == "[phone:***67]"
    assert captured.metadata["a2_proactive"]["preferred_time"] == "[provided]"


def test_authoritative_gate_does_not_turn_presale_refund_followup_into_p0() -> None:
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="Посмотрите программу и расписание, а если появится вопрос по группе — я помогу сориентироваться.",
        topic_id="theme:013_schedule",
    )
    context = {
        "active_brand": "unpk",
        "recent_messages": [
            "Клиент: А если не подойдёт, можно будет вернуть деньги?",
            "Ответ: Да, при досрочном отказе возвращается остаток неистраченных средств.",
        ],
        "conversation_intent_plan": {
            "primary_intent": "schedule",
            "risk_signals": [],
            "route_bias": "bot_answer_self_for_pilot",
        },
        "dialogue_memory_view": {
            "p0_latch": {
                "active": True,
                "codes": ["refund"],
                "primary_risk": "refund",
                "had_hard_p0_claim": True,
            }
        },
    }

    gated = apply_authoritative_output_gate(
        result,
        client_message="Понял, спасибо. Посмотрю программу и расписание",
        context=context,
    )

    assert gated.route == "bot_answer_self_for_pilot"
    findings = gated.metadata["authoritative_output_gate"]["findings"]
    assert all(item["code"] not in {"hard_p0", "zero_collect_required"} for item in findings)


def test_authoritative_gate_keeps_payment_dispute_latch_p0_on_followup() -> None:
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="Расписание можно посмотреть в карточке группы.",
        topic_id="theme:013_schedule",
    )
    context = {
        "active_brand": "unpk",
        "recent_messages": [
            "Клиент: Я оплатил, но в системе нет моего платежа, деньги списали!",
            "Ответ: Приняли вопрос по оплате. Передам его менеджеру.",
        ],
        "conversation_intent_plan": {
            "primary_intent": "schedule",
            "risk_signals": [],
            "route_bias": "bot_answer_self_for_pilot",
        },
        "dialogue_memory_view": {
            "p0_latch": {
                "active": True,
                "codes": ["payment_dispute"],
                "primary_risk": "payment_dispute",
                "had_hard_p0_claim": True,
            }
        },
    }

    gated = apply_authoritative_output_gate(
        result,
        client_message="Понял, спасибо. Посмотрю программу и расписание",
        context=context,
    )

    assert gated.route == "manager_only"
    findings = gated.metadata["authoritative_output_gate"]["findings"]
    assert any(item["code"] == "hard_p0" for item in findings)


def test_output_sanitizer_keeps_clean_detail_handoff_unchanged() -> None:
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="Чтобы не ошибиться, менеджер уточнит именно про дни и время занятий нужной группы и вернется с ответом.",
        topic_id="service:S2_unclear",
    )

    gated = apply_authoritative_output_gate(
        result,
        client_message="Какие дни занятий?",
        context={"active_brand": "foton", OUTPUT_SANITIZER_ENV: "1"},
    )

    assert gated.draft_text == result.draft_text
    assert "output_sanitizer" not in gated.metadata


def test_output_sanitizer_keeps_clean_client_answer_unchanged() -> None:
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="Да, пробное занятие есть — менеджер подберёт вариант записи.",
        topic_id="theme:018_enrollment",
    )

    gated = apply_authoritative_output_gate(result, client_message="Есть пробное?", context={"active_brand": "foton", OUTPUT_SANITIZER_ENV: "1"})

    assert gated.draft_text == result.draft_text
    assert "output_sanitizer" not in gated.metadata
    assert gated.metadata["authoritative_output_gate"]["action"] == "pass"


def test_output_sanitizer_is_off_by_default() -> None:
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="[manager] Передам вопрос менеджеру, чтобы он проверил актуальные условия.",
        topic_id="service:S2_unclear",
    )

    gated = apply_authoritative_output_gate(result, client_message="Можете уточнить?", context={"active_brand": "foton"})

    assert gated.draft_text == result.draft_text
    assert "output_sanitizer" not in gated.metadata


def test_night_hours_note_is_off_by_default() -> None:
    result = SubscriptionDraftResult(
        route="draft_for_manager",
        draft_text="Передам вопрос менеджеру, он вернётся с ответом.",
        topic_id="service:S2_unclear",
    )

    gated = apply_authoritative_output_gate(
        result,
        client_message="Когда ответят?",
        context={"active_brand": "foton", "now_msk_hour": 22},
    )

    assert gated.draft_text == result.draft_text
    assert "night_hours_note_applied" not in gated.safety_flags


def test_night_hours_note_skips_daytime_and_adds_once_at_night() -> None:
    result = SubscriptionDraftResult(
        route="draft_for_manager",
        draft_text="Передам вопрос менеджеру, он вернётся с ответом.",
        topic_id="service:S2_unclear",
    )
    base_context = {"active_brand": "foton", subscription_llm.NIGHT_HOURS_NOTE_ENV: "1"}

    daytime = apply_authoritative_output_gate(
        result,
        client_message="Когда ответят?",
        context={**base_context, "now_msk_hour": 12},
    )
    nighttime = apply_authoritative_output_gate(
        result,
        client_message="Когда ответят?",
        context={**base_context, "now_msk_hour": 22},
    )
    repeated = apply_authoritative_output_gate(
        nighttime,
        client_message="Когда ответят?",
        context={**base_context, "now_msk_hour": 22},
    )

    assert daytime.draft_text == result.draft_text
    assert nighttime.draft_text.count(subscription_llm.NIGHT_HOURS_NOTE_TEXT) == 1
    assert repeated.draft_text.count(subscription_llm.NIGHT_HOURS_NOTE_TEXT) == 1
    assert "night_hours_note_applied" in nighttime.safety_flags
    assert nighttime.metadata["night_hours_note"]["hour_msk"] == 22


def test_night_hours_note_covers_p0_manager_text() -> None:
    result = SubscriptionDraftResult(
        route="manager_only",
        draft_text=PAYMENT_DISPUTE_SAFE_TEXT,
        topic_id="theme:003_payment_status",
        safety_flags=("payment_dispute_manager_only",),
    )

    gated = apply_authoritative_output_gate(
        result,
        client_message="Деньги списали дважды",
        context={
            "active_brand": "foton",
            subscription_llm.NIGHT_HOURS_NOTE_ENV: "1",
            "now_msk_hour": 23,
        },
    )

    assert gated.route == "manager_only"
    assert gated.draft_text.count(subscription_llm.NIGHT_HOURS_NOTE_TEXT) == 1
    assert "night_hours_note_applied" in gated.safety_flags


def test_authoritative_output_gate_blocks_operational_specificity_without_fact() -> None:
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="Приезжайте в офис, оформим запись на площадке.",
        topic_id="theme:018_enrollment",
    )

    gated = apply_authoritative_output_gate(result, client_message="Как записаться?", context={"active_brand": "foton"})

    assert gated.route == "draft_for_manager"
    assert "authoritative_output_gate_blocked" in gated.safety_flags
    gate = gated.metadata["authoritative_output_gate"]
    assert gate["action"] == "downgrade"
    assert "unsupported_offline_visit_invitation" in {item["code"] for item in gate["findings"]}


def test_authoritative_output_gate_allows_clean_backed_range_answer() -> None:
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="Да, 10 класс подходит: очные курсы Фотона рассчитаны на 5-11 классы.",
        topic_id="theme:016_program",
        metadata={
            "dialogue_contract_pipeline": {
                "retrieved_facts": {
                    "foton.regular.offline.grades": "Фотон: очные курсы рассчитаны на 5-11 классы.",
                }
            }
        },
    )

    gated = apply_authoritative_output_gate(
        result,
        client_message="Для 10 класса есть очно?",
        context={"active_brand": "foton"},
    )

    assert gated.route == "bot_answer_self_for_pilot"
    gate = gated.metadata["authoritative_output_gate"]
    assert gate["action"] == "pass"
    assert gate["findings"] == []


def test_authoritative_output_gate_is_downgrade_only_and_does_not_promote_routes() -> None:
    result = SubscriptionDraftResult(
        route="draft_for_manager",
        draft_text="Передам менеджеру, он уточнит детали по нужной программе.",
        topic_id="theme:016_program",
    )

    gated = apply_authoritative_output_gate(
        result,
        client_message="Есть программа?",
        context={
            "active_brand": "foton",
            "confirmed_facts": {"program": "Фотон: есть программы по математике и информатике."},
        },
    )

    assert gated.route == "draft_for_manager"
    assert "authoritative_output_gate_blocked" not in gated.safety_flags
    assert gated.draft_text == result.draft_text


def test_a2_contact_capture_creates_warm_handoff_without_echoing_pii() -> None:
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="Да, курс есть. Если удобно, передам менеджеру — подскажите телефон и когда лучше связаться?",
        topic_id="theme:020_enrollment",
        safety_flags=("rules_engine_a2_offer_callback",),
        metadata={"a2_proactive": {"step": "offer_callback"}},
    )

    captured = apply_a2_proactive_layer(
        result,
        client_message="Мой телефон +7 999 123-45-67, удобно завтра вечером",
        context={"active_brand": "foton", "a_proactive_enabled": True},
    )

    assert captured.route == "draft_for_manager"
    assert captured.manager_followup_required is True
    assert "a2_proactive_contact_captured" in captured.safety_flags
    assert "+7" not in captured.draft_text
    assert "999" not in captured.draft_text
    assert "завтра вечером" not in captured.draft_text.casefold()
    assert captured.metadata["a2_proactive"]["phone_masked"] == "[phone:***67]"
    assert captured.metadata["a2_proactive"]["preferred_time"] == "[provided]"
    assert captured.metadata["a2_proactive"]["crm_write"] is False


def test_a2_contact_capture_uses_known_phone_and_p0_blocks_capture() -> None:
    base = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="Если удобно, передам менеджеру — подскажите, когда лучше связаться?",
        topic_id="theme:020_enrollment",
        safety_flags=("rules_engine_a2_offer_callback",),
        metadata={"a2_proactive": {"step": "offer_callback"}},
    )

    known_phone = apply_a2_proactive_layer(
        base,
        client_message="Лучше после 18",
        context={"active_brand": "foton", "a_proactive_enabled": True, "known_slots": {"phone_known": True}},
    )
    p0 = apply_a2_proactive_layer(
        replace(base, route="manager_only", safety_flags=("high_risk_manager_only",)),
        client_message="Верните деньги, мой телефон +7 999 123-45-67",
        context={"active_brand": "foton", "a_proactive_enabled": True},
    )

    assert known_phone.route == "draft_for_manager"
    assert known_phone.metadata["a2_proactive"]["phone_masked"] == "[known_phone]"
    assert "после 18" not in known_phone.draft_text
    assert p0.route == "manager_only"
    assert "a2_proactive_contact_captured" not in p0.safety_flags


def test_a2_gate_blocks_fake_enrollment_and_pii_echo_when_flagged() -> None:
    fake_done = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="Я вас записал на курс, приходите завтра.",
        topic_id="theme:020_enrollment",
        metadata={"a2_proactive": {"step": "offer_callback"}},
    )
    pii_echo = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="Передам номер +7 999 123-45-67 менеджеру.",
        topic_id="theme:020_enrollment",
        metadata={"a2_proactive": {"step": "offer_callback"}},
    )

    fake_gated = apply_authoritative_output_gate(fake_done, client_message="Запишите меня", context={"active_brand": "foton"})
    pii_gated = apply_authoritative_output_gate(
        pii_echo,
        client_message="Мой телефон +7 999 123-45-67",
        context={"active_brand": "foton"},
    )

    assert fake_gated.route == "manager_only"
    assert "fake_enrollment_claim" in {item["code"] for item in fake_gated.metadata["authoritative_output_gate"]["findings"]}
    assert pii_gated.route == "manager_only"
    assert "proactive_pii_echo" in {item["code"] for item in pii_gated.metadata["authoritative_output_gate"]["findings"]}
    assert "+7 999" not in pii_gated.draft_text


def test_a2_gate_flags_question_barrage_and_rich_format_limits_emoji() -> None:
    barrage = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="Подскажите класс? Предмет? Когда удобно?",
        topic_id="theme:020_enrollment",
        metadata={"a2_proactive": {"step": "offer_callback"}},
    )
    emoji = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="Да, передам менеджеру 🙂👍✨",
        topic_id="theme:020_enrollment",
        metadata={"a2_proactive": {"step": "offer_callback"}},
    )
    serious = SubscriptionDraftResult(
        route="manager_only",
        draft_text="Приняли обращение 😌",
        topic_id="theme:020_enrollment",
        safety_flags=("complaint_apology_guarded",),
    )

    barrage_gated = apply_authoritative_output_gate(barrage, client_message="Хочу обсудить курс", context={"active_brand": "foton"})
    emoji_clean = apply_a2_proactive_layer(emoji, client_message="Хочу обсудить курс", context={"active_brand": "foton", "a_rich_format_enabled": True})
    serious_clean = apply_a2_proactive_layer(
        serious,
        client_message="Ребёнок ничего не понял, хочу жалобу",
        context={"active_brand": "foton", "a_rich_format_enabled": True},
    )

    assert barrage_gated.route == "draft_for_manager"
    assert "proactive_too_many_questions" in {item["code"] for item in barrage_gated.metadata["authoritative_output_gate"]["findings"]}
    assert len([char for char in emoji_clean.draft_text if ord(char) > 0x2600]) <= 1
    assert "🙂" not in serious_clean.draft_text


def test_v2_unsupported_promise_guard_numeric_siblings_from_rfk() -> None:
    cases = (
        (
            "При онлайн-обучении скидка на второй предмет составляет 20%.",
            {"discounts.second_subject.online.pct": "УНПК: при онлайн-обучении скидка на второй предмет составляет 20%."},
            False,
        ),
        (
            "При онлайн-обучении скидка на второй предмет составляет 25%.",
            {"discounts.second_subject.online.pct": "УНПК: при онлайн-обучении скидка на второй предмет составляет 20%."},
            True,
        ),
        (
            "Для 9 класса онлайн-курс стоит 69 900 ₽.",
            {"prices.online.year": "УНПК: онлайн-курс для 9 класса, год — 69 900 ₽."},
            False,
        ),
        (
            "Для 9 класса онлайн-курс стоит 70 900 ₽.",
            {"prices.online.year": "УНПК: онлайн-курс для 9 класса, год — 69 900 ₽."},
            True,
        ),
        (
            "Эта цена действует до 1 июля.",
            {"prices.before_2026_07_01": "УНПК: ранняя цена действует до 1 июля."},
            False,
        ),
        (
            "Эта цена действует до 15 мая.",
            {},
            True,
        ),
        (
            "По результатам ученики могут набрать 100 баллов.",
            {"results.max_score": "УНПК: по результатам ученики могут набрать 100 баллов."},
            False,
        ),
        (
            "По результатам ученики могут набрать 100 баллов.",
            {},
            True,
        ),
    )

    for draft_text, retrieved_facts, should_block in cases:
        result = SubscriptionDraftResult(
            route="bot_answer_self_for_pilot",
            draft_text=draft_text,
            message_type="question",
            topic_id="theme:005_discounts",
            topic_confidence=0.91,
            metadata={"dialogue_contract_pipeline": {"retrieved_facts": retrieved_facts}},
        )

        guarded = apply_unsupported_promise_guard(result, context={"active_brand": "unpk"})

        if should_block:
            assert guarded.route == "manager_only", draft_text
            assert "unsupported_promise_detected" in guarded.safety_flags, draft_text
        else:
            assert guarded.route == "bot_answer_self_for_pilot", draft_text
            assert "unsupported_promise_detected" not in guarded.safety_flags, draft_text


def test_volna_peresborki_semantic_coverage_allows_rephrased_verified_numeric_fact() -> None:
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="На второй предмет действует скидка 20%.",
        message_type="question",
        topic_id="theme:005_discounts",
        topic_confidence=0.91,
        metadata={
            "dialogue_contract_pipeline": {
                "retrieved_facts": {
                    "discounts.second_subject.offline.pct": (
                        "Фотон: для второго и последующих очных предметов одного ребёнка скидка составляет 20 процентов."
                    )
                }
            }
        },
    )

    guarded = apply_unsupported_promise_guard(result, context={"active_brand": "foton"})

    assert guarded.route == "bot_answer_self_for_pilot"
    assert "unsupported_promise_detected" not in guarded.safety_flags


def test_volna_peresborki_semantic_coverage_negative_controls_block_real_fabrication() -> None:
    cases = (
        ("скидка 25%", ("УНПК: средний результат ЕГЭ выше среднего по стране на 25 баллов.",)),
        ("скидка 25%", ("УНПК: скидка на второй предмет составляет 20%.",)),
        ("до 15 мая", ("УНПК: ранняя цена действует до 1 июля.",)),
        ("Фотон: скидка 20%", ("УНПК: скидка на второй предмет составляет 20%.",)),
        ("70 900 ₽", ("УНПК: онлайн-курс для 9 класса, год — 69 900 ₽.",)),
        ("обычно есть утренние группы", ("УНПК: по расписанию обычно доступны группы в вечернее время.",)),
        ("занятия проходят по будням", ("УНПК: обычно бывают разные слоты, в том числе по выходным.",)),
        ("обычно есть выходные группы", ("УНПК: обычно есть группы в будние дни.",)),
    )
    for claim, facts in cases:
        assert not _claim_supported_by_facts(claim, facts), claim


def test_step4_keep_answer_supported_allows_rephrasing_but_keeps_hard_anchors() -> None:
    assert _keep_answer_supported(
        "На второй предмет действует скидка 20%.",
        ("Фотон: для второго и последующих очных предметов одного ребёнка скидка составляет 20 процентов.",),
    )
    assert not _keep_answer_supported(
        "Год стоит 70 900 ₽.",
        ("УНПК: онлайн-курс для 9 класса, год — 69 900 ₽.",),
    )
    assert not _keep_answer_supported(
        "Фотон: скидка 20%.",
        ("УНПК: скидка на второй предмет составляет 20%.",),
    )
    assert not _keep_answer_supported(
        "Менеджер вернётся завтра.",
        ("Менеджер свяжется сегодня.",),
    )


def test_step4_keep_answer_flag_uses_verifier_not_substring_for_informational_yield() -> None:
    facts = {
        "discount.second_subject": (
            "Фотон: для второго и последующих очных предметов одного ребёнка скидка составляет 20 процентов."
        )
    }
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="На второй предмет действует скидка 20%.",
        message_type="question",
        topic_id="theme:005_discounts",
        metadata=_a2_pipeline_metadata(
            question="Есть скидка на второй предмет?",
            facts=facts,
            recovery_candidate="",
        ),
    )

    assert _verified_informational_answer(
        result,
        client_message="Есть скидка на второй предмет?",
        context={"active_brand": "foton", "TELEGRAM_STEP4_KEEP_ANSWER": "1"},
    )


def test_step4_keep_answer_does_not_bypass_output_verifier_for_non_numeric_fabrication() -> None:
    facts = {"platform.webinars": "УНПК: онлайн-вебинары проходят на платформе МТС Линк."}
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="Онлайн-занятия проходят в Zoom.",
        message_type="question",
        topic_id="theme:014_format",
        metadata=_a2_pipeline_metadata(
            question="Где проходят онлайн-занятия?",
            facts=facts,
            recovery_candidate="",
        ),
    )

    assert not _verified_informational_answer(
        result,
        client_message="Где проходят онлайн-занятия?",
        context={"active_brand": "unpk", "TELEGRAM_STEP4_KEEP_ANSWER": "1"},
    )


def test_volna_peresborki_operational_guard_uses_retrieved_fact_metadata_semantically() -> None:
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="Обычно есть вечерние группы.",
        message_type="question",
        topic_id="theme:013_schedule",
        metadata={
            "dialogue_contract_pipeline": {
                "retrieved_facts": {
                    "schedule.guidance": "УНПК: по расписанию обычно доступны группы в вечернее время."
                }
            }
        },
    )

    guarded = apply_unconfirmed_operational_specificity_guard(
        result,
        context={"active_brand": "unpk", "facts_stale": True},
    )

    assert guarded.route == "bot_answer_self_for_pilot"
    assert "unsupported_schedule_assumption_detected" not in guarded.safety_flags


def test_volna_peresborki_operational_guard_blocks_wrong_scope_schedule_claim() -> None:
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="Обычно есть субботние группы.",
        message_type="question",
        topic_id="theme:013_schedule",
        metadata={
            "dialogue_contract_pipeline": {
                "retrieved_facts": {
                    "schedule.guidance": "УНПК: по расписанию обычно доступны группы в вечернее время."
                }
            }
        },
    )

    guarded = apply_unconfirmed_operational_specificity_guard(
        result,
        context={"active_brand": "unpk"},
    )

    assert guarded.route == "manager_only"
    assert "unsupported_schedule_assumption_detected" in guarded.safety_flags


def test_volna_peresborki_fresh_fact_texts_keeps_verified_fresh_facts_despite_global_stale_flag() -> None:
    context = {
        "facts_stale": True,
        "facts_context": {
            "fresh": True,
            "client_safe_fact_verified": True,
            "confirmed_facts": {
                "discounts.second_subject.offline.pct": "Фотон: скидка на второй предмет составляет 20%."
            },
        },
    }

    assert "Фотон: скидка на второй предмет составляет 20%." in _fresh_fact_texts(context)


def test_volna_peresborki_fresh_fact_texts_still_drops_unverified_stale_facts() -> None:
    context = {
        "facts_stale": True,
        "facts_context": {
            "confirmed_facts": {
                "discounts.second_subject.offline.pct": "Фотон: скидка на второй предмет составляет 20%."
            },
        },
    }

    assert _fresh_fact_texts(context) == ()


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


def test_antirepeat_strict_replaces_repeat_against_any_prior_bot_turn() -> None:
    repeated = (
        "По этому вопросу менеджер проверит детали и вернётся с ответом. "
        "Сейчас точный порядок лучше уточнить отдельно."
    )
    base = parse_llm_json(
        '{"route":"draft_for_manager","draft_text":"'
        + repeated
        + '","message_type":"question","topic_id":"theme:013_schedule","confidence_theme":0.86,'
        '"missing_facts":["schedule.current"]}'
    )

    result = apply_humanity_guards(
        base,
        client_message="А конкретно по каким дням занятия?",
        context={
            "active_brand": "unpk",
            "antirepeat_strict_enabled": True,
            "recent_messages": [
                f"Ответ: {repeated}",
                "Клиент: понятно",
                "Ответ: Другой промежуточный ответ без повторения.",
            ],
            "conversation_intent_plan": {
                "primary_intent": "schedule",
                "topic_id": "theme:013_schedule",
                "fact_scope": "class_schedule",
                "blocked_neighbor_scopes": ["office_hours"],
                "required_fact_keys": ["schedule.current"],
            },
            "facts_context": {"facts_missing": True, "required_fact_keys": ["schedule.current"]},
        },
    )

    assert result.draft_text != repeated
    assert "дни и время занятий" in result.draft_text
    assert "humanity_strict_antirepeat_fallback_applied" in result.safety_flags


def test_safe_fallback_draft_text_antirepeat_covers_battle_fallback() -> None:
    base = parse_llm_json(
        '{"route":"draft_for_manager","draft_text":"'
        + SAFE_FALLBACK_DRAFT_TEXT
        + '","message_type":"question","topic_id":"service:S2_unclear","confidence_theme":0.8}'
    )

    result = apply_humanity_guards(
        base,
        client_message="уточните дату старта",
        context={"recent_messages": [f"Ответ: {SAFE_FALLBACK_DRAFT_TEXT}"]},
    )

    assert result.draft_text != SAFE_FALLBACK_DRAFT_TEXT
    assert "спасибо за сообщение" not in result.draft_text.casefold()
    assert "humanity_strict_antirepeat_fallback_applied" in result.safety_flags


def test_p0_final_override_rotates_repeat_without_partial_value() -> None:
    base = parse_llm_json(
        '{"route":"bot_answer_self_for_pilot","draft_text":"Верните деньги, напишите номер договора.",'
        '"message_type":"question","topic_id":"theme:009_refund","confidence_theme":0.96}'
    )

    result = apply_high_risk_content_guards(
        base,
        client_message="Верните деньги, я недовольна.",
        context={"recent_messages": [f"Ответ: {REFUND_ZERO_COLLECT_SAFE_TEXT}"]},
    )

    assert result.route == "manager_only"
    assert result.draft_text != REFUND_ZERO_COLLECT_SAFE_TEXT
    assert "возврат" in result.draft_text.casefold()
    assert "ничего дополнительно" in result.draft_text.casefold()
    assert "скидк" not in result.draft_text.casefold()
    assert "договор" not in result.draft_text.casefold()


def test_antirepeat_strict_keeps_dry_p0_repeat() -> None:
    base = parse_llm_json(
        '{"route":"manager_only","draft_text":"'
        + REFUND_ZERO_COLLECT_SAFE_TEXT
        + '","message_type":"question","topic_id":"theme:009_refund","confidence_theme":0.96,'
        '"safety_flags":["high_risk_manager_only","zero_collect_refund_guarded"]}'
    )

    result = apply_humanity_guards(
        base,
        client_message="Верните деньги.",
        context={
            "antirepeat_strict_enabled": True,
            "recent_messages": [f"Ответ: {REFUND_ZERO_COLLECT_SAFE_TEXT}"],
        },
    )

    assert result.draft_text == REFUND_ZERO_COLLECT_SAFE_TEXT
    assert "humanity_strict_antirepeat_fallback_applied" not in result.safety_flags


def _step2b1_pipeline_metadata(question: str, facts: dict[str, str]) -> dict:
    return {
        "dialogue_contract_pipeline": {
            "contract": _route_shield_contract(question=question, answerability="answer_self", keys=tuple(facts.keys())),
            "retrieved_facts": facts,
            "retrieved_fact_keys": list(facts.keys()),
        }
    }


def _step2b1_context(*, brand: str, intent: str, question: str, facts: dict[str, str]) -> dict:
    topic_id = {
        "teacher": "theme:017_teachers",
        "recording": "theme:018_materials_homework",
        "address": "theme:015_address",
        "document": "theme:012_certificates",
        "matkap": "theme:007_matkap_payment",
        "tax": "theme:008_tax_deduction",
        "olympiad_online": "theme:016_program",
        "platform_access": "theme:024_account_access",
        "installment": "theme:006_installment",
        "payment_method": "theme:002_payment_method",
        "payment_by_invoice_monthly": "theme:002_payment_method",
        "discount": "theme:005_discounts",
        "pricing": "theme:001_pricing",
        "format": "theme:014_format",
        "trial": "theme:023_trial_class",
        "camp": "theme:026_camp_general",
        "live_availability": "theme:026_camp_general",
        "enrollment_process": "theme:020_enrollment",
        "refund_policy": "theme:020_enrollment",
    }.get(intent, "service:S5_general_consultation")
    return {
        "active_brand": brand,
        "client_message": question,
        "conversation_intent_plan": {
            "active_brand": brand,
            "primary_intent": intent,
            "topic_id": topic_id,
            "direct_question": question,
            "answer_policy": "answer_directly_if_fact_verified",
            "route_bias": "bot_answer_self_for_pilot",
            "required_fact_keys": list(facts.keys()),
        },
        "autonomy_policy": {
            "allow_autonomous": True,
            "allow_default_autonomy": True,
            "allowed_topic_ids": [topic_id],
        },
        "confirmed_facts": facts,
    }


def _step2b1_result(*, question: str, facts: dict[str, str], topic_id: str = "service:S5_general_consultation") -> SubscriptionDraftResult:
    return SubscriptionDraftResult(
        route="draft_for_manager",
        draft_text="Передам менеджеру, он уточнит и вернется с ответом.",
        topic_id=topic_id,
        metadata=_step2b1_pipeline_metadata(question, facts),
    )


def _step3a_result_with_planner(
    *,
    question: str,
    facts: dict[str, str],
    planner_intent: str,
    planner_confidence: float,
    planner_subvariant: str = "",
    planner_slots: dict[str, str] | None = None,
    is_p0: bool = False,
) -> SubscriptionDraftResult:
    result = _step2b1_result(question=question, facts=facts)
    metadata = dict(result.metadata)
    pipeline = dict(metadata["dialogue_contract_pipeline"])
    contract = dict(pipeline["contract"])
    contract.update(
        {
            "planner_intent": planner_intent,
            "planner_subvariant": planner_subvariant,
            "planner_slots": dict(planner_slots or {}),
            "planner_confidence": planner_confidence,
            "is_p0": is_p0,
        }
    )
    pipeline["contract"] = contract
    metadata["dialogue_contract_pipeline"] = pipeline
    return replace(result, metadata=metadata)


def test_step2b1_address_fact_still_blocked_for_non_address_question() -> None:
    facts = {"rules_registry.contact_address.foton.address": "Фотон: адрес очных занятий — Москва, Верхняя Красносельская ул., 30."}
    question = "Сколько стоит онлайн-курс по математике?"

    findings = verify_dialogue_contract_output(
        "Фотон: Москва, Верхняя Красносельская ул., 30.",
        facts=facts,
        active_brand="foton",
        contract=AnswerContract(active_brand="foton", current_question=question, answerability="answer_self"),
        client_message=question,
    )

    assert any(finding.code == "wrong_intent_fact" for finding in findings)


def test_autonomy_scope_precision_profile_default_on_and_explicit_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(AUTONOMY_SCOPE_PRECISION_ENV, raising=False)

    assert autonomy_scope_precision_enabled({}) is False
    assert autonomy_scope_precision_enabled({DIRECT_PATH_PILOT_CONFIG_ENV: DIRECT_PATH_PILOT_CONFIG_VERSION}) is True
    assert autonomy_scope_precision_enabled({AUTONOMY_SCOPE_PRECISION_ENV: "1"}) is True
    assert autonomy_scope_precision_enabled({
        DIRECT_PATH_PILOT_CONFIG_ENV: DIRECT_PATH_PILOT_CONFIG_VERSION,
        AUTONOMY_SCOPE_PRECISION_ENV: "0",
    }) is False
    monkeypatch.setenv(DIRECT_PATH_PILOT_CONFIG_ENV, DIRECT_PATH_PILOT_CONFIG_VERSION)
    assert autonomy_scope_precision_enabled() is True
    monkeypatch.setenv(AUTONOMY_SCOPE_PRECISION_ENV, "0")
    assert autonomy_scope_precision_enabled() is False
    assert AUTONOMY_SCOPE_PRECISION_ENV in subscription_llm.DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS


def test_autonomy_scope_precision_address_synonym_is_flagged(monkeypatch: pytest.MonkeyPatch) -> None:
    facts = {"rules_registry.contact_address.foton.address": "Фотон: адрес очных занятий — Москва, Верхняя Красносельская ул., 30."}
    question = "Как доехать до Фотона на очные занятия?"
    draft = "Фотон: Москва, Верхняя Красносельская ул., 30."
    contract = AnswerContract(active_brand="foton", current_question=question, answerability="answer_self")

    monkeypatch.delenv(AUTONOMY_SCOPE_PRECISION_ENV, raising=False)
    off_findings = verify_dialogue_contract_output(
        draft,
        facts=facts,
        active_brand="foton",
        contract=contract,
        client_message=question,
    )

    monkeypatch.setenv(AUTONOMY_SCOPE_PRECISION_ENV, "1")
    on_findings = verify_dialogue_contract_output(
        draft,
        facts=facts,
        active_brand="foton",
        contract=contract,
        client_message=question,
    )

    assert any(finding.code == "wrong_intent_fact" for finding in off_findings)
    assert not any(finding.code == "wrong_intent_fact" for finding in on_findings)


def test_autonomy_scope_precision_c1_address_fact_still_blocked_for_price_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(AUTONOMY_SCOPE_PRECISION_ENV, "1")
    facts = {"rules_registry.contact_address.foton.address": "Фотон: адрес очных занятий — Москва, Верхняя Красносельская ул., 30."}
    question = "Сколько стоит онлайн-курс по математике?"

    findings = verify_dialogue_contract_output(
        "Фотон: Москва, Верхняя Красносельская ул., 30.",
        facts=facts,
        active_brand="foton",
        contract=AnswerContract(active_brand="foton", current_question=question, answerability="answer_self"),
        client_message=question,
    )

    assert any(finding.code == "wrong_intent_fact" for finding in findings)


def test_autonomy_scope_precision_c3_lvsh_out_of_context_still_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(AUTONOMY_SCOPE_PRECISION_ENV, "1")
    facts = {
        "lvsh_mendeleevo_2026.address.client_safe_text": "Фотон: ЛВШ Менделеево проходит в кампусе МФТИ.",
    }
    question = "Как доехать до очных занятий в Москве?"

    findings = verify_dialogue_contract_output(
        "Фотон: ЛВШ Менделеево проходит в кампусе МФТИ.",
        facts=facts,
        active_brand="foton",
        contract=AnswerContract(active_brand="foton", current_question=question, answerability="answer_self"),
        client_message=question,
    )

    assert any(finding.code == "wrong_intent_fact" for finding in findings)


def _semantic_verifier_base_result(text: str, *, route: str = "bot_answer_self_for_pilot") -> SubscriptionDraftResult:
    return SubscriptionDraftResult(
        route=route,
        draft_text=text,
        topic_id="theme:024_advice",
        metadata={
            "dialogue_contract_pipeline": {
                "retrieved_facts": {
                    "program.basic": "Фотон: есть базовый и продвинутый уровень.",
                    "enrollment.remote": "Фотон: оформление проходит дистанционно, менеджер помогает с договором.",
                },
                "retrieved_fact_keys": ["program.basic", "enrollment.remote"],
            }
        },
    )


@pytest.mark.parametrize(
    ("text", "finding", "expected_action"),
    [
        (
            "Курс ОГЭ здесь базовый, он подойдёт для выравнивания пробелов.",
            {"code": "derived_product_claim", "span": "подойдёт для выравнивания пробелов", "relation_to_base": "absent"},
            "downgrade_keep_text",
        ),
        (
            "После оплаты по оферте запись считается подтверждённой.",
            {
                "code": "derived_product_claim",
                "span": "оплата по оферте = подтверждение записи",
                "relation_to_base": "adjacent",
                "nearest_fact_key": "enrollment.remote",
            },
            "downgrade_keep_text",
        ),
        (
            "Обычная группа — это базовый уровень для тех, кто начинает с азов.",
            {"code": "derived_product_claim", "span": "обычная группа — базовый уровень", "relation_to_base": "absent"},
            "downgrade_keep_text",
        ),
        (
            "Обычно за год-два большинство ребят закрывают пробелы.",
            {"code": "invented_generalization", "span": "за год-два большинство ребят", "relation_to_base": "absent"},
            "annotate",
        ),
        (
            "Обычно в очном курсе такие темы разбирают на практике.",
            {"code": "derived_product_claim", "span": "обычно в очном курсе", "relation_to_base": "absent"},
            "downgrade_keep_text",
        ),
    ],
)
def test_semantic_output_verifier_flags_regrade_cases_with_expected_actions(text, finding, expected_action) -> None:
    base = _semantic_verifier_base_result(text)

    checked = apply_semantic_output_verifier(
        base,
        client_message="Подскажите по курсу",
        context={SEMANTIC_OUTPUT_VERIFIER_ENV: True, "active_brand": "foton"},
        verifier_fn=lambda _prompt: {"findings": [finding]},
    )
    gated = apply_authoritative_output_gate(
        checked,
        client_message="Подскажите по курсу",
        context={"active_brand": "foton"},
    )

    assert checked.metadata["semantic_output_verifier"]["checked"] is True
    assert checked.metadata["semantic_output_verifier"]["findings"][0]["code"] == finding["code"]
    assert gated.metadata["authoritative_output_gate"]["action"] == expected_action
    assert gated.draft_text == text
    if expected_action == "downgrade_keep_text":
        assert gated.route == "draft_for_manager"
        assert "authoritative_output_gate_blocked" in gated.safety_flags
        assert f"authoritative_gate:{finding['code']}" in gated.safety_flags
        assert gated.error is None
        assert gated.metadata["semantic_output_verifier"]["fallback_reason"] == SEMANTIC_VERIFIER_DOWNGRADE_REASON
    else:
        assert gated.route == base.route
        assert "authoritative_output_gate_blocked" not in gated.safety_flags
    assert any("Смысловой верификатор" in item for item in gated.manager_checklist)


def test_semantic_output_verifier_keeps_false_cases_and_prompt_controls() -> None:
    prompt = build_semantic_output_verifier_prompt(
        bot_text="Есть базовый и продвинутый уровень.",
        client_message="Есть уровень попроще?",
        facts={"program.basic": "Фотон: есть базовый и продвинутый уровень."},
        active_brand="foton",
        route="bot_answer_self_for_pilot",
    )
    assert "relation_to_base" in prompt
    assert "каноничную фразу разделения брендов" in prompt
    assert "Очный курс физики есть" in prompt
    assert "Олимпиадная физика есть онлайн и очно" in prompt
    assert "Забронирую место на Сретенке" in prompt
    assert "порядок записи не подтверждён" in prompt
    assert "Помогу с оформлением" in prompt
    assert "подберём подходящий вариант" in prompt
    assert "НЕ individual_diagnosis" in prompt
    assert "цена очного формата не подтверждает онлайн-контекст" in prompt

    base = _semantic_verifier_base_result("Есть базовый и продвинутый уровень.")
    checked = apply_semantic_output_verifier(
        base,
        client_message="Есть уровень попроще?",
        context={SEMANTIC_OUTPUT_VERIFIER_ENV: True, "active_brand": "foton"},
        verifier_fn=lambda _prompt: {"findings": []},
    )
    gated = apply_authoritative_output_gate(checked, client_message="Есть уровень попроще?", context={"active_brand": "foton"})

    assert gated.draft_text == base.draft_text
    assert gated.route == base.route
    assert gated.metadata["authoritative_output_gate"]["action"] == "pass"


def test_semantic_output_verifier_price_scope_few_shot_reads_foton_prices_from_kb(tmp_path: Path) -> None:
    snapshot = tmp_path / "kb_release_v3_snapshot.json"
    snapshot.write_text(
        json.dumps(
            {
                "facts": [
                    {
                        "fact_key": "prices_regular_2026_27.offline_5_11_class.before_2026_07_01.semester",
                        "brand": "foton",
                        "allowed_for_client_answer": True,
                        "forbidden_for_client": False,
                        "internal_only": False,
                        "valid_until": "2099-07-01",
                        "client_safe_text": "Фотон: цены на 2026/27 учебный год, 5-11 класс, очно, семестр — 44 600 ₽.",
                    },
                    {
                        "fact_key": "prices_regular_2026_27.offline_5_11_class.before_2026_07_01.year",
                        "brand": "foton",
                        "allowed_for_client_answer": True,
                        "forbidden_for_client": False,
                        "internal_only": False,
                        "valid_until": "2099-07-01",
                        "client_safe_text": "Фотон: цены на 2026/27 учебный год, 5-11 класс, очно, год — 74 500 ₽.",
                    },
                    {
                        "fact_key": "prices_regular_2026_27.offline_5_11_class.before_2026_07_01.semester",
                        "brand": "unpk",
                        "allowed_for_client_answer": True,
                        "valid_until": "2099-07-01",
                        "client_safe_text": "УНПК: цены на 2026/27 учебный год, 5-11 класс, очно, семестр — 49 000 ₽.",
                    },
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    prompt = build_semantic_output_verifier_prompt(
        bot_text="Стоимость онлайн-курса такая же, как очно.",
        client_message="А онлайн?",
        facts={"prices.offline": "Фотон: очные цены есть только для очного формата."},
        active_brand="foton",
        route="bot_answer_self_for_pilot",
        context={"snapshot_path": str(snapshot)},
    )

    assert "Фотон: цены на 2026/27 учебный год, 5-11 класс, очно, семестр — 44 600 ₽." in prompt
    assert "Фотон: цены на 2026/27 учебный год, 5-11 класс, очно, год — 74 500 ₽." in prompt
    assert "Стоимость курса — 44 600 ₽ или 74 500 ₽" in prompt
    assert "49 000 ₽" not in prompt
    assert "82 000 ₽" not in prompt


def test_semantic_output_verifier_price_scope_few_shot_ignores_expired_kb_prices(tmp_path: Path) -> None:
    snapshot = tmp_path / "kb_release_v3_snapshot.json"
    snapshot.write_text(
        json.dumps(
            {
                "facts": [
                    {
                        "fact_key": "prices_regular_2026_27.offline_5_11_class.before_2026_07_01.semester",
                        "brand": "foton",
                        "allowed_for_client_answer": True,
                        "valid_until": "2000-01-01",
                        "client_safe_text": "Фотон: цены на 2026/27 учебный год, 5-11 класс, очно, семестр — 44 600 ₽.",
                    },
                    {
                        "fact_key": "prices_regular_2026_27.offline_5_11_class.before_2026_07_01.year",
                        "brand": "foton",
                        "allowed_for_client_answer": True,
                        "valid_until": "2000-01-01",
                        "client_safe_text": "Фотон: цены на 2026/27 учебный год, 5-11 класс, очно, год — 74 500 ₽.",
                    },
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    prompt = build_semantic_output_verifier_prompt(
        bot_text="Стоимость онлайн-курса такая же, как очно.",
        client_message="А онлайн?",
        facts={"prices.offline": "Фотон: очные цены есть только для очного формата."},
        active_brand="foton",
        route="bot_answer_self_for_pilot",
        context={"snapshot_path": str(snapshot)},
    )

    assert "Ответ переносит очную цену в онлайн-контекст" in prompt
    assert "44 600 ₽" not in prompt
    assert "74 500 ₽" not in prompt
    assert "49 000 ₽" not in prompt
    assert "82 000 ₽" not in prompt


@pytest.mark.parametrize(
    "text",
    [
        "Помогу с оформлением.",
        "Помогу записаться к старту.",
        "Менеджер сверит наличие мест и свяжется с вами.",
        "Подберём подходящий вариант группы.",
    ],
)
def test_semantic_output_verifier_keeps_service_next_steps_cross_model_replay(text: str) -> None:
    base = _semantic_verifier_base_result(text)
    results = []
    for fake_model in (lambda _prompt: {"findings": []}, lambda _prompt: '{"findings":[]}'):
        checked = apply_semantic_output_verifier(
            base,
            client_message="Как записаться?",
            context={SEMANTIC_OUTPUT_VERIFIER_ENV: True, "active_brand": "foton"},
            verifier_fn=fake_model,
        )
        results.append(apply_authoritative_output_gate(checked, client_message="Как записаться?", context={"active_brand": "foton"}))

    assert [item.route for item in results] == [base.route, base.route]
    assert [item.metadata["authoritative_output_gate"]["action"] for item in results] == ["pass", "pass"]
    assert all(item.draft_text == text for item in results)


def test_semantic_output_verifier_keeps_online_price_context_real_finding_cross_model_replay() -> None:
    base = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="Стоимость курса — 44 600 ₽ или 74 500 ₽.",
        topic_id="theme:001_pricing",
        metadata={
            "dialogue_contract_pipeline": {
                "retrieved_facts": {
                    "prices.offline": "Фотон: очные цены 44 600 ₽ и 74 500 ₽; онлайн-цена не указана."
                },
                "retrieved_fact_keys": ["prices.offline"],
            }
        },
    )
    payload = {
        "findings": [
            {
                "code": "derived_product_claim",
                "span": "44 600 ₽ или 74 500 ₽",
                "relation_to_base": "adjacent",
                "nearest_fact_key": "prices.offline",
            }
        ]
    }
    results = []
    for fake_model in (lambda _prompt: payload, lambda _prompt: json.dumps(payload, ensure_ascii=False)):
        checked = apply_semantic_output_verifier(
            base,
            client_message="А онлайн?",
            context={SEMANTIC_OUTPUT_VERIFIER_ENV: True, "active_brand": "foton"},
            verifier_fn=fake_model,
        )
        results.append(apply_authoritative_output_gate(checked, client_message="А онлайн?", context={"active_brand": "foton"}))

    assert [item.route for item in results] == ["draft_for_manager", "draft_for_manager"]
    assert [item.metadata["authoritative_output_gate"]["action"] for item in results] == ["downgrade_keep_text", "downgrade_keep_text"]
    assert all(item.metadata["semantic_output_verifier"]["findings"][0]["relation_to_base"] == "adjacent" for item in results)


def test_semantic_output_regen_prompt_forbids_edit_comments() -> None:
    prompt = build_semantic_output_regen_prompt(
        bot_text="Обычная группа — это базовый уровень.",
        client_message="Есть уровень попроще?",
        facts={"program.basic": "Фотон: есть базовый и продвинутый уровень."},
        findings=[{"code": "derived_product_claim", "span": "базовый уровень"}],
    )

    assert "Верни ТОЛЬКО текст ответа клиенту" in prompt
    assert "Заменяю только этот абзац" in prompt
    assert "Остальной текст без изменений" in prompt


def test_semantic_output_verifier_never_unblocks_deterministic_brand_gate() -> None:
    base = _semantic_verifier_base_result("У Фотона и УНПК одинаковые условия.")

    checked = apply_semantic_output_verifier(
        base,
        client_message="Сравните Фотон и УНПК",
        context={SEMANTIC_OUTPUT_VERIFIER_ENV: True, "active_brand": "foton"},
        verifier_fn=lambda _prompt: {"findings": []},
    )
    gated = apply_authoritative_output_gate(checked, client_message="Сравните Фотон и УНПК", context={"active_brand": "foton"})

    assert gated.route == "manager_only"
    assert gated.draft_text == SAFE_FALLBACK_DRAFT_TEXT
    assert "brand_leak" in {item["code"] for item in gated.metadata["authoritative_output_gate"]["findings"]}


def test_semantic_output_verifier_fail_soft_retries_once_on_timeout() -> None:
    base = _semantic_verifier_base_result("Да, дочка справится.")
    calls = 0

    def timeout(_prompt: str):
        nonlocal calls
        calls += 1
        raise subprocess.TimeoutExpired(cmd=["semantic"], timeout=30)

    checked = apply_semantic_output_verifier(
        base,
        client_message="Дочка справится?",
        context={SEMANTIC_OUTPUT_VERIFIER_ENV: True, "active_brand": "foton"},
        verifier_fn=timeout,
    )

    meta = checked.metadata["semantic_output_verifier"]
    assert calls == 2
    assert meta["unavailable"] is True
    assert meta["retry_attempted"] is True
    assert meta["fallback_reason"] == "semantic_verifier_unavailable"
    assert checked.route == base.route
    assert checked.draft_text == base.draft_text
    assert any("недоступен" in item for item in checked.manager_checklist)


def test_presale_semantic_output_verifier_reports_provider_rc_error() -> None:
    calls = 0

    def failing_runner(cmd, **kwargs):
        nonlocal calls
        calls += 1
        return subprocess.CompletedProcess(cmd, 7, stdout="", stderr="auth failed")

    provider = SubscriptionLlmDraftProvider(runner=failing_runner)
    context = {
        SEMANTIC_OUTPUT_VERIFIER_ENV: True,
        PRESALE_VERIFIER_FAILSOFT_ENV: "1",
        "active_brand": "foton",
    }
    base = _semantic_verifier_base_result("Да, очная группа есть.")

    checked = apply_semantic_output_verifier(
        base,
        client_message="Есть очная группа?",
        context=context,
        verifier_fn=provider._semantic_output_verifier_runner_for_context(context),
    )

    meta = checked.metadata["semantic_output_verifier"]
    assert calls == 2
    assert meta["checked"] is False
    assert meta["unavailable"] is True
    assert "provider_error rc=7" in meta["error"]
    assert checked.route == base.route
    assert checked.draft_text == base.draft_text


def test_semantic_output_verifier_absorbs_diagnosis_cases_any_route_and_hedged_false_case() -> None:
    substantive = _semantic_verifier_base_result(
        "По таким вводным слишком тяжело быть не должно: ритм посильный.",
        route="manager_only",
    )
    checked = apply_semantic_output_verifier(
        substantive,
        client_message="Дочке не будет тяжело?",
        context={SEMANTIC_OUTPUT_VERIFIER_ENV: True, "active_brand": "foton"},
        verifier_fn=lambda _prompt: {
            "findings": [
                {
                    "code": "individual_diagnosis",
                    "span": "слишком тяжело быть не должно",
                    "relation_to_base": "absent",
                }
            ]
        },
    )
    gated = apply_authoritative_output_gate(checked, client_message="Дочке не будет тяжело?", context={"active_brand": "foton"})
    assert checked.metadata["semantic_output_verifier"]["checked"] is True
    assert gated.route == "manager_only"
    assert gated.draft_text == substantive.draft_text
    assert gated.metadata["authoritative_output_gate"]["action"] == "downgrade_keep_text"

    hedged = _semantic_verifier_base_result(
        "Заочно не буду обещать: уровень лучше сверить с преподавателем, менеджер поможет подобрать группу."
    )
    hedged_checked = apply_semantic_output_verifier(
        hedged,
        client_message="Дочка справится?",
        context={SEMANTIC_OUTPUT_VERIFIER_ENV: True, "active_brand": "foton"},
        verifier_fn=lambda _prompt: {"findings": [{"code": "individual_diagnosis", "span": "уровень лучше сверить"}]},
    )
    assert hedged_checked.metadata["semantic_output_verifier"]["findings"] == []


def test_semantic_output_verifier_skips_only_locked_or_pure_handoff_texts() -> None:
    pure = SubscriptionDraftResult(
        route="manager_only",
        draft_text="Приняли обращение, передам менеджеру.",
        safety_flags=("high_risk_manager_only",),
    )
    calls = 0

    def verifier(_prompt: str):
        nonlocal calls
        calls += 1
        return {"findings": [{"code": "individual_diagnosis"}]}

    checked = apply_semantic_output_verifier(
        pure,
        client_message="Верните деньги, ребёнок не справится",
        context={SEMANTIC_OUTPUT_VERIFIER_ENV: True, "active_brand": "foton"},
        verifier_fn=verifier,
    )

    assert calls == 0
    assert checked.metadata["semantic_output_verifier"]["skipped"] is True


def test_semantic_output_verifier_skips_service_handoff_without_factual_claim() -> None:
    base = SubscriptionDraftResult(
        route="draft_for_manager",
        draft_text="Помогу с оформлением: менеджер сверит детали и свяжется.",
        topic_id="theme:020_enrollment",
    )

    def verifier(_prompt: str):
        raise AssertionError("service-only handoff must not call semantic verifier")

    checked = apply_semantic_output_verifier(
        base,
        client_message="Как оформить?",
        context={SEMANTIC_OUTPUT_VERIFIER_ENV: True, "active_brand": "foton"},
        verifier_fn=verifier,
    )

    assert checked.metadata["semantic_output_verifier"]["skipped"] is True
    assert checked.metadata["semantic_output_verifier"]["skip_reason"] == "pure_handoff"


def test_wave1_verifier_handoff_claims_off_keeps_current_pure_handoff_skip() -> None:
    base = SubscriptionDraftResult(
        route="draft_for_manager",
        draft_text="У нас сильные преподаватели, передам менеджеру.",
        topic_id="theme:016_program",
    )
    calls = 0

    def verifier(_prompt: str):
        nonlocal calls
        calls += 1
        return {"findings": [{"code": "derived_product_claim"}]}

    checked = apply_semantic_output_verifier(
        base,
        client_message="Сильные преподаватели?",
        context={SEMANTIC_OUTPUT_VERIFIER_ENV: True, VERIFIER_HANDOFF_CLAIMS_ENV: "0", "active_brand": "foton"},
        verifier_fn=verifier,
    )

    assert calls == 0
    assert checked.metadata["semantic_output_verifier"]["skipped"] is True
    assert checked.metadata["semantic_output_verifier"]["skip_reason"] == "pure_handoff"


def test_wave1_verifier_handoff_claims_on_keeps_canonical_template_skipped() -> None:
    base = SubscriptionDraftResult(
        route="draft_for_manager",
        draft_text=SAFE_FALLBACK_DRAFT_TEXT,
        topic_id="theme:001_pricing",
    )

    def verifier(_prompt: str):
        raise AssertionError("canonical pure handoff must stay skipped")

    checked = apply_semantic_output_verifier(
        base,
        client_message="Сколько стоит?",
        context={SEMANTIC_OUTPUT_VERIFIER_ENV: True, VERIFIER_HANDOFF_CLAIMS_ENV: "1", "active_brand": "foton"},
        verifier_fn=verifier,
    )

    assert checked.metadata["semantic_output_verifier"]["skipped"] is True
    assert checked.metadata["semantic_output_verifier"]["skip_reason"] == "pure_handoff"


def test_wave1_verifier_handoff_claims_on_checks_substantive_handoff() -> None:
    base = SubscriptionDraftResult(
        route="draft_for_manager",
        draft_text="У нас сильные преподаватели, передам менеджеру.",
        topic_id="theme:016_program",
    )
    calls = 0

    def verifier(_prompt: str):
        nonlocal calls
        calls += 1
        return {
            "findings": [
                {
                    "code": "derived_product_claim",
                    "span": "сильные преподаватели",
                    "relation_to_base": "absent",
                }
            ]
        }

    checked = apply_semantic_output_verifier(
        base,
        client_message="Сильные преподаватели?",
        context={SEMANTIC_OUTPUT_VERIFIER_ENV: True, VERIFIER_HANDOFF_CLAIMS_ENV: "1", "active_brand": "foton"},
        verifier_fn=verifier,
    )
    gated = apply_authoritative_output_gate(checked, client_message="Сильные преподаватели?", context={"active_brand": "foton"})

    assert calls == 1
    assert checked.metadata["semantic_output_verifier"]["checked"] is True
    assert checked.metadata["semantic_output_verifier"]["finding_codes"] == ["derived_product_claim"]
    assert gated.route == "draft_for_manager"
    assert gated.metadata["authoritative_output_gate"]["action"] == "downgrade_keep_text"


def test_wave1_verifier_handoff_claims_on_keeps_p0_and_brand_gates() -> None:
    p0 = SubscriptionDraftResult(
        route="manager_only",
        draft_text="Приняли обращение, передам менеджеру.",
        topic_id="theme:009_refund",
        safety_flags=("high_risk_manager_only",),
    )

    p0_checked = apply_semantic_output_verifier(
        p0,
        client_message="Верните деньги, ребёнок не справится",
        context={SEMANTIC_OUTPUT_VERIFIER_ENV: True, VERIFIER_HANDOFF_CLAIMS_ENV: "1", "active_brand": "foton"},
        verifier_fn=lambda _prompt: {"findings": [{"code": "derived_product_claim"}]},
    )
    brand_checked = apply_semantic_output_verifier(
        _semantic_verifier_base_result("У Фотона и УНПК одинаковые условия."),
        client_message="Сравните Фотон и УНПК",
        context={SEMANTIC_OUTPUT_VERIFIER_ENV: True, VERIFIER_HANDOFF_CLAIMS_ENV: "1", "active_brand": "foton"},
        verifier_fn=lambda _prompt: {"findings": []},
    )
    brand_gated = apply_authoritative_output_gate(brand_checked, client_message="Сравните Фотон и УНПК", context={"active_brand": "foton"})

    assert p0_checked.metadata["semantic_output_verifier"]["skip_reason"] == "locked_p0_or_high_risk_deferral"
    assert brand_gated.route == "manager_only"
    assert "brand_leak" in {item["code"] for item in brand_gated.metadata["authoritative_output_gate"]["findings"]}


def test_semantic_output_verifier_checks_handoff_with_factual_claim_sentence() -> None:
    base = SubscriptionDraftResult(
        route="draft_for_manager",
        draft_text="Передам вопрос менеджеру, он сверит детали. Обычно в очном курсе такие темы разбирают на практике.",
        topic_id="theme:016_program",
    )
    calls = 0

    def verifier(_prompt: str):
        nonlocal calls
        calls += 1
        return {
            "findings": [
                {
                    "code": "derived_product_claim",
                    "span": "обычно в очном курсе",
                    "relation_to_base": "absent",
                }
            ]
        }

    checked = apply_semantic_output_verifier(
        base,
        client_message="Как идут занятия?",
        context={SEMANTIC_OUTPUT_VERIFIER_ENV: True, "active_brand": "foton"},
        verifier_fn=verifier,
    )

    assert calls == 1
    assert checked.metadata["semantic_output_verifier"]["checked"] is True
    assert checked.metadata["semantic_output_verifier"]["findings"][0]["code"] == "derived_product_claim"


def test_semantic_output_verifier_regen_once_then_full_gate_runs_with_context() -> None:
    base = _semantic_verifier_base_result("Обычная группа — это базовый уровень.", route="draft_for_manager")
    verifier_calls = 0

    def verifier(_prompt: str):
        nonlocal verifier_calls
        verifier_calls += 1
        if verifier_calls == 1:
            return {"findings": [{"code": "derived_product_claim", "span": "базовый уровень"}]}
        return {"findings": []}

    regen_calls = 0

    def regen(_prompt: str) -> str:
        nonlocal regen_calls
        regen_calls += 1
        return "УНПК: есть базовый уровень."

    checked = apply_semantic_output_verifier(
        base,
        client_message="Есть уровень попроще?",
        context={SEMANTIC_OUTPUT_VERIFIER_ENV: True, "active_brand": "foton"},
        verifier_fn=verifier,
        regen_fn=regen,
    )
    gated = apply_authoritative_output_gate(checked, client_message="Есть уровень попроще?", context={"active_brand": "foton"})

    assert verifier_calls == 2
    assert regen_calls == 1
    assert checked.metadata["semantic_output_verifier"]["regen_attempted"] is True
    assert checked.metadata["semantic_output_verifier"]["regen_accepted"] is True
    assert gated.draft_text == SAFE_FALLBACK_DRAFT_TEXT
    assert "brand_leak" in {item["code"] for item in gated.metadata["authoritative_output_gate"]["findings"]}


def test_semantic_output_verifier_regens_autonomous_text_but_keeps_manager_route() -> None:
    base = _semantic_verifier_base_result("Обычная группа — это базовый уровень.", route="bot_answer_self_for_pilot")
    verifier_calls = 0

    def verifier(_prompt: str):
        nonlocal verifier_calls
        verifier_calls += 1
        if verifier_calls == 1:
            return {"findings": [{"code": "derived_product_claim", "span": "базовый уровень"}]}
        return {"findings": []}

    def regen(_prompt: str) -> str:
        return "Заочно не буду обещать уровень: менеджер поможет подобрать подходящую группу."

    checked = apply_semantic_output_verifier(
        base,
        client_message="Есть уровень попроще?",
        context={SEMANTIC_OUTPUT_VERIFIER_ENV: True, "active_brand": "foton"},
        verifier_fn=verifier,
        regen_fn=regen,
    )
    gated = apply_authoritative_output_gate(checked, client_message="Есть уровень попроще?", context={"active_brand": "foton"})

    assert verifier_calls == 2
    assert checked.metadata["semantic_output_verifier"]["regen_attempted"] is True
    assert checked.metadata["semantic_output_verifier"]["regen_accepted"] is True
    assert checked.route == "draft_for_manager"
    assert gated.route == "draft_for_manager"
    assert gated.draft_text == "Заочно не буду обещать уровень: менеджер поможет подобрать подходящую группу."


def test_semantic_output_verifier_cross_model_replay_fixture_is_consistent() -> None:
    base = _semantic_verifier_base_result("После оплаты по оферте запись считается подтверждённой.")
    payload = {
        "findings": [
            {
                "code": "derived_product_claim",
                "span": "запись считается подтверждённой",
                "relation_to_base": "adjacent",
                "nearest_fact_key": "enrollment.remote",
            }
        ]
    }
    results = []
    for fake_model in (lambda _prompt: payload, lambda _prompt: json.dumps(payload, ensure_ascii=False)):
        checked = apply_semantic_output_verifier(
            base,
            client_message="Как записаться?",
            context={SEMANTIC_OUTPUT_VERIFIER_ENV: True, "active_brand": "foton"},
            verifier_fn=fake_model,
        )
        results.append(apply_authoritative_output_gate(checked, client_message="Как записаться?", context={"active_brand": "foton"}))

    assert [item.route for item in results] == ["draft_for_manager", "draft_for_manager"]
    assert [item.metadata["authoritative_output_gate"]["action"] for item in results] == ["downgrade_keep_text", "downgrade_keep_text"]


def test_semantic_diagnosis_guard_rewrites_claude_paraphrase_real_text() -> None:
    base = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="Да, с тройками можно идти: сын сможет влиться в группу, отдельно догонять заранее не нужно.",
        topic_id="theme:024_advice",
        safety_flags=("rules_engine_phase2_anxiety_level_fit",),
    )

    result = apply_semantic_diagnosis_guard(
        base,
        client_message="У сына тройки, сможет ли он влиться?",
        context={
            "active_brand": "unpk",
            "semantic_diagnosis_guard_enabled": True,
            "semantic_diagnosis_classifier_fn": lambda _prompt: {
                "individual_diagnosis": True,
                "span": "сын сможет влиться",
                "reason": "уверенная оценка конкретного ребёнка",
            },
        },
    )

    text = result.draft_text.casefold()
    assert result.route == "bot_answer_self_for_pilot"
    assert "semantic_diagnosis_guard_rewritten" in result.safety_flags
    assert "сможет влиться" not in text
    assert "с тройками можно идти" not in text
    assert "заочно не буду оценивать" in text
    assert "преподавател" in text
    assert "менеджер" in text
    assert result.metadata["semantic_diagnosis_guard"]["individual_diagnosis"] is True
    assert result.metadata["semantic_diagnosis_guard"]["rewritten"] is True
    gated = apply_authoritative_output_gate(
        result,
        client_message="У сына тройки, сможет ли он влиться?",
        context={"active_brand": "unpk"},
    )
    assert gated.draft_text == result.draft_text
    assert "authoritative_output_gate_blocked" not in gated.safety_flags


def test_semantic_diagnosis_guard_rewrites_manager_only_substantive_real_text() -> None:
    base = SubscriptionDraftResult(
        route="manager_only",
        draft_text=(
            "По таким вводным слишком тяжело быть не должно: ритм посильный, "
            "а группу подберут под ребёнка."
        ),
        topic_id="theme:024_advice",
        safety_flags=("high_risk_manager_only",),
    )
    calls: list[str] = []

    def classifier(prompt: str) -> dict[str, object]:
        calls.append(prompt)
        return {
            "individual_diagnosis": True,
            "span": "слишком тяжело быть не должно",
            "reason": "косвенная оценка нагрузки конкретного ребёнка",
        }

    result = apply_semantic_diagnosis_guard(
        base,
        client_message="Дочка тревожится, ей не будет слишком тяжело?",
        context={
            "active_brand": "foton",
            "semantic_diagnosis_guard_enabled": True,
            "semantic_diagnosis_classifier_fn": classifier,
        },
    )

    text = result.draft_text.casefold()
    assert calls, "classifier must run for substantive manager_only drafts"
    assert "слишком тяжело" not in text
    assert "посильный ритм" not in text
    assert "подберут под ребёнка" not in text
    assert result.route == "manager_only"
    assert "semantic_diagnosis_guard_rewritten" in result.safety_flags
    assert result.metadata["semantic_diagnosis_guard"]["checked"] is True
    assert result.metadata["semantic_diagnosis_guard"]["rewritten"] is True


def test_semantic_diagnosis_guard_keeps_general_program_info_false_case() -> None:
    base = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="На платформе есть базовый уровень — он для тех, кто начинает с азов.",
        topic_id="theme:024_advice",
        safety_flags=("rules_engine_phase2_anxiety_capability",),
    )

    result = apply_semantic_diagnosis_guard(
        base,
        client_message="Есть уровень попроще?",
        context={
            "active_brand": "foton",
            "semantic_diagnosis_guard_enabled": True,
            "semantic_diagnosis_classifier_fn": lambda _prompt: {
                "individual_diagnosis": False,
                "span": "",
                "reason": "общая справка",
            },
        },
    )

    assert result.draft_text == base.draft_text
    assert "semantic_diagnosis_guard_rewritten" not in result.safety_flags
    assert result.metadata["semantic_diagnosis_guard"]["fallback_reason"] == "not_individual_diagnosis"


def test_semantic_diagnosis_guard_keeps_manager_only_general_info_false_case() -> None:
    base = SubscriptionDraftResult(
        route="manager_only",
        draft_text="Есть базовый уровень и формат мини-группы; менеджер поможет подобрать подходящую группу.",
        topic_id="theme:024_advice",
        safety_flags=("draft_for_manager",),
    )
    called = False

    def classifier(_prompt: str) -> dict[str, object]:
        nonlocal called
        called = True
        return {"individual_diagnosis": False, "span": "", "reason": "общая справка"}

    result = apply_semantic_diagnosis_guard(
        base,
        client_message="Есть уровень попроще?",
        context={
            "active_brand": "foton",
            "semantic_diagnosis_guard_enabled": True,
            "semantic_diagnosis_classifier_fn": classifier,
        },
    )

    assert called is True
    assert result.draft_text == base.draft_text
    assert result.route == "manager_only"
    assert "semantic_diagnosis_guard_rewritten" not in result.safety_flags
    assert result.metadata["semantic_diagnosis_guard"]["fallback_reason"] == "not_individual_diagnosis"


def test_semantic_diagnosis_guard_keeps_already_hedged_transfer() -> None:
    base = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="Уровень лучше сверить на пробном занятии: преподаватель сориентирует, а менеджер поможет подобрать группу.",
        topic_id="theme:024_advice",
    )

    result = apply_semantic_diagnosis_guard(
        base,
        client_message="Дочка справится?",
        context={
            "active_brand": "foton",
            "semantic_diagnosis_guard_enabled": True,
            "semantic_diagnosis_classifier_fn": lambda _prompt: {
                "individual_diagnosis": True,
                "span": "уровень лучше сверить",
                "reason": "модель перестраховалась",
            },
        },
    )

    assert result.draft_text == base.draft_text
    assert "semantic_diagnosis_guard_rewritten" not in result.safety_flags
    assert result.metadata["semantic_diagnosis_guard"]["fallback_reason"] == "already_hedged_and_transferred"


def test_semantic_diagnosis_guard_fail_soft_on_classifier_error() -> None:
    base = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="Да, дочка справится.",
        topic_id="theme:024_advice",
    )

    def broken(_prompt: str):
        raise RuntimeError("classifier down")

    result = apply_semantic_diagnosis_guard(
        base,
        client_message="Дочка справится?",
        context={
            "active_brand": "foton",
            "semantic_diagnosis_guard_enabled": True,
            "semantic_diagnosis_classifier_fn": broken,
        },
    )

    assert result.draft_text == base.draft_text
    assert "semantic_diagnosis_guard_rewritten" not in result.safety_flags
    assert result.metadata["semantic_diagnosis_guard"]["fallback_reason"] == "classifier_error"


def test_semantic_diagnosis_guard_does_not_touch_p0_manager_only() -> None:
    base = SubscriptionDraftResult(
        route="manager_only",
        draft_text="Приняли обращение, передам менеджеру.",
        topic_id="theme:009_refund",
        safety_flags=("high_risk_manager_only",),
    )
    called = False

    def classifier(_prompt: str):
        nonlocal called
        called = True
        return {"individual_diagnosis": True}

    result = apply_semantic_diagnosis_guard(
        base,
        client_message="Верните деньги, ребёнок не справится",
        context={
            "active_brand": "foton",
            "semantic_diagnosis_guard_enabled": True,
            "semantic_diagnosis_classifier_fn": classifier,
        },
    )

    assert result.route == "manager_only"
    assert result.draft_text == base.draft_text
    assert called is False
    assert result.metadata["semantic_diagnosis_guard"]["fallback_reason"] == "locked_p0_or_high_risk_deferral"


def test_semantic_diagnosis_prompt_contains_true_false_controls() -> None:
    prompt = build_semantic_diagnosis_prompt(
        client_message="С тройками можно?",
        bot_text="Да, с тройками можно идти.",
    )

    assert "с тройками можно идти" in prompt
    assert "слишком тяжело быть не должно" in prompt
    assert "посильный ритм" in prompt
    assert "есть базовый и продвинутый уровень" in prompt
    assert "Верни СТРОГО JSON" in prompt


def test_a_thread_context_carries_only_current_selling_slots_without_brand_override() -> None:
    contract = {
        "current_question": "А очно тогда сколько?",
        "planner_slots": {},
        "known_slots": {},
    }
    context = {
        "active_brand": "foton",
        "TELEGRAM_A_THREAD": True,
        "dialogue_memory_view": {
            "known_slots": {"grade": {"value": "10"}, "format": {"value": "онлайн"}},
            "topic_focus": {"subject": "информатика", "format": "онлайн", "active_brand": "unpk"},
        },
    }

    threaded = _context_with_selling_thread_slots(context, contract=contract, client_message="А очно тогда сколько?")
    off = _context_with_selling_thread_slots({**context, "TELEGRAM_A_THREAD": False}, contract=contract, client_message="А очно тогда сколько?")

    assert threaded is not None
    assert threaded["selling_thread_slots"]["grade"] == "10"
    assert threaded["selling_thread_slots"]["subject"] == "информатика"
    assert threaded["selling_thread_slots"]["format"] == "очно"
    assert threaded["selling_thread_slots"]["active_brand"] == "foton"
    assert off == {**context, "TELEGRAM_A_THREAD": False}


def test_step4_phase2_demolition_registry_keeps_only_safety_specs() -> None:
    names = {spec.name for spec in DIALOGUE_CONTRACT_V2_TEMPLATE_REGISTRY}

    assert names == {"cross_brand", "terminal", "result_guarantee", "admission_guarantee"}
    assert not {
        "matkap",
        "tax",
        "olympiad_online",
        "trial",
    }.intersection(names)


class _DirectPathProvider(SubscriptionLlmDraftProvider):
    def __init__(self, result: SubscriptionDraftResult) -> None:
        super().__init__()
        self.result = result
        self.calls = 0
        self.last_prompt = ""

    def _direct_path_draft_runner(self, prompt: str) -> SubscriptionDraftResult:
        self.calls += 1
        self.last_prompt = prompt
        return self.result


class _DirectPathSequenceProvider(SubscriptionLlmDraftProvider):
    def __init__(self, *results: SubscriptionDraftResult | Exception) -> None:
        super().__init__()
        self.results = list(results)
        self.calls = 0
        self.prompts: list[str] = []

    def _direct_path_draft_runner(self, prompt: str) -> SubscriptionDraftResult:
        self.calls += 1
        self.prompts.append(prompt)
        if not self.results:
            raise AssertionError("unexpected direct path draft call")
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class _DirectPathRetrieverProvider(_DirectPathProvider):
    def __init__(self, result: SubscriptionDraftResult, retriever_payload: Mapping[str, object] | Exception) -> None:
        super().__init__(result)
        self.retriever_payload = retriever_payload
        self.retriever_calls = 0
        self.last_retriever_prompt = ""

    def _direct_path_llm_retrieve_runner(self, prompt: str) -> Mapping[str, object] | str:
        self.retriever_calls += 1
        self.last_retriever_prompt = prompt
        if isinstance(self.retriever_payload, Exception):
            raise self.retriever_payload
        return self.retriever_payload


class _DirectPathShadowProvider(_DirectPathProvider):
    def __init__(self, result: SubscriptionDraftResult, shadow_payload: Mapping[str, object] | Exception) -> None:
        super().__init__(result)
        self.shadow_payload = shadow_payload
        self.shadow_calls = 0
        self.last_shadow_prompt = ""

    def _direct_path_slot_topic_shadow_runner(self, prompt: str) -> Mapping[str, object] | str:
        self.shadow_calls += 1
        self.last_shadow_prompt = prompt
        if isinstance(self.shadow_payload, Exception):
            raise self.shadow_payload
        return self.shadow_payload


def test_tz137_slot_topic_shadow_default_off_does_not_call_runner() -> None:
    provider = _DirectPathShadowProvider(
        SubscriptionDraftResult(route="bot_answer_self_for_pilot", draft_text="Подскажите класс ученика."),
        shadow_payload=AssertionError("shadow runner must not be called with flag OFF"),
    )

    result = provider.build_draft(
        "А по истории?",
        context={"active_brand": "foton", DIRECT_PATH_ENV: "1"},
    )

    assert provider.shadow_calls == 0
    assert result.route == "bot_answer_self_for_pilot"
    assert "slot_topic_shadow" not in result.metadata["direct_path"]


def test_tz137_slot_topic_shadow_on_logs_metadata_without_prompt_or_output_diff() -> None:
    base_result = SubscriptionDraftResult(route="bot_answer_self_for_pilot", draft_text="Подскажите класс ученика.")
    off_provider = _DirectPathProvider(base_result)
    on_provider = _DirectPathShadowProvider(
        base_result,
        shadow_payload={
            "model_slots": {"grade": "11", "subject": "история", "format": "онлайн"},
            "model_topic": "pricing",
            "evidence_quote": "по истории для 11",
            "confidence": 0.81,
        },
    )
    context = {
        "active_brand": "foton",
        DIRECT_PATH_ENV: "1",
        "conversation_intent_plan": {"known_slots": {"grade": "10"}, "primary_intent": "pricing"},
    }

    off = off_provider.build_draft("А по истории для 11?", context=context)
    on = on_provider.build_draft(
        "А по истории для 11?",
        context={**context, subscription_llm.DIRECT_SLOT_TOPIC_SHADOW_ENV: "1"},
    )
    off_meta = dict(off.metadata["direct_path"])
    on_meta = dict(on.metadata["direct_path"])
    shadow = on_meta.pop("slot_topic_shadow")

    assert on_provider.shadow_calls == 1
    assert on.route == off.route
    assert on.draft_text == off.draft_text
    assert on_provider.last_prompt == off_provider.last_prompt
    assert on_meta == off_meta
    assert shadow["used"] is True
    assert shadow["model_slots"]["subject"] == "история"
    assert shadow["plan_primary_intent"] == "pricing"


def test_tz137_slot_topic_shadow_fail_soft_keeps_output() -> None:
    provider = _DirectPathShadowProvider(
        SubscriptionDraftResult(route="bot_answer_self_for_pilot", draft_text="Подскажите класс ученика."),
        shadow_payload=subprocess.TimeoutExpired(cmd="shadow", timeout=1),
    )

    result = provider.build_draft(
        "А по истории?",
        context={
            "active_brand": "foton",
            DIRECT_PATH_ENV: "1",
            subscription_llm.DIRECT_SLOT_TOPIC_SHADOW_ENV: "1",
        },
    )

    assert result.route == "bot_answer_self_for_pilot"
    assert result.draft_text == "Подскажите класс ученика."
    assert result.metadata["direct_path"]["slot_topic_shadow"]["fallback_reason"] == "timeout"


def test_direct_path_bot_safe_memory_step_guard_runs_without_semantic_verifier(tmp_path: Path) -> None:
    snapshot_path = _write_wave6_snapshot(tmp_path)
    provider = _DirectPathProvider(
        SubscriptionDraftResult(
            route="bot_answer_self_for_pilot",
            draft_text="Да, место уже забронировано, заявка подтверждена.",
            metadata={"direct_path": {"model_response": "raw"}},
            safety_flags=(),
        )
    )

    result = provider.build_draft(
        "Что дальше с записью?",
        context={
            "active_brand": "foton",
            DIRECT_PATH_ENV: "1",
            "snapshot_path": str(snapshot_path),
            "TELEGRAM_BOT_SAFE_CRM_CONTEXT": "1",
            "TELEGRAM_BOT_SAFE_MEMORY_STEP_GUARD": "1",
            "timeline_context": {
                "source": "customer_timeline_bot_context",
                "found": True,
                "bot_context": {
                    "allowed_only": True,
                    "items": [
                        {
                            "chunk_id": "chunk-foton",
                            "chunk_type": "bot_safe_summary",
                            "text": "Фотон: клиент обсуждал запись.",
                            "next_step_status": "needs_manager_review",
                            "relevance_tags": ["bot_safe", "structured", "foton"],
                            "allowed_for_bot": True,
                            "requires_manager_review": False,
                        }
                    ],
                },
            },
        },
    )

    assert provider.calls == 1
    assert result.route == "draft_for_manager"
    assert "Уточню актуальный шаг с менеджером" in result.draft_text
    assert "забронировано" not in result.draft_text
    assert "bot_safe_memory_unconfirmed_step_detected" in result.safety_flags
    assert result.metadata["bot_safe_memory_step_guard"]["applied"] is True
    assert result.metadata["authoritative_output_gate"]["checked"] is True


def test_direct_path_bot_safe_memory_step_guard_is_noop_when_memory_off(tmp_path: Path) -> None:
    snapshot_path = _write_wave6_snapshot(tmp_path)
    provider = _DirectPathProvider(
        SubscriptionDraftResult(
            route="bot_answer_self_for_pilot",
            draft_text="Да, место уже забронировано, заявка подтверждена.",
            metadata={"direct_path": {"model_response": "raw"}},
            safety_flags=(),
        )
    )

    result = provider.build_draft(
        "Что дальше с записью?",
        context={
            "active_brand": "foton",
            DIRECT_PATH_ENV: "1",
            "snapshot_path": str(snapshot_path),
            "TELEGRAM_BOT_SAFE_CRM_CONTEXT": "0",
            "timeline_context": {
                "source": "customer_timeline_bot_context",
                "found": True,
                "bot_context": {
                    "allowed_only": True,
                    "items": [
                        {
                            "chunk_id": "chunk-foton",
                            "chunk_type": "bot_safe_summary",
                            "text": "Фотон: клиент обсуждал запись.",
                            "next_step_status": "needs_manager_review",
                            "relevance_tags": ["bot_safe", "structured", "foton"],
                            "allowed_for_bot": True,
                            "requires_manager_review": False,
                        }
                    ],
                },
            },
        },
    )

    assert provider.calls == 1
    assert result.draft_text == "Да, место уже забронировано, заявка подтверждена."
    assert "bot_safe_memory_unconfirmed_step_detected" not in result.safety_flags
    assert "bot_safe_memory_step_guard" not in result.metadata


def test_direct_path_rewrites_unconfirmed_contact_data_claim(tmp_path: Path) -> None:
    snapshot_path = _write_wave6_snapshot(tmp_path)
    provider = _DirectPathProvider(
        SubscriptionDraftResult(
            route="bot_answer_self_for_pilot",
            draft_text="Телефон повторно присылать не нужно, он уже есть в диалоге.",
            safety_flags=(),
        )
    )

    result = provider.build_draft(
        "Хочу записаться на курс",
        context={
            "active_brand": "foton",
            DIRECT_PATH_ENV: "1",
            "snapshot_path": str(snapshot_path),
            "recent_messages": ["Клиент: Хочу записаться на курс"],
        },
    )

    assert provider.calls == 1
    assert result.route == "bot_answer_self_for_pilot"
    assert result.draft_text == "Повторно указывать не обязательно — менеджер сверит по системе."
    assert "unconfirmed_contact_data_claim_rewritten" in result.safety_flags


def test_direct_path_keeps_contact_claim_when_client_sent_phone(tmp_path: Path) -> None:
    snapshot_path = _write_wave6_snapshot(tmp_path)
    provider = _DirectPathProvider(
        SubscriptionDraftResult(
            route="bot_answer_self_for_pilot",
            draft_text="Телефон повторно присылать не нужно, он уже есть в диалоге.",
            safety_flags=(),
        )
    )

    result = provider.build_draft(
        "Мой телефон +7 999 123-45-67",
        context={
            "active_brand": "foton",
            DIRECT_PATH_ENV: "1",
            "snapshot_path": str(snapshot_path),
            "recent_messages": ["Клиент: Мой телефон +7 999 123-45-67"],
        },
    )

    assert provider.calls == 1
    assert result.draft_text == "Телефон повторно присылать не нужно, он уже есть в диалоге."
    assert "unconfirmed_contact_data_claim_rewritten" not in result.safety_flags


def test_direct_path_rewrites_no_memory_better_start_frame(tmp_path: Path) -> None:
    snapshot_path = _write_wave6_snapshot(tmp_path)
    provider = _DirectPathProvider(
        SubscriptionDraftResult(
            route="bot_answer_self_for_pilot",
            draft_text="Лучше начать с класса ученика, чтобы подобрать группу.",
            safety_flags=(),
        )
    )

    result = provider.build_draft(
        "Что нужно для подбора?",
        context={
            "active_brand": "foton",
            DIRECT_PATH_ENV: "1",
            "snapshot_path": str(snapshot_path),
        },
    )

    assert provider.calls == 1
    assert "лучше начать" not in result.draft_text.casefold()
    assert "Предлагаю начать с класса ученика" in result.draft_text
    assert "no_memory_step_frame_rewritten" in result.safety_flags


def test_direct_path_rewrites_no_memory_next_step_synonym_frame(tmp_path: Path) -> None:
    snapshot_path = _write_wave6_snapshot(tmp_path)
    provider = _DirectPathProvider(
        SubscriptionDraftResult(
            route="bot_answer_self_for_pilot",
            draft_text=(
                "Дальше нужно подобрать онлайн-группу по уровню для 7 класса по математике. "
                "Онлайн-занятия в Фотоне проходят на SohoLMS."
            ),
            safety_flags=(),
        )
    )

    result = provider.build_draft(
        "Онлайн удобнее. Что дальше нужно сделать?",
        context={
            "active_brand": "foton",
            DIRECT_PATH_ENV: "1",
            "snapshot_path": str(snapshot_path),
        },
    )

    assert provider.calls == 1
    assert result.route == "bot_answer_self_for_pilot"
    assert "дальше нужно" not in result.draft_text.casefold()
    assert "Уточните, пожалуйста" in result.draft_text
    assert "класс ученика" in result.draft_text
    assert "предмет" in result.draft_text
    assert "формат" in result.draft_text
    assert "уровень подготовки" in result.draft_text
    assert "no_memory_step_frame_rewritten" in result.safety_flags
    assert "manager_approval_required" in result.safety_flags
    assert "no_auto_send" in result.safety_flags


def test_direct_path_keeps_no_memory_neutral_wait_frame(tmp_path: Path) -> None:
    snapshot_path = _write_wave6_snapshot(tmp_path)
    provider = _DirectPathProvider(
        SubscriptionDraftResult(
            route="bot_answer_self_for_pilot",
            draft_text="Дальше нужно дождаться ответа менеджера, чтобы не ошибиться.",
            safety_flags=(),
        )
    )

    result = provider.build_draft(
        "Что дальше?",
        context={
            "active_brand": "foton",
            DIRECT_PATH_ENV: "1",
            "snapshot_path": str(snapshot_path),
        },
    )

    assert provider.calls == 1
    assert result.draft_text == "Дальше нужно дождаться ответа менеджера, чтобы не ошибиться."
    assert "no_memory_step_frame_rewritten" not in result.safety_flags


def test_direct_path_does_not_rewrite_no_memory_payment_frame_as_step_question(tmp_path: Path) -> None:
    snapshot_path = _write_wave6_snapshot(tmp_path)
    provider = _DirectPathProvider(
        SubscriptionDraftResult(
            route="bot_answer_self_for_pilot",
            draft_text="Дальше нужно оплатить курс до завтра.",
            safety_flags=(),
        )
    )

    result = provider.build_draft(
        "Что дальше?",
        context={
            "active_brand": "foton",
            DIRECT_PATH_ENV: "1",
            "snapshot_path": str(snapshot_path),
        },
    )

    assert provider.calls == 1
    assert "no_memory_step_frame_rewritten" not in result.safety_flags
    assert "Уточните, пожалуйста, оплатить курс" not in result.draft_text


def test_direct_path_memory_step_guard_rewrites_synonym_frame_for_review_status(tmp_path: Path) -> None:
    snapshot_path = _write_wave6_snapshot(tmp_path)
    provider = _DirectPathProvider(
        SubscriptionDraftResult(
            route="bot_answer_self_for_pilot",
            draft_text=(
                "Следующий шаг — понять класс ребёнка, чтобы подобрать подходящую онлайн-группу "
                "по 4 предметам. Подскажите, пожалуйста, в каком классе ребёнок?"
            ),
            safety_flags=(),
        )
    )

    result = provider.build_draft(
        "Тогда какой сейчас следующий шаг?",
        context={
            "active_brand": "foton",
            DIRECT_PATH_ENV: "1",
            "snapshot_path": str(snapshot_path),
            "TELEGRAM_BOT_SAFE_CRM_CONTEXT": "1",
            "TELEGRAM_BOT_SAFE_MEMORY_STEP_GUARD": "1",
            "timeline_context": {
                "source": "customer_timeline_bot_context",
                "found": True,
                "bot_context": {
                    "allowed_only": True,
                    "items": [
                        {
                            "chunk_id": "chunk-foton",
                            "chunk_type": "bot_safe_summary",
                            "text": "Бренд: Фотон. Следующий шаг требует проверки менеджером.",
                            "next_step_status": "needs_manager_review",
                            "relevance_tags": ["bot_safe", "structured", "foton"],
                            "allowed_for_bot": True,
                            "requires_manager_review": False,
                        }
                    ],
                },
            },
        },
    )

    assert provider.calls == 1
    assert result.route == "bot_answer_self_for_pilot"
    assert "следующий шаг" not in result.draft_text.casefold()
    assert "Уточните, пожалуйста" in result.draft_text
    assert "класс ученика" in result.draft_text
    assert "предмет" in result.draft_text
    assert "формат" in result.draft_text
    assert "bot_safe_memory_unconfirmed_step_detected" in result.safety_flags
    assert "manager_approval_required" in result.safety_flags
    assert "no_auto_send" in result.safety_flags


def test_direct_path_final_bot_safe_memory_guard_catches_post_layer_soft_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot_path = _write_wave6_snapshot(tmp_path)
    provider = _DirectPathProvider(
        SubscriptionDraftResult(
            route="bot_answer_self_for_pilot",
            draft_text="Подскажите, пожалуйста, класс ученика.",
            metadata={"direct_path": {"model_response": "raw"}},
            safety_flags=(),
        )
    )

    def late_soft_step(result: SubscriptionDraftResult, *, client_message: str, context: Mapping[str, object] | None = None) -> SubscriptionDraftResult:
        return replace(result, draft_text=result.draft_text + " Следующий шаг — уточнить класс ученика.")

    monkeypatch.setattr(subscription_provider, "apply_tone_close_detect_layer", late_soft_step)

    result = provider.build_draft(
        "Что дальше?",
        context={
            "active_brand": "foton",
            DIRECT_PATH_ENV: "1",
            "snapshot_path": str(snapshot_path),
            "TELEGRAM_BOT_SAFE_CRM_CONTEXT": "1",
            "TELEGRAM_BOT_SAFE_MEMORY_STEP_GUARD": "1",
            "timeline_context": {
                "source": "customer_timeline_bot_context",
                "found": True,
                "bot_context": {
                    "allowed_only": True,
                    "items": [
                        {
                            "chunk_id": "chunk-foton",
                            "chunk_type": "bot_safe_summary",
                            "text": "Фотон: клиент обсуждал запись.",
                            "next_step_status": "empty",
                            "relevance_tags": ["bot_safe", "structured", "foton"],
                            "allowed_for_bot": True,
                            "requires_manager_review": False,
                        }
                    ],
                },
            },
        },
    )

    assert provider.calls == 1
    assert result.route == "bot_answer_self_for_pilot"
    assert "следующий шаг" not in result.draft_text.casefold()
    assert "Уточните, пожалуйста, класс ученика" in result.draft_text
    assert "bot_safe_memory_unconfirmed_step_detected" in result.safety_flags


def test_direct_path_deal_action_off_keeps_service_topic_parity() -> None:
    provider = _DirectPathProvider(
        SubscriptionDraftResult(
            route="bot_answer_self_for_pilot",
            draft_text="Стоимость онлайн-курса физики для 8 класса — 47 250 ₽.",
            message_type="question",
            topic_id="service:S2_unclear",
            metadata={
                "direct_path": {
                    "retrieved_facts": {
                        "foton_online_price_physics_8": "Фотон: онлайн-курс физики для 8 класса стоит 47 250 ₽."
                    },
                    "wide_fact_exact_keys": ["foton_online_price_physics_8"],
                },
                "action_proposal": {"action": "answer_only", "confidence": 0.8},
            },
        )
    )

    result = provider.build_draft(
        "Какая стоимость онлайн-физики для 8 класса?",
        context={
            "active_brand": "foton",
            DIRECT_PATH_ENV: "1",
            "client_safe_fact_verified": True,
            "autonomy_policy": {"allow_autonomous": True, "allowed_topic_ids": ["theme:001_pricing"]},
            "conversation_intent_plan": {"topic_id": "theme:001_pricing", "primary_intent": "pricing"},
        },
    )

    assert result.topic_id == "service:S2_unclear"
    assert "direct_path_autonomy_topic_from_plan" not in result.safety_flags
    assert "action_decision" not in result.metadata


def test_direct_path_deal_action_autonomy_uses_intent_topic() -> None:
    provider = _DirectPathProvider(
        SubscriptionDraftResult(
            route="bot_answer_self_for_pilot",
            draft_text="Стоимость онлайн-курса физики для 8 класса — 47 250 ₽.",
            message_type="question",
            topic_id="service:S2_unclear",
            metadata={
                "direct_path": {
                    "retrieved_facts": {
                        "foton_online_price_physics_8": "Фотон: онлайн-курс физики для 8 класса стоит 47 250 ₽."
                    },
                    "wide_fact_exact_keys": ["foton_online_price_physics_8"],
                },
                "action_proposal": {"action": "answer_only", "confidence": 0.8},
            },
        )
    )

    result = provider.build_draft(
        "Какая стоимость онлайн-физики для 8 класса?",
        context={
            "active_brand": "foton",
            DIRECT_PATH_ENV: "1",
            subscription_llm.DEAL_ACTION_DECISION_ENV: "1",
            "client_safe_fact_verified": True,
            "autonomy_policy": {"allow_autonomous": True, "allowed_topic_ids": ["theme:001_pricing"]},
            "conversation_intent_plan": {"topic_id": "theme:001_pricing", "primary_intent": "pricing"},
        },
    )

    decision = result.metadata["action_decision"]
    assert result.topic_id == "theme:001_pricing"
    assert result.route == "bot_answer_self_for_pilot"
    assert "direct_path_autonomy_topic_from_plan" in result.safety_flags
    assert "autonomy_default_cautious_topic_not_allowed" not in result.safety_flags
    assert result.metadata["direct_path_autonomy_topic_from"] == "service:S2_unclear"
    assert decision["action"] == "answer_only"
    assert decision["requires_manager_approval"] is False


DEFAULT_SNAPSHOT_PATH = Path("product_data/knowledge_base/kb_release_20260612_v6_7_staging_r4_1/kb_release_v3_snapshot.json")
V67_SNAPSHOT_PATH = DEFAULT_SNAPSHOT_PATH


def _wide_pack_context(
    *,
    brand: str,
    message: str,
    known_slots: Mapping[str, str] | None = None,
    primary_intent: str = "pricing",
) -> dict[str, object]:
    return {
        "active_brand": brand,
        "snapshot_path": str(DEFAULT_SNAPSHOT_PATH),
        "conversation_intent_plan": {
            "primary_intent": primary_intent,
            "answer_topics": [primary_intent],
            "required_fact_keys": ["prices.current"] if primary_intent == "pricing" else [],
        },
        "known_slots": dict(known_slots or {}),
        "recent_messages": [f"Клиент: {message}"],
    }


def _wide_pack_text(pack: Mapping[str, object], keys: Sequence[str] | None = None) -> str:
    facts = pack.get("facts") if isinstance(pack.get("facts"), Mapping) else {}
    meta = pack.get("fact_metadata") if isinstance(pack.get("fact_metadata"), Mapping) else {}
    selected = keys or tuple(facts.keys())
    return _direct_path_render_fact_block(facts, fact_metadata=meta, keys=tuple(str(key) for key in selected))


def test_template_from_kb_off_keeps_literal_terminal_template() -> None:
    context = {
        "active_brand": "foton",
        "snapshot_path": str(V67_SNAPSHOT_PATH),
        subscription_llm.TEMPLATE_FROM_KB_ENV: "0",
    }

    text = subscription_llm._terminal_safe_template(
        SubscriptionDraftResult(route="draft_for_manager", draft_text=""),
        client_message="Где вы в Москве?",
        context=context,
    )

    assert text == ADDRESS_FOTON_MOSCOW_SAFE_TEXT


def test_template_from_kb_renders_address_and_contacts_from_v67_snapshot() -> None:
    foton_context = {
        "active_brand": "foton",
        "snapshot_path": str(V67_SNAPSHOT_PATH),
        subscription_llm.TEMPLATE_FROM_KB_ENV: "1",
    }
    unpk_context = {
        "active_brand": "unpk",
        "snapshot_path": str(V67_SNAPSHOT_PATH),
        subscription_llm.TEMPLATE_FROM_KB_ENV: "1",
    }

    foton_address = subscription_llm._terminal_safe_template(
        SubscriptionDraftResult(route="draft_for_manager", draft_text=""),
        client_message="Где вы в Москве?",
        context=foton_context,
    )
    assert "Верхняя Красносельская ул., 30" in foton_address
    assert "Красносельская" in foton_address
    assert foton_address != ADDRESS_FOTON_MOSCOW_SAFE_TEXT

    foton_contacts = subscription_llm._terminal_safe_template(
        SubscriptionDraftResult(route="draft_for_manager", draft_text=""),
        client_message="Дайте телефон и почту, пожалуйста",
        context=foton_context,
    )
    assert "8 (495) 500-25-88" in foton_contacts
    assert "8 (800) 550-25-88" in foton_contacts
    assert "edu@cdpofoton.ru" in foton_contacts
    assert foton_context["template_from_kb_trace"][-1]["fact_key"] == "contacts_foton.phone+toll_free+email"

    unpk_contacts = subscription_llm._terminal_safe_template(
        SubscriptionDraftResult(route="draft_for_manager", draft_text=""),
        client_message="Дайте телефон, пожалуйста",
        context=unpk_context,
    )
    assert "+7 (495) 150-81-51" in unpk_contacts
    assert "8 (800) 500-81-51" in unpk_contacts
    assert "edu@kmipt.ru" in unpk_contacts
    rendered_phone = subscription_llm._direct_path_template_from_fact(
        active_brand="unpk",
        fact_key="contacts_unpk.phone",
        literal_text="literal",
        neutral_fallback="fallback",
        context=unpk_context,
        render=subscription_llm._direct_path_fact_value,
    )
    assert rendered_phone == "+7 (495) 150-81-51"


def test_template_from_kb_pilot_gold_renders_wave1_templates_from_default_snapshot(monkeypatch) -> None:
    for key in (TEMPLATE_FROM_KB_ENV, DIRECT_PATH_PILOT_CONFIG_ENV):
        monkeypatch.delenv(key, raising=False)
    context = {
        "snapshot_path": str(DEFAULT_SNAPSHOT_PATH),
        DIRECT_PATH_PILOT_CONFIG_ENV: DIRECT_PATH_PILOT_CONFIG_VERSION,
    }

    cases = (
        (
            "foton",
            "Где вы в Москве?",
            "Верхняя Красносельская ул., 30",
            ADDRESS_FOTON_MOSCOW_SAFE_TEXT,
        ),
        (
            "unpk",
            "Где в Москве обычные занятия?",
            "Сретенка, 20",
            ADDRESS_UNPK_MOSCOW_REGULAR_SAFE_TEXT,
        ),
        (
            "unpk",
            "Какие площадки?",
            "Площадки УНПК:",
            ADDRESS_UNPK_SAFE_TEXT,
        ),
    )

    for brand, message, expected, literal in cases:
        rendered = subscription_llm._terminal_safe_template(
            SubscriptionDraftResult(route="draft_for_manager", draft_text=""),
            client_message=message,
            context={**context, "active_brand": brand},
        )
        assert expected in rendered
        assert rendered != literal
        assert "лучше уточнить" not in rendered.casefold()


def test_template_from_kb_contact_trace_is_visible_in_direct_metadata(monkeypatch) -> None:
    for key in (TEMPLATE_FROM_KB_ENV, DIRECT_PATH_PILOT_CONFIG_ENV, LLM_RETRIEVE_ENV):
        monkeypatch.delenv(key, raising=False)
    context = {
        "active_brand": "foton",
        "snapshot_path": str(DEFAULT_SNAPSHOT_PATH),
        DIRECT_PATH_PILOT_CONFIG_ENV: DIRECT_PATH_PILOT_CONFIG_VERSION,
        LLM_RETRIEVE_ENV: "0",
    }
    provider = _DirectPathProvider(SubscriptionDraftResult(route="draft_for_manager", draft_text="Дайте контакты."))

    result = provider.build_draft("Дайте телефон и почту", context=context)

    trace = result.metadata["direct_path"]["template_from_kb_trace"]
    assert trace[-1]["fact_key"] == "direct_path.wide_fact_pack"
    assert trace[-1]["outcome"] == "hit"
    assert trace[-1]["selected_category"] == "contact"
    assert "contacts_foton.email" in trace[-1]["exact_keys"]
    assert result.metadata["template_from_kb_trace"] == trace


def test_template_from_kb_pilot_gold_explicit_off_returns_literal(monkeypatch) -> None:
    for key in (TEMPLATE_FROM_KB_ENV, DIRECT_PATH_PILOT_CONFIG_ENV):
        monkeypatch.delenv(key, raising=False)
    context = {
        "active_brand": "foton",
        "snapshot_path": str(DEFAULT_SNAPSHOT_PATH),
        DIRECT_PATH_PILOT_CONFIG_ENV: DIRECT_PATH_PILOT_CONFIG_VERSION,
        TEMPLATE_FROM_KB_ENV: "0",
    }

    rendered = subscription_llm._terminal_safe_template(
        SubscriptionDraftResult(route="draft_for_manager", draft_text=""),
        client_message="Где вы в Москве?",
        context=context,
    )

    assert rendered == ADDRESS_FOTON_MOSCOW_SAFE_TEXT


def test_template_from_kb_uses_neutral_fallback_for_missing_or_foreign_fact() -> None:
    context = {
        "active_brand": "foton",
        "snapshot_path": str(V67_SNAPSHOT_PATH),
        subscription_llm.TEMPLATE_FROM_KB_ENV: "1",
    }

    text = subscription_llm._direct_path_template_from_fact(
        active_brand="foton",
        fact_key="contacts_unpk.phone",
        literal_text="literal",
        neutral_fallback="Актуальные контакты лучше уточнить у менеджера.",
        context=context,
    )

    assert text == "Актуальные контакты лучше уточнить у менеджера."


def test_direct_path_contact_question_selects_contact_facts_from_snapshot() -> None:
    pack = _direct_path_context_fact_pack(
        {
            "active_brand": "foton",
            "snapshot_path": str(DEFAULT_SNAPSHOT_PATH),
            "conversation_intent_plan": {
                "primary_intent": "enrollment",
                "answer_topics": ["enrollment", "format"],
            },
        },
        client_message="Дайте телефон и почту",
    )

    assert "contact" in str(pack["selected_category"])
    assert "contacts_foton.phone" in pack["facts"]
    assert "contacts_foton.email" in pack["facts"]


def test_terminal_contact_request_ignores_client_own_contact() -> None:
    text = subscription_llm._terminal_safe_template(
        SubscriptionDraftResult(route="draft_for_manager", draft_text=""),
        client_message="Мой телефон +7 999 000-00-00, моя почта test@example.com",
        context={
            "active_brand": "foton",
            "snapshot_path": str(DEFAULT_SNAPSHOT_PATH),
            TEMPLATE_FROM_KB_ENV: "1",
        },
    )

    assert text == ""


def test_direct_path_contact_facts_do_not_answer_class_schedule() -> None:
    pack = _direct_path_context_fact_pack(
        {
            "active_brand": "foton",
            "snapshot_path": str(DEFAULT_SNAPSHOT_PATH),
            "conversation_intent_plan": {
                "primary_intent": "schedule",
                "answer_topics": ["schedule"],
            },
        },
        client_message="По каким дням занятия?",
    )

    assert "schedule" in str(pack["selected_category"])
    assert "contact" not in str(pack["selected_category"])


def _write_wave6_snapshot(tmp_path: Path) -> Path:
    snapshot = {
        "facts": [
            {
                "brand": "foton",
                "fact_key": "foton.price.online",
                "fact_type": "price",
                "product": "regular_course",
                "allowed_for_client_answer": True,
                "forbidden_for_client": False,
                "internal_only": False,
                "client_safe_text": "Фотон: онлайн-курс стоит 74 500 ₽ за год.",
            },
            {
                "brand": "foton",
                "fact_key": "foton.enrollment.next_step",
                "fact_type": "enrollment",
                "product": "regular_course",
                "allowed_for_client_answer": True,
                "forbidden_for_client": False,
                "internal_only": False,
                "client_safe_text": "Фотон: после оплаты менеджер помогает оформить заявку и подобрать группу.",
            },
            {
                "brand": "foton",
                "fact_key": "foton.schedule",
                "fact_type": "schedule",
                "product": "regular_course",
                "allowed_for_client_answer": True,
                "forbidden_for_client": False,
                "internal_only": False,
                "client_safe_text": "Фотон: расписание подбирается по классу и формату.",
            },
            {
                "brand": "unpk",
                "fact_key": "unpk.price.offline",
                "fact_type": "price",
                "product": "regular_course",
                "allowed_for_client_answer": True,
                "forbidden_for_client": False,
                "internal_only": False,
                "client_safe_text": "УНПК МФТИ: очный курс стоит 49 000 ₽ за семестр.",
            },
        ]
    }
    path = tmp_path / "wave6_snapshot.json"
    path.write_text(json.dumps(snapshot, ensure_ascii=False), encoding="utf-8")
    return path


def test_direct_path_wide_pack_price_close_contains_unpk_offline_price_pair() -> None:
    message = "Сколько стоит очно физика 9 класс?"
    pack = _direct_path_context_fact_pack(
        _wide_pack_context(brand="unpk", message=message),
        client_message=message,
    )

    exact_text = _wide_pack_text(pack, pack["exact_keys"])
    assert str(pack["selected_category"]).startswith("pricing")
    assert "49 000" in exact_text
    assert "82 000" in exact_text
    assert len(pack["facts"]) <= 60


def test_direct_path_wide_pack_is_brand_isolated_for_both_brands() -> None:
    for brand in ("foton", "unpk"):
        pack = _direct_path_context_fact_pack(
            _wide_pack_context(brand=brand, message="Сколько стоит курс?", primary_intent="pricing"),
            client_message="Сколько стоит курс?",
        )
        metadata = pack["fact_metadata"]
        assert metadata
        assert {item["brand"] for item in metadata.values()} == {brand}


def test_direct_path_wide_pack_serializes_only_client_safe_fields() -> None:
    pack = _direct_path_context_fact_pack(
        _wide_pack_context(brand="foton", message="Сколько стоит онлайн?", primary_intent="pricing"),
        client_message="Сколько стоит онлайн?",
    )
    text = _wide_pack_text(pack)
    assert "internal_text" not in text
    assert "manager_check" not in text
    assert "Скорняжн" not in text
    assert "лиценз" not in text.casefold()


def test_direct_path_wide_pack_excludes_expired_client_safe_fact(tmp_path) -> None:
    snapshot = {
        "facts": [
            {
                "brand": "foton",
                "fact_key": "expired.price",
                "fact_type": "price",
                "product": "regular_course",
                "allowed_for_client_answer": True,
                "forbidden_for_client": False,
                "internal_only": False,
                "valid_until": "2026-05-15",
                "client_safe_text": "Фотон: старая цена — 1 000 ₽.",
            },
            {
                "brand": "foton",
                "fact_key": "fresh.price",
                "fact_type": "price",
                "product": "regular_course",
                "allowed_for_client_answer": True,
                "forbidden_for_client": False,
                "internal_only": False,
                "valid_until": "2027-08-31",
                "client_safe_text": "Фотон: новая цена — 2 000 ₽.",
            },
        ]
    }
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False), encoding="utf-8")

    pack = _direct_path_context_fact_pack(
        {
            "active_brand": "foton",
            "snapshot_path": str(snapshot_path),
            "conversation_intent_plan": {"primary_intent": "pricing", "answer_topics": ["pricing"]},
        },
        client_message="Сколько стоит?",
    )
    text = _wide_pack_text(pack)

    assert "старая цена" not in text
    assert "новая цена" in text


def test_direct_path_wide_pack_schedule_stays_under_limit_and_keeps_exact_block() -> None:
    message = "Когда занятия и какое расписание?"
    pack = _direct_path_context_fact_pack(
        _wide_pack_context(brand="unpk", message=message, primary_intent="schedule"),
        client_message=message,
    )
    facts = pack["facts"]
    meta = pack["fact_metadata"]
    assert facts
    assert pack["exact_keys"]
    assert len(facts) <= 60
    assert sum(len(_wide_pack_text(pack, [key])) for key in facts) <= DIRECT_PATH_WIDE_FACT_CHAR_LIMIT
    assert _direct_path_render_fact_block(facts, fact_metadata=meta, keys=pack["exact_keys"])


def test_direct_path_wide_pack_marks_scope_conflict_as_adjacent() -> None:
    message = "Сколько стоит физика 9 класс?"
    pack = _direct_path_context_fact_pack(
        _wide_pack_context(brand="unpk", message=message, known_slots={"format": "очно"}, primary_intent="pricing"),
        client_message=message,
    )
    exact_text = _wide_pack_text(pack, pack["exact_keys"]).casefold()
    adjacent_text = _wide_pack_text(pack, pack["adjacent_keys"]).casefold()
    assert "очно" in exact_text
    assert "49 000" in exact_text
    assert "онлайн" in adjacent_text


def test_wave6_llm_retriever_prompt_tells_model_to_restore_incomplete_question() -> None:
    prompt = subscription_llm.build_direct_path_llm_retriever_prompt(
        "А по физике?",
        context={
            "recent_messages": ("Клиент: Сколько стоит очная математика 9 класс?", "Бот: Очный курс стоит 49 000 ₽."),
            "known_slots": {"format": "очно", "grade": "9"},
        },
        candidates=[
            {
                "fact_key": "foton.physics.offline.price",
                "client_safe_text": "Очный курс физики для 9 класса стоит 49 000 ₽.",
                "fact_type": "price",
                "product": "offline",
            }
        ],
    )

    assert "Если текущий вопрос неполный" in prompt
    assert "восстанови его по последним репликам диалога" in prompt
    assert "А по физике?" in prompt
    assert "Сколько стоит очная математика 9 класс?" in prompt
    assert "foton.physics.offline.price" in prompt


def test_wave6_llm_retriever_prompt_keeps_standalone_question_verbatim() -> None:
    question = "Сколько стоит очная физика для 9 класса?"

    prompt = subscription_llm.build_direct_path_llm_retriever_prompt(
        question,
        context={"recent_messages": ("Клиент: Здравствуйте",)},
        candidates=[{"fact_key": "foton.physics.offline.price", "client_safe_text": "Очная физика стоит 49 000 ₽."}],
    )

    assert f"Вопрос клиента:\n{question}\n\n" in prompt
    assert "foton.physics.offline.price" in prompt


def test_wave6_llm_retrieve_off_parity_keeps_keyword_pack(tmp_path: Path) -> None:
    snapshot_path = _write_wave6_snapshot(tmp_path)
    context = {
        "active_brand": "foton",
        "snapshot_path": str(snapshot_path),
        "conversation_intent_plan": {"primary_intent": "pricing", "answer_topics": ["pricing"]},
    }
    calls = 0

    def retriever(_: str) -> Mapping[str, object]:
        nonlocal calls
        calls += 1
        raise AssertionError("retriever must not be called with flag OFF")

    keyword = _direct_path_context_fact_pack(context, client_message="Сколько стоит?")
    off = _direct_path_context_fact_pack(
        {**context, LLM_RETRIEVE_ENV: "0"},
        client_message="Сколько стоит?",
        retriever_fn=retriever,
    )

    assert off == keyword
    assert calls == 0


def test_tz137_direct_plan_known_slots_flag_reads_serialized_known_slots() -> None:
    context = {
        "conversation_intent_plan": {
            "known_slots": {"grade": "8", "subject": "физика"},
            "slots": {"grade": "7"},
        }
    }

    off = subscription_llm._direct_path_known_slots(context)
    on = subscription_llm._direct_path_known_slots({**context, subscription_llm.DIRECT_PLAN_KNOWN_SLOTS_ENV: "1"})
    replay = subscription_llm._direct_path_known_slots(
        {"conversation_intent_plan": {"slots": {"grade": "9"}}, subscription_llm.DIRECT_PLAN_KNOWN_SLOTS_ENV: "1"}
    )

    assert off["grade"] == "7"
    assert on["grade"] == "8"
    assert on["subject"] == "физика"
    assert replay["grade"] == "9"


def test_tz137_keyword_fallback_relevance_drops_irrelevant_top_n() -> None:
    records = [
        {
            "brand": "foton",
            "fact_key": "foton.tax.license",
            "fact_type": "tax",
            "product": "documents",
            "allowed_for_client_answer": True,
            "client_safe_text": "Фотон: для налогового вычета используется лицензия.",
        }
    ]
    legacy = {"legacy.safe": "Уточните параметры вопроса, чтобы сориентировать корректно."}

    off = subscription_llm._direct_path_keyword_fact_pack_from_records(
        records,
        legacy=legacy,
        active_brand="foton",
        context={},
        client_message="Подскажите",
        max_facts=3,
        max_chars=4000,
    )
    on = subscription_llm._direct_path_keyword_fact_pack_from_records(
        records,
        legacy=legacy,
        active_brand="foton",
        context={subscription_llm.DIRECT_KEYWORD_FALLBACK_RELEVANCE_ENV: "1"},
        client_message="Подскажите",
        max_facts=3,
        max_chars=4000,
    )

    assert "foton.tax.license" in off["facts"]
    assert on["facts"] == legacy
    assert on["selected_category"] == "legacy_context"


def test_tz137_keyword_fallback_relevance_keeps_relevant_price_fact() -> None:
    records = [
        {
            "brand": "foton",
            "fact_key": "foton.price.online",
            "fact_type": "price",
            "product": "regular_course",
            "allowed_for_client_answer": True,
            "client_safe_text": "Фотон: онлайн-курс стоит 74 500 ₽ за год.",
        }
    ]

    pack = subscription_llm._direct_path_keyword_fact_pack_from_records(
        records,
        legacy={},
        active_brand="foton",
        context={subscription_llm.DIRECT_KEYWORD_FALLBACK_RELEVANCE_ENV: "1"},
        client_message="Сколько стоит?",
        max_facts=3,
        max_chars=4000,
    )

    assert pack["exact_keys"] == ["foton.price.online"]
    assert "74 500" in _wide_pack_text(pack)


def test_tz137_route_rubric_regenerates_empty_selection_open_question_only_with_flag() -> None:
    result = SubscriptionDraftResult(route="draft_for_manager", draft_text="Передам менеджеру.")
    fact_pack = {"llm_retrieve": {"fallback": True, "fallback_reason": "empty_selection"}}
    context = {
        subscription_llm.ROUTE_RUBRIC_ENV: "1",
        "conversation_intent_plan": {"direct_question": "Сколько стоит?"},
    }

    assert (
        subscription_llm._direct_path_route_rubric_should_regenerate(
            result,
            context=context,
            facts={},
            model_called=True,
            fact_pack=fact_pack,
        )
        is False
    )
    assert (
        subscription_llm._direct_path_route_rubric_should_regenerate(
            result,
            context={**context, subscription_llm.DIRECT_KEYWORD_FALLBACK_RELEVANCE_ENV: "1"},
            facts={},
            model_called=True,
            fact_pack=fact_pack,
        )
        is True
    )


def test_tz137_keyword_fallback_reask_turns_empty_selection_handoff_into_question() -> None:
    result = SubscriptionDraftResult(
        route="draft_for_manager",
        draft_text="Передам менеджеру.",
        metadata={"direct_path": {"llm_retrieve": {"fallback": True, "fallback_reason": "empty_selection"}}},
    )

    changed = subscription_llm.apply_direct_keyword_fallback_reask_layer(
        result,
        context={
            subscription_llm.DIRECT_KEYWORD_FALLBACK_RELEVANCE_ENV: "1",
            "conversation_intent_plan": {"direct_question": "Сколько стоит?", "requested_slots": ["grade"]},
        },
    )
    p0 = subscription_llm.apply_direct_keyword_fallback_reask_layer(
        replace(result, risk_level="high", safety_flags=("payment_dispute",)),
        context={
            subscription_llm.DIRECT_KEYWORD_FALLBACK_RELEVANCE_ENV: "1",
            "conversation_intent_plan": {"direct_question": "Верните деньги", "requested_slots": ["grade"]},
        },
    )

    assert changed.route == "bot_answer_self_for_pilot"
    assert "класс" in changed.draft_text.casefold()
    assert changed.metadata["direct_path"]["route_after"] == "bot_answer_self_for_pilot"
    assert p0.route == "draft_for_manager"


def test_wave6_llm_retrieve_selects_enrollment_fact_for_paid_next_step(tmp_path: Path) -> None:
    snapshot_path = _write_wave6_snapshot(tmp_path)
    message = "Оплатила, что дальше?"
    context = {
        "active_brand": "foton",
        "snapshot_path": str(snapshot_path),
        LLM_RETRIEVE_ENV: "1",
        "conversation_intent_plan": {"primary_intent": "pricing", "answer_topics": ["pricing"]},
    }

    pack = _direct_path_context_fact_pack(
        context,
        client_message=message,
        retriever_fn=lambda prompt: {"exact_ids": ["foton.enrollment.next_step"], "adjacent_ids": ["foton.schedule"]},
    )

    assert pack["selected_category"] == "llm_retrieve"
    assert pack["exact_keys"] == ["foton.enrollment.next_step"]
    assert "foton.schedule" in pack["adjacent_keys"]
    assert "после оплаты" in _wide_pack_text(pack, pack["exact_keys"]).casefold()
    assert pack["llm_retrieve"]["used"] is True


def test_tz110_retriever_flags_off_keep_id_only_contract(tmp_path: Path) -> None:
    snapshot_path = _write_wave6_snapshot(tmp_path)
    prompt_seen = ""
    context = {
        "active_brand": "foton",
        "snapshot_path": str(snapshot_path),
        LLM_RETRIEVE_ENV: "1",
        "conversation_intent_plan": {
            "primary_intent": "pricing",
            "answer_topics": ["pricing"],
            "required_fact_keys": ["prices.current"],
        },
    }

    def retriever(prompt: str) -> Mapping[str, object]:
        nonlocal prompt_seen
        prompt_seen = prompt
        return {"exact_ids": ["foton.price.online"], "adjacent_ids": []}

    pack = _direct_path_context_fact_pack(context, client_message="Сколько стоит?", retriever_fn=retriever)

    assert "needed_facts" not in prompt_seen
    assert "required_fact_keys" in prompt_seen
    assert "prices.current" in prompt_seen
    assert pack["llm_retrieve"]["mode"] == "id_only"
    assert pack["llm_retrieve"]["needed_facts"] == []
    assert pack["llm_retrieve"]["keyword_required_fact_keys"] == ["prices.current"]


def test_tz110_need_shadow_logs_declaration_without_changing_selection(tmp_path: Path) -> None:
    snapshot_path = _write_wave6_snapshot(tmp_path)
    base_context = {
        "active_brand": "foton",
        "snapshot_path": str(snapshot_path),
        LLM_RETRIEVE_ENV: "1",
        "conversation_intent_plan": {
            "primary_intent": "pricing",
            "answer_topics": ["pricing"],
            "required_fact_keys": ["prices.current"],
        },
    }
    payload = {
        "needed_facts": [
            {
                "theme": "pricing",
                "fact_type": "price",
                "brand": "foton",
                "grade": "не указано",
                "subject": "не указано",
                "format": "онлайн",
                "product": "regular_course",
                "why_needed": "клиент спрашивает стоимость",
                "importance": "required",
            }
        ],
        "exact_ids": ["foton.price.online"],
        "adjacent_ids": ["foton.schedule"],
    }

    off = _direct_path_context_fact_pack(base_context, client_message="Сколько стоит?", retriever_fn=lambda prompt: payload)
    shadow = _direct_path_context_fact_pack(
        {**base_context, RETRIEVER_NEED_SHADOW_ENV: "1"},
        client_message="Сколько стоит?",
        retriever_fn=lambda prompt: payload,
    )

    assert shadow["facts"] == off["facts"]
    assert shadow["exact_keys"] == off["exact_keys"]
    assert shadow["adjacent_keys"] == off["adjacent_keys"]
    assert shadow["llm_retrieve"]["mode"] == "need_shadow"
    assert shadow["llm_retrieve"]["need_shadow_enabled"] is True
    assert shadow["llm_retrieve"]["model_driven"] is False
    assert shadow["llm_retrieve"]["needed_facts"][0]["fact_type"] == "price"
    assert shadow["llm_retrieve"]["declaration_comparison"]["model_fact_types"] == ["price"]
    assert shadow["llm_retrieve"]["keyword_required_fact_keys"] == ["prices.current"]


def test_tz110_model_driven_strips_required_fact_keys_from_retriever_prompt_but_keeps_metadata(tmp_path: Path) -> None:
    snapshot_path = _write_wave6_snapshot(tmp_path)
    prompt_seen = ""
    calls = 0
    context = {
        "active_brand": "foton",
        "snapshot_path": str(snapshot_path),
        LLM_RETRIEVE_ENV: "1",
        ASSUMED_SCOPE_GUARD_ENV: "1",
        RETRIEVER_MODEL_DRIVEN_ENV: "1",
        "conversation_intent_plan": {
            "primary_intent": "pricing",
            "answer_topics": ["pricing"],
            "required_fact_keys": ["prices.current"],
        },
    }

    def retriever(prompt: str) -> Mapping[str, object]:
        nonlocal calls, prompt_seen
        calls += 1
        prompt_seen = prompt
        return {
            "needed_facts": [
                {
                    "theme": "pricing",
                    "fact_type": "price",
                    "brand": "foton",
                    "why_needed": "клиент спрашивает стоимость",
                    "importance": "required",
                }
            ],
            "exact_ids": ["foton.price.online"],
            "adjacent_ids": [],
        }

    pack = _direct_path_context_fact_pack(context, client_message="Сколько стоит?", retriever_fn=retriever)

    assert "required_fact_keys" not in prompt_seen
    assert "prices.current" not in prompt_seen
    assert "сам по смыслу определи" in prompt_seen
    assert calls == 1
    assert pack["llm_retrieve"]["mode"] == "model_driven"
    assert pack["llm_retrieve"]["model_driven"] is True
    assert pack["llm_retrieve"]["keyword_required_fact_keys"] == ["prices.current"]


def test_tz119_model_driven_requires_assumed_scope_guard(tmp_path: Path) -> None:
    snapshot_path = _write_wave6_snapshot(tmp_path)
    prompt_seen = ""
    context = {
        "active_brand": "foton",
        "snapshot_path": str(snapshot_path),
        LLM_RETRIEVE_ENV: "1",
        RETRIEVER_MODEL_DRIVEN_ENV: "1",
        "conversation_intent_plan": {
            "primary_intent": "pricing",
            "answer_topics": ["pricing"],
            "required_fact_keys": ["prices.current"],
        },
    }

    def retriever(prompt: str) -> Mapping[str, object]:
        nonlocal prompt_seen
        prompt_seen = prompt
        return {"exact_ids": ["foton.price.online"], "adjacent_ids": []}

    pack = _direct_path_context_fact_pack(context, client_message="Сколько стоит?", retriever_fn=retriever)

    assert "required_fact_keys" in prompt_seen
    assert "сам по смыслу определи" not in prompt_seen
    assert pack["llm_retrieve"]["mode"] == "id_only"
    assert pack["llm_retrieve"]["model_driven"] is False
    assert subscription_llm._retriever_model_driven_enabled(context) is False
    assert subscription_llm._retriever_model_driven_enabled({**context, ASSUMED_SCOPE_GUARD_ENV: "1"}) is True


def test_tz110_model_driven_requires_needed_fact_declaration_and_falls_back_to_keyword(tmp_path: Path) -> None:
    snapshot_path = _write_wave6_snapshot(tmp_path)
    context = {
        "active_brand": "foton",
        "snapshot_path": str(snapshot_path),
        LLM_RETRIEVE_ENV: "1",
        ASSUMED_SCOPE_GUARD_ENV: "1",
        RETRIEVER_MODEL_DRIVEN_ENV: "1",
        "conversation_intent_plan": {"primary_intent": "pricing", "answer_topics": ["pricing"]},
    }
    keyword = _direct_path_context_fact_pack({**context, LLM_RETRIEVE_ENV: "0"}, client_message="Сколько стоит?")

    pack = _direct_path_context_fact_pack(
        context,
        client_message="Сколько стоит?",
        retriever_fn=lambda prompt: {"exact_ids": ["foton.price.online"], "adjacent_ids": []},
    )

    assert pack["facts"] == keyword["facts"]
    assert pack["llm_retrieve"]["fallback"] is True
    assert pack["llm_retrieve"]["fallback_reason"] == "missing_needed_facts"
    assert pack["llm_retrieve"]["needed_fact_declaration_missing"] is True


def test_tz110_llm_retrieve_logs_scope_demoted_ids_for_wrong_scope_exact_selection(tmp_path: Path) -> None:
    snapshot = {
        "facts": [
            {
                "brand": "foton",
                "fact_key": "foton.physics9.online.price",
                "fact_type": "price",
                "product": "regular_course",
                "allowed_for_client_answer": True,
                "client_safe_text": "Фотон: физика 9 класс онлайн стоит 29 750 ₽ за семестр.",
            },
            {
                "brand": "foton",
                "fact_key": "foton.physics9.offline.price",
                "fact_type": "price",
                "product": "regular_course",
                "allowed_for_client_answer": True,
                "client_safe_text": "Фотон: физика 9 класс очно стоит 44 600 ₽ за семестр.",
            },
        ]
    }
    snapshot_path = tmp_path / "scope_snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False), encoding="utf-8")

    pack = _direct_path_context_fact_pack(
        {
            "active_brand": "foton",
            "snapshot_path": str(snapshot_path),
            LLM_RETRIEVE_ENV: "1",
            RETRIEVER_NEED_SHADOW_ENV: "1",
            "known_slots": {"format": "очно", "grade": "9", "subject": "физика"},
        },
        client_message="Сколько стоит очно физика 9 класс?",
        retriever_fn=lambda prompt: {
            "needed_facts": [
                {
                    "theme": "pricing",
                    "fact_type": "price",
                    "brand": "foton",
                    "format": "очно",
                    "why_needed": "клиент спрашивает цену очного формата",
                    "importance": "required",
                }
            ],
            "exact_ids": ["foton.physics9.online.price"],
            "adjacent_ids": ["foton.physics9.offline.price"],
        },
    )

    assert "foton.physics9.online.price" not in pack["exact_keys"]
    assert "foton.physics9.online.price" in pack["adjacent_keys"]
    assert "foton.physics9.offline.price" in pack["exact_keys"]
    assert pack["llm_retrieve"]["model_selected_exact_ids"] == ["foton.physics9.online.price"]
    assert pack["llm_retrieve"]["selected_exact_ids"] == ["foton.physics9.offline.price"]
    assert pack["llm_retrieve"]["scope_demoted_ids"] == ["foton.physics9.online.price"]


def test_tz119_unconfirmed_context_grade_is_soft_scope_not_hard_demotion(tmp_path: Path) -> None:
    snapshot = {
        "facts": [
            {
                "brand": "foton",
                "fact_key": "foton.physics5.online.price",
                "fact_type": "price",
                "product": "regular_course",
                "allowed_for_client_answer": True,
                "client_safe_text": "Фотон: физика 5 класс онлайн стоит 29 750 ₽ за семестр.",
            },
            {
                "brand": "foton",
                "fact_key": "foton.physics4.online.price",
                "fact_type": "price",
                "product": "regular_course",
                "allowed_for_client_answer": True,
                "client_safe_text": "Фотон: физика 4 класс онлайн стоит 29 750 ₽ за семестр.",
            },
        ]
    }
    snapshot_path = tmp_path / "assumed_scope_snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False), encoding="utf-8")
    prompt_seen = ""

    def retriever(prompt: str) -> Mapping[str, object]:
        nonlocal prompt_seen
        prompt_seen = prompt
        return {
            "needed_facts": [
                {
                    "theme": "pricing",
                    "fact_type": "price",
                    "brand": "foton",
                    "grade": "4",
                    "why_needed": "клиент спрашивает цену",
                    "importance": "required",
                }
            ],
            "exact_ids": ["foton.physics5.online.price"],
            "adjacent_ids": [],
        }

    context = {
        "active_brand": "foton",
        "snapshot_path": str(snapshot_path),
        LLM_RETRIEVE_ENV: "1",
        RETRIEVER_NEED_SHADOW_ENV: "1",
        ASSUMED_SCOPE_GUARD_ENV: "1",
        "dialogue_memory_view": {"known_slots": {"grade": "4", "subject": "физика"}},
    }

    pack = _direct_path_context_fact_pack(context, client_message="Сколько стоит?", retriever_fn=retriever)

    assert '"grade": {"status": "assumed_from_context", "value": "4"}' in prompt_seen
    assert pack["exact_keys"] == ["foton.physics5.online.price"]
    assert pack["llm_retrieve"]["scope_demoted_ids"] == []
    direct_meta = subscription_llm._direct_path_metadata(
        attempted=True,
        model_called=True,
        facts=pack["facts"],
        fact_pack=pack,
        context=context,
    )
    assert direct_meta["assumed_scope_guard"]["slot_provenance"]["grade"]["status"] == "assumed_from_context"


def test_tz119_confirmed_grade_still_scope_demotes_wrong_fact(tmp_path: Path) -> None:
    snapshot = {
        "facts": [
            {
                "brand": "foton",
                "fact_key": "foton.physics5.online.price",
                "fact_type": "price",
                "product": "regular_course",
                "allowed_for_client_answer": True,
                "client_safe_text": "Фотон: физика 5 класс онлайн стоит 29 750 ₽ за семестр.",
            }
        ]
    }
    snapshot_path = tmp_path / "confirmed_scope_snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False), encoding="utf-8")
    context = {
        "active_brand": "foton",
        "snapshot_path": str(snapshot_path),
        LLM_RETRIEVE_ENV: "1",
        RETRIEVER_NEED_SHADOW_ENV: "1",
        ASSUMED_SCOPE_GUARD_ENV: "1",
        "dialogue_memory_view": {
            "slot_provenance": {
                "grade": {
                    "value": "4",
                    "source": "memory_provenance",
                    "quote": "У ребёнка 4 класс",
                }
            }
        },
    }

    pack = _direct_path_context_fact_pack(
        context,
        client_message="Сколько стоит?",
        retriever_fn=lambda _: {
            "needed_facts": [{"theme": "pricing", "fact_type": "price", "brand": "foton", "importance": "required"}],
            "exact_ids": ["foton.physics5.online.price"],
            "adjacent_ids": [],
        },
    )

    assert pack["exact_keys"] == []
    assert "foton.physics5.online.price" in pack["adjacent_keys"]
    assert pack["llm_retrieve"]["scope_demoted_ids"] == ["foton.physics5.online.price"]
    assert subscription_llm._direct_path_slot_provenance(context)["grade"]["source"] == "memory_provenance"


def test_tz119_assumed_context_scope_guard_reasks_without_manager_handoff() -> None:
    result = subscription_llm.apply_assumed_scope_guard(
        SubscriptionDraftResult(
            route="bot_answer_self_for_pilot",
            draft_text="Для 4 класса онлайн стоимость 29 750 ₽ за семестр.",
        ),
        context={
            ASSUMED_SCOPE_GUARD_ENV: "1",
            "dialogue_memory_view": {"known_slots": {"grade": "4", "format": "онлайн"}},
        },
    )

    assert result.route == "bot_answer_self_for_pilot"
    assert "Правильно ли я понимаю" in result.draft_text
    assert "29 750" not in result.draft_text
    assert "assumed_scope_guard_reask" in result.safety_flags
    assert result.metadata["assumed_scope_guard"]["action"] == "reask_assumed_parameter"
    asserted_keys = {item["key"] for item in result.metadata["assumed_scope_guard"]["asserted_assumed_slots"]}
    assert "grade" in asserted_keys


def test_tz119_confirmed_slot_quote_prevents_reask_on_ellipsis() -> None:
    original = "Для 4 класса онлайн подойдёт регулярный курс."
    result = subscription_llm.apply_assumed_scope_guard(
        SubscriptionDraftResult(route="bot_answer_self_for_pilot", draft_text=original),
        context={
            ASSUMED_SCOPE_GUARD_ENV: "1",
            "dialogue_memory_view": {
                "slot_provenance": {
                    "grade": {
                        "value": "4",
                        "source": "memory_provenance",
                        "quote": "У нас 4 класс",
                    }
                },
                "do_not_reask_slots": ["grade"],
            },
        },
    )

    assert result.draft_text == original
    assert result.metadata["assumed_scope_guard"]["action"] == "pass"


def test_tz119_assumed_scope_guard_skips_p0_risk() -> None:
    original = "Для 4 класса можно оформить возврат по правилам."
    result = subscription_llm.apply_assumed_scope_guard(
        SubscriptionDraftResult(
            route="manager_only",
            risk_level="high",
            safety_flags=("p0_refund",),
            draft_text=original,
        ),
        context={
            ASSUMED_SCOPE_GUARD_ENV: "1",
            "dialogue_memory_view": {"crm_known_slots": {"grade": "4"}},
        },
    )

    assert result.draft_text == original
    assert result.route == "manager_only"
    assert result.metadata["assumed_scope_guard"]["action"] == "skipped_p0_or_risk"


def test_tz119_draft_prompt_marks_assumed_slots_only_when_flag_enabled() -> None:
    context = {
        "active_brand": "foton",
        "dialogue_memory_view": {"crm_known_slots": {"grade": "4", "subject": "физика"}},
    }

    off_prompt = subscription_llm._build_direct_path_prompt("Сколько стоит?", context=context)
    on_prompt = subscription_llm._build_direct_path_prompt(
        "Сколько стоит?",
        context={**context, ASSUMED_SCOPE_GUARD_ENV: "1"},
    )

    assert "assumed_from_context" not in off_prompt
    assert "Правило неподтверждённых параметров" not in off_prompt
    assert "assumed_from_context" in on_prompt
    assert "Не называй итоговые цены" in on_prompt
    assert '"grade": {' in on_prompt


def test_wave6_llm_retrieve_supplements_price_and_schedule_for_known_course(tmp_path: Path) -> None:
    snapshot = {
        "facts": [
            {
                "brand": "foton",
                "fact_key": "foton.physics8.online.schedule",
                "fact_type": "schedule",
                "product": "regular_course",
                "allowed_for_client_answer": True,
                "client_safe_text": "Физика, 8 класс, обычная группа, онлайн: воскресенье 14:30-16:30, старт 20.09.2026.",
            },
            {
                "brand": "foton",
                "fact_key": "foton.physics8.offline.schedule",
                "fact_type": "schedule",
                "product": "regular_course",
                "allowed_for_client_answer": True,
                "client_safe_text": "Физика, 8 класс, базовая группа, очно: воскресенье 10:00-12:00, старт 13.09.2026.",
            },
            {
                "brand": "foton",
                "fact_key": "foton.online.5_11.semester",
                "fact_type": "price",
                "product": "regular_course",
                "allowed_for_client_answer": True,
                "client_safe_text": "Фотон: цены на 2026/27 учебный год, 5-11 класс, онлайн, семестр — 29 750 ₽.",
            },
            {
                "brand": "foton",
                "fact_key": "foton.offline.5_11.semester",
                "fact_type": "price",
                "product": "regular_course",
                "allowed_for_client_answer": True,
                "client_safe_text": "Фотон: цены на 2026/27 учебный год, 5-11 класс, очно, семестр — 44 600 ₽.",
            },
        ]
    }
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False), encoding="utf-8")
    context = {
        "active_brand": "foton",
        "snapshot_path": str(snapshot_path),
        LLM_RETRIEVE_ENV: "1",
        "known_slots": {"grade": "8", "subject": "физика", "format": "онлайн"},
    }

    pack = _direct_path_context_fact_pack(
        context,
        client_message="онлайн",
        retriever_fn=lambda prompt: {
            "exact_ids": ["foton.physics8.online.schedule"],
            "adjacent_ids": ["foton.online.5_11.semester"],
        },
    )
    exact_text = _wide_pack_text(pack, pack["exact_keys"])

    assert "foton.physics8.online.schedule" in pack["exact_keys"]
    assert "foton.online.5_11.semester" in pack["exact_keys"]
    assert "foton.offline.5_11.semester" not in pack["facts"]
    assert "14:30-16:30" in exact_text
    assert "29 750" in exact_text
    assert pack["llm_retrieve"]["supplemented_exact_ids"] == ["foton.online.5_11.semester"]


def test_wave6_llm_retrieve_brand_isolation_filters_candidates_before_model(tmp_path: Path) -> None:
    snapshot_path = _write_wave6_snapshot(tmp_path)
    seen_prompt = ""

    def retriever(prompt: str) -> Mapping[str, object]:
        nonlocal seen_prompt
        seen_prompt = prompt
        return {"exact_ids": ["unpk.price.offline", "foton.enrollment.next_step"], "adjacent_ids": []}

    pack = _direct_path_context_fact_pack(
        {
            "active_brand": "foton",
            "snapshot_path": str(snapshot_path),
            LLM_RETRIEVE_ENV: "1",
        },
        client_message="Оплатила, что дальше?",
        retriever_fn=retriever,
    )

    assert "unpk.price.offline" not in seen_prompt
    assert "УНПК" not in seen_prompt
    assert "unpk.price.offline" not in pack["facts"]
    assert set(pack["facts"]) == {"foton.enrollment.next_step"}
    assert pack["llm_retrieve"]["invalid_ids"] == ["unpk.price.offline"]
    assert {item["brand"] for item in pack["fact_metadata"].values()} == {"foton"}


def test_wave6_llm_retrieve_fail_soft_falls_back_to_keyword(tmp_path: Path) -> None:
    snapshot_path = _write_wave6_snapshot(tmp_path)
    context = {
        "active_brand": "foton",
        "snapshot_path": str(snapshot_path),
        LLM_RETRIEVE_ENV: "1",
        "conversation_intent_plan": {"primary_intent": "pricing", "answer_topics": ["pricing"]},
    }
    keyword = _direct_path_context_fact_pack({**context, LLM_RETRIEVE_ENV: "0"}, client_message="Сколько стоит?")

    pack = _direct_path_context_fact_pack(
        context,
        client_message="Сколько стоит?",
        retriever_fn=lambda prompt: (_ for _ in ()).throw(subprocess.TimeoutExpired(cmd="retriever", timeout=1)),
    )

    assert pack["facts"] == keyword["facts"]
    assert pack["exact_keys"] == keyword["exact_keys"]
    assert pack["adjacent_keys"] == keyword["adjacent_keys"]
    assert pack["llm_retrieve"]["fallback"] is True
    assert pack["llm_retrieve"]["fallback_reason"] == "timeout"


def test_wave6_llm_retrieve_discards_hallucinated_ids_and_uses_valid_selection(tmp_path: Path) -> None:
    snapshot_path = _write_wave6_snapshot(tmp_path)
    pack = _direct_path_context_fact_pack(
        {
            "active_brand": "foton",
            "snapshot_path": str(snapshot_path),
            LLM_RETRIEVE_ENV: "1",
        },
        client_message="Оплатила, что дальше?",
        retriever_fn=lambda prompt: {"exact_ids": ["missing.fact", "foton.enrollment.next_step"], "adjacent_ids": []},
    )

    assert "missing.fact" not in pack["facts"]
    assert pack["exact_keys"] == ["foton.enrollment.next_step"]
    assert pack["llm_retrieve"]["invalid_ids"] == ["missing.fact"]


def test_wave6_llm_retrieve_only_hallucinated_ids_falls_back_to_keyword(tmp_path: Path) -> None:
    snapshot_path = _write_wave6_snapshot(tmp_path)
    context = {
        "active_brand": "foton",
        "snapshot_path": str(snapshot_path),
        LLM_RETRIEVE_ENV: "1",
        "conversation_intent_plan": {"primary_intent": "pricing", "answer_topics": ["pricing"]},
    }
    keyword = _direct_path_context_fact_pack({**context, LLM_RETRIEVE_ENV: "0"}, client_message="Сколько стоит?")

    pack = _direct_path_context_fact_pack(
        context,
        client_message="Сколько стоит?",
        retriever_fn=lambda prompt: {"exact_ids": ["missing.fact"], "adjacent_ids": []},
    )

    assert pack["facts"] == keyword["facts"]
    assert pack["llm_retrieve"]["fallback"] is True
    assert pack["llm_retrieve"]["fallback_reason"] == "empty_selection"
    assert pack["llm_retrieve"]["invalid_ids"] == ["missing.fact"]


def test_wave6_llm_retrieve_p0_preblock_skips_retriever_and_direct_model(tmp_path: Path) -> None:
    snapshot_path = _write_wave6_snapshot(tmp_path)
    provider = _DirectPathRetrieverProvider(
        SubscriptionDraftResult(route="bot_answer_self_for_pilot", draft_text="Этого текста быть не должно."),
        retriever_payload={"exact_ids": ["foton.enrollment.next_step"], "adjacent_ids": []},
    )

    result = provider.build_draft(
        "С карты списали дважды, верните деньги",
        context={
            "active_brand": "foton",
            "snapshot_path": str(snapshot_path),
            DIRECT_PATH_ENV: "1",
            LLM_RETRIEVE_ENV: "1",
        },
    )

    assert provider.retriever_calls == 0
    assert provider.calls == 0
    assert result.route == "manager_only"
    assert result.metadata["direct_path"]["preblocked"] is True
    assert result.metadata["direct_path"]["selected_category"] == "preblocked_before_llm_retrieve"


def test_wave1_number_scope_aware_wrong_scope_downgrades_direct_path_text() -> None:
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="Очно для 9 класса стоит 29 750 ₽.",
        topic_id="theme:001_pricing",
        metadata={
            "direct_path": {
                "enabled": True,
                "direct_path_attempted": True,
                "retrieved_facts": {
                    "price.online": "Онлайн для 9 класса стоит 29 750 ₽.",
                },
            }
        },
    )

    gated = apply_authoritative_output_gate(
        result,
        client_message="сколько стоит очно физика 9 класс?",
        context={
            "active_brand": "foton",
            "TELEGRAM_A_FREE_NUMBER_GATE": "1",
            "TELEGRAM_NUMBER_GATE_SCOPE_AWARE": "1",
        },
    )

    gate = gated.metadata["authoritative_output_gate"]
    assert gated.route == "draft_for_manager"
    assert gated.draft_text == result.draft_text
    assert gate["action"] == "downgrade_keep_text"
    assert "wrong_scope" in {item["code"] for item in gate["findings"]}
    assert "direct_path_gate_text_preserved" in gated.safety_flags


def test_direct_path_preblocks_p0_without_model_call() -> None:
    provider = _DirectPathProvider(
        SubscriptionDraftResult(route="bot_answer_self_for_pilot", draft_text="Этого текста быть не должно.")
    )
    result = provider.build_draft(
        "С карты списали дважды, верните деньги",
        context={"active_brand": "foton", DIRECT_PATH_ENV: "1"},
    )

    assert provider.calls == 0
    assert result.route == "manager_only"
    direct = result.metadata["direct_path"]
    assert direct["preblocked"] is True
    assert direct["model_called"] is False
    assert direct["reason_class"] == "p0_deferral"
    assert result.metadata["authoritative_output_gate"]["checked"] is True


def test_direct_path_p0_complaint_preblock_has_no_manager_deadline() -> None:
    provider = _DirectPathProvider(
        SubscriptionDraftResult(route="bot_answer_self_for_pilot", draft_text="Этого текста быть не должно.")
    )
    result = provider.build_draft(
        "Жалоба: преподаватель оскорбил ребенка на занятии.",
        context={"active_brand": "foton", DIRECT_PATH_ENV: "1"},
    )

    assert provider.calls == 0
    assert result.route == "manager_only"
    lowered = result.draft_text.casefold()
    assert "завтра" not in lowered
    assert "утром" not in lowered
    assert "в течение" not in lowered
    assert result.metadata["direct_path"]["reason_class"] == "p0_deferral"


def test_direct_path_child_incident_complaint_preblocks_without_collecting_details() -> None:
    provider = _DirectPathProvider(
        SubscriptionDraftResult(route="bot_answer_self_for_pilot", draft_text="Этого текста быть не должно.")
    )
    result = provider.build_draft(
        "Ребёнка унизили на занятии, я этого так не оставлю.",
        context={"active_brand": "unpk", DIRECT_PATH_ENV: "1"},
    )

    assert provider.calls == 0
    assert result.route == "manager_only"
    assert result.metadata["direct_path"]["preblocked"] is True
    assert result.metadata["direct_path"]["reason_class"] == "p0_deferral"
    assert "ребён" not in result.draft_text.casefold()
    assert "напишите" not in result.draft_text.casefold()


def test_direct_path_child_safety_complaint_preblocks_from_first_turn() -> None:
    provider = _DirectPathProvider(
        SubscriptionDraftResult(route="bot_answer_self_for_pilot", draft_text="Этого текста быть не должно.")
    )
    result = provider.build_draft(
        "Ребёнок остался один, никто не следил.",
        context={"active_brand": "foton", DIRECT_PATH_ENV: "1"},
    )

    assert provider.calls == 0
    assert result.route == "manager_only"
    assert result.metadata["direct_path"]["preblocked"] is True
    assert result.metadata["direct_path"]["reason_evidence"]["p0_kind"] == "complaint"


def test_direct_path_benign_teacher_minute_question_is_not_p0() -> None:
    provider = _DirectPathProvider(
        SubscriptionDraftResult(route="bot_answer_self_for_pilot", draft_text="Да, подскажу, как обычно устроен присмотр.")
    )
    result = provider.build_draft(
        "Педагог вышел на минуту — это нормально?",
        context={"active_brand": "foton", DIRECT_PATH_ENV: "1", "confirmed_facts": {"format.foton": "Фотон: занятия ведёт преподаватель."}},
    )

    assert provider.calls == 1
    assert result.route == "bot_answer_self_for_pilot"


def test_direct_path_model_p0_prompt_is_flagged_off_by_default() -> None:
    context = {"active_brand": "foton", DIRECT_PATH_ENV: "1"}

    off_prompt = subscription_llm._build_direct_path_prompt("Нужна помощь по спорной оплате.", context=context)
    on_prompt = subscription_llm._build_direct_path_prompt(
        "Нужна помощь по спорной оплате.",
        context={**context, subscription_llm.DIRECT_PATH_MODEL_P0_ENV: "1"},
    )

    assert "is_p0" not in off_prompt
    assert '"manager_only"' not in off_prompt
    assert "спорную оплату" in on_prompt
    assert '"is_p0": false' in on_prompt
    assert '"p0_kind": "none|payment_dispute|refund|complaint|legal_threat"' in on_prompt
    assert '"route": "bot_answer_self_for_pilot" | "draft_for_manager" | "manager_only"' in on_prompt
    assert "дорого/подумаю" in on_prompt
    assert "cancellation_service_request" not in on_prompt
    assert "paid_operation_context" not in on_prompt


def test_p0_model_classes_v2_prompt_is_profile_on_and_history_aware_when_enabled() -> None:
    context = {
        "active_brand": "foton",
        DIRECT_PATH_ENV: "1",
        subscription_llm.DIRECT_PATH_MODEL_P0_ENV: "1",
        "recent_messages": [
            "Клиент: Мы оплатили июльскую смену, но мест в нужной группе нет.",
            "Клиент: Что можно сделать?",
        ],
    }
    profile_context = {
        **context,
        subscription_llm.DIRECT_PATH_PILOT_CONFIG_ENV: subscription_llm.DIRECT_PATH_PILOT_CONFIG_VERSION,
    }

    off_prompt = subscription_llm._build_direct_path_prompt("Что можно сделать?", context=context)
    on_prompt = subscription_llm._build_direct_path_prompt(
        "Что можно сделать?",
        context=profile_context,
    )

    assert subscription_llm.P0_MODEL_CLASSES_V2_ENV in subscription_llm.DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS
    assert subscription_llm._p0_model_classes_v2_enabled(context) is False
    assert subscription_llm._p0_model_classes_v2_enabled(profile_context) is True
    assert subscription_llm._p0_model_classes_v2_enabled({**profile_context, subscription_llm.P0_MODEL_CLASSES_V2_ENV: "0"}) is False
    assert "cancellation_service_request" not in off_prompt
    assert "contract_dispute" not in off_prompt
    assert "paid_operation_context" not in off_prompt
    assert '"p0_kind": "none|payment_dispute|refund|complaint|legal_threat|' in on_prompt
    assert "cancellation_service_request" in on_prompt
    assert "contract_dispute" in on_prompt
    assert "paid_operation_context" in on_prompt
    assert "оценивай не только текущую реплику" in on_prompt
    assert "Сам факт оплаты не делает обращение P0" in on_prompt
    assert "Оплата вчера при будущих занятиях" in on_prompt
    assert "без жалобы, возврата и уже наступившего отсутствия доступа" in on_prompt
    assert "оплаченная смена/курс/запись" in on_prompt


def test_intent_model_led_default_off_and_enabled_by_pilot_profile() -> None:
    assert subscription_llm._intent_model_led_enabled({}) is False
    assert subscription_llm._intent_model_led_enabled({subscription_llm.INTENT_MODEL_LED_ENV: "1"}) is True
    assert subscription_llm._intent_model_led_enabled({subscription_llm.INTENT_MODEL_LED_ENV: "0"}) is False
    assert subscription_llm.INTENT_MODEL_LED_ENV in subscription_llm.DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS
    assert (
        subscription_llm._intent_model_led_enabled(
            {subscription_llm.DIRECT_PATH_PILOT_CONFIG_ENV: subscription_llm.DIRECT_PATH_PILOT_CONFIG_VERSION}
        )
        is True
    )
    assert (
        subscription_llm._intent_model_led_enabled(
            {
                subscription_llm.DIRECT_PATH_PILOT_CONFIG_ENV: subscription_llm.DIRECT_PATH_PILOT_CONFIG_VERSION,
                subscription_llm.INTENT_MODEL_LED_ENV: "0",
            }
        )
        is False
    )


def test_intent_model_led_prompt_block_is_flagged_only() -> None:
    context = {"active_brand": "foton", DIRECT_PATH_ENV: "1"}

    off_prompt = subscription_llm._build_direct_path_prompt("Привезу ребёнка сразу на место.", context=context)
    explicit_off_prompt = subscription_llm._build_direct_path_prompt(
        "Привезу ребёнка сразу на место.",
        context={**context, subscription_llm.INTENT_MODEL_LED_ENV: "0"},
    )
    on_prompt = subscription_llm._build_direct_path_prompt(
        "Привезу ребёнка сразу на место.",
        context={**context, subscription_llm.INTENT_MODEL_LED_ENV: "1"},
    )

    assert "model_intent" not in off_prompt
    assert "Смысловой intent_model_led" not in off_prompt
    assert "model_intent" not in explicit_off_prompt
    assert "Смысловой intent_model_led" in on_prompt
    assert '"model_intent": {"primary_intent": "live_availability|schedule|address|camp|price_fix|off_topic|other"' in on_prompt
    assert "off_topic ставь только" in on_prompt
    assert "«место» как территория/площадка/место занятий" in on_prompt
    assert "настоящего вопроса о наличии мест/броней/свободной группе" in on_prompt
    assert "«когда привезу/подъеду» — other" in on_prompt
    assert "«закрепить материал/навык» — other" in on_prompt


def test_direct_path_payload_parses_model_intent_metadata() -> None:
    result = _normalize_direct_path_payload(
        {
            "route": "bot_answer_self_for_pilot",
            "draft_text": "Подскажу по адресу.",
            "model_intent": {
                "primary_intent": "venue",
                "scope": "moscow_regular",
                "sense": "venue",
                "confidence": 0.82,
                "reason": "клиент спрашивает про площадку",
            },
        }
    )

    signal = result.metadata["direct_path_model_intent"]
    assert signal["primary_intent"] == "address"
    assert signal["scope"] == "moscow_regular"
    assert signal["sense"] == "venue"
    assert signal["confidence"] == 0.82


def test_intent_model_led_false_live_availability_keeps_direct_answer() -> None:
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="Да, привозите ребёнка на площадку по адресу из расписания.",
        topic_id="theme:013_schedule",
        metadata={
            "direct_path_model_intent": {
                "primary_intent": "address",
                "scope": "venue",
                "sense": "venue",
                "confidence": 0.9,
                "reason": "место означает площадку, не наличие мест",
            }
        },
    )
    context = {
        "active_brand": "foton",
        DIRECT_PATH_ENV: "1",
        subscription_llm.INTENT_MODEL_LED_ENV: "1",
        "conversation_intent_plan": {
            "schema_version": "test",
            "primary_intent": "live_availability",
            "topic_id": "theme:026_camp_general",
            "answer_policy": "answer_safe_parts_then_manager_live_check",
            "route_bias": "draft_for_manager",
            "keyword_signals": ["live_availability"],
        },
    }

    guarded = apply_conversation_intent_plan_guard(
        result,
        client_message="Я привезу ребёнка сразу на место, верно?",
        context=context,
    )

    assert guarded.route == "bot_answer_self_for_pilot"
    assert "conversation_intent_plan_live_availability" not in guarded.safety_flags
    assert "conversation_intent_plan_live_check_handoff" not in guarded.safety_flags
    assert guarded.metadata["intent_model_led"]["applied_primary_intent"] == "address"
    assert guarded.metadata["conversation_intent_primary_intent"] == "address"


def test_intent_model_led_true_live_availability_no_longer_hands_off_without_frame() -> None:
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="Я уточню по группе.",
        topic_id="theme:026_camp_general",
        metadata={
            "direct_path_model_intent": {
                "primary_intent": "live_availability",
                "scope": "seats",
                "sense": "seats",
                "confidence": 0.93,
                "reason": "клиент спрашивает о свободных местах",
            }
        },
    )
    context = {
        "active_brand": "foton",
        DIRECT_PATH_ENV: "1",
        subscription_llm.INTENT_MODEL_LED_ENV: "1",
        "conversation_intent_plan": {
            "schema_version": "test",
            "primary_intent": "live_availability",
            "topic_id": "theme:026_camp_general",
            "answer_policy": "answer_safe_parts_then_manager_live_check",
            "route_bias": "draft_for_manager",
            "keyword_signals": ["live_availability"],
        },
    }

    guarded = apply_conversation_intent_plan_guard(
        result,
        client_message="Есть ли места в группе?",
        context=context,
    )

    assert guarded.route == "bot_answer_self_for_pilot"
    assert "conversation_intent_plan_live_availability" not in guarded.safety_flags
    assert "conversation_intent_plan_live_check_handoff" not in guarded.safety_flags
    assert guarded.metadata["intent_model_led"]["applied_primary_intent"] == "live_availability"


def test_intent_model_led_does_not_demote_explicit_availability_question() -> None:
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="Да, привозите ребёнка на площадку по адресу из расписания.",
        topic_id="theme:013_schedule",
        metadata={
            "direct_path_model_intent": {
                "primary_intent": "address",
                "scope": "venue",
                "sense": "venue",
                "confidence": 0.99,
                "reason": "ошибочный тестовый сигнал",
            }
        },
    )
    context = {
        "active_brand": "foton",
        DIRECT_PATH_ENV: "1",
        subscription_llm.INTENT_MODEL_LED_ENV: "1",
        "conversation_intent_plan": {
            "schema_version": "test",
            "primary_intent": "live_availability",
            "topic_id": "theme:026_camp_general",
            "direct_question": "Есть ли места в группе?",
            "answer_policy": "answer_safe_parts_then_manager_live_check",
            "route_bias": "draft_for_manager",
            "keyword_signals": ["live_availability"],
        },
    }

    guarded = apply_conversation_intent_plan_guard(
        result,
        client_message="Есть ли места в группе?",
        context=context,
    )

    assert guarded.route == "bot_answer_self_for_pilot"
    assert "conversation_intent_plan_live_availability" not in guarded.safety_flags
    assert "conversation_intent_plan_live_check_handoff" not in guarded.safety_flags
    assert guarded.metadata["intent_model_led"]["applied"] is False
    assert guarded.metadata["intent_model_led"]["skip_reason"] == "explicit_live_availability_floor"


def test_intent_model_led_low_confidence_does_not_demote_live_availability() -> None:
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="Да, привозите ребёнка на площадку по адресу из расписания.",
        topic_id="theme:013_schedule",
        metadata={
            "direct_path_model_intent": {
                "primary_intent": "address",
                "scope": "venue",
                "sense": "venue",
                "confidence": 0.2,
                "reason": "неуверенный тестовый сигнал",
            }
        },
    )
    context = {
        "active_brand": "foton",
        DIRECT_PATH_ENV: "1",
        subscription_llm.INTENT_MODEL_LED_ENV: "1",
        "conversation_intent_plan": {
            "schema_version": "test",
            "primary_intent": "live_availability",
            "topic_id": "theme:026_camp_general",
            "answer_policy": "answer_safe_parts_then_manager_live_check",
            "route_bias": "draft_for_manager",
            "keyword_signals": ["live_availability"],
        },
    }

    guarded = apply_conversation_intent_plan_guard(
        result,
        client_message="Я привезу ребёнка сразу на место, верно?",
        context=context,
    )

    assert guarded.route == "bot_answer_self_for_pilot"
    assert "conversation_intent_plan_live_availability" not in guarded.safety_flags
    assert guarded.metadata["intent_model_led"]["applied"] is False
    assert guarded.metadata["intent_model_led"]["skip_reason"] == "low_confidence"


def test_intent_model_led_is_ignored_when_flag_off() -> None:
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="Да, привозите ребёнка на площадку по адресу из расписания.",
        topic_id="theme:013_schedule",
        metadata={"direct_path_model_intent": {"primary_intent": "address", "confidence": 0.9}},
    )
    context = {
        "active_brand": "foton",
        DIRECT_PATH_ENV: "1",
        "conversation_intent_plan": {
            "schema_version": "test",
            "primary_intent": "live_availability",
            "topic_id": "theme:026_camp_general",
            "answer_policy": "answer_safe_parts_then_manager_live_check",
            "route_bias": "draft_for_manager",
            "keyword_signals": ["live_availability"],
        },
    }

    guarded = apply_conversation_intent_plan_guard(
        result,
        client_message="Я привезу ребёнка сразу на место, верно?",
        context=context,
    )

    assert guarded.route == "bot_answer_self_for_pilot"
    assert "conversation_intent_plan_live_availability" not in guarded.safety_flags
    assert "intent_model_led" not in guarded.metadata


def test_direct_path_provider_uses_single_model_intent_signal_for_plan_guard() -> None:
    provider = _DirectPathProvider(
        SubscriptionDraftResult(
            route="bot_answer_self_for_pilot",
            draft_text="Да, привозите ребёнка на площадку по адресу из расписания.",
            topic_id="theme:013_schedule",
            metadata={
                "direct_path_model_intent": {
                    "primary_intent": "address",
                    "sense": "venue",
                    "confidence": 0.91,
                    "reason": "место означает площадку",
                }
            },
        )
    )

    result = provider.build_draft(
        "Привезу ребёнка сразу на место?",
        context={
            "active_brand": "foton",
            DIRECT_PATH_ENV: "1",
            subscription_llm.INTENT_MODEL_LED_ENV: "1",
            "conversation_intent_plan": {
                "schema_version": "test",
                "primary_intent": "live_availability",
                "topic_id": "theme:026_camp_general",
                "answer_policy": "answer_safe_parts_then_manager_live_check",
                "route_bias": "draft_for_manager",
                "keyword_signals": ["live_availability"],
            },
            "confirmed_facts": {"address.foton": "Фотон: занятия проходят на площадке по адресу из расписания."},
        },
    )

    assert provider.calls == 1
    assert result.route == "bot_answer_self_for_pilot"
    assert "conversation_intent_plan_live_availability" not in result.safety_flags
    assert result.metadata["intent_model_led"]["applied_primary_intent"] == "address"


def test_p0_model_led_filters_confusion_complaint_without_touching_off() -> None:
    message = "А тестирование нужно? Ребёнок в 6 классе, я просто не понимаю, нас уже в группу или сначала тест?"

    assert detect_high_risk_input_markers(message) == ("complaint",)
    assert detect_high_risk_input_markers(message, context={subscription_llm.P0_MODEL_LED_ENV: "1"}) == ()


def test_p0_model_led_prompt_calibrates_complaint_vs_confusion() -> None:
    context = {
        "active_brand": "foton",
        DIRECT_PATH_ENV: "1",
        subscription_llm.P0_MODEL_LED_ENV: "1",
    }

    prompt = subscription_llm._build_direct_path_prompt(
        "А тестирование нужно? Ребёнок в 6 классе, я просто не понимаю, нас уже в группу или сначала тест?",
        context=context,
    )

    assert '"is_p0": false' in prompt
    assert '"p0_kind": "none|payment_dispute|refund|complaint|legal_threat"' in prompt
    assert "Растерянность, уточнение порядка или тревога без претензии" in prompt
    assert "is_p0=false и отвечай полезно по фактам" in prompt


def test_p0_model_led_confusion_is_not_direct_preblock() -> None:
    provider = _DirectPathProvider(
        SubscriptionDraftResult(
            route="bot_answer_self_for_pilot",
            draft_text="Да, сначала подскажем порядок тестирования и записи.",
            risk_level="low",
            metadata={
                "direct_path_model_p0": {
                    "is_p0": False,
                    "risk_level": "low",
                    "p0_kind": "none",
                    "model_reason": "уточнение порядка без жалобы",
                }
            },
        )
    )

    result = provider.build_draft(
        "А тестирование нужно? Ребёнок в 6 классе, я просто не понимаю, нас уже в группу или сначала тест?",
        context={
            "active_brand": "foton",
            DIRECT_PATH_ENV: "1",
            subscription_llm.P0_MODEL_LED_ENV: "1",
            "confirmed_facts": {"testing.foton": "Фотон: тестирование помогает подобрать группу."},
        },
    )

    assert provider.calls == 1
    assert result.route == "bot_answer_self_for_pilot"
    assert "complaint_apology_guarded" not in result.safety_flags
    assert result.metadata["direct_path"]["preblocked"] is False


@pytest.mark.parametrize(
    "message",
    (
        "Жалоба: преподаватель оскорбил ребенка на занятии.",
        "Это безобразие, как накричали на ребёнка.",
        "Ребёнок один остался, никто не подошёл.",
        "Напишу везде какие вы, если не решите ситуацию.",
    ),
)
def test_p0_model_led_literal_complaints_stay_direct_preblock(message: str) -> None:
    provider = _DirectPathProvider(
        SubscriptionDraftResult(route="bot_answer_self_for_pilot", draft_text="Этого текста быть не должно.")
    )

    result = provider.build_draft(
        message,
        context={
            "active_brand": "foton",
            DIRECT_PATH_ENV: "1",
            subscription_llm.P0_MODEL_LED_ENV: "1",
        },
    )

    assert provider.calls == 0
    assert result.route == "manager_only"
    assert result.metadata["direct_path"]["preblocked"] is True
    assert result.metadata["direct_path"]["reason_evidence"]["p0_kind"] == "complaint"
    assert "complaint_apology_guarded" in result.safety_flags


@pytest.mark.parametrize(
    ("message", "expected_code"),
    (
        ("С карты списали дважды, верните деньги.", "payment_dispute"),
        ("Верните деньги обратно.", "refund"),
        ("Если не решите, пойду в суд.", "legal"),
    ),
)
def test_p0_model_led_keeps_refund_legal_payment_preblock(message: str, expected_code: str) -> None:
    provider = _DirectPathProvider(
        SubscriptionDraftResult(route="bot_answer_self_for_pilot", draft_text="Этого текста быть не должно.")
    )

    result = provider.build_draft(
        message,
        context={
            "active_brand": "foton",
            DIRECT_PATH_ENV: "1",
            subscription_llm.P0_MODEL_LED_ENV: "1",
        },
    )

    assert provider.calls == 0
    assert result.route == "manager_only"
    assert result.metadata["direct_path"]["preblocked"] is True
    assert expected_code in result.metadata["direct_path"]["reason_evidence"]["p0_kind"]
    assert "manager_approval_required" in result.safety_flags
    assert "no_auto_send" in result.safety_flags


def test_answerability_shadow_prompt_is_absent_without_pilot_profile() -> None:
    context = {"active_brand": "foton", DIRECT_PATH_ENV: "1"}

    prompt = subscription_llm._build_direct_path_prompt("Можно записаться на пробное?", context=context)

    assert "Теневая самооценка ответуемости" not in prompt
    assert '"can_answer_self"' not in prompt
    assert '"self_missing_facts"' not in prompt
    assert '"supporting_facts"' not in prompt
    assert '"why_manager"' not in prompt


def test_answerability_shadow_profile_keeps_direct_prompt_byte_identical() -> None:
    context = {
        "active_brand": "foton",
        DIRECT_PATH_ENV: "1",
        DIRECT_PATH_PILOT_CONFIG_ENV: DIRECT_PATH_PILOT_CONFIG_VERSION,
    }

    prompt = subscription_llm._build_direct_path_prompt("Можно записаться на пробное?", context=context)
    legacy_override_prompt = subscription_llm._build_direct_path_prompt(
        "Можно записаться на пробное?",
        context={**context, "TELEGRAM_ANSWERABILITY_SHADOW": "0"},
    )

    assert subscription_llm._answerability_shadow_enabled({"TELEGRAM_ANSWERABILITY_SHADOW": "1"}) is False
    assert subscription_llm._answerability_shadow_enabled(
        {**context, "TELEGRAM_ANSWERABILITY_SHADOW": "0"}
    ) is True
    assert prompt == legacy_override_prompt
    assert "Теневая самооценка ответуемости" not in prompt
    assert '"can_answer_self"' not in prompt
    assert '"self_missing_facts"' not in prompt
    assert '"supporting_facts"' not in prompt
    assert '"why_manager"' not in prompt


def test_answerability_trace_off_is_not_added_to_metadata() -> None:
    result = subscription_llm._direct_path_finalize_metadata(
        SubscriptionDraftResult(
            route="draft_for_manager",
            draft_text="Передам менеджеру.",
            metadata={
                "direct_path": {"model_called": True, "reason_class": "policy_permission"},
                "semantic_output_verifier": {"action": "downgrade", "finding_codes": ["unsupported_claim"]},
                "authoritative_output_gate": {
                    "action": "downgrade",
                    "route_before": "bot_answer_self_for_pilot",
                    "route_after": "draft_for_manager",
                    "findings": [{"code": "unbacked_fact", "source": "number_gate"}],
                },
                "answerability_self": {"can_answer_self": "no"},
            },
        ),
        before_gate_route="bot_answer_self_for_pilot",
        client_message="Можно записаться?",
        context={"active_brand": "foton"},
    )

    assert "answerability_trace" not in result.metadata


def test_answerability_trace_on_summarizes_existing_downgrade_causes() -> None:
    result = subscription_llm._direct_path_finalize_metadata(
        SubscriptionDraftResult(
            route="draft_for_manager",
            draft_text="Передам менеджеру.",
            metadata={
                "direct_path": {"model_called": True, "reason_class": "policy_permission"},
                "semantic_output_verifier": {"action": "downgrade", "finding_codes": ["unsupported_claim"]},
                "authoritative_output_gate": {
                    "action": "downgrade",
                    "route_before": "bot_answer_self_for_pilot",
                    "route_after": "draft_for_manager",
                    "findings": [{"code": "unbacked_fact", "source": "number_gate"}],
                },
                "answerability_self": {"can_answer_self": "no"},
            },
        ),
        before_gate_route="bot_answer_self_for_pilot",
        client_message="Можно записаться?",
        context={
            "active_brand": "foton",
            DIRECT_PATH_PILOT_CONFIG_ENV: DIRECT_PATH_PILOT_CONFIG_VERSION,
        },
    )

    trace = result.metadata["answerability_trace"]
    assert trace["route_before_gate"] == "bot_answer_self_for_pilot"
    assert trace["route_after"] == "draft_for_manager"
    assert "semantic_output_verifier" in trace["lowering_layers"]
    assert "authoritative_output_gate" in trace["lowering_layers"]
    assert trace["semantic_output_verifier"]["finding_codes"] == ["unsupported_claim"]
    assert trace["authoritative_output_gate"]["findings"][0]["code"] == "unbacked_fact"
    assert trace["answerability_self"] == {"can_answer_self": "no"}


def test_pilot_gold_answerability_shadow_changes_only_trace_metadata() -> None:
    def source_result() -> SubscriptionDraftResult:
        return SubscriptionDraftResult(
            route="draft_for_manager",
            draft_text="Передам менеджеру, чтобы проверить наличие места.",
            safety_flags=("manager_approval_required",),
            context_used=("schedule.fact",),
            metadata={
                "direct_path": {"model_called": True, "reason_class": "policy_permission"},
                "authoritative_output_gate": {"action": "pass", "findings": []},
                "answerability_self": {
                    "can_answer_self": "no",
                    "why_manager": "нужно проверить наличие места",
                },
            },
        )

    profile_context = {
        "active_brand": "foton",
        DIRECT_PATH_PILOT_CONFIG_ENV: DIRECT_PATH_PILOT_CONFIG_VERSION,
    }
    off = subscription_llm._direct_path_finalize_metadata(
        source_result(),
        before_gate_route="draft_for_manager",
        client_message="Есть места в группе?",
        context={"active_brand": "foton"},
    )
    on = subscription_llm._direct_path_finalize_metadata(
        source_result(),
        before_gate_route="draft_for_manager",
        client_message="Есть места в группе?",
        context=profile_context,
    )

    assert off.route == on.route
    assert off.draft_text == on.draft_text
    assert off.safety_flags == on.safety_flags
    assert off.context_used == on.context_used
    off_metadata = dict(off.metadata)
    on_metadata = dict(on.metadata)
    assert "answerability_trace" not in off_metadata
    assert "answerability_trace" in on_metadata
    on_metadata.pop("answerability_trace")
    assert on_metadata == off_metadata


def test_direct_path_model_p0_off_keeps_previous_route_and_text() -> None:
    provider = _DirectPathProvider(
        SubscriptionDraftResult(
            route="bot_answer_self_for_pilot",
            draft_text="Можем посмотреть скидку и варианты оплаты.",
            risk_level="high",
            metadata={
                "direct_path_model_p0": {
                    "is_p0": True,
                    "risk_level": "high",
                    "p0_kind": "payment_dispute",
                    "model_reason": "клиент пишет про спорную оплату",
                }
            },
        )
    )

    result = provider.build_draft(
        "Нужна помощь по спорной ситуации с оплатой.",
        context={"active_brand": "foton", DIRECT_PATH_ENV: "1"},
    )

    assert provider.calls == 1
    assert result.route == "bot_answer_self_for_pilot"
    assert result.draft_text == "Можем посмотреть скидку и варианты оплаты."
    assert "direct_path_model_p0_payment_dispute" not in result.safety_flags


def test_direct_path_model_p0_payment_dispute_routes_before_gate_and_replaces_sales_text() -> None:
    provider = _DirectPathProvider(
        SubscriptionDraftResult(
            route="bot_answer_self_for_pilot",
            draft_text="Можем посмотреть скидку и варианты оплаты.",
            risk_level="high",
            metadata={
                "direct_path_model_p0": {
                    "is_p0": True,
                    "risk_level": "high",
                    "p0_kind": "payment_dispute",
                    "model_reason": "спорная ситуация с оплатой",
                }
            },
        )
    )

    result = provider.build_draft(
        "Нужна помощь по спорной ситуации с оплатой.",
        context={
            "active_brand": "foton",
            DIRECT_PATH_ENV: "1",
            subscription_llm.DIRECT_PATH_MODEL_P0_ENV: "1",
        },
    )

    gate = result.metadata["authoritative_output_gate"]
    direct_p0 = result.metadata["direct_path_model_p0"]
    assert provider.calls == 1
    assert result.route == "manager_only"
    assert "скидк" not in result.draft_text.casefold()
    assert "варианты оплаты" not in result.draft_text.casefold()
    assert "direct_path_model_p0_payment_dispute" in result.safety_flags
    assert "payment_dispute" in result.safety_flags
    assert direct_p0["p0_kind"] == "payment_dispute"
    assert direct_p0["model_reason"] == "спорная ситуация с оплатой"
    assert direct_p0["floor_reason"] == ""
    assert gate["route_before"] == "manager_only"
    assert gate["action"] == "block"
    assert "hard_p0" in {item["code"] for item in gate["findings"]}


def test_p0_model_led_model_complaint_routes_manager_before_gate() -> None:
    provider = _DirectPathProvider(
        SubscriptionDraftResult(
            route="bot_answer_self_for_pilot",
            draft_text="Давайте посмотрим, какие есть группы и скидки.",
            risk_level="low",
            metadata={
                "direct_path_model_p0": {
                    "is_p0": True,
                    "risk_level": "high",
                    "p0_kind": "complaint",
                    "model_reason": "клиент описывает реальную претензию к занятию",
                }
            },
        )
    )

    result = provider.build_draft(
        "Преподаватель грубо разговаривал с ребёнком, я недовольна занятием.",
        context={
            "active_brand": "foton",
            DIRECT_PATH_ENV: "1",
            subscription_llm.P0_MODEL_LED_ENV: "1",
            subscription_llm.SEMANTIC_OUTPUT_VERIFIER_ENV: "1",
            "confirmed_facts": {"schedule.foton": "Фотон: расписание подбирает менеджер."},
        },
    )

    direct_p0 = result.metadata["direct_path_model_p0"]
    assert provider.calls == 1
    assert result.route == "manager_only"
    assert "скидк" not in result.draft_text.casefold()
    assert "direct_path_model_p0_complaint" in result.safety_flags
    assert "complaint" in result.safety_flags
    assert "manager_approval_required" in result.safety_flags
    assert "no_auto_send" in result.safety_flags
    assert direct_p0["p0_kind"] == "complaint"
    assert direct_p0["source"] == "model_p0"
    assert direct_p0["model_reason"] == "клиент описывает реальную претензию к занятию"


@pytest.mark.parametrize(
    ("kind", "message", "history", "legacy_kind"),
    (
        (
            "cancellation_service_request",
            "Можете подсказать порядок по нашей записи?",
            (),
            "refund",
        ),
        (
            "contract_dispute",
            "Посмотрите, пожалуйста, наш документ.",
            (),
            "legal_threat",
        ),
        (
            "paid_operation_context",
            "Что можно сделать?",
            (
                "Клиент: Мы уже оплатили июльскую смену.",
                "Клиент: Но нужных мест нет, обсуждали перенос на другую смену.",
            ),
            "refund",
        ),
    ),
)
def test_p0_model_classes_v2_routes_new_kinds_to_manager_and_legacy_hard_flags(
    kind: str,
    message: str,
    history: tuple[str, ...],
    legacy_kind: str,
) -> None:
    provider = _DirectPathProvider(
        SubscriptionDraftResult(
            route="bot_answer_self_for_pilot",
            draft_text="Можем подобрать варианты и рассказать про оплату.",
            risk_level="high",
            metadata={
                "direct_path_model_p0": {
                    "is_p0": True,
                    "risk_level": "high",
                    "p0_kind": kind,
                    "model_reason": "модель увидела новый P0-класс",
                }
            },
        )
    )

    result = provider.build_draft(
        message,
        context={
            "active_brand": "foton",
            DIRECT_PATH_ENV: "1",
            subscription_llm.DIRECT_PATH_MODEL_P0_ENV: "1",
            subscription_llm.P0_MODEL_CLASSES_V2_ENV: "1",
            "recent_messages": history,
        },
    )

    direct_p0 = result.metadata["direct_path_model_p0"]
    assert provider.calls == 1
    assert result.route == "manager_only"
    assert "варианты и рассказать про оплату" not in result.draft_text.casefold()
    assert direct_p0["p0_kind"] == kind
    assert direct_p0["legacy_p0_kind"] == legacy_kind
    assert f"direct_path_model_p0_{kind}" in result.safety_flags
    assert kind in result.safety_flags
    assert f"direct_path_model_p0_{legacy_kind}" in result.safety_flags
    assert legacy_kind in result.safety_flags
    assert "manager_approval_required" in result.safety_flags
    assert "no_auto_send" in result.safety_flags
    gate = result.metadata["authoritative_output_gate"]
    assert gate["route_before"] == "manager_only"
    assert "hard_p0" in {item["code"] for item in gate["findings"]}


def test_p0_model_classes_v2_unknown_when_flag_off_falls_back_to_legacy_complaint() -> None:
    provider = _DirectPathProvider(
        SubscriptionDraftResult(
            route="bot_answer_self_for_pilot",
            draft_text="Можем обсудить варианты.",
            risk_level="high",
            metadata={
                "direct_path_model_p0": {
                    "is_p0": True,
                    "risk_level": "high",
                    "p0_kind": "cancellation_service_request",
                    "model_reason": "новый класс пришёл без флага v2",
                }
            },
        )
    )

    result = provider.build_draft(
        "Можете подсказать порядок по нашей записи?",
        context={
            "active_brand": "foton",
            DIRECT_PATH_ENV: "1",
            subscription_llm.DIRECT_PATH_MODEL_P0_ENV: "1",
        },
    )

    direct_p0 = result.metadata["direct_path_model_p0"]
    assert result.route == "manager_only"
    assert direct_p0["p0_kind"] == "complaint"
    assert "direct_path_model_p0_cancellation_service_request" not in result.safety_flags
    assert "direct_path_model_p0_complaint" in result.safety_flags


def test_direct_path_model_p0_latches_next_neutral_turn() -> None:
    first_provider = _DirectPathProvider(
        SubscriptionDraftResult(
            route="bot_answer_self_for_pilot",
            draft_text="Можем посмотреть скидку и варианты оплаты.",
            risk_level="high",
            metadata={
                "direct_path_model_p0": {
                    "is_p0": True,
                    "risk_level": "high",
                    "p0_kind": "payment_dispute",
                    "model_reason": "спорная ситуация с оплатой",
                }
            },
        )
    )
    context = {
        "active_brand": "foton",
        DIRECT_PATH_ENV: "1",
        subscription_llm.DIRECT_PATH_MODEL_P0_ENV: "1",
    }
    first = first_provider.build_draft("Нужна помощь по спорной ситуации с оплатой.", context=context)
    memory = build_dialogue_memory(
        active_brand="foton",
        current_message="Нужна помощь по спорной ситуации с оплатой.",
        context=context,
    )
    memory = update_dialogue_memory_after_answer(
        memory,
        answer_text=first.draft_text,
        route=first.route,
        safety_flags=first.safety_flags,
    )

    follow_provider = _DirectPathProvider(
        SubscriptionDraftResult(route="bot_answer_self_for_pilot", draft_text="Этого текста быть не должно.")
    )
    follow = follow_provider.build_draft(
        "А когда удобно?",
        context={**context, "dialogue_memory_view": memory.to_json_dict()},
    )

    assert memory.p0_latch.active is True
    assert "payment_dispute" in memory.p0_latch.codes
    assert follow_provider.calls == 0
    assert follow.route == "manager_only"
    assert follow.metadata["direct_path"]["preblocked"] is True
    assert follow.metadata["direct_path"]["preblock_reason"] == "p0_pre_gate"


def test_direct_path_model_p0_benign_messages_stay_autonomous() -> None:
    for message in ("Дорого, подумаю.", "А можно вернуть деньги гипотетически, если передумаем?"):
        provider = _DirectPathProvider(
            SubscriptionDraftResult(
                route="bot_answer_self_for_pilot",
                draft_text="Да, подскажу условия по фактам.",
                risk_level="low",
                metadata={
                    "direct_path_model_p0": {
                        "is_p0": False,
                        "risk_level": "low",
                        "p0_kind": "none",
                        "model_reason": "обычный вопрос без претензии",
                    }
                },
            )
        )

        result = provider.build_draft(
            message,
            context={
                "active_brand": "foton",
                DIRECT_PATH_ENV: "1",
                subscription_llm.DIRECT_PATH_MODEL_P0_ENV: "1",
                "confirmed_facts": {"payment.foton": "Фотон: условия оплаты объясняет менеджер."},
            },
        )

        assert provider.calls == 1
        assert result.route == "bot_answer_self_for_pilot"
        assert "direct_path_model_p0_payment_dispute" not in result.safety_flags
        assert "authoritative_gate:hard_p0" not in result.safety_flags


def test_direct_path_prompt_forbids_manager_deadline_and_unconfirmed_phone_for_night_lead() -> None:
    provider = _DirectPathProvider(
        SubscriptionDraftResult(route="draft_for_manager", draft_text="Менеджер свяжется и поможет подобрать группу.")
    )
    provider.build_draft(
        "Сейчас ночь, менеджер завтра утром позвонит? Телефон у вас уже есть?",
        context={
            "active_brand": "foton",
            DIRECT_PATH_ENV: "1",
            "confirmed_facts": {"trial.foton": "Фотон: пробное занятие есть."},
            "recent_messages": ["Клиент: Сейчас ночь, можно записаться?"],
        },
    )

    prompt = provider.last_prompt.casefold()
    assert "не обещай действия и сроки от имени менеджера" in prompt
    assert "«менеджер свяжется» без срока" in prompt
    assert "нельзя «свяжется завтра/утром/в течение n»" in prompt
    assert "не утверждай, что телефон или контакт уже есть" in prompt
    assert "имя ребёнка можно использовать, если" in prompt
    assert "телефон или фио целиком не дублируй" in prompt


def test_route_rubric_enabled_by_pilot_gold_profile(monkeypatch) -> None:
    for key in (subscription_llm.ROUTE_RUBRIC_ENV, DIRECT_PATH_PILOT_CONFIG_ENV):
        monkeypatch.delenv(key, raising=False)

    context = {DIRECT_PATH_PILOT_CONFIG_ENV: DIRECT_PATH_PILOT_CONFIG_VERSION}

    assert subscription_llm.ROUTE_RUBRIC_ENV in subscription_llm.DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS
    assert subscription_llm._route_rubric_enabled(context) is True
    assert subscription_llm._route_rubric_enabled({**context, subscription_llm.ROUTE_RUBRIC_ENV: "0"}) is False
    assert subscription_llm._route_rubric_enabled({"route_rubric_enabled": "1"}) is True


def test_direct_path_prompt_hides_technical_fact_key_deadline() -> None:
    fact_key = "prices_regular_2026_27.online_5_11_class.before_2026_08_01.year"
    fact_text = "Фотон: цены на 2026/27 учебный год, 5-11 класс, онлайн, год — 47 250 ₽."
    context = {
        "active_brand": "foton",
        "confirmed_facts": {fact_key: fact_text},
    }

    prompt = subscription_llm._build_direct_path_prompt("Сколько стоит год?", context=context)
    pack = subscription_llm._direct_path_context_fact_pack(context, client_message="Сколько стоит год?")

    assert fact_key not in prompt
    assert "1 августа" not in prompt.casefold()
    assert fact_text in prompt
    assert fact_key in pack["facts"]
    assert context["confirmed_facts"] == {fact_key: fact_text}


def test_route_rubric_prompt_off_golden_and_on_adds_rubric(monkeypatch) -> None:
    for key in (subscription_llm.ROUTE_RUBRIC_ENV,):
        monkeypatch.delenv(key, raising=False)
    context = {
        "active_brand": "foton",
        "confirmed_facts": {"fact.price": "Фотон: годовой курс стоит 59 000 ₽."},
        "recent_messages": ["Клиент: Сколько стоит?"],
    }

    off_prompt = subscription_llm._build_direct_path_prompt("Сколько стоит?", context=context)
    expected_off = """Ты — менеджер-консультант учебного центра Фотон. Тебе пишет родитель с задачей
про ребёнка. Твоя цель — реально помочь разобраться и довести до записи на
подходящий курс. Продажа — это помощь: польза с первого ответа, предугадывай
следующий вопрос, веди к понятному шагу. Не дави: честность важнее сделки.
Числа, даты и условия — только из фактов; чего нет в фактах — скажи честно
и предложи шаг. Если правило безопасности или передача менеджеру противоречат
записи — правило важнее. Не обещай действия и сроки от имени менеджера: можно
написать «менеджер свяжется» без срока, но нельзя «свяжется завтра/утром/в течение N»
или гарантировать действие. Не утверждай, что телефон или контакт уже есть у центра,
если это не подтверждено в памяти или фактах. Имя ребёнка можно использовать, если
клиент сам его назвал; телефон или ФИО целиком не дублируй.

Дополнение к числам: каждую цену, дату, процент, длительность и количество называй вместе с форматом,
классом или продуктом того факта, из которого взял число. Если скоуп факта не совпадает с вопросом — не называй число.

Активный бренд: Фотон (foton).
Текущее сообщение клиента:
Сколько стоит?

Факты по вашему вопросу:
- Подтверждённый факт: Фотон: годовой курс стоит 59 000 ₽.

Смежные факты — используй только если вопрос реально про это:
(нет подтверждённых фактов в этом блоке)

Память диалога:
{}

Известные слоты:
{}

Последние реплики:
Клиент: Сколько стоит?

Верни только JSON без Markdown и без комментариев:
{
  "route": "bot_answer_self_for_pilot" | "draft_for_manager",
  "draft_text": "текст для клиента",
  "manager_checklist": [],
  "missing_facts": [],
  "context_used": []
}
"""
    assert off_prompt == expected_off

    on_prompt = subscription_llm._build_direct_path_prompt(
        "Сколько стоит?",
        context={**context, subscription_llm.ROUTE_RUBRIC_ENV: "1"},
    )
    expected_on = """Ты — менеджер-консультант учебного центра Фотон. Тебе пишет родитель с задачей
про ребёнка. Твоя цель — реально помочь разобраться и довести до записи на
подходящий курс. Продажа — это помощь: польза с первого ответа, предугадывай
следующий вопрос, веди к понятному шагу. Не дави: честность важнее сделки.
Числа, даты и условия — только из фактов; чего нет в фактах — скажи честно
и предложи шаг. Если правило безопасности или передача менеджеру противоречат
записи — правило важнее. Не обещай действия и сроки от имени менеджера: можно
написать «менеджер свяжется» без срока только в черновике для менеджера, но нельзя «свяжется завтра/утром/в течение N»
или гарантировать действие. Не утверждай, что телефон или контакт уже есть у центра,
если это не подтверждено в памяти или фактах. Имя ребёнка можно использовать, если
клиент сам его назвал; телефон или ФИО целиком не дублируй.

Выбор маршрута:
- "bot_answer_self_for_pilot" — когда факты из блока «Факты по вашему вопросу» покрывают вопрос клиента и не требуется действие менеджера. Отвечай по фактам уверенно и не обещай, что «менеджер свяжется», — ты уже отвечаешь. Смежные факты покрытием НЕ считаются: на их основе самостоятельный ответ не выбирай.
- "draft_for_manager" — когда фактов не хватает, нужно ДЕЙСТВИЕ или проверка менеджера (оформить запись, отправить документы, проверить оплату, персональные данные) или вопрос требует личной оценки. Обязательно заполни missing_facts: какого факта или какой проверки не хватает. В черновике пиши содержательный ответ по фактам для менеджера — а не «передам менеджеру» как весь текст.
Развилка по процессам: РАССКАЗАТЬ, как устроен процесс (как проходит запись, что после оплаты, есть лист ожидания), — это самостоятельный ответ по факту процесса. ВЫПОЛНИТЬ действие по просьбе клиента («запишите меня», «пришлите договор», «проверьте оплату») — это draft_for_manager.
Запрещено вычислять новые числа: не выводи проценты, скидки, суммы и итоги из других цен («за два предмета выйдет…», «это получается N%»). Называй только числа, которые есть в фактах дословно или назвал сам клиент. Не подтверждай расчёты клиента («у меня выходит N, верно?») — точный расчёт и итог по нескольким предметам или со скидками подтвердит менеджер.
Избегай сравнительных оценок форматов/программ без факта («очно удобнее…») — вместо этого предложи признак выбора вопросом.
Запрещено: выбирать "draft_for_manager" на всякий случай при полных фактах.

Дополнение к числам: каждую цену, дату, процент, длительность и количество называй вместе с форматом,
классом или продуктом того факта, из которого взял число. Если скоуп факта не совпадает с вопросом — не называй число.

Активный бренд: Фотон (foton).
Текущее сообщение клиента:
Сколько стоит?

Факты по вашему вопросу:
- Подтверждённый факт: Фотон: годовой курс стоит 59 000 ₽.

Смежные факты — используй только если вопрос реально про это:
(нет подтверждённых фактов в этом блоке)

Память диалога:
{}

Известные слоты:
{}

Последние реплики:
Клиент: Сколько стоит?

Верни только JSON без Markdown и без комментариев:
{
  "route": "bot_answer_self_for_pilot" | "draft_for_manager",
  "draft_text": "текст для клиента",
  "manager_checklist": [],
  "missing_facts": [],
  "context_used": []
}
"""

    assert off_prompt != on_prompt
    assert on_prompt == expected_on
    assert "Выбор маршрута:" in on_prompt
    assert "Смежные факты покрытием НЕ считаются" in on_prompt
    assert "только в черновике для менеджера" in on_prompt
    assert "Запрещено вычислять новые числа" in on_prompt
    assert "Избегай сравнительных оценок форматов/программ без факта" in on_prompt
    assert "можно\nнаписать «менеджер свяжется» без срока, но нельзя" not in on_prompt


def test_route_rubric_regenerates_unjustified_deferral_once() -> None:
    provider = _DirectPathSequenceProvider(
        SubscriptionDraftResult(route="draft_for_manager", draft_text="Передам менеджеру."),
        SubscriptionDraftResult(route="bot_answer_self_for_pilot", draft_text="Годовой курс стоит 59 000 ₽."),
    )

    result = provider.build_draft(
        "Сколько стоит год?",
        context={
            "active_brand": "foton",
            DIRECT_PATH_ENV: "1",
            subscription_llm.ROUTE_RUBRIC_ENV: "1",
            "confirmed_facts": {"fact.price": "Фотон: годовой курс стоит 59 000 ₽."},
        },
    )

    direct = result.metadata["direct_path"]
    assert provider.calls == 2
    assert result.route == "bot_answer_self_for_pilot"
    assert "Предыдущий JSON-ответ модели" in provider.prompts[1]
    assert '"route": "draft_for_manager"' in provider.prompts[1]
    assert "missing_facts пуст" in provider.prompts[1]
    assert direct["rubric_enabled"] is True
    assert direct["rubric_regenerated"] is True
    assert direct["rubric_reason"] == "missing_justification"
    assert direct["direct_path_regenerated"] is False


def test_route_rubric_no_regen_matrix_and_no_code_route_promotion(tmp_path: Path) -> None:
    common = {
        "active_brand": "foton",
        DIRECT_PATH_ENV: "1",
        subscription_llm.ROUTE_RUBRIC_ENV: "1",
        "confirmed_facts": {"fact.price": "Фотон: годовой курс стоит 59 000 ₽."},
    }

    self_provider = _DirectPathSequenceProvider(
        SubscriptionDraftResult(route="bot_answer_self_for_pilot", draft_text="Годовой курс стоит 59 000 ₽.")
    )
    self_result = self_provider.build_draft("Сколько стоит год?", context=common)
    assert self_provider.calls == 1
    assert self_result.route == "bot_answer_self_for_pilot"

    missing_provider = _DirectPathSequenceProvider(
        SubscriptionDraftResult(route="draft_for_manager", draft_text="Нужно проверить.", missing_facts=("наличие мест",))
    )
    missing_result = missing_provider.build_draft("Есть места?", context=common)
    assert missing_provider.calls == 1
    assert missing_result.route == "draft_for_manager"

    no_facts_provider = _DirectPathSequenceProvider(
        SubscriptionDraftResult(route="draft_for_manager", draft_text="Передам менеджеру.")
    )
    no_facts_result = no_facts_provider.build_draft(
        "Неизвестный вопрос",
        context={
            "active_brand": "foton",
            "snapshot_path": str(tmp_path / "missing_snapshot.json"),
            DIRECT_PATH_ENV: "1",
            subscription_llm.ROUTE_RUBRIC_ENV: "1",
        },
    )
    assert no_facts_provider.calls == 1
    assert no_facts_result.metadata["direct_path"]["wide_facts_count"] == 0

    off_provider = _DirectPathSequenceProvider(
        SubscriptionDraftResult(route="draft_for_manager", draft_text="Факт есть, но route не повышаем кодом.")
    )
    off_result = off_provider.build_draft(
        "Сколько стоит год?",
        context={
            "active_brand": "foton",
            DIRECT_PATH_ENV: "1",
            subscription_llm.ROUTE_RUBRIC_ENV: "0",
            "confirmed_facts": {"fact.price": "Фотон: годовой курс стоит 59 000 ₽."},
        },
    )
    assert off_provider.calls == 1
    assert off_result.route == "draft_for_manager"
    assert off_result.metadata["direct_path"]["rubric_enabled"] is False
    assert off_result.metadata["direct_path"]["rubric_regenerated"] is False

    preblock_provider = _DirectPathSequenceProvider(
        SubscriptionDraftResult(route="bot_answer_self_for_pilot", draft_text="Этого текста быть не должно.")
    )
    preblock_result = preblock_provider.build_draft(
        "Сколько стоит?",
        context={
            "active_brand": "unknown",
            DIRECT_PATH_ENV: "1",
            subscription_llm.ROUTE_RUBRIC_ENV: "1",
            "confirmed_facts": {"fact.price": "Фотон: годовой курс стоит 59 000 ₽."},
        },
    )
    assert preblock_provider.calls == 0
    assert preblock_result.metadata["direct_path"]["model_called"] is False


def test_route_rubric_regen_error_keeps_first_result() -> None:
    first = SubscriptionDraftResult(route="draft_for_manager", draft_text="Передам менеджеру.")
    provider = _DirectPathSequenceProvider(first, RuntimeError("temporary outage"))

    result = provider.build_draft(
        "Сколько стоит год?",
        context={
            "active_brand": "foton",
            DIRECT_PATH_ENV: "1",
            subscription_llm.ROUTE_RUBRIC_ENV: "1",
            "confirmed_facts": {"fact.price": "Фотон: годовой курс стоит 59 000 ₽."},
        },
    )

    direct = result.metadata["direct_path"]
    assert provider.calls == 2
    assert result.route == "draft_for_manager"
    assert result.draft_text == first.draft_text
    assert direct["rubric_regenerated"] is False
    assert str(direct["rubric_reason"]).startswith("regen_failed:temporary outage")


def test_route_rubric_regenerated_self_still_passes_authoritative_gate() -> None:
    provider = _DirectPathSequenceProvider(
        SubscriptionDraftResult(route="draft_for_manager", draft_text="Передам менеджеру."),
        SubscriptionDraftResult(route="bot_answer_self_for_pilot", draft_text="Можно оплатить за 2-3 месяца."),
    )

    result = provider.build_draft(
        "Можно оплатить помесячно?",
        context={
            "active_brand": "foton",
            DIRECT_PATH_ENV: "1",
            subscription_llm.ROUTE_RUBRIC_ENV: "1",
            "TELEGRAM_A_FREE_NUMBER_GATE": "1",
            "confirmed_facts": {
                "payment.foton.installment": "Фотон: рассрочка доступна на 6, 10 или 12 месяцев."
            },
        },
    )

    gate = result.metadata["authoritative_output_gate"]
    assert provider.calls == 2
    assert result.route == "manager_only"
    assert gate["action"] == "block"
    assert "unsupported_product_number" in {item["code"] for item in gate["findings"]}
    assert result.metadata["direct_path"]["rubric_regenerated"] is True
    assert result.metadata["direct_path"]["reason_class"] == "output_safety"


def test_route_rubric_deferral_text_in_self_metadata() -> None:
    provider = _DirectPathSequenceProvider(
        SubscriptionDraftResult(route="bot_answer_self_for_pilot", draft_text="Передам вопрос менеджеру.")
    )

    result = provider.build_draft(
        "Спасибо, поняла.",
        context={
            "active_brand": "foton",
            DIRECT_PATH_ENV: "1",
            subscription_llm.ROUTE_RUBRIC_ENV: "1",
            "confirmed_facts": {"fact.process": "Фотон: запись оформляется через менеджера."},
        },
    )

    assert result.route == "bot_answer_self_for_pilot"
    assert result.metadata["direct_path"]["deferral_text_in_self"] is True


def test_direct_path_output_sanitizer_removes_client_phone_and_child_name_echo() -> None:
    provider = _DirectPathProvider(
        SubscriptionDraftResult(
            route="draft_for_manager",
            draft_text="Приняла: Иванов Артём, телефон: +7 999 123-45-67. Передам менеджеру, он свяжется.",
            topic_id="theme:020_enrollment",
        )
    )
    client_message = "Добрый вечер. Ребёнок Иванов Артём, 9 класс. Телефон +7 999 123-45-67. Можно записаться?"

    result = provider.build_draft(
        client_message,
        context={
            "active_brand": "foton",
            DIRECT_PATH_ENV: "1",
            "confirmed_facts": {"enrollment.foton": "Для записи менеджер помогает подобрать группу и оформить заявку."},
        },
    )

    assert provider.calls == 1
    assert result.route == "draft_for_manager"
    assert result.draft_text == "Приняла: Артём, телефон: [данные у менеджера]. Передам менеджеру, он свяжется."
    assert "Иванов" not in result.draft_text
    assert "Артём" in result.draft_text
    assert "+7 999" not in result.draft_text
    assert "123-45-67" not in result.draft_text
    assert result.metadata["output_sanitizer"]["applied"] is True
    assert result.metadata["output_sanitizer"]["enabled"] is False
    assert {"client_phone_echo", "client_name_echo"}.issubset(set(result.metadata["output_sanitizer"]["reasons"]))
    assert result.metadata["authoritative_output_gate"]["action"] == "pass"
    assert result.metadata["direct_path"]["model_called"] is True
    assert result.metadata["direct_path"]["downgraded"] is False


def test_direct_path_output_sanitizer_allows_named_child_name_declension() -> None:
    provider = _DirectPathProvider(
        SubscriptionDraftResult(
            route="draft_for_manager",
            draft_text="Поняла. Подскажите, какой предмет нужен Петру?",
            topic_id="theme:020_enrollment",
        )
    )
    client_message = "Записывайте: Иванов Пётр, 9 класс. Хотим подобрать курс."

    result = provider.build_draft(
        client_message,
        context={
            "active_brand": "foton",
            DIRECT_PATH_ENV: "1",
            "confirmed_facts": {"enrollment.foton": "Для записи менеджер помогает подобрать группу и оформить заявку."},
        },
    )

    assert provider.calls == 1
    assert result.route == "draft_for_manager"
    assert "Петру" in result.draft_text
    assert "данные ребёнка" not in result.draft_text
    assert "output_sanitizer" not in result.metadata or result.metadata["output_sanitizer"].get("applied") is not True
    assert result.metadata["authoritative_output_gate"]["checked"] is True


def test_direct_path_output_sanitizer_allows_client_names_from_recent_window() -> None:
    provider = _DirectPathProvider(
        SubscriptionDraftResult(
            route="draft_for_manager",
            draft_text="Спасибо, Ирина! По сыну Артёму менеджер подберёт группу.",
            topic_id="theme:020_enrollment",
        )
    )

    result = provider.build_draft(
        "Спасибо, жду.",
        context={
            "active_brand": "foton",
            DIRECT_PATH_ENV: "1",
            "confirmed_facts": {"enrollment.foton": "Для записи менеджер помогает подобрать группу и оформить заявку."},
            "recent_messages": [
                "Клиент: Я Ирина, мама Артёма.",
                "Ответ: Подскажу по записи.",
            ],
        },
    )

    assert provider.calls == 1
    assert result.route == "draft_for_manager"
    assert "Ирина" in result.draft_text
    assert "Артёму" in result.draft_text
    assert "output_sanitizer" not in result.metadata or result.metadata["output_sanitizer"].get("applied") is not True
    assert result.metadata["authoritative_output_gate"]["checked"] is True


def test_direct_path_output_sanitizer_masks_unmentioned_child_name() -> None:
    provider = _DirectPathProvider(
        SubscriptionDraftResult(
            route="bot_answer_self_for_pilot",
            draft_text="Записала Артёма на консультацию по математике.",
            topic_id="theme:020_enrollment",
        )
    )

    result = provider.build_draft(
        "Запишите Кирилла на математику.",
        context={
            "active_brand": "foton",
            DIRECT_PATH_ENV: "1",
            "confirmed_facts": {"enrollment.foton": "Для записи менеджер помогает подобрать группу и оформить заявку."},
        },
    )

    assert provider.calls == 1
    assert result.route == "bot_answer_self_for_pilot"
    assert "Артёма" not in result.draft_text
    assert "данные ребёнка" in result.draft_text
    assert "client_name_echo" in result.metadata["output_sanitizer"]["reasons"]
    assert any("Артёма" in item for item in result.manager_checklist)
    assert result.metadata["authoritative_output_gate"]["checked"] is True


def test_direct_path_output_sanitizer_masks_client_phone_from_recent_window() -> None:
    provider = _DirectPathProvider(
        SubscriptionDraftResult(
            route="draft_for_manager",
            draft_text="Контакт +7 999 123-45-67 передам менеджеру.",
            topic_id="theme:020_enrollment",
        )
    )

    result = provider.build_draft(
        "Спасибо, жду.",
        context={
            "active_brand": "foton",
            DIRECT_PATH_ENV: "1",
            "confirmed_facts": {"enrollment.foton": "Для записи менеджер помогает подобрать группу и оформить заявку."},
            "dialogue_memory_view": {
                "recent_turns": [
                    {"role": "client", "text": "Телефон +7 999 123-45-67, меня зовут Ирина."},
                    {"role": "bot", "text": "Передам менеджеру."},
                ]
            },
        },
    )

    assert provider.calls == 1
    assert result.route == "draft_for_manager"
    assert "+7 999" not in result.draft_text
    assert "123-45-67" not in result.draft_text
    assert result.draft_text == "Записала, передам менеджеру — он свяжется с вами."
    assert "client_phone_echo" in result.metadata["output_sanitizer"]["reasons"]
    assert result.metadata["authoritative_output_gate"]["checked"] is True


def test_presale_direct_path_prompt_filters_pii_slots_but_keeps_safe_slots() -> None:
    provider = _DirectPathProvider(
        SubscriptionDraftResult(route="bot_answer_self_for_pilot", draft_text="Да, подскажу по физике.")
    )

    provider.build_draft(
        "Подскажите курс.",
        context={
            "active_brand": "foton",
            DIRECT_PATH_ENV: "1",
            PRESALE_PII_MEMORY_ENV: "1",
            "known_slots": {
                "subject": "физика",
                "grade": "9",
                "client_name": "Ирина",
                "phone": "+7 999 123-45-67",
            },
            "dialogue_memory_view": {
                "known_slots": {"subject": "физика", "client_name": "Ирина"},
                "crm_known_slots": {"child_name": "Артём", "phone": "+7 999 123-45-67"},
                "conversation_summary_short": "Ирина просит курс для Артёма, телефон +7 999 123-45-67.",
            },
            "confirmed_facts": {"format.foton": "Фотон: есть очные и онлайн-занятия."},
        },
    )

    assert provider.calls == 1
    assert "физика" in provider.last_prompt
    assert '"grade": "9"' in provider.last_prompt
    assert "Ирина" not in provider.last_prompt
    assert "Артём" in provider.last_prompt
    assert "+7 999" not in provider.last_prompt
    assert "conversation_summary_short" not in provider.last_prompt


def test_presale_direct_path_prompt_masks_current_and_recent_pii_but_keeps_child_first_name() -> None:
    provider = _DirectPathProvider(
        SubscriptionDraftResult(route="draft_for_manager", draft_text="Пётр, передам заявку менеджеру.")
    )

    provider.build_draft(
        "Запишите нас: Иванов Пётр, 9 класс, телефон 8-900-123-45-67",
        context={
            "active_brand": "unpk",
            DIRECT_PATH_ENV: "1",
            PRESALE_PII_MEMORY_ENV: "1",
            "recent_messages": [
                "Клиент: Родитель: Иванова Мария Сергеевна, почта maria@example.com",
            ],
            "confirmed_facts": {"enrollment.unpk": "УНПК: менеджер помогает оформить заявку."},
        },
    )

    assert provider.calls == 1
    assert "Пётр" in provider.last_prompt
    assert "Иванов" not in provider.last_prompt
    assert "Иванова" not in provider.last_prompt
    assert "Мария" not in provider.last_prompt
    assert "8-900" not in provider.last_prompt
    assert "maria@example.com" not in provider.last_prompt


def test_presale_output_sanitizer_masks_names_from_memory_slots() -> None:
    provider = _DirectPathProvider(
        SubscriptionDraftResult(
            route="draft_for_manager",
            draft_text="Спасибо, Ирина! По сыну Артёму менеджер подберёт группу.",
            topic_id="theme:020_enrollment",
        )
    )

    result = provider.build_draft(
        "Спасибо, жду.",
        context={
            "active_brand": "foton",
            DIRECT_PATH_ENV: "1",
            PRESALE_PII_MEMORY_ENV: "1",
            "confirmed_facts": {"enrollment.foton": "Для записи менеджер помогает подобрать группу и оформить заявку."},
            "dialogue_memory_view": {
                "crm_known_slots": {"client_name": "Ирина", "child_name": "Артём"},
            },
        },
    )

    assert provider.calls == 1
    assert result.draft_text == "Спасибо, [данные у менеджера]! По сыну Артёму менеджер подберёт группу."
    assert "Ирина" not in result.draft_text
    assert "Артём" in result.draft_text
    assert "client_name_echo" in result.metadata["output_sanitizer"]["reasons"]
    assert any("Ирина" in item and "Артём" in item for item in result.manager_checklist)


def test_presale_output_sanitizer_masks_inflected_single_names_from_memory_slots() -> None:
    provider = _DirectPathProvider(
        SubscriptionDraftResult(
            route="draft_for_manager",
            draft_text="Передайте Ирине: для Артёма есть группа.",
            topic_id="theme:020_enrollment",
        )
    )

    result = provider.build_draft(
        "Спасибо, жду.",
        context={
            "active_brand": "foton",
            DIRECT_PATH_ENV: "1",
            PRESALE_PII_MEMORY_ENV: "1",
            "confirmed_facts": {"enrollment.foton": "Для записи менеджер помогает подобрать группу и оформить заявку."},
            "dialogue_memory_view": {
                "crm_known_slots": {"client_name": "Ирина", "child_name": "Артём"},
            },
        },
    )

    assert provider.calls == 1
    assert "Ирине" not in result.draft_text
    assert "Артёма" in result.draft_text
    assert "[данные у менеджера]" in result.draft_text
    assert "client_name_echo" in result.metadata["output_sanitizer"]["reasons"]


def test_presale_output_sanitizer_keeps_child_first_name_and_moves_full_pii_to_checklist() -> None:
    provider = _DirectPathProvider(
        SubscriptionDraftResult(
            route="draft_for_manager",
            draft_text="Записала Иванова Петра, телефон 8-900-123-45-67 и почту maria@example.com передам менеджеру.",
            topic_id="theme:020_enrollment",
        )
    )

    result = provider.build_draft(
        "Запишите нас: Иванов Пётр, 9 класс, телефон 8-900-123-45-67, почта maria@example.com",
        context={
            "active_brand": "unpk",
            DIRECT_PATH_ENV: "1",
            PRESALE_PII_MEMORY_ENV: "1",
            "confirmed_facts": {"enrollment.unpk": "УНПК: менеджер помогает оформить заявку."},
        },
    )

    assert provider.calls == 1
    assert "Пётр" in result.draft_text
    assert "Иванов" not in result.draft_text
    assert "8-900" not in result.draft_text
    assert "maria@example.com" not in result.draft_text
    assert "[данные у менеджера]" in result.draft_text
    assert "client_name_echo" in result.metadata["output_sanitizer"]["reasons"]
    assert "client_phone_echo" in result.metadata["output_sanitizer"]["reasons"]
    assert "client_email_echo" in result.metadata["output_sanitizer"]["reasons"]
    checklist = "\n".join(result.manager_checklist)
    assert "Иванов Пётр" in checklist
    assert "8-900-123-45-67" in checklist
    assert "maria@example.com" in checklist


def test_direct_path_output_sanitizer_keeps_capitalized_non_name_words() -> None:
    provider = _DirectPathProvider(
        SubscriptionDraftResult(
            route="draft_for_manager",
            draft_text="Да, Москва подходит как ориентир по площадке.",
            topic_id="theme:015_address",
        )
    )
    client_message = "Москва удобна, подскажите площадку."

    result = provider.build_draft(
        client_message,
        context={
            "active_brand": "foton",
            DIRECT_PATH_ENV: "1",
            "confirmed_facts": {"address.foton": "Занятия проходят в Москве."},
            "recent_messages": ["Клиент: Я Москва использую как ориентир по дороге."],
        },
    )

    assert provider.calls == 1
    assert result.route == "draft_for_manager"
    assert result.draft_text == "Да, Москва подходит как ориентир по площадке."
    assert "output_sanitizer" not in result.metadata or result.metadata["output_sanitizer"].get("applied") is not True


def test_pii_relation_stopwords_flag_keeps_family_words(monkeypatch) -> None:
    text = "У меня сын в 7 классе и дочь в 4-м"

    monkeypatch.setenv(DIRECT_PATH_PILOT_CONFIG_ENV, DIRECT_PATH_PILOT_CONFIG_VERSION)
    monkeypatch.setenv(subscription_llm.PII_RELATION_STOPWORDS_ENV, "0")
    off_text, off_reasons = subscription_llm._sanitize_client_pii_echo(text, client_message=text)
    assert off_text == "У меня [данные у менеджера] в 7 классе и дочь в 4-м"
    assert "client_name_echo" in off_reasons

    monkeypatch.delenv(subscription_llm.PII_RELATION_STOPWORDS_ENV, raising=False)
    on_text, on_reasons = subscription_llm._sanitize_client_pii_echo(text, client_message=text)
    assert on_text == text
    assert on_reasons == ()


def test_pii_relation_stopwords_flag_still_masks_unmentioned_name(monkeypatch) -> None:
    monkeypatch.setenv(DIRECT_PATH_PILOT_CONFIG_ENV, DIRECT_PATH_PILOT_CONFIG_VERSION)
    monkeypatch.delenv(subscription_llm.PII_RELATION_STOPWORDS_ENV, raising=False)

    sanitized, reasons = subscription_llm._sanitize_client_pii_echo(
        "Для Ирины подберём группу.",
        client_message="Спасибо, жду.",
    )

    assert "Ирины" not in sanitized
    assert "данные ребёнка" in sanitized
    assert "client_name_echo" in reasons


def test_pii_sanitizer_keeps_address_toponyms(monkeypatch) -> None:
    monkeypatch.setenv(DIRECT_PATH_PILOT_CONFIG_ENV, DIRECT_PATH_PILOT_CONFIG_VERSION)

    sanitized, reasons = subscription_llm._sanitize_client_pii_echo(
        "Для Сретенка, 20 ближайшее метро — Чистые Пруды.",
        client_message="Как доехать до московской площадки УНПК?",
    )

    assert sanitized == "Для Сретенка, 20 ближайшее метро — Чистые Пруды."
    assert reasons == ()


def test_direct_path_p0_preblock_stays_manager_only_with_output_sanitizer() -> None:
    provider = _DirectPathProvider(
        SubscriptionDraftResult(route="bot_answer_self_for_pilot", draft_text="Этого текста быть не должно.")
    )

    result = provider.build_draft(
        "С карты списали дважды, верните деньги. Телефон +7 999 123-45-67",
        context={"active_brand": "foton", DIRECT_PATH_ENV: "1", OUTPUT_SANITIZER_ENV: "1"},
    )

    assert provider.calls == 0
    assert result.route == "manager_only"
    assert result.metadata["direct_path"]["preblocked"] is True
    assert result.metadata["direct_path"]["reason_class"] == "p0_deferral"
    assert result.metadata["authoritative_output_gate"]["checked"] is True


def test_direct_path_overrides_pipeline_and_keeps_clean_close() -> None:
    provider = _DirectPathProvider(
        SubscriptionDraftResult(
            route="bot_answer_self_for_pilot",
            draft_text="Рада была помочь! Возвращайтесь, если появятся вопросы.",
        )
    )
    result = provider.build_draft(
        "Спасибо!",
        context={
            "active_brand": "foton",
            DIRECT_PATH_ENV: "1",
            "TELEGRAM_DIALOGUE_CONTRACT_PIPELINE": "1",
            "confirmed_facts": {"trial.foton": "Фотон: пробное занятие есть."},
        },
    )

    assert provider.calls == 1
    assert "Ты — менеджер-консультант учебного центра Фотон" in provider.last_prompt
    assert "Фотон: пробное занятие есть." in provider.last_prompt
    assert result.route == "bot_answer_self_for_pilot"
    assert result.draft_text == "Рада была помочь! Возвращайтесь, если появятся вопросы."
    assert "dialogue_contract_pipeline" not in result.metadata
    assert "close_detect" not in result.metadata
    assert result.metadata["direct_path"]["text_composition_source"] == "direct_path_model"
    assert result.metadata["direct_path"]["wide_facts_count"] == 1
    assert result.metadata["direct_path"]["selected_category"] == "legacy_context"


def test_direct_path_unsupported_product_number_is_downgraded_by_gate() -> None:
    provider = _DirectPathProvider(
        SubscriptionDraftResult(
            route="bot_answer_self_for_pilot",
            draft_text="Можно оплатить за 2-3 месяца, так будет удобнее.",
        )
    )
    result = provider.build_draft(
        "Можно оплатить помесячно?",
        context={
            "active_brand": "foton",
            DIRECT_PATH_ENV: "1",
            "TELEGRAM_A_FREE_NUMBER_GATE": "1",
            "confirmed_facts": {
                "payment.foton.installment": "Фотон: рассрочка доступна на 6, 10 или 12 месяцев."
            },
        },
    )

    gate = result.metadata["authoritative_output_gate"]
    assert result.route == "manager_only"
    assert gate["action"] == "block"
    assert result.draft_text == SAFE_FALLBACK_DRAFT_TEXT
    assert "unsupported_product_number" in {item["code"] for item in gate["findings"]}
    assert result.metadata["direct_path"]["downgraded"] is True
    assert result.metadata["direct_path"]["reason_class"] == "output_safety"


def test_direct_path_derived_product_number_keeps_text_with_addressed_checklist() -> None:
    provider = _DirectPathProvider(
        SubscriptionDraftResult(
            route="bot_answer_self_for_pilot",
            draft_text="За два предмета выйдет 181 740 ₽, это выгоднее.",
        )
    )
    result = provider.build_draft(
        "Сколько будет за два предмета?",
        context={
            "active_brand": "foton",
            DIRECT_PATH_ENV: "1",
            "confirmed_facts": {
                "price.semester": "Фотон: семестр стоит 49 000 ₽.",
                "price.year": "Фотон: год стоит 82 000 ₽.",
            },
        },
    )

    gate = result.metadata["authoritative_output_gate"]
    codes = {item["code"] for item in gate["findings"]}
    assert result.route == "draft_for_manager"
    assert result.draft_text == "За два предмета выйдет 181 740 ₽, это выгоднее."
    assert gate["action"] == "downgrade_keep_text"
    assert "derived_product_number" in codes
    assert "direct_path_gate_text_preserved" in result.safety_flags
    assert any("Проверьте 181 740 ₽ — вычислено ботом, в прайсе нет." == item for item in result.manager_checklist)
    assert "derived_product_number" not in subscription_llm.DIRECT_PATH_REPLACE_TEXT_GATE_CODES


def test_direct_path_derived_product_number_allows_fact_and_client_numbers() -> None:
    fact_provider = _DirectPathProvider(
        SubscriptionDraftResult(
            route="bot_answer_self_for_pilot",
            draft_text="Семестр — 49 000 ₽, год — 82 000 ₽.",
        )
    )
    fact_result = fact_provider.build_draft(
        "Сколько стоит очно?",
        context={
            "active_brand": "unpk",
            DIRECT_PATH_ENV: "1",
            "confirmed_facts": {
                "price.offline": "УНПК: очные группы стоят: семестр — 49 000 ₽, год — 82 000 ₽."
            },
        },
    )
    fact_gate = fact_result.metadata["authoritative_output_gate"]
    assert "derived_product_number" not in {item["code"] for item in fact_gate["findings"]}

    client_provider = _DirectPathProvider(
        SubscriptionDraftResult(
            route="bot_answer_self_for_pilot",
            draft_text="16,4% как точный итог не подтверждаю: менеджер сверит расчёт.",
        )
    )
    client_result = client_provider.build_draft(
        "У меня выходит 16,4%, верно?",
        context={
            "active_brand": "foton",
            DIRECT_PATH_ENV: "1",
            "confirmed_facts": {"price.year": "Фотон: год стоит 82 000 ₽."},
        },
    )
    client_gate = client_result.metadata["authoritative_output_gate"]
    assert "derived_product_number" not in {item["code"] for item in client_gate["findings"]}


def test_direct_path_hard_gate_generic_replacement_avoids_repeat() -> None:
    provider = _DirectPathProvider(
        SubscriptionDraftResult(
            route="bot_answer_self_for_pilot",
            draft_text="Можно оплатить за 2-3 месяца, так будет удобнее.",
        )
    )
    result = provider.build_draft(
        "Можно оплатить помесячно?",
        context={
            "active_brand": "foton",
            DIRECT_PATH_ENV: "1",
            "TELEGRAM_A_FREE_NUMBER_GATE": "1",
            "confirmed_facts": {
                "payment.foton.installment": "Фотон: рассрочка доступна на 6, 10 или 12 месяцев."
            },
            "recent_messages": [f"Ответ: {SAFE_FALLBACK_DRAFT_TEXT}"],
        },
    )

    assert result.route == "manager_only"
    assert result.draft_text != SAFE_FALLBACK_DRAFT_TEXT
    assert "менеджер" in result.draft_text.casefold()


def test_direct_path_soft_gate_finding_keeps_model_text_for_manager() -> None:
    text = "Очная площадка на Сретенке."
    provider = _DirectPathProvider(
        SubscriptionDraftResult(
            route="bot_answer_self_for_pilot",
            draft_text=text,
        )
    )
    result = provider.build_draft(
        "Где вы находитесь?",
        context={
            "active_brand": "foton",
            DIRECT_PATH_ENV: "1",
            "confirmed_facts": {
                "address.foton": "Фотон: очная площадка — Москва, Верхняя Красносельская ул., 30."
            },
        },
    )

    gate = result.metadata["authoritative_output_gate"]
    codes = {item["code"] for item in gate["findings"]}
    assert provider.calls == 1
    assert result.route == "draft_for_manager"
    assert result.draft_text == text
    assert gate["action"] == "downgrade_keep_text"
    assert "unsupported_entity" in codes
    assert "direct_path_gate_text_preserved" in result.safety_flags
    assert result.metadata["direct_path"]["downgraded"] is True
    assert result.metadata["direct_path"]["reason_class"] == "output_safety"


def test_direct_path_brand_leak_is_downgraded_by_gate() -> None:
    provider = _DirectPathProvider(
        SubscriptionDraftResult(
            route="bot_answer_self_for_pilot",
            draft_text="В Фотоне есть пробное, а в УНПК условия похожие.",
        )
    )
    result = provider.build_draft(
        "Есть пробное?",
        context={
            "active_brand": "foton",
            DIRECT_PATH_ENV: "1",
            "confirmed_facts": {"trial.foton": "Фотон: пробное занятие есть."},
        },
    )

    gate = result.metadata["authoritative_output_gate"]
    assert result.route == "manager_only"
    assert gate["action"] == "block"
    assert result.draft_text == SAFE_FALLBACK_DRAFT_TEXT
    assert "brand_leak" in {item["code"] for item in gate["findings"]}
    assert result.metadata["direct_path"]["downgraded"] is True


def test_direct_path_real_manager_gold_pack_lints_examples() -> None:
    payload = yaml.safe_load(DIRECT_PATH_REAL_MANAGER_GOLD_PACK_PATH.read_text(encoding="utf-8"))
    examples = payload["examples"]

    assert len(examples) == 12
    assert payload["source"] == "real_manager_tg"
    for item in examples:
        assert item["mission_gold"] is True
        assert item["brand"] in {"foton", "unpk"}
        manager_text = item["manager_response_masked"]
        prompt_example = item["prompt_example"]
        assert "₽" not in manager_text
        assert "+7" not in manager_text
        assert "8 (" not in manager_text
        assert "[" in manager_text and "]" in manager_text or item["topic"] in {"close", "docs", "enrollment", "join_mid", "payment_flex", "value"}
        assert "[" not in prompt_example and "]" not in prompt_example
        if item["brand"] == "foton":
            assert "УНПК" not in manager_text
        if item["brand"] == "unpk":
            assert "Фотон" not in manager_text


def test_direct_path_real_manager_gold_v2_pack_lints_examples() -> None:
    pack_path = (
        DIRECT_PATH_REAL_MANAGER_GOLD_PACK_PATH.parent
        / "real_manager_gold_v2_2026-06-11.yaml"
    )
    payload = yaml.safe_load(pack_path.read_text(encoding="utf-8"))
    examples = payload["examples"]

    assert len(examples) == 38
    assert payload["schema_version"] == "real_manager_gold_v2_2026_06_11"
    by_id = {item["id"]: item for item in examples}
    assert "Да, вариант на полгода есть" in by_id["foton_installment_01"]["manager_response_masked"]
    assert "неделя ничего не решает" in by_id["foton_think_over_camp_01"]["manager_response_masked"]
    assert "следит куратор" in by_id["unpk_anxiety_join_mid_01"]["manager_response_masked"]

    for item in examples:
        assert item["mission_gold"] is True
        assert item["brand"] in {"foton", "unpk"}
        assert item.get("source")
        manager_text = item["manager_response_masked"]
        prompt_example = item["prompt_example"]
        assert "₽" not in manager_text
        assert "+7" not in manager_text
        assert "8 (" not in manager_text
        bare_numbers = []
        for match in re.finditer(r"\b\d+\b", manager_text):
            before = manager_text[: match.start()]
            if before.rfind("[") <= before.rfind("]"):
                bare_numbers.append(match.group(0))
        assert bare_numbers == []
        assert "[" not in prompt_example and "]" not in prompt_example
        if item["brand"] == "foton":
            assert "УНПК" not in manager_text
        if item["brand"] == "unpk":
            assert "Фотон" not in manager_text


def test_direct_path_real_manager_gold_is_gated_by_flag() -> None:
    provider = _DirectPathProvider(
        SubscriptionDraftResult(route="bot_answer_self_for_pilot", draft_text="Да, подскажу по рассрочке.")
    )
    provider.build_draft(
        "Рассрочка есть?",
        context={
            "active_brand": "foton",
            DIRECT_PATH_ENV: "1",
            "confirmed_facts": {"installment.foton": "Фотон: доступны варианты на 6, 10 или 12 месяцев."},
        },
    )

    assert "Живые образцы менеджерского стиля" not in provider.last_prompt

    provider_with_gold = _DirectPathProvider(
        SubscriptionDraftResult(route="bot_answer_self_for_pilot", draft_text="Да, подскажу по рассрочке.")
    )
    result = provider_with_gold.build_draft(
        "Рассрочка есть?",
        context={
            "active_brand": "foton",
            DIRECT_PATH_ENV: "1",
            BOT_GOLD_REAL_ENV: "1",
            "conversation_intent_plan": {"primary_intent": "installment"},
            "confirmed_facts": {"installment.foton": "Фотон: доступны варианты на 6, 10 или 12 месяцев."},
        },
    )

    assert "Живые образцы менеджерского стиля" in provider_with_gold.last_prompt
    assert result.metadata["direct_path"]["gold_real_enabled"] is True
    assert result.metadata["direct_path"]["gold_real_example_ids"]


def test_direct_path_real_manager_gold_pack_env_overrides_examples(monkeypatch) -> None:
    pack_path = (
        DIRECT_PATH_REAL_MANAGER_GOLD_PACK_PATH.parent
        / "real_manager_gold_v2_2026-06-11.yaml"
    )
    monkeypatch.setenv(subscription_llm.BOT_GOLD_REAL_PACK_ENV, str(pack_path))

    provider = _DirectPathProvider(
        SubscriptionDraftResult(route="bot_answer_self_for_pilot", draft_text="Да, подскажу по рассрочке.")
    )
    result = provider.build_draft(
        "Рассрочка на полгода есть?",
        context={
            "active_brand": "foton",
            DIRECT_PATH_ENV: "1",
            BOT_GOLD_REAL_ENV: "1",
            "conversation_intent_plan": {"primary_intent": "installment"},
            "confirmed_facts": {"installment.foton": "Фотон: доступны варианты на 6, 10 или 12 месяцев."},
        },
    )

    assert "Да, вариант на полгода есть" in provider.last_prompt
    assert "foton_installment_01" in result.metadata["direct_path"]["gold_real_example_ids"]
    assert result.metadata["direct_path"]["gold_pack_version"] == "real_manager_gold_v2_2026-06-11"


def test_direct_path_pilot_gold_v1_enables_direct_and_gold_without_extra_flags() -> None:
    provider = _DirectPathProvider(
        SubscriptionDraftResult(route="bot_answer_self_for_pilot", draft_text="Да, подскажу по рассрочке.")
    )
    result = provider.build_draft(
        "Рассрочка есть?",
        context={
            "active_brand": "foton",
            DIRECT_PATH_PILOT_CONFIG_ENV: DIRECT_PATH_PILOT_CONFIG_VERSION,
            "confirmed_facts": {"installment.foton": "Фотон: доступны варианты на 6, 10 или 12 месяцев."},
        },
    )

    assert provider.calls == 1
    assert "Живые образцы менеджерского стиля" in provider.last_prompt
    assert result.metadata["direct_path"]["pilot_config"] == DIRECT_PATH_PILOT_CONFIG_VERSION
    assert result.metadata["direct_path"]["gold_real_enabled"] is True


def test_pilot_gold_v1_enables_full_battle_profile_flags(monkeypatch) -> None:
    for key in (
        DIRECT_PATH_ENV,
        BOT_GOLD_REAL_ENV,
        SEMANTIC_OUTPUT_VERIFIER_ENV,
        OUTPUT_SANITIZER_ENV,
        LLM_RETRIEVE_ENV,
        NUMBER_GATE_SCOPE_AWARE_ENV,
        VERIFIER_HANDOFF_CLAIMS_ENV,
        PRESALE_SAFETY_ENV,
        PRESALE_PII_MEMORY_ENV,
        PRESALE_VERIFIER_FAILSOFT_ENV,
        PRESALE_META_RU_ENV,
        PRESALE_SOURCE_ID_ENV,
        subscription_llm.DEAL_ACTION_DECISION_ENV,
        subscription_llm.DIRECT_PATH_MODEL_P0_ENV,
        subscription_llm.MEMORY_PROVENANCE_ENV,
        subscription_llm.MEMORY_PROVENANCE_COMPACT_ENV,
        subscription_llm.PII_RELATION_STOPWORDS_ENV,
        subscription_llm.MEMORY_CHILD_ELLIPSIS_ENV,
        subscription_llm.PRICE_AXES_SELECTOR_ENV,
        subscription_llm.PRICE_AXES_CLEAN_DEFER_ENV,
        ASSUMED_SCOPE_GUARD_ENV,
        RETRIEVER_MODEL_DRIVEN_ENV,
        RETRIEVER_NEED_SHADOW_ENV,
        AUTONOMY_SCOPE_PRECISION_ENV,
        TEMPLATE_FROM_KB_ENV,
        DIRECT_PATH_PILOT_CONFIG_ENV,
    ):
        monkeypatch.delenv(key, raising=False)

    context = {DIRECT_PATH_PILOT_CONFIG_ENV: DIRECT_PATH_PILOT_CONFIG_VERSION}
    legacy_context: dict[str, str] = {}

    assert _direct_path_gold_real_enabled(context) is True
    assert _semantic_output_verifier_enabled(context) is True
    assert _output_sanitizer_enabled(context) is True
    assert subscription_llm._llm_retrieve_enabled(context) is True
    assert number_gate_scope_aware_enabled(context) is True
    assert autonomy_scope_precision_enabled(context) is True
    assert _verifier_handoff_claims_enabled(context) is True
    assert _presale_safety_enabled(context, subflag=PRESALE_PII_MEMORY_ENV) is True
    assert _presale_safety_enabled(context, subflag=PRESALE_VERIFIER_FAILSOFT_ENV) is True
    assert _presale_safety_enabled(context, subflag=PRESALE_META_RU_ENV) is True
    assert _presale_safety_enabled(context, subflag=PRESALE_SOURCE_ID_ENV) is True
    assert subscription_llm._deal_action_decision_enabled(context) is True
    assert subscription_llm._direct_path_model_p0_enabled(context) is True
    assert subscription_llm._deal_action_decision_enabled(legacy_context) is False
    assert subscription_llm._direct_path_model_p0_enabled(legacy_context) is False
    assert subscription_llm._template_from_kb_enabled(context) is True
    assert TONE_CLOSE_DETECT_ENV in subscription_llm.DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS
    assert TONE_RICH_FORMAT_ENV in subscription_llm.DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS
    assert subscription_llm.A_RICH_FORMAT_ENV in subscription_llm.DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS
    assert subscription_llm._a2_rich_format_enabled(context) is True
    assert subscription_llm.MEMORY_PROVENANCE_ENV in subscription_llm.DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS
    assert subscription_llm.MEMORY_PROVENANCE_COMPACT_ENV in subscription_llm.DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS
    assert subscription_llm.PII_RELATION_STOPWORDS_ENV in subscription_llm.DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS
    assert subscription_llm.MEMORY_CHILD_ELLIPSIS_ENV in subscription_llm.DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS
    assert subscription_llm.PRICE_AXES_SELECTOR_ENV in subscription_llm.DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS
    assert subscription_llm.PRICE_AXES_CLEAN_DEFER_ENV in subscription_llm.DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS
    assert AUTONOMY_SCOPE_PRECISION_ENV in subscription_llm.DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS
    assert ASSUMED_SCOPE_GUARD_ENV not in subscription_llm.DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS
    assert RETRIEVER_NEED_SHADOW_ENV not in subscription_llm.DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS
    assert RETRIEVER_MODEL_DRIVEN_ENV not in subscription_llm.DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS
    assert subscription_llm.DEAL_ACTION_DECISION_ENV in subscription_llm.DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS
    assert subscription_llm.DIRECT_PATH_MODEL_P0_ENV in subscription_llm.DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS
    assert subscription_llm.P0_MODEL_LED_ENV in subscription_llm.DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS
    assert subscription_llm.P0_MODEL_CLASSES_V2_ENV in subscription_llm.DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS
    assert subscription_llm.DIRECT_P0_TEXT_HYGIENE_ENV in subscription_llm.DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS
    assert subscription_llm._assumed_scope_guard_enabled(context) is False
    assert subscription_llm._retriever_need_shadow_enabled(context) is False
    assert subscription_llm._retriever_model_driven_enabled(context) is False
    assert subscription_llm._answerability_shadow_enabled(context) is True
    assert subscription_llm._retriever_need_shadow_enabled({**context, RETRIEVER_NEED_SHADOW_ENV: "1"}) is True
    assert subscription_llm._retriever_model_driven_enabled({**context, RETRIEVER_MODEL_DRIVEN_ENV: "1"}) is False
    assert subscription_llm._retriever_model_driven_enabled(
        {**context, ASSUMED_SCOPE_GUARD_ENV: "1", RETRIEVER_MODEL_DRIVEN_ENV: "1"}
    ) is True
    assert subscription_llm._answerability_shadow_enabled(legacy_context) is False
    assert subscription_llm._deal_action_decision_enabled({**context, subscription_llm.DEAL_ACTION_DECISION_ENV: "1"}) is True
    assert subscription_llm._deal_action_decision_enabled({**context, subscription_llm.DEAL_ACTION_DECISION_ENV: "0"}) is False
    assert subscription_llm._direct_path_model_p0_enabled({**context, subscription_llm.DIRECT_PATH_MODEL_P0_ENV: "1"}) is True
    assert subscription_llm._direct_path_model_p0_enabled({**context, subscription_llm.DIRECT_PATH_MODEL_P0_ENV: "0"}) is True
    assert (
        subscription_llm._direct_path_model_p0_enabled(
            {
                **context,
                subscription_llm.DIRECT_PATH_MODEL_P0_ENV: "0",
                subscription_llm.P0_MODEL_LED_ENV: "0",
            }
        )
        is False
    )


def test_pilot_gold_v1_llm_retrieve_explicit_zero_keeps_keyword_pack(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv(LLM_RETRIEVE_ENV, raising=False)
    snapshot_path = _write_wave6_snapshot(tmp_path)
    context = {
        "active_brand": "foton",
        "snapshot_path": str(snapshot_path),
        DIRECT_PATH_PILOT_CONFIG_ENV: DIRECT_PATH_PILOT_CONFIG_VERSION,
        "conversation_intent_plan": {"primary_intent": "pricing", "answer_topics": ["pricing"]},
    }
    calls = 0

    def retriever(_: str) -> Mapping[str, object]:
        nonlocal calls
        calls += 1
        raise AssertionError("explicit TELEGRAM_LLM_RETRIEVE=0 must keep keyword selection")

    keyword = _direct_path_context_fact_pack(
        {**context, LLM_RETRIEVE_ENV: "0"},
        client_message="Сколько стоит?",
    )
    off = _direct_path_context_fact_pack(
        {**context, LLM_RETRIEVE_ENV: "0"},
        client_message="Сколько стоит?",
        retriever_fn=retriever,
    )

    assert subscription_llm._llm_retrieve_enabled(context) is True
    assert subscription_llm._llm_retrieve_enabled({**context, LLM_RETRIEVE_ENV: "0"}) is False
    assert off == keyword
    assert off["selected_category"] != "llm_retrieve"
    assert calls == 0


def test_pilot_gold_v1_explicit_override_is_visible_in_metadata(monkeypatch) -> None:
    for key in (
        DIRECT_PATH_ENV,
        BOT_GOLD_REAL_ENV,
        SEMANTIC_OUTPUT_VERIFIER_ENV,
        OUTPUT_SANITIZER_ENV,
        LLM_RETRIEVE_ENV,
        NUMBER_GATE_SCOPE_AWARE_ENV,
        VERIFIER_HANDOFF_CLAIMS_ENV,
        PRESALE_SAFETY_ENV,
        PRESALE_PII_MEMORY_ENV,
        PRESALE_VERIFIER_FAILSOFT_ENV,
        PRESALE_META_RU_ENV,
        PRESALE_SOURCE_ID_ENV,
        subscription_llm.MEMORY_PROVENANCE_ENV,
        subscription_llm.MEMORY_PROVENANCE_COMPACT_ENV,
        subscription_llm.PII_RELATION_STOPWORDS_ENV,
        subscription_llm.MEMORY_CHILD_ELLIPSIS_ENV,
        TEMPLATE_FROM_KB_ENV,
        DIRECT_PATH_PILOT_CONFIG_ENV,
    ):
        monkeypatch.delenv(key, raising=False)

    provider = _DirectPathProvider(
        SubscriptionDraftResult(route="bot_answer_self_for_pilot", draft_text="Да, подскажу по рассрочке.")
    )
    result = provider.build_draft(
        "Рассрочка есть?",
        context={
            "active_brand": "foton",
            DIRECT_PATH_PILOT_CONFIG_ENV: DIRECT_PATH_PILOT_CONFIG_VERSION,
            SEMANTIC_OUTPUT_VERIFIER_ENV: "0",
            LLM_RETRIEVE_ENV: "0",
            TEMPLATE_FROM_KB_ENV: "0",
            "confirmed_facts": {"installment.foton": "Фотон: доступны варианты на 6, 10 или 12 месяцев."},
        },
    )

    assert _semantic_output_verifier_enabled(
        {DIRECT_PATH_PILOT_CONFIG_ENV: DIRECT_PATH_PILOT_CONFIG_VERSION, SEMANTIC_OUTPUT_VERIFIER_ENV: "0"}
    ) is False
    assert result.metadata["direct_path"]["pilot_profile_overrides"] == {
        SEMANTIC_OUTPUT_VERIFIER_ENV: "0",
        LLM_RETRIEVE_ENV: "0",
        TEMPLATE_FROM_KB_ENV: "0",
    }
    assert subscription_llm._template_from_kb_enabled(
        {DIRECT_PATH_PILOT_CONFIG_ENV: DIRECT_PATH_PILOT_CONFIG_VERSION, TEMPLATE_FROM_KB_ENV: "0"}
    ) is False


def test_without_pilot_config_profile_flags_keep_default_off(monkeypatch) -> None:
    for key in (
        DIRECT_PATH_ENV,
        BOT_GOLD_REAL_ENV,
        SEMANTIC_OUTPUT_VERIFIER_ENV,
        OUTPUT_SANITIZER_ENV,
        LLM_RETRIEVE_ENV,
        NUMBER_GATE_SCOPE_AWARE_ENV,
        VERIFIER_HANDOFF_CLAIMS_ENV,
        PRESALE_SAFETY_ENV,
        PRESALE_PII_MEMORY_ENV,
        PRESALE_VERIFIER_FAILSOFT_ENV,
        PRESALE_META_RU_ENV,
        PRESALE_SOURCE_ID_ENV,
        subscription_llm.MEMORY_PROVENANCE_ENV,
        subscription_llm.MEMORY_PROVENANCE_COMPACT_ENV,
        subscription_llm.PII_RELATION_STOPWORDS_ENV,
        subscription_llm.MEMORY_CHILD_ELLIPSIS_ENV,
        TEMPLATE_FROM_KB_ENV,
        DIRECT_PATH_PILOT_CONFIG_ENV,
    ):
        monkeypatch.delenv(key, raising=False)

    context: dict[str, object] = {}

    assert _direct_path_gold_real_enabled(context) is False
    assert _semantic_output_verifier_enabled(context) is False
    assert _output_sanitizer_enabled(context) is False
    assert number_gate_scope_aware_enabled(context) is False
    assert _verifier_handoff_claims_enabled(context) is False
    assert _presale_safety_enabled(context, subflag=PRESALE_PII_MEMORY_ENV) is False
    assert subscription_llm._template_from_kb_enabled(context) is False


def test_direct_path_legacy_context_filters_unsafe_upstream_facts() -> None:
    pack = _direct_path_context_fact_pack(
        {
            "active_brand": "foton",
            "confirmed_facts": {
                "valid.fact": {
                    "brand": "foton",
                    "allowed_for_client_answer": True,
                    "forbidden_for_client": False,
                    "internal_only": False,
                    "valid_until": "2027-08-31",
                    "client_safe_text": "Фотон: безопасный факт для клиента.",
                },
                "wrong.brand": {
                    "brand": "unpk",
                    "allowed_for_client_answer": True,
                    "client_safe_text": "УНПК: чужой бренд.",
                },
                "not.client.safe": {
                    "brand": "foton",
                    "allowed_for_client_answer": False,
                    "client_safe_text": "Фотон: не клиентский факт.",
                },
                "expired.fact": {
                    "brand": "foton",
                    "allowed_for_client_answer": True,
                    "valid_until": "2020-01-01",
                    "client_safe_text": "Фотон: устаревший факт.",
                },
            },
        },
        client_message="Расскажите условия",
    )

    facts = pack["facts"]
    assert "valid.fact" in facts
    assert "wrong.brand" not in facts
    assert "not.client.safe" not in facts
    assert "expired.fact" not in facts
    assert facts["valid.fact"] == "Фотон: безопасный факт для клиента."


def test_direct_path_gate_downgrades_manager_deadline_promise_but_keeps_text() -> None:
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="Менеджер свяжется завтра утром и поможет оформить запись.",
        metadata={"direct_path": {"enabled": True, "direct_path_attempted": True}},
    )

    gated = apply_authoritative_output_gate(result)
    gate = gated.metadata["authoritative_output_gate"]

    assert gated.route == "draft_for_manager"
    assert gated.draft_text == result.draft_text
    assert gate["action"] == "downgrade_keep_text"
    assert "unsupported_manager_deadline_promise" in {item["code"] for item in gate["findings"]}
    assert "direct_path_gate_text_preserved" in gated.safety_flags


def test_direct_path_gate_allows_manager_contact_without_deadline() -> None:
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="Менеджер свяжется и поможет оформить запись.",
        metadata={"direct_path": {"enabled": True, "direct_path_attempted": True}},
    )

    gated = apply_authoritative_output_gate(result)

    assert gated.route == "bot_answer_self_for_pilot"
    assert gated.draft_text == result.draft_text
    assert gated.metadata["authoritative_output_gate"]["action"] == "pass"


def test_direct_path_gate_downgrades_future_booking_commitment_but_keeps_text() -> None:
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="Запишем вас в группу и закрепим место.",
        metadata={"direct_path": {"enabled": True, "direct_path_attempted": True}},
    )

    gated = apply_authoritative_output_gate(result, client_message="Что дальше?", context={"active_brand": "foton"})

    gate = gated.metadata["authoritative_output_gate"]
    assert gated.route == "draft_for_manager"
    assert gated.draft_text == result.draft_text
    assert gate["action"] == "downgrade_keep_text"
    assert "unsafe_future_commitment" in {item["code"] for item in gate["findings"]}
    assert "direct_path_gate_text_preserved" in gated.safety_flags


@pytest.mark.parametrize(
    "draft_text",
    (
        "Я запишу вас в группу и закреплю место.",
        "Забронирую вам место в группе.",
        "Я верну вам деньги за оплату.",
        "Оформлю возврат оплаты.",
        "Выставлю счет.",
    ),
)
def test_direct_path_gate_downgrades_first_person_future_commitments(draft_text: str) -> None:
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text=draft_text,
        metadata={"direct_path": {"enabled": True, "direct_path_attempted": True}},
    )

    gated = apply_authoritative_output_gate(result, client_message="Что дальше?", context={"active_brand": "foton"})

    gate = gated.metadata["authoritative_output_gate"]
    assert gated.route in {"draft_for_manager", "manager_only"}
    assert gate["action"] in {"downgrade_keep_text", "block"}
    assert {"unsafe_future_commitment", "p0_promise"} & {item["code"] for item in gate["findings"]}


def test_direct_path_gate_keeps_safe_manager_check_before_booking() -> None:
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="Сначала менеджер проверит наличие. Если место есть, он подскажет оформление заявки.",
        metadata={"direct_path": {"enabled": True, "direct_path_attempted": True}},
    )

    gated = apply_authoritative_output_gate(result, client_message="Есть места?", context={"active_brand": "foton"})

    assert gated.route == "bot_answer_self_for_pilot"
    assert gated.draft_text == result.draft_text
    assert gated.metadata["authoritative_output_gate"]["action"] == "pass"


def test_brand_guard_blocks_any_brand_token_when_active_brand_unknown() -> None:
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="Фотон поможет подобрать группу по математике.",
    )

    guarded = apply_brand_separation_guard(result, client_message="Что выбрать?", context={"active_brand": "unknown"})

    assert guarded.route == "manager_only"
    assert "brand_unknown_client_text_blocked" in guarded.safety_flags
    assert guarded.metadata["forbidden_brand_terms"] == ["foton"]


def test_brand_guard_blocks_two_brands_even_when_active_brand_known() -> None:
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="В Фотоне и УНПК МФТИ условия похожи, можно выбрать любой вариант.",
    )

    guarded = apply_brand_separation_guard(result, client_message="Что выбрать?", context={"active_brand": "foton"})

    assert guarded.route == "manager_only"
    assert "cross_brand_client_text_blocked" in guarded.safety_flags
    assert set(guarded.metadata["forbidden_brand_terms"]) == {"foton", "unpk"}


def test_direct_path_real_manager_gold_p0_preblock_still_skips_model() -> None:
    provider = _DirectPathProvider(
        SubscriptionDraftResult(route="bot_answer_self_for_pilot", draft_text="Этого текста быть не должно.")
    )
    result = provider.build_draft(
        "Списали дважды, верните деньги",
        context={"active_brand": "foton", DIRECT_PATH_ENV: "1", BOT_GOLD_REAL_ENV: "1"},
    )

    assert provider.calls == 0
    assert provider.last_prompt == ""
    assert result.route == "manager_only"
    assert result.metadata["direct_path"]["preblocked"] is True
    assert result.metadata["direct_path"]["gold_real_enabled"] is False


def test_prose_model_led_default_off_and_enabled_in_pilot_profile() -> None:
    assert PROSE_MODEL_LED_ENV in subscription_llm.DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS
    assert subscription_llm._prose_model_led_enabled({}) is False
    assert subscription_llm._prose_model_led_enabled({PROSE_MODEL_LED_ENV: "1"}) is True
    assert subscription_llm._prose_model_led_enabled({PROSE_MODEL_LED_ENV: "0"}) is False


def test_prose_model_led_prompt_block_is_flagged_only() -> None:
    base_context = {"active_brand": "foton", DIRECT_PATH_ENV: "1"}

    off_prompt = _build_direct_path_prompt("Есть места на 8 класс?", context=base_context, facts={})
    on_prompt = _build_direct_path_prompt(
        "Есть места на 8 класс?",
        context={**base_context, PROSE_MODEL_LED_ENV: "1"},
        facts={},
    )

    assert "Качество текста:" not in off_prompt
    assert "Качество текста:" in on_prompt
    assert "Не начинай с казённых фраз" in on_prompt
    assert "не обещай место" in on_prompt


def test_prose_model_led_verified_fact_fallback_removes_robotic_opening() -> None:
    result = SubscriptionDraftResult(
        route="draft_for_manager",
        topic_id="theme:001_pricing",
        draft_text="Передам менеджеру.",
    )
    context = {
        "confirmed_facts": {
            "price": "Фотон: очный курс для 8 класса стоит 44 600 ₽ за семестр.",
        }
    }

    off = subscription_llm._promoted_verified_fact_text(result, context=context, client_message="Сколько стоит?")
    on = subscription_llm._promoted_verified_fact_text(
        result,
        context={**context, PROSE_MODEL_LED_ENV: "1"},
        client_message="Сколько стоит?",
    )

    assert "По проверенным условиям" in off
    assert not off.startswith("Да,")
    assert "сориентирую по проверенной" not in on.casefold()
    assert "44 600 ₽" in on


def test_prose_model_led_off_is_noop_for_placeholder_and_repeat() -> None:
    text = "По местам не буду обещать без проверки. Передам менеджеру, чтобы он проверил наличие. [данные у менеджера]"
    result = SubscriptionDraftResult(route="draft_for_manager", draft_text=text)
    context = {
        "recent_messages": [f"bot: {text}"],
        "known_slots": {"grade": "8", "subject": "математика"},
    }

    guarded = apply_prose_model_led_quality_guard(result, client_message="Есть места?", context=context)

    assert guarded is result
    assert guarded.draft_text == text


def test_prose_model_led_removes_internal_placeholders_from_client_text() -> None:
    result = SubscriptionDraftResult(
        route="draft_for_manager",
        draft_text="Записала [данные у менеджера]. Передам менеджеру. [...]",
    )

    guarded = apply_prose_model_led_quality_guard(result, context={PROSE_MODEL_LED_ENV: "1"})

    assert "[данные у менеджера]" not in guarded.draft_text
    assert "[...]" not in guarded.draft_text
    assert "prose_model_led:internal_client_placeholder" in guarded.safety_flags
    assert guarded.metadata["prose_model_led"]["placeholder_removed"] is True


def test_prose_model_led_removes_meta_fact_phrase_before_client_output() -> None:
    result = SubscriptionDraftResult(
        route="draft_for_manager",
        draft_text="Точной ссылки на тест у меня сейчас в фактах нет, поэтому её нужно проверить. Менеджер подскажет форму.",
    )

    guarded = apply_prose_model_led_quality_guard(result, context={PROSE_MODEL_LED_ENV: "1"})

    assert "в фактах нет" not in guarded.draft_text.casefold()
    assert "Эту деталь нужно проверить у менеджера" in guarded.draft_text
    assert "prose_model_led:meta_fact_phrase" in guarded.safety_flags


def test_prose_model_led_rewrites_unsupported_send_action() -> None:
    result = SubscriptionDraftResult(
        route="draft_for_manager",
        draft_text="По онлайн-формату можно посмотреть фрагмент занятия. Прикрепляю фрагмент и инструкцию по записи.",
    )

    guarded = apply_prose_model_led_quality_guard(result, context={PROSE_MODEL_LED_ENV: "1"})

    assert "Прикрепляю" not in guarded.draft_text
    assert "должен отправить менеджер" in guarded.draft_text
    assert "prose_model_led:unsupported_send_action" in guarded.safety_flags


def test_prose_model_led_rephrases_repeated_availability_handoff() -> None:
    previous = "По местам не буду обещать без проверки по вашему запросу (8 класс, математика). Передам менеджеру, чтобы он проверил наличие по конкретной группе."
    result = SubscriptionDraftResult(route="draft_for_manager", draft_text=previous)
    context = {
        PROSE_MODEL_LED_ENV: "1",
        "known_slots": {"grade": "8", "subject": "математика"},
        "recent_messages": [f"bot: {previous}"],
    }

    guarded = apply_prose_model_led_quality_guard(result, client_message="А места есть?", context=context)

    assert guarded.draft_text != previous
    assert "По местам не буду обещать без проверки" not in guarded.draft_text
    assert "Заранее место не подтверждаю" in guarded.draft_text
    assert "prose_model_led:near_repeat_rephrased" in guarded.safety_flags


def test_prose_model_led_does_not_rewrite_p0_safe_templates() -> None:
    previous = COMPLAINT_SAFE_TEXT
    result = SubscriptionDraftResult(
        route="manager_only",
        topic_id="theme:009_refund",
        draft_text=COMPLAINT_SAFE_TEXT,
        safety_flags=("complaint_apology_guarded", "manager_approval_required", "no_auto_send"),
    )
    context = {PROSE_MODEL_LED_ENV: "1", "recent_messages": [f"bot: {previous}"]}

    guarded = apply_prose_model_led_quality_guard(result, client_message="Жалоба", context=context)

    assert guarded.draft_text == COMPLAINT_SAFE_TEXT
    assert not any(str(flag).startswith("prose_model_led:near_repeat") for flag in guarded.safety_flags)
