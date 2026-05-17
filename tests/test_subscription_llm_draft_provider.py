from __future__ import annotations

import subprocess
from pathlib import Path

from mango_mvp.channels.subscription_llm import (
    CodexExecConfig,
    CodexExecDraftProvider,
    DraftGenerationResult,
    FakeDraftProvider,
    contains_bot_identity_disclosure,
    detect_high_risk_input_markers,
    parse_llm_json,
)


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


def test_draft_text_does_not_disclose_bot_identity() -> None:
    result = parse_llm_json('{"route":"draft_for_manager","draft_text":"Как ИИ я могу подсказать."}')

    assert result.route == "manager_only"
    assert "bot_identity_disclosure" in result.safety_flags
    assert contains_bot_identity_disclosure("Я бот и нейросеть")
    for phrase in ("я бот", "как ИИ", "нейросеть", "искусственный интеллект", "GPT", "Claude", "Codex"):
        assert contains_bot_identity_disclosure(f"Тест: {phrase}")


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

    assert result.topic_id == "theme:001_pricing"
    assert result.route == "manager_only"
    assert "high_risk_input_manager_only" in result.safety_flags
    assert result.metadata["forced_route_high_risk_input"] == ["refund"]


def test_high_risk_input_marker_coverage_for_russian_forms() -> None:
    cases = {
        "refund": [
            "Возврат",
            "вернуть деньги",
            "возвращу оплату",
            "верните оплату",
            "возвратить платеж",
            "расторгнуть договор",
            "отказаться от обучения",
            "забрать деньги",
            "возрат денег",
            "ВОЗВРАТ платежа",
        ],
        "matkap": [
            "маткапитал",
            "материнский капитал",
            "материнский сертификат",
            "семейный сертификат",
        ],
        "tax": [
            "налоговый вычет",
            "вернуть 13%",
            "справка для налоговой",
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
        "payment_status": [
            "оплатил",
            "оплатила",
            "провели платёж",
            "списали",
            "списание",
            "зачислили",
            "получили деньги",
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
