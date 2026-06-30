from __future__ import annotations

import pytest

import mango_mvp.channels.subscription_llm as subscription_llm
from mango_mvp.channels.subscription_llm import (
    DIRECT_PATH_ENV,
    DIRECT_PATH_PILOT_CONFIG_ENV,
    DIRECT_PATH_PILOT_CONFIG_VERSION,
    SEMANTIC_FRAME_SHADOW_ENV,
    SubscriptionDraftResult,
    SubscriptionLlmDraftProvider,
    _build_direct_path_prompt,
    _normalize_direct_path_payload,
)


def test_semantic_frame_shadow_flag_is_default_off_and_not_profile_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(SEMANTIC_FRAME_SHADOW_ENV, raising=False)
    profile_context = {DIRECT_PATH_PILOT_CONFIG_ENV: DIRECT_PATH_PILOT_CONFIG_VERSION}

    assert subscription_llm._semantic_frame_shadow_enabled({}) is False
    assert subscription_llm._semantic_frame_shadow_enabled(profile_context) is False
    assert SEMANTIC_FRAME_SHADOW_ENV not in subscription_llm.DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS
    assert subscription_llm._semantic_frame_shadow_enabled({SEMANTIC_FRAME_SHADOW_ENV: "1"}) is True
    assert subscription_llm._semantic_frame_shadow_enabled({"semantic_frame_shadow": "1"}) is True
    assert subscription_llm._semantic_frame_shadow_enabled({SEMANTIC_FRAME_SHADOW_ENV: "0"}) is False


def test_semantic_frame_shadow_prompt_is_explicitly_flagged() -> None:
    context = {"active_brand": "foton", DIRECT_PATH_ENV: "1"}
    profile_context = {
        **context,
        DIRECT_PATH_PILOT_CONFIG_ENV: DIRECT_PATH_PILOT_CONFIG_VERSION,
    }

    off_prompt = _build_direct_path_prompt("Можно записаться на курс?", context=context)
    profile_prompt = _build_direct_path_prompt("Можно записаться на курс?", context=profile_context)
    on_prompt = _build_direct_path_prompt(
        "Можно записаться на курс?",
        context={**context, SEMANTIC_FRAME_SHADOW_ENV: "1"},
    )

    assert '"semantic_frame"' not in off_prompt
    assert "SemanticFrame SHADOW" not in off_prompt
    assert '"semantic_frame"' not in profile_prompt
    assert "SemanticFrame SHADOW" not in profile_prompt
    assert '"semantic_frame"' in on_prompt
    assert "SemanticFrame SHADOW" in on_prompt
    assert "не меняй из-за него route, draft_text" in on_prompt
    assert '"must_handoff": false' in on_prompt


def test_direct_path_payload_stores_semantic_frame_shadow_metadata_only() -> None:
    off_result = _normalize_direct_path_payload(
        {
            "route": "bot_answer_self_for_pilot",
            "draft_text": "Да, можно записаться на онлайн-курс.",
            "semantic_frame": {"intent": "enrollment_question"},
        }
    )
    assert "semantic_frame_shadow" not in off_result.metadata

    result = _normalize_direct_path_payload(
        {
            "route": "bot_answer_self_for_pilot",
            "draft_text": "Да, можно записаться на онлайн-курс.",
            "semantic_frame": {
                "intent": "enrollment_question",
                "risk_class": "safe",
                "deal_stage": "interest",
                "payment_readiness": "considering",
                "requested_product": {
                    "brand": "foton",
                    "subject": "physics",
                    "grade": "7",
                    "format": "online",
                    "venue": "online",
                    "program_kind": "regular",
                    "raw_text": "онлайн-курс физики 7 класс",
                },
                "requested_action": "enroll",
                "answerability": "answer_self",
                "must_handoff": False,
                "evidence": ["клиент спрашивает про запись без P0, телефон +7 900 111-22-33, id 15930"],
                "confidence": 0.91,
            },
        },
        include_semantic_frame_shadow=True,
    )

    assert result.route == "bot_answer_self_for_pilot"
    assert result.draft_text == "Да, можно записаться на онлайн-курс."
    frame = result.metadata["semantic_frame_shadow"]
    assert frame == {
        "schema_version": "semantic_frame_shadow_v1_2026_06_30",
        "intent": "enrollment_question",
        "risk_class": "safe",
        "deal_stage": "interest",
        "payment_readiness": "considering",
        "requested_product": {
            "brand": "foton",
            "subject": "physics",
            "grade": "7",
            "format": "online",
            "venue": "online",
            "program_kind": "regular",
            "raw_text": "онлайн-курс физики 7 класс",
        },
        "requested_action": "enroll",
        "answerability": "yes",
        "must_handoff": False,
        "evidence": ["клиент спрашивает про запись без P0, телефон [phone], id [id]"],
        "confidence": 0.91,
    }


class _SemanticFrameFakeProvider(SubscriptionLlmDraftProvider):
    def __init__(self, result: SubscriptionDraftResult) -> None:
        super().__init__()
        self.result = result
        self.calls = 0
        self.last_prompt = ""

    def _direct_path_draft_runner(self, prompt: str) -> SubscriptionDraftResult:
        self.calls += 1
        self.last_prompt = prompt
        return self.result


def test_semantic_frame_shadow_does_not_change_route_text_or_call_count() -> None:
    provider = _SemanticFrameFakeProvider(
        SubscriptionDraftResult(
            route="draft_for_manager",
            draft_text="Менеджер проверит наличие места и подскажет следующий шаг.",
            metadata={
                "semantic_frame_shadow": {
                    "schema_version": "semantic_frame_shadow_v1_2026_06_30",
                    "intent": "live_availability",
                    "must_handoff": True,
                }
            },
        )
    )

    result = provider.build_draft(
        "Есть места в онлайн-группе?",
        context={"active_brand": "foton", DIRECT_PATH_ENV: "1", SEMANTIC_FRAME_SHADOW_ENV: "1"},
    )

    assert provider.calls == 1
    assert '"semantic_frame"' in provider.last_prompt
    assert result.route == "draft_for_manager"
    assert result.draft_text == "Менеджер проверит наличие места и подскажет следующий шаг."
    assert result.metadata["direct_path"]["semantic_frame_shadow"]["intent"] == "live_availability"
