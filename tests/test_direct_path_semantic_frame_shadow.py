from __future__ import annotations

import json

import pytest

import mango_mvp.channels.subscription_llm as subscription_llm
from mango_mvp.channels.subscription_llm import (
    DIRECT_PATH_ENV,
    DIRECT_PATH_PILOT_CONFIG_ENV,
    DIRECT_PATH_PILOT_CONFIG_VERSION,
    SEMANTIC_FRAME_DECISION_SHADOW_ENV,
    SEMANTIC_FRAME_MANAGER_ACTION_GATE_ENV,
    SEMANTIC_FRAME_POSTHOC_SHADOW_ENV,
    SEMANTIC_FRAME_SELF_ANSWER_SHADOW_ENV,
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


def test_semantic_frame_decision_shadow_flag_is_default_off_and_not_profile_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(SEMANTIC_FRAME_DECISION_SHADOW_ENV, raising=False)
    profile_context = {DIRECT_PATH_PILOT_CONFIG_ENV: DIRECT_PATH_PILOT_CONFIG_VERSION}

    assert subscription_llm._semantic_frame_decision_shadow_enabled({}) is False
    assert subscription_llm._semantic_frame_decision_shadow_enabled(profile_context) is False
    assert SEMANTIC_FRAME_DECISION_SHADOW_ENV not in subscription_llm.DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS
    assert subscription_llm._semantic_frame_decision_shadow_enabled({SEMANTIC_FRAME_DECISION_SHADOW_ENV: "1"}) is True
    assert subscription_llm._semantic_frame_decision_shadow_enabled({"semantic_frame_decision_shadow": "1"}) is True
    assert subscription_llm._semantic_frame_decision_shadow_enabled({SEMANTIC_FRAME_DECISION_SHADOW_ENV: "0"}) is False


def test_semantic_frame_manager_action_gate_flag_is_default_off_and_not_profile_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(SEMANTIC_FRAME_MANAGER_ACTION_GATE_ENV, raising=False)
    profile_context = {DIRECT_PATH_PILOT_CONFIG_ENV: DIRECT_PATH_PILOT_CONFIG_VERSION}

    assert subscription_llm._semantic_frame_manager_action_gate_enabled({}) is False
    assert subscription_llm._semantic_frame_manager_action_gate_enabled(profile_context) is False
    assert SEMANTIC_FRAME_MANAGER_ACTION_GATE_ENV not in subscription_llm.DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS
    assert subscription_llm._semantic_frame_manager_action_gate_enabled({SEMANTIC_FRAME_MANAGER_ACTION_GATE_ENV: "1"}) is True
    assert subscription_llm._semantic_frame_manager_action_gate_enabled({"semantic_frame_manager_action_gate": "1"}) is True
    assert subscription_llm._semantic_frame_manager_action_gate_enabled({SEMANTIC_FRAME_MANAGER_ACTION_GATE_ENV: "0"}) is False


def test_semantic_frame_self_answer_shadow_flag_is_default_off_and_not_profile_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(SEMANTIC_FRAME_SELF_ANSWER_SHADOW_ENV, raising=False)
    profile_context = {DIRECT_PATH_PILOT_CONFIG_ENV: DIRECT_PATH_PILOT_CONFIG_VERSION}

    assert subscription_llm._semantic_frame_self_answer_shadow_enabled({}) is False
    assert subscription_llm._semantic_frame_self_answer_shadow_enabled(profile_context) is False
    assert SEMANTIC_FRAME_SELF_ANSWER_SHADOW_ENV not in subscription_llm.DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS
    assert subscription_llm._semantic_frame_self_answer_shadow_enabled({SEMANTIC_FRAME_SELF_ANSWER_SHADOW_ENV: "1"}) is True
    assert subscription_llm._semantic_frame_self_answer_shadow_enabled({"semantic_frame_self_answer_shadow": "1"}) is True
    assert subscription_llm._semantic_frame_self_answer_shadow_enabled({SEMANTIC_FRAME_SELF_ANSWER_SHADOW_ENV: "0"}) is False


def test_semantic_frame_posthoc_shadow_flag_is_default_off_and_not_profile_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(SEMANTIC_FRAME_POSTHOC_SHADOW_ENV, raising=False)
    profile_context = {DIRECT_PATH_PILOT_CONFIG_ENV: DIRECT_PATH_PILOT_CONFIG_VERSION}

    assert subscription_llm._semantic_frame_posthoc_shadow_enabled({}) is False
    assert subscription_llm._semantic_frame_posthoc_shadow_enabled(profile_context) is False
    assert SEMANTIC_FRAME_POSTHOC_SHADOW_ENV not in subscription_llm.DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS
    assert subscription_llm._semantic_frame_posthoc_shadow_enabled({SEMANTIC_FRAME_POSTHOC_SHADOW_ENV: "1"}) is True
    assert subscription_llm._semantic_frame_posthoc_shadow_enabled({"semantic_frame_posthoc_shadow": "1"}) is True
    assert subscription_llm._semantic_frame_posthoc_shadow_enabled({SEMANTIC_FRAME_POSTHOC_SHADOW_ENV: "0"}) is False


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
    assert "semantic_frame" not in off_result.metadata
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
    frame = result.metadata["semantic_frame"]
    assert frame == {
        "schema_version": "semantic_frame_v1_2026_07_01",
        "legacy_schema_version": "semantic_frame_shadow_v1_2026_06_30",
        "mode": "shadow",
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
        "evidence": ["клиент спрашивает про запись без P0, телефон [phone], id [id]"],
        "confidence": 0.91,
    }
    assert result.metadata["semantic_frame_shadow"] == frame


class _SemanticFrameFakeProvider(SubscriptionLlmDraftProvider):
    def __init__(
        self,
        result: SubscriptionDraftResult,
        *,
        posthoc_payload: dict | None = None,
        posthoc_error: Exception | None = None,
    ) -> None:
        super().__init__()
        self.result = result
        self.posthoc_payload = posthoc_payload
        self.posthoc_error = posthoc_error
        self.calls = 0
        self.posthoc_calls = 0
        self.last_prompt = ""
        self.last_posthoc_prompt = ""

    def _direct_path_draft_runner(self, prompt: str) -> SubscriptionDraftResult:
        self.calls += 1
        self.last_prompt = prompt
        return self.result

    def _direct_path_semantic_frame_shadow_runner(self, prompt: str) -> str:
        self.posthoc_calls += 1
        self.last_posthoc_prompt = prompt
        if self.posthoc_error is not None:
            raise self.posthoc_error
        return json.dumps(self.posthoc_payload or {}, ensure_ascii=False)


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
    assert result.metadata["direct_path"]["semantic_frame"]["intent"] == "live_availability"
    assert result.metadata["direct_path"]["semantic_frame"]["mode"] == "shadow"
    assert result.metadata["direct_path"]["semantic_frame_shadow"]["intent"] == "live_availability"
    assert "frame_decision_shadow" not in result.metadata


def test_semantic_frame_posthoc_shadow_adds_metadata_only_after_final_result() -> None:
    base_result = SubscriptionDraftResult(
        route="draft_for_manager",
        draft_text="Менеджер проверит наличие места и подскажет следующий шаг.",
        safety_flags=("manager_approval_required", "no_auto_send"),
        manager_checklist=("Проверить наличие места.",),
    )
    off_provider = _SemanticFrameFakeProvider(base_result)
    off_result = off_provider.build_draft(
        "Есть места в онлайн-группе?",
        context={"active_brand": "foton", DIRECT_PATH_ENV: "1"},
    )
    provider = _SemanticFrameFakeProvider(
        base_result,
        posthoc_payload={
            "semantic_frame": {
                "intent": "live_availability",
                "risk_class": "manager_action",
                "deal_stage": "closing",
                "payment_readiness": "considering",
                "requested_product": {"brand": "foton", "raw_text": "онлайн-группа"},
                "requested_action": "check_availability",
                "answerability": "manager_only",
                "must_handoff": True,
                "evidence": ["нужно проверить место, телефон +7 900 111-22-33"],
                "confidence": 0.9,
            }
        },
    )

    result = provider.build_draft(
        "Есть места в онлайн-группе?",
        context={"active_brand": "foton", DIRECT_PATH_ENV: "1", SEMANTIC_FRAME_POSTHOC_SHADOW_ENV: "1"},
    )

    assert off_provider.calls == 1
    assert off_provider.posthoc_calls == 0
    assert provider.calls == 1
    assert provider.posthoc_calls == 1
    assert '"semantic_frame"' not in provider.last_prompt
    assert "SemanticFrame SHADOW" not in provider.last_prompt
    assert result.route == off_result.route
    assert result.draft_text == off_result.draft_text
    assert result.safety_flags == off_result.safety_flags
    assert result.manager_checklist == off_result.manager_checklist
    frame = result.metadata["semantic_frame"]
    assert frame["intent"] == "live_availability"
    assert frame["evidence"] == ["нужно проверить место, телефон [phone]"]
    assert result.metadata["semantic_frame_shadow"] == frame
    assert result.metadata["direct_path"]["semantic_frame"] == frame
    assert result.metadata["semantic_frame_posthoc_shadow"]["status"] == "ok"


def test_semantic_frame_posthoc_shadow_error_is_fail_soft() -> None:
    base_result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="Да, можно записаться на онлайн-курс.",
        safety_flags=("no_auto_send",),
    )
    off_provider = _SemanticFrameFakeProvider(base_result)
    off_result = off_provider.build_draft(
        "Можно записаться на курс?",
        context={"active_brand": "foton", DIRECT_PATH_ENV: "1"},
    )
    provider = _SemanticFrameFakeProvider(
        base_result,
        posthoc_error=RuntimeError("temporary frame failure"),
    )

    result = provider.build_draft(
        "Можно записаться на курс?",
        context={"active_brand": "foton", DIRECT_PATH_ENV: "1", SEMANTIC_FRAME_POSTHOC_SHADOW_ENV: "1"},
    )

    assert off_provider.calls == 1
    assert off_provider.posthoc_calls == 0
    assert provider.calls == 1
    assert provider.posthoc_calls == 1
    assert result.route == off_result.route
    assert result.draft_text == off_result.draft_text
    assert result.safety_flags == off_result.safety_flags
    assert result.manager_checklist == off_result.manager_checklist
    assert "semantic_frame" not in result.metadata
    assert result.metadata["semantic_frame_posthoc_shadow"]["status"] == "provider_error"


def test_semantic_frame_decision_shadow_adds_metadata_only() -> None:
    base_result = SubscriptionDraftResult(
        route="draft_for_manager",
        draft_text="Менеджер проверит наличие места и подскажет следующий шаг.",
        safety_flags=("manager_approval_required",),
        manager_checklist=("Проверить наличие места.",),
        metadata={
            "direct_path_model_p0": {"is_p0": False},
            "semantic_frame": {
                "schema_version": "semantic_frame_v1_2026_07_01",
                "mode": "shadow",
                "intent": "live_availability",
                "risk_class": "manager_action",
                "deal_stage": "closing",
                "payment_readiness": "considering",
                "requested_action": "check_availability",
                "answerability": "manager_only",
                "must_handoff": True,
                "confidence": 0.88,
                "evidence": ["нужно проверить место"],
            },
        },
    )
    off_provider = _SemanticFrameFakeProvider(base_result)
    off_result = off_provider.build_draft(
        "Есть места в онлайн-группе?",
        context={
            "active_brand": "foton",
            DIRECT_PATH_ENV: "1",
            SEMANTIC_FRAME_SHADOW_ENV: "1",
        },
    )
    provider = _SemanticFrameFakeProvider(base_result)

    result = provider.build_draft(
        "Есть места в онлайн-группе?",
        context={
            "active_brand": "foton",
            DIRECT_PATH_ENV: "1",
            SEMANTIC_FRAME_SHADOW_ENV: "1",
            SEMANTIC_FRAME_DECISION_SHADOW_ENV: "1",
        },
    )

    assert provider.calls == 1
    assert off_provider.calls == 1
    assert result.route == off_result.route == "draft_for_manager"
    assert result.draft_text == off_result.draft_text == "Менеджер проверит наличие места и подскажет следующий шаг."
    assert result.safety_flags == off_result.safety_flags
    assert result.manager_checklist == off_result.manager_checklist
    assert "frame_decision_shadow" not in off_result.metadata
    shadow = result.metadata["frame_decision_shadow"]
    assert shadow["schema_version"] == "semantic_frame_decision_shadow_v1_2026_07_01"
    assert shadow["status"] == "observed"
    assert shadow["frame"]["intent"] == "live_availability"
    assert shadow["actual"]["route_after"] == "draft_for_manager"
    assert shadow["comparisons"]["must_handoff_vs_route"] == "match"
    assert shadow["comparisons"]["p0_vs_actual"] == "match"
    assert result.metadata["direct_path"]["frame_decision_shadow"] == shadow


def test_semantic_frame_decision_shadow_reports_missing_frame_without_behavior_change() -> None:
    provider = _SemanticFrameFakeProvider(
        SubscriptionDraftResult(
            route="bot_answer_self_for_pilot",
            draft_text="Да, можно записаться на онлайн-курс.",
        )
    )

    result = provider.build_draft(
        "Можно записаться на курс?",
        context={
            "active_brand": "foton",
            DIRECT_PATH_ENV: "1",
            SEMANTIC_FRAME_DECISION_SHADOW_ENV: "1",
        },
    )

    assert provider.calls == 1
    assert result.route == "bot_answer_self_for_pilot"
    assert result.draft_text == "Да, можно записаться на онлайн-курс."
    assert result.metadata["frame_decision_shadow"]["status"] == "no_frame"


def _safe_self_answer_frame(*, confidence: float = 0.94, valid_until: str = "2026-12-31") -> dict:
    return {
        "semantic_frame_posthoc_shadow": {"status": "ok"},
        "semantic_frame": {
            "schema_version": "semantic_frame_v1_2026_07_01",
            "mode": "shadow",
            "intent": "price_question",
            "risk_class": "safe",
            "deal_stage": "interest",
            "payment_readiness": "asking_price",
            "requested_product": {"brand": "foton", "raw_text": "онлайн-курс"},
            "requested_action": "answer_question",
            "answerability": "answer_self",
            "must_handoff": False,
            "confidence": confidence,
        },
        "direct_path": {
            "selected_category": "pricing",
            "wide_fact_exact_keys": ["foton.online.price"],
            "wide_fact_metadata": {
                "foton.online.price": {
                    "brand": "foton",
                    "client_safe": "true",
                    "valid_until": valid_until,
                }
            },
        },
    }


def test_semantic_frame_self_answer_shadow_metadata_only_would_demote() -> None:
    base_result = SubscriptionDraftResult(
        route="draft_for_manager",
        draft_text="Онлайн-курс стоит 24 000 рублей за семестр.",
        safety_flags=("manager_approval_required", "no_auto_send"),
        manager_checklist=("Проверить перед отправкой.",),
        metadata=_safe_self_answer_frame(),
    )

    result = subscription_llm.apply_semantic_frame_self_answer_shadow(
        base_result,
        context={
            "active_brand": "foton",
            SEMANTIC_FRAME_SELF_ANSWER_SHADOW_ENV: "1",
        },
    )

    assert result.route == "draft_for_manager"
    assert result.draft_text == "Онлайн-курс стоит 24 000 рублей за семестр."
    assert result.safety_flags == base_result.safety_flags
    assert result.manager_checklist == base_result.manager_checklist
    shadow = result.metadata["semantic_frame_self_answer_shadow"]
    assert shadow["status"] == "would_demote_to_self"
    assert shadow["route_before"] == "draft_for_manager"
    assert shadow["route_after_if_active"] == "bot_answer_self_for_pilot"
    assert shadow["guards"]["freshness"]["ok"] is True
    assert result.metadata["direct_path"]["semantic_frame_self_answer_shadow"] == shadow


def test_semantic_frame_self_answer_shadow_blocks_p0_even_when_frame_says_safe() -> None:
    metadata = _safe_self_answer_frame()
    metadata["direct_path_model_p0"] = {"is_p0": True, "p0_kind": "refund_claim"}
    base_result = SubscriptionDraftResult(
        route="draft_for_manager",
        draft_text="Передам вопрос менеджеру.",
        risk_level="high",
        safety_flags=("refund", "manager_approval_required", "no_auto_send"),
        metadata=metadata,
    )

    result = subscription_llm.apply_semantic_frame_self_answer_shadow(
        base_result,
        context={
            "active_brand": "foton",
            SEMANTIC_FRAME_SELF_ANSWER_SHADOW_ENV: "1",
        },
    )

    shadow = result.metadata["semantic_frame_self_answer_shadow"]
    assert result.route == "draft_for_manager"
    assert shadow["status"] == "blocked"
    assert shadow["reason"] == "protected_p0"
    assert shadow["route_after_if_active"] == "draft_for_manager"


def test_semantic_frame_self_answer_shadow_requires_high_confidence_and_fresh_fact() -> None:
    low_conf = SubscriptionDraftResult(
        route="draft_for_manager",
        draft_text="Онлайн-курс стоит 24 000 рублей за семестр.",
        metadata=_safe_self_answer_frame(confidence=0.89),
    )
    stale = SubscriptionDraftResult(
        route="draft_for_manager",
        draft_text="Онлайн-курс стоит 24 000 рублей за семестр.",
        metadata=_safe_self_answer_frame(valid_until="2026-01-01"),
    )

    context = {
        "active_brand": "foton",
        SEMANTIC_FRAME_SELF_ANSWER_SHADOW_ENV: "1",
    }
    low_result = subscription_llm.apply_semantic_frame_self_answer_shadow(low_conf, context=context)
    stale_result = subscription_llm.apply_semantic_frame_self_answer_shadow(stale, context=context)

    assert low_result.metadata["semantic_frame_self_answer_shadow"]["reason"] == "low_confidence"
    assert stale_result.metadata["semantic_frame_self_answer_shadow"]["reason"] == "no_fresh_client_safe_exact_fact"
    assert low_result.route == stale_result.route == "draft_for_manager"


def test_semantic_frame_self_answer_shadow_blocks_sanitizer_and_deferral_flags() -> None:
    sanitized = _safe_self_answer_frame()
    sanitized["direct_path"]["deferral_text_in_self"] = True
    base_result = SubscriptionDraftResult(
        route="draft_for_manager",
        draft_text="Если нужно уточнить по вашей группе, менеджер проверит.",
        safety_flags=("output_sanitizer:client_name_echo", "prose_model_led:internal_client_placeholder"),
        metadata=sanitized,
    )

    result = subscription_llm.apply_semantic_frame_self_answer_shadow(
        base_result,
        context={
            "active_brand": "foton",
            SEMANTIC_FRAME_SELF_ANSWER_SHADOW_ENV: "1",
        },
    )

    shadow = result.metadata["semantic_frame_self_answer_shadow"]
    assert shadow["status"] == "blocked"
    assert shadow["reason"] == "blocking_safety_flags"
    assert "output_sanitizer:client_name_echo" in shadow["guards"]["blocking_flags"]
    assert result.route == "draft_for_manager"


def test_semantic_frame_self_answer_shadow_requires_answer_question_action() -> None:
    metadata = _safe_self_answer_frame()
    metadata["semantic_frame"]["requested_action"] = "check_availability"
    base_result = SubscriptionDraftResult(
        route="draft_for_manager",
        draft_text="Менеджер проверит наличие места.",
        metadata=metadata,
    )

    result = subscription_llm.apply_semantic_frame_self_answer_shadow(
        base_result,
        context={
            "active_brand": "foton",
            SEMANTIC_FRAME_SELF_ANSWER_SHADOW_ENV: "1",
        },
    )

    shadow = result.metadata["semantic_frame_self_answer_shadow"]
    assert shadow["status"] == "blocked"
    assert shadow["reason"] == "requested_action_not_answer_question"
    assert result.route == "draft_for_manager"


def test_semantic_frame_self_answer_shadow_ignores_inline_non_posthoc_frame() -> None:
    metadata = _safe_self_answer_frame()
    metadata.pop("semantic_frame_posthoc_shadow")
    base_result = SubscriptionDraftResult(
        route="draft_for_manager",
        draft_text="Онлайн-курс стоит 24 000 рублей за семестр.",
        metadata=metadata,
    )

    result = subscription_llm.apply_semantic_frame_self_answer_shadow(
        base_result,
        context={
            "active_brand": "foton",
            SEMANTIC_FRAME_SELF_ANSWER_SHADOW_ENV: "1",
        },
    )

    shadow = result.metadata["semantic_frame_self_answer_shadow"]
    assert shadow["status"] == "blocked"
    assert shadow["reason"] == "frame_not_posthoc"
    assert result.route == "draft_for_manager"


def test_semantic_frame_manager_action_gate_is_off_noop() -> None:
    base_result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="Да, можно записаться на онлайн-курс.",
        metadata={
            "semantic_frame_posthoc_shadow": {"status": "ok"},
            "semantic_frame": {
                "intent": "live_availability",
                "risk_class": "manager_action",
                "deal_stage": "closing",
                "payment_readiness": "considering",
                "requested_action": "check_availability",
                "answerability": "manager_only",
                "must_handoff": True,
                "confidence": 0.9,
            },
        },
    )
    provider = _SemanticFrameFakeProvider(base_result)

    result = provider.build_draft(
        "Есть места в онлайн-группе?",
        context={"active_brand": "foton", DIRECT_PATH_ENV: "1"},
    )

    assert provider.calls == 1
    assert result.route == "bot_answer_self_for_pilot"
    assert result.draft_text == "Да, можно записаться на онлайн-курс."
    assert "semantic_frame_manager_action_gate" not in result.metadata


def test_semantic_frame_manager_action_gate_promotes_strong_manager_action() -> None:
    base_result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="Да, можно записаться на онлайн-курс.",
        metadata={
            "semantic_frame_posthoc_shadow": {"status": "ok"},
            "semantic_frame": {
                "intent": "live_availability",
                "risk_class": "manager_action",
                "deal_stage": "closing",
                "payment_readiness": "considering",
                "requested_action": "check_availability",
                "answerability": "manager_only",
                "must_handoff": True,
                "confidence": 0.9,
                "evidence": ["нужно проверить место"],
            },
        },
    )
    provider = _SemanticFrameFakeProvider(base_result)

    result = provider.build_draft(
        "Есть места в онлайн-группе?",
        context={
            "active_brand": "foton",
            DIRECT_PATH_ENV: "1",
            SEMANTIC_FRAME_MANAGER_ACTION_GATE_ENV: "1",
            SEMANTIC_FRAME_DECISION_SHADOW_ENV: "1",
        },
    )

    assert provider.calls == 1
    assert result.route == "draft_for_manager"
    assert result.draft_text == "Да, можно записаться на онлайн-курс."
    assert "semantic_frame_manager_action_gate" in result.safety_flags
    assert "SemanticFrame: проверить действие менеджера перед ответом клиенту." in result.manager_checklist
    gate = result.metadata["semantic_frame_manager_action_gate"]
    assert gate["status"] == "promoted_to_draft_for_manager"
    assert gate["reason"] == "manager_action:check_availability"
    assert gate["route_before"] == "bot_answer_self_for_pilot"
    assert gate["route_after"] == "draft_for_manager"
    assert result.metadata["direct_path"]["semantic_frame_manager_action_gate"] == gate
    shadow = result.metadata["frame_decision_shadow"]
    assert shadow["actual"]["route_after"] == "draft_for_manager"
    assert shadow["comparisons"]["must_handoff_vs_route"] == "match"


def test_semantic_frame_manager_action_gate_keeps_safe_or_forward_payment_self_answer() -> None:
    safe_result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="Оплатить можно по ссылке, которую пришлёт менеджер после оформления.",
        metadata={
            "semantic_frame_posthoc_shadow": {"status": "ok"},
            "semantic_frame": {
                "intent": "payment_link_request",
                "risk_class": "manager_action",
                "deal_stage": "closing",
                "payment_readiness": "ready_to_pay",
                "requested_action": "send_payment_link",
                "answerability": "manager_only",
                "must_handoff": True,
                "confidence": 0.92,
            },
        },
    )
    provider = _SemanticFrameFakeProvider(safe_result)

    result = provider.build_draft(
        "А сюда можно ссылку на оплату?",
        context={
            "active_brand": "foton",
            DIRECT_PATH_ENV: "1",
            SEMANTIC_FRAME_MANAGER_ACTION_GATE_ENV: "1",
        },
    )

    assert provider.calls == 1
    assert result.route == "bot_answer_self_for_pilot"
    assert result.draft_text == "Оплатить можно по ссылке, которую пришлёт менеджер после оформления."
    gate = result.metadata["semantic_frame_manager_action_gate"]
    assert gate["status"] == "pass"
    assert gate["reason"] == "unsupported_manager_action"


def test_semantic_frame_manager_action_gate_does_not_promote_interest_availability() -> None:
    base_result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="Да, очные курсы есть, расскажу по формату и направлениям.",
        metadata={
            "semantic_frame_posthoc_shadow": {"status": "ok"},
            "semantic_frame": {
                "intent": "course_existence_question",
                "risk_class": "manager_action",
                "deal_stage": "interest",
                "payment_readiness": "none",
                "requested_action": "check_availability",
                "answerability": "manager_only",
                "must_handoff": True,
                "confidence": 0.92,
            },
        },
    )
    provider = _SemanticFrameFakeProvider(base_result)

    result = provider.build_draft(
        "Есть очные курсы?",
        context={
            "active_brand": "foton",
            DIRECT_PATH_ENV: "1",
            SEMANTIC_FRAME_MANAGER_ACTION_GATE_ENV: "1",
        },
    )

    assert provider.calls == 1
    assert result.route == "bot_answer_self_for_pilot"
    gate = result.metadata["semantic_frame_manager_action_gate"]
    assert gate["status"] == "pass"
    assert gate["reason"] == "unsupported_manager_action"


def test_semantic_frame_manager_action_gate_does_not_trust_string_false_must_handoff() -> None:
    base_result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="Да, можно начать с пробного занятия.",
        metadata={
            "semantic_frame_posthoc_shadow": {"status": "ok"},
            "semantic_frame": {
                "intent": "trial_lesson_question",
                "risk_class": "safe",
                "deal_stage": "interest",
                "payment_readiness": "none",
                "requested_action": "enroll",
                "answerability": "answer_self",
                "must_handoff": "false",
                "confidence": 0.93,
            },
        },
    )
    provider = _SemanticFrameFakeProvider(base_result)

    result = provider.build_draft(
        "Можно пробное?",
        context={
            "active_brand": "foton",
            DIRECT_PATH_ENV: "1",
            SEMANTIC_FRAME_MANAGER_ACTION_GATE_ENV: "1",
            SEMANTIC_FRAME_DECISION_SHADOW_ENV: "1",
        },
    )

    assert provider.calls == 1
    assert result.route == "bot_answer_self_for_pilot"
    gate = result.metadata["semantic_frame_manager_action_gate"]
    assert gate["status"] == "pass"
    assert gate["reason"] == "risk_class_not_manager_action"
    shadow = result.metadata["frame_decision_shadow"]
    assert shadow["frame"]["must_handoff"] is False
    assert shadow["comparisons"]["must_handoff_vs_route"] == "match"


def test_semantic_frame_manager_action_gate_does_not_lower_existing_handoff() -> None:
    base_result = SubscriptionDraftResult(
        route="manager_only",
        draft_text="Передам вопрос менеджеру.",
        risk_level="high",
        metadata={
            "semantic_frame_posthoc_shadow": {"status": "ok"},
            "semantic_frame": {
                "intent": "payment_receipt_check",
                "risk_class": "manager_action",
                "deal_stage": "post_payment",
                "payment_readiness": "paid",
                "requested_action": "handoff_manager",
                "answerability": "manager_only",
                "must_handoff": True,
                "confidence": 0.91,
            },
        },
    )
    provider = _SemanticFrameFakeProvider(base_result)

    result = provider.build_draft(
        "Я оплатил, чек пришёл?",
        context={
            "active_brand": "foton",
            DIRECT_PATH_ENV: "1",
            SEMANTIC_FRAME_MANAGER_ACTION_GATE_ENV: "1",
        },
    )

    assert provider.calls == 1
    assert result.route == "manager_only"
    gate = result.metadata["semantic_frame_manager_action_gate"]
    assert gate["status"] == "pass"
    assert gate["reason"] == "manager_action:handoff_manager"
    assert gate["route_before"] == "manager_only"
    assert gate["route_after"] == "manager_only"


def test_semantic_frame_manager_action_gate_ignores_inline_non_posthoc_frame() -> None:
    base_result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="Да, можно записаться на онлайн-курс.",
        metadata={
            "semantic_frame": {
                "intent": "live_availability",
                "risk_class": "manager_action",
                "deal_stage": "closing",
                "payment_readiness": "considering",
                "requested_action": "check_availability",
                "answerability": "manager_only",
                "must_handoff": True,
                "confidence": 0.95,
            },
        },
    )
    provider = _SemanticFrameFakeProvider(base_result)

    result = provider.build_draft(
        "Есть места в онлайн-группе?",
        context={
            "active_brand": "foton",
            DIRECT_PATH_ENV: "1",
            SEMANTIC_FRAME_MANAGER_ACTION_GATE_ENV: "1",
        },
    )

    assert provider.calls == 1
    assert result.route == "bot_answer_self_for_pilot"
    gate = result.metadata["semantic_frame_manager_action_gate"]
    assert gate["status"] == "frame_not_posthoc"
    assert gate["route_before"] == "bot_answer_self_for_pilot"
    assert gate["route_after"] == "bot_answer_self_for_pilot"
