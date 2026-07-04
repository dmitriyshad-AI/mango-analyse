from __future__ import annotations

from mango_mvp.channels.subscription_llm_parts.contracts import SubscriptionDraftResult
from mango_mvp.channels.subscription_llm_parts.policy_routing import (
    OFF_TOPIC_FOTON_SAFE_TEXT,
    _conversation_intent_plan_with_model_led,
    apply_conversation_intent_plan_guard,
    apply_dialogue_contract_v2_template_dispatcher,
)
from mango_mvp.channels.subscription_llm_parts.provider import apply_semantic_reading_trace_finalize
from mango_mvp.channels.subscription_llm_parts.provider import SubscriptionLlmDraftProvider
from mango_mvp.channels.subscription_llm_parts.reliable_answerer import (
    RELIABLE_ANSWERER_STEP1_ENV,
    apply_reliable_answerer_output_guard,
)
from mango_mvp.channels.subscription_llm_parts.semantic_reading import (
    SEMANTIC_READING_CLASSES_ENV,
    append_reading_trace_record,
    semantic_reading_trace_record,
)
from mango_mvp.channels.subscription_llm_parts.support import INTENT_MODEL_LED_ENV


def _semantic_result(*, primary_intent: str = "faq", sense: str = "place") -> SubscriptionDraftResult:
    return SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="Места есть, можно записаться.",
        metadata={
            "direct_path_model_intent": {
                "primary_intent": primary_intent,
                "sense": sense,
                "scope": "место проведения",
                "confidence": 0.92,
            },
            "semantic_frame": {
                "source": "inline",
                "requested_action": "answer_question",
                "requested_product": {"grade": "9 класс", "subject": "физика", "format": "онлайн"},
                "confidence": 0.91,
            },
        },
    )


class _FakeDirectPipelineProvider(SubscriptionLlmDraftProvider):
    def __init__(self, result: SubscriptionDraftResult) -> None:
        super().__init__(runner=lambda *args, **kwargs: None)
        self._result = result

    def _build_direct_path_draft(self, client_message: str, *, context=None) -> SubscriptionDraftResult:  # type: ignore[override]
        del client_message, context
        return self._result


def test_direct_path_pipeline_off_masks_keeps_metadata_without_trace(monkeypatch) -> None:
    monkeypatch.delenv(SEMANTIC_READING_CLASSES_ENV, raising=False)
    baseline = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="Здравствуйте! Адрес занятий — ул. Краснопрудная, 28.",
        metadata={"direct_path": {"enabled": True, "model_called": True}, "stable_marker": "base"},
    )
    provider = _FakeDirectPipelineProvider(baseline)

    result = provider.build_draft("Где проходят занятия?", context={"TELEGRAM_DIRECT_PATH": "1"})

    assert result.route == baseline.route
    assert result.draft_text == baseline.draft_text
    assert result.metadata == baseline.metadata
    assert "semantic_reading_trace" not in result.metadata
    assert "semantic_reading_trace" not in result.metadata["direct_path"]


def test_append_reading_trace_record_mirrors_direct_path_and_finalizer_noops_without_trace() -> None:
    result = SubscriptionDraftResult(metadata={"direct_path": {"model_called": True}})

    assert apply_semantic_reading_trace_finalize(result) is result

    metadata = append_reading_trace_record(
        result.metadata,
        semantic_reading_trace_record(
            reading_class="sense_seats",
            enabled=True,
            status="no_op",
            reason="guard_not_triggered",
        ),
    )
    assert metadata["semantic_reading_trace"][0]["class"] == "sense_seats"
    assert metadata["direct_path"]["semantic_reading_trace"] == metadata["semantic_reading_trace"]


def test_append_reading_trace_record_truncates() -> None:
    metadata: dict[str, object] = {}
    for index in range(5):
        metadata = append_reading_trace_record(
            metadata,
            semantic_reading_trace_record(
                reading_class=f"class_{index}",
                enabled=True,
                status="no_op",
            ),
            max_records=3,
        )

    records = metadata["semantic_reading_trace"]
    assert len(records) == 3
    assert records[-1]["status"] == "truncated"


def test_sense_seats_trace_is_suppressed_when_reliable_step1_off() -> None:
    result = apply_reliable_answerer_output_guard(
        _semantic_result(),
        client_message="Какое место проведения?",
        context={SEMANTIC_READING_CLASSES_ENV: "sense_seats"},
    )

    trace = result.metadata["semantic_reading_trace"]
    assert trace[0]["class"] == "sense_seats"
    assert trace[0]["status"] == "suppressed"
    assert trace[0]["reason"] == "reliable_step1_off"
    assert result.route == "bot_answer_self_for_pilot"


def test_sense_seats_not_seats_does_not_disable_availability_promise_floor() -> None:
    result = apply_reliable_answerer_output_guard(
        _semantic_result(),
        client_message="Какое место проведения занятий?",
        context={
            SEMANTIC_READING_CLASSES_ENV: "sense_seats",
            RELIABLE_ANSWERER_STEP1_ENV: "1",
        },
    )

    trace = result.metadata["semantic_reading_trace"]
    assert trace[0]["decision"] == "not_seats"
    assert trace[0]["reason"] == "availability_promise_floor_kept"
    assert result.route == "draft_for_manager"
    assert "reliable_answerer_availability_promise_blocked" in result.safety_flags


def test_off_topic_reading_adds_terminal_template_and_trace() -> None:
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="Расскажу про Фотон.",
        metadata={
            "direct_path_model_intent": {
                "primary_intent": "off_topic",
                "sense": "other",
                "scope": "crypto",
                "confidence": 0.93,
            },
            "semantic_frame": {
                "source": "inline",
                "requested_action": "answer_question",
                "confidence": 0.91,
            },
        },
    )

    guarded = apply_dialogue_contract_v2_template_dispatcher(
        result,
        client_message="А что думаете про крипту?",
        context={"active_brand": "foton", SEMANTIC_READING_CLASSES_ENV: "off_topic"},
    )

    assert guarded.draft_text == OFF_TOPIC_FOTON_SAFE_TEXT
    assert guarded.metadata["semantic_reading_trace"][0]["class"] == "off_topic"
    assert guarded.metadata["semantic_reading_trace"][0]["status"] == "applied"


def test_off_topic_model_intent_remains_metadata_only_for_conversation_plan() -> None:
    plan = {
        "primary_intent": "live_availability",
        "keyword_prefilter_intents": ["live_availability"],
    }
    result = SubscriptionDraftResult(
        metadata={
            "direct_path_model_intent": {
                "primary_intent": "off_topic",
                "confidence": 0.96,
            }
        }
    )

    updated, trace = _conversation_intent_plan_with_model_led(
        plan,
        result,
        context={INTENT_MODEL_LED_ENV: "1", SEMANTIC_READING_CLASSES_ENV: "off_topic"},
        client_message="А расскажите про крипту?",
    )

    assert updated == plan
    assert trace["skip_reason"] == "off_topic_metadata_only"


def test_intent_actions_explicit_inline_check_availability_preserves_live_availability_floor() -> None:
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        topic_id="theme:001_pricing",
        draft_text="Да, можно записаться.",
        metadata={
            "semantic_frame": {
                "source": "inline",
                "requested_action": "check_availability",
                "answerability": "manager_only",
                "must_handoff": True,
                "risk_class": "manager_action",
                "confidence": 0.92,
            }
        },
    )
    guarded = apply_conversation_intent_plan_guard(
        result,
        client_message="Есть места на смену 6-17 июля?",
        context={
            SEMANTIC_READING_CLASSES_ENV: "intent_actions",
            "conversation_intent_plan": {"primary_intent": "schedule", "topic_id": "theme:013_schedule"},
        },
    )

    assert guarded.route == "draft_for_manager"
    assert "conversation_intent_plan_live_availability" in guarded.safety_flags
    assert "semantic_frame_intent_actions_live_availability" in guarded.safety_flags
    trace = guarded.metadata["semantic_reading_trace"][0]
    assert trace["class"] == "intent_actions"
    assert trace["status"] == "applied"
    assert trace["decision"] == "frame_check_availability"


def test_intent_actions_check_availability_adds_live_flag_even_when_legacy_already_manager_draft() -> None:
    result = SubscriptionDraftResult(
        route="draft_for_manager",
        topic_id="theme:001_pricing",
        draft_text="Менеджер проверит.",
        metadata={
            "semantic_frame": {
                "source": "inline",
                "requested_action": "check_availability",
                "answerability": "manager_only",
                "must_handoff": True,
                "risk_class": "manager_action",
                "confidence": 0.91,
            }
        },
    )
    guarded = apply_conversation_intent_plan_guard(
        result,
        client_message="Можно попасть в группу сейчас?",
        context={
            SEMANTIC_READING_CLASSES_ENV: "intent_actions",
            "conversation_intent_plan": {"primary_intent": "schedule", "topic_id": "theme:013_schedule"},
        },
    )

    assert guarded.route == "draft_for_manager"
    assert "conversation_intent_plan_live_availability" in guarded.safety_flags
    trace = guarded.metadata["semantic_reading_trace"][0]
    assert trace["decision"] == "frame_check_availability"
    assert trace["status"] == "applied"


def test_intent_actions_check_availability_does_not_lower_manager_only() -> None:
    result = SubscriptionDraftResult(
        route="manager_only",
        topic_id="theme:019b_negative_feedback",
        draft_text="Передам вопрос менеджеру.",
        safety_flags=("manager_approval_required", "no_auto_send", "high_risk_manager_only"),
        metadata={
            "semantic_frame": {
                "source": "inline",
                "requested_action": "check_availability",
                "answerability": "manager_only",
                "must_handoff": True,
                "risk_class": "manager_action",
                "confidence": 0.95,
            }
        },
    )
    guarded = apply_conversation_intent_plan_guard(
        result,
        client_message="Есть места? И хочу пожаловаться.",
        context={
            SEMANTIC_READING_CLASSES_ENV: "intent_actions",
            "conversation_intent_plan": {"primary_intent": "schedule", "topic_id": "theme:013_schedule"},
        },
    )

    assert guarded.route == "manager_only"
    assert "conversation_intent_plan_live_availability" in guarded.safety_flags
    assert "high_risk_manager_only" in guarded.safety_flags


def test_intent_actions_non_check_availability_actions_are_noop() -> None:
    result = SubscriptionDraftResult(
        route="manager_only",
        topic_id="theme:019b_negative_feedback",
        draft_text="Передам вопрос менеджеру.",
        safety_flags=("manager_approval_required", "no_auto_send", "high_risk_manager_only"),
        metadata={
            "semantic_frame": {
                "source": "inline",
                "requested_action": "answer_question",
                "answerability": "answer_self",
                "must_handoff": "false",
                "risk_class": "safe",
                "confidence": 0.98,
            }
        },
    )
    guarded = apply_conversation_intent_plan_guard(
        result,
        client_message="Это справочный вопрос.",
        context={
            SEMANTIC_READING_CLASSES_ENV: "intent_actions",
            "conversation_intent_plan": {"primary_intent": "refund", "topic_id": "theme:019b_negative_feedback"},
        },
    )

    assert guarded.route == "manager_only"
    assert "high_risk_manager_only" in guarded.safety_flags
    trace = guarded.metadata["semantic_reading_trace"][0]
    assert trace["status"] == "no_op"
    assert trace["metadata"]["must_handoff"] is False


def test_intent_actions_explicit_class_does_not_apply_legacy_false_p0_repair() -> None:
    result = SubscriptionDraftResult(
        route="manager_only",
        topic_id="theme:029_legal_question",
        topic_confidence=0.84,
        draft_text="Можно оформить дистанционно. Передам менеджеру запрос на запись.",
        safety_flags=("manager_approval_required", "high_risk_manager_only"),
        metadata={
            "semantic_frame": {
                "source": "inline",
                "requested_action": "answer_question",
                "answerability": "answer_self",
                "must_handoff": "false",
                "risk_class": "safe",
                "confidence": 0.96,
            }
        },
    )
    guarded = apply_conversation_intent_plan_guard(
        result,
        client_message="А чтобы записаться или с менеджером обсудить, надо приезжать или можно дистанционно?",
        context={
            SEMANTIC_READING_CLASSES_ENV: "intent_actions",
            "conversation_intent_plan": {
                "primary_intent": "format",
                "topic_id": "theme:014_format",
                "answer_policy": "answer_directly_if_fact_verified",
                "route_bias": "bot_answer_self_for_pilot",
                "risk_signals": [],
            },
        },
    )

    assert guarded.route == "manager_only"
    assert "conversation_intent_plan_false_p0_repaired" not in guarded.metadata
    assert "high_risk_manager_only" in guarded.safety_flags
    trace = guarded.metadata["semantic_reading_trace"][0]
    assert trace["status"] == "no_op"
    assert trace["decision"] == "original"
    assert trace["conflict_with"] == ["legacy_route"]
    assert trace["metadata"]["legacy_route"] == "draft_for_manager"


def test_intent_actions_invalid_requested_action_fails_closed_to_original() -> None:
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        topic_id="theme:013_schedule",
        draft_text="Да, можно записаться.",
        metadata={
            "semantic_frame": {
                "source": "inline",
                "requested_action": "reserve_live_seat",
                "answerability": "manager_only",
                "must_handoff": True,
                "risk_class": "manager_action",
                "confidence": 0.99,
            }
        },
    )
    guarded = apply_conversation_intent_plan_guard(
        result,
        client_message="Есть места?",
        context={SEMANTIC_READING_CLASSES_ENV: "intent_actions"},
    )

    assert guarded.route == "bot_answer_self_for_pilot"
    assert "conversation_intent_plan_live_availability" not in guarded.safety_flags
    trace = guarded.metadata["semantic_reading_trace"][0]
    assert trace["status"] == "fail_closed"
    assert trace["reason"] == "invalid_requested_action"
    assert trace["decision"] == "original"


def test_intent_actions_check_availability_does_not_lower_blocked_route() -> None:
    result = SubscriptionDraftResult(
        route="blocked",
        topic_id="theme:013_schedule",
        draft_text="",
        safety_flags=("authoritative_output_gate_blocked", "no_auto_send"),
        metadata={
            "semantic_frame": {
                "source": "inline",
                "requested_action": "check_availability",
                "answerability": "manager_only",
                "must_handoff": True,
                "risk_class": "manager_action",
                "confidence": 0.96,
            }
        },
    )
    guarded = apply_conversation_intent_plan_guard(
        result,
        client_message="Есть места?",
        context={SEMANTIC_READING_CLASSES_ENV: "intent_actions"},
    )

    assert guarded.route == "blocked"
    assert "authoritative_output_gate_blocked" in guarded.safety_flags
    assert "conversation_intent_plan_live_availability" in guarded.safety_flags


def test_intent_actions_posthoc_frame_fails_closed_to_legacy() -> None:
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        topic_id="theme:001_pricing",
        draft_text="Да, можно записаться.",
        metadata={
            "semantic_frame": {
                "source": "posthoc",
                "requested_action": "check_availability",
                "answerability": "manager_only",
                "must_handoff": True,
                "risk_class": "manager_action",
                "confidence": 0.98,
            }
        },
    )
    guarded = apply_conversation_intent_plan_guard(
        result,
        client_message="Есть места?",
        context={
            SEMANTIC_READING_CLASSES_ENV: "intent_actions",
            "conversation_intent_plan": {"primary_intent": "schedule", "topic_id": "theme:013_schedule"},
        },
    )

    assert guarded.route == "bot_answer_self_for_pilot"
    assert "conversation_intent_plan_live_availability" not in guarded.safety_flags
    trace = guarded.metadata["semantic_reading_trace"][0]
    assert trace["status"] == "fail_closed"
    assert trace["reason"] == "source_not_inline"


def test_intent_actions_low_confidence_frame_fails_closed_to_legacy() -> None:
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        topic_id="theme:001_pricing",
        draft_text="Да, можно записаться.",
        metadata={
            "semantic_frame": {
                "source": "inline",
                "requested_action": "check_availability",
                "answerability": "manager_only",
                "must_handoff": True,
                "risk_class": "manager_action",
                "confidence": 0.42,
            }
        },
    )
    guarded = apply_conversation_intent_plan_guard(
        result,
        client_message="Есть места?",
        context={
            SEMANTIC_READING_CLASSES_ENV: "intent_actions",
            "conversation_intent_plan": {"primary_intent": "schedule", "topic_id": "theme:013_schedule"},
        },
    )

    assert guarded.route == "bot_answer_self_for_pilot"
    assert "conversation_intent_plan_live_availability" not in guarded.safety_flags
    trace = guarded.metadata["semantic_reading_trace"][0]
    assert trace["status"] == "fail_closed"
    assert trace["reason"] == "low_confidence"


def test_intent_actions_no_frame_keeps_original_when_legacy_was_not_active() -> None:
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        topic_id="theme:001_pricing",
        draft_text="Да, можно записаться.",
        metadata={},
    )
    guarded = apply_conversation_intent_plan_guard(
        result,
        client_message="Есть места?",
        context={
            SEMANTIC_READING_CLASSES_ENV: "intent_actions",
            "conversation_intent_plan": {"primary_intent": "live_availability", "topic_id": "theme:026_camp_general"},
        },
    )

    assert guarded.route == "bot_answer_self_for_pilot"
    assert "conversation_intent_plan_live_availability" not in guarded.safety_flags
    trace = guarded.metadata["semantic_reading_trace"][0]
    assert trace["status"] == "fail_closed"
    assert trace["reason"] == "no_frame"
    assert trace["decision"] == "original"
    assert trace["conflict_with"] == ["legacy_route"]


def test_intent_actions_no_frame_keeps_legacy_when_existing_pipeline_would_apply_it() -> None:
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        topic_id="theme:001_pricing",
        draft_text="Да, можно записаться.",
        metadata={"direct_path_model_intent": {"primary_intent": "live_availability"}},
    )
    guarded = apply_conversation_intent_plan_guard(
        result,
        client_message="Есть места?",
        context={
            INTENT_MODEL_LED_ENV: "1",
            SEMANTIC_READING_CLASSES_ENV: "intent_actions",
            "conversation_intent_plan": {"primary_intent": "live_availability", "topic_id": "theme:026_camp_general"},
        },
    )

    assert guarded.route == "draft_for_manager"
    assert "conversation_intent_plan_live_availability" in guarded.safety_flags
    trace = guarded.metadata["semantic_reading_trace"][0]
    assert trace["status"] == "fail_closed"
    assert trace["reason"] == "no_frame"
    assert trace["decision"] == "legacy"
