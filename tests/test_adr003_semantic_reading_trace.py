from __future__ import annotations

from mango_mvp.channels.subscription_llm_parts.contracts import SubscriptionDraftResult
from mango_mvp.channels.subscription_llm_parts.policy_routing import (
    OFF_TOPIC_FOTON_SAFE_TEXT,
    _conversation_intent_plan_with_model_led,
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
