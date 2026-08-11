from __future__ import annotations

import pytest

from mango_mvp.channels.subscription_llm_parts.contracts import SubscriptionDraftResult
from mango_mvp.channels.subscription_llm_parts.policy_routing import (
    OFF_TOPIC_FOTON_SAFE_TEXT,
    apply_known_context_redundant_question_guard,
)
from mango_mvp.channels.subscription_llm_parts.provider import apply_semantic_frame_manager_action_gate
from mango_mvp.channels.subscription_llm_parts.provider import apply_semantic_reading_trace_finalize
from mango_mvp.channels.subscription_llm_parts.provider import SubscriptionLlmDraftProvider
from mango_mvp.channels.subscription_llm_parts.reliable_answerer import (
    RELIABLE_ANSWERER_STEP1_ENV,
    apply_reliable_answerer_output_guard,
)
from mango_mvp.channels.subscription_llm_parts.semantic_reading import (
    READING_APPLY_CLASSES_ENV,
    SEMANTIC_READING_CLASSES_ENV,
    append_reading_trace_record,
    semantic_reading_trace_record,
)
from mango_mvp.channels.subscription_llm_parts.support import (
    DIRECT_PATH_PILOT_CONFIG_ENV,
    DIRECT_PATH_PILOT_CONFIG_VERSION,
    INTENT_MODEL_LED_ENV,
    SEATS_DEFAULT_OPEN_REGULAR_SAFE_TEXT,
)


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
        metadata={
            "direct_path": {
                "enabled": True,
                "model_called": True,
                "retrieved_facts": {"address.foton": "Адрес занятий — ул. Краснопрудная, 28."},
            },
            "dialogue_contract_pipeline": {
                "contract": {
                    "current_question": "Какой адрес занятий?",
                    "subquestions": [
                        {
                            "text": "Какой адрес занятий?",
                            "answerable": "self",
                            "needed_fact_keys": ["address.foton"],
                        }
                    ],
                    "answerability": "answer_self",
                },
                "retrieved_facts": {"address.foton": "Адрес занятий — ул. Краснопрудная, 28."},
            },
            "stable_marker": "base",
        },
    )
    provider = _FakeDirectPipelineProvider(baseline)

    result = provider.build_draft(
        "Какой адрес занятий?",
        context={
            "TELEGRAM_DIRECT_PATH": "1",
            "active_brand": "foton",
            "confirmed_facts": {"address": "Адрес занятий — ул. Краснопрудная, 28."},
        },
    )

    assert result.route == baseline.route
    assert result.draft_text == baseline.draft_text
    assert result.metadata["stable_marker"] == "base"
    assert result.metadata["authoritative_output_gate"]["action"] == "pass"
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




















def _live_status_frame_result(
    action: str,
    *,
    route: str = "bot_answer_self_for_pilot",
    source: str = "inline",
    confidence: float = 0.95,
    risk_class: str = "safe",
    must_handoff: bool = False,
    product: dict[str, str] | None = None,
) -> SubscriptionDraftResult:
    return SubscriptionDraftResult(
        route=route,
        topic_id="theme:013_schedule",
        draft_text="Да, по проверенным фактам сориентирую.",
        metadata={
            "semantic_frame": {
                "source": source,
                "requested_action": action,
                "answerability": "answer_self",
                "must_handoff": must_handoff,
                "risk_class": risk_class,
                "confidence": confidence,
                "requested_product": product or {"grade": "8", "subject": "математика", "format": "очно"},
            }
        },
    )


def test_route_templates_trace_records_known_context_reask_guard() -> None:
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="Подскажите класс ребёнка?",
        metadata={
            "semantic_frame": {
                "source": "inline",
                "requested_action": "answer_question",
                "confidence": 0.91,
            }
        },
    )

    guarded = apply_known_context_redundant_question_guard(
        result,
        client_message="А сколько стоит?",
        context={
            SEMANTIC_READING_CLASSES_ENV: "route_templates",
            "dialogue_memory_view": {"known_slots": {"grade": "8"}},
        },
    )

    assert guarded.route == "draft_for_manager"
    trace = guarded.metadata["semantic_reading_trace"][0]
    assert trace["class"] == "route_templates"
    assert trace["metadata"]["stage"] == "redundant_guard"
    assert "grade" in trace["metadata"]["repeated_fields"]


def test_live_status_read_records_reliable_answerer_facets_and_keeps_floor() -> None:
    result = apply_reliable_answerer_output_guard(
        _semantic_result(),
        client_message="Есть места на смену?",
        context={
            SEMANTIC_READING_CLASSES_ENV: "live_status_read",
            RELIABLE_ANSWERER_STEP1_ENV: "1",
        },
    )

    assert result.route == "draft_for_manager"
    assert "reliable_answerer_availability_promise_blocked" in result.safety_flags
    trace = result.metadata["semantic_reading_trace"][0]
    assert trace["class"] == "live_status_read"
    assert trace["status"] == "applied"
    assert trace["decision"] == "legacy_availability_promise_blocked"
    assert trace["metadata"]["stage"] == "reliable_answerer_output_guard"
    assert trace["metadata"]["availability_promise_detected"] is True


def test_reliable_answerer_allows_only_marked_seats_default_open_template() -> None:
    result = apply_reliable_answerer_output_guard(
        SubscriptionDraftResult(
            route="bot_answer_self_for_pilot",
            draft_text=SEATS_DEFAULT_OPEN_REGULAR_SAFE_TEXT,
            safety_flags=("seats_default_open_regular_groups",),
            metadata={
                "seats_default_open_regular_groups": True,
                "availability_promise_allowlist": "seats_default_open_regular_groups",
                "direct_path": {"seats_default_open_regular_groups": True},
            },
        ),
        client_message="Есть места?",
        context={
            SEMANTIC_READING_CLASSES_ENV: "live_status_read",
            RELIABLE_ANSWERER_STEP1_ENV: "1",
        },
    )

    assert result.route == "bot_answer_self_for_pilot"
    assert "reliable_answerer_availability_promise_blocked" not in result.safety_flags
    trace = result.metadata["semantic_reading_trace"][0]
    assert trace["status"] == "no_op"


def test_reliable_answerer_blocks_forged_seats_default_open_metadata_with_unsafe_text() -> None:
    result = apply_reliable_answerer_output_guard(
        SubscriptionDraftResult(
            route="bot_answer_self_for_pilot",
            draft_text="Да, места есть, запишем вас в группу.",
            safety_flags=("seats_default_open_regular_groups",),
            metadata={
                "seats_default_open_regular_groups": True,
                "availability_promise_allowlist": "seats_default_open_regular_groups",
                "direct_path": {"seats_default_open_regular_groups": True},
            },
        ),
        client_message="Есть места?",
        context={
            SEMANTIC_READING_CLASSES_ENV: "live_status_read",
            RELIABLE_ANSWERER_STEP1_ENV: "1",
        },
    )

    assert result.route == "draft_for_manager"
    assert "reliable_answerer_availability_promise_blocked" in result.safety_flags


def test_semantic_frame_manager_action_gate_allows_marked_seats_default_open_template() -> None:
    result = apply_semantic_frame_manager_action_gate(
        SubscriptionDraftResult(
            route="bot_answer_self_for_pilot",
            draft_text=SEATS_DEFAULT_OPEN_REGULAR_SAFE_TEXT,
            safety_flags=("seats_default_open_regular_groups",),
            metadata={
                "seats_default_open_regular_groups": True,
                "availability_promise_allowlist": "seats_default_open_regular_groups",
                "direct_path": {"seats_default_open_regular_groups": True},
                "semantic_frame_posthoc_shadow": {"status": "ok"},
                "semantic_frame": {
                    "source": "posthoc",
                    "requested_action": "check_availability",
                    "risk_class": "manager_action",
                    "answerability": "manager_only",
                    "must_handoff": True,
                    "confidence": 0.95,
                },
            },
        ),
        context={"TELEGRAM_SEMANTIC_FRAME_MANAGER_ACTION_GATE": "1"},
    )

    assert result.route == "bot_answer_self_for_pilot"
    trace = result.metadata["semantic_frame_manager_action_gate"]
    assert trace["status"] == "pass"
    assert trace["reason"] == "seats_default_open_regular_groups_allowlist"


def test_semantic_frame_manager_action_gate_blocks_forged_seats_default_open_metadata() -> None:
    result = apply_semantic_frame_manager_action_gate(
        SubscriptionDraftResult(
            route="bot_answer_self_for_pilot",
            draft_text="Да, места есть, запишем вас в группу.",
            safety_flags=("seats_default_open_regular_groups",),
            metadata={
                "seats_default_open_regular_groups": True,
                "availability_promise_allowlist": "seats_default_open_regular_groups",
                "direct_path": {"seats_default_open_regular_groups": True},
                "semantic_frame_posthoc_shadow": {"status": "ok"},
                "semantic_frame": {
                    "source": "posthoc",
                    "requested_action": "check_availability",
                    "risk_class": "manager_action",
                    "answerability": "manager_only",
                    "must_handoff": True,
                    "deal_stage": "closing",
                    "confidence": 0.95,
                },
            },
        ),
        context={"TELEGRAM_SEMANTIC_FRAME_MANAGER_ACTION_GATE": "1"},
    )

    assert result.route == "draft_for_manager"
    assert result.metadata["semantic_frame_manager_action_gate"]["status"] == "promoted_to_draft_for_manager"
