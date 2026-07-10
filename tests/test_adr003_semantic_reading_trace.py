from __future__ import annotations

import pytest

from mango_mvp.channels.subscription_llm_parts.contracts import SubscriptionDraftResult
from mango_mvp.channels.answer_quality_rewriter import apply_answer_quality_rewriter
from mango_mvp.channels.subscription_llm_parts.policy_routing import (
    OFF_TOPIC_FOTON_SAFE_TEXT,
    SEATS_DEFAULT_OPEN_ENV,
    SEATS_DEFAULT_OPEN_REGULAR_SAFE_TEXT,
    _conversation_intent_plan_with_model_led,
    _route_templates_transition_trace,
    apply_conversation_intent_plan_guard,
    apply_dialogue_contract_v2_template_dispatcher,
    apply_known_context_redundant_question_guard,
    apply_live_status_read_plan_trace,
)
from mango_mvp.channels.subscription_llm_parts.post_layers import apply_humanity_guards
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
            "conversation_intent_plan": {"primary_intent": "live_availability", "topic_id": "theme:026_camp_general"},
        },
    )

    assert guarded.route == "draft_for_manager"
    assert "conversation_intent_plan_live_availability" in guarded.safety_flags
    assert "semantic_frame_intent_actions_live_availability" in guarded.safety_flags
    trace = guarded.metadata["semantic_reading_trace"][0]
    assert trace["class"] == "intent_actions"
    assert trace["status"] == "applied"


def test_live_status_read_records_conversation_plan_shadow_without_behavior_change() -> None:
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        topic_id="theme:001_pricing",
        draft_text="Стоимость смены зависит от программы.",
        metadata={
            "semantic_frame": {
                "source": "inline",
                "requested_action": "answer_question",
                "requested_product": {
                    "grade": "8 класс",
                    "subject": "физика",
                    "format": "очно",
                    "raw_text": "8 класс, физика, ФИО Иванов",
                },
                "answerability": "answer_self",
                "must_handoff": False,
                "risk_class": "safe",
                "confidence": 0.95,
            }
        },
    )

    guarded = apply_conversation_intent_plan_guard(
        result,
        client_message="Есть места на смену 6-17 июля?",
        context={
            SEMANTIC_READING_CLASSES_ENV: "live_status_read",
            "conversation_intent_plan": {"primary_intent": "live_availability", "topic_id": "theme:026_camp_general"},
        },
    )

    assert guarded.route == "bot_answer_self_for_pilot"
    assert "conversation_intent_plan_live_availability" not in guarded.safety_flags
    trace = guarded.metadata["semantic_reading_trace"][0]
    assert trace["class"] == "live_status_read"
    assert trace["status"] == "shadow_only"
    assert trace["metadata"]["stage"] == "conversation_intent_plan"
    assert trace["metadata"]["frame_requested_product"] == {"grade": "8 класс", "subject": "физика", "format": "очно"}
    assert "Иванов" not in str(trace)


def test_live_status_read_plan_observer_does_not_apply_legacy_route() -> None:
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        topic_id="theme:013_schedule",
        draft_text="Передам менеджеру, чтобы он проверил наличие.",
        metadata={
            "semantic_frame": {
                "source": "inline",
                "requested_action": "check_availability",
                "requested_product": {"grade": "9 класс", "subject": "математика"},
                "confidence": 0.94,
            }
        },
    )

    traced = apply_live_status_read_plan_trace(
        result,
        client_message="Есть ли свободные места в группе 9 класса по математике?",
        context={
            SEMANTIC_READING_CLASSES_ENV: "live_status_read",
            "conversation_intent_plan": {"primary_intent": "live_availability", "topic_id": "theme:013_schedule"},
        },
    )

    assert traced.route == "bot_answer_self_for_pilot"
    assert "conversation_intent_plan_live_availability" not in traced.safety_flags
    trace = traced.metadata["semantic_reading_trace"][0]
    assert trace["class"] == "live_status_read"
    assert trace["metadata"]["stage"] == "conversation_intent_plan_observer"
    assert trace["decision"] == "legacy_not_live_status"
    assert trace["changed_fields"] == []


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


def test_live_status_apply_check_availability_sets_guard_when_legacy_missed() -> None:
    result = apply_conversation_intent_plan_guard(
        _live_status_frame_result("check_availability"),
        client_message="Сколько стоит и есть ли сейчас места?",
        context={
            SEMANTIC_READING_CLASSES_ENV: "live_status_read",
            READING_APPLY_CLASSES_ENV: "live_status_read/conversation_intent_plan",
            "conversation_intent_plan": {"primary_intent": "pricing", "topic_id": "theme:013_schedule"},
        },
    )

    assert result.route == "draft_for_manager"
    assert "conversation_intent_plan_live_availability" in result.safety_flags
    assert "conversation_intent_plan_live_check_handoff" in result.safety_flags
    assert "semantic_frame_live_status_read_live_availability" in result.safety_flags
    trace = result.metadata["semantic_reading_trace"][0]
    assert trace["class"] == "live_status_read"
    assert trace["status"] == "applied"
    assert trace["decision"] == "frame_check_availability"
    assert trace["metadata"]["apply_enabled"] is True
    assert trace["conflict_with"] == ["legacy_missing_live_status"]


def test_live_status_apply_allows_manager_action_availability_frame() -> None:
    result = apply_conversation_intent_plan_guard(
        _live_status_frame_result("check_availability", risk_class="manager_action", must_handoff=True),
        client_message="Есть ли сейчас места?",
        context={
            SEMANTIC_READING_CLASSES_ENV: "live_status_read",
            READING_APPLY_CLASSES_ENV: "live_status_read/conversation_intent_plan",
            "conversation_intent_plan": {"primary_intent": "pricing", "topic_id": "theme:001_pricing"},
        },
    )

    assert result.route == "draft_for_manager"
    assert "conversation_intent_plan_live_availability" in result.safety_flags
    trace = result.metadata["semantic_reading_trace"][0]
    assert trace["status"] == "applied"
    assert trace["decision"] == "frame_check_availability"


def test_live_status_default_open_regular_group_answers_self_when_flag_enabled() -> None:
    result = apply_conversation_intent_plan_guard(
        _live_status_frame_result("check_availability", risk_class="manager_action", must_handoff=True),
        client_message="Есть ли места в группе 9 класса по математике?",
        context={
            SEMANTIC_READING_CLASSES_ENV: "live_status_read,intent_actions",
            READING_APPLY_CLASSES_ENV: "live_status_read/conversation_intent_plan",
            SEATS_DEFAULT_OPEN_ENV: "1",
            "active_brand": "foton",
            "conversation_intent_plan": {"primary_intent": "pricing", "topic_id": "theme:001_pricing"},
        },
    )

    assert result.route == "bot_answer_self_for_pilot"
    assert result.draft_text == SEATS_DEFAULT_OPEN_REGULAR_SAFE_TEXT
    assert "места есть" not in result.draft_text.casefold()
    assert "идёт набор" in result.draft_text.casefold()
    assert "seats_default_open_regular_groups" in result.safety_flags
    traces = result.metadata["semantic_reading_trace"]
    assert traces[0]["decision"] == "frame_check_availability_default_open"
    assert traces[-1]["reason"] == "seats_default_open_regular_groups"


@pytest.mark.parametrize(
    ("client_message", "product", "expected_reason"),
    [
        (
            "Есть места в ЛВШ?",
            {"grade": "8", "subject": "физика", "format": "очно", "program_kind": "camp", "raw_text": "ЛВШ Менделеево"},
            "camp_or_shift_floor",
        ),
        (
            "Забронируйте место в группе.",
            {"grade": "8", "subject": "математика", "format": "очно", "program_kind": "regular", "raw_text": "регулярная группа"},
            "booking_operation_floor",
        ),
        (
            "Есть места на индивидуальные занятия?",
            {"grade": "8", "subject": "математика", "format": "очно", "program_kind": "individual", "raw_text": "индивидуальные занятия"},
            "individual_floor",
        ),
    ],
)
def test_live_status_default_open_keeps_exception_floors(
    client_message: str,
    product: dict[str, str],
    expected_reason: str,
) -> None:
    result = apply_conversation_intent_plan_guard(
        _live_status_frame_result("check_availability", risk_class="manager_action", must_handoff=True, product=product),
        client_message=client_message,
        context={
            SEMANTIC_READING_CLASSES_ENV: "live_status_read",
            READING_APPLY_CLASSES_ENV: "live_status_read/conversation_intent_plan",
            SEATS_DEFAULT_OPEN_ENV: "1",
            "active_brand": "foton",
            "conversation_intent_plan": {"primary_intent": "pricing", "topic_id": "theme:001_pricing"},
        },
    )

    assert result.route == "draft_for_manager"
    assert result.draft_text != SEATS_DEFAULT_OPEN_REGULAR_SAFE_TEXT
    assert "seats_default_open_regular_groups" not in result.safety_flags
    trace = result.metadata["semantic_reading_trace"][0]
    assert trace["decision"] == "frame_check_availability"
    assert trace["reason"] == f"frame_live_availability:{expected_reason}"


@pytest.mark.parametrize(
    ("context_extra", "expected_reason"),
    [
        ({}, "brand_floor"),
        ({"active_brand": "foton"}, "brand_floor"),
    ],
)
def test_live_status_default_open_requires_active_brand_and_product_brand_match(
    context_extra: dict[str, str],
    expected_reason: str,
) -> None:
    product = {"grade": "8", "subject": "математика", "format": "очно", "brand": "УНПК"}
    result = apply_conversation_intent_plan_guard(
        _live_status_frame_result("check_availability", risk_class="manager_action", must_handoff=True, product=product),
        client_message="Есть места в группе?",
        context={
            SEMANTIC_READING_CLASSES_ENV: "live_status_read",
            READING_APPLY_CLASSES_ENV: "live_status_read/conversation_intent_plan",
            SEATS_DEFAULT_OPEN_ENV: "1",
            "conversation_intent_plan": {"primary_intent": "pricing", "topic_id": "theme:001_pricing"},
            **context_extra,
        },
    )

    assert result.route == "draft_for_manager"
    assert result.draft_text != SEATS_DEFAULT_OPEN_REGULAR_SAFE_TEXT
    trace = result.metadata["semantic_reading_trace"][0]
    assert trace["reason"] == f"frame_live_availability:{expected_reason}"


@pytest.mark.parametrize(
    ("client_message", "expected_reason"),
    [
        ("Оформить место в группе можно?", "booking_operation_floor"),
        ("Запишите нас в группу", "booking_operation_floor"),
        ("Сколько мест в группе обычно?", "group_size_question_floor"),
    ],
)
def test_live_status_default_open_blocks_operations_and_group_size_questions(
    client_message: str,
    expected_reason: str,
) -> None:
    result = apply_conversation_intent_plan_guard(
        _live_status_frame_result("check_availability", risk_class="manager_action", must_handoff=True),
        client_message=client_message,
        context={
            SEMANTIC_READING_CLASSES_ENV: "live_status_read",
            READING_APPLY_CLASSES_ENV: "live_status_read/conversation_intent_plan",
            SEATS_DEFAULT_OPEN_ENV: "1",
            "active_brand": "foton",
            "conversation_intent_plan": {"primary_intent": "pricing", "topic_id": "theme:001_pricing"},
        },
    )

    assert result.route == "draft_for_manager"
    assert result.draft_text != SEATS_DEFAULT_OPEN_REGULAR_SAFE_TEXT
    trace = result.metadata["semantic_reading_trace"][0]
    assert trace["reason"] == f"frame_live_availability:{expected_reason}"


def test_live_status_apply_preserves_legacy_when_both_see_availability() -> None:
    result = apply_conversation_intent_plan_guard(
        _live_status_frame_result("check_availability"),
        client_message="Есть свободные места?",
        context={
            SEMANTIC_READING_CLASSES_ENV: "live_status_read",
            READING_APPLY_CLASSES_ENV: "live_status_read/conversation_intent_plan",
            "conversation_intent_plan": {"primary_intent": "live_availability", "topic_id": "theme:026_camp_general"},
        },
    )

    assert result.route == "draft_for_manager"
    assert "conversation_intent_plan_live_availability" in result.safety_flags
    trace = result.metadata["semantic_reading_trace"][0]
    assert trace["status"] == "applied"
    assert trace["decision"] == "frame_check_availability"
    assert trace["metadata"]["legacy_live_status"] is False
    assert trace["conflict_with"] == ["legacy_missing_live_status"]


def test_live_status_apply_keeps_paid_floor_even_when_frame_is_availability() -> None:
    paid = _live_status_frame_result("check_availability", route="manager_only")
    paid = SubscriptionDraftResult(
        route=paid.route,
        topic_id=paid.topic_id,
        draft_text=paid.draft_text,
        safety_flags=("direct_path_model_p0_paid_operation_context", "high_risk_manager_only"),
        metadata=paid.metadata,
    )

    result = apply_conversation_intent_plan_guard(
        paid,
        client_message="Оплатили заранее, подтверждения места нет.",
        context={
            SEMANTIC_READING_CLASSES_ENV: "live_status_read",
            READING_APPLY_CLASSES_ENV: "live_status_read/conversation_intent_plan",
            "conversation_intent_plan": {"primary_intent": "live_availability", "topic_id": "theme:003_payment_status"},
        },
    )

    assert result.route == "manager_only"
    assert "high_risk_manager_only" in result.safety_flags
    trace = result.metadata["semantic_reading_trace"][0]
    assert trace["status"] == "fail_closed"
    assert trace["reason"] == "hard_route_floor"


def test_live_status_default_open_does_not_bypass_paid_floor() -> None:
    paid = _live_status_frame_result("check_availability", route="manager_only")
    paid = SubscriptionDraftResult(
        route=paid.route,
        topic_id=paid.topic_id,
        draft_text=paid.draft_text,
        safety_flags=("direct_path_model_p0_paid_operation_context", "high_risk_manager_only"),
        metadata=paid.metadata,
    )

    result = apply_conversation_intent_plan_guard(
        paid,
        client_message="Оплатили заранее, подтверждения места нет.",
        context={
            SEMANTIC_READING_CLASSES_ENV: "live_status_read",
            READING_APPLY_CLASSES_ENV: "live_status_read/conversation_intent_plan",
            SEATS_DEFAULT_OPEN_ENV: "1",
            "active_brand": "foton",
            "conversation_intent_plan": {"primary_intent": "live_availability", "topic_id": "theme:003_payment_status"},
        },
    )

    assert result.route == "manager_only"
    assert result.draft_text != SEATS_DEFAULT_OPEN_REGULAR_SAFE_TEXT
    trace = result.metadata["semantic_reading_trace"][0]
    assert trace["status"] == "fail_closed"
    assert trace["reason"] == "hard_route_floor"


@pytest.mark.parametrize(
    ("flags", "expected_reason"),
    [
        (("brand_separation_guarded",), "brand_floor"),
        (("payment_confirmation_without_two_sources",), "payment_confirmation_floor"),
    ],
)
def test_live_status_apply_keeps_brand_and_payment_floors(flags: tuple[str, ...], expected_reason: str) -> None:
    guarded = _live_status_frame_result("check_availability")
    guarded = SubscriptionDraftResult(
        route=guarded.route,
        topic_id=guarded.topic_id,
        draft_text=guarded.draft_text,
        safety_flags=flags,
        metadata=guarded.metadata,
    )

    result = apply_conversation_intent_plan_guard(
        guarded,
        client_message="Есть места?",
        context={
            SEMANTIC_READING_CLASSES_ENV: "live_status_read",
            READING_APPLY_CLASSES_ENV: "live_status_read/conversation_intent_plan",
            "conversation_intent_plan": {"primary_intent": "pricing", "topic_id": "theme:001_pricing"},
        },
    )

    assert result.route == "bot_answer_self_for_pilot"
    for flag in flags:
        assert flag in result.safety_flags
    assert "conversation_intent_plan_live_availability" not in result.safety_flags
    trace = result.metadata["semantic_reading_trace"][0]
    assert trace["status"] == "fail_closed"
    assert trace["reason"] == expected_reason


@pytest.mark.parametrize(
    "guarded",
    [
        SubscriptionDraftResult(
            route="bot_answer_self_for_pilot",
            topic_id="theme:009_refund",
            draft_text="Да, по проверенным фактам сориентирую.",
            metadata=_live_status_frame_result("check_availability").metadata,
        ),
        SubscriptionDraftResult(
            route="bot_answer_self_for_pilot",
            topic_id="theme:013_schedule",
            draft_text="Да, по проверенным фактам сориентирую.",
            metadata={
                **_live_status_frame_result("check_availability").metadata,
                "direct_path_model_p0": {"is_p0": True, "risk_level": "high", "p0_kind": "refund"},
            },
        ),
    ],
)
def test_live_status_apply_keeps_high_risk_floor_without_p0_flags(guarded: SubscriptionDraftResult) -> None:
    result = apply_conversation_intent_plan_guard(
        guarded,
        client_message="Есть места?",
        context={
            SEMANTIC_READING_CLASSES_ENV: "live_status_read",
            READING_APPLY_CLASSES_ENV: "live_status_read/conversation_intent_plan",
            "conversation_intent_plan": {"primary_intent": "pricing", "topic_id": "theme:001_pricing"},
        },
    )

    assert result.route == "bot_answer_self_for_pilot"
    assert "conversation_intent_plan_live_availability" not in result.safety_flags
    trace = result.metadata["semantic_reading_trace"][0]
    assert trace["status"] == "fail_closed"
    assert trace["reason"] == "p0_or_high_risk_floor"


def test_live_status_apply_keeps_semantic_risk_class_floor() -> None:
    result = apply_conversation_intent_plan_guard(
        _live_status_frame_result("check_availability", risk_class="payment_dispute"),
        client_message="Оплатили заранее, а подтверждения места нет.",
        context={
            SEMANTIC_READING_CLASSES_ENV: "live_status_read",
            READING_APPLY_CLASSES_ENV: "live_status_read/conversation_intent_plan",
            "conversation_intent_plan": {"primary_intent": "pricing", "topic_id": "theme:001_pricing"},
        },
    )

    assert result.route == "bot_answer_self_for_pilot"
    assert "conversation_intent_plan_live_availability" not in result.safety_flags
    trace = result.metadata["semantic_reading_trace"][0]
    assert trace["status"] == "fail_closed"
    assert trace["reason"] == "risk_class_floor"


def test_live_status_apply_preserves_legacy_topic_when_frame_adds_availability_guard() -> None:
    original = _live_status_frame_result("check_availability")

    result = apply_conversation_intent_plan_guard(
        original,
        client_message="Какая цена и есть ли места?",
        context={
            SEMANTIC_READING_CLASSES_ENV: "live_status_read",
            READING_APPLY_CLASSES_ENV: "live_status_read/conversation_intent_plan",
            "conversation_intent_plan": {"primary_intent": "pricing", "topic_id": "theme:001_pricing"},
        },
    )

    assert result.route == "draft_for_manager"
    assert result.topic_id == "theme:001_pricing"
    assert "conversation_intent_plan_live_availability" in result.safety_flags
    trace = result.metadata["semantic_reading_trace"][0]
    assert trace["status"] == "applied"
    assert trace["decision"] == "frame_check_availability"


@pytest.mark.parametrize("action", ["send_document", "enroll"])
def test_live_status_apply_clears_false_legacy_for_document_or_enroll_without_availability(action: str) -> None:
    result = apply_conversation_intent_plan_guard(
        _live_status_frame_result(action),
        client_message="Это справка, не бронирование.",
        context={
            SEMANTIC_READING_CLASSES_ENV: "live_status_read",
            READING_APPLY_CLASSES_ENV: "live_status_read/conversation_intent_plan",
            "conversation_intent_plan": {"primary_intent": "live_availability", "topic_id": "theme:026_camp_general"},
        },
    )

    assert result.route == "bot_answer_self_for_pilot"
    assert "conversation_intent_plan_live_availability" not in result.safety_flags
    trace = result.metadata["semantic_reading_trace"][0]
    assert trace["status"] == "applied"
    assert trace["decision"] == "frame_not_live_status"
    assert trace["reason"] == "frame_no_live_status"
    assert trace["conflict_with"] == []


@pytest.mark.parametrize(
    ("source", "confidence", "expected_reason"),
    [
        ("posthoc", 0.95, "source_not_inline"),
        ("inline", 0.70, "low_confidence"),
    ],
)
def test_live_status_apply_fail_closed_on_non_inline_or_low_confidence(source: str, confidence: float, expected_reason: str) -> None:
    result = apply_conversation_intent_plan_guard(
        _live_status_frame_result("check_availability", source=source, confidence=confidence),
        client_message="Есть места?",
        context={
            SEMANTIC_READING_CLASSES_ENV: "live_status_read",
            READING_APPLY_CLASSES_ENV: "live_status_read/conversation_intent_plan",
            "conversation_intent_plan": {"primary_intent": "pricing", "topic_id": "theme:013_schedule"},
        },
    )

    assert result.route == "bot_answer_self_for_pilot"
    trace = result.metadata["semantic_reading_trace"][0]
    assert trace["status"] == "fail_closed"
    assert trace["reason"] == expected_reason


def test_live_status_apply_class_without_reading_class_is_noop() -> None:
    original = _live_status_frame_result("check_availability")

    result = apply_conversation_intent_plan_guard(
        original,
        client_message="Есть места?",
        context={
            READING_APPLY_CLASSES_ENV: "live_status_read/conversation_intent_plan",
            "conversation_intent_plan": {"primary_intent": "pricing", "topic_id": "theme:013_schedule"},
        },
    )

    assert result.route == original.route
    assert result.draft_text == original.draft_text
    assert result.safety_flags == original.safety_flags
    assert "semantic_reading_trace" not in result.metadata


def test_live_status_apply_keeps_trace_when_intent_actions_also_enabled() -> None:
    result = apply_conversation_intent_plan_guard(
        _live_status_frame_result("check_availability"),
        client_message="Сколько стоит и есть ли сейчас места?",
        context={
            SEMANTIC_READING_CLASSES_ENV: "live_status_read,intent_actions",
            READING_APPLY_CLASSES_ENV: "live_status_read/conversation_intent_plan",
            "conversation_intent_plan": {"primary_intent": "pricing", "topic_id": "theme:001_pricing"},
        },
    )

    traces = result.metadata["semantic_reading_trace"]
    assert [trace["class"] for trace in traces] == ["live_status_read", "intent_actions"]
    assert traces[0]["status"] == "applied"
    assert traces[0]["decision"] == "frame_check_availability"
    assert result.topic_id == "theme:001_pricing"
    assert "conversation_intent_plan_live_availability" in result.safety_flags
    assert "semantic_frame_live_status_read_live_availability" in result.safety_flags


def test_live_status_apply_clears_false_legacy_even_when_intent_actions_also_enabled() -> None:
    result = apply_conversation_intent_plan_guard(
        _live_status_frame_result("send_document"),
        client_message="Это справка, не бронирование.",
        context={
            SEMANTIC_READING_CLASSES_ENV: "live_status_read,intent_actions",
            READING_APPLY_CLASSES_ENV: "live_status_read/conversation_intent_plan",
            "conversation_intent_plan": {"primary_intent": "live_availability", "topic_id": "theme:026_camp_general"},
        },
    )

    traces = result.metadata["semantic_reading_trace"]
    assert [trace["class"] for trace in traces] == ["live_status_read", "intent_actions"]
    assert traces[0]["decision"] == "frame_not_live_status"
    assert "conversation_intent_plan_live_availability" not in result.safety_flags
    assert result.route == "bot_answer_self_for_pilot"


def test_route_templates_class_records_same_stage_legacy_shadow_without_profile_default() -> None:
    result = _semantic_result(primary_intent="schedule", sense="answer_question")

    guarded = apply_conversation_intent_plan_guard(
        result,
        client_message="Когда занятия?",
        context={
            SEMANTIC_READING_CLASSES_ENV: "route_templates",
            "conversation_intent_plan": {"primary_intent": "schedule", "topic_id": "theme:013_schedule"},
        },
    )

    assert guarded.route == result.route
    assert guarded.draft_text == result.draft_text
    trace = guarded.metadata["semantic_reading_trace"][0]
    assert trace["class"] == "route_templates"
    assert trace["metadata"]["stage"] == "autonomy_matrix"
    assert trace["metadata"]["chosen"] == "legacy_more_conservative"


def test_route_templates_apply_keeps_live_availability_floor_when_frame_is_safe() -> None:
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        topic_id="theme:001_pricing",
        draft_text="Есть программа для 8 класса.",
        metadata={
            "semantic_frame": {
                "source": "inline",
                "requested_action": "answer_question",
                "answerability": "answer_self",
                "must_handoff": False,
                "risk_class": "safe",
                "confidence": 0.96,
            }
        },
    )

    guarded = apply_conversation_intent_plan_guard(
        result,
        client_message="Есть места на смену 6-17 июля?",
        context={
            SEMANTIC_READING_CLASSES_ENV: "route_templates",
            READING_APPLY_CLASSES_ENV: "route_templates/autonomy_matrix",
            "conversation_intent_plan": {"primary_intent": "live_availability", "topic_id": "theme:026_camp_general"},
        },
    )

    assert guarded.route == "bot_answer_self_for_pilot"
    assert "conversation_intent_plan_live_availability" not in guarded.safety_flags
    trace = guarded.metadata["semantic_reading_trace"][0]
    assert trace["class"] == "route_templates"
    assert trace["status"] == "fail_closed"
    assert trace["reason"] == "manual_approval_floor"
    assert trace["decision"] == "legacy_more_conservative"


def test_route_templates_apply_can_restore_safe_original_without_text_replacement_or_floor() -> None:
    original = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        topic_id="theme:013_schedule",
        draft_text="Занятия проходят онлайн.",
        metadata={
            "semantic_frame": {
                "source": "inline",
                "requested_action": "answer_question",
                "answerability": "answer_self",
                "must_handoff": False,
                "risk_class": "safe",
                "confidence": 0.95,
            }
        },
    )
    legacy = SubscriptionDraftResult(
        route="draft_for_manager",
        topic_id="theme:013_schedule",
        draft_text=original.draft_text,
        safety_flags=("manager_approval_required", "no_auto_send", "autonomy_default_cautious_topic_not_allowed"),
        metadata=original.metadata,
    )

    guarded = _route_templates_transition_trace(
        original,
        legacy,
        context={
            SEMANTIC_READING_CLASSES_ENV: "route_templates",
            READING_APPLY_CLASSES_ENV: "route_templates/autonomy_matrix",
        },
        stage="autonomy_matrix",
        reason="unit_safe_apply",
    )

    assert guarded.route == "bot_answer_self_for_pilot"
    assert guarded.draft_text == original.draft_text
    trace = guarded.metadata["semantic_reading_trace"][0]
    assert trace["status"] == "applied"
    assert trace["decision"] == "frame_safe_original"
    assert trace["metadata"]["text_replacement"] is False


def test_route_templates_apply_keeps_brand_floor() -> None:
    original = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        topic_id="theme:013_schedule",
        draft_text="Занятия проходят онлайн.",
        metadata={
            "semantic_frame": {
                "source": "inline",
                "requested_action": "answer_question",
                "answerability": "answer_self",
                "must_handoff": False,
                "risk_class": "safe",
                "confidence": 0.96,
            }
        },
    )
    legacy = SubscriptionDraftResult(
        route="draft_for_manager",
        topic_id=original.topic_id,
        draft_text=original.draft_text,
        safety_flags=("brand_separation_guarded", "manager_approval_required", "no_auto_send"),
        metadata=original.metadata,
    )

    guarded = _route_templates_transition_trace(
        original,
        legacy,
        context={
            SEMANTIC_READING_CLASSES_ENV: "route_templates",
            READING_APPLY_CLASSES_ENV: "route_templates/autonomy_matrix",
        },
        stage="autonomy_matrix",
        reason="unit_brand_floor",
    )

    assert guarded.route == "draft_for_manager"
    trace = guarded.metadata["semantic_reading_trace"][0]
    assert trace["status"] == "fail_closed"
    assert trace["reason"] == "brand_floor"
    assert trace["decision"] == "legacy_more_conservative"


def test_route_templates_apply_keeps_payment_confirmation_floor() -> None:
    original = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        topic_id="theme:002_payment_method",
        draft_text="Оплата прошла, доступ будет открыт.",
        metadata={
            "semantic_frame": {
                "source": "inline",
                "requested_action": "answer_question",
                "answerability": "answer_self",
                "must_handoff": False,
                "risk_class": "safe",
                "confidence": 0.96,
            }
        },
    )
    legacy = SubscriptionDraftResult(
        route="draft_for_manager",
        topic_id=original.topic_id,
        draft_text=original.draft_text,
        safety_flags=("payment_confirmation_without_two_sources", "manager_approval_required", "no_auto_send"),
        metadata=original.metadata,
    )

    guarded = _route_templates_transition_trace(
        original,
        legacy,
        context={
            SEMANTIC_READING_CLASSES_ENV: "route_templates",
            READING_APPLY_CLASSES_ENV: "route_templates/autonomy_matrix",
        },
        stage="autonomy_matrix",
        reason="unit_payment_floor",
    )

    assert guarded.route == "draft_for_manager"
    trace = guarded.metadata["semantic_reading_trace"][0]
    assert trace["status"] == "fail_closed"
    assert trace["reason"] == "payment_confirmation_floor"


def test_route_templates_apply_keeps_topic_id_floor() -> None:
    original = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        topic_id="theme:013_schedule",
        draft_text="Занятия проходят онлайн.",
        metadata={
            "semantic_frame": {
                "source": "inline",
                "requested_action": "answer_question",
                "answerability": "answer_self",
                "must_handoff": False,
                "risk_class": "safe",
                "confidence": 0.96,
            }
        },
    )
    legacy = SubscriptionDraftResult(
        route="draft_for_manager",
        topic_id="theme:026_camp_general",
        draft_text=original.draft_text,
        safety_flags=("autonomy_default_cautious_topic_not_allowed", "manager_approval_required", "no_auto_send"),
        metadata=original.metadata,
    )

    guarded = _route_templates_transition_trace(
        original,
        legacy,
        context={
            SEMANTIC_READING_CLASSES_ENV: "route_templates",
            READING_APPLY_CLASSES_ENV: "route_templates/autonomy_matrix",
        },
        stage="autonomy_matrix",
        reason="unit_topic_floor",
    )

    assert guarded.route == "draft_for_manager"
    trace = guarded.metadata["semantic_reading_trace"][0]
    assert trace["status"] == "fail_closed"
    assert trace["reason"] == "topic_id_floor"
    assert "topic_id" in trace["changed_fields"]


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


def test_rewrite_quality_class_records_rewriter_shadow_without_route_change() -> None:
    result = SubscriptionDraftResult(
        route="manager_only",
        draft_text="Передам вопрос менеджеру.",
        safety_flags=("manager_approval_required",),
        metadata={
            "semantic_frame": {
                "source": "inline",
                "requested_action": "answer_question",
                "confidence": 0.93,
            }
        },
    )

    guarded = apply_answer_quality_rewriter(
        result,
        client_message="Хочу пожаловаться.",
        context={SEMANTIC_READING_CLASSES_ENV: "rewrite_quality"},
    )

    assert guarded.route == result.route
    assert guarded.draft_text == result.draft_text
    trace = guarded.metadata["semantic_reading_trace"][0]
    assert trace["class"] == "rewrite_quality"
    assert trace["metadata"]["stage"] == "rewriter"
    assert trace["metadata"]["text_replacement"] is False


def test_post_semantics_class_records_humanity_text_replacement() -> None:
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="В фактах нет точной информации, передам.",
        metadata={
            "semantic_frame": {
                "source": "inline",
                "requested_action": "answer_question",
                "confidence": 0.9,
            }
        },
    )

    guarded = apply_humanity_guards(
        result,
        client_message="Есть информация?",
        context={SEMANTIC_READING_CLASSES_ENV: "post_semantics"},
    )

    assert guarded.draft_text != result.draft_text
    trace = guarded.metadata["semantic_reading_trace"][0]
    assert trace["class"] == "post_semantics"
    assert trace["metadata"]["stage"] == "humanity"
    assert trace["metadata"]["text_replacement"] is True


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


def test_intent_actions_missing_frame_with_live_plan_fails_closed_to_manager() -> None:
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

    assert guarded.route == "draft_for_manager"
    assert "conversation_intent_plan_live_availability" in guarded.safety_flags
    assert "semantic_frame_intent_actions_live_availability" in guarded.safety_flags
    trace = guarded.metadata["semantic_reading_trace"][0]
    assert trace["status"] == "fail_closed"
    assert trace["reason"] == "no_frame"
    assert trace["decision"] == "conversation_plan_live_availability_floor"
    assert "route" in trace["changed_fields"]


def test_intent_actions_missing_frame_with_legacy_live_floor_signal_fails_closed_to_manager() -> None:
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        topic_id="theme:001_pricing",
        draft_text="Да, можно записаться.",
        metadata={},
    )
    guarded = apply_conversation_intent_plan_guard(
        result,
        client_message="Можно закрепить место на ЛВШ для 8 класса?",
        context={
            SEMANTIC_READING_CLASSES_ENV: "intent_actions",
            "conversation_intent_plan": {
                "primary_intent": "camp",
                "topic_id": "theme:026_camp_general",
            },
            "conversation_intent_plan_internal": {
                "primary_intent": "camp",
                "topic_id": "theme:026_camp_general",
                "legacy_live_availability_floor_signal": True,
            },
        },
    )

    assert guarded.route == "draft_for_manager"
    assert "conversation_intent_plan_live_availability" in guarded.safety_flags
    trace = guarded.metadata["semantic_reading_trace"][0]
    assert trace["status"] == "fail_closed"
    assert trace["reason"] == "no_frame"
    assert trace["decision"] == "conversation_plan_live_availability_floor"


def test_intent_actions_invalid_frame_with_live_plan_fails_closed_to_manager() -> None:
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        topic_id="theme:001_pricing",
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
        context={
            SEMANTIC_READING_CLASSES_ENV: "intent_actions",
            "conversation_intent_plan": {"primary_intent": "live_availability", "topic_id": "theme:026_camp_general"},
        },
    )

    assert guarded.route == "draft_for_manager"
    assert "conversation_intent_plan_live_availability" in guarded.safety_flags
    trace = guarded.metadata["semantic_reading_trace"][0]
    assert trace["status"] == "fail_closed"
    assert trace["reason"] == "invalid_requested_action"
    assert trace["decision"] == "conversation_plan_live_availability_floor"


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
            "conversation_intent_plan": {"primary_intent": "schedule", "topic_id": "theme:013_schedule"},
        },
    )

    assert guarded.route == "bot_answer_self_for_pilot"
    assert "conversation_intent_plan_live_availability" not in guarded.safety_flags
    trace = guarded.metadata["semantic_reading_trace"][0]
    assert trace["status"] == "fail_closed"
    assert trace["reason"] == "no_frame"
    assert trace["decision"] == "original"
    assert trace["conflict_with"] == []


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
    assert trace["decision"] == "conversation_plan_live_availability_floor"
