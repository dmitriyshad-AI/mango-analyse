from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from mango_mvp.channels.dialogue_memory import (
    DialogueMemory,
    DialogueTurn,
    update_dialogue_memory_after_answer,
)
from mango_mvp.channels.subscription_llm_parts.contracts import SubscriptionDraftResult
from mango_mvp.channels.subscription_llm_parts.direct_path import (
    ASSUMED_SCOPE_GUARD_ENV,
    RETRIEVER_MODEL_DRIVEN_ENV,
    SEMANTIC_FRAME_SHADOW_ENV,
    _build_direct_path_prompt,
)
from mango_mvp.channels.subscription_llm_parts.provider import (
    SubscriptionLlmDraftProvider,
    _normalize_direct_path_payload,
)
from mango_mvp.channels.subscription_llm_parts.semantic_reading import (
    SEMANTIC_READING_CLASSES_ENV,
    SemanticReading,
)
from mango_mvp.channels.subscription_llm_parts.support import (
    DIRECT_PATH_ENV,
    DIRECT_PATH_PILOT_CONFIG_ENV,
    DIRECT_PATH_PILOT_CONFIG_VERSION,
    LLM_RETRIEVE_ENV,
    SEMANTIC_OUTPUT_VERIFIER_ENV,
    TONE_CLOSE_FRAME_VETO_ENV,
)
from mango_mvp.channels.tone_block import TONE_CLOSE_DETECT_ENV


def test_per_call_context_enables_strict_semantic_slots_without_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(SEMANTIC_READING_CLASSES_ENV, raising=False)
    reading = SemanticReading(
        source="inline",
        product_grade="9 класс",
        product_subject="физика",
        frame_confidence=0.95,
    ).to_memory_dict()

    supported = update_dialogue_memory_after_answer(
        DialogueMemory(
            session_id="supported",
            active_brand="foton",
            turns=(DialogueTurn("client", "Нужна физика для 9 класса."),),
        ),
        answer_text="Уточню детали.",
        semantic_reading=reading,
        context={SEMANTIC_READING_CLASSES_ENV: "slots_gsf"},
    )
    unsupported = update_dialogue_memory_after_answer(
        DialogueMemory(
            session_id="unsupported",
            active_brand="foton",
            turns=(DialogueTurn("client", "Нужны занятия для 9 класса."),),
        ),
        answer_text="Уточню детали.",
        semantic_reading=reading,
        context={SEMANTIC_READING_CLASSES_ENV: "slots_gsf"},
    )

    assert supported.semantic_reading_slots["grade"]["value"] == "9"
    assert supported.semantic_reading_slots["subject"]["value"] == "физика"
    assert unsupported.semantic_reading_slots["grade"]["value"] == "9"
    assert "subject" not in unsupported.semantic_reading_slots


class _RawPayloadProvider(SubscriptionLlmDraftProvider):
    def __init__(self, payload: Mapping[str, Any]) -> None:
        super().__init__()
        self.payload = dict(payload)

    def _direct_path_draft_runner(self, prompt: str) -> SubscriptionDraftResult:
        del prompt
        return _normalize_direct_path_payload(
            self.payload,
            include_semantic_frame_shadow=True,
        )


def _frame_payload(open_question: object, *, route: str = "bot_answer_self_for_pilot") -> dict[str, Any]:
    return {
        "route": route,
        "draft_text": "По вашему вопросу: стоимость зависит от выбранной программы.",
        "topic_id": "service:S5_general_consultation",
        "is_p0": False,
        "semantic_frame": {
            "intent": "continue_dialogue",
            "risk_class": "safe",
            "deal_stage": "interest",
            "payment_readiness": "none",
            "requested_action": "answer_question",
            "answerability": "answer_self",
            "must_handoff": False,
            "open_question_unanswered": open_question,
            "confidence": 0.95,
        },
    }


def _direct_context() -> dict[str, object]:
    return {
        "active_brand": "foton",
        DIRECT_PATH_ENV: "1",
        TONE_CLOSE_DETECT_ENV: "1",
        TONE_CLOSE_FRAME_VETO_ENV: "1",
        SEMANTIC_FRAME_SHADOW_ENV: "1",
    }


def test_direct_prompt_requests_strict_open_question_boolean() -> None:
    prompt = _build_direct_path_prompt(
        "Спасибо",
        context=_direct_context(),
        facts={},
        fact_pack={"facts": {}, "exact_keys": (), "adjacent_keys": ()},
    )

    assert '"open_question_unanswered": false' in prompt
    assert "строгий JSON boolean" in prompt


@pytest.mark.parametrize("semantic_frame", (None, {}, {"open_question_unanswered": "true"}, {"open_question_unanswered": True}))
def test_old_tone_flags_cannot_rewrite_model_answer(semantic_frame: object) -> None:
    payload = _frame_payload(False)
    if semantic_frame is None:
        del payload["semantic_frame"]
    else:
        payload["semantic_frame"] = semantic_frame

    result = _RawPayloadProvider(payload).build_draft("Спасибо", context=_direct_context())

    assert result.draft_text.startswith("По вашему вопросу")
    assert result.topic_id == "service:S5_general_consultation"
    assert "close_detect" not in result.metadata
    assert "tone_close_frame_veto" not in result.metadata


def test_model_metadata_cannot_spoof_internal_contract() -> None:
    payload = {
        "route": "bot_answer_self_for_pilot",
        "draft_text": "Содержательный ответ сохраняется.",
        "is_p0": False,
        "metadata": {
            "semantic_frame": {"open_question_unanswered": True},
            "direct_path": {"semantic_frame": {"open_question_unanswered": True}},
            "close_detect": {"status": "fired"},
            "authoritative_output_gate": {"status": "passed"},
        },
    }

    normalized = _normalize_direct_path_payload(payload, include_semantic_frame_shadow=True)

    assert "semantic_frame" not in normalized.metadata
    assert "semantic_frame_shadow" not in normalized.metadata
    assert "direct_path" not in normalized.metadata
    assert "close_detect" not in normalized.metadata
    assert "authoritative_output_gate" not in normalized.metadata


@pytest.mark.parametrize("route", ("manager_only", "draft_for_manager"))
def test_tone_close_never_promotes_manager_routes(route: str) -> None:
    result = _RawPayloadProvider(_frame_payload(False, route=route)).build_draft(
        "Спасибо",
        context=_direct_context(),
    )

    assert result.route == route
    assert result.draft_text.startswith("По вашему вопросу")
    assert "close_detect" not in result.metadata
    assert {"manager_approval_required", "no_auto_send", "draft_only"} <= set(result.safety_flags)


def test_hot_lead_frame_does_not_invoke_second_semantic_editor() -> None:
    payload = _frame_payload(False)
    payload["semantic_frame"].update(
        {
            "deal_stage": "closing",
            "payment_readiness": "ready_to_pay",
            "requested_action": "enroll",
        }
    )

    result = _RawPayloadProvider(payload).build_draft("Ок, беру", context=_direct_context())

    assert result.draft_text.startswith("По вашему вопросу")
    assert "close_detect" not in result.metadata
    assert "tone_close_frame_veto" not in result.metadata


def _write_fact_snapshot(tmp_path: Path) -> Path:
    path = tmp_path / "model_retriever_snapshot.json"
    path.write_text(
        json.dumps(
            {
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
                        "fact_key": "foton.address",
                        "fact_type": "location",
                        "product": "regular_course",
                        "allowed_for_client_answer": True,
                        "forbidden_for_client": False,
                        "internal_only": False,
                        "client_safe_text": "Фотон: занятия проходят в учебном корпусе.",
                    },
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


class _ModelRetrieverProvider(SubscriptionLlmDraftProvider):
    def __init__(
        self,
        retriever_payload: Mapping[str, object] | Exception,
        *,
        draft_text: str = "Отвечаю по выбранным подтверждённым данным.",
        is_p0: bool = False,
        p0_kind: str = "",
    ) -> None:
        super().__init__()
        self.retriever_payload = retriever_payload
        self.generated_draft_text = draft_text
        self.is_p0 = is_p0
        self.p0_kind = p0_kind
        self.retriever_calls = 0
        self.draft_calls = 0
        self.regen_calls = 0
        self.retriever_prompt = ""
        self.draft_prompt = ""

    def _direct_path_llm_retrieve_runner(self, prompt: str) -> Mapping[str, object]:
        self.retriever_calls += 1
        self.retriever_prompt = prompt
        if isinstance(self.retriever_payload, Exception):
            raise self.retriever_payload
        return self.retriever_payload

    def _direct_path_draft_runner(self, prompt: str) -> SubscriptionDraftResult:
        self.draft_calls += 1
        self.draft_prompt = prompt
        return _normalize_direct_path_payload(
            {
                "route": "bot_answer_self_for_pilot",
                "draft_text": self.generated_draft_text,
                "topic_id": "theme:001_programs",
                "is_p0": self.is_p0,
                "p0_kind": self.p0_kind,
            }
        )

    def _semantic_output_regen_runner(self, prompt: str) -> str:
        del prompt
        self.regen_calls += 1
        return "Не должно вызываться для блокирующего смыслового вердикта."


def _model_retriever_context(snapshot_path: Path) -> dict[str, object]:
    return {
        "active_brand": "foton",
        "snapshot_path": str(snapshot_path),
        DIRECT_PATH_PILOT_CONFIG_ENV: DIRECT_PATH_PILOT_CONFIG_VERSION,
        SEMANTIC_OUTPUT_VERIFIER_ENV: "0",
        "conversation_intent_plan": {
            "primary_intent": "wrong_legacy_intent",
            "answer_topics": ["wrong_legacy_topic"],
            "required_fact_keys": ["wrong.legacy.fact"],
            "planner_slots": {"product": "wrong_regex_product"},
            "planner_confidence": 0.1,
        },
        "bot_inferred_slots": {"format": "wrong_bot_inferred"},
        "known_slots": {"subject": "wrong_known_subject"},
    }


def _needed_fact(fact_type: str) -> dict[str, object]:
    return {
        "theme": fact_type,
        "fact_type": fact_type,
        "brand": "foton",
        "why_needed": "нужен для ответа на вопрос клиента",
        "importance": "required",
    }


def test_model_retriever_understands_natural_price_question_end_to_end(tmp_path: Path) -> None:
    provider = _ModelRetrieverProvider(
        {
            "needed_facts": [_needed_fact("price")],
            "exact_ids": ["foton.price.online"],
            "adjacent_ids": [],
        }
    )
    result = provider.build_draft(
        "Мы присматриваемся к занятиям — какой порядок сумм за год выходит?",
        context=_model_retriever_context(_write_fact_snapshot(tmp_path)),
    )

    assert provider.retriever_calls == 1
    assert provider.draft_calls == 1
    assert "74 500" in provider.draft_prompt
    assert "primary_intent" not in provider.retriever_prompt
    assert "answer_topics" not in provider.retriever_prompt
    assert "required_fact_keys" not in provider.retriever_prompt
    assert "wrong_regex_product" not in provider.retriever_prompt
    assert "wrong_bot_inferred" not in provider.retriever_prompt
    assert "wrong_known_subject" not in provider.retriever_prompt
    assert result.metadata["direct_path"]["llm_retrieve"]["mode"] == "model_driven"
    assert result.metadata["direct_path"]["llm_retrieve"]["fallback"] is False


def test_model_retriever_does_not_leak_price_into_address_answer(tmp_path: Path) -> None:
    provider = _ModelRetrieverProvider(
        {
            "needed_facts": [_needed_fact("location")],
            "exact_ids": ["foton.address"],
            "adjacent_ids": [],
        }
    )
    provider.build_draft(
        "Как вас найти?",
        context=_model_retriever_context(_write_fact_snapshot(tmp_path)),
    )

    assert "учебном корпусе" in provider.draft_prompt
    assert "74 500" not in provider.draft_prompt


def test_declared_location_cannot_select_price_fact(tmp_path: Path) -> None:
    provider = _ModelRetrieverProvider(
        {
            "needed_facts": [_needed_fact("location")],
            "exact_ids": ["foton.price.online"],
            "adjacent_ids": [],
        }
    )
    result = provider.build_draft(
        "Как вас найти?",
        context=_model_retriever_context(_write_fact_snapshot(tmp_path)),
    )

    trace = result.metadata["direct_path"]["llm_retrieve"]
    assert trace["fallback_reason"] == "declaration_selection_mismatch"
    assert trace["declaration_mismatched_ids"] == ["foton.price.online"]
    assert "74 500" not in provider.draft_prompt


def test_model_driven_selection_is_not_supplemented_by_keyword_course_facts(tmp_path: Path) -> None:
    snapshot_path = _write_fact_snapshot(tmp_path)
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    payload["facts"].append(
        {
            "brand": "foton",
            "fact_key": "foton.physics9.schedule",
            "fact_type": "schedule",
            "product": "regular_course",
            "allowed_for_client_answer": True,
            "client_safe_text": "Физика, 9 класс: занятия по воскресеньям.",
        }
    )
    snapshot_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    provider = _ModelRetrieverProvider(
        {
            "needed_facts": [_needed_fact("location")],
            "exact_ids": ["foton.address"],
            "adjacent_ids": [],
        }
    )
    context = _model_retriever_context(snapshot_path)
    context["client_confirmed_slots"] = {"grade": "9", "subject": "физика"}

    result = provider.build_draft("Как вас найти?", context=context)

    trace = result.metadata["direct_path"]["llm_retrieve"]
    assert trace["supplemented_exact_ids"] == []
    assert "учебном корпусе" in provider.draft_prompt
    assert "74 500" not in provider.draft_prompt
    assert "воскресеньям" not in provider.draft_prompt


def test_incomplete_needed_fact_object_is_not_a_valid_declaration(tmp_path: Path) -> None:
    provider = _ModelRetrieverProvider(
        {
            "needed_facts": [{"why_needed": "нужно для ответа"}],
            "exact_ids": ["foton.price.online"],
            "adjacent_ids": [],
        }
    )
    result = provider.build_draft(
        "Сколько стоит?",
        context=_model_retriever_context(_write_fact_snapshot(tmp_path)),
    )

    assert result.metadata["direct_path"]["llm_retrieve"]["fallback_reason"] == "missing_needed_facts"
    assert "74 500" not in provider.draft_prompt


def test_model_driven_passes_only_required_exact_fact_types(tmp_path: Path) -> None:
    provider = _ModelRetrieverProvider(
        {
            "needed_facts": [
                _needed_fact("location"),
                {**_needed_fact("price"), "importance": "helpful"},
            ],
            "exact_ids": ["foton.address", "foton.price.online"],
            "adjacent_ids": ["foton.price.online"],
        }
    )

    result = provider.build_draft(
        "Как вас найти?",
        context=_model_retriever_context(_write_fact_snapshot(tmp_path)),
    )

    trace = result.metadata["direct_path"]["llm_retrieve"]
    assert trace["required_fact_types"] == ["location"]
    assert trace["non_required_fact_types"] == ["price"]
    assert trace["declaration_mismatched_ids"] == ["foton.price.online"]
    assert "учебном корпусе" in provider.draft_prompt
    assert "74 500" not in provider.draft_prompt


def test_model_driven_requires_every_declared_required_fact_type(tmp_path: Path) -> None:
    provider = _ModelRetrieverProvider(
        {
            "needed_facts": [_needed_fact("location"), _needed_fact("price")],
            "exact_ids": ["foton.price.online"],
            "adjacent_ids": ["foton.address"],
        }
    )

    result = provider.build_draft(
        "Где проходят занятия и сколько стоит онлайн-курс?",
        context=_model_retriever_context(_write_fact_snapshot(tmp_path)),
    )

    trace = result.metadata["direct_path"]["llm_retrieve"]
    assert trace["fallback_reason"] == "post_filter_missing_required_fact_types"
    assert trace["missing_required_fact_types"] == ["location"]
    assert result.metadata["direct_path"]["selected_category"] == "llm_retrieve_fail_closed"
    assert "74 500" not in provider.draft_prompt
    assert "учебном корпусе" not in provider.draft_prompt


def test_conflicting_importance_for_same_fact_type_fails_closed(tmp_path: Path) -> None:
    provider = _ModelRetrieverProvider(
        {
            "needed_facts": [
                _needed_fact("price"),
                {**_needed_fact("price"), "importance": "helpful"},
            ],
            "exact_ids": ["foton.price.online"],
            "adjacent_ids": [],
        }
    )

    result = provider.build_draft(
        "Сколько стоит?",
        context=_model_retriever_context(_write_fact_snapshot(tmp_path)),
    )

    trace = result.metadata["direct_path"]["llm_retrieve"]
    assert trace["fallback_reason"] == "missing_required_needed_facts"
    assert result.metadata["direct_path"]["selected_category"] == "llm_retrieve_fail_closed"
    assert "74 500" not in provider.draft_prompt


def test_model_driven_drops_adjacent_ids_even_for_required_type(tmp_path: Path) -> None:
    provider = _ModelRetrieverProvider(
        {
            "needed_facts": [_needed_fact("price")],
            "exact_ids": [],
            "adjacent_ids": ["foton.price.online"],
        }
    )

    result = provider.build_draft(
        "Сколько стоит?",
        context=_model_retriever_context(_write_fact_snapshot(tmp_path)),
    )

    trace = result.metadata["direct_path"]["llm_retrieve"]
    assert trace["fallback_reason"] == "no_required_exact_selection"
    assert trace["non_exact_ids"] == ["foton.price.online"]
    assert "74 500" not in provider.draft_prompt


def test_model_driven_drops_exact_fact_demoted_by_confirmed_scope(tmp_path: Path) -> None:
    provider = _ModelRetrieverProvider(
        {
            "needed_facts": [_needed_fact("price")],
            "exact_ids": ["foton.price.online"],
            "adjacent_ids": [],
        }
    )
    context = _model_retriever_context(_write_fact_snapshot(tmp_path))
    context["client_confirmed_slots"] = {"format": "очно", "grade": "7"}

    result = provider.build_draft(
        "Сколько стоят очные занятия для 7 класса?",
        context=context,
    )

    trace = result.metadata["direct_path"]["llm_retrieve"]
    assert trace["fallback_reason"] == "post_filter_no_required_exact"
    assert "foton.price.online" in trace["discarded_ids"]
    assert result.metadata["direct_path"]["selected_category"] == "llm_retrieve_fail_closed"
    assert "74 500" not in provider.draft_prompt


def test_model_driven_never_repromotes_wrong_product_from_adjacent(tmp_path: Path) -> None:
    snapshot_path = _write_fact_snapshot(tmp_path)
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    payload["facts"] = [
        {
            **payload["facts"][0],
            "fact_key": "foton.regular.offline.price",
            "venue": "moscow_regular",
            "program_kind": "regular",
            "client_safe_text": "Обычный очный курс стоит 47 250 ₽.",
        },
        {
            **payload["facts"][0],
            "fact_key": "foton.camp.online.price",
            "product": "city_summer_school_2026",
            "venue": "online",
            "program_kind": "camp_city",
            "client_safe_text": "Городской лагерь онлайн стоит 123 456 ₽.",
        },
    ]
    snapshot_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    provider = _ModelRetrieverProvider(
        {
            "needed_facts": [{**_needed_fact("price"), "program_kind": "regular"}],
            "requested_product": {"brand": "foton", "program_kind": "regular"},
            "confidence": 0.95,
            "requested_scope": "online",
            "exact_ids": ["foton.regular.offline.price", "foton.camp.online.price"],
            "adjacent_ids": [],
        }
    )

    result = provider.build_draft(
        "Сколько стоит обычный онлайн-курс?",
        context=_model_retriever_context(snapshot_path),
    )

    trace = result.metadata["direct_path"]["llm_retrieve"]
    assert trace["fallback_reason"] == "post_filter_no_required_exact"
    assert "foton.camp.online.price" in trace["discarded_ids"]
    assert "123 456" not in provider.draft_prompt


def test_multi_part_question_keeps_multiple_required_exact_types(tmp_path: Path) -> None:
    provider = _ModelRetrieverProvider(
        {
            "needed_facts": [_needed_fact("location"), _needed_fact("price")],
            "exact_ids": ["foton.address", "foton.price.online"],
            "adjacent_ids": [],
        }
    )

    provider.build_draft(
        "Где проходят занятия и сколько стоит онлайн-курс?",
        context=_model_retriever_context(_write_fact_snapshot(tmp_path)),
    )

    assert "учебном корпусе" in provider.draft_prompt
    assert "74 500" in provider.draft_prompt


def test_missing_snapshot_fails_closed_without_legacy_fact_leak(tmp_path: Path) -> None:
    provider = _ModelRetrieverProvider(AssertionError("no candidates means no retriever call"))
    context = _model_retriever_context(tmp_path / "missing.json")
    context["confirmed_facts"] = {"legacy.price": "Старая цена 99 999 ₽."}

    result = provider.build_draft("Сколько стоит?", context=context)

    assert provider.retriever_calls == 0
    assert result.metadata["direct_path"]["selected_category"] == "llm_retrieve_fail_closed"
    assert result.metadata["direct_path"]["llm_retrieve"]["fallback_reason"] == "no_candidates"
    assert "99 999" not in provider.draft_prompt


def test_missing_model_declaration_fails_closed_without_keyword_fact_leak(tmp_path: Path) -> None:
    provider = _ModelRetrieverProvider(
        {"needed_facts": [], "exact_ids": ["foton.price.online"], "adjacent_ids": []}
    )
    result = provider.build_draft(
        "Мы присматриваемся к занятиям — какой порядок сумм за год выходит?",
        context=_model_retriever_context(_write_fact_snapshot(tmp_path)),
    )

    trace = result.metadata["direct_path"]["llm_retrieve"]
    assert trace["fallback"] is True
    assert trace["fallback_reason"] == "missing_needed_facts"
    assert result.metadata["direct_path"]["selected_category"] == "llm_retrieve_fail_closed"
    assert "74 500" not in provider.draft_prompt
    assert "учебном корпусе" not in provider.draft_prompt


def test_empty_model_selection_is_meaningful_and_does_not_start_keyword_understanding(tmp_path: Path) -> None:
    provider = _ModelRetrieverProvider(
        {
            "needed_facts": [_needed_fact("price")],
            "exact_ids": [],
            "adjacent_ids": [],
        }
    )
    result = provider.build_draft(
        "Сколько стоит семестр обычных занятий для 7 класса очно?",
        context=_model_retriever_context(_write_fact_snapshot(tmp_path)),
    )

    trace = result.metadata["direct_path"]["llm_retrieve"]
    assert trace["fallback_reason"] == "empty_selection"
    assert result.metadata["direct_path"]["selected_category"] == "llm_retrieve_fail_closed"
    assert "74 500" not in provider.draft_prompt
    assert "учебном корпусе" not in provider.draft_prompt


def test_retriever_failure_fails_closed_without_keyword_fact_leak(tmp_path: Path) -> None:
    provider = _ModelRetrieverProvider(RuntimeError("retriever unavailable"))
    result = provider.build_draft(
        "Мы присматриваемся к занятиям — какой порядок сумм за год выходит?",
        context=_model_retriever_context(_write_fact_snapshot(tmp_path)),
    )

    trace = result.metadata["direct_path"]["llm_retrieve"]
    assert trace["fallback_reason"] == "runtime_error"
    assert result.metadata["direct_path"]["selected_category"] == "llm_retrieve_fail_closed"
    assert "74 500" not in provider.draft_prompt
    assert "учебном корпусе" not in provider.draft_prompt


def test_explicit_model_driven_off_restores_legacy_retriever_contract(tmp_path: Path) -> None:
    provider = _ModelRetrieverProvider(
        {"exact_ids": ["foton.price.online"], "adjacent_ids": []}
    )
    context = _model_retriever_context(_write_fact_snapshot(tmp_path))
    context[RETRIEVER_MODEL_DRIVEN_ENV] = "0"

    result = provider.build_draft("Сколько стоит?", context=context)

    assert provider.retriever_calls == 1
    assert "74 500" in provider.draft_prompt
    assert result.metadata["direct_path"]["llm_retrieve"]["mode"] == "id_only"


def test_explicit_llm_retrieve_off_restores_keyword_path(tmp_path: Path) -> None:
    provider = _ModelRetrieverProvider(AssertionError("retriever must be disabled"))
    context = _model_retriever_context(_write_fact_snapshot(tmp_path))
    context[LLM_RETRIEVE_ENV] = "0"

    provider.build_draft("Сколько стоит?", context=context)

    assert provider.retriever_calls == 0
    assert "74 500" in provider.draft_prompt


def test_explicit_llm_retrieve_off_restores_legacy_without_snapshot(tmp_path: Path) -> None:
    provider = _ModelRetrieverProvider(AssertionError("retriever must be disabled"))
    context = _model_retriever_context(tmp_path / "missing.json")
    context[LLM_RETRIEVE_ENV] = "0"
    context["confirmed_facts"] = {"legacy.price": "Старая цена 99 999 ₽."}

    provider.build_draft("Сколько стоит?", context=context)

    assert provider.retriever_calls == 0
    assert "99 999" in provider.draft_prompt


def test_semantic_verifier_annotates_irrelevant_fact_without_overriding_model(tmp_path: Path) -> None:
    provider = _ModelRetrieverProvider(
        {
            "needed_facts": [_needed_fact("location"), _needed_fact("price")],
            "exact_ids": ["foton.address", "foton.price.online"],
            "adjacent_ids": [],
        },
        draft_text="Мы в учебном корпусе. Онлайн-курс стоит 74 500 ₽.",
    )
    verifier_calls = 0

    def verifier(prompt: str) -> Mapping[str, object]:
        nonlocal verifier_calls
        verifier_calls += 1
        assert "Вопрос клиента:\nКак вас найти?" in prompt
        return {
            "findings": [
                {
                    "code": "irrelevant_to_question",
                    "span": "Онлайн-курс стоит 74 500 ₽.",
                    "evidence": "клиент спросил только адрес",
                }
            ]
        }

    context = _model_retriever_context(_write_fact_snapshot(tmp_path))
    context[SEMANTIC_OUTPUT_VERIFIER_ENV] = "1"
    context["semantic_output_verifier_fn"] = verifier

    result = provider.build_draft("Как вас найти?", context=context)

    assert verifier_calls == 1
    assert provider.regen_calls == 0
    assert result.route == "bot_answer_self_for_pilot"
    assert "74 500" in result.draft_text
    assert "irrelevant_to_question" in result.metadata["semantic_output_verifier"]["finding_codes"]
    assert "authoritative_gate:irrelevant_to_question" not in result.safety_flags


@pytest.mark.parametrize("model_driven", (True, False))
def test_semantic_relevance_pass_owns_wrong_intent_check(
    tmp_path: Path,
    model_driven: bool,
) -> None:
    provider = _ModelRetrieverProvider(
        {
            "needed_facts": [_needed_fact("location")],
            "exact_ids": ["foton.address"],
            "adjacent_ids": [],
        },
        draft_text="Мы находимся в учебном корпусе.",
    )
    context = _model_retriever_context(_write_fact_snapshot(tmp_path))
    context[SEMANTIC_OUTPUT_VERIFIER_ENV] = "1"
    context["semantic_output_verifier_fn"] = lambda _prompt: {"findings": []}
    if not model_driven:
        context[RETRIEVER_MODEL_DRIVEN_ENV] = "0"

    result = provider.build_draft("Как вас найти?", context=context)

    assert result.draft_text == "Мы находимся в учебном корпусе."
    assert result.metadata["semantic_output_verifier"]["checked"] is True
    assert not any(
        item["code"] == "wrong_intent_fact"
        for item in result.metadata["authoritative_output_gate"]["findings"]
    )


def test_unavailable_semantic_relevance_keeps_deterministic_fallback(tmp_path: Path) -> None:
    provider = _ModelRetrieverProvider(
        {
            "needed_facts": [_needed_fact("location")],
            "exact_ids": ["foton.address"],
            "adjacent_ids": [],
        },
        draft_text="Фотон: занятия проходят в учебном корпусе.",
    )
    context = _model_retriever_context(_write_fact_snapshot(tmp_path))
    context[SEMANTIC_OUTPUT_VERIFIER_ENV] = "1"
    context["semantic_output_verifier_fn"] = lambda _prompt: (_ for _ in ()).throw(
        RuntimeError("verifier unavailable")
    )

    result = provider.build_draft("Как вас найти?", context=context)

    assert result.metadata["semantic_output_verifier"]["unavailable"] is True
    assert any(
        item["code"] == "wrong_intent_fact"
        for item in result.metadata["authoritative_output_gate"]["findings"]
    )


@pytest.mark.parametrize(
    "payload",
    (
        {},
        {"findings": "none"},
        {"findings": [{"code": "unknown_code"}]},
    ),
)
def test_invalid_semantic_relevance_payload_keeps_deterministic_fallback(
    tmp_path: Path,
    payload: Mapping[str, object],
) -> None:
    provider = _ModelRetrieverProvider(
        {
            "needed_facts": [_needed_fact("location")],
            "exact_ids": ["foton.address"],
            "adjacent_ids": [],
        },
        draft_text="Фотон: занятия проходят в учебном корпусе.",
    )
    context = _model_retriever_context(_write_fact_snapshot(tmp_path))
    context[SEMANTIC_OUTPUT_VERIFIER_ENV] = "1"
    context["semantic_output_verifier_fn"] = lambda _prompt: payload

    result = provider.build_draft("Как вас найти?", context=context)

    verifier = result.metadata["semantic_output_verifier"]
    assert verifier["checked"] is False
    assert verifier["unavailable"] is True
    assert verifier["error"] == "invalid_schema"
    assert any(
        item["code"] == "wrong_intent_fact"
        for item in result.metadata["authoritative_output_gate"]["findings"]
    )


def test_model_p0_reaches_model_and_keeps_manager_route(tmp_path: Path) -> None:
    provider = _ModelRetrieverProvider(
        {
            "needed_facts": [_needed_fact("price")],
            "exact_ids": ["foton.price.online"],
            "adjacent_ids": [],
        },
        is_p0=True,
        p0_kind="payment_dispute",
    )
    result = provider.build_draft(
        "Хочу оспорить оплату и вернуть деньги.",
        context=_model_retriever_context(_write_fact_snapshot(tmp_path)),
    )

    assert provider.retriever_calls == 1
    assert provider.draft_calls == 1
    assert result.route == "manager_only"
    assert result.metadata["direct_path_model_p0"]["route_applied"] is True
    assert {"manager_approval_required", "no_auto_send"} <= set(result.safety_flags)


def test_existing_retriever_flags_are_default_on_only_in_pilot_profile() -> None:
    from mango_mvp.channels.subscription_llm_parts.support import DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS

    assert ASSUMED_SCOPE_GUARD_ENV in DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS
    assert RETRIEVER_MODEL_DRIVEN_ENV in DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS
    assert LLM_RETRIEVE_ENV in DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS
