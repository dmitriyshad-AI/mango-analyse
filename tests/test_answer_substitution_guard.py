from __future__ import annotations

from dataclasses import replace

import pytest

from mango_mvp.channels.subscription_llm_parts import provider as provider_module
from mango_mvp.channels.subscription_llm_parts.contracts import SubscriptionDraftResult
from mango_mvp.channels.subscription_llm_parts.post_layers import (
    _direct_path_finalize_metadata,
    apply_authoritative_output_gate,
)
from mango_mvp.channels.subscription_llm_parts.provider import SubscriptionLlmDraftProvider


class _LateMutationProvider(SubscriptionLlmDraftProvider):
    def _build_direct_path_draft(self, client_message: str, *, context=None) -> SubscriptionDraftResult:
        return SubscriptionDraftResult(
            route="bot_answer_self_for_pilot",
            draft_text="Подтверждённая информация по курсу.",
            message_type="question",
            topic_id="theme:016_program",
            metadata={"direct_path": {"enabled": True, "direct_path_attempted": True}},
        )


def test_final_gate_catches_mutation_after_earlier_provider_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    def inject_foreign_brand(result: SubscriptionDraftResult, *, context=None) -> SubscriptionDraftResult:
        return replace(result, route="bot_answer_self_for_pilot", draft_text="У УНПК МФТИ условия такие же.")

    monkeypatch.setattr(provider_module, "apply_semantic_frame_decision_shadow", inject_foreign_brand)

    result = _LateMutationProvider().build_draft("Что входит в курс?", context={"active_brand": "foton"})

    gate = result.metadata["authoritative_output_gate"]
    assert result.route not in {"bot_answer_self", "bot_answer_self_for_pilot"}
    assert "brand_leak" in {item["code"] for item in gate["findings"]}
    assert result.metadata["direct_path"]["route_after"] == result.route == gate["route_after"]


def test_direct_path_finalize_metadata_does_not_duplicate_template_trace() -> None:
    record = {"fact_key": "price.foton", "source": "kb"}
    result = SubscriptionDraftResult(
        route="draft_for_manager",
        draft_text="Менеджер проверит детали.",
        metadata={"direct_path": {"template_from_kb_trace": [record]}},
    )
    context = {"template_from_kb_trace": [record]}

    first = _direct_path_finalize_metadata(result, before_gate_route="draft_for_manager", context=context)
    second = _direct_path_finalize_metadata(first, before_gate_route="draft_for_manager", context=context)

    assert second.metadata["direct_path"]["template_from_kb_trace"] == [record]


def test_authoritative_output_gate_is_monotone_on_second_pass() -> None:
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="У УНПК МФТИ условия такие же.",
        metadata={"direct_path": {"enabled": True}},
    )

    first = apply_authoritative_output_gate(
        result,
        client_message="Что входит в курс Фотона?",
        context={"active_brand": "foton"},
    )
    second = apply_authoritative_output_gate(
        first,
        client_message="Что входит в курс Фотона?",
        context={"active_brand": "foton"},
    )

    assert second.route == first.route
    assert second.draft_text == first.draft_text
    assert second.safety_flags == first.safety_flags
    assert second.manager_checklist == first.manager_checklist
    assert second.metadata["authoritative_output_gate"]["findings"] == first.metadata["authoritative_output_gate"]["findings"]
