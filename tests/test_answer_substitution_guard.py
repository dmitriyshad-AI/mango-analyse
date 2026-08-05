from __future__ import annotations

from dataclasses import replace

import pytest

from mango_mvp.channels.subscription_llm_parts import provider as provider_module
from mango_mvp.channels.subscription_llm_parts.contracts import SubscriptionDraftResult
from mango_mvp.channels.subscription_llm_parts.policy_routing import apply_autonomy_matrix_guard
from mango_mvp.channels.subscription_llm_parts.post_layers import (
    _direct_path_finalize_metadata,
    apply_authoritative_output_gate,
)
from mango_mvp.channels.subscription_llm_parts.provider import SubscriptionLlmDraftProvider


@pytest.mark.parametrize(
    ("client_message", "topic_id", "unrelated_fact"),
    (
        ("Оставлю email для оформления.", "theme:020_enrollment", "Адрес филиала: Сретенка, 20."),
        ("Как оформить оплату маткапиталом?", "theme:007_matkap_payment", "Курс стоит 44 600 рублей."),
        ("Ребёнок уже записан в городской лагерь, как оплатить?", "theme:026_camp_general", "ЛВШ проходит в Менделеево."),
        ("Когда подписать документы?", "theme:011_contract", "Для лагеря нужна медицинская справка."),
        ("Памятка не пришла на почту.", "theme:016_program", "Есть программы по химии и английскому."),
        ("Можно посмотреть договор до оплаты?", "theme:011_contract", "Занятия проходят очно и онлайн."),
        ("Есть вопросы к договору перед оплатой.", "theme:011_contract", "Адрес площадки: Чистые пруды."),
    ),
)
def test_low_value_manager_draft_is_not_replaced_or_promoted(
    client_message: str,
    topic_id: str,
    unrelated_fact: str,
) -> None:
    draft_text = "Передам вопрос менеджеру, чтобы он проверил детали."
    result = SubscriptionDraftResult(
        route="draft_for_manager",
        draft_text=draft_text,
        message_type="question",
        topic_id=topic_id,
    )
    context = {
        "active_brand": "foton",
        "autonomy_policy": {"allow_autonomous": True, "allowed_topic_ids": [topic_id]},
        "facts_context": {"client_safe": True, "fresh": True, "facts_missing": False},
        "confirmed_facts": {"unrelated": unrelated_fact},
    }

    guarded = apply_autonomy_matrix_guard(result, client_message=client_message, context=context)

    assert guarded.route == "draft_for_manager"
    assert guarded.draft_text == draft_text
    assert unrelated_fact not in guarded.draft_text
    assert "autonomy_matrix_kept_unverified_draft" in guarded.safety_flags
    assert "autonomy_verified_fact_answer_template_applied" not in guarded.safety_flags


def test_useful_verified_draft_can_still_be_promoted_without_text_rewrite() -> None:
    draft_text = (
        "Для 8 класса очный курс по математике рассчитан на системную подготовку: "
        "занятия идут в группе, программа курса охватывает школьные темы и задачи повышенной сложности. "
        "Можно выбрать подходящий формат и продолжить оформление."
    )
    result = SubscriptionDraftResult(
        route="draft_for_manager",
        draft_text=draft_text,
        message_type="question",
        topic_id="theme:016_program",
        metadata={
            "direct_path": {"retrieved_facts": {"program": draft_text}},
            "dialogue_contract_pipeline": {
                "contract": {
                    "current_question": "Что входит в курс?",
                    "subquestions": [
                        {
                            "text": "Что входит в курс?",
                            "answerable": "self",
                            "needed_fact_keys": ["program"],
                        }
                    ],
                    "answerability": "answer_self",
                },
                "retrieved_facts": {"program": draft_text},
            },
        },
    )
    context = {
        "active_brand": "foton",
        "autonomy_policy": {"allow_autonomous": True, "allowed_topic_ids": ["theme:016_program"]},
        "facts_context": {"client_safe": True, "fresh": True, "facts_missing": False},
        "confirmed_facts": {"program": draft_text},
    }

    guarded = apply_autonomy_matrix_guard(result, client_message="Что входит в курс?", context=context)

    assert guarded.route == "bot_answer_self_for_pilot"
    assert guarded.draft_text == draft_text
    assert "autonomy_matrix_promoted_safe_draft" in guarded.safety_flags


def test_long_unverified_manager_draft_is_not_promoted() -> None:
    draft_text = (
        "Можно рассмотреть несколько вариантов обучения и выбрать удобный формат для семьи. "
        "Программа строится последовательно, а преподаватель помогает двигаться дальше. "
        "Подробности зависят от конкретной группы и ситуации ребёнка."
    )
    result = SubscriptionDraftResult(
        route="draft_for_manager",
        draft_text=draft_text,
        message_type="question",
        topic_id="theme:016_program",
    )
    context = {
        "active_brand": "foton",
        "autonomy_policy": {"allow_autonomous": True, "allowed_topic_ids": ["theme:016_program"]},
        "facts_context": {"client_safe": True, "fresh": True, "facts_missing": False},
    }

    guarded = apply_autonomy_matrix_guard(result, client_message="Что входит в курс?", context=context)

    assert guarded.route == "draft_for_manager"
    assert guarded.draft_text == draft_text
    assert "autonomy_matrix_kept_unverified_draft" in guarded.safety_flags


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
