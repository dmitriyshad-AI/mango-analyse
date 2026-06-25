from __future__ import annotations

from mango_mvp.channels.subscription_llm_parts.contracts import SubscriptionDraftResult
from mango_mvp.channels.subscription_llm_parts.direct_path import _build_direct_path_prompt
from mango_mvp.channels.subscription_llm_parts.policy_routing import apply_autonomy_matrix_guard
from mango_mvp.channels.subscription_llm_parts.reliable_answerer import (
    RELIABLE_ANSWERER_STEP1_ENV,
    apply_reliable_answerer_output_guard,
    build_answer_coverage_plan,
    reliable_answerer_step1_enabled,
)
from mango_mvp.channels.subscription_llm_parts.support import (
    DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS,
    DIRECT_PATH_PILOT_CONFIG_ENV,
)


def _price_fact_pack(*, venue: str = "online") -> dict:
    return {
        "facts": {
            "price.foton.online.semester": "Фотон: онлайн-курс стоит 37 000 рублей за семестр.",
        },
        "exact_keys": ["price.foton.online.semester"],
        "adjacent_keys": [],
        "selected_category": "llm_retrieve",
        "fact_metadata": {
            "price.foton.online.semester": {
                "brand": "foton",
                "fact_type": "price",
                "product": "regular",
                "venue": venue,
                "program_kind": "regular",
            }
        },
        "llm_retrieve": {"venue_scope": {"requested_scope": "online"}},
    }


def test_reliable_answerer_step1_default_off_and_not_in_pilot_profile(monkeypatch) -> None:
    monkeypatch.delenv(RELIABLE_ANSWERER_STEP1_ENV, raising=False)

    assert reliable_answerer_step1_enabled({DIRECT_PATH_PILOT_CONFIG_ENV: "pilot_gold_v1"}) is False
    assert RELIABLE_ANSWERER_STEP1_ENV not in DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS


def test_reliable_answerer_prompt_block_only_when_flag_enabled() -> None:
    fact_pack = _price_fact_pack()
    off_prompt = _build_direct_path_prompt(
        "Сколько стоит онлайн?",
        context={"active_brand": "foton"},
        fact_pack=fact_pack,
    )
    on_prompt = _build_direct_path_prompt(
        "Сколько стоит онлайн?",
        context={"active_brand": "foton", RELIABLE_ANSWERER_STEP1_ENV: "1"},
        fact_pack=fact_pack,
    )

    assert "Надёжный ответчик" not in off_prompt
    assert "Надёжный ответчик" in on_prompt
    assert "не сдавай весь ответ" in on_prompt


def test_venue_sensitive_facet_without_venue_is_not_covered() -> None:
    plan = build_answer_coverage_plan(
        "Сколько стоит очно?",
        fact_pack=_price_fact_pack(venue="any"),
        context={"active_brand": "foton", RELIABLE_ANSWERER_STEP1_ENV: "1"},
    )

    assert "price" not in {item["facet"] for item in plan["covered_facets"]}
    assert "price" in {item["facet"] for item in plan["blocked_facets"]}


def test_general_platform_fact_can_cover_without_specific_venue() -> None:
    pack = {
        "facts": {"platform.foton": "Фотон: онлайн-занятия проходят на платформе МТС Линк."},
        "exact_keys": ["platform.foton"],
        "adjacent_keys": [],
        "fact_metadata": {
            "platform.foton": {"brand": "foton", "fact_type": "platform", "product": "regular", "venue": "any"}
        },
    }

    plan = build_answer_coverage_plan(
        "На какой платформе онлайн?",
        fact_pack=pack,
        context={"active_brand": "foton", RELIABLE_ANSWERER_STEP1_ENV: "1"},
    )

    assert "platform" in {item["facet"] for item in plan["covered_facets"]}


def test_availability_promise_is_blocked_by_deterministic_guard() -> None:
    plan = build_answer_coverage_plan(
        "Есть места?",
        fact_pack=_price_fact_pack(),
        context={"active_brand": "foton", RELIABLE_ANSWERER_STEP1_ENV: "1"},
    )
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="Да, места есть, запишем вас в группу.",
        metadata={"answer_coverage_plan": plan},
    )

    guarded = apply_reliable_answerer_output_guard(
        result,
        client_message="Есть места?",
        context={"active_brand": "foton", RELIABLE_ANSWERER_STEP1_ENV: "1"},
    )

    assert guarded.route == "draft_for_manager"
    assert "reliable_answerer_availability_promise_blocked" in guarded.safety_flags
    assert "Наличие места" in guarded.draft_text


def test_live_status_guard_preserves_partial_answer_as_manager_draft() -> None:
    plan = build_answer_coverage_plan(
        "Сколько стоит и есть ли места?",
        fact_pack=_price_fact_pack(),
        context={"active_brand": "foton", RELIABLE_ANSWERER_STEP1_ENV: "1"},
    )
    base = SubscriptionDraftResult(
        route="draft_for_manager",
        draft_text="Онлайн-курс стоит 37 000 рублей за семестр.",
        message_type="question",
        topic_id="theme:001_pricing",
        missing_facts=("availability_by_group_or_shift",),
        metadata={"answer_coverage_plan": plan},
    )

    guarded = apply_autonomy_matrix_guard(
        base,
        client_message="Сколько стоит и есть ли места?",
        context={
            "active_brand": "foton",
            RELIABLE_ANSWERER_STEP1_ENV: "1",
            "autonomy_policy": {"allow_autonomous": True, "allowed_topic_ids": ["theme:001_pricing"]},
            "facts_context": {"client_safe": True, "fresh": True, "facts_missing": False},
            "confirmed_facts": {"price.foton.online.semester": "Фотон: онлайн-курс стоит 37 000 рублей за семестр."},
        },
    )

    assert guarded.route == "draft_for_manager"
    assert "37 000" in guarded.draft_text
    assert "до live-проверки" in guarded.draft_text
    assert "reliable_answerer_live_status_partial_preserved" in guarded.safety_flags


def test_live_status_guard_without_covered_facet_keeps_safe_handoff() -> None:
    plan = build_answer_coverage_plan(
        "Есть места?",
        fact_pack=_price_fact_pack(),
        context={"active_brand": "foton", RELIABLE_ANSWERER_STEP1_ENV: "1"},
    )
    base = SubscriptionDraftResult(
        route="draft_for_manager",
        draft_text="Передам менеджеру, он проверит наличие мест.",
        message_type="question",
        topic_id="theme:026_camp_general",
        missing_facts=("availability_by_group_or_shift",),
        metadata={"answer_coverage_plan": plan},
    )

    guarded = apply_autonomy_matrix_guard(
        base,
        client_message="Есть места?",
        context={
            "active_brand": "foton",
            RELIABLE_ANSWERER_STEP1_ENV: "1",
            "autonomy_policy": {"allow_autonomous": True, "allowed_topic_ids": ["theme:026_camp_general"]},
            "facts_context": {"client_safe": True, "fresh": True, "facts_missing": False},
            "confirmed_facts": {"price.foton.online.semester": "Фотон: онлайн-курс стоит 37 000 рублей за семестр."},
        },
    )

    assert guarded.route == "draft_for_manager"
    assert "не буду обещать без проверки" in guarded.draft_text
    assert "reliable_answerer_live_status_partial_preserved" not in guarded.safety_flags
