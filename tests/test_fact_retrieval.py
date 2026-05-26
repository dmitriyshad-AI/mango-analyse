from __future__ import annotations

from mango_mvp.channels.fact_retrieval import key_matches, select_confirmed_facts
from mango_mvp.channels.held_state import HeldState, update_held
from mango_mvp.channels.semantic_roles import tag_message_roles


UNPK = [
    {"fact_key": "locations_unpk.addresses.1.address", "brand": "unpk", "scopes": set(), "text": "УНПК: Сретенка, 20."},
    {"fact_key": "locations_unpk.addresses.0.address", "brand": "unpk", "scopes": set(), "text": "УНПК: Долгопрудный, Институтский 9."},
    {"fact_key": "payment_options.available_schedules.3.year.discount_extra", "brand": "unpk", "scopes": set(), "text": "УНПК: за год скидка 14%."},
    {"fact_key": "prices_regular_2026_27.online_olympiad_phystech_9_and_11.classes", "brand": "unpk", "scopes": {"olympiad_online"}, "text": "Физтех онлайн указан для 9 и 11 классов."},
    {"fact_key": "objection_responses.inconvenient_time.1", "brand": "unpk", "scopes": set(), "text": "Разные слоты по выходным."},
    {"fact_key": "discounts_multichild_condition_client_text", "brand": "unpk", "scopes": {"discount_multichild"}, "text": "многодетным по удостоверению."},
    {"fact_key": "discounts_stacking_rule", "brand": "unpk", "scopes": {"discount_stacking"}, "text": "скидки не суммируются."},
    {"fact_key": "matkap.sfr_review_timing", "brand": "unpk", "scopes": {"matkap_process"}, "text": "СФР рассматривает до 10 рабочих дней + до 5."},
    *[{"fact_key": f"noise.fact_{i}", "brand": "unpk", "scopes": set(), "text": f"шум {i}"} for i in range(15)],
]

FOTON_CAMP = [
    {"fact_key": "ls_city_2026_foton.schedule", "brand": "foton", "scopes": {"city_day_camp"}, "text": "городская летняя школа, без проживания, пн-пт."},
    {"fact_key": "lvsh_mendeleevo_foton.lodging", "brand": "foton", "scopes": {"residential_lvsh"}, "text": "выездная ЛВШ Менделеево, проживание, 5-раз питание."},
]


def _fact_key_in(result: list[object], needle: str) -> bool:
    return any(needle in str(item.get("fact_key") or "") for item in result if isinstance(item, dict))


def test_address_recall_with_scopeless_fact() -> None:
    result = select_confirmed_facts(UNPK, active_brand="unpk", required_fact_keys=["locations.current"], k=10)

    assert _fact_key_in(result, "locations_unpk.addresses.1.address")


def test_discount_alias_matches_payment_options_year_discount() -> None:
    assert key_matches("discounts.current", "payment_options.available_schedules.3.year.discount_extra")
    assert key_matches("discounts_year.current", "payment_options.available_schedules.3.year.discount_extra")

    result = select_confirmed_facts(UNPK, active_brand="unpk", required_fact_keys=["discounts_year.current", "discounts.current"], k=10)

    assert _fact_key_in(result, "year.discount_extra")


def test_answer_fact_survives_cap() -> None:
    result = select_confirmed_facts(UNPK, active_brand="unpk", required_fact_keys=["discounts_year.current", "discounts.current"], k=10)

    assert _fact_key_in(result, "year.discount_extra")


def test_specific_discount_key_prioritizes_year_answer_before_broad_discount_noise() -> None:
    result = select_confirmed_facts(UNPK, active_brand="unpk", required_fact_keys=["discounts_year.current", "discounts.current"], k=10)

    assert "year.discount_extra" in str(result[0].get("fact_key"))


def test_olympiad_online_alias_recalls_phystech_classes() -> None:
    result = select_confirmed_facts(
        UNPK,
        active_brand="unpk",
        required_fact_keys=["olympiad_online.current", "programs.current"],
        active_topics=["olympiad_online"],
        blocked_scopes=["regular_online"],
        k=10,
    )

    assert _fact_key_in(result, "online_olympiad_phystech_9_and_11.classes")


def test_schedule_weekend_alias_recalls_inconvenient_time_fact() -> None:
    result = select_confirmed_facts(
        UNPK,
        active_brand="unpk",
        required_fact_keys=["schedule_weekend.current", "schedule.current"],
        active_topics=["schedule"],
        k=10,
    )

    assert _fact_key_in(result, "objection_responses.inconvenient_time.1")


def test_no_cross_brand_fact() -> None:
    mixed = [*UNPK, {"fact_key": "foton.price", "brand": "foton", "scopes": set(), "text": "Фотон цена"}]
    result = select_confirmed_facts(mixed, active_brand="unpk", required_fact_keys=["prices.current"])

    assert not any(item.get("brand") == "foton" for item in result)


def test_foreign_scope_blocked_but_city_fact_kept() -> None:
    result = select_confirmed_facts(
        FOTON_CAMP,
        active_brand="foton",
        required_fact_keys=["programs.current"],
        active_topics=["camp"],
        blocked_scopes=["residential_lvsh"],
    )

    assert not _fact_key_in(result, "lvsh_mendeleevo")
    assert _fact_key_in(result, "ls_city_2026_foton")


def test_followup_keeps_retrieval_keys_in_held_state() -> None:
    held = HeldState()
    roles0 = tag_message_roles("если оплатить за год, скидка будет?")
    held = update_held(
        held,
        "если оплатить за год, скидка будет?",
        roles0,
        p0_required=False,
        required_fact_keys=("discounts.current",),
    )
    roles1 = tag_message_roles("а за семестр?")
    held = update_held(held, "а за семестр?", roles1, p0_required=False, required_fact_keys=())

    retrieval = held.retrieval_context()
    assert "discounts.current" in retrieval["required_fact_keys"]
    result = select_confirmed_facts(UNPK, active_brand="unpk", required_fact_keys=retrieval["required_fact_keys"])
    assert _fact_key_in(result, "year.discount_extra")
