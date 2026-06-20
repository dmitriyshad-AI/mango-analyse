from __future__ import annotations

"""Recall-first confirmed fact selection.

This layer promotes facts that answer the requested semantic key while still
blocking facts from explicitly forbidden neighbouring scopes.
"""

from typing import Mapping, Sequence

from mango_mvp.knowledge_base.price_axes_catalog import (
    price_axes_selector_enabled,
    select_price_fact_for_query,
)


FACT_RETRIEVAL_SCHEMA_VERSION = "fact_retrieval_v1_2026_05_25"

KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "prices": ("price", "prices", "tuition", "year", "semester", "cost", "стоим"),
    "discounts": ("discount", "discounts", "payment_options"),
    "discounts_year": ("year.discount", "year_discount", "discount_extra", "available_schedules", "год"),
    "discounts_semester": ("semester.discount", "semester_discount", "discount_extra", "available_schedules", "семестр"),
    "installment_terms": ("installment", "dolyami", "payment_options", "rassroch"),
    "trial_class": ("trial", "fragment", "фрагмент", "probn"),
    "trial_online_fragment": ("trial", "fragment", "фрагмент", "online_trial"),
    "programs": ("program", "camp", "ls_city", "lvsh", "senior_school", "direction", "subject", "olympiad"),
    "availability": ("availability", "seats", "places", "mest"),
    "schedule": ("schedule", "weekly_lessons", "available_schedules", "days", "lesson"),
    "schedule_weekend": ("objection_responses.inconvenient_time", "weekend_slots", "выходн", "слоты"),
    "online_recordings": ("online_recording", "online_recordings", "online_platform.recording", "recording", "запис", "сохран"),
    "offline_recordings": ("offline_recording", "offline_recordings", "recording", "materials"),
    "recordings": ("recording", "online_platform.recording", "materials", "запис", "сохран"),
    "teachers": ("teacher", "teachers", "theme_17_teachers", "преподав", "педагог", "учитель"),
    "olympiad_online": (
        "olympiad_online",
        "online_olympiad",
        "olympiad_phystech",
        "phystech",
        "физтех",
        "физтех онлайн",
        "9 и 11",
    ),
    "formats": ("format", "online", "offline", "ochno"),
    "locations": ("location", "address", "addresses"),
    "documents": ("document", "spravka", "certificate", "cert"),
    "platform": ("online_platform", "student_account_access", "platform", "личный кабинет", "платформ", "мтс линк"),
    "platform_documents": ("electronic_document_flow", "electronic_documents", "электронный документооборот", "скан-коп"),
    "refund_policy": ("refund_policy", "refund", "returns", "возврат"),
    "matkap_documents": ("matkap", "sfr", "materin"),
    "matkap_timeline": ("matkap.timeline", "sfr_review", "transfer_days", "total_max_days", "маткап", "сфр"),
    "tax_deduction_procedure": ("tax", "deduction", "vychet", "ndfl", "fns"),
}


def key_matches(required_key: str, fact_key: str) -> bool:
    head = str(required_key or "").split(".")[0].strip().casefold()
    if not head:
        return False
    value = str(fact_key or "").casefold()
    aliases = KEY_ALIASES.get(head, (head,))
    return any(alias in value for alias in aliases)


def select_confirmed_facts(
    candidates: Sequence[Mapping[str, object]],
    *,
    active_brand: str = "",
    required_fact_keys: Sequence[str] = (),
    active_topics: Sequence[str] = (),
    blocked_scopes: Sequence[str] = (),
    query: str = "",
    k: int = 10,
) -> list[Mapping[str, object]]:
    required = [str(item) for item in required_fact_keys if str(item or "").strip()]
    topics = {str(item) for item in active_topics if str(item or "").strip()}
    blocked = {str(item) for item in blocked_scopes if str(item or "").strip()}
    brand = str(active_brand or "").strip()

    virtual_price_facts: list[Mapping[str, object]] = []
    if query and price_axes_selector_enabled() and any(str(key or "").split(".", 1)[0] == "prices" for key in required):
        raw_facts = [fact.get("__fact") if isinstance(fact.get("__fact"), Mapping) else fact for fact in candidates]
        selected_price_fact = select_price_fact_for_query(raw_facts, active_brand=brand, query=query)
        if selected_price_fact:
            virtual_price_facts.append(
                {
                    "__fact": selected_price_fact,
                    "__score": 10_000,
                    "__index": -1,
                    "brand": str(selected_price_fact.get("brand") or "").strip(),
                    "fact_key": str(selected_price_fact.get("fact_key") or "").strip(),
                    "scopes": set(),
                }
            )

    answer_facts: list[tuple[int, int, Mapping[str, object]]] = []
    rest: list[tuple[int, Mapping[str, object]]] = []

    for fact in candidates:
        fact_brand = str(fact.get("brand") or "").strip()
        if brand and fact_brand and fact_brand != brand:
            continue
        scopes = {str(scope) for scope in (fact.get("scopes") or []) if str(scope).strip()}
        if scopes & blocked:
            continue
        fact_key = str(fact.get("fact_key") or "")
        matched_indexes = [idx for idx, required_key in enumerate(required) if key_matches(required_key, fact_key)]
        if matched_indexes:
            score = int(fact.get("__score") or 0) if isinstance(fact, Mapping) else 0
            answer_facts.append((min(matched_indexes), score, fact))
            continue
        score = 0
        if scopes & topics:
            score += 10
        rest.append((score, fact))

    answer_facts.sort(key=lambda item: (item[0], -item[1]))
    rest.sort(key=lambda item: -item[0])
    result = [*virtual_price_facts, *[fact for _, _, fact in answer_facts], *[fact for _, fact in rest]]
    if virtual_price_facts:
        seen_ids: set[str] = set()
        deduped: list[Mapping[str, object]] = []
        for fact in result:
            raw_fact = fact.get("__fact") if isinstance(fact.get("__fact"), Mapping) else fact
            fact_id = str(raw_fact.get("fact_id") or raw_fact.get("id") or raw_fact.get("fact_key") or "")
            if fact_id and fact_id in seen_ids:
                continue
            if fact_id:
                seen_ids.add(fact_id)
            deduped.append(fact)
        result = deduped
    limit = max(k, len(answer_facts))
    return result[:limit]
