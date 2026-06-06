from __future__ import annotations

"""Recall-first confirmed fact selection.

This layer promotes facts that answer the requested semantic key while still
blocking facts from explicitly forbidden neighbouring scopes.
"""

from typing import Mapping, Sequence


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
    k: int = 10,
) -> list[Mapping[str, object]]:
    required = [str(item) for item in required_fact_keys if str(item or "").strip()]
    topics = {str(item) for item in active_topics if str(item or "").strip()}
    blocked = {str(item) for item in blocked_scopes if str(item or "").strip()}
    brand = str(active_brand or "").strip()

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
    result = [fact for _, _, fact in answer_facts] + [fact for _, fact in rest]
    limit = max(k, len(answer_facts))
    return result[:limit]
