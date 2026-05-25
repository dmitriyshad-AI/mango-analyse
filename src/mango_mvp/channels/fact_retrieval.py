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
    "installment_terms": ("installment", "dolyami", "payment_options", "rassroch"),
    "trial_class": ("trial", "fragment", "фрагмент", "probn"),
    "programs": ("program", "camp", "ls_city", "lvsh", "senior_school", "direction", "subject", "olympiad"),
    "availability": ("availability", "seats", "places", "mest"),
    "schedule": ("schedule", "weekly_lessons", "available_schedules", "days", "lesson"),
    "formats": ("format", "online", "offline", "ochno"),
    "locations": ("location", "address", "addresses"),
    "documents": ("document", "spravka", "certificate", "cert"),
    "matkap_documents": ("matkap", "sfr", "materin"),
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

    answer_facts: list[Mapping[str, object]] = []
    rest: list[tuple[int, Mapping[str, object]]] = []

    for fact in candidates:
        fact_brand = str(fact.get("brand") or "").strip()
        if brand and fact_brand and fact_brand != brand:
            continue
        scopes = {str(scope) for scope in (fact.get("scopes") or []) if str(scope).strip()}
        if scopes & blocked:
            continue
        fact_key = str(fact.get("fact_key") or "")
        if any(key_matches(required_key, fact_key) for required_key in required):
            answer_facts.append(fact)
            continue
        score = 0
        if scopes & topics:
            score += 10
        rest.append((score, fact))

    rest.sort(key=lambda item: -item[0])
    result = [*answer_facts, *(fact for _, fact in rest)]
    limit = max(k, len(answer_facts))
    return result[:limit]
