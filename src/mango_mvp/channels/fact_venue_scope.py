from __future__ import annotations

import os
from typing import Any, Mapping, Optional, Sequence


FACT_VENUE_SCOPE_ENV = "TELEGRAM_FACT_VENUE_SCOPE"

VENUE_SCOPE_VALUES = frozenset({"moscow_regular", "dolgoprudny", "lvsh_mendeleevo", "online"})
VENUE_SCOPE_ANY = "any"
VENUE_SCOPE_UNSPECIFIED = "unspecified"
PROGRAM_KIND_VALUES = frozenset({"regular", "camp_city", "camp_lvsh", "olympiad"})

VENUE_SCOPE_LABELS: Mapping[str, str] = {
    "moscow_regular": "Москва/регулярные очные занятия",
    "dolgoprudny": "Долгопрудный",
    "lvsh_mendeleevo": "ЛВШ Менделеево",
    "online": "онлайн",
    "any": "любая площадка",
    "unspecified": "не уточнено",
}

_TRUTHY_VALUES = {"1", "true", "yes", "y", "on", "да", "вкл"}


def _explicit_venue_scope_setting(context: Optional[Mapping[str, Any]] = None) -> Optional[bool]:
    if isinstance(context, Mapping):
        for key in (FACT_VENUE_SCOPE_ENV, "fact_venue_scope", "fact_venue_scope_enabled"):
            if key in context:
                return str(context.get(key) or "").strip().casefold() in _TRUTHY_VALUES
    if FACT_VENUE_SCOPE_ENV in os.environ:
        return str(os.getenv(FACT_VENUE_SCOPE_ENV) or "").strip().casefold() in _TRUTHY_VALUES
    return None


def venue_scope_enabled(context: Optional[Mapping[str, Any]] = None) -> bool:
    explicit = _explicit_venue_scope_setting(context)
    if explicit is not None:
        return explicit
    from mango_mvp.channels.subscription_llm_parts.support import _pilot_profile_default_on_flag_enabled

    return _pilot_profile_default_on_flag_enabled(
        context,
        FACT_VENUE_SCOPE_ENV,
        aliases=("fact_venue_scope", "fact_venue_scope_enabled"),
    )


def normalize_requested_scope(value: Any) -> str:
    text = str(value or "").strip().casefold().replace("-", "_").replace(" ", "_")
    aliases = {
        "moscow": "moscow_regular",
        "москва": "moscow_regular",
        "regular_moscow": "moscow_regular",
        "regular_offline": "moscow_regular",
        "dolgoprudnyi": "dolgoprudny",
        "долгопрудный": "dolgoprudny",
        "lvsh": "lvsh_mendeleevo",
        "лвш": "lvsh_mendeleevo",
        "mendeleevo": "lvsh_mendeleevo",
        "менделеево": "lvsh_mendeleevo",
        "онлайн": "online",
        "not_specified": "unspecified",
        "unknown": "unspecified",
        "none": "unspecified",
        "": "unspecified",
    }
    text = aliases.get(text, text)
    if text in VENUE_SCOPE_VALUES:
        return text
    if text in {VENUE_SCOPE_ANY, VENUE_SCOPE_UNSPECIFIED}:
        return text
    return VENUE_SCOPE_UNSPECIFIED


def normalize_fact_venue(value: Any) -> str:
    text = str(value or "").strip().casefold().replace("-", "_").replace(" ", "_")
    if text in VENUE_SCOPE_VALUES:
        return text
    if text == VENUE_SCOPE_ANY:
        return text
    return VENUE_SCOPE_ANY


def normalize_program_kind(value: Any) -> str:
    text = str(value or "").strip().casefold().replace("-", "_").replace(" ", "_")
    if text in PROGRAM_KIND_VALUES:
        return text
    if text == VENUE_SCOPE_ANY:
        return text
    return VENUE_SCOPE_ANY


def fact_venue(fact: Mapping[str, Any]) -> str:
    value = fact.get("venue")
    if not value and isinstance(fact.get("metadata"), Mapping):
        value = fact["metadata"].get("venue")  # type: ignore[index]
    return normalize_fact_venue(value)


def fact_program_kind(fact: Mapping[str, Any]) -> str:
    value = fact.get("program_kind")
    if not value and isinstance(fact.get("metadata"), Mapping):
        value = fact["metadata"].get("program_kind")  # type: ignore[index]
    return normalize_program_kind(value)


def venue_scope_specific(value: Any) -> bool:
    return normalize_requested_scope(value) in VENUE_SCOPE_VALUES


def fact_foreign_to_requested(fact: Mapping[str, Any], requested_scope: str) -> bool:
    requested = normalize_requested_scope(requested_scope)
    if requested not in VENUE_SCOPE_VALUES:
        return False
    venue = fact_venue(fact)
    return venue not in {VENUE_SCOPE_ANY, requested}


def has_requested_venue_fact(facts: Sequence[Mapping[str, Any]], requested_scope: str) -> bool:
    requested = normalize_requested_scope(requested_scope)
    if requested not in VENUE_SCOPE_VALUES:
        return False
    return any(fact_venue(fact) == requested for fact in facts)


def venue_scope_label(value: Any) -> str:
    return VENUE_SCOPE_LABELS.get(normalize_requested_scope(value), str(value or "").strip())
