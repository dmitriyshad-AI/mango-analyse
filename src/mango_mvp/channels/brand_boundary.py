"""Single structured owner of client-visible brand separation markers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class BrandBoundaryRule:
    category: str
    marker: str
    aliases: tuple[str, ...] = ()
    word_prefixes: tuple[str, ...] = ()
    exception: str = ""


BRAND_BOUNDARY_RULES: Mapping[str, tuple[BrandBoundaryRule, ...]] = {
    "foton": (
        BrandBoundaryRule("brand", "унпк", ("унпк", "унпк мфти", "унфт", "unpk")),
        BrandBoundaryRule("legal_name", "ано дпо", ("ано дпо", "ноу унпк")),
        BrandBoundaryRule("domain_or_account", "kmipt", ("kmipt ru", "unpk mipt")),
        BrandBoundaryRule("exclusive_address", "сретенка", word_prefixes=("сретенк",)),
        BrandBoundaryRule("exclusive_address", "институтский", word_prefixes=("институтск",)),
        BrandBoundaryRule("exclusive_address", "пацаева", word_prefixes=("пацаев",)),
        BrandBoundaryRule("contextual_entity", "мфти", ("мфти",), exception="foton_academic_fact"),
        BrandBoundaryRule("contextual_entity", "mipt", ("mipt",), exception="foton_academic_fact"),
    ),
    "unpk": (
        BrandBoundaryRule(
            "brand", "фотон", ("фотон", "фотона", "фотоне", "фотоном", "фотону", "foton"),
            exception="physical_photon",
        ),
        BrandBoundaryRule("legal_name", "цдпо", ("цдпо", "црдо", "цдпо фотон", "црдо фотон")),
        BrandBoundaryRule("domain_or_account", "cdpofoton", ("cdpofoton", "cdpofoton ru", "foton edu")),
        BrandBoundaryRule("commercial_marker", "долями", ("долями",)),
        BrandBoundaryRule(
            "commercial_marker", "т-банк",
            ("тбанк", "т банк", "tbank", "t bank", "tбанк", "тbank", "t банк", "т bank", "tinkoff", "тинькофф"),
        ),
    ),
}


# Compatibility surface for dynamic reexports. New decisions use the structured rules above.
LEGACY_FOREIGN_TOKENS: Mapping[str, tuple[str, ...]] = {
    "foton": ("унпк", "унпк мфти", "мфти", "kmipt", "@unpk", "ноу унпк", "ано дпо"),
    "unpk": ("фотон", "цдпо", "црдо", "cdpofoton", "foton", "долями", "т-банк"),
}


_ACADEMIC_ROLE_PREFIXES = ("преподавател", "вожат", "выпускник")
_PHOTON_WORDS = frozenset(("фотон", "фотона", "фотоне", "фотоном", "фотону"))


def foreign_brand_rule(
    text: str,
    *,
    active_brand: str,
    facts: Mapping[str, str],
) -> BrandBoundaryRule | None:
    source = str(text or "")
    words = _words(source)
    rules = BRAND_BOUNDARY_RULES.get(str(active_brand or "").strip().casefold(), ())
    for rule in rules:
        if not _rule_matches(words, rule):
            continue
        if rule.exception == "physical_photon" and _physical_photon(words):
            continue
        if rule.exception == "foton_academic_fact" and _academic_mfti_is_fact_backed(words, facts):
            continue
        return rule
    return None


def _words(text: str) -> tuple[str, ...]:
    normalized = str(text or "").casefold().replace("ё", "е")
    return tuple("".join(char if char.isalnum() else " " for char in normalized).split())


def _rule_matches(words: tuple[str, ...], rule: BrandBoundaryRule) -> bool:
    padded = f" {' '.join(words)} "
    if any(f" {' '.join(_words(alias))} " in padded for alias in rule.aliases):
        return True
    return any(word.startswith(prefix) for word in words for prefix in rule.word_prefixes)


def _academic_mfti_is_fact_backed(words: tuple[str, ...], facts: Mapping[str, str]) -> bool:
    if not _academic_mfti_context(words):
        return False
    return any(_academic_mfti_context(_words(str(value or ""))) for value in facts.values())


def _academic_mfti_context(words: tuple[str, ...]) -> bool:
    indices = [index for index, word in enumerate(words) if word in {"мфти", "mipt"}]
    return any(
        any(role.startswith(prefix) for role in words[max(0, index - 12) : index + 13] for prefix in _ACADEMIC_ROLE_PREFIXES)
        for index in indices
    )


def _physical_photon(words: tuple[str, ...]) -> bool:
    photon_indices = [index for index, word in enumerate(words) if word in _PHOTON_WORDS]
    return bool(photon_indices) and all(
        any(word.startswith(("квант", "частиц")) for word in words[max(0, index - 3) : index + 4])
        for index in photon_indices
    )


__all__ = (
    "BRAND_BOUNDARY_RULES",
    "BrandBoundaryRule",
    "LEGACY_FOREIGN_TOKENS",
    "foreign_brand_rule",
)
