from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional


# Tallanto ``type_client_c`` stores the last completed grade, not the grade the
# student is entering. Keep the source vocabulary closed and deterministic.
_FINISHED_GRADE_BY_STUDENT_TYPE = {f"{grade}_klass": grade for grade in range(1, 11)}
_EXPLICIT_GRADUATE_TYPES = frozenset({"vypusknik"})
_KNOWN_OUT_OF_SCOPE_TYPES = frozenset({"11_klass", "listener"})
_EXACT_STUDENT_TYPE_ALIASES = {
    **{f"{grade} класс": f"{grade}_klass" for grade in range(1, 12)},
    "выпускник": "vypusknik",
    "слушатель": "listener",
}

FAMILY_LINK_SCOPE_CANONICAL_IN = "canonical_in_scope"
FAMILY_LINK_SCOPE_LEGACY_NUMERIC = "legacy_numeric_in_scope"
FAMILY_LINK_SCOPE_LEGACY_NUMERIC_OUT = "legacy_numeric_out_of_scope"
FAMILY_LINK_SCOPE_OUT = "out_of_scope"
FAMILY_LINK_SCOPE_INVALID = "invalid_or_unknown"


def finished_grade_from_student_type(value: Any) -> Optional[int]:
    return _FINISHED_GRADE_BY_STUDENT_TYPE.get(_normalized(value))


def next_grade_from_student_type(value: Any) -> Optional[int]:
    finished = finished_grade_from_student_type(value)
    return finished + 1 if finished is not None else None


def is_explicit_graduate_student_type(value: Any) -> bool:
    return _normalized(value) in _EXPLICIT_GRADUATE_TYPES


def student_type_in_timeline_scope(value: Any) -> bool:
    """The owner selected Tallanto's completed grades 1..10 as current scope."""
    return finished_grade_from_student_type(value) is not None


def family_link_student_type_contract(
    record: Mapping[str, Any],
) -> tuple[bool, bool, bool, set[int]]:
    """Return (has_contract, in_scope, explicit_graduate, target_grades)."""
    contract, _state = _family_link_contract_and_scope(record)
    return contract


def family_link_timeline_scope_state(record: Mapping[str, Any]) -> str:
    """Classify one family link without treating unknown data as an eligible child."""
    _contract, state = _family_link_contract_and_scope(record)
    return state


def family_timeline_scope_decision(states: Iterable[str]) -> str:
    """Aggregate child states; invalid blocks, canonical in-scope outranks terminal siblings."""
    values = tuple(states)
    if not values or FAMILY_LINK_SCOPE_INVALID in values:
        return FAMILY_LINK_SCOPE_INVALID
    if FAMILY_LINK_SCOPE_CANONICAL_IN in values:
        return FAMILY_LINK_SCOPE_CANONICAL_IN
    if FAMILY_LINK_SCOPE_LEGACY_NUMERIC in values:
        return FAMILY_LINK_SCOPE_LEGACY_NUMERIC
    if FAMILY_LINK_SCOPE_OUT in values:
        return FAMILY_LINK_SCOPE_OUT
    if all(value == FAMILY_LINK_SCOPE_LEGACY_NUMERIC_OUT for value in values):
        return FAMILY_LINK_SCOPE_OUT
    return FAMILY_LINK_SCOPE_INVALID


def _family_link_contract_and_scope(
    record: Mapping[str, Any],
) -> tuple[tuple[bool, bool, bool, set[int]], str]:
    current_values = record.get("student_types")
    legacy_values = record.get("grades")
    if isinstance(current_values, list) and any(_normalized(value) for value in current_values):
        values = tuple(_normalized(value) for value in current_values if _normalized(value))
        cache_required = True
    elif isinstance(legacy_values, list):
        values = tuple(_normalized(value) for value in legacy_values if _normalized(value))
        cache_required = False
    else:
        return (False, False, False, set()), FAMILY_LINK_SCOPE_INVALID
    if not values:
        return (False, False, False, set()), FAMILY_LINK_SCOPE_INVALID

    canonical = tuple(value for value in values if value in _FINISHED_GRADE_BY_STUDENT_TYPE)
    graduates = tuple(value for value in values if value in _EXPLICIT_GRADUATE_TYPES)
    known_out = tuple(value for value in values if value in _KNOWN_OUT_OF_SCOPE_TYPES)
    recognized = set(canonical) | set(graduates) | set(known_out)
    unknown = tuple(value for value in values if value not in recognized)
    if unknown:
        if not cache_required and len(values) == 1 and values[0].isdecimal():
            grade = int(values[0])
            if 1 <= grade <= 10:
                return (False, False, False, set()), FAMILY_LINK_SCOPE_LEGACY_NUMERIC
            if grade == 11:
                return (True, False, False, set()), FAMILY_LINK_SCOPE_LEGACY_NUMERIC_OUT
        fail_closed = (True, False, True, set()) if cache_required else (False, False, False, set())
        return fail_closed, FAMILY_LINK_SCOPE_INVALID
    if canonical and (graduates or known_out):
        return (True, False, True, set()), FAMILY_LINK_SCOPE_INVALID

    target_grades = {_FINISHED_GRADE_BY_STUDENT_TYPE[value] + 1 for value in canonical}
    explicit_graduate = bool(graduates)
    in_scope = bool(canonical)
    if cache_required and not _family_link_cache_matches(
        record,
        target_grades=target_grades,
        in_scope=in_scope,
        explicit_graduate=explicit_graduate,
    ):
        return (True, False, True, set()), FAMILY_LINK_SCOPE_INVALID
    if in_scope:
        return (True, True, False, target_grades), FAMILY_LINK_SCOPE_CANONICAL_IN
    return (True, False, explicit_graduate, set()), FAMILY_LINK_SCOPE_OUT


def _family_link_cache_matches(
    record: Mapping[str, Any],
    *,
    target_grades: set[int],
    in_scope: bool,
    explicit_graduate: bool,
) -> bool:
    raw_targets = record.get("target_grades")
    if not isinstance(raw_targets, list) or any(not str(value).strip().isdecimal() for value in raw_targets):
        return False
    persisted_targets = {int(value) for value in raw_targets}
    return (
        persisted_targets == target_grades
        and record.get("timeline_scope_eligible") is in_scope
        and record.get("explicit_graduate") is explicit_graduate
    )


def _normalized(value: Any) -> str:
    normalized = str(value or "").strip().casefold()
    return _EXACT_STUDENT_TYPE_ALIASES.get(normalized, normalized)


__all__ = [
    "FAMILY_LINK_SCOPE_CANONICAL_IN",
    "FAMILY_LINK_SCOPE_INVALID",
    "FAMILY_LINK_SCOPE_LEGACY_NUMERIC",
    "FAMILY_LINK_SCOPE_LEGACY_NUMERIC_OUT",
    "FAMILY_LINK_SCOPE_OUT",
    "family_link_timeline_scope_state",
    "finished_grade_from_student_type",
    "family_link_student_type_contract",
    "family_timeline_scope_decision",
    "is_explicit_graduate_student_type",
    "next_grade_from_student_type",
    "student_type_in_timeline_scope",
]
