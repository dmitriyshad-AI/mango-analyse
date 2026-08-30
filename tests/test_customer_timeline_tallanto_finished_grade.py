from __future__ import annotations

import pytest

from mango_mvp.customer_timeline.tallanto_finished_grade import (
    FAMILY_LINK_SCOPE_CANONICAL_IN,
    FAMILY_LINK_SCOPE_INVALID,
    FAMILY_LINK_SCOPE_LEGACY_NUMERIC,
    FAMILY_LINK_SCOPE_OUT,
    family_link_student_type_contract,
    family_link_timeline_scope_state,
    family_timeline_scope_decision,
    finished_grade_from_student_type,
    is_explicit_graduate_student_type,
    next_grade_from_student_type,
    student_type_in_timeline_scope,
)


@pytest.mark.parametrize(
    ("raw", "finished", "next_grade", "in_scope", "graduate"),
    (
        ("8_klass", 8, 9, True, False),
        ("8 класс", 8, 9, True, False),
        ("10_klass", 10, 11, True, False),
        ("10 класс", 10, 11, True, False),
        ("vypusknik", None, None, False, True),
        ("Выпускник", None, None, False, True),
        ("11_klass", None, None, False, False),
        ("11 класс", None, None, False, False),
        ("Listener", None, None, False, False),
        ("Слушатель", None, None, False, False),
        (None, None, None, False, False),
    ),
)
def test_tallanto_completed_grade_contract(raw, finished, next_grade, in_scope, graduate) -> None:
    assert finished_grade_from_student_type(raw) == finished
    assert next_grade_from_student_type(raw) == next_grade
    assert student_type_in_timeline_scope(raw) is in_scope
    assert is_explicit_graduate_student_type(raw) is graduate


def test_family_link_contract_reads_legacy_source_value_and_validates_current_cache() -> None:
    assert family_link_student_type_contract({"grades": ["8_klass"]}) == (
        True, True, False, {9},
    )
    assert family_link_student_type_contract({"grades": ["vypusknik"]}) == (
        True, False, True, set(),
    )
    assert family_link_student_type_contract({
        "student_types": ["8_klass"],
        "target_grades": [8],
        "timeline_scope_eligible": True,
        "explicit_graduate": False,
    }) == (True, False, True, set())
    assert family_link_timeline_scope_state({"student_types": [], "grades": ["8"]}) == (
        FAMILY_LINK_SCOPE_LEGACY_NUMERIC
    )


def test_family_scope_uses_tri_state_and_never_treats_unknown_as_candidate() -> None:
    graduate = family_link_timeline_scope_state({"grades": ["vypusknik"]})
    listener = family_link_timeline_scope_state({"grades": ["Listener"]})
    legacy_eleven = family_link_timeline_scope_state({"grades": ["11_klass"]})
    finished_eight = family_link_timeline_scope_state({"grades": ["8_klass"]})
    numeric_eight = family_link_timeline_scope_state({"grades": ["8"]})
    arbitrary_unknown = family_link_timeline_scope_state({"grades": ["something_else"]})

    assert (graduate, listener, legacy_eleven) == (FAMILY_LINK_SCOPE_OUT,) * 3
    assert finished_eight == FAMILY_LINK_SCOPE_CANONICAL_IN
    assert numeric_eight == FAMILY_LINK_SCOPE_LEGACY_NUMERIC
    assert arbitrary_unknown == FAMILY_LINK_SCOPE_INVALID
    assert family_timeline_scope_decision((graduate, listener)) == FAMILY_LINK_SCOPE_OUT
    assert family_timeline_scope_decision((graduate, legacy_eleven)) == FAMILY_LINK_SCOPE_OUT
    assert family_timeline_scope_decision((graduate, finished_eight)) == FAMILY_LINK_SCOPE_CANONICAL_IN
    assert family_timeline_scope_decision((graduate, numeric_eight)) == FAMILY_LINK_SCOPE_LEGACY_NUMERIC
    assert family_timeline_scope_decision((finished_eight, arbitrary_unknown)) == FAMILY_LINK_SCOPE_INVALID
