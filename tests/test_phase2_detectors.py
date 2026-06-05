from __future__ import annotations

from mango_mvp.insights.phase2_detectors import (
    NEGATIVE_EXAMPLES,
    POSITIVE_ANXIETY_EXAMPLES,
    POSITIVE_OBJECTION_EXAMPLES,
    detect_anxiety,
    detect_objection,
)


def test_phase2_objection_detector_matches_positive_examples() -> None:
    for text, expected in POSITIVE_OBJECTION_EXAMPLES:
        assert detect_objection(text) == expected, text


def test_phase2_anxiety_detector_matches_positive_examples() -> None:
    for text, expected in POSITIVE_ANXIETY_EXAMPLES:
        assert detect_anxiety(text) == expected, text


def test_phase2_detectors_keep_neutral_info_questions_clean() -> None:
    for text in NEGATIVE_EXAMPLES:
        assert detect_objection(text) is None, text
        assert detect_anxiety(text) is None, text
