from __future__ import annotations

from mango_mvp.channels.dialogue_memory import MEMORY_PROVENANCE_ENV, build_dialogue_memory
from mango_mvp.channels.new_lead_funnel import (
    ANCHORED_BARE_GRADE_ENV,
    build_lead_funnel_state,
    extract_format,
    extract_grade,
)
from scripts.run_tz124_slot_anchor_pack import main as run_tz124_pack


def _memory_view(text: str, monkeypatch, *, flag: str = "1", brand: str = "unpk") -> dict:
    monkeypatch.setenv(MEMORY_PROVENANCE_ENV, "1")
    monkeypatch.setenv(ANCHORED_BARE_GRADE_ENV, flag)
    return dict(
        build_dialogue_memory(
            current_message=text,
            active_brand=brand,
            session_id=f"tz124:{brand}:{text}",
        ).to_prompt_view()
    )


def test_tz124_off_flag_keeps_main_behavior_for_bare_grade_and_format_choice(monkeypatch) -> None:
    view = _memory_view("физика 8 онлайн", monkeypatch, flag="0")

    assert "grade" not in view["known_slots"]
    assert extract_grade("физика 8 онлайн") == ""

    monkeypatch.setenv(ANCHORED_BARE_GRADE_ENV, "0")
    assert extract_format("онлайн и очно сравниваю") == "offline"


def test_tz124_p1_bare_grade_with_subject_and_format_goes_through_provenance(monkeypatch) -> None:
    view = _memory_view("физика 8 онлайн", monkeypatch, flag="1")

    assert view["known_slots"]["grade"] == "8"
    assert view["known_slots"]["subject"] == "физика"
    assert view["known_slots"]["format"] == "онлайн"
    assert view["client_confirmed_slots"]["grade"] == "8"
    assert view["slot_sources"]["grade"] == "memory_provenance"
    assert "физика 8 онлайн" in view["slot_provenance"]["grade"]["quote"]

    assert extract_grade("физика 8 онлайн") == "8"


def test_tz124_p2_explicit_class_route_case_still_extracts_without_parser_change(monkeypatch) -> None:
    view = _memory_view("8 класс математика онлайн", monkeypatch, flag="1", brand="foton")

    assert view["known_slots"]["grade"] == "8"
    assert view["known_slots"]["subject"] == "математика"
    assert view["known_slots"]["format"] == "онлайн"


def test_tz124_p3_p4_bare_and_exam_anchors(monkeypatch) -> None:
    p3 = _memory_view("информатика 10 очно", monkeypatch, flag="1")
    p4 = _memory_view("9 класс, ОГЭ", monkeypatch, flag="1")

    assert p3["known_slots"]["grade"] == "10"
    assert p3["known_slots"]["subject"] == "информатика"
    assert p3["known_slots"]["format"] == "очно"
    assert extract_grade("информатика 10 очно") == "10"
    assert p4["known_slots"]["grade"] == "9"
    assert extract_grade("9 класс, ОГЭ") == "9"


def test_tz124_p5_ellipsis_keeps_compact_grade_without_reask(monkeypatch) -> None:
    monkeypatch.setenv(MEMORY_PROVENANCE_ENV, "1")
    monkeypatch.setenv(ANCHORED_BARE_GRADE_ENV, "1")
    initial = build_dialogue_memory(
        current_message="физика 8 онлайн",
        active_brand="unpk",
        session_id="tz124-p5",
    )
    followup = build_dialogue_memory(
        current_message="а по физике?",
        active_brand="unpk",
        previous_memory=initial,
        session_id="tz124-p5",
    ).to_prompt_view()

    assert followup["known_slots"]["grade"] == "8"
    assert "grade" in followup["do_not_ask_again"]


def test_tz124_n1_n4_number_traps_do_not_extract_grade(monkeypatch) -> None:
    cases = {
        "N1_phone": "звоните на 8 800 555 35 35, физика онлайн",
        "N2_age": "нам с 8 лет можно? физика онлайн",
        "N3_time_colon": "в 8:00 удобно? физика онлайн",
        "N3_time_preposition": "в 8 удобно? физика онлайн",
        "N4_count": "у меня 2 детей, физика онлайн",
        "N4_lessons": "8 занятий по физике онлайн",
        "N_money": "8 тыс за физику онлайн?",
        "N_range": "8-10 августа физика онлайн",
        "N_date": "8 июня физика онлайн",
    }
    for text in cases.values():
        view = _memory_view(text, monkeypatch, flag="1")
        assert "grade" not in view["known_slots"], text
        assert extract_grade(text) == "", text


def test_tz124_n5_multi_subject_does_not_create_single_confirmed_subject_scope(monkeypatch) -> None:
    view = _memory_view("информатика и математика 8 онлайн", monkeypatch, flag="1")

    assert view["known_slots"]["grade"] == "8"
    assert "subject" not in view["known_slots"]
    assert "subject" not in view["client_confirmed_slots"]


def test_tz124_n6_online_and_offline_comparison_keeps_format_empty(monkeypatch) -> None:
    view = _memory_view("онлайн и очно сравниваю, физика 8", monkeypatch, flag="1")

    assert view["known_slots"]["grade"] == "8"
    assert view["known_slots"]["subject"] == "физика"
    assert "format" not in view["known_slots"]
    assert extract_format("онлайн и очно сравниваю") == ""


def test_tz124_n7_n8_p0_still_wins(monkeypatch) -> None:
    p0 = _memory_view("дважды списали, верните", monkeypatch, flag="1")
    composite = _memory_view("дважды списали, верните; а в каком классе берёте?", monkeypatch, flag="1")

    assert p0["handoff_state"] == "required"
    assert "p0" in p0["risk_flags"]
    assert composite["handoff_state"] == "required"
    assert "p0" in composite["risk_flags"]
    assert "grade" not in composite["known_slots"]


def test_tz124_n9_phone_balance_does_not_extract_grade(monkeypatch) -> None:
    view = _memory_view("посмотрите баланс по +7 916 123 45 67", monkeypatch, flag="1")

    assert "grade" not in view["known_slots"]
    assert extract_grade("посмотрите баланс по +7 916 123 45 67") == ""


def test_tz124_flat_new_lead_funnel_gets_bare_grade_only_when_enabled(monkeypatch) -> None:
    monkeypatch.setenv(ANCHORED_BARE_GRADE_ENV, "0")
    off = build_lead_funnel_state(
        "физика 8 онлайн",
        active_brand="unpk",
        topic_id="theme:001_pricing",
    )
    assert off.known_slots.grade == ""

    monkeypatch.setenv(ANCHORED_BARE_GRADE_ENV, "1")
    on = build_lead_funnel_state(
        "физика 8 онлайн",
        active_brand="unpk",
        topic_id="theme:001_pricing",
    )
    assert on.known_slots.grade == "8"
    assert "grade" not in on.missing_slots


def test_tz124_runner_writes_parallel_off_on_pack(tmp_path) -> None:
    out_dir = tmp_path / "tz124"

    assert run_tz124_pack(["--out-dir", str(out_dir), "--parallel", "4"]) == 0

    summary = __import__("json").loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["parallel"] == 4
    assert summary["mode_counts"] == {"off": 15, "on": 15}
    assert summary["gate_passed"] is True
    assert summary["llm_calls_total"] == 0
    assert summary["stop_conditions"]["false_grade_from_number_trap"] is False
    assert summary["stop_conditions"]["price_under_extracted_class"] is False
    transcripts = (out_dir / "transcripts.md").read_text(encoding="utf-8")
    assert "P1 / on" in transcripts
    assert "MEMORY_GRADE: 8" in transcripts
