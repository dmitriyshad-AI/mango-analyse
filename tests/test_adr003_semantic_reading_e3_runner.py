from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts.validate_adr003_e3_leg import validate_leg
from scripts.run_telegram_dynamic_client_sim import write_progress_json


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_adr003_semantic_reading_e3_paired.sh"
FINAL_PACHKA_RUNNER = ROOT / "scripts" / "run_adr003_final_pachka_pair.sh"
VALIDATOR = ROOT / "scripts" / "validate_adr003_e3_leg.py"


def _runner_text() -> str:
    return RUNNER.read_text(encoding="utf-8")


def _validator_text() -> str:
    return VALIDATOR.read_text(encoding="utf-8")


def _final_pachka_runner_text() -> str:
    return FINAL_PACHKA_RUNNER.read_text(encoding="utf-8")


def test_adr003_e3_runner_shell_syntax() -> None:
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)


def test_adr003_final_pachka_runner_shell_syntax() -> None:
    subprocess.run(["bash", "-n", str(FINAL_PACHKA_RUNNER)], check=True)


def test_adr003_final_pachka_runner_emulates_old_profile_in_b_leg() -> None:
    text = _final_pachka_runner_text()

    assert 'OLD_READING_CLASSES="sense_seats,slots_gsf,off_topic,intent_actions,live_status_read"' in text
    assert 'OLD_APPLY_CLASSES="live_status_read/conversation_intent_plan"' in text
    for flag in (
        "TELEGRAM_FACT_SELECT_FRAME",
        "TELEGRAM_TONE_CLOSE_FRAME_VETO",
        "TELEGRAM_P0_MODEL_LED",
        "TELEGRAM_PROSE_MODEL_LED",
        "TELEGRAM_PAYMENT_REFUND_DISPUTE_SPLIT",
        "TELEGRAM_SEATS_DEFAULT_OPEN",
        "TELEGRAM_P0_LATCH_AUTORELEASE_V2",
    ):
        assert flag in text
    assert 'b_env=(' in text
    b_run_section = text[text.index("run_b_leg()") : text.index("run_report()", text.index("run_b_leg()"))]
    assert "--allow-non-pilot-profile" in b_run_section
    assert 'TELEGRAM_SEMANTIC_READING_CLASSES="$OLD_READING_CLASSES"' in text
    assert 'TELEGRAM_READING_APPLY_CLASSES="$OLD_APPLY_CLASSES"' in text
    assert '"$flag=0"' in text
    assert '--forbid-trace-class "$TARGET_READING_CLASSES"' in text


def test_adr003_final_pachka_runner_uses_current_profile_in_on_leg() -> None:
    text = _final_pachka_runner_text()
    on_section = text[text.index("on_env=(") : text.index("b_env=(")]
    on_run_section = text[text.index("run_on_leg()") : text.index("run_b_leg()", text.index("run_on_leg()"))]

    assert "-u TELEGRAM_SEMANTIC_READING_CLASSES" in on_section
    assert "-u TELEGRAM_READING_APPLY_CLASSES" in on_section
    assert '"current HEAD pilot profile as-is; no manual env for target flags"' in text
    assert '--require-trace-class "$TARGET_READING_CLASSES"' in text
    assert 'RUN_ORDER="${RUN_ORDER:-ON_FIRST}"' in text
    assert "--allow-non-pilot-profile" not in on_run_section


def test_adr003_e3_runner_avoids_empty_expect_arg_array_for_bash32() -> None:
    text = _runner_text()

    assert "expect_arg=()" not in text
    assert "${expect_arg[@]}" not in text
    assert "--expect-trace" in text


def test_adr003_e3_runner_validates_eligible_frame_rate_not_all_turns() -> None:
    text = _runner_text()
    validator_text = _validator_text()

    assert "scripts/validate_adr003_e3_leg.py" in text
    assert '--forbid-trace-class "$TARGET_READING_CLASSES"' in text
    assert '--require-trace-class "$TARGET_READING_CLASSES"' in text
    assert "eligible_frame_rate" in validator_text
    assert "preblocked_p0=" in validator_text
    assert "timeouts=" in validator_text
    assert "model_called_eligible=" in validator_text
    assert "gate_blocked_turns=" in validator_text
    assert "semantic frame metadata on" not in text


def test_adr003_e3_runner_can_append_target_reading_class_on_top_of_profile_default() -> None:
    text = _runner_text()

    assert 'TARGET_READING_CLASS="${TARGET_READING_CLASS:-}"' in text
    assert 'TARGET_READING_CLASSES="${TARGET_READING_CLASSES:-$TARGET_READING_CLASS}"' in text
    base_section = text[
        text.index('BASE_READING_CLASSES="${READING_CLASSES:-') : text.index("TARGET_READING_CLASS=", text.index("BASE_READING_CLASSES="))
    ]
    assert "PROFILE_READING_CLASSES" in base_section
    assert "PILOT_PROFILE_DEFAULT_READING_CLASSES" in text
    assert "already in profile/base READING_CLASSES" in text
    assert 'READING_CLASSES="${READING_CLASSES},${target_reading_class}"' in text
    assert '"TARGET_READING_CLASSES": target_reading_classes' in text
    assert "-u TELEGRAM_SEMANTIC_READING_CLASSES" in text


def test_adr003_e3_runner_supports_target_class_list_and_on_apply_only() -> None:
    text = _runner_text()
    baseline_section = text[text.index("== B:") : text.index("validate_leg B 0", text.index("== B:"))]
    on_section = text[text.index("== ON:") : text.index("validate_leg ON 1", text.index("== ON:"))]

    assert 'TARGET_APPLY_CLASSES="${TARGET_APPLY_CLASSES:-}"' in text
    assert 'BASE_APPLY_CLASSES="${READING_APPLY_CLASSES:-$PROFILE_APPLY_CLASSES}"' in text
    assert 'APPLY_CLASSES="$(merge_csv "$BASE_APPLY_CLASSES" "$TARGET_APPLY_CLASSES")"' in text
    assert "-u TELEGRAM_READING_APPLY_CLASSES" in text
    assert "TELEGRAM_READING_APPLY_CLASSES=" not in baseline_section
    assert 'TELEGRAM_READING_APPLY_CLASSES="$APPLY_CLASSES"' in on_section
    assert "target_apply_classes=$TARGET_APPLY_CLASSES" in text
    assert "apply_classes=$APPLY_CLASSES" in text


def test_adr003_e3_runner_supports_on_first_order() -> None:
    text = _runner_text()

    assert 'RUN_ORDER="${RUN_ORDER:-B_FIRST}"' in text
    assert 'RUN_ORDER must be B_FIRST or ON_FIRST' in text
    assert 'if [[ "$RUN_ORDER" == "ON_FIRST" ]]' in text
    on_first_section = text[text.index('if [[ "$RUN_ORDER" == "ON_FIRST" ]]') :]
    assert on_first_section.index("run_on_leg") < on_first_section.index("run_b_leg")
    assert '"RUN_ORDER": run_order' in text


def test_adr003_e3_runner_baseline_uses_profile_reading_classes() -> None:
    text = _runner_text()
    baseline_section = text[text.index("== B:") : text.index("validate_leg B 0", text.index("== B:"))]
    on_section = text[text.index("== ON:") : text.index("validate_leg ON 1", text.index("== ON:"))]

    assert "TELEGRAM_SEMANTIC_READING_CLASSES=" not in baseline_section
    assert 'TELEGRAM_SEMANTIC_READING_CLASSES="$READING_CLASSES"' in on_section


def test_adr003_e3_runner_does_not_force_reliable_answerer_step1() -> None:
    text = _runner_text()
    base_section = text[text.index("base_env=(") : text.index("validate_leg()", text.index("base_env=("))]
    manifest_section = text[text.index('"required_env": {') : text.index("},\n}", text.index('"required_env": {'))]

    assert "TELEGRAM_RELIABLE_ANSWERER_STEP1=1" not in base_section
    assert '"TELEGRAM_RELIABLE_ANSWERER_STEP1"' not in manifest_section


def test_adr003_e3_runner_has_resume_on_report_mode() -> None:
    text = _runner_text()

    resume_index = text.index("mode=resume-on-report")
    full_run_index = text.index("ADR003 E3 paired")
    assert "--resume-on-report" in text
    assert "validate_leg B 0" in text[resume_index:full_run_index]
    assert "run_on_leg" in text[resume_index:full_run_index]
    assert "run_report" in text[resume_index:full_run_index]
    assert "Done resume-on-report" in text[resume_index:full_run_index]


def test_adr003_e3_runner_writes_manifest_after_red_report() -> None:
    text = _runner_text()

    assert text.count("run_report || report_rc=$?") == 2
    assert text.count('exit "$report_rc"') == 2
    assert text.rindex("run_report || report_rc=$?") < text.rindex("write_sha_manifest")


def test_adr003_e3_runner_passes_progress_json_to_both_legs() -> None:
    text = _runner_text()

    assert "--progress-json \"$OUT/B/progress.json\"" in text
    assert "--progress-leg B" in text
    assert "--progress-json \"$OUT/ON/progress.json\"" in text
    assert "--progress-leg ON" in text


def _e3_summary() -> dict:
    return {
        "run_config": {
            "key_flags": {
                "profile": {"env": "pilot_gold_v1", "effective": True},
            },
        },
        "llm_calls": {"bot_direct_draft": 1},
    }


def _e3_turn(**overrides: object) -> dict:
    turn = {
        "turn": 1,
        "client_message": "Есть занятия?",
        "bot_route": "draft_for_manager",
        "bot_text": "Менеджер проверит.",
        "bot_safety_flags": ["manager_approval_required"],
        "bot_direct_path": {"model_called": True, "preblocked": False},
        "bot_semantic_frame": {
            "intent": "product_info",
            "risk_class": "safe",
            "deal_stage": "research",
            "payment_readiness": "unknown",
            "requested_product": {"brand": "foton"},
            "requested_action": "ask_info",
            "answerability": "can_answer",
            "must_handoff": False,
            "evidence": ["вопрос справочный"],
            "confidence": 0.9,
        },
    }
    turn.update(overrides)
    return turn


def _write_e3_fixture(tmp_path: Path, turn: dict) -> tuple[Path, Path]:
    summary = tmp_path / "dynamic_summary.json"
    transcripts = tmp_path / "dynamic_dialog_transcripts.jsonl"
    summary.write_text(json.dumps(_e3_summary(), ensure_ascii=False), encoding="utf-8")
    transcripts.write_text(
        json.dumps({"dialog_id": "d1", "brand": "foton", "turns": [turn]}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary, transcripts


def _write_e3_dialogs(tmp_path: Path, dialogs: list[dict]) -> tuple[Path, Path]:
    summary = tmp_path / "dynamic_summary.json"
    transcripts = tmp_path / "dynamic_dialog_transcripts.jsonl"
    summary.write_text(json.dumps(_e3_summary(), ensure_ascii=False), encoding="utf-8")
    transcripts.write_text(
        "".join(json.dumps(dialog, ensure_ascii=False) + "\n" for dialog in dialogs),
        encoding="utf-8",
    )
    return summary, transcripts


def test_e3_leg_validator_counts_gate_blocked_as_valid_product_event(tmp_path: Path) -> None:
    summary, transcripts = _write_e3_fixture(
        tmp_path,
        _e3_turn(
            bot_provider_error="authoritative_output_gate_blocked",
            bot_safety_flags=["manager_approval_required", "authoritative_output_gate_blocked"],
        ),
    )

    result = validate_leg(summary_path=summary, transcripts_path=transcripts, leg="B", expect_trace=False)

    assert result["status"] == "valid"
    assert result["gate_blocked_turns"] == 1


def test_e3_leg_validator_tolerates_sparse_timeouts(tmp_path: Path) -> None:
    dialogs = [
        {"dialog_id": "timeout_dialog", "brand": "foton", "run_status": "timeout", "turns": []},
        {
            "dialog_id": "mixed_dialog",
            "brand": "foton",
            "run_status": "completed",
            "turns": [
                _e3_turn(turn=1, bot_provider_error="timeout", bot_semantic_frame={}),
                _e3_turn(turn=2, bot_provider_error="timeout", bot_semantic_frame={}),
                *[_e3_turn(turn=index) for index in range(3, 10)],
            ],
        },
    ]
    summary, transcripts = _write_e3_dialogs(tmp_path, dialogs)

    result = validate_leg(summary_path=summary, transcripts_path=transcripts, leg="B", expect_trace=False)

    assert result["status"] == "valid"
    assert result["timeouts"] == 3
    assert result["timeout_turns"] == 2
    assert result["timeout_dialogs"] == 1
    assert result["timeout_tolerated"] is True


def test_e3_leg_validator_excludes_model_not_called_from_frame_denominator(tmp_path: Path) -> None:
    summary, transcripts = _write_e3_dialogs(
        tmp_path,
        [
            {
                "dialog_id": "guarded",
                "brand": "foton",
                "run_status": "completed",
                "turns": [
                    _e3_turn(
                        turn=1,
                        bot_direct_path={
                            "model_called": False,
                            "preblocked": True,
                            "preblock_reason": "reliable_answerer_p0_bypass",
                        },
                        bot_semantic_frame={},
                    ),
                    _e3_turn(turn=2),
                ],
            },
        ],
    )

    result = validate_leg(summary_path=summary, transcripts_path=transcripts, leg="B", expect_trace=False)

    assert result["status"] == "valid"
    assert result["model_not_called"] == 1
    assert result["model_called_eligible"] == 1
    assert result["frames"] == 1


def test_e3_leg_validator_allows_profile_traces_while_forbidding_target_class(tmp_path: Path) -> None:
    summary, transcripts = _write_e3_fixture(
        tmp_path,
        _e3_turn(bot_semantic_reading_trace=[{"class": "sense_seats", "status": "no_op"}]),
    )

    result = validate_leg(
        summary_path=summary,
        transcripts_path=transcripts,
        leg="B",
        expect_trace=False,
        forbid_trace_class="intent_actions",
    )

    assert result["status"] == "valid"
    assert result["trace_turns"] == 1
    assert result["forbidden_trace_turns"] == 0
    assert result["forbidden_trace_turns_by_class"] == {"intent_actions": 0}


def test_e3_leg_validator_rejects_target_trace_in_baseline(tmp_path: Path) -> None:
    summary, transcripts = _write_e3_fixture(
        tmp_path,
        _e3_turn(bot_semantic_reading_trace=[{"class": "intent_actions", "status": "applied"}]),
    )

    with pytest.raises(SystemExit):
        validate_leg(
            summary_path=summary,
            transcripts_path=transcripts,
            leg="B",
            expect_trace=False,
            forbid_trace_class="intent_actions",
        )


def test_e3_leg_validator_handles_trace_class_lists(tmp_path: Path) -> None:
    summary, transcripts = _write_e3_fixture(
        tmp_path,
        _e3_turn(
            bot_semantic_reading_trace=[
                {"class": "route_templates", "status": "applied"},
                {"class": "live_status_read", "status": "applied"},
            ]
        ),
    )

    result = validate_leg(
        summary_path=summary,
        transcripts_path=transcripts,
        leg="ON",
        expect_trace=True,
        require_trace_class="route_templates,live_status_read",
    )

    assert result["required_trace_classes"] == ["route_templates", "live_status_read"]
    assert result["required_trace_turns_by_class"] == {
        "route_templates": 1,
        "live_status_read": 1,
    }


def test_adr003_e3_runner_rejects_profile_default_target() -> None:
    text = _runner_text()

    assert "already in profile/base READING_CLASSES" in text
    assert "would not be an attributable B/ON target" in text
    assert "exit 2" in text[text.index("already in profile/base READING_CLASSES") :]


def test_e3_leg_validator_requires_target_trace_in_on_leg(tmp_path: Path) -> None:
    summary, transcripts = _write_e3_fixture(
        tmp_path,
        _e3_turn(bot_semantic_reading_trace=[{"class": "intent_actions", "status": "applied"}]),
    )

    result = validate_leg(
        summary_path=summary,
        transcripts_path=transcripts,
        leg="ON",
        expect_trace=True,
        require_trace_class="intent_actions",
    )

    assert result["required_trace_turns"] == 1


def test_e3_leg_validator_fails_when_timeouts_exceed_budget(tmp_path: Path) -> None:
    dialogs = [
        {"dialog_id": "timeout_dialog", "brand": "foton", "run_status": "timeout", "turns": []},
        {
            "dialog_id": "mixed_dialog",
            "brand": "foton",
            "run_status": "completed",
            "turns": [
                _e3_turn(turn=1, bot_provider_error="timeout", bot_semantic_frame={}),
                _e3_turn(turn=2, bot_provider_error="timeout", bot_semantic_frame={}),
                _e3_turn(turn=3, bot_provider_error="timeout", bot_semantic_frame={}),
                *[_e3_turn(turn=index) for index in range(4, 10)],
            ],
        },
    ]
    summary, transcripts = _write_e3_dialogs(tmp_path, dialogs)

    with pytest.raises(SystemExit):
        validate_leg(summary_path=summary, transcripts_path=transcripts, leg="B", expect_trace=False)


def test_e3_leg_validator_fails_on_infra_provider_error(tmp_path: Path) -> None:
    summary, transcripts = _write_e3_fixture(tmp_path, _e3_turn(bot_provider_error="invalid_json"))

    with pytest.raises(SystemExit):
        validate_leg(summary_path=summary, transcripts_path=transcripts, leg="B", expect_trace=False)


def test_progress_json_is_atomic_and_human_readable(tmp_path: Path) -> None:
    progress_path = tmp_path / "B" / "progress.json"
    write_progress_json(
        progress_path,
        leg="B",
        summary={
            "totals": {
                "dialogs": 3,
                "pass": 1,
                "pass_with_notes": 1,
                "fail": 1,
            },
            "hard_gate_failure_dialogs": ["d_fail"],
        },
        total=146,
        last_dialog_id="d3",
    )

    payload = json.loads(progress_path.read_text(encoding="utf-8"))
    assert payload["leg"] == "B"
    assert payload["done_dialogs"] == 3
    assert payload["total"] == 146
    assert payload["pass"] == 1
    assert payload["pass_with_notes"] == 1
    assert payload["fail"] == 1
    assert payload["hard_gate_dialog_ids"] == ["d_fail"]
    assert payload["last_dialog_id"] == "d3"
    assert payload["updated_at"]
    assert not (tmp_path / "B" / "progress.json.tmp").exists()
