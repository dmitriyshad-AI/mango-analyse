from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts.validate_adr003_e3_leg import validate_leg
from scripts.run_telegram_dynamic_client_sim import write_progress_json


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_adr003_semantic_reading_e3_paired.sh"
VALIDATOR = ROOT / "scripts" / "validate_adr003_e3_leg.py"


def _runner_text() -> str:
    return RUNNER.read_text(encoding="utf-8")


def _validator_text() -> str:
    return VALIDATOR.read_text(encoding="utf-8")


def test_adr003_e3_runner_shell_syntax() -> None:
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)


def test_adr003_e3_runner_validates_eligible_frame_rate_not_all_turns() -> None:
    text = _runner_text()
    validator_text = _validator_text()

    assert "scripts/validate_adr003_e3_leg.py" in text
    assert "eligible_frame_rate" in validator_text
    assert "preblocked_p0=" in validator_text
    assert "timeouts=" in validator_text
    assert "model_called_eligible=" in validator_text
    assert "gate_blocked_turns=" in validator_text
    assert "semantic frame metadata on" not in text


def test_adr003_e3_runner_has_resume_on_report_mode() -> None:
    text = _runner_text()

    resume_index = text.index("mode=resume-on-report")
    full_run_index = text.index("ADR003 E3 paired")
    assert "--resume-on-report" in text
    assert "validate_leg B 0" in text[resume_index:full_run_index]
    assert "run_on_leg" in text[resume_index:full_run_index]
    assert "run_report" in text[resume_index:full_run_index]
    assert "Done resume-on-report" in text[resume_index:full_run_index]


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
