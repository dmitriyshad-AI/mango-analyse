from __future__ import annotations

import json

from scripts.run_tz123_target_question_set import main as run_target_set


def test_tz123_target_question_set_off_handoff_on_single_question(tmp_path) -> None:
    out_dir = tmp_path / "target"

    assert run_target_set(["--out-dir", str(out_dir), "--parallel", "4"]) == 0

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["cases_total"] == 10
    assert summary["off_handoff"] == 10
    assert summary["on_fired"] == 10
    assert summary["on_by_slot"] == {"format": 2, "grade": 3, "subject": 3, "time": 2}
    assert summary["failed_checks"] == []
    assert summary["gate_passed"] is True
    assert summary["llm_calls_total"] == 0

    transcripts = (out_dir / "transcripts.md").read_text(encoding="utf-8")
    assert "OFF_ROUTE: draft_for_manager" in transcripts
    assert "ON_ROUTE: bot_answer_self_for_pilot" in transcripts
    assert "Подскажите, пожалуйста" in transcripts
