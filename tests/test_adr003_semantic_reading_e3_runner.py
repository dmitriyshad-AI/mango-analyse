from __future__ import annotations

import json
import subprocess
from pathlib import Path

from scripts.run_telegram_dynamic_client_sim import write_progress_json


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_adr003_semantic_reading_e3_paired.sh"


def _runner_text() -> str:
    return RUNNER.read_text(encoding="utf-8")


def test_adr003_e3_runner_shell_syntax() -> None:
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)


def test_adr003_e3_runner_validates_eligible_frame_rate_not_all_turns() -> None:
    text = _runner_text()

    assert "eligible_frame_rate" in text
    assert "preblocked_p0=" in text
    assert "timeouts=" in text
    assert "model_called_eligible=" in text
    assert "P0_PREBLOCK_REASONS" in text
    assert "direct_path_preblocked_p0" in text
    assert "provider errors" in text
    assert "value != \"timeout\"" in text
    assert "semantic frame metadata on" not in text


def test_adr003_e3_runner_passes_progress_json_to_both_legs() -> None:
    text = _runner_text()

    assert "--progress-json \"$OUT/B/progress.json\"" in text
    assert "--progress-leg B" in text
    assert "--progress-json \"$OUT/ON/progress.json\"" in text
    assert "--progress-leg ON" in text


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
