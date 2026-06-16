from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.run_tz121_outcome_b_micro_shadow import main as outcome_b_micro_main


def test_tz121_outcome_b_micro_shadow_reports_allowed_and_blocked_flips(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"

    assert outcome_b_micro_main(["--out-dir", str(out_dir)]) == 0

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["mode"] == "shadow"
    assert summary["rows_total"] == 10
    assert summary["allowed_primary_flip"] == "won_paid_or_active->known_student_or_lead"
    assert summary["allowed_flip_rows"] == 2
    assert summary["allowed_flip_wrong"] == 0
    assert summary["payment_pending_flip_rows"] == 1
    assert summary["payment_pending_flip_primary_blocked"] == 1
    assert summary["primary_run"] is False
    assert summary["stop_for_regrede"] is True
    assert summary["llm_calls_total"] == 0
    assert summary["safety"]["writes_crm"] is False
    assert summary["safety"]["runs_full_set"] is False

    rows = list(csv.DictReader((out_dir / "tz121_b_outcome_trace.csv").open(encoding="utf-8-sig")))
    assert {row["id"] for row in rows if row["primary_allowed"] == "Да"} == {"b01", "b02"}
    payment_pending = next(row for row in rows if row["id"] == "b06")
    assert payment_pending["flip"] == "won_paid_or_active->payment_pending"
    assert payment_pending["primary_allowed"] == "Нет"
    assert "запрещен" in payment_pending["rationale"]


def test_tz121_outcome_b_micro_runner_rejects_non_shadow_mode(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="shadow-only"):
        outcome_b_micro_main(["--out-dir", str(tmp_path / "out"), "--mode", "primary"])
