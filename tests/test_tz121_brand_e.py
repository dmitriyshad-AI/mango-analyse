from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.run_tz121_brand_e_micro_shadow import main as brand_e_micro_main


def test_tz121_brand_e_micro_shadow_reports_root_matching_and_fail_closed(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"

    assert brand_e_micro_main(["--out-dir", str(out_dir)]) == 0

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["mode"] == "shadow"
    assert summary["rows_total"] == 12
    assert summary["model_correct"] == 12
    assert summary["model_break_rows"] == 0
    assert summary["foton_gold_rows"] == 7
    assert summary["foton_unknown_cyrillic_v2"] == 0
    assert summary["cross_brand_rows"] == 2
    assert summary["cross_brand_fail_closed"] == 2
    assert summary["llm_calls_total"] == 0
    assert summary["primary_run"] is False
    assert summary["stop_for_regrede"] is True
    assert summary["safety"]["unknown_fail_closed"] is True
    assert summary["safety"]["runs_full_set"] is False

    rows = list(csv.DictReader((out_dir / "tz121_e_brand_trace.csv").open(encoding="utf-8-sig")))
    assert next(row for row in rows if row["id"] == "e08")["flip"] == "unknown->unpk"
    assert next(row for row in rows if row["id"] == "e09")["model"] == "unknown"
    assert next(row for row in rows if row["id"] == "e10")["model"] == "unknown"


def test_tz121_brand_e_micro_runner_rejects_non_shadow_mode(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="shadow-only"):
        brand_e_micro_main(["--out-dir", str(tmp_path / "out"), "--mode", "primary"])
