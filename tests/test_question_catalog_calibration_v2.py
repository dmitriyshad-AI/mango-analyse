from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

from mango_mvp.question_catalog.calibration_metrics import (
    compute_classification_metrics,
    validate_labeled_rows,
)


def test_compute_classification_metrics_macro_f1_and_worst_recall() -> None:
    rows = [
        {"human_label": "theme:a", "predicted_theme_id": "theme:a"},
        {"human_label": "theme:a", "predicted_theme_id": "theme:b"},
        {"human_label": "theme:b", "predicted_theme_id": "theme:b"},
        {"human_label": "theme:c", "predicted_theme_id": "theme:a"},
    ]

    metrics = compute_classification_metrics(rows)

    assert metrics.total == 4
    assert metrics.correct == 2
    assert metrics.accuracy == 0.5
    assert metrics.label_count == 3
    assert round(metrics.macro_f1, 4) == 0.3889
    assert metrics.worst_recall(limit=1)[0].label == "theme:c"


def test_validate_labeled_rows_rejects_bad_labels_and_duplicates() -> None:
    rows = [
        {"question_id": "q1", "human_label": "theme:001_pricing"},
        {"question_id": "q1", "human_label": "theme:999_fake"},
        {"question_id": "", "human_label": ""},
    ]

    errors = validate_labeled_rows(rows, {"theme:001_pricing"})

    assert any("duplicate question_id q1" in error for error in errors)
    assert any("invalid human_label theme:999_fake" in error for error in errors)
    assert any("missing question_id" in error for error in errors)
    assert any("missing human_label" in error for error in errors)


def test_calibration_script_rule_mode_writes_report(tmp_path: Path) -> None:
    input_path = tmp_path / "sample.csv"
    out_dir = tmp_path / "out"
    with input_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "question_id",
                "raw_text",
                "source",
                "extracted_params",
                "rule_based_theme_id",
                "human_label",
                "human_label_notes",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "question_id": "q1",
                "raw_text": "Сколько стоит курс по математике?",
                "source": "test",
                "extracted_params": "{}",
                "rule_based_theme_id": "theme:001_pricing",
                "human_label": "theme:001_pricing",
                "human_label_notes": "",
            }
        )

    env = {**os.environ, "PYTHONPATH": "src"}
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_question_catalog_llm_calibration_v2.py",
            "--input",
            str(input_path),
            "--out-dir",
            str(out_dir),
            "--mode",
            "rule",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    report_path = Path(payload["report"])
    assert payload["mode"] == "rule"
    assert report_path.exists()
    report = report_path.read_text(encoding="utf-8")
    assert "not the D.2 acceptance run" in report
    assert "First Mismatches" in report
