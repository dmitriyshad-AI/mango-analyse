from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.run_tz121_question_catalog_c_hybrid_shadow import main as c_hybrid_main


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_tz121_question_catalog_c_hybrid_shadow_uses_followup_guard(tmp_path: Path) -> None:
    input_path = tmp_path / "predictions.csv"
    guard_path = tmp_path / "guard.csv"
    out_dir = tmp_path / "out"
    _write_csv(
        input_path,
        [
            {
                "question_id": "q_guard",
                "raw_text": "Сколько стоит с учетом скидки?",
                "human_label": "theme:005_discounts",
                "rule_theme_id": "theme:005_discounts",
                "model_theme_id": "theme:001_pricing",
                "model_confidence": "0.91",
                "source": "synthetic",
            },
            {
                "question_id": "q_model",
                "raw_text": "Можно оплатить наличными?",
                "human_label": "theme:002_payment_method",
                "rule_theme_id": "theme:003_payment_status",
                "model_theme_id": "theme:002_payment_method",
                "model_confidence": "0.86",
                "source": "synthetic",
            },
            {
                "question_id": "q_service_model",
                "raw_text": "Спасибо, больше не актуально",
                "human_label": "service:S1_non_question",
                "rule_theme_id": "service:S5_general_consultation",
                "model_theme_id": "service:S1_non_question",
                "model_confidence": "0.84",
                "source": "synthetic",
            },
        ],
    )
    _write_csv(
        guard_path,
        [
            {
                "question_id": "q_guard",
                "human_rule": "theme:005_discounts",
                "model": "theme:001_pricing",
                "regression_class": "discount_vs_price",
                "note": "стоимость с учетом скидки остается скидочной темой",
            }
        ],
    )

    assert c_hybrid_main(["--input", str(input_path), "--guard-review", str(guard_path), "--out-dir", str(out_dir)]) == 0

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["mode"] == "shadow"
    assert summary["hybrid_vs_gold"]["correct"] == 3
    assert summary["target_passed"] is True
    assert summary["llm_calls_total"] == 0
    assert summary["primary_run"] is False

    rows = list(csv.DictReader((out_dir / "tz121_c_question_catalog_hybrid_trace.csv").open(encoding="utf-8-sig")))
    by_id = {row["id"]: row for row in rows}
    assert by_id["q_guard"]["hybrid"] == "theme:005_discounts"
    assert by_id["q_guard"]["guard_reason"] == "followup_regression:discount_vs_price"
    assert by_id["q_model"]["hybrid"] == "theme:002_payment_method"
    assert by_id["q_service_model"]["hybrid"] == "service:S1_non_question"
    assert all(row["input_fragment"] == "redacted calibration question" for row in rows)


def test_tz121_question_catalog_c_hybrid_primary_is_offline_only(tmp_path: Path) -> None:
    input_path = tmp_path / "predictions.csv"
    guard_path = tmp_path / "guard.csv"
    out_dir = tmp_path / "out"
    _write_csv(
        input_path,
        [
            {
                "question_id": "q_primary",
                "raw_text": "Можно оплатить наличными?",
                "human_label": "theme:002_payment_method",
                "rule_theme_id": "theme:003_payment_status",
                "model_theme_id": "theme:002_payment_method",
                "model_confidence": "0.86",
                "source": "synthetic",
            }
        ],
    )
    _write_csv(
        guard_path,
        [
            {
                "question_id": "unused",
                "human_rule": "theme:005_discounts",
                "model": "theme:001_pricing",
                "regression_class": "discount_vs_price",
                "note": "",
            }
        ],
    )

    assert (
        c_hybrid_main(
            ["--input", str(input_path), "--guard-review", str(guard_path), "--out-dir", str(out_dir), "--mode", "primary"]
        )
        == 0
    )

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["mode"] == "primary"
    assert summary["primary_run"] is True
    assert summary["stop_for_regrede"] is False
    assert summary["llm_calls_total"] == 0
    assert summary["safety"]["rebuilds_main_catalog"] is False
    assert summary["safety"]["writes_db"] is False
