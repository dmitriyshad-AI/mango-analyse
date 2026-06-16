from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.run_tz121_deal_a_gold_measure import main as deal_a_main


def _write_results(path: Path) -> None:
    rows = [
        {
            "case_id": "a01",
            "heuristic_analysis": {
                "brand": "foton",
                "pipeline_name": "Лиды",
                "status_name": "Закрыто и не реализовано",
                "loss_reason_summary": "Недозвон",
                "close_verdict": "follow_up_needed",
                "premature_close_risk": "medium",
            },
            "llm_analysis": {
                "close_verdict": "follow_up_needed",
                "premature_close_risk": "medium",
                "confidence": 0.83,
                "evidence_signals": ["Недозвон без подтвержденного отказа."],
            },
        },
        {
            "case_id": "a02",
            "heuristic_analysis": {
                "brand": "unpk",
                "pipeline_name": "Лиды",
                "status_name": "Закрыто и не реализовано",
                "loss_reason_summary": "Архив\u2028 (нет связи)",
                "close_verdict": "follow_up_needed",
                "premature_close_risk": "medium",
            },
            "llm_analysis": {
                "close_verdict": "closed_too_early",
                "premature_close_risk": "medium",
                "confidence": 0.91,
                "evidence_signals": ["Нет истории контактов."],
            },
        },
        {
            "case_id": "a03",
            "heuristic_analysis": {
                "brand": "foton",
                "pipeline_name": "Сделки B2C",
                "status_name": "Закрыто и не реализовано",
                "loss_reason_summary": "Не актуально",
                "close_verdict": "manual_review",
                "premature_close_risk": "manual_review",
            },
            "llm_analysis": {
                "close_verdict": "manual_review",
                "premature_close_risk": "manual_review",
                "confidence": 0.35,
                "conflict_flags": ["Недостаточно истории для автоматического вывода."],
            },
        },
    ]
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def test_tz121_deal_a_gold_measure_is_shadow_only_and_conservative(tmp_path: Path) -> None:
    results = tmp_path / "results.jsonl"
    out_dir = tmp_path / "out"
    _write_results(results)

    assert deal_a_main(["--results", str(results), "--out-dir", str(out_dir), "--write-gold"]) == 0

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["mode"] == "shadow"
    assert summary["records_total"] == 3
    assert summary["brand_counts"] == {"foton": 2, "unpk": 1}
    assert summary["rule_exact_vs_gold"]["correct"] == 2
    assert summary["model_exact_vs_gold"]["correct"] == 2
    assert summary["high_confidence_wrong_count"] == 1
    assert summary["llm_calls_total"] == 0
    assert summary["primary_run"] is False
    assert summary["stop_for_regrede"] is True
    assert summary["safety"]["calls_model"] is False
    assert summary["safety"]["writes_amo"] is False
    assert summary["safety"]["writes_tallanto"] is False
    assert summary["safety"]["reads_live_crm"] is False

    gold_rows = list(csv.DictReader((out_dir / "deal_a_gold_labels.csv").open(encoding="utf-8-sig")))
    gold_by_id = {row["case_id"]: row for row in gold_rows}
    assert gold_by_id["a02"]["gold_verdict"] == "manual_review"
    assert gold_by_id["a02"]["gold_risk"] == "manual_review"
    assert gold_by_id["a02"]["gold_reason"] == "archive_no_contact_needs_manual_review_before_auto_conclusion"

    trace_rows = list(csv.DictReader((out_dir / "tz121_a_deal_gold_trace.csv").open(encoding="utf-8-sig")))
    by_id = {row["id"]: row for row in trace_rows}
    assert by_id["a01"]["error_type"] == "both_correct"
    assert by_id["a02"]["error_type"] == "both_wrong"
    assert by_id["a02"]["input_fragment"] == "redacted closed-deal dossier"
    assert by_id["a03"]["rationale"] == "Недостаточно истории для автоматического вывода."


def test_tz121_deal_a_gold_measure_rejects_primary_mode(tmp_path: Path) -> None:
    results = tmp_path / "results.jsonl"
    _write_results(results)

    with pytest.raises(SystemExit, match="shadow-only"):
        deal_a_main(["--results", str(results), "--out-dir", str(tmp_path / "out"), "--mode", "primary"])
