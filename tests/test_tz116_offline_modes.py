from __future__ import annotations

import csv
import json
from pathlib import Path

from mango_mvp.customer_timeline.canonical_readonly_import import infer_brand
from mango_mvp.insights.outcome_linker import classify_tallanto_rows
from scripts.evaluate_tz116_mono_role_assignment import main as mono_eval_main
from scripts.run_tz116_crm_llm_offline_measure import main as crm_measure_main
from scripts.run_tz116_question_catalog_offline_measure import main as qc_measure_main


def test_outcome_linker_default_off_preserves_legacy_negation_behavior() -> None:
    signal = classify_tallanto_rows(
        [
            {
                "tallanto_id": "t1",
                "student_type": "8 класс",
                "history_raw": "Клиент не оплатил и не записался в группу.",
            }
        ]
    )

    assert signal.label == "won_paid_or_active"
    assert "outcome_model_shadow" not in signal.metadata


def test_outcome_linker_shadow_reports_negation_aware_disagreement_without_changing_final_label() -> None:
    signal = classify_tallanto_rows(
        [
            {
                "tallanto_id": "t1",
                "student_type": "8 класс",
                "history_raw": "Клиент не оплатил и не записался в группу.",
            }
        ],
        outcome_model_mode="shadow",
    )

    assert signal.label == "won_paid_or_active"
    shadow = signal.metadata["outcome_model_shadow"]
    assert shadow["legacy_label"] == "won_paid_or_active"
    assert shadow["semantic_label"] == "known_student_or_lead"
    assert shadow["label_changed"] is True


def test_outcome_linker_primary_uses_negation_aware_signal_for_synthetic_input() -> None:
    signal = classify_tallanto_rows(
        [
            {
                "tallanto_id": "t1",
                "student_type": "8 класс",
                "history_raw": "Не отказались, оплатили и ждут первое занятие.",
            }
        ],
        outcome_model_mode="primary",
    )

    assert signal.label == "won_paid_or_active"
    assert "tallanto_history_has_affirmed_refusal_terms" not in signal.reasons


def test_infer_brand_default_legacy_and_cyrillic_v2_are_separate() -> None:
    assert infer_brand(["Фотон и УНПК"]) == "unpk"
    assert infer_brand(["Фотон и УНПК"], mode="cyrillic_v2") == "unknown"
    assert infer_brand(["МПК МФТИ"], mode="cyrillic_v2") == "unpk"
    assert infer_brand(["У Н П К М Ф Т И"], mode="cyrillic_v2") == "unpk"
    assert infer_brand(["ЦДПО"], mode="cyrillic_v2") == "foton"
    assert infer_brand(["просто МФТИ"], mode="cyrillic_v2") == "unknown"
    assert infer_brand(["фотонный эффект"], mode="cyrillic_v2") == "unknown"
    assert infer_brand(["unpkg пакет"], mode="cyrillic_v2") == "unknown"


def test_crm_llm_offline_measure_shadow_never_allows_writeback(tmp_path: Path) -> None:
    input_path = tmp_path / "crm.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "case_id": "case-1",
                "heuristic_analysis": {
                    "close_verdict": "closed_valid",
                    "premature_close_risk": "no_risk",
                    "match_confidence": 0.95,
                    "analysis_source": "heuristic",
                },
                "llm_analysis": {
                    "close_verdict": "reopen_recommended",
                    "premature_close_risk": "high",
                    "confidence": 0.91,
                    "needs_manual_review": False,
                    "conflict_flags": [],
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"

    assert crm_measure_main(["--input", str(input_path), "--out-dir", str(out_dir), "--mode", "shadow"]) == 0

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["llm_calls_total"] == 0
    assert summary["safety"]["writes_amo"] is False
    rows = list(csv.DictReader((out_dir / "crm_llm_offline_measure_rows.csv").open(encoding="utf-8-sig")))
    assert rows[0]["final_writeback_allowed"] == "Нет"
    assert "shadow_mode" in rows[0]["final_writeback_blockers"]
    assert "offline_measure_no_writeback" in rows[0]["final_writeback_blockers"]


def test_question_catalog_offline_measure_shadow_uses_precomputed_model_without_live_call(tmp_path: Path) -> None:
    input_path = tmp_path / "questions.csv"
    with input_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["question_id", "raw_text", "human_label", "model_theme_id"])
        writer.writeheader()
        writer.writerow(
            {
                "question_id": "q1",
                "raw_text": "Сколько стоит курс?",
                "human_label": "theme:001_pricing",
                "model_theme_id": "theme:001_pricing",
            }
        )
    out_dir = tmp_path / "qc"

    assert qc_measure_main(["--input", str(input_path), "--out-dir", str(out_dir), "--mode", "shadow"]) == 0

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["llm_calls_total"] == 0
    assert summary["safety"]["rebuilds_main_catalog"] is False
    rows = list(csv.DictReader((out_dir / "question_catalog_offline_predictions.csv").open(encoding="utf-8-sig")))
    assert rows[0]["classification_method"] == "rule_shadow"
    assert rows[0]["model_comparison"] in {"agree", "disagree"}


def test_mono_role_assignment_eval_uses_only_synthetic_roles(tmp_path: Path) -> None:
    input_path = tmp_path / "mono.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "case_id": "m1",
                "gold_roles": ["manager", "client"],
                "rule_roles": ["manager", "manager"],
                "model_roles": ["manager", "client"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "mono"

    assert mono_eval_main(["--input", str(input_path), "--out-dir", str(out_dir), "--mode", "shadow"]) == 0

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["llm_calls_total"] == 0
    assert summary["safety"]["calls_openai"] is False
    rows = list(csv.DictReader((out_dir / "mono_role_assignment_eval_rows.csv").open(encoding="utf-8-sig")))
    assert rows[0]["shadow_rule_model_disagreement"] == "1"
