from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

from mango_mvp.customer_timeline.canonical_readonly_import import infer_brand
from mango_mvp.insights.outcome_linker import classify_tallanto_rows
from mango_mvp.services.transcribe import TranscribeService
from scripts.build_tz116_crm_fixed_snapshot import build_heuristic, select_cases
from scripts.evaluate_tz116_mono_role_assignment import main as mono_eval_main
from scripts.run_tz116_be_real_measure import main as be_measure_main
from scripts.run_tz116_crm_llm_offline_measure import main as crm_measure_main
from scripts.run_tz116_mono_role_gold50_measure import main as mono_gold50_main
from scripts.run_tz116_mono_role_shadow_real import main as mono_real_main
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


def test_crm_llm_offline_measure_shadow_can_use_codex_source_without_writeback(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "crm_codex.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "case_id": "case-1",
                "dossier": {"lead": {"id": 1}, "call_history": []},
                "heuristic_analysis": {
                    "close_verdict": "closed_valid",
                    "premature_close_risk": "no_risk",
                    "match_confidence": 0.95,
                    "analysis_source": "heuristic",
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    class FakeAnalyzer:
        def analyze(self, *, dossier, heuristic_analysis):  # noqa: ANN001
            return {
                "close_verdict": "follow_up_needed",
                "premature_close_risk": "medium",
                "confidence": 0.9,
                "needs_manual_review": False,
                "conflict_flags": [],
                "llm_provider": "codex_cli",
            }

    monkeypatch.setattr(
        "scripts.run_tz116_crm_llm_offline_measure.build_codex_analyzer",
        lambda _args: FakeAnalyzer(),
    )
    out_dir = tmp_path / "out_codex"

    assert crm_measure_main(
        [
            "--input",
            str(input_path),
            "--out-dir",
            str(out_dir),
            "--mode",
            "shadow",
            "--llm-source",
            "codex",
        ]
    ) == 0

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["llm_source"] == "codex"
    assert summary["llm_calls_total"] == 1
    assert summary["safety"]["model_transport"] == "codex_cli"
    assert summary["safety"]["uses_openai_api_key"] is False
    rows = list(csv.DictReader((out_dir / "crm_llm_offline_measure_rows.csv").open(encoding="utf-8-sig")))
    assert rows[0]["final_writeback_allowed"] == "Нет"
    assert "shadow_mode" in rows[0]["final_writeback_blockers"]


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
    assert summary["rule_vs_gold"]["total"] == 1
    assert summary["model_vs_gold"]["correct"] == 1
    rows = list(csv.DictReader((out_dir / "question_catalog_offline_predictions.csv").open(encoding="utf-8-sig")))
    assert rows[0]["classification_method"] == "rule_shadow"
    assert rows[0]["model_comparison"] in {"agree", "disagree"}


def test_question_catalog_offline_measure_shadow_can_use_codex_source(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "questions_codex.csv"
    with input_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["question_id", "raw_text", "human_label"])
        writer.writeheader()
        writer.writerow(
            {
                "question_id": "q1",
                "raw_text": "Сколько стоит курс?",
                "human_label": "theme:001_pricing",
            }
        )

    def fake_codex(prompt, *, codex_bin, candidate, timeout_sec):  # noqa: ANN001
        assert "q1" in prompt
        return json.dumps(
            {
                "items": [
                    {
                        "question_item_id": "q1",
                        "theme_id": "theme:001_pricing",
                        "confidence": 0.93,
                        "reasoning": "вопрос о цене",
                    }
                ]
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr("scripts.run_tz116_question_catalog_offline_measure.call_codex_batch", fake_codex)
    out_dir = tmp_path / "qc_codex"

    assert qc_measure_main(
        [
            "--input",
            str(input_path),
            "--out-dir",
            str(out_dir),
            "--mode",
            "shadow",
            "--llm-source",
            "codex",
            "--batch-size",
            "1",
        ]
    ) == 0

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["llm_source"] == "codex"
    assert summary["llm_calls_total"] == 1
    assert summary["safety"]["model_transport"] == "codex_cli"
    assert summary["safety"]["uses_openai_api_key"] is False
    rows = list(csv.DictReader((out_dir / "question_catalog_offline_predictions.csv").open(encoding="utf-8-sig")))
    assert rows[0]["classification_method"] == "rule_shadow_codex"
    assert rows[0]["model_theme_id"] == "theme:001_pricing"
    assert summary["model_vs_gold"]["correct"] == 1


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


def _write_canonical_mono_db(path: Path) -> None:
    with sqlite3.connect(path) as con:
        con.execute(
            """
            CREATE TABLE canonical_calls (
                canonical_call_id INTEGER PRIMARY KEY,
                source_filename TEXT,
                started_at TEXT,
                manager_name TEXT,
                duration_sec REAL,
                transcript_text TEXT,
                transcript_variants_json TEXT,
                has_transcript_variants_json INTEGER
            )
            """
        )
        payload = {
            "mode": "mono_or_fallback",
            "full": {
                "final": (
                    "Добрый день, вас беспокоит учебный центр. "
                    "Здравствуйте, можно стоимость курса? "
                    "Да, подскажите класс ребенка. "
                    "Девятый класс, нужна подготовка к ОГЭ."
                )
            },
            "role_assignment": {"applied": False, "mode": "off", "meta": None},
        }
        con.execute(
            """
            INSERT INTO canonical_calls (
                canonical_call_id, source_filename, started_at, manager_name, duration_sec,
                transcript_text, transcript_variants_json, has_transcript_variants_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                1,
                "synthetic.mp3",
                "2026-06-01T10:00:00+00:00",
                "Иванов",
                180.0,
                payload["full"]["final"],
                json.dumps(payload, ensure_ascii=False),
                1,
            ),
        )


def test_real_mono_shadow_runner_default_off_reads_sqlite_without_assignment(tmp_path: Path) -> None:
    db = tmp_path / "canonical.db"
    _write_canonical_mono_db(db)
    out_dir = tmp_path / "real_off"

    assert mono_real_main(["--db", str(db), "--out-dir", str(out_dir), "--mode", "off", "--limit", "1"]) == 0

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["mode"] == "off"
    assert summary["llm_calls_total"] == 0
    assert summary["safety"]["reads_db_mode"] == "ro"
    assert summary["safety"]["runs_asr"] is False
    rows = list(csv.DictReader((out_dir / "mono_role_shadow_real_rows.csv").open(encoding="utf-8-sig")))
    assert rows[0]["status"] == "off"


def test_real_mono_shadow_runner_shadow_uses_codex_selective_without_openai_api(tmp_path: Path, monkeypatch) -> None:
    db = tmp_path / "canonical.db"
    _write_canonical_mono_db(db)
    out_dir = tmp_path / "real_shadow"

    def fake_codex(self, turns, manager_name):  # noqa: ANN001
        return self._normalize_role_assignment_payload(  # noqa: SLF001
            {
                "roles": ["manager" if idx % 2 == 0 else "client" for idx, _ in enumerate(turns)],
                "confidence": 0.9,
                "notes": "synthetic",
            },
            turns=turns,
            manager_name=manager_name,
            provider="codex_cli",
        )

    monkeypatch.setattr(TranscribeService, "_assign_roles_with_codex", fake_codex)

    assert mono_real_main(["--db", str(db), "--out-dir", str(out_dir), "--mode", "shadow", "--limit", "1"]) == 0

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["mode"] == "shadow"
    assert summary["model_transport"] == "codex_cli"
    assert summary["safety"]["writes_db"] is False
    assert summary["safety"]["calls_openai_api"] is False
    assert summary["safety"]["uses_openai_api_key"] is False
    rows = list(csv.DictReader((out_dir / "mono_role_shadow_real_rows.csv").open(encoding="utf-8-sig")))
    assert rows[0]["status"] in {"assigned", "not_assigned"}


def test_real_mono_primary_is_blocked_until_gold_regrede(tmp_path: Path) -> None:
    db = tmp_path / "canonical.db"
    _write_canonical_mono_db(db)

    try:
        mono_real_main(["--db", str(db), "--out-dir", str(tmp_path / "primary"), "--mode", "primary", "--limit", "1"])
    except SystemExit as exc:
        assert "primary is blocked" in str(exc)
    else:
        raise AssertionError("primary mode must be blocked without explicit flag")

    try:
        mono_real_main(
            [
                "--db",
                str(db),
                "--out-dir",
                str(tmp_path / "primary_flag"),
                "--mode",
                "primary",
                "--limit",
                "1",
                "--allow-primary-after-gold-regrede",
            ]
        )
    except SystemExit as exc:
        assert "primary is blocked" in str(exc)
    else:
        raise AssertionError("primary mode must stay blocked even with legacy flag")


def test_mono_role_gold50_measure_calls_codex_only_for_low_confidence(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "gold50.csv"
    turns = [
        {"i": 1, "start": 0.0, "text": "угу"},
        {"i": 2, "start": 5.0, "text": "ага"},
    ]
    with input_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "canonical_call_id",
                "source_filename",
                "started_at",
                "manager_name",
                "duration_sec",
                "turn_count",
                "gold_roles",
                "notes_for_reviewer",
                "turns_json",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "canonical_call_id": "1",
                "source_filename": "synthetic.mp3",
                "started_at": "2026-06-01T10:00:00+00:00",
                "manager_name": "Иванов",
                "duration_sec": "60",
                "turn_count": "2",
                "gold_roles": json.dumps(["manager", "client"]),
                "notes_for_reviewer": "",
                "turns_json": json.dumps(turns, ensure_ascii=False),
            }
        )

    def fake_codex(self, turns, manager_name):  # noqa: ANN001
        return self._normalize_role_assignment_payload(  # noqa: SLF001
            {"roles": ["manager", "client"], "confidence": 0.93, "notes": "synthetic"},
            turns=turns,
            manager_name=manager_name,
            provider="codex_cli",
        )

    monkeypatch.setattr(TranscribeService, "_assign_roles_with_codex", fake_codex)
    out_dir = tmp_path / "gold50_out"

    assert mono_gold50_main(["--input", str(input_path), "--out-dir", str(out_dir), "--mode", "shadow"]) == 0

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["calls_total"] == 1
    assert summary["rule_low_confidence_calls"] == 1
    assert summary["codex_called_calls"] == 1
    assert summary["llm_calls_total"] == 1
    assert summary["model_vs_gold"]["exact_correct"] == 1
    rows = list(csv.DictReader((out_dir / "mono_role_gold50_measure_rows.csv").open(encoding="utf-8-sig")))
    assert rows[0]["selected_provider"] == "codex_cli"
    assert rows[0]["model_exact_vs_gold"] == "Да"


def test_be_real_measure_counts_negation_shadow_and_brand_flips(tmp_path: Path) -> None:
    tallanto = tmp_path / "tallanto.csv"
    with tallanto.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["tallanto_id", "phone_parent", "phone_extra", "phones_joined", "history_raw", "student_type", "branch", "responsible"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "tallanto_id": "t1",
                "phone_parent": "+79990000000",
                "phone_extra": "",
                "phones_joined": "+79990000000",
                "history_raw": "Клиент не оплатил и не записался.",
                "student_type": "Слушатель",
                "branch": "",
                "responsible": "",
            }
        )
    chains = tmp_path / "chains.csv"
    with chains.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["client_key", "phone"])
        writer.writeheader()
        writer.writerow({"client_key": "+79990000000", "phone": "+79990000000"})
    master = tmp_path / "master.csv"
    with master.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", "branch"])
        writer.writeheader()
        writer.writerow({"name": "Фотон и УНПК", "branch": ""})
        writer.writerow({"name": "ЦДПО", "branch": ""})
    amo_contacts = tmp_path / "amo_contacts.csv"
    amo_deals = tmp_path / "amo_deals.csv"
    for path in (amo_contacts, amo_deals):
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["name"])
            writer.writeheader()
            writer.writerow({"name": "МПК МФТИ"})

    out_dir = tmp_path / "be_out"
    assert be_measure_main(
        [
            "--out-dir",
            str(out_dir),
            "--tallanto-contacts",
            str(tallanto),
            "--client-chains",
            str(chains),
            "--master-contacts",
            str(master),
            "--amo-contacts",
            str(amo_contacts),
            "--amo-deals",
            str(amo_deals),
        ]
    ) == 0

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["llm_calls_total"] == 0
    assert summary["b_outcome_negation_shadow"]["client_chain_rows_changed"] == 1
    assert summary["e_brand_infer"]["total_changed_rows"] >= 1


def test_crm_fixed_snapshot_selection_keeps_brand_split_and_skips_conflicts() -> None:
    def lead(lead_id: int, brand_text: str, *, closed_at: int = 1000) -> dict[str, object]:
        return {
            "id": lead_id,
            "name": brand_text,
            "pipeline_id": 8938034,
            "status_id": 143,
            "closed_at": closed_at,
            "custom_fields_values": [
                {
                    "field_name": "Организация",
                    "values": [{"value": brand_text}],
                },
                {
                    "field_name": "Причина отказа (лид)",
                    "values": [{"value": "Не актуально"}],
                },
            ],
        }

    selected = select_cases(
        [
            lead(1, "ЦДПО ФОТОН"),
            lead(2, "АНО УНПК МФТИ"),
            lead(3, "Фотон УНПК"),
        ],
        per_brand=1,
        pipeline_ids={8938034},
    )

    assert [item["id"] for item in selected] == [1, 2]
    heuristic = build_heuristic(
        lead=selected[0],
        brand="foton",
        pipeline_name="Лиды",
        status_name="Закрыто и не реализовано",
        loss_reason="Не актуально",
    )
    assert heuristic["close_verdict"] == "manual_review"
    assert heuristic["analysis_source"] == "heuristic"
