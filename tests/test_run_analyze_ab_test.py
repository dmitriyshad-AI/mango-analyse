from __future__ import annotations

import importlib.util
import json
import sqlite3
from argparse import Namespace
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "run_analyze_ab_test.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("run_analyze_ab_test", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_explicit_ab_arms() -> None:
    script = _load_script()
    args = Namespace(
        arms=[
            "mini_v6:gpt-5.4-mini:compact",
            "mini_v7:gpt-5.4-mini:full",
        ],
        models=["ignored"],
        prompt_profile="compact",
    )

    assert script.parse_arms(args) == [
        {"name": "mini_v6", "model": "gpt-5.4-mini", "prompt_profile": "compact"},
        {"name": "mini_v7", "model": "gpt-5.4-mini", "prompt_profile": "full"},
    ]


def test_run_analyze_sets_actual_analyze_model_env(tmp_path: Path) -> None:
    script = _load_script()
    captured_env: dict[str, str] = {}

    def fake_run(cmd, *, capture_output, text, env, check):
        _ = (capture_output, text, check)
        captured_env.update(env)
        return CompletedProcess(cmd, 0, stdout="ok", stderr="")

    with patch.object(script.subprocess, "run", side_effect=fake_run), patch.object(
        script.time,
        "monotonic",
        side_effect=[10.0, 12.5],
    ):
        rc, elapsed, stdout, stderr = script.run_analyze(
            cli_path=Path("/tmp/run-cli.sh"),
            db_path=tmp_path / "test.db",
            model="gpt-5.4-mini",
            reasoning="high",
            provider="codex_cli",
            timeout_sec=180,
            export_dir=tmp_path / "exports",
            limit=3,
            prompt_profile="compact",
        )

    assert rc == 0
    assert elapsed == 2.5
    assert stdout == "ok"
    assert stderr == ""
    assert captured_env["CODEX_ANALYZE_MODEL"] == "gpt-5.4-mini"
    assert captured_env["CODEX_MERGE_MODEL"] == "gpt-5.4-mini"
    assert captured_env["ANALYZE_PROMPT_PROFILE"] == "compact"
    assert captured_env["TRANSCRIPT_EXPORT_DIR"] == ""


def test_prepare_db_copy_accepts_canonical_calls_source(tmp_path: Path) -> None:
    script = _load_script()
    db_path = tmp_path / "canonical.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE canonical_calls (
                canonical_call_id INTEGER PRIMARY KEY,
                source_file TEXT,
                source_filename TEXT,
                source_call_id TEXT,
                duration_sec REAL,
                phone TEXT,
                manager_name TEXT,
                direction TEXT,
                started_at TEXT,
                transcription_status TEXT,
                resolve_status TEXT,
                analysis_status TEXT,
                sync_status TEXT,
                dead_letter_stage TEXT,
                transcript_manager TEXT,
                transcript_client TEXT,
                transcript_text TEXT,
                transcript_variants_json TEXT,
                resolve_json TEXT,
                resolve_quality_score REAL,
                analysis_json TEXT,
                amocrm_contact_id INTEGER,
                amocrm_lead_id INTEGER,
                last_error TEXT,
                created_at TEXT,
                selected_updated_at TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO canonical_calls (
                canonical_call_id, source_file, source_filename, source_call_id,
                duration_sec, phone, manager_name, direction, started_at,
                transcription_status, resolve_status, analysis_status, sync_status,
                dead_letter_stage, transcript_text, created_at, selected_updated_at
            )
            VALUES (
                42, '/tmp/a.mp3', 'a.mp3', 'mango-42',
                180, '+79990000000', 'Менеджер', 'inbound', '2026-01-01 10:00:00',
                'done', 'done', 'done', 'pending',
                '', 'Клиент спрашивает про математику.', '2026-01-01 10:01:00', '2026-01-01 10:02:00'
            )
            """
        )
        conn.commit()
    finally:
        conn.close()

    script.prepare_db_copy(db_path, [42])

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT id, analysis_status, analysis_json, analysis_worker_id FROM call_records WHERE id=42"
        ).fetchone()
    finally:
        conn.close()
    assert row == (42, "pending", None, None)


def test_summarize_call_reads_meta_and_redacts_personal_data() -> None:
    script = _load_script()
    row = {
        "id": 101,
        "source_filename": "2026-01-01__79991234567__Иванова Анна.mp3",
        "duration_sec": 180,
        "manager_name": "Петрова",
        "phone": "+79991234567",
        "analysis_status": "done",
        "last_error": "Ошибка по телефону +79991234567",
        "analysis_json": json.dumps(
            {
                "analysis_meta": {
                    "analysis_model": "gpt-5.4-mini",
                    "analysis_provider": "codex_cli",
                    "analysis_prompt_version": "v6",
                },
                "history_summary": "Клиент +79991234567 спросил про math@example.com",
                "structured_fields": {
                    "contacts": {"email": "math@example.com"},
                    "student": {},
                    "interests": {"subjects": ["математика"]},
                    "next_step": {"action": "Позвонить +79991234567"},
                },
                "target_product": "годовые курсы",
                "objections": ["дорого"],
            },
            ensure_ascii=False,
        ),
    }

    summary = script.summarize_call(row)

    assert "source_filename" not in summary
    assert summary["source_filename_sha256"]
    assert summary["phone_masked"] == "7***4567"
    assert summary["email_present"] is True
    assert summary["analysis_model"] == "gpt-5.4-mini"
    assert summary["analysis_prompt_version"] == "v6"
    assert "history_summary" not in summary
    assert "next_step_action" not in summary
    assert summary["history_summary_present"] is True
    assert summary["history_summary_len"] > 0
    assert summary["next_step_action_present"] is True
    assert summary["next_step_action_len"] > 0
    assert "[phone]" in summary["analysis_error"]


def test_coverage_matrix_includes_prompt_version() -> None:
    script = _load_script()
    record = {
        "arm": "mini_v6",
        "model": "gpt-5.4-mini",
        "provider": "codex_cli",
        "prompt_profile": "compact",
        "returncode": 0,
        "metrics": {"analysis_model_missing": 0, "analysis_prompt_version_missing": 0},
        "elapsed_sec": 3.0,
    }
    row = script.build_coverage_row(
        record,
        [
            {
                "analysis_status": "done",
                "analysis_prompt_version": "v6",
                "target_product": "годовые курсы",
                "next_step_action_present": True,
                "next_step_action_len": 19,
                "objections": [],
                "history_summary_present": True,
                "history_summary_len": 16,
            }
        ],
    )

    assert row["prompt_version"] == "v6"
    assert row["target_product_present_pct"] == 100.0
    assert row["next_step_action_present_pct"] == 100.0
