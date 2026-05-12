from __future__ import annotations

import csv
import json
import sqlite3
from dataclasses import replace
from pathlib import Path

from mango_mvp.db import build_session_factory, init_db
from mango_mvp.models import CallRecord
from mango_mvp.quality.transcript_quality_hard_gate_backfill_dry_run import (
    HardGateBackfillDryRunConfig,
    build_hard_gate_backfill_dry_run,
)
from mango_mvp.services.analyze import AnalyzeService
from tests.test_dialogue_format import make_settings


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _row_analysis_json(db_path: Path, source_filename: str) -> str:
    con = sqlite3.connect(db_path)
    try:
        row = con.execute(
            "select analysis_json from call_records where source_filename = ?",
            (source_filename,),
        ).fetchone()
        assert row is not None
        return str(row[0] or "")
    finally:
        con.close()


def _analysis_payload(
    call_type: str,
    *,
    sales_leak: bool,
) -> str:
    structured_fields = {
        "people": {},
        "contacts": {"phone_from_filename": "+79161234567"},
        "student": {},
        "interests": {"products": [], "format": [], "subjects": [], "exam_targets": []},
        "commercial": {"price_sensitivity": None, "budget": None, "discount_interest": None},
        "objections": [],
        "next_step": {"action": None, "due": None},
        "lead_priority": "cold",
    }
    payload = {
        "analysis_schema_version": "v2",
        "history_summary": "Нет содержательного диалога.",
        "structured_fields": structured_fields,
        "quality_flags": {"call_type": call_type},
        "target_product": None,
        "follow_up_score": 0,
        "tags": [call_type],
    }
    if sales_leak:
        payload.update(
            {
                "history_summary": "Клиент заинтересовался летним лагерем и попросил отправить ссылку на оплату.",
                "target_product": "летний лагерь",
                "follow_up_score": 90,
                "next_step": "Отправить ссылку на оплату",
                "objections": ["цена"],
                "tags": ["sales_call"],
            }
        )
        payload["quality_flags"] = {"call_type": "sales_call"}
        payload["structured_fields"] = {
            **structured_fields,
            "interests": {
                "products": ["летний лагерь"],
                "format": [],
                "subjects": ["математика"],
                "exam_targets": [],
            },
            "commercial": {"price_sensitivity": "high", "budget": "до 100000", "discount_interest": True},
            "objections": ["цена"],
            "next_step": {"action": "Отправить ссылку на оплату", "due": "завтра"},
            "lead_priority": "hot",
        }
    return json.dumps(payload, ensure_ascii=False)


def test_hard_gate_backfill_dry_run_reports_updates_without_db_write(tmp_path: Path) -> None:
    db_path = tmp_path / "calls.db"
    settings = replace(make_settings(), database_url=f"sqlite:///{db_path}")
    init_db(settings)
    session_factory = build_session_factory(settings)
    stale_json = _analysis_payload("sales_call", sales_leak=True)
    clean_json = _analysis_payload("non_conversation", sales_leak=False)

    with session_factory() as session:
        session.add_all(
            [
                CallRecord(
                    source_file=str(tmp_path / "stale_voicemail.mp3"),
                    source_filename="stale_voicemail.mp3",
                    phone="+79161234567",
                    manager_name="Петрова Анна",
                    transcription_status="done",
                    resolve_status="done",
                    analysis_status="done",
                    duration_sec=24,
                    transcript_text=(
                        "MANAGER:\n"
                        "Добрый день, это Фотон, хотели рассказать про летний лагерь.\n\n"
                        "CLIENT:\n"
                        "Абонент сейчас не может ответить на ваш звонок. "
                        "Оставьте сообщение после звукового сигнала."
                    ),
                    analysis_json=stale_json,
                ),
                CallRecord(
                    source_file=str(tmp_path / "clean_voicemail.mp3"),
                    source_filename="clean_voicemail.mp3",
                    phone="+79160000000",
                    manager_name="Петрова Анна",
                    transcription_status="done",
                    resolve_status="done",
                    analysis_status="done",
                    duration_sec=12,
                    transcript_text=(
                        "CLIENT:\n"
                        "Абонент сейчас не может ответить на ваш звонок. "
                        "Оставьте сообщение после звукового сигнала."
                    ),
                    analysis_json=clean_json,
                ),
            ]
        )
        session.commit()

    out_root = tmp_path / "dry_run"
    summary = build_hard_gate_backfill_dry_run(
        HardGateBackfillDryRunConfig(
            database_url=settings.database_url,
            out_root=out_root,
        ),
        analyze_service=AnalyzeService(settings),
    )

    assert summary["terminal_rows_selected"] == 2
    assert summary["rows_scanned_by_normalizer"] == 2
    assert summary["would_update"] == 1
    assert summary["parse_errors"] == 0
    assert summary["transition_counts"]["sales_call->non_conversation"] == 1
    assert summary["update_reason_counts"]["call_type_to_non_conversation"] == 1
    assert summary["update_reason_counts"]["clear_sales_fields"] == 1
    assert _row_analysis_json(db_path, "stale_voicemail.mp3") == stale_json

    saved_summary = json.loads((out_root / "summary.json").read_text(encoding="utf-8"))
    assert saved_summary["would_update"] == 1
    assert (out_root / "HARD_GATE_BACKFILL_DRY_RUN.md").exists()
    assert (out_root / "would_update_candidates.csv").exists()
    assert (out_root / "all_results.csv").exists()

    rows = _read_csv(out_root / "would_update_candidates.csv")
    assert len(rows) == 1
    row = rows[0]
    assert row["source_filename"] == "stale_voicemail.mp3"
    assert row["current_call_type"] == "sales_call"
    assert row["normalized_call_type"] == "non_conversation"
    assert row["phone"] == "+79161234567"
    assert row["normalized_follow_up_score"] == "0"
    assert row["normalized_next_step"] == ""
    assert row["normalized_products"] == ""
    assert row["normalized_objections"] == ""
    assert row["hard_validation_applied"] == "True"
