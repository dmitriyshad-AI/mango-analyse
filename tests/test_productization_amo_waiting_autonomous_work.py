from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

from mango_mvp.productization.amo_waiting_autonomous_work import build_amo_waiting_autonomous_work


def test_waiting_work_prepares_non_duplicate_refresh_and_readback_batches(tmp_path: Path) -> None:
    project = _fixture_project(tmp_path)

    summary = build_amo_waiting_autonomous_work(project_root=project, out_root=project / "stable_runtime" / "waiting")

    counts = summary["counts"]
    assert counts["text_quality_review_rows"] == 2
    assert counts["text_quality_cleared_rows"] == 2
    assert counts["non_duplicate_live_candidate_rows"] == 1
    assert counts["contact_id_mismatch_rows"] == 1
    assert counts["refresh_candidate_rows"] == 1
    assert counts["readback_missing_rows"] == 1
    assert summary["policy"]["write_crm"] is False
    assert summary["policy"]["live_write_executed"] is False
    out = project / "stable_runtime" / "waiting"
    assert list(csv.DictReader((out / "text_quality_cleared_candidates_ru.csv").open(encoding="utf-8-sig")))[0]["AMO contact IDs"] == "222"
    assert list(csv.DictReader((out / "already_written_refresh_candidates_ru.csv").open(encoding="utf-8-sig")))[0]["AMO contact IDs"] == "111"
    assert "execute-live-write" not in (out / "next_refresh_real_tunnel_dry_run_command.sh").read_text(encoding="utf-8")
    assert (out / "dashboard.html").exists()


def test_waiting_work_cli_runs_read_only(tmp_path: Path) -> None:
    project = _fixture_project(tmp_path)
    completed = subprocess.run(
        [
            "python3",
            "scripts/run_amo_waiting_autonomous_work.py",
            "--project-root",
            str(project),
            "--out-root",
            str(project / "stable_runtime" / "waiting"),
            "--analysis-date",
            "2026-05-11",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "prepared_safe_next_batches" in completed.stdout


def _fixture_project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    stable = project / "stable_runtime"
    queue = stable / "amo_writeback_queue_20260510_v2_production"
    export = stable / "export"
    reports = stable / "amocrm_runtime" / "contact_writebacks" / "run1" / "readback_gate"
    queue.mkdir(parents=True)
    export.mkdir(parents=True)
    reports.mkdir(parents=True)
    source_csv = export / "amo_export_ready_ru.csv"
    source_rows = [
        _source_row("+79000000001", "111", "written text row", written="written", written_contact_id="111"),
        _source_row("+79000000002", "222", "clear text candidate"),
        _source_row("+79000000003", "333", "mismatch row"),
        _source_row("+79000000004", "444", "already written missing readback", written="written", written_contact_id="444"),
    ]
    _write_csv(source_csv, source_rows)
    _write_json(export / "summary.json", {"tenant_config": {"path": str(project / "tenant.json")}})
    _write_json(stable / "CURRENT_RUNTIME.json", {"paths": {"amo_export_ready_csv": str(source_csv), "active_export_summary": str(export / "summary.json")}})
    _write_csv(queue / "needs_text_quality_review.csv", [source_rows[0], source_rows[1]])
    mismatch = dict(source_rows[2])
    mismatch["dry_run_contact_id"] = "999"
    mismatch["queue_reason"] = "dry_run_contact_id_mismatch"
    _write_csv(queue / "blocked_contact_id_mismatch.csv", [mismatch])
    _write_csv(queue / "already_written.csv", [source_rows[3]])
    _write_csv(
        reports / "readback_report.csv",
        [
            {
                "row_index": "1",
                "source_status": "written",
                "phone": "+79000000001",
                "contact_id": "111",
                "contact_name": "Test",
                "readback_status": "evaluated",
                "decision": "allow",
                "risk_types": "",
                "field::Статус матчинга": "old",
                "field::AI-приоритет": "cold",
                "field::AI-рекомендованный следующий шаг": "Старый шаг",
                "field::Последняя AI-сводка": "Старая сводка",
                "field::Авто история общения": "Старая история",
            }
        ],
    )
    return project


def _source_row(phone: str, contact_id: str, summary: str, *, written: str = "", written_contact_id: str = "") -> dict[str, str]:
    return {
        "queue_bucket": "",
        "queue_reason": "",
        "source_row_index": contact_id,
        "normalized_phone": phone,
        "source_amo_contact_ids": contact_id,
        "effective_contact_id": contact_id,
        "written_status": written,
        "written_contact_id": written_contact_id,
        "written_report": "report.csv" if written else "",
        "Телефон клиента": phone,
        "AMO contact IDs": contact_id,
        "Статус матчинга Tallanto": "exact",
        "Краткое резюме последнего свежего звонка": summary,
        "Краткая история общения": "Клиент интересуется летним лагерем для ребенка. Следующий шаг: отправить материалы.",
        "Хронология общения (последние 5 касаний)": "01.04.2026 — менеджер (sales_call): летний лагерь",
        "Возражения": "Актуальные: нужно согласовать даты",
        "Следующий шаг": "Отправить материалы",
        "Рекомендуемая дата следующего контакта": "2026-05-11",
        "Приоритет лида": "cold",
        "Вероятность продажи, %": "40",
        "CRM writeback policy": "live_update_ready",
        "CRM writeback blockers": "",
        "AMO entity policy": "update_existing_single_amo_contact",
        "Готово к записи в AMO": "Да",
    }


def _write_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
