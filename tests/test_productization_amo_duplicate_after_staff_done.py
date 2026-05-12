from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

from mango_mvp.productization.amo_duplicate_after_staff_done import build_amo_duplicate_after_staff_done_pipeline


def test_after_staff_done_pipeline_waits_without_recheck_report(tmp_path: Path) -> None:
    project = _fixture_project(tmp_path)
    summary = build_amo_duplicate_after_staff_done_pipeline(project_root=project, out_root=project / "stable_runtime" / "after_staff")

    assert summary["status"] == "waiting_for_staff_done_and_recheck"
    assert summary["candidate_rows"] == 0
    assert summary["blocked_rows"] == 1
    assert summary["policy"]["manual_intake_required"] is False
    assert (project / "stable_runtime" / "after_staff" / "next_quality_gate_command.sh").exists()


def test_after_staff_done_pipeline_builds_candidate_batch_from_green_recheck(tmp_path: Path) -> None:
    project = _fixture_project(tmp_path)
    report = _fixture_report(project, contact_id="112")
    out = project / "stable_runtime" / "after_staff"

    summary = build_amo_duplicate_after_staff_done_pipeline(project_root=project, report_dir=report, out_root=out, analysis_date="2026-05-11")

    assert summary["status"] == "ready_for_quality_gate"
    assert summary["candidate_rows"] == 1
    assert summary["blocked_rows"] == 0
    rows = list(csv.DictReader((out / "post_merge_live_candidates_ru.csv").open(encoding="utf-8-sig")))
    assert rows[0]["AMO contact IDs"] == "112"
    assert rows[0]["CRM writeback policy"] == "live_update_ready"
    assert rows[0]["Готово к записи в AMO"] == "Да"
    assert "--expected-dry-run 1" in (out / "next_real_tunnel_dry_run_command.sh").read_text(encoding="utf-8")
    assert "run_crm_writeback_quality_gate.py" in (out / "next_quality_gate_command.sh").read_text(encoding="utf-8")


def test_after_staff_done_cli_is_non_live(tmp_path: Path) -> None:
    project = _fixture_project(tmp_path)
    completed = subprocess.run(
        [
            "python3",
            "scripts/run_amo_duplicate_after_staff_done.py",
            "--project-root",
            str(project),
            "--out-root",
            str(project / "stable_runtime" / "after_staff"),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "waiting_for_staff_done_and_recheck" in completed.stdout
    assert "execute-live-write" not in (project / "stable_runtime" / "after_staff" / "next_real_tunnel_dry_run_command.sh").read_text(encoding="utf-8")


def _fixture_project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    stable = project / "stable_runtime"
    export_root = stable / "strict_export"
    duplicate_root = stable / "amo_duplicate_resolution_20260511_v1"
    export_root.mkdir(parents=True)
    duplicate_root.mkdir(parents=True)
    active_csv = export_root / "amo_export_ready_ru.csv"
    _write_csv(
        active_csv,
        [
            {
                "Телефон клиента": "+79000000001",
                "AMO contact IDs": "111",
                "AMO lead IDs": "lead1",
                "Тип последнего свежего звонка": "sales_call",
                "Краткая история общения": "История клиента",
                "Краткое резюме последнего свежего звонка": "Сводка",
                "Следующий шаг": "Отправить материалы",
                "Приоритет лида": "warm",
                "Вероятность продажи, %": "65",
                "CRM writeback policy": "live_update_ready",
                "CRM writeback blockers": "",
                "AMO entity policy": "update_existing_single_amo_contact",
                "Готово к записи в AMO": "Да",
                "Причина статуса AMO": "готово",
            }
        ],
    )
    export_summary = export_root / "summary.json"
    _write_json(export_summary, {"tenant_config": {"path": str(project / "tenant_config.json")}})
    _write_json(project / "tenant_config.json", {"schema_version": "tenant_config_v1", "tenant_id": "test"})
    _write_json(
        stable / "CURRENT_RUNTIME.json",
        {
            "paths": {
                "amo_export_ready_csv": str(active_csv),
                "active_export_summary": str(export_summary),
                "stage15_summary": str(stable / "stage15" / "summary.json"),
            }
        },
    )
    _write_json(stable / "stage15" / "summary.json", {"passed": True, "readiness": {"crm_quality_writeback_ready": True}})
    _write_csv(
        duplicate_root / "post_merge_recheck_input_ru.csv",
        [
            {
                "Телефон клиента": "79000000001",
                "AMO contact IDs": "111",
                "Manual resolution id": "row0001_79000000001",
            }
        ],
    )
    _write_csv(
        duplicate_root / "candidate_contacts.csv",
        [
            {"resolution_id": "row0001_79000000001", "phone": "79000000001", "candidate_contact_id": "111"},
            {"resolution_id": "row0001_79000000001", "phone": "79000000001", "candidate_contact_id": "112"},
        ],
    )
    _write_csv(duplicate_root / "duplicate_merge_queue.csv", [{"resolution_id": "row0001_79000000001", "phone": "79000000001"}])
    return project


def _fixture_report(project: Path, *, contact_id: str) -> Path:
    report = project / "stable_runtime" / "amocrm_runtime" / "contact_writebacks" / "20260511T000000Z"
    report.mkdir(parents=True)
    input_csv = project / "stable_runtime" / "amo_duplicate_resolution_20260511_v1" / "post_merge_recheck_input_ru.csv"
    _write_json(
        report / "contact_writeback_summary.json",
        {"mode": "dry_run", "live_write": False, "input": str(input_csv), "total_rows": 1, "dry_run": 1, "skipped": 0, "failed": 0},
    )
    _write_csv(
        report / "contact_writeback_report.csv",
        [{"row_index": "1", "mode": "dry_run", "phone": "79000000001", "status": "dry_run", "reason": "live_write_not_confirmed", "contact_id": contact_id}],
    )
    return report


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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
