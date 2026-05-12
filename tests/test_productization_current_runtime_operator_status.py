from __future__ import annotations

import csv
import json
from pathlib import Path

from mango_mvp.productization.current_runtime import build_current_runtime_contract
from mango_mvp.productization.operator_status import build_operator_status


def test_current_runtime_contract_binds_post_backfill_artifacts(tmp_path: Path) -> None:
    project = _fixture_project(tmp_path, ready_rows=0)
    out = project / "stable_runtime" / "CURRENT_RUNTIME.json"

    contract = build_current_runtime_contract(project_root=project, out_path=out)

    assert out.exists()
    assert contract["summary"]["validation_ok"] is True
    assert contract["summary"]["active_export_name"] == "sales_master_export_20260510_after_quality_backfill_v5_crm_text_quality_strict"
    assert contract["summary"]["canonical_missing_asr"] == 0
    assert contract["summary"]["canonical_missing_ra"] == 0
    assert contract["paths"]["amo_queue_summary"].endswith("amo_writeback_queue_current/summary.json")
    gates = {gate["gate"]: gate["passed"] for gate in contract["gates"]}
    assert gates["ACTIVE_EXPORT_NOT_LEGACY_APRIL"] is True
    assert gates["CALL_PROCESSING_READINESS_GREEN"] is True


def test_current_runtime_contract_blocks_legacy_april_pointer(tmp_path: Path) -> None:
    project = _fixture_project(tmp_path, ready_rows=0, export_name="sales_master_export_20260424_legacy")

    contract = build_current_runtime_contract(project_root=project, out_path=project / "stable_runtime" / "CURRENT_RUNTIME.json")

    gates = {gate["gate"]: gate["passed"] for gate in contract["gates"]}
    assert contract["summary"]["validation_ok"] is False
    assert gates["ACTIVE_EXPORT_NOT_LEGACY_APRIL"] is False


def test_operator_status_writes_dashboard_and_russian_queue(tmp_path: Path) -> None:
    project = _fixture_project(tmp_path, ready_rows=0)
    runtime_path = project / "stable_runtime" / "CURRENT_RUNTIME.json"
    build_current_runtime_contract(project_root=project, out_path=runtime_path)
    out_root = project / "stable_runtime" / "operator_status"

    status = build_operator_status(project_root=project, runtime_contract_path=runtime_path, out_root=out_root)

    assert status["summary"]["runtime_validation_ok"] is True
    assert status["summary"]["crm_writeback_live_allowed_now"] is False
    assert status["summary"]["blocked_semantics"] == "runtime_blocked_plus_production_blocking_reasons"
    assert status["summary"]["contact_id_mismatch_blocked_rows"] == 1
    assert status["summary"]["queue_already_written_rows"] == 2
    assert status["summary"]["queue_manual_resolution_rows"] == 3
    assert status["summary"]["duplicate_merge_required_rows"] == 1
    assert status["summary"]["duplicate_contact_mismatch_rows"] == 1
    assert status["summary"]["duplicate_after_staff_done_status"] == "waiting_for_staff_done_and_recheck"
    assert status["summary"]["duplicate_after_staff_done_blocked_rows"] == 1
    assert status["summary"]["waiting_work_status"] == "prepared_safe_next_batches"
    assert status["summary"]["waiting_work_non_duplicate_candidate_rows"] == 1
    assert status["summary"]["waiting_work_refresh_candidate_rows"] == 4
    assert status["summary"]["waiting_work_readback_missing_rows"] == 2
    assert status["summary"]["waiting_work_live_write_allowed_now"] is False
    assert status["amo_production_loop"]["waiting_work_dry_run_allowed_now"] is True
    assert "no_ready_single_contact_rows_for_next_live_stage" in status["amo_production_loop"]["blocking_reasons"]
    assert "run_waiting_work_refresh_real_tunnel_dry_run" in {
        action["action"] for action in status["amo_production_loop"]["next_operator_actions"]
    }
    assert (out_root / "operator_status.json").exists()
    assert "Mango Analyse Operator Status" in (out_root / "operator_dashboard.html").read_text(encoding="utf-8")
    rows = list(csv.DictReader((out_root / "crm_queue_operator.csv").open("r", encoding="utf-8-sig")))
    assert {row["status_ru"] for row in rows} >= {
        "Уже записано и подтверждено предыдущими этапами",
        "Нужен ручной выбор AMO-контакта",
        "Заблокировано: AMO dry-run нашел другой contact_id",
        "Нужна проверка текста перед записью",
    }


def test_operator_status_allows_next_stage_only_when_ready_rows_exist(tmp_path: Path) -> None:
    project = _fixture_project(tmp_path, ready_rows=1)
    runtime_path = project / "stable_runtime" / "CURRENT_RUNTIME.json"
    build_current_runtime_contract(project_root=project, out_path=runtime_path)

    status = build_operator_status(project_root=project, runtime_contract_path=runtime_path, out_root=project / "stable_runtime" / "operator_status")

    assert status["summary"]["crm_writeback_live_allowed_now"] is False
    assert status["summary"]["queue_ready_rows"] == 1
    assert status["amo_production_loop"]["dry_run_allowed_now"] is True
    assert status["amo_production_loop"]["live_write_approval_required"] is True
    assert status["amo_production_loop"]["next_operator_actions"][0]["action"] == "prepare_next_live_stage"


def _fixture_project(tmp_path: Path, *, ready_rows: int, export_name: str = "sales_master_export_20260510_after_quality_backfill_v5_crm_text_quality_strict") -> Path:
    project = tmp_path / "project"
    stable = project / "stable_runtime"
    stable.mkdir(parents=True)
    (stable / "CANONICAL_EXPORT.txt").write_text(export_name, encoding="utf-8")

    export_root = stable / export_name
    export_root.mkdir()
    export_csv = export_root / "amo_export_ready_ru.csv"
    _write_csv(
        export_csv,
        [
            _source_row("ready_single_contact_not_written", "+79000000001"),
            _source_row("already_written", "+79000000002"),
            _source_row("needs_manager_review_multi_contact", "+79000000003"),
            _source_row("blocked_contact_id_mismatch", "+79000000004"),
            _source_row("needs_text_quality_review", "+79000000005"),
        ],
    )

    canonical_root = stable / "canonical_master_20260510_after_quality_backfill_v1"
    canonical_root.mkdir()
    canonical_db = canonical_root / "canonical_calls_master.db"
    canonical_db.write_bytes(b"sqlite-placeholder")
    _write_json(
        canonical_root / "summary.json",
        {
            "validation": {"passed": True},
            "canonical_db": {"path": str(canonical_db), "passed": True},
            "actionable_source_audio": 12,
            "missing_asr_actionable": 0,
            "missing_full_ra_actionable": 0,
        },
    )

    stage15_root = stable / "transcript_quality_stage15_export_gate_20260510_v11_frozen_gate"
    stage15_root.mkdir()
    stage15_summary = stage15_root / "summary.json"
    _write_json(stage15_summary, {"passed": True, "readiness": {"crm_quality_writeback_ready": True}})

    _write_json(
        export_root / "summary.json",
        {
            "canonical_db": str(canonical_db),
            "stage15_summary": str(stage15_summary),
            "amo_export_ready_rows": 5,
            "output_files": {"amo_export_ready_csv": str(export_csv)},
        },
    )

    crm_gate_root = stable / "crm_writeback_quality_gate_current"
    crm_gate_root.mkdir()
    _write_json(
        crm_gate_root / "summary.json",
        {
            "input": str(export_csv),
            "passed": True,
            "blocking_rows": 0,
            "population_recall": {"passed_for_live": True},
        },
    )

    queue_root = stable / "amo_writeback_queue_current"
    queue_root.mkdir()
    bucket_counts = {
        "ready_single_contact_not_written": ready_rows,
        "needs_manager_review_multi_contact": 1,
        "blocked_contact_id_mismatch": 1,
        "needs_text_quality_review": 1,
        "deferred_non_sales_or_service": 0,
        "already_written": 2 - ready_rows,
    }
    outputs = {"buckets": {}}
    for bucket, count in bucket_counts.items():
        rows = [_source_row(bucket, f"+790000001{idx}") for idx in range(count)]
        path = queue_root / f"{bucket}.csv"
        _write_csv(path, rows)
        outputs["buckets"][bucket] = {"csv": str(path), "xlsx": ""}
    _write_json(
        queue_root / "summary.json",
        {
            "input_csv": str(export_csv),
            "bucket_counts": bucket_counts,
            "outputs": outputs,
        },
    )

    after_staff_root = stable / "amo_duplicate_after_staff_done_20260511_v1"
    after_staff_root.mkdir()
    _write_json(
        after_staff_root / "summary.json",
        {
            "status": "waiting_for_staff_done_and_recheck",
            "candidate_rows": 0,
            "blocked_rows": 1,
            "outputs": {"candidate_csv": str(after_staff_root / "post_merge_live_candidates_ru.csv")},
        },
    )

    waiting_work_root = stable / "amo_waiting_autonomous_work_20260511_v1"
    waiting_work_root.mkdir()
    _write_json(
        waiting_work_root / "summary.json",
        {
            "status": "prepared_safe_next_batches",
            "counts": {
                "non_duplicate_live_candidate_rows": 1,
                "refresh_candidate_rows": 4,
                "readback_missing_rows": 2,
                "contact_id_mismatch_rows": 1,
                "already_written_rows": 5,
                "text_quality_review_rows": 3,
                "text_quality_cleared_rows": 3,
            },
            "outputs": {
                "text_quality_cleared_candidates_csv": str(waiting_work_root / "text_quality_cleared_candidates_ru.csv"),
                "already_written_refresh_candidates_csv": str(waiting_work_root / "already_written_refresh_candidates_ru.csv"),
                "readback_missing_written_rows_csv": str(waiting_work_root / "readback_missing_written_rows.csv"),
            },
            "next_actions": [
                {"action": "run_refresh_quality_gate_and_dry_run", "rows": 4},
                {"action": "run_readback_for_missing_written_rows", "rows": 2},
            ],
        },
    )

    readback_root = stable / "amocrm_runtime" / "contact_writebacks"
    for name in ("20260510T175140Z", "20260510T180418Z"):
        gate = readback_root / name / "readback_gate"
        gate.mkdir(parents=True)
        _write_json(gate / "summary.json", {"passed": True, "evaluated_rows": 20, "expected_evaluated": 20, "expected_count_mismatch": False})

    product_root = project / "_local_archive_mango_api_downloads_20260507" / "product_appliance"
    product_root.mkdir(parents=True)
    (product_root / "mango_product_appliance.sqlite").write_bytes(b"sqlite-placeholder")
    quarantine = project / "_cleanup_quarantine_20260510_stage2"
    quarantine.mkdir()
    (quarantine / "MANIFEST.csv").write_text("original_path,quarantine_path\n", encoding="utf-8")
    return project


def _source_row(bucket: str, phone: str) -> dict[str, str]:
    return {
        "queue_bucket": bucket,
        "queue_reason": bucket,
        "source_row_index": "1",
        "normalized_phone": phone,
        "source_amo_contact_ids": "123",
        "effective_contact_id": "123",
        "written_status": "written" if bucket == "already_written" else "",
        "dry_run_status": "dry_run" if bucket == "ready_single_contact_not_written" else "",
        "Телефон клиента": phone,
        "AMO contact IDs": "123",
        "ФИО родителя": "Иванов",
        "ФИО ребенка": "Петр",
        "Дата последнего свежего звонка": "2026-04-01",
        "Тип последнего свежего звонка": "sales_call",
        "Приоритет лида": "warm",
        "Вероятность продажи, %": "65",
        "Следующий шаг": "Позвонить",
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
