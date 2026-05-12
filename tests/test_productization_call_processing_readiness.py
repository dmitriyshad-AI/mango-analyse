from __future__ import annotations

import json
from pathlib import Path

from mango_mvp.productization.call_processing_readiness import build_call_processing_readiness_report
from scripts import mango_office_call_processing_readiness


def test_call_processing_readiness_passes_green_current_chain(tmp_path: Path) -> None:
    project = build_project_fixture(tmp_path)

    report = build_call_processing_readiness_report(project_root=project)

    assert report["summary"]["validation_ok"] is True
    assert report["summary"]["canonical_missing_asr"] == 0
    assert report["summary"]["canonical_missing_ra"] == 0
    assert report["summary"]["safe_writeback_pending_rows"] == 0
    assert report["summary"]["stage1_writeback_complete"] is True
    assert report["safety"]["write_crm"] is False
    assert all(gate["passed"] for gate in report["gates"] if gate["severity"] == "block")


def test_call_processing_readiness_blocks_missing_ra(tmp_path: Path) -> None:
    project = build_project_fixture(tmp_path)
    canonical_summary = project / "stable_runtime" / "canonical_master_current" / "summary.json"
    payload = json.loads(canonical_summary.read_text(encoding="utf-8"))
    payload["missing_full_ra_actionable"] = 3
    canonical_summary.write_text(json.dumps(payload), encoding="utf-8")

    report = build_call_processing_readiness_report(project_root=project)

    assert report["summary"]["validation_ok"] is False
    assert any(gate["gate"] == "NO_MISSING_ASR_OR_RA" and not gate["passed"] for gate in report["gates"])


def test_call_processing_readiness_blocks_quality_gate_input_drift(tmp_path: Path) -> None:
    project = build_project_fixture(tmp_path)
    crm_summary = project / "stable_runtime" / "crm_writeback_quality_gate_current" / "summary.json"
    payload = json.loads(crm_summary.read_text(encoding="utf-8"))
    payload["input"] = str(project / "stable_runtime" / "old_export" / "amo_export_ready_ru.csv")
    crm_summary.write_text(json.dumps(payload), encoding="utf-8")

    report = build_call_processing_readiness_report(project_root=project)

    assert report["summary"]["validation_ok"] is False
    assert any(gate["gate"] == "CRM_GATE_INPUT_MATCHES_EXPORT" and not gate["passed"] for gate in report["gates"])


def test_call_processing_readiness_cli_writes_report(tmp_path: Path) -> None:
    project = build_project_fixture(tmp_path)
    out = project / "stable_runtime" / "call_processing_readiness" / "report.json"

    rc = mango_office_call_processing_readiness.main(["--project-root", str(project), "--out", str(out)])

    assert rc == 0
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["summary"]["processing_pipeline_ready"] is True


def build_project_fixture(tmp_path: Path) -> Path:
    project = tmp_path / "repo"
    stable = project / "stable_runtime"
    export_root = stable / "sales_master_export_current"
    canonical_root = stable / "canonical_master_current"
    stage15_root = stable / "stage15_current"
    crm_root = stable / "crm_writeback_quality_gate_current"
    queue_root = stable / "amo_writeback_queue_current"
    readback_a = stable / "amocrm_runtime" / "contact_writebacks" / "run-a" / "readback_gate_expected20"
    readback_b = stable / "amocrm_runtime" / "contact_writebacks" / "run-b" / "readback_gate_expected20"
    quarantine = project / "_cleanup_quarantine_20260510_stage2"
    for path in [export_root, canonical_root, stage15_root, crm_root, queue_root, readback_a, readback_b, quarantine]:
        path.mkdir(parents=True, exist_ok=True)

    canonical_db = canonical_root / "canonical_calls_master.db"
    canonical_db.write_text("", encoding="utf-8")
    amo_csv = export_root / "amo_export_ready_ru.csv"
    amo_csv.write_text("phone\n", encoding="utf-8")
    (stable / "CANONICAL_EXPORT.txt").write_text("sales_master_export_current\n", encoding="utf-8")
    (quarantine / "MANIFEST.csv").write_text("original_path,quarantine_path\n", encoding="utf-8")

    write_json(
        canonical_root / "summary.json",
        {
            "actionable_source_audio": 10,
            "missing_asr_actionable": 0,
            "missing_full_ra_actionable": 0,
            "validation": {"passed": True},
            "canonical_db": {"path": str(canonical_db), "passed": True},
        },
    )
    write_json(
        stage15_root / "summary.json",
        {"passed": True, "readiness": {"crm_quality_writeback_ready": True}},
    )
    write_json(
        export_root / "summary.json",
        {
            "canonical_db": str(canonical_db),
            "stage15_summary": str(stage15_root / "summary.json"),
            "amo_export_ready_rows": 3,
            "output_files": {"amo_export_ready_csv": str(amo_csv)},
        },
    )
    write_json(
        crm_root / "summary.json",
        {
            "input": str(amo_csv),
            "passed": True,
            "blocking_rows": 0,
            "population_recall": {"passed_for_live": True},
            "crm_text_quality": {"passed_for_live": True},
        },
    )
    write_json(
        queue_root / "summary.json",
        {
            "input_csv": str(amo_csv),
            "bucket_counts": {
                "ready_single_contact_not_written": 0,
                "already_written": 3,
            },
        },
    )
    for path in [readback_a / "summary.json", readback_b / "summary.json"]:
        write_json(
            path,
            {
                "passed": True,
                "evaluated_rows": 20,
                "expected_evaluated": 20,
                "expected_count_mismatch": False,
            },
        )
    return project


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
