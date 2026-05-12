from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from mango_mvp.productization.project_cleanup_manifest import (
    MANIFEST_COLUMNS,
    ProjectCleanupManifestConfig,
    build_project_cleanup_manifest,
)
from scripts import build_project_cleanup_manifest as cli


GENERATED_AT = datetime(2026, 5, 11, 9, 0, tzinfo=timezone.utc)


def test_cleanup_manifest_excludes_current_runtime_paths_and_fresh_audits(tmp_path: Path) -> None:
    project = _fixture_project(tmp_path)
    out_root = project / "stable_runtime" / "project_cleanup_manifest_20260511_v1"

    summary = build_project_cleanup_manifest(
        ProjectCleanupManifestConfig(
            project_root=project,
            out_root=out_root,
            generated_at=GENERATED_AT,
            fresh_audit_days=1,
        )
    )

    assert summary["safety"] == {
        "read_only_scan": True,
        "deletes_files": False,
        "moves_files": False,
        "quarantines_files": False,
        "writes_only_report_artifacts": True,
        "destructive_operations_available": False,
    }
    assert summary["candidate_rows"] > 0
    assert (out_root / "manifest.csv").exists()
    assert (out_root / "manifest.json").exists()
    assert (out_root / "summary.json").exists()

    rows = json.loads((out_root / "manifest.json").read_text(encoding="utf-8"))
    by_path = {row["candidate_path"]: row for row in rows}
    assert "stable_runtime/sales_master_export_20260509_old/summary.json" not in by_path
    assert "stable_runtime/sales_master_export_20260511_active" not in by_path
    assert "stable_runtime/canonical_master_20260511_active" not in by_path
    assert "stable_runtime/crm_writeback_quality_gate_20260511_active" not in by_path
    assert "stable_runtime/amo_writeback_queue_20260511_active" not in by_path
    assert "stable_runtime/CURRENT_RUNTIME.json" not in by_path
    assert "audits/_inbox/amo_manual_resolution_operator_status_20260511_v1" not in by_path
    assert "audits/_results/2026-05-10_amo_stage40_readback" not in by_path

    old_export = by_path["stable_runtime/sales_master_export_20260509_old"]
    assert old_export["category"] == "superseded_strict_export"
    assert old_export["replacement_path"] == "stable_runtime/sales_master_export_20260511_active"
    assert old_export["safe_to_quarantine"] is True
    assert old_export["requires_human_review"] is True

    old_audit = by_path["audits/_inbox/legacy_audit_20260501_v1"]
    assert old_audit["category"] == "historical_audit_pack"
    assert old_audit["safe_to_quarantine"] is True
    assert old_audit["requires_human_review"] is True

    local_junk = by_path[".DS_Store"]
    assert local_junk["category"] == "local_os_metadata"
    assert local_junk["safe_to_quarantine"] is True
    assert local_junk["requires_human_review"] is False

    review_artifact = by_path["stable_runtime/external_m1_batch_20260501"]
    assert review_artifact["category"] == "runtime_manual_review_required"
    assert review_artifact["safe_to_quarantine"] is False
    assert review_artifact["requires_human_review"] is True


def test_cleanup_manifest_csv_has_required_columns_and_cli_writes_default_shape(tmp_path: Path) -> None:
    project = _fixture_project(tmp_path)
    out_root = project / "stable_runtime" / "cleanup_cli"

    rc = cli.main(
        [
            "--project-root",
            str(project),
            "--out-root",
            str(out_root),
            "--generated-at",
            GENERATED_AT.isoformat(),
            "--fresh-audit-days",
            "1",
        ]
    )

    assert rc == 0
    rows = _read_csv(out_root / "manifest.csv")
    assert rows
    assert list(rows[0].keys()) == MANIFEST_COLUMNS
    by_path = {row["candidate_path"]: row for row in rows}
    assert by_path["stable_runtime/crm_writeback_quality_gate_20260509_old"]["category"] == (
        "superseded_crm_writeback_quality_gate"
    )
    assert by_path["stable_runtime/amo_writeback_queue_20260509_old"]["safe_to_quarantine"] == "true"
    assert by_path["stable_runtime/external_m1_batch_20260501"]["safe_to_quarantine"] == "false"

    summary = json.loads((out_root / "summary.json").read_text(encoding="utf-8"))
    assert summary["outputs"]["manifest_csv"] == str(out_root / "manifest.csv")
    assert summary["safety"]["deletes_files"] is False
    assert summary["safety"]["moves_files"] is False


def _fixture_project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    stable = project / "stable_runtime"
    stable.mkdir(parents=True)
    (project / ".DS_Store").write_text("local metadata", encoding="utf-8")
    (project / "_external_handoffs").mkdir()

    active_export = stable / "sales_master_export_20260511_active"
    active_export.mkdir()
    active_export_summary = active_export / "summary.json"
    active_export_summary.write_text("{}", encoding="utf-8")
    active_export_csv = active_export / "amo_export_ready_ru.csv"
    active_export_csv.write_text("phone\n+79000000000\n", encoding="utf-8")

    old_export = stable / "sales_master_export_20260509_old"
    old_export.mkdir()
    (old_export / "summary.json").write_text("{}", encoding="utf-8")

    active_canonical = stable / "canonical_master_20260511_active"
    active_canonical.mkdir()
    active_canonical_db = active_canonical / "canonical_calls_master.db"
    active_canonical_db.write_bytes(b"sqlite")
    active_canonical_summary = active_canonical / "summary.json"
    active_canonical_summary.write_text("{}", encoding="utf-8")
    (stable / "canonical_master_20260509_old").mkdir()

    active_stage15 = stable / "transcript_quality_stage15_export_gate_20260511_active"
    active_stage15.mkdir()
    active_stage15_summary = active_stage15 / "summary.json"
    active_stage15_summary.write_text("{}", encoding="utf-8")
    (stable / "transcript_quality_stage15_export_gate_20260509_old").mkdir()

    active_quality = stable / "crm_writeback_quality_gate_20260511_active"
    active_quality.mkdir()
    active_quality_summary = active_quality / "summary.json"
    active_quality_summary.write_text("{}", encoding="utf-8")
    (stable / "crm_writeback_quality_gate_20260509_old").mkdir()

    active_queue = stable / "amo_writeback_queue_20260511_active"
    active_queue.mkdir()
    active_queue_summary = active_queue / "summary.json"
    active_queue_summary.write_text("{}", encoding="utf-8")
    (stable / "amo_writeback_queue_20260509_old").mkdir()

    (stable / "external_m1_batch_20260501").mkdir()
    (stable / "project_inventory_20260509_v1").mkdir()
    (stable / "CURRENT_RUNTIME.json").write_text(
        json.dumps(
            {
                "paths": {
                    "active_export_root": str(active_export),
                    "active_export_summary": str(active_export_summary),
                    "amo_export_ready_csv": str(active_export_csv),
                    "canonical_db": str(active_canonical_db),
                    "canonical_summary": str(active_canonical_summary),
                    "stage15_summary": str(active_stage15_summary),
                    "crm_quality_summary": str(active_quality_summary),
                    "amo_queue_summary": str(active_queue_summary),
                    "canonical_export_pointer": str(stable / "CANONICAL_EXPORT.txt"),
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (stable / "CANONICAL_EXPORT.txt").write_text(active_export.name, encoding="utf-8")

    old_inbox = project / "audits" / "_inbox" / "legacy_audit_20260501_v1"
    old_inbox.mkdir(parents=True)
    (old_inbox / "README.md").write_text("old", encoding="utf-8")
    fresh_inbox = project / "audits" / "_inbox" / "amo_manual_resolution_operator_status_20260511_v1"
    fresh_inbox.mkdir()
    (fresh_inbox / "README.md").write_text("fresh", encoding="utf-8")
    fresh_result = project / "audits" / "_results" / "2026-05-10_amo_stage40_readback"
    fresh_result.mkdir(parents=True)
    (fresh_result / "CLAUDE_REAUDIT_RESULT.md").write_text("fresh", encoding="utf-8")
    old_result = project / "audits" / "_results" / "2026-05-01_old_result"
    old_result.mkdir()
    (old_result / "result.md").write_text("old", encoding="utf-8")

    return project


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))
