from __future__ import annotations

import csv
import json
from pathlib import Path

from mango_mvp.productization.runtime_artifact_index import build_runtime_artifact_index


def test_runtime_artifact_index_classifies_current_blocked_and_legacy(tmp_path: Path) -> None:
    project = tmp_path / "project"
    stable = project / "stable_runtime"
    stable.mkdir(parents=True)

    active_export = stable / "sales_master_export_20260510_after_quality_backfill_v5_crm_text_quality_strict"
    active_export.mkdir()
    _write_json(active_export / "summary.json", {"passed": True})
    queue = stable / "amo_writeback_queue_20260510_v2_production"
    queue.mkdir()
    _write_json(queue / "summary.json", {"passed": True})
    blocked = stable / "amo_duplicate_after_staff_done_20260511_v1"
    blocked.mkdir()
    _write_json(blocked / "summary.json", {"status": "waiting_for_staff_done_and_recheck", "blocked_rows": 13})
    legacy = stable / "sales_master_export_20260424_legacy"
    legacy.mkdir()
    _write_json(legacy / "summary.json", {"passed": True})
    audit_only = stable / "claude_stage15_v3"
    audit_only.mkdir()
    _write_json(audit_only / "summary.json", {"passed": True})
    invalid = stable / "amocrm_runtime" / "contact_writebacks" / "bad"
    invalid.mkdir(parents=True)
    (invalid / "summary.json").write_text("{bad", encoding="utf-8")

    _write_json(
        stable / "CURRENT_RUNTIME.json",
        {
            "paths": {
                "active_export_root": str(active_export),
                "amo_queue_summary": str(queue / "summary.json"),
            }
        },
    )
    cleanup = stable / "project_cleanup_manifest_20260511_v1"
    cleanup.mkdir()
    with (cleanup / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["path"])
        writer.writeheader()
        writer.writerow({"path": str(legacy.relative_to(project))})

    out = stable / "runtime_artifact_index"
    report = build_runtime_artifact_index(project_root=project, out_root=out)

    by_path = {entry["path"]: entry for entry in report["entries"]}
    assert by_path[str(active_export.relative_to(project))]["category"] == "active_current"
    assert by_path[str(queue.relative_to(project))]["category"] == "active_current"
    assert by_path[str(blocked.relative_to(project))]["category"] == "blocked"
    assert by_path[str(legacy.relative_to(project))]["category"] == "legacy_candidate"
    assert by_path[str(audit_only.relative_to(project))]["category"] == "audit_only"
    assert by_path[str(invalid.relative_to(project))]["category"] == "blocked"
    assert by_path[str(invalid.relative_to(project))]["valid_json"] is False
    assert report["summary"]["read_only_scan"] is True
    assert (out / "artifact_index.json").exists()
    assert (out / "artifact_index.csv").exists()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
