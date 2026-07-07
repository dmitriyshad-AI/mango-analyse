from __future__ import annotations

import json
from pathlib import Path

from scripts.publish_snapshot import build_snapshot, preflight, reader_smoke
from scripts.publish_snapshot.common import backup_plan_report, classify_publish_worktree_status, copy_verified
from tests.test_customer_timeline_read_api import seed_timeline_db


def _config(tmp_path: Path, prod: Path, staging: Path) -> Path:
    cfg = {
        "schema_version": "publish_snapshot_config_v1",
        "package_name": "test",
        "tenant_id": "foton",
        "staging_db": str(staging),
        "prod_db": str(prod),
        "snapshot_root": str(tmp_path / "snapshots"),
        "backup_root": str(tmp_path / "prod_backups"),
        "backup_async_copy_root": str(tmp_path / "openclaw_backups"),
        "required_free_copies": 1,
        "count_tables": ["customer_identities", "timeline_events", "bot_context_chunks"],
        "control_customers": [{"customer_id": "customer:0", "expected_found": True}],
        "readers": [],
    }
    path = tmp_path / "publish_config.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


def test_build_snapshot_vacuum_into_and_manifest_then_reader_smoke(tmp_path: Path) -> None:
    prod_dir = tmp_path / "prod"
    staging_dir = tmp_path / "staging"
    prod_dir.mkdir()
    staging_dir.mkdir()
    prod, _prod_customer = seed_timeline_db(prod_dir)
    staging, staging_customer = seed_timeline_db(staging_dir)
    cfg = _config(tmp_path, prod, staging)
    payload = json.loads(cfg.read_text(encoding="utf-8"))
    payload["control_customers"] = [
        {
            "customer_id": staging_customer,
            "expected_found": True,
            "expected_counts": {
                "events_total": 1,
                "bot_context_chunks_total": 2,
                "allowed_chunks": 1,
                "review_required_chunks": 1,
                "derived_signals_total": 1,
            },
        }
    ]
    cfg.write_text(json.dumps(payload), encoding="utf-8")

    report, ok = build_snapshot.build_snapshot(cfg, execute=True, snapshot_name="prod_test")

    assert ok is True
    snapshot_db = Path(report["snapshot_db"])
    manifest = json.loads((snapshot_db.parent / "build_manifest.json").read_text(encoding="utf-8"))
    assert manifest["quick_check"] == "ok"
    assert manifest["counts"]["timeline_events"] >= 1

    smoke_report, smoke_ok = reader_smoke.smoke(cfg, snapshot_db=snapshot_db)
    assert smoke_ok is True
    assert smoke_report["internal_control_customers"][0]["found"] is True
    assert smoke_report["internal_control_customers"][0]["count_mismatches"] == {}

    payload["control_customers"][0]["expected_counts"]["events_total"] = 999
    cfg.write_text(json.dumps(payload), encoding="utf-8")
    mismatch_report, mismatch_ok = reader_smoke.smoke(cfg, snapshot_db=snapshot_db)
    assert mismatch_ok is False
    assert mismatch_report["internal_control_customers"][0]["count_mismatches"]["events_total"]["actual"] == 1


def test_preflight_blocks_dirty_reader_worktree(tmp_path: Path) -> None:
    prod_dir = tmp_path / "prod"
    staging_dir = tmp_path / "staging"
    prod_dir.mkdir()
    staging_dir.mkdir()
    prod, _prod_customer = seed_timeline_db(prod_dir)
    staging, _staging_customer = seed_timeline_db(staging_dir)
    cfg = json.loads(_config(tmp_path, prod, staging).read_text(encoding="utf-8"))
    cfg["readers"] = [{"name": "reader", "worktree": str(tmp_path), "stop_command": ["true"], "start_command": ["true"]}]
    cfg_path = tmp_path / "publish_config_dirty.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    report, ok = preflight.build_report(cfg_path)

    assert ok is False
    assert report["readers"][0]["git_status_clean"] is False


def test_publish_worktree_status_allows_data_untracked() -> None:
    status = (
        "?? product_data/telegram_dynamic_test_sets/sample.jsonl\n"
        "?? tasks/_running/task.md\n"
        "?? README.local.md\n"
    )

    report = classify_publish_worktree_status(status)

    assert report["clean_for_publish"] is True
    assert report["tracked_blockers"] == []
    assert report["untracked_code_blockers"] == []
    assert report["untracked_allowed"] == [
        "?? product_data/telegram_dynamic_test_sets/sample.jsonl",
        "?? tasks/_running/task.md",
        "?? README.local.md",
    ]


def test_publish_worktree_status_blocks_code_untracked_and_tracked_changes() -> None:
    status = (
        " M docs/DECISIONS_LOG.md\n"
        "?? src/mango_mvp/new_module.py\n"
        "?? scripts/new_tool.py\n"
        "?? product_data/telegram_dynamic_test_sets/sample.jsonl\n"
    )

    report = classify_publish_worktree_status(status)

    assert report["clean_for_publish"] is False
    assert report["tracked_blockers"] == [" M docs/DECISIONS_LOG.md"]
    assert report["untracked_code_blockers"] == ["?? src/mango_mvp/new_module.py", "?? scripts/new_tool.py"]
    assert report["untracked_allowed"] == ["?? product_data/telegram_dynamic_test_sets/sample.jsonl"]


def test_backup_plan_allows_same_disk_with_verified_async_copy(tmp_path: Path) -> None:
    source = tmp_path / "prod.sqlite"
    source.write_text("backup-source", encoding="utf-8")

    report = backup_plan_report(
        source,
        tmp_path / "prod_backups",
        tmp_path / "openclaw_backups",
        required_bytes=source.stat().st_size,
    )
    first = copy_verified(source, tmp_path / "prod_backups" / "copy.sqlite")
    second = copy_verified(Path(first["target"]), tmp_path / "openclaw_backups" / "copy.sqlite")

    assert report["ok"] is True
    assert report["policy"] == "same_disk_verified_backup_plus_yandex_async_copy"
    assert first["source_sha256"] == first["target_sha256"]
    assert second["source_sha256"] == second["target_sha256"]
