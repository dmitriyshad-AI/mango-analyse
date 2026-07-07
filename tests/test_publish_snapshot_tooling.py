from __future__ import annotations

import json
from pathlib import Path

from scripts.publish_snapshot import build_snapshot, preflight, reader_smoke
from tests.test_customer_timeline_read_api import seed_timeline_db


def _config(tmp_path: Path, prod: Path, staging: Path) -> Path:
    cfg = {
        "schema_version": "publish_snapshot_config_v1",
        "package_name": "test",
        "tenant_id": "foton",
        "staging_db": str(staging),
        "prod_db": str(prod),
        "snapshot_root": str(tmp_path / "snapshots"),
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
    payload["control_customers"] = [{"customer_id": staging_customer, "expected_found": True}]
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
