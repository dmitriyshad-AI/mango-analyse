from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from mango_mvp.productization.product_api import ProductApiFacade, build_product_api_readiness_report
from mango_mvp.productization.product_db import initialize_product_db
from scripts import mango_office_product_api_readiness


def test_product_api_facade_returns_read_only_endpoints(tmp_path: Path) -> None:
    product_root = tmp_path / "product_appliance"
    product_db = product_root / "mango_product_appliance.sqlite"
    initialize_product_db(product_db, product_root)

    api = ProductApiFacade(product_root=product_root, product_db_path=product_db, workspace_root=Path.cwd())

    dashboard = api.dashboard_summary()
    capture = api.capture_recent(limit=10)
    scheduler = api.scheduler_runs(limit=10)
    scheduler_health = api.scheduler_health()
    scheduler_control = api.scheduler_control_plane()
    lifecycle = api.lifecycle_readiness()
    crm_mapping = api.crm_mapping_preview()
    asr_gate = api.asr_gate_status()
    writeback = api.writeback_previews()
    queue = api.processing_queue(limit=10)
    knowledge = api.knowledge_playbook()
    operator_status = api.operator_runtime_status()
    manual_resolution = api.manual_resolution_status()
    waiting_work = api.waiting_autonomous_work_status()
    dry_run_readiness = api.writeback_dry_run_readiness()
    settings = api.settings_adapters()
    appliance = api.appliance_dashboard()

    assert dashboard["endpoint"] == "GET /dashboard/summary"
    assert dashboard["actions"]["write_runtime_db"] is False
    assert capture["items"] == []
    assert scheduler["items"] == []
    assert scheduler_health["endpoint"] == "GET /scheduler/health"
    assert scheduler_control["endpoint"] == "GET /scheduler/control-plane"
    assert lifecycle["mode"] == "waiting_for_lifecycle_report"
    assert crm_mapping["endpoint"] == "GET /crm/mapping-preview"
    assert asr_gate["stage25"]["run_asr"] is False
    assert asr_gate["stage25"]["dispatch_allowed"] is False
    assert writeback["write_crm"] is False
    assert writeback["preview"]["summary"]["write_crm"] is False
    assert queue["queue_policy"]["auto_trigger_enabled"] is False
    assert operator_status["endpoint"] == "GET /operator/status"
    assert manual_resolution["endpoint"] == "GET /manual-resolution/status"
    assert waiting_work["endpoint"] == "GET /waiting-work/status"
    assert waiting_work["write_crm"] is False
    assert dry_run_readiness["endpoint"] == "GET /writeback/dry-run-readiness"
    assert dry_run_readiness["live_write_allowed_now"] is False
    assert knowledge["current_mode"] == "schema_only"
    assert settings["adapters"]["database"]["profile"] == "sqlite_single_writer_v1"
    assert appliance["schema_version"] == "appliance_dashboard_v1"
    assert appliance["endpoint"] == "GET /dashboard/appliance"
    assert appliance["actions"]["write_crm"] is False
    assert appliance["safety"]["run_asr"] is False
    assert "capture" in appliance["panels"]
    assert "lifecycle" in appliance["panels"]
    assert "leadership_snapshot" in appliance["panels"]
    assert "stage_rollout" in appliance["panels"]
    assert "cleanup" in appliance["panels"]
    assert appliance["panels"]["scheduler"]["health"]["endpoint"] == "GET /scheduler/health"


def test_product_api_operator_status_uses_latest_artifact(tmp_path: Path) -> None:
    product_root = tmp_path / "product_appliance"
    product_db = product_root / "mango_product_appliance.sqlite"
    initialize_product_db(product_db, product_root)
    old_status = tmp_path / "stable_runtime" / "operator_status_20260510_v1" / "operator_status.json"
    latest_status = tmp_path / "stable_runtime" / "operator_status_20260511_v1" / "operator_status.json"
    old_status.parent.mkdir(parents=True)
    latest_status.parent.mkdir(parents=True)
    old_status.write_text(json.dumps({"summary": {"marker": "old"}}, ensure_ascii=False), encoding="utf-8")
    latest_status.write_text(json.dumps({"summary": {"marker": "latest"}}, ensure_ascii=False), encoding="utf-8")

    api = ProductApiFacade(product_root=product_root, product_db_path=product_db, workspace_root=tmp_path)
    status = api.operator_runtime_status()

    assert status["status"] == "loaded"
    assert status["summary"]["marker"] == "latest"
    assert status["actions"]["write_crm"] is False


def test_product_api_readiness_report_writes_json(tmp_path: Path) -> None:
    product_root = tmp_path / "product_appliance"
    product_db = product_root / "mango_product_appliance.sqlite"
    initialize_product_db(product_db, product_root)
    out = product_root / "product_api_readiness" / "report.json"

    report = build_product_api_readiness_report(
        product_root=product_root,
        product_db_path=product_db,
        out_path=out,
        workspace_root=Path.cwd(),
    )

    assert report["summary"]["validation_ok"] is True
    assert report["summary"]["read_only"] is True
    assert sorted(report["endpoints"]) == [
        "appliance_dashboard",
        "asr_gate_status",
        "capture_recent",
        "crm_mapping_preview",
        "dashboard_summary",
        "knowledge_playbook",
        "lifecycle_readiness",
        "manual_resolution_status",
        "operator_runtime_status",
        "processing_queue",
        "scheduler_control_plane",
        "scheduler_health",
        "scheduler_runs",
        "settings_adapters",
        "waiting_autonomous_work_status",
        "writeback_dry_run_readiness",
        "writeback_previews",
    ]
    assert json.loads(out.read_text(encoding="utf-8"))["summary"]["endpoints"] == 17


def test_product_api_readiness_blocks_missing_product_db(tmp_path: Path) -> None:
    product_root = tmp_path / "product_appliance"
    product_root.mkdir()

    report = build_product_api_readiness_report(
        product_root=product_root,
        product_db_path=product_root / "mango_product_appliance.sqlite",
        workspace_root=Path.cwd(),
    )

    assert report["summary"]["report_generated_ok"] is True
    assert report["summary"]["validation_ok"] is False
    assert report["summary"]["product_db_present"] is False
    assert "product_db_missing" in report["summary"]["blocked_reasons"]


def test_product_api_readiness_blocks_empty_sqlite_db(tmp_path: Path) -> None:
    product_root = tmp_path / "product_appliance"
    product_root.mkdir()
    product_db = product_root / "mango_product_appliance.sqlite"
    sqlite3.connect(product_db).close()

    report = build_product_api_readiness_report(
        product_root=product_root,
        product_db_path=product_db,
        workspace_root=Path.cwd(),
    )

    assert report["summary"]["validation_ok"] is False
    assert report["summary"]["product_db_present"] is True
    assert "product_db_invalid" in report["summary"]["blocked_reasons"]
    assert report["endpoints"]["dashboard_summary"]["summary"]["blocked"] > 0


def test_product_api_refuses_missing_db_outside_product_root(tmp_path: Path) -> None:
    product_root = tmp_path / "product_appliance"
    product_root.mkdir()

    with pytest.raises(ValueError, match="product DB"):
        build_product_api_readiness_report(
            product_root=product_root,
            product_db_path=tmp_path / "outside" / "mango_product_appliance.sqlite",
            workspace_root=Path.cwd(),
        )


def test_product_api_refuses_stable_runtime_paths(tmp_path: Path) -> None:
    product_root = tmp_path / "product_appliance"
    product_root.mkdir()

    with pytest.raises(ValueError, match="stable_runtime"):
        ProductApiFacade(
            product_root=product_root,
            product_db_path=tmp_path / "stable_runtime" / "mango_product_appliance.sqlite",
            workspace_root=Path.cwd(),
        )


def test_product_api_cli_writes_report(tmp_path: Path) -> None:
    product_root = tmp_path / "product_appliance"
    product_db = product_root / "mango_product_appliance.sqlite"
    initialize_product_db(product_db, product_root)
    out = product_root / "product_api_readiness" / "report.json"

    rc = mango_office_product_api_readiness.main(
        [
            "--product-root",
            str(product_root),
            "--product-db",
            str(product_db),
            "--out",
            str(out),
        ]
    )

    saved = json.loads(out.read_text(encoding="utf-8"))
    assert rc == 0
    assert saved["summary"]["read_only"] is True
    assert saved["safety"]["run_asr"] is False
