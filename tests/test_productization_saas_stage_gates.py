from __future__ import annotations

import json
from pathlib import Path

import pytest

from mango_mvp.productization.product_db import initialize_product_db
from mango_mvp.productization.saas_stage_gates import build_saas_stage_gates_report
from scripts import mango_office_saas_stage_gates


def test_saas_stage_gates_report_handles_missing_product_db(tmp_path: Path) -> None:
    product_root = tmp_path / "product_appliance"
    product_root.mkdir()
    out = product_root / "saas_stage_gates" / "report.json"

    report = build_saas_stage_gates_report(
        product_root=product_root,
        product_db_path=product_root / "mango_product_appliance.sqlite",
        out_path=out,
        workspace_root=Path.cwd(),
    )

    assert report["summary"]["report_generated_ok"] is True
    assert report["summary"]["validation_ok"] is False
    assert report["summary"]["stage_inputs_valid"] is False
    assert report["summary"]["saas_ready"] is False
    assert report["summary"]["stages_total"] == 9
    assert report["product_db_audit"]["present"] is False
    assert "product_db_missing" in report["stage_index"]["local_product_api_single_writer"]["blockers"]
    assert report["writeback_policy"]["write_crm"] is False
    assert report["safety"]["run_asr"] is False
    assert out.exists()


def test_saas_stage_gates_report_builds_surface_contracts_with_product_db(tmp_path: Path) -> None:
    product_root = tmp_path / "product_appliance"
    product_db = product_root / "mango_product_appliance.sqlite"
    initialize_product_db(product_db, product_root)
    (product_root / "asr_worker_sandbox_execution_request_stage25").mkdir(parents=True)
    (product_root / "asr_worker_sandbox_execution_request_stage25" / "asr_worker_sandbox_execution_request_stage25.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    live_dir = product_root / "raw_payload_archive" / "live_shadow_poll"
    live_dir.mkdir(parents=True)
    (live_dir / "live_shadow_poll_foton.jsonl").write_text("{}\n", encoding="utf-8")
    out = product_root / "saas_stage_gates" / "report.json"

    report = build_saas_stage_gates_report(
        product_root=product_root,
        product_db_path=product_db,
        out_path=out,
        workspace_root=Path.cwd(),
    )

    assert report["summary"]["validation_ok"] is True
    assert report["artifact_state"]["asr_stage25_request_present"] is True
    assert report["artifact_state"]["live_shadow_poll_archive_present"] is True
    assert report["api_surface_contract"]["schema_version"] == "product_api_surface_contract_v1"
    assert report["summary"]["stages_total"] == 9
    assert len(report["api_surface_contract"]["endpoints"]) == 8
    assert report["ui_surface_contract"]["blocked_actions"] == [
        "download_audio",
        "run_asr",
        "run_ra",
        "write_crm",
        "write_runtime_db",
    ]
    assert report["deployment_profile"]["profile"] == "sqlite_single_writer_v1"
    assert "multi_client_readiness" in report["stage_index"]


def test_saas_stage_gates_refuses_stable_runtime_output(tmp_path: Path) -> None:
    product_root = tmp_path / "product_appliance"
    product_root.mkdir()

    with pytest.raises(ValueError, match="stable_runtime"):
        build_saas_stage_gates_report(
            product_root=product_root,
            product_db_path=product_root / "mango_product_appliance.sqlite",
            out_path=tmp_path / "stable_runtime" / "report.json",
            workspace_root=Path.cwd(),
        )


def test_saas_stage_gates_refuses_missing_db_outside_product_root(tmp_path: Path) -> None:
    product_root = tmp_path / "product_appliance"
    product_root.mkdir()

    with pytest.raises(ValueError, match="product DB"):
        build_saas_stage_gates_report(
            product_root=product_root,
            product_db_path=tmp_path / "outside" / "mango_product_appliance.sqlite",
            out_path=product_root / "saas_stage_gates" / "report.json",
            workspace_root=Path.cwd(),
        )


def test_saas_stage_gates_cli_writes_report(tmp_path: Path) -> None:
    product_root = tmp_path / "product_appliance"
    product_db = product_root / "mango_product_appliance.sqlite"
    initialize_product_db(product_db, product_root)
    out = product_root / "saas_stage_gates" / "report.json"

    rc = mango_office_saas_stage_gates.main(
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
    assert saved["summary"]["stages_total"] == 9
    assert saved["summary"]["validation_ok"] is True
    assert saved["writeback_policy"]["current_mode"] == "preview_only"
