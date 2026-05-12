from __future__ import annotations

import json
from pathlib import Path

import pytest

from mango_mvp.productization.product_api import ProductApiFacade
from mango_mvp.productization.product_api_http import (
    build_product_api_http_readiness_report,
    render_appliance_dashboard_html,
    route_product_api_request,
)
from mango_mvp.productization.product_db import initialize_product_db
from scripts import mango_office_product_api_http


def test_product_api_http_routes_read_only_endpoints(tmp_path: Path) -> None:
    product_root = tmp_path / "product_appliance"
    product_db = product_root / "mango_product_appliance.sqlite"
    initialize_product_db(product_db, product_root)
    api = ProductApiFacade(product_root=product_root, product_db_path=product_db, workspace_root=Path.cwd())

    status, payload = route_product_api_request(api, "GET", "/dashboard/summary")
    appliance_status, appliance_payload = route_product_api_request(api, "GET", "/dashboard/appliance")
    mutation_status, mutation_payload = route_product_api_request(api, "POST", "/dashboard/summary")
    missing_status, missing_payload = route_product_api_request(api, "GET", "/missing")

    assert status == 200
    assert payload["payload"]["endpoint"] == "GET /dashboard/summary"
    assert appliance_status == 200
    assert appliance_payload["payload"]["schema_version"] == "appliance_dashboard_v1"
    assert appliance_payload["payload"]["safety"]["write_crm"] is False
    writeback_status, writeback_payload = route_product_api_request(api, "GET", "/writeback/previews?stage=batch_50&limit=5")
    scheduler_health_status, scheduler_health_payload = route_product_api_request(api, "GET", "/scheduler/health")
    scheduler_control_status, scheduler_control_payload = route_product_api_request(api, "GET", "/scheduler/control-plane")
    lifecycle_status, lifecycle_payload = route_product_api_request(api, "GET", "/processing/lifecycle")
    crm_mapping_status, crm_mapping_payload = route_product_api_request(api, "GET", "/crm/mapping-preview")
    operator_status, operator_payload = route_product_api_request(api, "GET", "/operator/status")
    manual_status, manual_payload = route_product_api_request(api, "GET", "/manual-resolution/status")
    waiting_status, waiting_payload = route_product_api_request(api, "GET", "/waiting-work/status")
    dry_ready_status, dry_ready_payload = route_product_api_request(api, "GET", "/writeback/dry-run-readiness")
    assert writeback_status == 200
    assert writeback_payload["payload"]["preview"]["summary"]["stage"] == "batch_50"
    assert writeback_payload["payload"]["write_crm"] is False
    assert scheduler_health_status == 200
    assert scheduler_health_payload["payload"]["endpoint"] == "GET /scheduler/health"
    assert scheduler_control_status == 200
    assert scheduler_control_payload["payload"]["endpoint"] == "GET /scheduler/control-plane"
    assert lifecycle_status == 200
    assert lifecycle_payload["payload"]["mode"] == "waiting_for_lifecycle_report"
    assert crm_mapping_status == 200
    assert crm_mapping_payload["payload"]["endpoint"] == "GET /crm/mapping-preview"
    assert operator_status == 200
    assert operator_payload["payload"]["endpoint"] == "GET /operator/status"
    assert manual_status == 200
    assert manual_payload["payload"]["write_crm"] is False
    assert waiting_status == 200
    assert waiting_payload["payload"]["endpoint"] == "GET /waiting-work/status"
    assert waiting_payload["payload"]["write_crm"] is False
    assert dry_ready_status == 200
    assert dry_ready_payload["payload"]["live_write_allowed_now"] is False
    filtered_status, filtered_payload = route_product_api_request(
        api,
        "GET",
        "/dashboard/appliance?q=CALL&capture_status=ready_for_capture&scheduler_status=succeeded",
    )
    assert filtered_status == 200
    assert filtered_payload["payload"]["filters"]["q"] == "CALL"
    capture_status, capture_payload = route_product_api_request(api, "GET", "/capture/recent?q=CALL&status=ready_for_capture")
    assert capture_status == 200
    assert capture_payload["payload"]["filters"]["status"] == "ready_for_capture"
    assert payload["actions"]["write_crm"] is False
    assert mutation_status == 405
    assert mutation_payload["actions"]["run_asr"] is False
    assert missing_status == 404
    assert "/dashboard/summary" in missing_payload["implemented_routes"]


def test_product_api_http_invalid_limit_returns_json_400(tmp_path: Path) -> None:
    product_root = tmp_path / "product_appliance"
    product_db = product_root / "mango_product_appliance.sqlite"
    initialize_product_db(product_db, product_root)
    api = ProductApiFacade(product_root=product_root, product_db_path=product_db, workspace_root=Path.cwd())

    status, payload = route_product_api_request(api, "GET", "/capture/recent?limit=abc")
    zero_status, zero_payload = route_product_api_request(api, "GET", "/scheduler/runs?limit=0")
    dashboard_status, dashboard_payload = route_product_api_request(api, "GET", "/dashboard/appliance?capture_limit=abc")

    assert status == 400
    assert payload["error"] == "bad_request"
    assert payload["actions"]["write_runtime_db"] is False
    assert zero_status == 400
    assert zero_payload["error"] == "bad_request"
    assert dashboard_status == 400
    assert "capture_limit" in dashboard_payload["detail"]


def test_product_api_http_readiness_report(tmp_path: Path) -> None:
    product_root = tmp_path / "product_appliance"
    product_db = product_root / "mango_product_appliance.sqlite"
    initialize_product_db(product_db, product_root)
    out = product_root / "product_api_http" / "report.json"

    report = build_product_api_http_readiness_report(
        product_root=product_root,
        product_db_path=product_db,
        out_path=out,
        workspace_root=Path.cwd(),
    )

    assert report["summary"]["validation_ok"] is True
    assert report["summary"]["routes"] == 17
    assert report["blocked_mutation_check"]["status"] == 405
    assert all(status == 200 for status in report["routes"]["checks"].values())
    assert json.loads(out.read_text(encoding="utf-8"))["summary"]["read_only"] is True


def test_product_api_http_refuses_outside_missing_db(tmp_path: Path) -> None:
    product_root = tmp_path / "product_appliance"
    product_root.mkdir()

    with pytest.raises(ValueError, match="product DB"):
        build_product_api_http_readiness_report(
            product_root=product_root,
            product_db_path=tmp_path / "outside" / "mango_product_appliance.sqlite",
            workspace_root=Path.cwd(),
        )


def test_product_api_http_cli_readiness(tmp_path: Path) -> None:
    product_root = tmp_path / "product_appliance"
    product_db = product_root / "mango_product_appliance.sqlite"
    initialize_product_db(product_db, product_root)
    out = product_root / "product_api_http" / "report.json"

    rc = mango_office_product_api_http.main(
        [
            "--product-root",
            str(product_root),
            "--product-db",
            str(product_db),
            "readiness",
            "--out",
            str(out),
        ]
    )

    saved = json.loads(out.read_text(encoding="utf-8"))
    assert rc == 0
    assert saved["summary"]["routes"] == 17
    assert saved["safety"]["write_runtime_db"] is False


def test_product_api_http_dashboard_html_shell_is_read_only() -> None:
    html = render_appliance_dashboard_html()

    assert "Mango Analyse Appliance" in html
    assert "/dashboard/appliance" in html
    assert "Lifecycle" in html
    assert "Safety" in html
    assert "Demo readiness" in html
    assert "Manager tasks" in html
    assert "Duplicate recheck" in html
    assert "Waiting work" in html
    assert "Stage50" in html
    assert "Stage86" in html
    assert "Cleanup" in html
    assert "filter-q" in html
    assert "Apply" in html
    assert "Write CRM" in html
    assert "Write runtime DB" in html
    assert "--execute-live-write" not in html
