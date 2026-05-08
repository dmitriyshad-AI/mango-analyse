from __future__ import annotations

import json
from pathlib import Path

import pytest

from mango_mvp.productization.product_api import ProductApiFacade
from mango_mvp.productization.product_api_http import build_product_api_http_readiness_report, route_product_api_request
from mango_mvp.productization.product_db import initialize_product_db
from scripts import mango_office_product_api_http


def test_product_api_http_routes_read_only_endpoints(tmp_path: Path) -> None:
    product_root = tmp_path / "product_appliance"
    product_db = product_root / "mango_product_appliance.sqlite"
    initialize_product_db(product_db, product_root)
    api = ProductApiFacade(product_root=product_root, product_db_path=product_db, workspace_root=Path.cwd())

    status, payload = route_product_api_request(api, "GET", "/dashboard/summary")
    mutation_status, mutation_payload = route_product_api_request(api, "POST", "/dashboard/summary")
    missing_status, missing_payload = route_product_api_request(api, "GET", "/missing")

    assert status == 200
    assert payload["payload"]["endpoint"] == "GET /dashboard/summary"
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

    assert status == 400
    assert payload["error"] == "bad_request"
    assert payload["actions"]["write_runtime_db"] is False
    assert zero_status == 400
    assert zero_payload["error"] == "bad_request"


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
    assert report["summary"]["routes"] == 8
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
    assert saved["summary"]["routes"] == 8
    assert saved["safety"]["write_runtime_db"] is False
