from __future__ import annotations

import json
from pathlib import Path

from mango_mvp.productization.demo_tenant import build_demo_tenant_product_root
from scripts import mango_office_demo_tenant


def test_demo_tenant_builds_isolated_product_root(tmp_path: Path) -> None:
    product_root = tmp_path / "demo_product_appliance"

    report = build_demo_tenant_product_root(product_root=product_root, replace_existing=True)

    assert report["summary"]["validation_ok"] is True
    assert report["summary"]["product_calls"] == 4
    assert report["summary"]["capture_inbox_items"] == 4
    assert report["summary"]["crm_entities"] == 4
    assert report["safety"]["fake_data_only"] is True
    assert report["safety"]["write_crm"] is False
    assert (product_root / "mango_product_appliance.sqlite").exists()
    assert (product_root / "crm_snapshots" / "amocrm_entities.json").exists()
    assert (product_root / "product_api_readiness" / "demo_api_readiness.json").exists()


def test_demo_tenant_cli_writes_report(tmp_path: Path) -> None:
    product_root = tmp_path / "demo_product_appliance"
    out = product_root / "demo_tenant_report.json"

    rc = mango_office_demo_tenant.main(["--product-root", str(product_root), "--out", str(out), "--replace"])

    saved = json.loads(out.read_text(encoding="utf-8"))
    assert rc == 0
    assert saved["summary"]["validation_ok"] is True
    assert "serve_dashboard" in saved["demo_commands"]


def test_demo_tenant_refuses_stable_runtime(tmp_path: Path) -> None:
    product_root = tmp_path / "stable_runtime" / "demo_product_appliance"

    rc = mango_office_demo_tenant.main(["--product-root", str(product_root), "--replace"])

    assert rc == 2
