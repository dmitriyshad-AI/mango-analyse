from __future__ import annotations

import json
from pathlib import Path

import pytest

from mango_mvp.productization.demo_tenant import build_demo_tenant_product_root
from mango_mvp.productization.tenant_isolation import build_tenant_isolation_report
from scripts import mango_office_tenant_isolation


def test_tenant_isolation_reports_counts_and_scaffolds_layout(tmp_path: Path) -> None:
    product_root = tmp_path / "product_appliance"
    build_demo_tenant_product_root(product_root=product_root, replace_existing=True)
    product_db = product_root / "mango_product_appliance.sqlite"
    out = product_root / "tenant_isolation" / "report.json"

    report = build_tenant_isolation_report(product_root, product_db, out_path=out, scaffold=True)

    assert report["summary"]["validation_ok"] is True
    assert report["summary"]["tenants"] == 1
    assert report["summary"]["scaffold_written"] is True
    assert report["table_counts"]["product_calls"]["demo_foton"] == 4
    assert (product_root / "tenants" / "demo_foton" / "README.md").exists()
    assert json.loads(out.read_text(encoding="utf-8"))["safety"]["write_crm"] is False


def test_tenant_isolation_cli_writes_report(tmp_path: Path) -> None:
    product_root = tmp_path / "product_appliance"
    build_demo_tenant_product_root(product_root=product_root, replace_existing=True)
    out = product_root / "tenant_isolation" / "report.json"

    rc = mango_office_tenant_isolation.main(
        [
            "--product-root",
            str(product_root),
            "--product-db",
            str(product_root / "mango_product_appliance.sqlite"),
            "--out",
            str(out),
        ]
    )

    saved = json.loads(out.read_text(encoding="utf-8"))
    assert rc == 0
    assert saved["summary"]["tenant_scoped_tables"] >= 4
    assert saved["safety"]["read_only_db"] is True


def test_tenant_isolation_refuses_stable_runtime_output(tmp_path: Path) -> None:
    product_root = tmp_path / "product_appliance"
    build_demo_tenant_product_root(product_root=product_root, replace_existing=True)

    with pytest.raises(ValueError, match="stable_runtime"):
        build_tenant_isolation_report(
            product_root,
            product_root / "mango_product_appliance.sqlite",
            out_path=product_root / "stable_runtime" / "tenant.json",
        )
