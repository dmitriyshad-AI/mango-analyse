from __future__ import annotations

import json
from pathlib import Path

from mango_mvp.productization.appliance_config_wizard import build_appliance_config_wizard_report
from mango_mvp.productization.demo_tenant import build_demo_tenant_product_root
from scripts import mango_office_appliance_config_wizard


def test_appliance_config_wizard_reports_installation_readiness(tmp_path: Path, monkeypatch) -> None:
    product_root = tmp_path / "product_appliance"
    build_demo_tenant_product_root(product_root=product_root, replace_existing=True)
    product_db = product_root / "mango_product_appliance.sqlite"
    snapshot = product_root / "crm_snapshots" / "amocrm_entities.json"
    out = product_root / "appliance_config_wizard" / "report.json"
    monkeypatch.delenv("MANGO_OFFICE_API_KEY", raising=False)
    monkeypatch.delenv("MANGO_OFFICE_API_SALT", raising=False)

    report = build_appliance_config_wizard_report(
        product_root=product_root,
        product_db_path=product_db,
        out_path=out,
        crm_snapshot_path=snapshot,
    )

    assert report["summary"]["validation_ok"] is True
    assert report["summary"]["blocked"] == 0
    assert report["summary"]["warnings"] >= 1
    assert any(check["name"] == "mango_credentials" and check["status"] == "WARN" for check in report["checks"])
    assert report["safety"]["write_crm"] is False
    assert json.loads(out.read_text(encoding="utf-8"))["summary"]["checks"] == 8
    assert "config_templates" in report


def test_appliance_config_wizard_can_require_mango_credentials(tmp_path: Path, monkeypatch) -> None:
    product_root = tmp_path / "product_appliance"
    build_demo_tenant_product_root(product_root=product_root, replace_existing=True)
    monkeypatch.delenv("MANGO_OFFICE_API_KEY", raising=False)
    monkeypatch.delenv("MANGO_OFFICE_API_SALT", raising=False)

    report = build_appliance_config_wizard_report(
        product_root=product_root,
        product_db_path=product_root / "mango_product_appliance.sqlite",
        require_mango_credentials=True,
    )

    assert report["summary"]["validation_ok"] is False
    assert any(check["name"] == "mango_credentials" and check["status"] == "BLOCK" for check in report["checks"])


def test_appliance_config_wizard_cli_writes_report(tmp_path: Path, monkeypatch) -> None:
    product_root = tmp_path / "product_appliance"
    build_demo_tenant_product_root(product_root=product_root, replace_existing=True)
    product_db = product_root / "mango_product_appliance.sqlite"
    out = product_root / "appliance_config_wizard" / "report.json"
    monkeypatch.setenv("MANGO_OFFICE_API_KEY", "key")
    monkeypatch.setenv("MANGO_OFFICE_API_SALT", "salt")

    rc = mango_office_appliance_config_wizard.main(
        [
            "--product-root",
            str(product_root),
            "--product-db",
            str(product_db),
            "--out",
            str(out),
            "--require-mango-credentials",
            "--write-templates",
        ]
    )

    saved = json.loads(out.read_text(encoding="utf-8"))
    assert rc == 0
    assert saved["summary"]["validation_ok"] is True
    assert saved["safety"]["live_crm_reads"] is False
    assert saved["safety"]["product_config_template_writes"] is True
    assert (product_root / "config" / "appliance.env.example").exists()
    assert "<put-client-vpbx-code-here>" in (product_root / "config" / "appliance.env.example").read_text(encoding="utf-8")
