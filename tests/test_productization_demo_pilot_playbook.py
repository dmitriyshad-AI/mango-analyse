from __future__ import annotations

import json
from pathlib import Path

import pytest

from mango_mvp.productization.demo_pilot_playbook import build_demo_pilot_playbook
from mango_mvp.productization.demo_tenant import build_demo_tenant_product_root
from scripts import mango_office_demo_pilot_playbook


def test_demo_pilot_playbook_writes_markdown_and_json(tmp_path: Path) -> None:
    product_root = tmp_path / "product_appliance"
    build_demo_tenant_product_root(product_root=product_root, replace_existing=True)
    out_dir = product_root / "demo_pilot_playbook"

    report = build_demo_pilot_playbook(product_root, product_root / "mango_product_appliance.sqlite", out_dir=out_dir)

    assert report["summary"]["validation_ok"] is True
    assert report["summary"]["demo_ready"] is True
    assert (out_dir / "demo_pilot_playbook.md").exists()
    assert (out_dir / "demo_pilot_playbook.json").exists()
    markdown = (out_dir / "demo_pilot_playbook.md").read_text(encoding="utf-8")
    assert "Safety Gates" in markdown
    assert "Processing quality acceptance" in markdown
    assert report["safety"]["write_crm"] is False


def test_demo_pilot_playbook_cli_writes_manifest(tmp_path: Path) -> None:
    product_root = tmp_path / "product_appliance"
    build_demo_tenant_product_root(product_root=product_root, replace_existing=True)
    out_dir = product_root / "demo_pilot_playbook"

    rc = mango_office_demo_pilot_playbook.main(
        [
            "--product-root",
            str(product_root),
            "--product-db",
            str(product_root / "mango_product_appliance.sqlite"),
            "--out-dir",
            str(out_dir),
        ]
    )

    saved = json.loads((out_dir / "demo_pilot_playbook_manifest.json").read_text(encoding="utf-8"))
    assert rc == 0
    assert saved["summary"]["pilot_ready_without_processing"] is True
    assert saved["safety"]["run_asr"] is False


def test_demo_pilot_playbook_refuses_outside_output(tmp_path: Path) -> None:
    product_root = tmp_path / "product_appliance"
    build_demo_tenant_product_root(product_root=product_root, replace_existing=True)

    with pytest.raises(ValueError, match="demo pilot playbook output"):
        build_demo_pilot_playbook(
            product_root,
            product_root / "mango_product_appliance.sqlite",
            out_dir=tmp_path / "outside",
        )
