from __future__ import annotations

import json
from pathlib import Path

import pytest

from mango_mvp.productization.appliance_service_pack import build_appliance_service_pack
from mango_mvp.productization.product_db import initialize_product_db
from scripts import mango_office_appliance_service_pack


def test_appliance_service_pack_writes_templates_without_starting_services(tmp_path: Path) -> None:
    product_root = tmp_path / "product appliance"
    product_db = product_root / "mango_product_appliance.sqlite"
    initialize_product_db(product_db, product_root)
    out_dir = product_root / "service_pack"

    report = build_appliance_service_pack(product_root, product_db, out_dir=out_dir)

    assert report["summary"]["templates_written"] == 4
    assert report["summary"]["installs_services"] is False
    assert report["summary"]["starts_services"] is False
    assert report["safety"]["run_asr"] is False
    assert (out_dir / "launchd" / "com.mango-analyse.dashboard.plist").exists()
    assert (out_dir / "systemd" / "mango-analyse-dashboard.service").exists()
    assert "serve --host 127.0.0.1" in (out_dir / "README.md").read_text(encoding="utf-8")


def test_appliance_service_pack_cli_writes_manifest(tmp_path: Path) -> None:
    product_root = tmp_path / "product_appliance"
    product_db = product_root / "mango_product_appliance.sqlite"
    initialize_product_db(product_db, product_root)
    out_dir = product_root / "service_pack"

    rc = mango_office_appliance_service_pack.main(
        ["--product-root", str(product_root), "--product-db", str(product_db), "--out-dir", str(out_dir)]
    )

    saved = json.loads((out_dir / "service_pack_manifest.json").read_text(encoding="utf-8"))
    assert rc == 0
    assert saved["summary"]["validation_ok"] is True
    assert saved["safety"]["starts_services"] is False


def test_appliance_service_pack_refuses_stable_runtime_output(tmp_path: Path) -> None:
    product_root = tmp_path / "product_appliance"
    product_db = product_root / "mango_product_appliance.sqlite"
    initialize_product_db(product_db, product_root)

    with pytest.raises(ValueError, match="stable_runtime"):
        build_appliance_service_pack(product_root, product_db, out_dir=product_root / "stable_runtime" / "service_pack")
