from __future__ import annotations

import json
from pathlib import Path

import pytest

from mango_mvp.productization.appliance_command_surface import build_appliance_command_surface
from scripts import mango_office_appliance


def test_appliance_command_surface_builds_safe_operator_flow(tmp_path: Path) -> None:
    product_root = tmp_path / "product appliance"
    product_db = product_root / "mango_product_appliance.sqlite"
    out = product_root / "appliance_command_surface" / "commands.json"

    report = build_appliance_command_surface(
        product_root=product_root,
        product_db_path=product_db,
        out_path=out,
        workspace_root=Path.cwd(),
    )

    assert report["summary"]["validation_ok"] is True
    assert report["summary"]["commands"] >= 8
    assert report["summary"]["warnings"] == 1
    assert report["operator_flow"][0] == "demo_or_bootstrap"
    assert report["safety"]["executes_commands"] is False
    assert "mango_office_product_api_http.py" in json.dumps(report, ensure_ascii=False)
    assert out.exists()


def test_appliance_command_surface_cli_writes_report(tmp_path: Path) -> None:
    product_root = tmp_path / "product_appliance"
    product_db = product_root / "mango_product_appliance.sqlite"
    out = product_root / "appliance_command_surface" / "commands.json"

    rc = mango_office_appliance.main(
        ["--product-root", str(product_root), "--product-db", str(product_db), "--out", str(out)]
    )

    saved = json.loads(out.read_text(encoding="utf-8"))
    assert rc == 0
    assert saved["summary"]["validation_ok"] is True
    assert saved["safety"]["run_asr"] is False


def test_appliance_command_surface_refuses_stable_runtime(tmp_path: Path) -> None:
    product_root = tmp_path / "stable_runtime" / "product_appliance"

    with pytest.raises(ValueError, match="stable_runtime"):
        build_appliance_command_surface(
            product_root=product_root,
            product_db_path=product_root / "mango_product_appliance.sqlite",
            workspace_root=Path.cwd(),
        )
