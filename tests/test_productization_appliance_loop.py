from __future__ import annotations

import json
from pathlib import Path

import pytest

from mango_mvp.productization.appliance_loop import build_autonomous_appliance_loop_dry_run
from mango_mvp.productization.product_db import initialize_product_db
from scripts import mango_office_appliance_loop_dry_run


def test_appliance_loop_dry_run_blocks_dangerous_actions(tmp_path: Path) -> None:
    product_root = tmp_path / "product_appliance"
    product_db = product_root / "mango_product_appliance.sqlite"
    initialize_product_db(product_db, product_root)
    out = product_root / "appliance_loop" / "report.json"

    report = build_autonomous_appliance_loop_dry_run(
        product_root=product_root,
        product_db_path=product_db,
        out_path=out,
        workspace_root=Path.cwd(),
    )

    assert report["summary"]["validation_ok"] is True
    assert report["summary"]["loop_ready"] is False
    assert report["summary"]["blocked_actions"] >= 3
    assert report["action_counts"]["BLOCK_ASR_AUTO_TRIGGER"] == 1
    assert report["action_counts"]["BLOCK_CRM_WRITEBACK"] == 1
    assert report["safety"]["download_audio"] is False
    assert report["safety"]["run_asr"] is False
    assert report["safety"]["write_crm"] is False
    assert out.exists()


def test_appliance_loop_refuses_outside_db(tmp_path: Path) -> None:
    product_root = tmp_path / "product_appliance"
    product_root.mkdir()

    with pytest.raises(ValueError, match="product DB"):
        build_autonomous_appliance_loop_dry_run(
            product_root=product_root,
            product_db_path=tmp_path / "outside" / "mango_product_appliance.sqlite",
            workspace_root=Path.cwd(),
        )


def test_appliance_loop_cli_writes_report(tmp_path: Path) -> None:
    product_root = tmp_path / "product_appliance"
    product_db = product_root / "mango_product_appliance.sqlite"
    initialize_product_db(product_db, product_root)
    out = product_root / "appliance_loop" / "report.json"

    rc = mango_office_appliance_loop_dry_run.main(
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
    assert saved["summary"]["validation_ok"] is True
    assert saved["safety"]["write_runtime_db"] is False
