from __future__ import annotations

import json
from pathlib import Path

import pytest

from mango_mvp.productization.crm_tallanto_mapping_preview import build_crm_tallanto_mapping_preview
from mango_mvp.productization.demo_tenant import build_demo_tenant_product_root
from scripts import mango_office_crm_tallanto_mapping_preview


def test_crm_tallanto_mapping_preview_uses_local_snapshots_only(tmp_path: Path) -> None:
    product_root = tmp_path / "product_appliance"
    build_demo_tenant_product_root(product_root=product_root, replace_existing=True)
    product_db = product_root / "mango_product_appliance.sqlite"
    tallanto_snapshot = product_root / "crm_snapshots" / "tallanto_entities.json"
    tallanto_snapshot.write_text(
        json.dumps(
            {
                "entities": [
                    {"id": "t-1", "entity_type": "student", "phone": "+79990000000", "name": "Tallanto A"},
                    {"id": "t-2", "entity_type": "student", "phone": "+79990000001", "name": "Tallanto B"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = build_crm_tallanto_mapping_preview(
        product_db_path=product_db,
        product_root=product_root,
        tallanto_snapshot_path=tallanto_snapshot,
        out_path=product_root / "crm_mapping_preview" / "report.json",
        limit=10,
    )

    assert report["summary"]["capture_rows_seen"] == 4
    assert report["summary"]["amo_resolved"] == 4
    assert report["summary"]["tallanto_resolved"] == 2
    assert report["summary"]["tallanto_missing"] == 2
    assert report["safety"]["live_crm_reads"] is False
    assert report["safety"]["write_tallanto"] is False


def test_crm_tallanto_mapping_preview_cli_writes_report(tmp_path: Path) -> None:
    product_root = tmp_path / "product_appliance"
    build_demo_tenant_product_root(product_root=product_root, replace_existing=True)
    product_db = product_root / "mango_product_appliance.sqlite"
    out = product_root / "crm_mapping_preview" / "report.json"

    rc = mango_office_crm_tallanto_mapping_preview.main(
        ["--product-root", str(product_root), "--product-db", str(product_db), "--out", str(out)]
    )

    saved = json.loads(out.read_text(encoding="utf-8"))
    assert rc == 0
    assert saved["summary"]["amo_resolved"] == 4
    assert saved["safety"]["write_crm"] is False


def test_crm_tallanto_mapping_preview_refuses_snapshot_outside_root(tmp_path: Path) -> None:
    product_root = tmp_path / "product_appliance"
    build_demo_tenant_product_root(product_root=product_root, replace_existing=True)

    with pytest.raises(ValueError, match="snapshot"):
        build_crm_tallanto_mapping_preview(
            product_db_path=product_root / "mango_product_appliance.sqlite",
            product_root=product_root,
            amo_snapshot_path=tmp_path / "outside.json",
        )
