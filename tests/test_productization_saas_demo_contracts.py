from __future__ import annotations

import json
from pathlib import Path

import pytest

from mango_mvp.productization.demo_tenant import build_demo_tenant_product_root
from mango_mvp.productization.saas_demo_contracts import build_dashboard_demo_readiness, build_snapshot_inventory


def test_snapshot_inventory_reads_json_and_csv_snapshots(tmp_path: Path) -> None:
    product_root = tmp_path / "product_appliance"
    snapshot_dir = product_root / "crm_snapshots"
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / "amocrm_entities.json").write_text(
        json.dumps({"entities": [{"entity_id": "1", "phone": "+79001234567"}]}, ensure_ascii=False),
        encoding="utf-8",
    )
    (snapshot_dir / "tallanto_entities.csv").write_text("entity_id,phone\n2,+79007654321\n", encoding="utf-8")

    report = build_snapshot_inventory(product_root)

    assert report["summary"]["snapshot_files"] == 2
    assert report["summary"]["entities"] == 2
    assert report["summary"]["phones_indexed"] == 2
    assert report["summary"]["warnings"] == 0


def test_dashboard_demo_readiness_tracks_panels_snapshots_and_artifacts(tmp_path: Path) -> None:
    product_root = tmp_path / "product_appliance"
    build_demo_tenant_product_root(product_root=product_root, replace_existing=True)
    product_db = product_root / "mango_product_appliance.sqlite"
    panels = {name: {} for name in ("capture", "processing_queue", "scheduler", "lifecycle", "writeback", "crm_mapping", "gates", "knowledge", "settings")}

    report = build_dashboard_demo_readiness(product_root, product_db, panels=panels)

    assert report["summary"]["validation_ok"] is True
    assert report["summary"]["snapshot_files"] == 1
    assert report["summary"]["demo_artifacts"] >= 1
    assert report["missing_panels"] == []
    assert report["safety"]["write_crm"] is False


def test_dashboard_demo_readiness_refuses_stable_runtime(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="stable_runtime"):
        build_snapshot_inventory(tmp_path / "stable_runtime")
