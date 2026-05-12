from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from mango_mvp.productization.crm_entity_resolver import (
    BLOCK_AMBIGUOUS_CRM_MATCH,
    BLOCK_NO_CRM_MATCH,
    RESOLVE_CRM_ENTITY,
    build_crm_entity_resolution_report,
    load_crm_entities,
)
from scripts import mango_office_crm_entity_resolver
from tests.test_productization_product_db import bootstrap_sample_product_db


def test_crm_entity_resolver_matches_product_calls_by_filename_phone(tmp_path: Path) -> None:
    product_root, product_db, _tenant_config = bootstrap_sample_product_db(tmp_path)
    snapshot = write_json_snapshot(product_root, [{"entity_id": 501, "entity_type": "lead", "phone": "+79990000000"}])

    report = build_crm_entity_resolution_report(
        product_db_path=product_db,
        product_root=product_root,
        crm_snapshot_path=snapshot,
        out_path=product_root / "crm_entity_resolver_stage6" / "report.json",
    )

    assert report["summary"]["validation_ok"] is True
    assert report["summary"]["calls_seen"] == 3
    assert report["summary"]["crm_entities_seen"] == 1
    assert report["action_counts"] == {RESOLVE_CRM_ENTITY: 3}
    assert report["items"][0]["crm_entity_id"] == "501"
    assert report["items"][0]["call_phone"] == "+79990000000"
    assert report["safety"]["live_crm_reads"] is False
    assert report["safety"]["write_crm"] is False


def test_crm_entity_resolver_blocks_ambiguous_phone_matches(tmp_path: Path) -> None:
    product_root, product_db, _tenant_config = bootstrap_sample_product_db(tmp_path)
    snapshot = write_json_snapshot(
        product_root,
        [
            {"entity_id": 501, "entity_type": "lead", "phone": "+79990000000"},
            {"entity_id": 777, "entity_type": "contact", "phone": "89990000000"},
        ],
    )

    report = build_crm_entity_resolution_report(product_db, product_root, snapshot)

    assert report["action_counts"] == {BLOCK_AMBIGUOUS_CRM_MATCH: 3}
    assert report["summary"]["blocked_ambiguous_crm_match"] == 3
    assert report["items"][0]["candidate_count"] == 2


def test_crm_entity_resolver_reports_no_match(tmp_path: Path) -> None:
    product_root, product_db, _tenant_config = bootstrap_sample_product_db(tmp_path)
    snapshot = write_json_snapshot(product_root, [{"entity_id": 501, "entity_type": "lead", "phone": "+79991111111"}])

    report = build_crm_entity_resolution_report(product_db, product_root, snapshot, limit=1)

    assert report["action_counts"] == {BLOCK_NO_CRM_MATCH: 1}
    assert report["summary"]["calls_seen"] == 1


def test_crm_entity_resolver_loads_csv_snapshot(tmp_path: Path) -> None:
    product_root, _product_db, _tenant_config = bootstrap_sample_product_db(tmp_path)
    snapshot = product_root / "crm_snapshots" / "amocrm_entities.csv"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    with snapshot.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["id", "type", "phone", "name"])
        writer.writeheader()
        writer.writerow({"id": "501", "type": "lead", "phone": "+79990000000", "name": "Lead 501"})

    rows = load_crm_entities(snapshot)

    assert rows[0]["entity_id"] == "501"
    assert rows[0]["entity_type"] == "lead"
    assert rows[0]["phones"] == ("+79990000000",)


def test_crm_entity_resolver_cli_writes_report(tmp_path: Path) -> None:
    product_root, product_db, _tenant_config = bootstrap_sample_product_db(tmp_path)
    snapshot = write_json_snapshot(product_root, [{"entity_id": 501, "entity_type": "lead", "phone": "+79990000000"}])
    out = product_root / "crm_entity_resolver_stage6" / "cli.json"

    rc = mango_office_crm_entity_resolver.main(
        [
            "--product-root",
            str(product_root),
            "--product-db",
            str(product_db),
            "--crm-snapshot",
            str(snapshot),
            "--out",
            str(out),
        ]
    )

    assert rc == 0
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["summary"]["resolve_crm_entity"] == 3
    assert saved["safety"]["write_crm"] is False


def test_crm_entity_resolver_refuses_snapshot_outside_product_root(tmp_path: Path) -> None:
    product_root, product_db, _tenant_config = bootstrap_sample_product_db(tmp_path)
    outside = tmp_path / "outside.json"
    outside.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="CRM snapshot"):
        build_crm_entity_resolution_report(product_db, product_root, outside)


def write_json_snapshot(product_root: Path, rows: list[dict]) -> Path:
    path = product_root / "crm_snapshots" / "amocrm_entities.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"entities": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
