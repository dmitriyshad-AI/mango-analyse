from __future__ import annotations

import json
from pathlib import Path

import pytest

from mango_mvp.productization.crm_writeback_preview import (
    BLOCK_MISSING_CRM_ENTITY,
    BLOCK_POLICY_FORBIDDEN,
    PREVIEW_READY,
    build_crm_writeback_preview,
    build_preview_item,
)
from scripts import mango_office_crm_writeback_preview
from tests.test_productization_product_db import bootstrap_sample_product_db


def test_crm_writeback_preview_builds_safe_blocked_diff_from_product_db(tmp_path: Path) -> None:
    product_root, product_db, _tenant_config = bootstrap_sample_product_db(tmp_path)

    report = build_crm_writeback_preview(
        product_db_path=product_db,
        product_root=product_root,
        out_path=product_root / "crm_writeback_preview_stage6" / "preview.json",
        stage="batch_10",
    )

    assert report["summary"]["validation_ok"] is True
    assert report["summary"]["write_crm"] is False
    assert report["summary"]["selected_items"] == 3
    assert report["action_counts"] == {BLOCK_MISSING_CRM_ENTITY: 3}
    assert report["policy_gates"]["live_write_enabled"] is False
    assert report["policy_gates"]["required_live_confirmation"] == "WRITE_AMO_LIVE"
    assert report["safety"]["write_crm"] is False
    assert report["items"][0]["diff"][0]["write_policy"] == "preview_only"


def test_crm_writeback_preview_uses_resolved_crm_entity_snapshot(tmp_path: Path) -> None:
    product_root, product_db, _tenant_config = bootstrap_sample_product_db(tmp_path)
    snapshot = product_root / "crm_snapshots" / "amocrm_entities.json"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_text(
        json.dumps({"entities": [{"entity_id": 501, "entity_type": "lead", "phone": "+79990000000"}]}),
        encoding="utf-8",
    )

    report = build_crm_writeback_preview(
        product_db_path=product_db,
        product_root=product_root,
        crm_snapshot_path=snapshot,
    )

    assert report["crm_resolution"]["enabled"] is True
    assert report["crm_resolution"]["summary"]["resolve_crm_entity"] == 3
    assert report["summary"]["blocked_missing_crm_entity"] == 0
    assert report["summary"]["blocked_missing_insight"] == 3
    assert report["action_counts"] == {"BLOCK_MISSING_INSIGHT": 3}
    assert report["items"][0]["crm_entity_id"] == "501"
    assert report["items"][0]["crm_resolution"]["action"] == "RESOLVE_CRM_ENTITY"


def test_crm_writeback_preview_item_can_be_ready_only_with_entity_and_insight() -> None:
    item = build_preview_item(
        {
            "tenant_id": "foton",
            "telephony_provider": "mango",
            "provider_call_id": "CALL-1",
            "event_key": "foton:mango:CALL-1",
            "crm_entity_id": "123",
            "crm_entity_type": "lead",
            "insight_summary": "Короткая сводка",
            "recommended_next_step": "Позвонить завтра",
            "ai_priority": "high",
        }
    )

    assert item["action"] == PREVIEW_READY
    assert item["confidence"] == 0.8
    assert item["policy"]["write_crm"] is False
    assert item["diff"][0]["write_policy"] == "preview_only"


def test_crm_writeback_preview_item_blocks_forbidden_actions() -> None:
    item = build_preview_item(
        {
            "tenant_id": "foton",
            "telephony_provider": "mango",
            "provider_call_id": "CALL-1",
            "event_key": "foton:mango:CALL-1",
            "crm_entity_id": "123",
            "crm_entity_type": "lead",
            "insight_summary": "Сводка",
            "requested_actions": json.dumps(["close_lead_automatically"]),
        }
    )

    assert item["action"] == BLOCK_POLICY_FORBIDDEN
    assert "close_lead_automatically" in item["blockers"]


def test_crm_writeback_preview_cli_writes_report(tmp_path: Path) -> None:
    product_root, product_db, _tenant_config = bootstrap_sample_product_db(tmp_path)
    out = product_root / "crm_writeback_preview_stage6" / "cli.json"

    rc = mango_office_crm_writeback_preview.main(
        [
            "--product-root",
            str(product_root),
            "--product-db",
            str(product_db),
            "--out",
            str(out),
            "--stage",
            "batch_50",
            "--limit",
            "2",
        ]
    )

    assert rc == 0
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["summary"]["stage"] == "batch_50"
    assert saved["summary"]["selected_items"] == 2
    assert saved["safety"]["write_crm"] is False


def test_crm_writeback_preview_cli_accepts_crm_snapshot(tmp_path: Path) -> None:
    product_root, product_db, _tenant_config = bootstrap_sample_product_db(tmp_path)
    snapshot = product_root / "crm_snapshots" / "amocrm_entities.json"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_text(
        json.dumps({"entities": [{"entity_id": 501, "entity_type": "lead", "phone": "+79990000000"}]}),
        encoding="utf-8",
    )
    out = product_root / "crm_writeback_preview_stage6" / "cli_snapshot.json"

    rc = mango_office_crm_writeback_preview.main(
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
    assert saved["summary"]["blocked_missing_crm_entity"] == 0
    assert saved["summary"]["blocked_missing_insight"] == 3


def test_crm_writeback_preview_refuses_unknown_stage(tmp_path: Path) -> None:
    product_root, product_db, _tenant_config = bootstrap_sample_product_db(tmp_path)

    with pytest.raises(ValueError, match="unknown writeback stage"):
        build_crm_writeback_preview(product_db, product_root, stage="live_now")
