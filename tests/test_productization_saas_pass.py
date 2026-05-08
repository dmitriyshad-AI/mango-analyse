from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from mango_mvp.productization.insight_seed import build_insight_seed_report
from mango_mvp.productization.manager_identity import install_manager_identity_map
from mango_mvp.productization.provider_metadata import PROVIDER_METADATA_TABLE, install_provider_metadata_sidecar
from mango_mvp.productization.repository import ProductRepository
from mango_mvp.productization.supervisor import build_supervisor_dry_run_report
from mango_mvp.productization.tenant_owner_mapping import (
    TENANT_OWNER_MAPPING_SCHEMA_VERSION,
    build_tenant_owner_mapping_draft,
)
from mango_mvp.productization.ui_contracts import build_dashboard_contract
from scripts import mango_office_saas_productization_audit
from tests.test_productization_manager_identity import make_manager_row, write_amo_users, write_mango_users
from tests.test_productization_provider_metadata import build_disposable_db


def build_sample_product_db(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    root = tmp_path / "quarantine"
    audio_dir = root / "audio"
    rows = [
        make_manager_row(audio_dir, "CALL-1", "101"),
        make_manager_row(audio_dir, "CALL-2", "101"),
        make_manager_row(audio_dir, "CALL-3", "102"),
    ]
    db_path, metadata_csv, out_root = build_disposable_db(tmp_path, rows)
    install_provider_metadata_sidecar(
        db_path=db_path,
        metadata_csv_path=metadata_csv,
        out_allowed_root=out_root,
        replace_existing=True,
    )
    with sqlite3.connect(db_path) as con:
        con.execute(f"update {PROVIDER_METADATA_TABLE} set raw_payload_ref = 'raw/payloads.jsonl#entry=test'")
        con.commit()
    mango_users_path = tmp_path / "config" / "mango_users.json"
    amo_users_path = tmp_path / "config" / "amo_users.json"
    write_mango_users(
        mango_users_path,
        [
            {"extension": "101", "name": "Анна Менеджер", "email": "anna@example.com"},
            {"extension": "102", "name": "Олег Менеджер", "email": "oleg@example.com"},
        ],
    )
    write_amo_users(amo_users_path, [{"id": 9001, "name": "Анна Менеджер", "email": "anna@example.com"}])
    install_manager_identity_map(
        db_path=db_path,
        mango_users_path=mango_users_path,
        amo_users_path=amo_users_path,
        out_allowed_root=out_root,
        replace_existing=True,
    )
    raw_payload = root / "raw_payload.jsonl"
    raw_payload.write_text('{"entry_id":"CALL-1"}\n{"entry_id":"CALL-2"}\n', encoding="utf-8")
    return db_path, root, audio_dir, raw_payload


def test_product_repository_reads_enriched_calls(tmp_path: Path) -> None:
    db_path, root, _audio_dir, _raw_payload = build_sample_product_db(tmp_path)
    repo = ProductRepository(db_path=db_path, out_allowed_root=root)

    summary = repo.summary()
    calls = repo.list_calls(limit=10)
    missing_owner_calls = repo.list_calls(limit=10, crm_owner_status="missing")

    assert summary.validation_ok is True
    assert summary.provider_metadata_rows == 3
    assert summary.enriched_view_rows == 3
    assert summary.calls_with_crm_owner == 2
    assert summary.manual_owner_review_items == 1
    assert len(calls) == 3
    assert {call.manager_display_name for call in calls} == {"Анна Менеджер", "Олег Менеджер"}
    assert len(missing_owner_calls) == 1
    assert missing_owner_calls[0].manager_extension == "102"


def test_tenant_owner_mapping_draft_marks_pending_decisions(tmp_path: Path) -> None:
    db_path, root, _audio_dir, _raw_payload = build_sample_product_db(tmp_path)

    report = build_tenant_owner_mapping_draft(
        db_path=db_path,
        out_allowed_root=root,
        out_path=root / "test_ingest" / "owner_draft.json",
    )

    assert report["summary"]["confirmed_candidates"] == 1
    assert report["summary"]["manual_decisions_required"] == 1
    assert report["summary"]["calls_pending"] == 1
    assert report["manual_review_items"][0]["manager_extension"] == "102"
    assert report["config_template"]["schema_version"] == TENANT_OWNER_MAPPING_SCHEMA_VERSION
    assert (root / "test_ingest" / "owner_draft.json").exists()


def test_supervisor_ui_and_insight_reports_share_repository(tmp_path: Path) -> None:
    db_path, root, audio_dir, raw_payload = build_sample_product_db(tmp_path)
    repo = ProductRepository(db_path=db_path, out_allowed_root=root)

    supervisor = build_supervisor_dry_run_report(repo, [raw_payload], audio_dir)
    ui_contract = build_dashboard_contract(repo, call_limit=2)
    insight = build_insight_seed_report(repo)

    assert supervisor["summary"]["validation_ok"] is True
    assert supervisor["summary"]["warning_steps"] == 1
    assert supervisor["steps"][-1]["name"] == "manager_identity"
    assert supervisor["steps"][-1]["blocked_by"] == ("tenant_owner_mapping_review_required",)
    assert ui_contract["schema_version"] == "saas_ui_contracts_v1"
    assert len(ui_contract["views"]["call_list"]["items"]) == 2
    assert ui_contract["views"]["manual_owner_review_queue"]["items"][0]["manager_extension"] == "102"
    assert "write_crm" in ui_contract["actions"]["blocked"]
    assert insight["summary"]["seeds"] == 2
    assert insight["summary"]["manual_owner_seeds"] == 1
    assert insight["items"][0]["evidence_refs"]


def test_saas_productization_audit_script_writes_all_reports(tmp_path: Path) -> None:
    db_path, root, audio_dir, raw_payload = build_sample_product_db(tmp_path)
    out_root = root / "test_ingest"
    out = out_root / "saas_audit.json"

    rc = mango_office_saas_productization_audit.main(
        [
            "--db",
            str(db_path),
            "--allowed-root",
            str(root),
            "--out-root",
            str(out_root),
            "--raw-payload",
            str(raw_payload),
            "--audio-dir",
            str(audio_dir),
            "--out",
            str(out),
            "--call-limit",
            "2",
        ]
    )

    assert rc == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["summary"]["validation_ok"] is True
    for path in data["outputs"].values():
        assert Path(path).exists()


def test_saas_productization_audit_script_refuses_output_outside_allowed_root(tmp_path: Path) -> None:
    db_path, root, audio_dir, raw_payload = build_sample_product_db(tmp_path)

    with pytest.raises(ValueError, match="out-root"):
        mango_office_saas_productization_audit.main(
            [
                "--db",
                str(db_path),
                "--allowed-root",
                str(root),
                "--out-root",
                str(tmp_path / "outside"),
                "--raw-payload",
                str(raw_payload),
                "--audio-dir",
                str(audio_dir),
                "--out",
                str(root / "test_ingest" / "saas_audit.json"),
            ]
        )
