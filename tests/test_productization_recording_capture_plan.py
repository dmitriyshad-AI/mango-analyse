from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from mango_mvp.productization.recording_capture_plan import (
    BLOCK_MISSING_RECORDING_REF,
    PLAN_DOWNLOAD_DRY_RUN,
    SKIP_DUPLICATE_RECORDING,
    audit_recording_capture_plan,
    build_recording_capture_plan,
)
from scripts import mango_office_recording_capture_plan
from tests.test_productization_product_db import bootstrap_sample_product_db


def test_recording_capture_plan_is_dry_run_and_flags_risks(tmp_path: Path) -> None:
    product_root, product_db, _tenant_config = bootstrap_sample_product_db(tmp_path)
    insert_inbox_item(product_db, event_key="foton:mango:CALL-A", provider_call_id="CALL-A", recording_ref="rec-a")
    insert_inbox_item(product_db, event_key="foton:mango:CALL-B", provider_call_id="CALL-B", recording_ref="rec-b")
    insert_inbox_item(product_db, event_key="foton:mango:CALL-C", provider_call_id="CALL-C", recording_ref="rec-a")
    insert_inbox_item(product_db, event_key="foton:mango:CALL-D", provider_call_id="CALL-D", recording_ref=None)
    recordings_dir = product_root / "recording_capture_dry_run" / "recordings"
    manifest = product_root / "recording_capture_dry_run" / "plan.jsonl"
    out = product_root / "recording_capture_dry_run" / "audit.json"

    report = build_recording_capture_plan(
        product_db_path=product_db,
        product_root=product_root,
        recordings_dir=recordings_dir,
        manifest_path=manifest,
        out_path=out,
    )

    assert report["summary"]["inbox_items_seen"] == 4
    assert report["summary"]["manifest_items"] == 4
    assert report["summary"]["plan_download_dry_run"] == 2
    assert report["summary"]["skip_duplicate_recording"] == 1
    assert report["summary"]["blocked_missing_recording_ref"] == 1
    assert report["summary"]["validation_ok"] is False
    assert report["action_counts"] == {
        BLOCK_MISSING_RECORDING_REF: 1,
        PLAN_DOWNLOAD_DRY_RUN: 2,
        SKIP_DUPLICATE_RECORDING: 1,
    }
    assert manifest.exists()
    assert out.exists()
    assert not recordings_dir.exists()
    rows = read_jsonl(manifest)
    assert {row["action"] for row in rows} == {
        BLOCK_MISSING_RECORDING_REF,
        PLAN_DOWNLOAD_DRY_RUN,
        SKIP_DUPLICATE_RECORDING,
    }
    assert all(row["download_audio"] is False for row in rows)
    assert all(row["run_asr"] is False for row in rows)
    assert all(row["run_ra"] is False for row in rows)
    assert all(row["write_runtime_db"] is False for row in rows)
    assert all(row["write_crm"] is False for row in rows)


def test_recording_capture_plan_filter_limit_and_cli_audit(tmp_path: Path) -> None:
    product_root, product_db, _tenant_config = bootstrap_sample_product_db(tmp_path)
    insert_inbox_item(
        product_db,
        event_key="foton:mango:CALL-A",
        provider_call_id="CALL-A",
        recording_ref="rec-a",
        manager_ref="101",
    )
    insert_inbox_item(
        product_db,
        event_key="foton:mango:CALL-B",
        provider_call_id="CALL-B",
        recording_ref="rec-b",
        manager_ref="102",
    )
    manifest = product_root / "recording_capture_dry_run" / "filtered.jsonl"
    build_out = product_root / "recording_capture_dry_run" / "filtered_build.json"
    audit_out = product_root / "recording_capture_dry_run" / "filtered_audit.json"

    build_rc = mango_office_recording_capture_plan.main(
        [
            "--product-root",
            str(product_root),
            "--product-db",
            str(product_db),
            "build",
            "--recordings-dir",
            str(product_root / "recording_capture_dry_run" / "recordings"),
            "--manifest",
            str(manifest),
            "--out",
            str(build_out),
            "--manager-ref",
            "102",
            "--limit",
            "1",
        ]
    )
    audit_rc = mango_office_recording_capture_plan.main(
        [
            "--product-root",
            str(product_root),
            "--product-db",
            str(product_db),
            "audit",
            "--manifest",
            str(manifest),
            "--out",
            str(audit_out),
        ]
    )

    assert build_rc == 0
    assert audit_rc == 0
    build_data = json.loads(build_out.read_text(encoding="utf-8"))
    audit_data = json.loads(audit_out.read_text(encoding="utf-8"))
    assert build_data["summary"]["inbox_items_seen"] == 1
    assert build_data["manager_ref_counts"] == {"102": 1}
    assert audit_data["summary"]["items"] == 1
    row = read_jsonl(manifest)[0]
    assert row["manager_ref"] == "102"
    assert row["action"] == PLAN_DOWNLOAD_DRY_RUN


def test_recording_capture_audit_rejects_target_paths_outside_product_root(tmp_path: Path) -> None:
    product_root, _product_db, _tenant_config = bootstrap_sample_product_db(tmp_path)
    manifest = product_root / "recording_capture_dry_run" / "bad.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        manifest,
        [
            {
                "schema_version": "recording_capture_plan_v1",
                "action": PLAN_DOWNLOAD_DRY_RUN,
                "target_audio_path": str(tmp_path / "outside" / "audio.mp3"),
            }
        ],
    )

    report = audit_recording_capture_plan(
        manifest_path=manifest,
        product_root=product_root,
        out_path=product_root / "recording_capture_dry_run" / "bad_audit.json",
    )

    assert report["summary"]["validation_ok"] is False
    assert report["summary"]["target_paths_outside_root"] == 1
    assert report["summary"]["blocked"] == 1


def test_recording_capture_plan_refuses_outputs_outside_product_root(tmp_path: Path) -> None:
    product_root, product_db, _tenant_config = bootstrap_sample_product_db(tmp_path)
    insert_inbox_item(product_db, event_key="foton:mango:CALL-A", provider_call_id="CALL-A", recording_ref="rec-a")

    with pytest.raises(ValueError, match="recording capture manifest"):
        build_recording_capture_plan(
            product_db_path=product_db,
            product_root=product_root,
            recordings_dir=product_root / "recording_capture_dry_run" / "recordings",
            manifest_path=tmp_path / "outside.jsonl",
        )


def insert_inbox_item(
    product_db: Path,
    *,
    event_key: str,
    provider_call_id: str,
    recording_ref: str | None,
    manager_ref: str = "101",
) -> None:
    now = "2026-05-07T06:00:00+00:00"
    with sqlite3.connect(product_db) as con:
        con.execute(
            """
            INSERT INTO capture_inbox_items (
                tenant_id, provider, event_key, provider_call_id, status,
                started_at, ended_at, direction, client_phone, manager_ref,
                recording_ref, audio_ref, decision_reason, candidate_json, event_json,
                first_seen_at, last_seen_at, enqueue_count
            )
            VALUES (
                'foton', 'mango', ?, ?, 'ready_for_capture',
                ?, '2026-05-07T06:05:00+00:00', 'outbound', '+79990000000', ?,
                ?, ?, 'ready_for_shadow_capture', '{}', '{}',
                ?, ?, 1
            )
            """,
            (event_key, provider_call_id, now, manager_ref, recording_ref, recording_ref, now, now),
        )
        con.commit()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
