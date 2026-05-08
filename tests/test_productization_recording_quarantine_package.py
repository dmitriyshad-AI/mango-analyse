from __future__ import annotations

import json
from pathlib import Path

import pytest

from mango_mvp.productization.capture_staging import file_sha256
from mango_mvp.productization.recording_quarantine_package import (
    build_recording_quarantine_plan,
    materialize_recording_quarantine_package,
)
from scripts import mango_office_recording_quarantine_package


def test_recording_quarantine_package_plans_and_materializes_nested_bridge_report(tmp_path: Path) -> None:
    product_root = tmp_path / "product_appliance"
    bridge_report = write_nested_bridge_report(product_root, count=2)
    package_root = product_root / "recording_quarantine_stage9"
    plan_path = package_root / "recording_quarantine_plan_stage9.json"

    plan = build_recording_quarantine_plan(
        source_bridge_plan_path=bridge_report,
        product_root=product_root,
        package_root=package_root,
        quarantine_dir=package_root / "audio",
        metadata_csv_path=package_root / "metadata.csv",
        plan_path=plan_path,
        normalized_bridge_plan_path=package_root / "normalized_bridge_plan_stage9.json",
    )
    first = materialize_recording_quarantine_package(
        plan_path=plan_path,
        product_root=product_root,
        out_path=package_root / "materialization.json",
        mode="copy",
    )
    second = materialize_recording_quarantine_package(
        plan_path=plan_path,
        product_root=product_root,
        out_path=package_root / "materialization_idempotency.json",
        mode="copy",
    )

    assert plan["summary"]["ready"] == 2
    assert plan["summary"]["blocked"] == 0
    assert plan["summary"]["metadata_rows"] == 2
    assert plan["path_audit"]["bridge_plan"]["blocked"] == 0
    assert plan["path_audit"]["quarantine_plan"]["blocked"] == 0
    assert plan["safety"]["runtime_db_writes"] is False
    assert (package_root / "metadata.csv").exists()
    assert first["summary"]["copied"] == 2
    assert first["summary"]["blocked"] == 0
    assert first["summary"]["target_audio_files"] == 2
    assert first["path_audit"]["materialized"]["blocked"] == 0
    assert second["summary"]["already_present"] == 2
    assert second["summary"]["copied"] == 0
    assert second["summary"]["blocked"] == 0


def test_recording_quarantine_package_cli_plan_and_materialize(tmp_path: Path) -> None:
    product_root = tmp_path / "product_appliance"
    bridge_report = write_nested_bridge_report(product_root, count=1)
    package_root = product_root / "recording_quarantine_stage9"
    plan_path = package_root / "plan.json"
    materialize_out = package_root / "materialize.json"

    plan_rc = mango_office_recording_quarantine_package.main(
        [
            "--product-root",
            str(product_root),
            "plan",
            "--bridge-plan",
            str(bridge_report),
            "--package-root",
            str(package_root),
            "--quarantine-dir",
            str(package_root / "audio"),
            "--metadata-csv",
            str(package_root / "metadata.csv"),
            "--normalized-bridge-plan",
            str(package_root / "normalized_bridge.json"),
            "--out",
            str(plan_path),
        ]
    )
    materialize_rc = mango_office_recording_quarantine_package.main(
        [
            "--product-root",
            str(product_root),
            "materialize",
            "--plan",
            str(plan_path),
            "--out",
            str(materialize_out),
            "--mode",
            "copy",
        ]
    )

    assert plan_rc == 0
    assert materialize_rc == 0
    assert json.loads(plan_path.read_text(encoding="utf-8"))["summary"]["ready"] == 1
    assert json.loads(materialize_out.read_text(encoding="utf-8"))["summary"]["copied"] == 1


def test_recording_quarantine_package_refuses_bridge_plan_outside_product_root(tmp_path: Path) -> None:
    product_root = tmp_path / "product_appliance"
    outside = tmp_path / "bridge.json"
    outside.write_text(json.dumps({"items": []}), encoding="utf-8")

    with pytest.raises(ValueError, match="source_bridge_plan_path"):
        build_recording_quarantine_plan(
            source_bridge_plan_path=outside,
            product_root=product_root,
            package_root=product_root / "recording_quarantine_stage9",
            quarantine_dir=product_root / "recording_quarantine_stage9" / "audio",
            metadata_csv_path=product_root / "recording_quarantine_stage9" / "metadata.csv",
            plan_path=product_root / "recording_quarantine_stage9" / "plan.json",
            normalized_bridge_plan_path=product_root / "recording_quarantine_stage9" / "normalized_bridge.json",
        )


def test_recording_quarantine_package_refuses_source_audio_outside_product_root(tmp_path: Path) -> None:
    product_root = tmp_path / "product_appliance"
    bridge_report = write_nested_bridge_report(product_root, count=1)
    outside_audio = tmp_path / "outside.mp3"
    outside_audio.write_bytes(b"outside")
    data = json.loads(bridge_report.read_text(encoding="utf-8"))
    data["bridge"]["items"][0]["local_audio_path"] = str(outside_audio)
    data["bridge"]["items"][0]["checksum_sha256"] = file_sha256(outside_audio)
    bridge_report.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="unsafe paths"):
        build_recording_quarantine_plan(
            source_bridge_plan_path=bridge_report,
            product_root=product_root,
            package_root=product_root / "recording_quarantine_stage9",
            quarantine_dir=product_root / "recording_quarantine_stage9" / "audio",
            metadata_csv_path=product_root / "recording_quarantine_stage9" / "metadata.csv",
            plan_path=product_root / "recording_quarantine_stage9" / "plan.json",
            normalized_bridge_plan_path=product_root / "recording_quarantine_stage9" / "normalized_bridge.json",
        )


def write_nested_bridge_report(product_root: Path, count: int) -> Path:
    items = []
    for index in range(1, count + 1):
        audio = product_root / "recording_capture_downloads" / "recordings" / f"call-{index}.mp3"
        audio.parent.mkdir(parents=True, exist_ok=True)
        audio.write_bytes(f"audio-{index}".encode("utf-8"))
        items.append(
            {
                "event_key": f"foton:mango:CALL-{index}",
                "provider_call_id": f"CALL-{index}",
                "recording_id": f"rec-{index}",
                "status": "would_import",
                "reason": "validated_capture_not_found_in_source_dir_or_db",
                "started_at": "2026-05-07T06:00:00+00:00",
                "started_at_msk": "2026-05-07T09:00:00+03:00",
                "direction": "outbound",
                "client_phone": "+79990000000",
                "manager_ref": "101",
                "local_audio_path": str(audio),
                "size_bytes": audio.stat().st_size,
                "checksum_sha256": file_sha256(audio),
                "duration_sec": 30.0,
                "proposed_filename": f"2026-05-07__09-00-00__79990000000__mango_101_CALL-{index}.mp3",
                "proposed_metadata": {
                    "source": "mango_api_capture",
                    "tenant_id": "foton",
                    "provider": "mango",
                    "event_key": f"foton:mango:CALL-{index}",
                    "provider_call_id": f"CALL-{index}",
                    "recording_id": f"rec-{index}",
                    "started_at_msk": "2026-05-07T09:00:00+03:00",
                    "client_phone": "+79990000000",
                    "manager_ref": "101",
                    "direction": "outbound",
                    "duration_sec": 30.0,
                    "checksum_sha256": file_sha256(audio),
                },
            }
        )
    path = product_root / "recording_bridge_stage8" / "bridge_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"summary": {"would_import": count}, "bridge": {"items": items}}), encoding="utf-8")
    return path
