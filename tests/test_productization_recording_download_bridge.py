from __future__ import annotations

import json
from pathlib import Path

import pytest

from mango_mvp.productization.capture_staging import file_sha256
from mango_mvp.productization.recording_capture_download import (
    DOWNLOADED_RECORDING,
    SKIP_ALREADY_DOWNLOADED,
)
from mango_mvp.productization.recording_download_bridge import (
    build_recording_download_bridge_dry_run,
)
from scripts import mango_office_recording_bridge_dry_run


def test_recording_download_bridge_converts_available_downloads_to_would_import(tmp_path: Path) -> None:
    product_root = tmp_path / "product_appliance"
    manifest = write_download_manifest(product_root, actions=[DOWNLOADED_RECORDING, SKIP_ALREADY_DOWNLOADED])
    capture_manifest = product_root / "recording_bridge_stage8" / "capture_manifest.jsonl"
    bridge_plan = product_root / "recording_bridge_stage8" / "bridge_plan.json"
    csv_path = product_root / "recording_bridge_stage8" / "bridge_plan.csv"

    report = build_recording_download_bridge_dry_run(
        download_manifest_path=manifest,
        product_root=product_root,
        capture_manifest_path=capture_manifest,
        bridge_plan_path=bridge_plan,
        source_dir=product_root / "recording_bridge_stage8" / "empty_source",
        csv_path=csv_path,
    )

    assert report["summary"]["download_manifest_rows"] == 2
    assert report["summary"]["latest_available_events"] == 2
    assert report["summary"]["converted_capture_manifest_rows"] == 2
    assert report["summary"]["would_import"] == 2
    assert report["summary"]["blocked"] == 0
    assert report["summary"]["validation_ok"] is True
    assert report["bridge"]["summary"]["bridge_status_counts"] == {"would_import": 2}
    assert capture_manifest.exists()
    assert csv_path.exists()
    rows = read_jsonl(capture_manifest)
    assert {row["status"] for row in rows} == {"downloaded"}
    assert {row["source_download_action"] for row in rows} == {DOWNLOADED_RECORDING, SKIP_ALREADY_DOWNLOADED}


def test_recording_download_bridge_cli_writes_report(tmp_path: Path) -> None:
    product_root = tmp_path / "product_appliance"
    manifest = write_download_manifest(product_root, actions=[DOWNLOADED_RECORDING])
    out = product_root / "recording_bridge_stage8" / "bridge_plan.json"

    rc = mango_office_recording_bridge_dry_run.main(
        [
            "--product-root",
            str(product_root),
            "--download-manifest",
            str(manifest),
            "--capture-manifest",
            str(product_root / "recording_bridge_stage8" / "capture_manifest.jsonl"),
            "--out",
            str(out),
            "--csv-out",
            str(product_root / "recording_bridge_stage8" / "bridge_plan.csv"),
            "--source-dir",
            str(product_root / "recording_bridge_stage8" / "empty_source"),
        ]
    )

    assert rc == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["summary"]["would_import"] == 1
    assert data["summary"]["validation_ok"] is True


def test_recording_download_bridge_refuses_outputs_outside_product_root(tmp_path: Path) -> None:
    product_root = tmp_path / "product_appliance"
    manifest = write_download_manifest(product_root, actions=[DOWNLOADED_RECORDING])

    with pytest.raises(ValueError, match="bridge plan"):
        build_recording_download_bridge_dry_run(
            download_manifest_path=manifest,
            product_root=product_root,
            capture_manifest_path=product_root / "recording_bridge_stage8" / "capture_manifest.jsonl",
            bridge_plan_path=tmp_path / "outside.json",
            source_dir=product_root / "recording_bridge_stage8" / "empty_source",
        )


def write_download_manifest(product_root: Path, actions: list[str]) -> Path:
    recordings_dir = product_root / "recording_capture_downloads" / "recordings"
    rows = []
    for index, action in enumerate(actions, 1):
        audio = recordings_dir / f"call-{index}.mp3"
        audio.parent.mkdir(parents=True, exist_ok=True)
        audio.write_bytes(f"audio-{index}".encode("utf-8"))
        rows.append(
            {
                "schema_version": "recording_capture_download_v1",
                "created_at": "2026-05-07T07:00:00+00:00",
                "action": action,
                "tenant_id": "foton",
                "provider": "mango",
                "event_key": f"foton:mango:CALL-{index}",
                "provider_call_id": f"CALL-{index}",
                "recording_id": f"rec-{index}",
                "recording_ref": f"rec-{index}",
                "started_at": "2026-05-07T06:00:00+00:00",
                "ended_at": "2026-05-07T06:00:30+00:00",
                "direction": "outbound",
                "client_phone": "+79990000000",
                "manager_ref": "101",
                "local_audio_path": str(audio),
                "size_bytes": audio.stat().st_size,
                "checksum_sha256": file_sha256(audio),
                "duration_sec": 30.0,
                "codec_name": "mp3",
                "channels": 2,
                "sample_rate": 8000,
                "download_audio": action == DOWNLOADED_RECORDING,
                "run_asr": False,
                "run_ra": False,
                "write_runtime_db": False,
                "write_crm": False,
            }
        )
    manifest = product_root / "recording_capture_downloads" / "recording_download_manifest_stage8.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    return manifest


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
