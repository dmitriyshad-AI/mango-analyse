from __future__ import annotations

import json
from pathlib import Path

import pytest

from mango_mvp.productization.recording_capture_download import (
    DOWNLOADED_RECORDING,
    PLAN_RECORDING_DOWNLOAD,
    SKIP_ALREADY_DOWNLOADED,
    audit_recording_capture_download,
    run_recording_capture_download,
)
from mango_mvp.productization.recording_capture_plan import PLAN_DOWNLOAD_DRY_RUN
from scripts import mango_office_recording_capture_download


class FakeDownloader:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Path]] = []

    def download(self, recording_id: str, target_path: Path) -> int:
        self.calls.append((recording_id, target_path))
        target_path.parent.mkdir(parents=True, exist_ok=True)
        payload = f"fake-mango-recording:{recording_id}".encode("utf-8")
        target_path.write_bytes(payload)
        return len(payload)


def test_recording_capture_download_executes_limited_manifest_item(tmp_path: Path) -> None:
    product_root = tmp_path / "product_appliance"
    source_plan = write_source_plan(product_root, count=2)
    recordings_dir = product_root / "recording_capture_downloads" / "recordings"
    download_manifest = product_root / "recording_capture_downloads" / "manifest.jsonl"
    out = product_root / "recording_capture_downloads" / "run_audit.json"
    downloader = FakeDownloader()

    report = run_recording_capture_download(
        source_plan_manifest_path=source_plan,
        product_root=product_root,
        recordings_dir=recordings_dir,
        download_manifest_path=download_manifest,
        out_path=out,
        downloader=downloader,
        execute=True,
        limit=1,
    )
    audit = audit_recording_capture_download(
        download_manifest_path=download_manifest,
        product_root=product_root,
        recordings_dir=recordings_dir,
        out_path=product_root / "recording_capture_downloads" / "verify.json",
    )

    assert report["summary"]["plan_items_seen"] == 2
    assert report["summary"]["selected_items"] == 1
    assert report["summary"]["downloaded_recording"] == 1
    assert report["summary"]["validation_ok"] is True
    assert report["safety"] == {
        "download_audio": True,
        "product_db_writes": False,
        "runtime_db_writes": False,
        "stable_runtime_writes": False,
        "run_asr": False,
        "run_ra": False,
        "write_crm": False,
    }
    assert len(downloader.calls) == 1
    row = read_jsonl(download_manifest)[0]
    assert row["action"] == DOWNLOADED_RECORDING
    assert row["size_bytes"] > 0
    assert row["checksum_sha256"]
    assert Path(row["local_audio_path"]).is_file()
    assert audit["summary"]["downloaded_latest_events"] == 1
    assert audit["summary"]["checksum_mismatches"] == 0
    assert audit["summary"]["validation_ok"] is True


def test_recording_capture_download_is_idempotent_for_existing_manifest(tmp_path: Path) -> None:
    product_root = tmp_path / "product_appliance"
    source_plan = write_source_plan(product_root, count=1)
    recordings_dir = product_root / "recording_capture_downloads" / "recordings"
    download_manifest = product_root / "recording_capture_downloads" / "manifest.jsonl"
    first_downloader = FakeDownloader()
    second_downloader = FakeDownloader()

    first = run_recording_capture_download(
        source_plan_manifest_path=source_plan,
        product_root=product_root,
        recordings_dir=recordings_dir,
        download_manifest_path=download_manifest,
        downloader=first_downloader,
        execute=True,
    )
    second = run_recording_capture_download(
        source_plan_manifest_path=source_plan,
        product_root=product_root,
        recordings_dir=recordings_dir,
        download_manifest_path=download_manifest,
        downloader=second_downloader,
        execute=True,
    )
    audit = audit_recording_capture_download(
        download_manifest_path=download_manifest,
        product_root=product_root,
        recordings_dir=recordings_dir,
    )

    assert first["summary"]["downloaded_recording"] == 1
    assert second["summary"]["skip_already_downloaded"] == 1
    assert audit["summary"]["available_latest_events"] == 1
    assert audit["summary"]["downloaded_latest_events"] == 0
    assert audit["summary"]["missing_files"] == 0
    assert audit["summary"]["checksum_mismatches"] == 0
    assert audit["summary"]["validation_ok"] is True
    assert len(first_downloader.calls) == 1
    assert len(second_downloader.calls) == 0
    assert [row["action"] for row in read_jsonl(download_manifest)] == [
        DOWNLOADED_RECORDING,
        SKIP_ALREADY_DOWNLOADED,
    ]


def test_recording_capture_download_cli_can_plan_without_execute(tmp_path: Path) -> None:
    product_root = tmp_path / "product_appliance"
    source_plan = write_source_plan(product_root, count=1)
    recordings_dir = product_root / "recording_capture_downloads" / "recordings"
    download_manifest = product_root / "recording_capture_downloads" / "manifest.jsonl"
    out = product_root / "recording_capture_downloads" / "run_audit.json"

    rc = mango_office_recording_capture_download.main(
        [
            "--product-root",
            str(product_root),
            "run",
            "--source-plan",
            str(source_plan),
            "--recordings-dir",
            str(recordings_dir),
            "--download-manifest",
            str(download_manifest),
            "--out",
            str(out),
            "--limit",
            "1",
        ]
    )

    assert rc == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["summary"]["execute"] is False
    assert data["summary"]["plan_recording_download"] == 1
    assert read_jsonl(download_manifest)[0]["action"] == PLAN_RECORDING_DOWNLOAD
    assert not recordings_dir.exists()


def test_recording_capture_download_audit_flags_checksum_mismatch(tmp_path: Path) -> None:
    product_root = tmp_path / "product_appliance"
    recordings_dir = product_root / "recording_capture_downloads" / "recordings"
    recordings_dir.mkdir(parents=True)
    audio = recordings_dir / "call.mp3"
    audio.write_bytes(b"real-content")
    manifest = product_root / "recording_capture_downloads" / "manifest.jsonl"
    write_jsonl(
        manifest,
        [
            {
                "schema_version": "recording_capture_download_v1",
                "action": DOWNLOADED_RECORDING,
                "event_key": "foton:mango:CALL-1",
                "recording_id": "rec-1",
                "local_audio_path": str(audio),
                "checksum_sha256": "wrong",
            }
        ],
    )

    audit = audit_recording_capture_download(
        download_manifest_path=manifest,
        product_root=product_root,
        recordings_dir=recordings_dir,
    )

    assert audit["summary"]["validation_ok"] is False
    assert audit["summary"]["checksum_mismatches"] == 1
    assert audit["summary"]["blocked"] == 1


def test_recording_capture_download_refuses_paths_outside_product_root(tmp_path: Path) -> None:
    product_root = tmp_path / "product_appliance"
    source_plan = write_source_plan(product_root, count=1)

    with pytest.raises(ValueError, match="download manifest"):
        run_recording_capture_download(
            source_plan_manifest_path=source_plan,
            product_root=product_root,
            recordings_dir=product_root / "recording_capture_downloads" / "recordings",
            download_manifest_path=tmp_path / "outside.jsonl",
        )


def write_source_plan(product_root: Path, count: int) -> Path:
    rows = []
    for index in range(count):
        call_id = f"CALL-{index + 1}"
        rows.append(
            {
                "schema_version": "recording_capture_plan_v1",
                "action": PLAN_DOWNLOAD_DRY_RUN,
                "tenant_id": "foton",
                "provider": "mango",
                "event_key": f"foton:mango:{call_id}",
                "provider_call_id": call_id,
                "capture_inbox_item_id": index + 1,
                "started_at": "2026-05-07T06:00:00+00:00",
                "ended_at": "2026-05-07T06:05:00+00:00",
                "direction": "outbound",
                "client_phone": "+79990000000",
                "manager_ref": "101",
                "recording_id": f"rec-{index + 1}",
                "recording_ref": f"rec-{index + 1}",
                "target_audio_path": str(
                    product_root
                    / "recording_capture_dry_run"
                    / "recordings"
                    / f"20260507T060000Z__mgr_101__call_{index + 1}__rec_{index + 1}.mp3"
                ),
            }
        )
    path = product_root / "recording_capture_dry_run" / "recording_capture_plan_stage6.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(path, rows)
    return path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
