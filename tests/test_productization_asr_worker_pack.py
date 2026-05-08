from __future__ import annotations

import json
from pathlib import Path

import pytest

from mango_mvp.productization.asr_worker_pack import build_asr_worker_pack
from mango_mvp.productization.processing_handoff import build_processing_handoff_dry_run
from scripts import mango_office_asr_worker_pack
from tests.test_productization_processing_handoff import build_asset_db, read_jsonl, sqlite_scalar


def test_asr_worker_pack_copies_audio_and_is_idempotent(tmp_path: Path) -> None:
    product_root, source_manifest = build_handoff_manifest(tmp_path, count=2)
    pack_root = product_root / "asr_worker_pack_stage13"
    pack_manifest = pack_root / "asr_worker_input_manifest.jsonl"

    first = build_asr_worker_pack(
        source_manifest_path=source_manifest,
        product_root=product_root,
        pack_root=pack_root,
        pack_manifest_path=pack_manifest,
        out_path=pack_root / "audit.json",
    )
    second = build_asr_worker_pack(
        source_manifest_path=source_manifest,
        product_root=product_root,
        pack_root=pack_root,
        pack_manifest_path=pack_manifest,
        out_path=pack_root / "audit_idempotency.json",
    )

    assert first["summary"]["validation_ok"] is True
    assert first["summary"]["copied"] == 2
    assert first["summary"]["pack_audio_files"] == 2
    assert first["action_counts"] == {"PACK_ASR_WORKER_ITEM": 2}
    assert second["summary"]["validation_ok"] is True
    assert second["summary"]["already_present"] == 2
    assert second["summary"]["manifest_sha256"] == first["summary"]["manifest_sha256"]
    assert second["action_counts"] == {"SKIP_ALREADY_PACKED": 2}
    rows = read_jsonl(pack_manifest)
    assert len(rows) == 2
    assert rows[0]["schema_version"] == "asr_worker_pack_v1"
    assert rows[0]["audio_rel_path"].startswith("audio/")
    assert rows[0]["planned_outputs_rel"]["transcript_json"].startswith("outputs/")
    assert (pack_root / rows[0]["audio_rel_path"]).exists()


def test_asr_worker_pack_dry_run_writes_manifest_without_audio(tmp_path: Path) -> None:
    product_root, source_manifest = build_handoff_manifest(tmp_path, count=1)
    pack_root = product_root / "asr_worker_pack_stage13"

    report = build_asr_worker_pack(
        source_manifest_path=source_manifest,
        product_root=product_root,
        pack_root=pack_root,
        pack_manifest_path=pack_root / "asr_worker_input_manifest.jsonl",
        out_path=pack_root / "audit.json",
        dry_run=True,
    )

    assert report["summary"]["validation_ok"] is True
    assert report["summary"]["manifest_rows"] == 1
    assert report["summary"]["pack_audio_files"] == 0
    assert report["safety"]["copies_audio"] is False
    assert not (pack_root / "audio").exists()


def test_asr_worker_pack_blocks_checksum_mismatch(tmp_path: Path) -> None:
    product_root, source_manifest = build_handoff_manifest(tmp_path, count=1)
    audio_path = Path(sqlite_scalar(product_root / "recording_quarantine_stage10" / "recording_asset_ingest.sqlite", "select audio_path from captured_recording_assets limit 1"))
    audio_path.write_bytes(b"changed-audio")

    report = build_asr_worker_pack(
        source_manifest_path=source_manifest,
        product_root=product_root,
        pack_root=product_root / "asr_worker_pack_stage13",
        pack_manifest_path=product_root / "asr_worker_pack_stage13" / "asr_worker_input_manifest.jsonl",
    )

    assert report["summary"]["validation_ok"] is False
    assert report["summary"]["blocked"] == 1
    assert report["action_counts"] == {"BLOCK_ASR_WORKER_PACK": 1}
    assert "checksum_sha256_mismatch" in report["items"][0]["blocked_reasons"]


def test_asr_worker_pack_refuses_outside_and_stable_runtime_paths(tmp_path: Path) -> None:
    product_root, source_manifest = build_handoff_manifest(tmp_path, count=1)

    with pytest.raises(ValueError, match="product root"):
        build_asr_worker_pack(
            source_manifest_path=source_manifest,
            product_root=product_root,
            pack_root=tmp_path / "outside_pack",
            pack_manifest_path=product_root / "asr_worker_pack_stage13" / "asr_worker_input_manifest.jsonl",
        )
    with pytest.raises(ValueError, match="stable_runtime"):
        build_asr_worker_pack(
            source_manifest_path=source_manifest,
            product_root=product_root,
            pack_root=product_root / "stable_runtime" / "pack",
            pack_manifest_path=product_root / "stable_runtime" / "pack" / "manifest.jsonl",
        )


def test_asr_worker_pack_script_writes_report(tmp_path: Path) -> None:
    product_root, source_manifest = build_handoff_manifest(tmp_path, count=1)
    pack_root = product_root / "asr_worker_pack_stage13"
    out = pack_root / "audit.json"
    pack_manifest = pack_root / "asr_worker_input_manifest.jsonl"

    rc = mango_office_asr_worker_pack.main(
        [
            "--product-root",
            str(product_root),
            "--source-manifest",
            str(source_manifest),
            "--pack-root",
            str(pack_root),
            "--pack-manifest",
            str(pack_manifest),
            "--out",
            str(out),
        ]
    )

    assert rc == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["summary"]["validation_ok"] is True
    assert data["summary"]["copied"] == 1
    assert data["safety"]["run_asr"] is False
    assert data["safety"]["runtime_db_writes"] is False
    assert pack_manifest.exists()


def build_handoff_manifest(tmp_path: Path, count: int) -> tuple[Path, Path]:
    product_root, asset_db = build_asset_db(tmp_path, count=count)
    out_dir = product_root / "processing_handoff_stage12"
    manifest = out_dir / "asr_handoff_manifest.jsonl"
    report = build_processing_handoff_dry_run(
        asset_db_path=asset_db,
        product_root=product_root,
        out_dir=out_dir,
        manifest_path=manifest,
        out_path=out_dir / "audit.json",
        package_ref="recording_quarantine_stage9",
    )
    assert report["summary"]["validation_ok"] is True
    return product_root, manifest
