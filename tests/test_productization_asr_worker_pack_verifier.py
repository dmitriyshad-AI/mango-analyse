from __future__ import annotations

import json
from pathlib import Path

import pytest

from mango_mvp.productization.asr_worker_pack import build_asr_worker_pack
from mango_mvp.productization.asr_worker_pack_verifier import verify_asr_worker_pack
from scripts import mango_office_asr_worker_pack_verify
from tests.test_productization_asr_worker_pack import build_handoff_manifest
from tests.test_productization_processing_handoff import read_jsonl


def test_asr_worker_pack_verifier_accepts_clean_pack(tmp_path: Path) -> None:
    product_root, pack_root, pack_manifest = build_pack(tmp_path, count=2)

    report = verify_asr_worker_pack(
        product_root=product_root,
        pack_root=pack_root,
        pack_manifest_path=pack_manifest,
        out_path=pack_root / "verify.json",
    )

    assert report["summary"]["validation_ok"] is True
    assert report["summary"]["ready_items"] == 2
    assert report["summary"]["blocked"] == 0
    assert report["summary"]["pack_audio_files"] == 2
    assert report["action_counts"] == {"VERIFY_ASR_WORKER_PACK_ITEM": 2}
    assert report["readiness_gate"]["ready_for_worker"] is True
    assert report["readiness_gate"]["worker_may_run_asr"] is False
    assert report["safety"]["read_only"] is True
    assert report["safety"]["run_asr"] is False


def test_asr_worker_pack_verifier_is_idempotent(tmp_path: Path) -> None:
    product_root, pack_root, pack_manifest = build_pack(tmp_path, count=2)

    first = verify_asr_worker_pack(
        product_root=product_root,
        pack_root=pack_root,
        pack_manifest_path=pack_manifest,
        out_path=pack_root / "verify.json",
    )
    second = verify_asr_worker_pack(
        product_root=product_root,
        pack_root=pack_root,
        pack_manifest_path=pack_manifest,
        out_path=pack_root / "verify_idempotency.json",
    )

    assert first["summary"]["manifest_sha256"] == second["summary"]["manifest_sha256"]
    assert second["summary"]["validation_ok"] is True


def test_asr_worker_pack_verifier_blocks_checksum_mismatch(tmp_path: Path) -> None:
    product_root, pack_root, pack_manifest = build_pack(tmp_path, count=1)
    row = read_jsonl(pack_manifest)[0]
    (pack_root / row["audio_rel_path"]).write_bytes(b"changed-audio")

    report = verify_asr_worker_pack(
        product_root=product_root,
        pack_root=pack_root,
        pack_manifest_path=pack_manifest,
    )

    assert report["summary"]["validation_ok"] is False
    assert report["summary"]["blocked"] == 1
    assert "audio_sha256_mismatch" in report["items"][0]["blocked_reasons"]


def test_asr_worker_pack_verifier_blocks_missing_audio(tmp_path: Path) -> None:
    product_root, pack_root, pack_manifest = build_pack(tmp_path, count=1)
    row = read_jsonl(pack_manifest)[0]
    (pack_root / row["audio_rel_path"]).unlink()

    report = verify_asr_worker_pack(
        product_root=product_root,
        pack_root=pack_root,
        pack_manifest_path=pack_manifest,
    )

    assert report["summary"]["validation_ok"] is False
    assert report["summary"]["blocked"] == 2
    assert "audio_missing" in report["items"][0]["blocked_reasons"]
    assert report["pack_audit"]["blocked_reasons"]["missing_expected_files"] == 1


def test_asr_worker_pack_verifier_blocks_path_traversal(tmp_path: Path) -> None:
    product_root, pack_root, pack_manifest = build_pack(tmp_path, count=1)
    rows = read_jsonl(pack_manifest)
    rows[0]["audio_rel_path"] = "../outside.mp3"
    write_jsonl(pack_manifest, rows)

    report = verify_asr_worker_pack(
        product_root=product_root,
        pack_root=pack_root,
        pack_manifest_path=pack_manifest,
    )

    assert report["summary"]["validation_ok"] is False
    assert report["summary"]["blocked"] >= 1
    assert "audio_rel_path_must_not_traverse" in report["items"][0]["blocked_reasons"]
    assert "audio_outside_pack_root" in report["items"][0]["blocked_reasons"]


def test_asr_worker_pack_verifier_refuses_outside_and_stable_runtime_paths(tmp_path: Path) -> None:
    product_root, pack_root, pack_manifest = build_pack(tmp_path, count=1)

    with pytest.raises(ValueError, match="product root"):
        verify_asr_worker_pack(
            product_root=product_root,
            pack_root=tmp_path / "outside",
            pack_manifest_path=pack_manifest,
        )
    with pytest.raises(ValueError, match="stable_runtime"):
        verify_asr_worker_pack(
            product_root=product_root,
            pack_root=product_root / "stable_runtime" / "pack",
            pack_manifest_path=product_root / "stable_runtime" / "pack" / "manifest.jsonl",
        )


def test_asr_worker_pack_verify_script_writes_report(tmp_path: Path) -> None:
    product_root, pack_root, pack_manifest = build_pack(tmp_path, count=1)
    out = pack_root / "verify.json"

    rc = mango_office_asr_worker_pack_verify.main(
        [
            "--product-root",
            str(product_root),
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
    assert data["summary"]["ready_items"] == 1
    assert data["safety"]["read_only"] is True
    assert data["safety"]["run_asr"] is False


def build_pack(tmp_path: Path, count: int) -> tuple[Path, Path, Path]:
    product_root, source_manifest = build_handoff_manifest(tmp_path, count=count)
    pack_root = product_root / "asr_worker_pack_stage13"
    pack_manifest = pack_root / "asr_worker_input_manifest.jsonl"
    report = build_asr_worker_pack(
        source_manifest_path=source_manifest,
        product_root=product_root,
        pack_root=pack_root,
        pack_manifest_path=pack_manifest,
        out_path=pack_root / "pack.json",
    )
    assert report["summary"]["validation_ok"] is True
    return product_root, pack_root, pack_manifest


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
