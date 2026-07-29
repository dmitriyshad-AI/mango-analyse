from __future__ import annotations

import shutil
import hashlib
from pathlib import Path

import pytest

from mango_mvp.customer_timeline import (
    approval_decisions,
    approved_context_pack,
    calls_two_processes,
    canonical_readonly_import,
    crm_export_package,
    ingestion,
    mail_stage2_ingest,
)
from mango_mvp.customer_timeline.safe_copy import file_sha256, safe_copy_prod_snapshot, sqlite_wal_path


def test_file_sha256_handles_empty_and_multi_chunk_files(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    multi_chunk = tmp_path / "multi"
    empty.write_bytes(b"")
    payload = b"a" * (1024 * 1024 + 17)
    multi_chunk.write_bytes(payload)

    assert file_sha256(empty) == hashlib.sha256(b"").hexdigest()
    assert file_sha256(multi_chunk) == hashlib.sha256(payload).hexdigest()
    assert file_sha256(str(multi_chunk)) == hashlib.sha256(payload).hexdigest()


def test_timeline_sha_helpers_reexport_one_canonical_implementation() -> None:
    assert ingestion.file_sha256 is file_sha256
    assert mail_stage2_ingest.file_sha256 is file_sha256
    assert approval_decisions.file_sha256 is file_sha256
    assert approved_context_pack.file_sha256 is file_sha256
    assert canonical_readonly_import.file_sha256 is file_sha256
    assert calls_two_processes.sha256_file is file_sha256
    assert crm_export_package._sha256_file is file_sha256


def test_safe_copy_prod_snapshot_copies_only_stable_sqlite_file(tmp_path: Path) -> None:
    source = tmp_path / "customer_timeline.sqlite"
    target = tmp_path / "copy" / "customer_timeline.sqlite"
    source.write_bytes(b"stable-db")

    manifest = safe_copy_prod_snapshot(source, target, sleep_fn=lambda _seconds: None)

    assert target.read_bytes() == b"stable-db"
    assert manifest["attempt_count"] == 1
    assert manifest["source_sha256"] == manifest["target_sha256"]
    assert manifest["wal_size"] == 0


def test_safe_copy_prod_snapshot_fails_after_bounded_non_empty_wal_retries(tmp_path: Path) -> None:
    source = tmp_path / "customer_timeline.sqlite"
    target = tmp_path / "copy.sqlite"
    source.write_bytes(b"stable-db")
    sqlite_wal_path(source).write_bytes(b"pending-wal")
    sleeps: list[float] = []

    with pytest.raises(RuntimeError, match="bounded retries"):
        safe_copy_prod_snapshot(source, target, retries=3, sleep_seconds=0.25, sleep_fn=sleeps.append)

    assert sleeps == [0.25, 0.25]
    assert not target.exists()


def test_safe_copy_prod_snapshot_fails_on_source_sha_drift(tmp_path: Path) -> None:
    source = tmp_path / "customer_timeline.sqlite"
    target = tmp_path / "copy.sqlite"
    source.write_bytes(b"before")

    def drifting_copy(src: Path, dst: Path) -> object:
        shutil.copy2(src, dst)
        src.write_bytes(b"after")
        return None

    with pytest.raises(RuntimeError, match="source file changed"):
        safe_copy_prod_snapshot(source, target, retries=1, copy_fn=drifting_copy, sleep_fn=lambda _seconds: None)

    assert target.read_bytes() == b"before"


def test_safe_copy_prod_snapshot_fails_on_target_hash_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "customer_timeline.sqlite"
    target = tmp_path / "copy.sqlite"
    source.write_bytes(b"source")

    def corrupting_copy(_src: Path, dst: Path) -> object:
        dst.write_bytes(b"corrupt")
        return None

    with pytest.raises(RuntimeError, match="target copy hash mismatched"):
        safe_copy_prod_snapshot(source, target, retries=1, copy_fn=corrupting_copy, sleep_fn=lambda _seconds: None)

    assert target.read_bytes() == b"corrupt"
