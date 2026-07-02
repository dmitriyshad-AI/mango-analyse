from __future__ import annotations

import hashlib
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional


CopyFn = Callable[[Path, Path], object]
SleepFn = Callable[[float], object]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sqlite_wal_path(sqlite_path: Path) -> Path:
    return Path(f"{sqlite_path}-wal")


def safe_copy_prod_snapshot(
    src: Path | str,
    dst: Path | str,
    *,
    retries: int = 3,
    sleep_seconds: float = 1.0,
    copy_fn: Optional[CopyFn] = None,
    sleep_fn: SleepFn = time.sleep,
) -> Mapping[str, Any]:
    """Copy a SQLite file only when the main file is stable and WAL is empty.

    This helper intentionally does not open SQLite. It must never checkpoint the
    source database; if WAL is non-empty or the source hash drifts, the caller
    gets a hard failure after bounded retries.
    """

    source = Path(src).expanduser().resolve(strict=True)
    target = Path(dst).expanduser().resolve(strict=False)
    if source == target:
        raise ValueError("safe_copy_prod_snapshot source and destination must differ")
    if retries < 1:
        raise ValueError("retries must be positive")
    copier = copy_fn or shutil.copy2
    target.parent.mkdir(parents=True, exist_ok=True)
    attempts: list[Mapping[str, Any]] = []
    wal = sqlite_wal_path(source)
    for attempt in range(1, retries + 1):
        wal_exists_before = wal.exists()
        wal_size_before = wal.stat().st_size if wal_exists_before else 0
        manifest_attempt: dict[str, Any] = {
            "attempt": attempt,
            "wal_path": str(wal),
            "wal_exists_before": wal_exists_before,
            "wal_size_before": wal_size_before,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        if wal_size_before != 0:
            manifest_attempt["status"] = "retry_wal_non_empty"
            attempts.append(manifest_attempt)
            if attempt < retries:
                sleep_fn(sleep_seconds)
            continue
        sha_before = file_sha256(source)
        manifest_attempt["source_sha256_before"] = sha_before
        manifest_attempt["source_size_before"] = source.stat().st_size
        copier(source, target)
        sha_after = file_sha256(source)
        wal_exists_after = wal.exists()
        wal_size_after = wal.stat().st_size if wal_exists_after else 0
        manifest_attempt.update(
            {
                "source_sha256_after": sha_after,
                "source_size_after": source.stat().st_size,
                "target_sha256": file_sha256(target),
                "target_size": target.stat().st_size,
                "wal_exists_after": wal_exists_after,
                "wal_size_after": wal_size_after,
            }
        )
        target_sha = str(manifest_attempt["target_sha256"])
        if sha_before == sha_after and target_sha == sha_after and wal_size_after == 0:
            manifest_attempt["status"] = "copied"
            attempts.append(manifest_attempt)
            return {
                "kind": "safe_copy_prod_snapshot",
                "source_path": str(source),
                "target_path": str(target),
                "attempts": attempts,
                "attempt_count": attempt,
                "source_sha256": sha_after,
                "target_sha256": manifest_attempt["target_sha256"],
                "wal_path": str(wal),
                "wal_size": wal_size_after,
                "copied_at": datetime.now(timezone.utc).isoformat(),
            }
        if sha_before != sha_after:
            manifest_attempt["status"] = "retry_source_changed"
        elif target_sha != sha_after:
            manifest_attempt["status"] = "retry_target_mismatch"
        else:
            manifest_attempt["status"] = "retry_wal_changed"
        attempts.append(manifest_attempt)
        if attempt < retries:
            sleep_fn(sleep_seconds)
    raise RuntimeError(
        "safe_copy_prod_snapshot failed: source WAL was non-empty, source file changed, "
        "or target copy hash mismatched after bounded retries; "
        + f"attempts={attempts!r}"
    )


__all__ = ["file_sha256", "safe_copy_prod_snapshot", "sqlite_wal_path"]
