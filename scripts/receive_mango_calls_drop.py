#!/usr/bin/env python3
"""Accept one sealed Mango calls drop on the Timeline host."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import sqlite3
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mango_mvp.productization.mango_calls_service_contract import (  # noqa: E402
    current_git_sha,
    git_worktree_is_clean,
)
from mango_mvp.productization.owner_only_io import path_has_cloud_marker  # noqa: E402

SCHEMA = "mango_calls_two_processes_v1"
DB_NAME = "mango_calls_ready.sqlite"
MANIFEST_NAME = "mango_calls_ready.manifest.json"
CONFIRMATION = "ACCEPT_MANGO_CALLS_REMOTE_DROP"
RESTORE_CONFIRMATION = "RESTORE_MANGO_CALLS_REMOTE_ROLLBACK"


def path_has_forbidden_part(path: Path) -> bool:
    parts = {part.casefold() for part in path.parts}
    return "stable_runtime" in parts or path_has_cloud_marker(path)


def assert_no_symlink_components(path: Path) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.exists() and current.is_symlink():
            raise RuntimeError("handoff path contains a symlink")


def secure_directory(path: Path, *, create: bool) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute() or path_has_forbidden_part(expanded):
        raise RuntimeError("handoff path is unsafe")
    assert_no_symlink_components(expanded)
    missing = not expanded.exists()
    if create:
        expanded.mkdir(parents=True, mode=0o700, exist_ok=True)
    if not expanded.is_dir() or expanded.is_symlink() or expanded.resolve() != expanded.absolute():
        raise RuntimeError("handoff directory is unsafe")
    if create:
        expanded.chmod(0o700)
        if missing:
            descriptor = os.open(expanded.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    return expanded.resolve()


def fsync_path(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def handoff_lock(pipeline_root: Path):
    root = secure_directory(pipeline_root, create=True)
    locks = secure_directory(root / "locks", create=True)
    lock_path = locks / "remote_drop.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        lock_path.chmod(0o600)
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("remote drop handoff is already running") from exc
        yield root


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sqlite_quick_check(path: Path) -> str:
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=60) as connection:
        connection.execute("PRAGMA query_only=ON")
        return str(connection.execute("PRAGMA quick_check").fetchone()[0])


def sqlite_integrity_check(path: Path) -> str:
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=60) as connection:
        connection.execute("PRAGMA query_only=ON")
        return str(connection.execute("PRAGMA integrity_check").fetchone()[0])


def load_package(directory: Path, expected_sha: str | None = None, *, exact: bool = True) -> tuple[Path, Mapping[str, Any]]:
    directory = directory.expanduser().resolve()
    db, manifest_path = directory / DB_NAME, directory / MANIFEST_NAME
    names = {path.name for path in directory.iterdir()} if directory.is_dir() else set()
    if not directory.is_dir() or (names != {DB_NAME, MANIFEST_NAME} if exact else not {DB_NAME, MANIFEST_NAME} <= names):
        raise RuntimeError("sealed drop must contain exactly DB and manifest")
    if any(not path.is_file() or path.is_symlink() for path in (db, manifest_path)):
        raise RuntimeError("sealed drop files must be regular files")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise RuntimeError("sealed drop manifest is invalid")
    actual_sha, actual_size = sha256_file(db), db.stat().st_size
    if (manifest.get("schema_version") != SCHEMA or manifest.get("status") != "ready"
            or manifest.get("sha256") != actual_sha or manifest.get("size_bytes") != actual_size
            or manifest.get("quick_check") != "ok" or sqlite_quick_check(db) != "ok"
            or sqlite_integrity_check(db) != "ok"):
        raise RuntimeError("sealed drop verification failed")
    if expected_sha is not None and actual_sha != expected_sha:
        raise RuntimeError("sealed drop SHA does not match expected SHA")
    return db, manifest


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_path(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def local_manifest(source: Mapping[str, Any], target_db: Path) -> dict[str, Any]:
    return {
        **source,
        "ready_db": str(target_db),
        "ready_mtime_ns": target_db.stat().st_mtime_ns,
        "received_at": datetime.now(timezone.utc).isoformat(),
        "remote_handoff": True,
    }


def stage_package(target_db: Path, target_manifest: Path, directory: Path) -> tuple[Path, Path]:
    descriptor, db_name = tempfile.mkstemp(prefix=f".{DB_NAME}.", suffix=".tmp", dir=directory)
    os.close(descriptor)
    temporary_db = Path(db_name)
    temporary_db.unlink()
    os.link(target_db, temporary_db)
    descriptor, name = tempfile.mkstemp(prefix=f".{MANIFEST_NAME}.", suffix=".tmp", dir=directory)
    manifest_temp = Path(name)
    try:
        os.fchmod(descriptor, 0o600)
        with target_manifest.open("rb") as source, os.fdopen(descriptor, "wb") as destination:
            shutil.copyfileobj(source, destination)
            destination.flush()
            os.fsync(destination.fileno())
    except Exception:
        temporary_db.unlink(missing_ok=True)
        if manifest_temp.exists():
            manifest_temp.unlink()
        raise
    return temporary_db, manifest_temp


def commit_staged_package(staged: tuple[Path, Path], directory: Path) -> None:
    temporary_db, manifest_temp = staged
    try:
        os.replace(temporary_db, directory / DB_NAME)
        os.replace(manifest_temp, directory / MANIFEST_NAME)
        fsync_path(directory)
    finally:
        temporary_db.unlink(missing_ok=True)
        manifest_temp.unlink(missing_ok=True)


def preserve_rollback(target_db: Path, target_manifest: Path) -> None:
    rollback = secure_directory(target_db.parent / "rollback", create=True)
    commit_staged_package(stage_package(target_db, target_manifest, rollback), rollback)


def cleanup_incoming(directory: Path) -> None:
    for name in (DB_NAME, MANIFEST_NAME):
        (directory / name).unlink(missing_ok=True)
    directory.rmdir()


def _accept_drop_locked(incoming: Path, root: Path, expected_sha: str, *, cleanup: bool) -> Mapping[str, Any]:
    source_db, source_manifest = load_package(incoming, expected_sha)
    drop = secure_directory(root / "drop", create=True)
    target_db, target_manifest = drop / DB_NAME, drop / MANIFEST_NAME
    if target_db.exists() and sha256_file(target_db) == expected_sha:
        if target_db.is_symlink() or sqlite_quick_check(target_db) != "ok":
            raise RuntimeError("existing target DB failed quick_check")
        target_db.chmod(0o600)
        atomic_json(target_manifest, local_manifest(source_manifest, target_db))
        status = "reused"
    else:
        if target_db.exists() or target_manifest.exists():
            if not target_db.exists() or not target_manifest.exists():
                raise RuntimeError("existing target drop is incomplete")
            load_package(drop, exact=False)
            preserve_rollback(target_db, target_manifest)
        descriptor, name = tempfile.mkstemp(prefix=f".{DB_NAME}.{expected_sha[:12]}.", suffix=".tmp", dir=drop)
        temporary = Path(name)
        try:
            os.fchmod(descriptor, 0o600)
            with source_db.open("rb") as source, os.fdopen(descriptor, "wb") as destination:
                shutil.copyfileobj(source, destination)
                destination.flush()
                os.fsync(destination.fileno())
            if sha256_file(temporary) != expected_sha or sqlite_quick_check(temporary) != "ok":
                raise RuntimeError("copied target DB verification failed")
            os.replace(temporary, target_db)
            fsync_path(drop)
            atomic_json(target_manifest, local_manifest(source_manifest, target_db))
        finally:
            if temporary.exists():
                temporary.unlink()
        status = "accepted"
    if cleanup:
        cleanup_incoming(incoming)
    return {"status": status, "sha256": expected_sha, "size_bytes": target_db.stat().st_size}


def accept_drop(incoming: Path, pipeline_root: Path, expected_sha: str, *, execute: bool,
                confirmation: str, cleanup: bool = False, lock_held: bool = False) -> Mapping[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
        raise RuntimeError("expected SHA is invalid")
    source = secure_directory(incoming, create=False)
    expanded_root = pipeline_root.expanduser()
    if not expanded_root.is_absolute() or path_has_forbidden_part(expanded_root):
        raise RuntimeError("pipeline root is unsafe")
    source_db, _ = load_package(source, expected_sha)
    if not execute:
        return {"status": "dry_run", "sha256": expected_sha, "size_bytes": source_db.stat().st_size}
    if confirmation != CONFIRMATION:
        raise RuntimeError("explicit receiver confirmation is required")
    if lock_held:
        root = secure_directory(expanded_root, create=True)
        return _accept_drop_locked(source, root, expected_sha, cleanup=cleanup)
    with handoff_lock(expanded_root) as root:
        return _accept_drop_locked(source, root, expected_sha, cleanup=cleanup)


def restore_rollback(pipeline_root: Path, *, execute: bool, confirmation: str) -> Mapping[str, Any]:
    if not execute:
        return {"status": "dry_run", "operation": "restore_rollback"}
    if confirmation != RESTORE_CONFIRMATION:
        raise RuntimeError("explicit rollback confirmation is required")
    with handoff_lock(pipeline_root) as root:
        drop = secure_directory(root / "drop", create=False)
        rollback = secure_directory(drop / "rollback", create=False)
        source_db, source_manifest = load_package(rollback, exact=False)
        target_db, target_manifest = drop / DB_NAME, drop / MANIFEST_NAME
        try:
            load_package(drop, exact=False)
        except (OSError, ValueError, RuntimeError):
            current_staged = None
        else:
            # A valid current generation must be preserved or restoration stops
            # before replacing either target file.
            current_staged = stage_package(target_db, target_manifest, rollback)
        descriptor, name = tempfile.mkstemp(prefix=f".{DB_NAME}.restore.", suffix=".tmp", dir=drop)
        temporary = Path(name)
        try:
            os.fchmod(descriptor, 0o600)
            with source_db.open("rb") as source, os.fdopen(descriptor, "wb") as destination:
                shutil.copyfileobj(source, destination)
                destination.flush()
                os.fsync(destination.fileno())
            expected_sha = str(source_manifest["sha256"])
            if sha256_file(temporary) != expected_sha or sqlite_integrity_check(temporary) != "ok":
                raise RuntimeError("rollback DB verification failed")
            os.replace(temporary, target_db)
            fsync_path(drop)
            atomic_json(target_manifest, local_manifest(source_manifest, target_db))
            load_package(drop, expected_sha, exact=False)
            if current_staged is not None:
                commit_staged_package(current_staged, rollback)
            return {"status": "restored", "sha256": expected_sha, "size_bytes": target_db.stat().st_size}
        finally:
            if temporary.exists():
                temporary.unlink()
            if current_staged is not None:
                for staged in current_staged:
                    staged.unlink(missing_ok=True)


def verify_repo(repo: Path, expected_sha: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", expected_sha):
        raise RuntimeError("expected code SHA is invalid")
    try:
        resolved_repo = repo.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError("receiver repository root is invalid") from exc
    if resolved_repo != ROOT.resolve(strict=True):
        raise RuntimeError("receiver repository root does not match executable")
    try:
        head = current_git_sha(resolved_repo)
    except (OSError, RuntimeError):
        head = ""
    if head != expected_sha or not git_worktree_is_clean(resolved_repo):
        raise RuntimeError("receiver repository revision is not exact and clean")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Accept a verified remote Mango calls drop.")
    parser.add_argument("--incoming-dir", type=Path)
    parser.add_argument("--pipeline-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--expected-sha", default="")
    parser.add_argument("--expected-code-sha", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirmation", default="")
    parser.add_argument("--cleanup-incoming", action="store_true")
    parser.add_argument("--restore-rollback", action="store_true")
    args = parser.parse_args(argv)
    try:
        verify_repo(args.repo_root.resolve(), args.expected_code_sha)
        if args.restore_rollback:
            result = restore_rollback(args.pipeline_root, execute=args.execute, confirmation=args.confirmation)
        else:
            if args.incoming_dir is None:
                raise RuntimeError("incoming directory is required")
            result = accept_drop(
                args.incoming_dir, args.pipeline_root, args.expected_sha, execute=args.execute,
                confirmation=args.confirmation, cleanup=args.cleanup_incoming,
            )
    except Exception as exc:
        result = {"status": "failed", "stop_reason": f"receiver_exception:{type(exc).__name__}"}
        print(json.dumps(result, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
