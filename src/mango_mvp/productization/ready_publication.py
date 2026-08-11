from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import stat
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Optional


READY_PUBLICATION_SCHEMA = "mango_calls_ready_publication_v1"
Checkpoint = Optional[Callable[[str], None]]


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_private_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    info = os.lstat(path)
    if (
        not stat.S_ISDIR(info.st_mode)
        or path.is_symlink()
        or info.st_uid != os.getuid()
    ):
        raise RuntimeError(f"ready publication directory is unsafe: {path}")
    path.chmod(0o700)


def _sha256_regular(path: Path, *, label: str) -> str:
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0),
    )
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.getuid()
            or before.st_nlink != 1
        ):
            raise RuntimeError(f"{label} is not a private regular file")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(descriptor)
        current = os.stat(path, follow_symlinks=False)
        identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        if identity != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) or (current.st_dev, current.st_ino) != (after.st_dev, after.st_ino):
            raise RuntimeError(f"{label} changed while hashing")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _digest_or_none(path: Path, *, label: str) -> Optional[str]:
    if not os.path.lexists(path):
        return None
    return _sha256_regular(path, label=label)


def _write_private_bytes(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(descriptor)
        os.fchmod(descriptor, 0o600)
    finally:
        os.close(descriptor)


def _write_private_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            os.fchmod(handle.fileno(), 0o600)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if os.path.lexists(path) and path.is_symlink():
            raise RuntimeError("ready publication JSON target is a symlink")
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _read_private_json(path: Path, *, label: str) -> Mapping[str, Any]:
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0),
    )
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.getuid()
            or before.st_nlink != 1
        ):
            raise RuntimeError(f"{label} is not a private regular file")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
        current = os.stat(path, follow_symlinks=False)
        if (
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or (current.st_dev, current.st_ino) != (after.st_dev, after.st_ino)
        ):
            raise RuntimeError(f"{label} changed while reading")
        raw = b"".join(chunks)
    finally:
        os.close(descriptor)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} is invalid") from exc
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} is invalid")
    return value


def _paths(ready_db: Path) -> Mapping[str, Path]:
    stem = f".{ready_db.name}.publication"
    transaction = ready_db.parent / f"{stem}.txn"
    return {
        "manifest": ready_db.with_suffix(".manifest.json"),
        "lock": ready_db.parent / f"{stem}.lock",
        "journal": ready_db.parent / f"{stem}.journal.json",
        "transaction": transaction,
        "new_db": transaction / "new.sqlite",
        "new_manifest": transaction / "new.manifest.json",
        "old_db": transaction / "old.sqlite",
        "old_manifest": transaction / "old.manifest.json",
    }


@contextmanager
def ready_publication_lock(ready_db: Path) -> Iterator[None]:
    paths = _paths(ready_db)
    _ensure_private_directory(ready_db.parent)
    descriptor = os.open(
        paths["lock"],
        os.O_RDWR
        | os.O_CREAT
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    handle = os.fdopen(descriptor, "a+b")
    try:
        opened = os.fstat(handle.fileno())
        current = os.stat(paths["lock"], follow_symlinks=False)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_uid != os.getuid()
            or opened.st_nlink != 1
            or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)
        ):
            raise RuntimeError("ready publication lock is unsafe")
        os.fchmod(handle.fileno(), 0o600)
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def inspect_ready_publication(ready_db: Path) -> Mapping[str, Any]:
    paths = _paths(ready_db)
    if not os.path.lexists(paths["journal"]):
        return {"status": "clean", "recovery_required": False}
    if os.path.lexists(paths["lock"]):
        try:
            lock_descriptor = os.open(
                paths["lock"],
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0),
            )
        except FileNotFoundError:
            if os.path.lexists(paths["lock"]):
                raise
        else:
            try:
                opened = os.fstat(lock_descriptor)
                try:
                    current = os.stat(paths["lock"], follow_symlinks=False)
                except FileNotFoundError:
                    current = None
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or opened.st_uid != os.getuid()
                    or opened.st_nlink != 1
                    or (
                        current is not None
                        and (opened.st_dev, opened.st_ino)
                        != (current.st_dev, current.st_ino)
                    )
                ):
                    raise RuntimeError("ready publication lock is unsafe")
                shared_lock_acquired = False
                try:
                    fcntl.flock(
                        lock_descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB
                    )
                    shared_lock_acquired = True
                except BlockingIOError:
                    return {
                        "status": "publication_active",
                        "recovery_required": False,
                    }
                finally:
                    if shared_lock_acquired:
                        try:
                            fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
                        except OSError:
                            pass
            finally:
                os.close(lock_descriptor)
    try:
        journal = _read_private_json(
            paths["journal"], label="ready publication journal"
        )
    except FileNotFoundError:
        if not os.path.lexists(paths["journal"]):
            return {"status": "clean", "recovery_required": False}
        raise
    if (
        journal.get("schema_version") != READY_PUBLICATION_SCHEMA
        or journal.get("ready_db") != ready_db.name
        or journal.get("ready_manifest") != paths["manifest"].name
    ):
        raise RuntimeError("ready publication journal identity is invalid")
    return {
        "status": "recovery_required",
        "recovery_required": True,
        "new_db_sha256": (journal.get("new") or {}).get("db_sha256"),
    }


def _copy_private(source: Path, target: Path) -> None:
    source_descriptor = os.open(
        source,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    target_descriptor = -1
    try:
        source_before = os.fstat(source_descriptor)
        if (
            not stat.S_ISREG(source_before.st_mode)
            or source_before.st_uid != os.getuid()
            or source_before.st_nlink != 1
        ):
            raise RuntimeError("ready publication source is unsafe")
        target_descriptor = os.open(
            target,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        digest = hashlib.sha256()
        while chunk := os.read(source_descriptor, 1024 * 1024):
            digest.update(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(target_descriptor, view)
                view = view[written:]
        os.fsync(target_descriptor)
        os.fchmod(target_descriptor, 0o600)
        os.utime(
            target,
            ns=(source_before.st_atime_ns, source_before.st_mtime_ns),
            follow_symlinks=False,
        )
        os.fsync(target_descriptor)
        source_after = os.fstat(source_descriptor)
        current = os.stat(source, follow_symlinks=False)
        if (
            (source_before.st_dev, source_before.st_ino, source_before.st_size,
             source_before.st_mtime_ns)
            != (source_after.st_dev, source_after.st_ino, source_after.st_size,
                source_after.st_mtime_ns)
            or (current.st_dev, current.st_ino)
            != (source_after.st_dev, source_after.st_ino)
        ):
            raise RuntimeError("ready publication source changed while copying")
        source_sha = digest.hexdigest()
    finally:
        if target_descriptor >= 0:
            os.close(target_descriptor)
        os.close(source_descriptor)
    if _sha256_regular(target, label="ready publication copy") != source_sha:
        raise RuntimeError("ready publication copy verification failed")


def _remove_transaction(paths: Mapping[str, Path]) -> None:
    paths["journal"].unlink(missing_ok=True)
    transaction = paths["transaction"]
    if os.path.lexists(transaction):
        info = os.lstat(transaction)
        if (
            not stat.S_ISDIR(info.st_mode)
            or transaction.is_symlink()
            or info.st_uid != os.getuid()
        ):
            raise RuntimeError("ready publication transaction directory is unsafe")
        shutil.rmtree(transaction)
    _fsync_directory(paths["journal"].parent)


def _replace_and_sync(source: Path, target: Path) -> None:
    if os.path.lexists(target) and target.is_symlink():
        raise RuntimeError("ready publication target is a symlink")
    os.replace(source, target)
    target.chmod(0o600)
    _fsync_directory(target.parent)


def _recover_ready_generation_locked(
    ready_db: Path, *, checkpoint: Checkpoint = None
) -> Mapping[str, Any]:
    paths = _paths(ready_db)
    if not os.path.lexists(paths["journal"]):
        if os.path.lexists(paths["transaction"]):
            _remove_transaction(paths)
        return {"status": "clean", "recovered": False}
    journal = _read_private_json(paths["journal"], label="ready publication journal")
    if (
        journal.get("schema_version") != READY_PUBLICATION_SCHEMA
        or journal.get("ready_db") != ready_db.name
        or journal.get("ready_manifest") != paths["manifest"].name
    ):
        raise RuntimeError("ready publication journal identity is invalid")
    old = journal.get("old")
    new = journal.get("new")
    if not isinstance(old, Mapping) or not isinstance(new, Mapping):
        raise RuntimeError("ready publication journal generations are invalid")
    old_present = old.get("present") is True
    old_db_sha = str(old.get("db_sha256") or "") or None
    old_manifest_sha = str(old.get("manifest_sha256") or "") or None
    new_db_sha = str(new.get("db_sha256") or "")
    new_manifest_sha = str(new.get("manifest_sha256") or "")
    if not new_db_sha or not new_manifest_sha or (
        old_present and (not old_db_sha or not old_manifest_sha)
    ):
        raise RuntimeError("ready publication journal digests are invalid")

    current_db_sha = _digest_or_none(ready_db, label="canonical ready database")
    current_manifest_sha = _digest_or_none(
        paths["manifest"], label="canonical ready manifest"
    )
    if current_db_sha not in {None, old_db_sha, new_db_sha}:
        raise RuntimeError("canonical ready database has an unknown generation")
    if current_manifest_sha not in {None, old_manifest_sha, new_manifest_sha}:
        raise RuntimeError("canonical ready manifest has an unknown generation")

    staged_new_db_sha = _digest_or_none(
        paths["new_db"], label="staged new ready database"
    )
    staged_new_manifest_sha = _digest_or_none(
        paths["new_manifest"], label="staged new ready manifest"
    )
    can_finish_forward = bool(
        (current_db_sha == new_db_sha or staged_new_db_sha == new_db_sha)
        and (
            current_manifest_sha == new_manifest_sha
            or staged_new_manifest_sha == new_manifest_sha
        )
    )
    if can_finish_forward:
        # Replace an explicitly staged database even when its bytes equal the
        # current generation.  Its manifest carries the staged file metadata;
        # keeping the old inode would make an evidence-only republish look
        # modified forever and defeat idempotent reuse.
        if current_db_sha != new_db_sha or staged_new_db_sha == new_db_sha:
            _replace_and_sync(paths["new_db"], ready_db)
            if checkpoint:
                checkpoint("db_replaced")
        if current_manifest_sha != new_manifest_sha:
            _replace_and_sync(paths["new_manifest"], paths["manifest"])
            if checkpoint:
                checkpoint("manifest_replaced")
        committed = "new"
    elif old_present:
        staged_old_db_sha = _digest_or_none(
            paths["old_db"], label="staged old ready database"
        )
        staged_old_manifest_sha = _digest_or_none(
            paths["old_manifest"], label="staged old ready manifest"
        )
        can_roll_back = bool(
            (current_db_sha == old_db_sha or staged_old_db_sha == old_db_sha)
            and (
                current_manifest_sha == old_manifest_sha
                or staged_old_manifest_sha == old_manifest_sha
            )
        )
        if not can_roll_back:
            raise RuntimeError("ready publication cannot finish or roll back safely")
        # As in the forward path, restore an explicitly staged database even
        # when the old and new generations have identical bytes.  The old
        # manifest records the old inode metadata; keeping the replacement
        # inode would make the rolled-back pair fail its own reuse check.
        if current_db_sha != old_db_sha or staged_old_db_sha == old_db_sha:
            _replace_and_sync(paths["old_db"], ready_db)
        if current_manifest_sha != old_manifest_sha:
            _replace_and_sync(paths["old_manifest"], paths["manifest"])
        committed = "old"
    else:
        raise RuntimeError("initial ready publication cannot be recovered safely")

    expected_db = new_db_sha if committed == "new" else old_db_sha
    expected_manifest = (
        new_manifest_sha if committed == "new" else old_manifest_sha
    )
    if (
        _sha256_regular(ready_db, label="recovered ready database") != expected_db
        or _sha256_regular(
            paths["manifest"], label="recovered ready manifest"
        )
        != expected_manifest
    ):
        raise RuntimeError("ready publication recovery verification failed")
    if checkpoint:
        checkpoint("verified")
    _remove_transaction(paths)
    return {"status": "recovered", "recovered": True, "generation": committed}


def recover_ready_generation(
    ready_db: Path, *, checkpoint: Checkpoint = None, lock_held: bool = False
) -> Mapping[str, Any]:
    if lock_held:
        return _recover_ready_generation_locked(ready_db, checkpoint=checkpoint)
    with ready_publication_lock(ready_db):
        return _recover_ready_generation_locked(ready_db, checkpoint=checkpoint)


def commit_ready_generation(
    ready_db: Path,
    staged_db: Path,
    manifest: Mapping[str, Any],
    *,
    checkpoint: Checkpoint = None,
) -> Mapping[str, Any]:
    with ready_publication_lock(ready_db):
        _recover_ready_generation_locked(ready_db)
        paths = _paths(ready_db)
        if os.path.lexists(paths["transaction"]):
            raise RuntimeError("ready publication transaction already exists")
        paths["transaction"].mkdir(mode=0o700)
        _ensure_private_directory(paths["transaction"])
        try:
            staged_db_sha = _sha256_regular(
                staged_db, label="staged ready database"
            )
            staged_size = staged_db.stat().st_size
            if (
                manifest.get("sha256") != staged_db_sha
                or int(manifest.get("size_bytes") or -1) != staged_size
                or manifest.get("ready_db") != str(ready_db)
            ):
                raise RuntimeError("staged ready manifest does not describe staged database")
            os.replace(staged_db, paths["new_db"])
            paths["new_db"].chmod(0o600)
            with paths["new_db"].open("rb") as handle:
                os.fsync(handle.fileno())
            manifest_bytes = (
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n"
            ).encode("utf-8")
            _write_private_bytes(paths["new_manifest"], manifest_bytes)

            canonical_db = _digest_or_none(
                ready_db, label="existing ready database"
            )
            canonical_manifest = _digest_or_none(
                paths["manifest"], label="existing ready manifest"
            )
            old: dict[str, Any] = {
                "present": False,
                "discarded_invalid": bool(
                    canonical_db is not None or canonical_manifest is not None
                ),
                "db_sha256": canonical_db,
                "manifest_sha256": canonical_manifest,
            }
            if canonical_db is not None and canonical_manifest is not None:
                try:
                    existing_manifest = _read_private_json(
                        paths["manifest"], label="existing ready manifest"
                    )
                except RuntimeError:
                    existing_manifest = {}
                existing_valid = bool(
                    existing_manifest.get("sha256") == canonical_db
                    and int(existing_manifest.get("size_bytes") or -1)
                    == ready_db.stat().st_size
                )
                if existing_valid:
                    _copy_private(ready_db, paths["old_db"])
                    _copy_private(paths["manifest"], paths["old_manifest"])
                    old.update(
                        {
                            "present": True,
                            "discarded_invalid": False,
                        }
                    )
            new_manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
            journal = {
                "schema_version": READY_PUBLICATION_SCHEMA,
                "ready_db": ready_db.name,
                "ready_manifest": paths["manifest"].name,
                "old": old,
                "new": {
                    "db_sha256": staged_db_sha,
                    "size_bytes": staged_size,
                    "manifest_sha256": new_manifest_sha,
                },
            }
            _fsync_directory(paths["transaction"])
            _write_private_json(paths["journal"], journal)
            if checkpoint:
                checkpoint("journal_written")
            result = _recover_ready_generation_locked(
                ready_db, checkpoint=checkpoint
            )
            if result.get("generation") != "new":
                raise RuntimeError(
                    "ready publication rolled back instead of committing new generation"
                )
            return result
        except Exception:
            # Once the journal exists, recovery evidence must remain untouched.
            if not os.path.lexists(paths["journal"]):
                _remove_transaction(paths)
            raise
