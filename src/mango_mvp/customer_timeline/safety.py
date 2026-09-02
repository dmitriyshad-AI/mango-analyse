from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import fcntl
import os
from pathlib import Path
import struct
import sys
from typing import Any, Iterator, Mapping, Optional


CUSTOMER_TIMELINE_SAFETY_SCHEMA_VERSION = "customer_timeline_safety_v1"
_MANAGED_STAGING_WRITE_ALLOWED_DBS: ContextVar[tuple[str, ...]] = ContextVar(
    "customer_timeline_managed_staging_write_allowed_dbs",
    default=(),
)
_MANAGED_STAGING_CHILD_PASS_FDS: ContextVar[tuple[int, ...]] = ContextVar(
    "customer_timeline_managed_staging_child_pass_fds",
    default=(),
)
_NIGHTLY_LOCK_SUFFIX = ".nightly_service.lock"
_MANAGED_WRITER_ENV_DB = "MANGO_CUSTOMER_TIMELINE_MANAGED_WRITER_DB"
_MANAGED_WRITER_ENV_LOCK = "MANGO_CUSTOMER_TIMELINE_MANAGED_WRITER_LOCK"
_MANAGED_WRITER_ENV_PARENT_PID = "MANGO_CUSTOMER_TIMELINE_MANAGED_WRITER_PARENT_PID"
_MANAGED_WRITER_ENV_PROOF_LOCK = "MANGO_CUSTOMER_TIMELINE_MANAGED_WRITER_PROOF_LOCK"
_MANAGED_WRITER_ENV_PROOF_FD = "MANGO_CUSTOMER_TIMELINE_MANAGED_WRITER_PROOF_FD"


def customer_timeline_safety_contract() -> Mapping[str, Any]:
    return {
        "schema_version": CUSTOMER_TIMELINE_SAFETY_SCHEMA_VERSION,
        "read_only_source_systems": True,
        "write_crm": False,
        "write_tallanto": False,
        "send_email": False,
        "send_messenger": False,
        "live_send": False,
        "run_asr": False,
        "run_ra": False,
        "write_runtime_db": False,
        "runtime_db_writes": False,
        "mutate_stable_runtime": False,
        "stable_runtime_writes": False,
        "delete_source_artifacts": False,
        "store_raw_files_in_sqlite": False,
        "identity_conflicts_auto_merge": False,
        "old_to_new_customer_id_mapping_required": True,
        "brand_blocks_identity_merge": False,
    }


def blocked_live_actions() -> tuple[str, ...]:
    return (
        "write_crm",
        "write_tallanto",
        "send_email",
        "send_messenger",
        "live_send",
        "run_asr",
        "run_ra",
        "write_runtime_db",
        "runtime_db_writes",
        "mutate_stable_runtime",
        "stable_runtime_writes",
        "delete_source_artifacts",
    )


def assert_customer_timeline_safety_contract(contract: Mapping[str, Any]) -> None:
    for action in blocked_live_actions():
        if contract.get(action) is not False:
            raise ValueError(f"customer timeline safety requires {action}=False")
    if contract.get("read_only_source_systems") is not True:
        raise ValueError("customer timeline safety requires read_only_source_systems=True")
    if contract.get("store_raw_files_in_sqlite") is not False:
        raise ValueError("customer timeline safety requires store_raw_files_in_sqlite=False")
    if contract.get("identity_conflicts_auto_merge") is not False:
        raise ValueError("customer timeline safety requires identity_conflicts_auto_merge=False")
    if contract.get("old_to_new_customer_id_mapping_required") is not True:
        raise ValueError("customer timeline safety requires old_to_new_customer_id_mapping_required=True")
    if contract.get("brand_blocks_identity_merge") is not False:
        raise ValueError("customer timeline safety requires brand_blocks_identity_merge=False")


def is_stable_runtime_path(path: Path | str) -> bool:
    return any(part.casefold() == "stable_runtime" for part in Path(path).parts)


def is_customer_timeline_prod_path(path: Path | str) -> bool:
    return any("customer_timeline_prod_" in part.casefold() for part in Path(path).parts)


def is_canonical_customer_timeline_staging_path(
    path: Path | str,
    *,
    allowed_root: Path | str,
) -> bool:
    """Identify the single-writer staging DB, including hard-link aliases."""

    resolved = Path(path).expanduser().resolve(strict=False)
    canonical = Path(allowed_root).expanduser().resolve(strict=False) / "customer_timeline_staging.sqlite"
    if resolved == canonical:
        return True
    try:
        return resolved.is_file() and canonical.is_file() and resolved.samefile(canonical)
    except OSError:
        return False


@contextmanager
def managed_staging_writer_scope(db_path: Optional[Path | str] = None) -> Iterator[None]:
    """Authorize the verified unified nightly writer in this execution context."""

    resolved_db = _resolve_managed_scope_db(db_path)
    lock_path = _managed_writer_lock_path(resolved_db)
    proof_lock_path = Path(str(lock_path) + ".owner")
    proof_handle = _held_managed_writer_proof_handle(lock_path)
    scope_marker = _MANAGED_STAGING_WRITE_ALLOWED_DBS.set((str(resolved_db),))
    child_fds_marker = _MANAGED_STAGING_CHILD_PASS_FDS.set((proof_handle.fileno(),))
    previous_env = {
        _MANAGED_WRITER_ENV_DB: os.environ.get(_MANAGED_WRITER_ENV_DB),
        _MANAGED_WRITER_ENV_LOCK: os.environ.get(_MANAGED_WRITER_ENV_LOCK),
        _MANAGED_WRITER_ENV_PARENT_PID: os.environ.get(_MANAGED_WRITER_ENV_PARENT_PID),
        _MANAGED_WRITER_ENV_PROOF_LOCK: os.environ.get(_MANAGED_WRITER_ENV_PROOF_LOCK),
        _MANAGED_WRITER_ENV_PROOF_FD: os.environ.get(_MANAGED_WRITER_ENV_PROOF_FD),
    }
    os.environ[_MANAGED_WRITER_ENV_DB] = str(resolved_db)
    os.environ[_MANAGED_WRITER_ENV_LOCK] = str(lock_path)
    os.environ[_MANAGED_WRITER_ENV_PARENT_PID] = str(os.getpid())
    os.environ[_MANAGED_WRITER_ENV_PROOF_LOCK] = str(proof_lock_path)
    os.environ[_MANAGED_WRITER_ENV_PROOF_FD] = str(proof_handle.fileno())
    try:
        yield
    finally:
        _MANAGED_STAGING_WRITE_ALLOWED_DBS.reset(scope_marker)
        _MANAGED_STAGING_CHILD_PASS_FDS.reset(child_fds_marker)
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def guard_managed_customer_timeline_staging_write(path: Path | str) -> Path:
    resolved = Path(path).expanduser().resolve(strict=False)
    canonical = resolved.parent / "customer_timeline_staging.sqlite"
    ownership_receipt = canonical.parent / "state" / "WRITER_OWNERSHIP.json"
    if (
        is_canonical_customer_timeline_staging_path(resolved, allowed_root=resolved.parent)
        and ownership_receipt.is_file()
        and not _managed_writer_scope_allows(resolved)
    ):
        raise ValueError("managed customer timeline staging writes require the unified nightly service")
    return resolved


def _resolve_managed_scope_db(db_path: Optional[Path | str]) -> Path:
    held_db_paths = _held_managed_scope_db_paths()
    if db_path is not None:
        resolved = Path(db_path).expanduser().resolve(strict=False)
        if str(resolved) not in {str(item) for item in held_db_paths}:
            raise ValueError("managed customer timeline writer scope requires a held nightly service lock")
        return resolved
    if len(held_db_paths) != 1:
        raise ValueError("managed customer timeline writer scope requires exactly one held nightly service lock")
    return held_db_paths[0]


def _held_managed_scope_db_paths() -> tuple[Path, ...]:
    try:
        from mango_mvp.customer_timeline.store import _held_customer_timeline_run_locks
    except Exception:
        return ()
    paths: list[Path] = []
    for raw_lock_path in _held_customer_timeline_run_locks():
        db_path = _db_path_from_lock_path(Path(raw_lock_path).expanduser().resolve(strict=False))
        if db_path is not None:
            paths.append(db_path)
    return tuple(paths)


def _managed_writer_lock_path(db_path: Path | str) -> Path:
    resolved = Path(db_path).expanduser().resolve(strict=False)
    return Path(str(resolved) + _NIGHTLY_LOCK_SUFFIX)


def _db_path_from_lock_path(lock_path: Path) -> Optional[Path]:
    text = str(lock_path)
    if not text.endswith(_NIGHTLY_LOCK_SUFFIX):
        return None
    return Path(text[: -len(_NIGHTLY_LOCK_SUFFIX)])


def _managed_writer_scope_allows(db_path: Path) -> bool:
    resolved = Path(db_path).expanduser().resolve(strict=False)
    if str(resolved) in _MANAGED_STAGING_WRITE_ALLOWED_DBS.get():
        return True
    env_db = os.environ.get(_MANAGED_WRITER_ENV_DB)
    env_lock = os.environ.get(_MANAGED_WRITER_ENV_LOCK)
    env_parent_pid = os.environ.get(_MANAGED_WRITER_ENV_PARENT_PID)
    env_proof_lock = os.environ.get(_MANAGED_WRITER_ENV_PROOF_LOCK)
    env_proof_fd = os.environ.get(_MANAGED_WRITER_ENV_PROOF_FD)
    if not all((env_db, env_lock, env_parent_pid, env_proof_lock, env_proof_fd)):
        return False
    if Path(env_db).expanduser().resolve(strict=False) != resolved:
        return False
    lock_path = Path(env_lock).expanduser().resolve(strict=False)
    if lock_path != _managed_writer_lock_path(resolved):
        return False
    if str(os.getppid()) != env_parent_pid or not _pid_is_alive(env_parent_pid):
        return False
    proof_lock_path = Path(env_proof_lock).expanduser().resolve(strict=False)
    if proof_lock_path != Path(str(lock_path) + ".owner"):
        return False
    return _inherited_run_lock_allows(
        lock_path,
        proof_lock_path,
        env_proof_fd,
        int(env_parent_pid),
    )


def managed_staging_writer_subprocess_pass_fds() -> tuple[int, ...]:
    """Return the held run-lock descriptor that authorizes one direct child."""

    return _MANAGED_STAGING_CHILD_PASS_FDS.get()


def _held_managed_writer_proof_handle(lock_path: Path) -> Any:
    from mango_mvp.customer_timeline.store import _held_customer_timeline_run_lock_proofs

    handle = _held_customer_timeline_run_lock_proofs().get(str(lock_path))
    if handle is None:
        raise ValueError("managed customer timeline writer scope requires its held lock handle")
    return handle


def _inherited_run_lock_allows(
    lock_path: Path,
    proof_lock_path: Path,
    raw_proof_fd: str,
    expected_parent_pid: int,
) -> bool:
    try:
        proof_fd = int(raw_proof_fd)
        inherited_stat = os.fstat(proof_fd)
        path_stat = proof_lock_path.stat()
    except (OSError, TypeError, ValueError):
        return False
    if (
        proof_fd <= 2
        or (inherited_stat.st_dev, inherited_stat.st_ino) != (path_stat.st_dev, path_stat.st_ino)
    ):
        return False
    return _run_lock_is_held(lock_path) and _record_lock_owner(proof_fd) == expected_parent_pid


def _run_lock_is_held(lock_path: Path) -> bool:
    try:
        handle = lock_path.open("r")
    except OSError:
        return False
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError):
            return True
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return False
    finally:
        handle.close()


def _record_lock_owner(lock_fd: int) -> Optional[int]:
    """Return the PID blocking a one-byte write lock on Darwin/Linux."""

    if sys.platform == "darwin":
        layout = "@qqihh"
        request = struct.pack(layout, 0, 1, 0, fcntl.F_WRLCK, os.SEEK_SET)
        result = struct.unpack(layout, fcntl.fcntl(lock_fd, fcntl.F_GETLK, request))
        lock_pid, lock_type = int(result[2]), int(result[3])
    else:
        layout = "@hhqqi4x"
        request = struct.pack(layout, fcntl.F_WRLCK, os.SEEK_SET, 0, 1, 0)
        result = struct.unpack(layout, fcntl.fcntl(lock_fd, fcntl.F_GETLK, request))
        lock_type, lock_pid = int(result[0]), int(result[4])
    return None if lock_type == fcntl.F_UNLCK else lock_pid


def _pid_is_alive(raw_pid: str) -> bool:
    try:
        pid = int(raw_pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def guard_customer_timeline_writable_path(path: Path | str) -> Path:
    resolved = Path(path).expanduser().resolve(strict=False)
    if is_customer_timeline_prod_path(resolved):
        raise ValueError(f"customer timeline prod is snapshot-only; write staging and publish atomically: {resolved}")
    if resolved.is_file() and resolved.stat().st_nlink > 1:
        raise ValueError(f"customer timeline writable path must not be a hard link: {resolved}")
    return resolved


def guard_customer_timeline_output_path(path: Path | str, allowed_root: Path | str) -> Path:
    resolved = Path(path).resolve(strict=False)
    root = Path(allowed_root).resolve(strict=False)
    if is_stable_runtime_path(resolved):
        raise ValueError(f"customer timeline output must not be under stable_runtime: {resolved}")
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"customer timeline output must stay under allowed root: {root}") from exc
    return resolved
