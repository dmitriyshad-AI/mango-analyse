from __future__ import annotations

import ctypes
import errno
import fcntl
import hashlib
import os
import stat
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Mapping, Optional

CLOUD_PATH_MARKERS = (
    "yandex.disk",
    "yandexdisk",
    "cloudstorage",
    "mobile documents",
    "icloud",
    "dropbox",
    "onedrive",
    "google drive",
    "googledrive",
)


def path_has_cloud_marker(path: Path) -> bool:
    lowered = str(path).casefold()
    return any(marker in lowered for marker in CLOUD_PATH_MARKERS)


def _file_identity(
    value: os.stat_result,
) -> tuple[int, int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _descriptor_has_extended_acl(descriptor: int, *, label: str) -> bool:
    """Fail closed when an owner-only descriptor grants access through an ACL."""
    if sys.platform == "darwin":
        try:
            libc = ctypes.CDLL(None, use_errno=True)
            acl_get_fd_np = libc.acl_get_fd_np
            acl_get_fd_np.argtypes = (ctypes.c_int, ctypes.c_int)
            acl_get_fd_np.restype = ctypes.c_void_p
            acl_free = libc.acl_free
            acl_free.argtypes = (ctypes.c_void_p,)
            acl_free.restype = ctypes.c_int
        except (AttributeError, OSError) as exc:
            raise RuntimeError(f"{label}_acl_check_failed") from exc
        ctypes.set_errno(0)
        acl = acl_get_fd_np(descriptor, 0x100)  # ACL_TYPE_EXTENDED
        if acl:
            acl_free(acl)
            return True
        if ctypes.get_errno() == errno.ENOENT:
            return False
        raise RuntimeError(f"{label}_acl_check_failed")
    if sys.platform.startswith("linux"):
        try:
            os.getxattr(descriptor, "system.posix_acl_access")
        except OSError as exc:
            no_acl_errors = {
                value
                for value in (
                    getattr(errno, "ENODATA", None),
                    getattr(errno, "ENOATTR", None),
                )
                if value is not None
            }
            if exc.errno in no_acl_errors:
                return False
            raise RuntimeError(f"{label}_acl_check_failed") from exc
        return True
    raise RuntimeError(f"{label}_acl_check_unsupported")


def _descriptor_resolved_path(descriptor: int, *, label: str) -> Path:
    """Return the path attached to an open descriptor and bind it to its inode."""
    try:
        if sys.platform == "darwin":
            raw = fcntl.fcntl(
                descriptor,
                fcntl.F_GETPATH,
                b"\0" * 1024,
            )
            value = os.fsdecode(raw.split(b"\0", 1)[0])
        elif sys.platform.startswith("linux"):
            value = os.readlink(f"/proc/self/fd/{descriptor}")
        else:
            raise RuntimeError(f"{label}_descriptor_path_unsupported")
        resolved = Path(value).resolve(strict=True)
        opened = os.fstat(descriptor)
        current = os.stat(resolved, follow_symlinks=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError(f"{label}_descriptor_path_unavailable") from exc
    if (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino):
        raise RuntimeError(f"{label}_descriptor_path_changed")
    return resolved


@contextmanager
def _stable_regular_descriptor(
    path: Path,
    *,
    label: str,
    owner_only_mode: Optional[int] = None,
) -> Iterable[int]:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RuntimeError(f"{label}_unsafe_or_missing") from exc
    try:
        opened = os.fstat(descriptor)
        current = os.lstat(path)
        if (
            not stat.S_ISREG(opened.st_mode)
            or not stat.S_ISREG(current.st_mode)
            or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)
        ):
            raise RuntimeError(f"{label}_must_be_regular_nofollow")
        if owner_only_mode is not None and (
            opened.st_uid != os.getuid()
            or stat.S_IMODE(opened.st_mode) != owner_only_mode
            or opened.st_nlink != 1
        ):
            raise RuntimeError(f"{label}_must_be_owner_only_{owner_only_mode:04o}")
        if owner_only_mode is not None and _descriptor_has_extended_acl(
            descriptor, label=label
        ):
            raise RuntimeError(f"{label}_must_not_have_extended_acl")
        yield descriptor
        after = os.fstat(descriptor)
        current_after = os.lstat(path)
        if (
            _file_identity(opened) != _file_identity(after)
            or (after.st_dev, after.st_ino)
            != (current_after.st_dev, current_after.st_ino)
            or not stat.S_ISREG(current_after.st_mode)
        ):
            raise RuntimeError(f"{label}_changed_while_reading")
        if owner_only_mode is not None and _descriptor_has_extended_acl(
            descriptor, label=label
        ):
            raise RuntimeError(f"{label}_must_not_have_extended_acl")
    finally:
        os.close(descriptor)


def read_stable_regular_bytes(
    path: Path,
    *,
    label: str,
    owner_only_mode: Optional[int] = None,
) -> bytes:
    chunks: list[bytes] = []
    with _stable_regular_descriptor(
        path, label=label, owner_only_mode=owner_only_mode
    ) as descriptor:
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
    return b"".join(chunks)


def read_stable_regular_bytes_with_path(
    path: Path,
    *,
    label: str,
    owner_only_mode: Optional[int] = None,
) -> tuple[bytes, Path]:
    """Read stable bytes and the canonical path of the exact opened inode."""
    chunks: list[bytes] = []
    with _stable_regular_descriptor(
        path, label=label, owner_only_mode=owner_only_mode
    ) as descriptor:
        resolved = _descriptor_resolved_path(descriptor, label=label)
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        resolved_after = _descriptor_resolved_path(descriptor, label=label)
        if resolved_after != resolved:
            raise RuntimeError(f"{label}_descriptor_path_changed_while_reading")
    return b"".join(chunks), resolved


def inspect_stable_regular_file(
    path: Path,
    *,
    label: str,
    require_owner: bool = False,
    require_single_link: bool = False,
    owner_only_mode: Optional[int] = None,
) -> Mapping[str, object]:
    """Bind a regular file to one inode and return content-free integrity data."""
    evidence: dict[str, object]
    with _stable_regular_descriptor(
        path,
        label=label,
        owner_only_mode=owner_only_mode,
    ) as descriptor:
        opened = os.fstat(descriptor)
        if require_owner and opened.st_uid != os.getuid():
            raise RuntimeError(f"{label}_must_be_owned_by_runtime_user")
        if require_single_link and opened.st_nlink != 1:
            raise RuntimeError(f"{label}_must_have_one_hardlink")
        if opened.st_size <= 0:
            raise RuntimeError(f"{label}_must_not_be_empty")
        resolved = _descriptor_resolved_path(descriptor, label=label)
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        evidence = {
            "resolved_path": resolved,
            "size_bytes": opened.st_size,
            "sha256": digest.hexdigest(),
        }
    return evidence


def copy_stable_regular_file_owner_only(
    source: Path,
    target: Path,
    *,
    label: str,
    expected_sha256: str,
    expected_size_bytes: int,
) -> Mapping[str, object]:
    """Stream one verified input inode into a new private regular file."""
    validate_owner_only_directory(
        target.parent,
        label=f"{label}_directory",
        owner_only_mode=0o700,
    )
    if os.path.lexists(target):
        raise RuntimeError(f"{label}_target_already_exists")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    size_bytes = 0
    try:
        os.fchmod(descriptor, 0o600)
        if _descriptor_has_extended_acl(descriptor, label=label):
            raise RuntimeError(f"{label}_temporary_has_extended_acl")
        with _stable_regular_descriptor(source, label=f"{label}_source") as source_fd:
            with os.fdopen(descriptor, "wb") as target_handle:
                descriptor = -1
                while chunk := os.read(source_fd, 1024 * 1024):
                    digest.update(chunk)
                    size_bytes += len(chunk)
                    target_handle.write(chunk)
                target_handle.flush()
                os.fsync(target_handle.fileno())
        if (
            size_bytes != expected_size_bytes
            or digest.hexdigest() != expected_sha256
        ):
            raise RuntimeError(f"{label}_source_binding_mismatch")
        if os.path.lexists(target):
            raise RuntimeError(f"{label}_target_already_exists")
        os.replace(temporary, target)
        directory_fd = os.open(
            target.parent,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        evidence = inspect_stable_regular_file(
            target,
            label=label,
            require_owner=True,
            require_single_link=True,
            owner_only_mode=0o600,
        )
        if (
            evidence.get("sha256") != expected_sha256
            or evidence.get("size_bytes") != expected_size_bytes
        ):
            raise RuntimeError(f"{label}_verification_failed")
        return evidence
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
        temporary.unlink(missing_ok=True)


def validate_owner_only_directory(
    path: Path,
    *,
    label: str,
    owner_only_mode: int = 0o700,
) -> Path:
    """Validate one directory by descriptor, including its extended ACL."""
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RuntimeError(f"{label}_unsafe_or_missing") from exc
    try:
        opened = os.fstat(descriptor)
        current = os.lstat(path)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or not stat.S_ISDIR(current.st_mode)
            or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)
            or opened.st_uid != os.getuid()
            or stat.S_IMODE(opened.st_mode) != owner_only_mode
        ):
            raise RuntimeError(
                f"{label}_must_be_owner_only_{owner_only_mode:04o}_directory"
            )
        if _descriptor_has_extended_acl(descriptor, label=label):
            raise RuntimeError(f"{label}_must_not_have_extended_acl")
        resolved = _descriptor_resolved_path(descriptor, label=label)
        after = os.fstat(descriptor)
        current_after = os.lstat(path)
        if (
            _file_identity(opened) != _file_identity(after)
            or (after.st_dev, after.st_ino)
            != (current_after.st_dev, current_after.st_ino)
            or not stat.S_ISDIR(current_after.st_mode)
        ):
            raise RuntimeError(f"{label}_changed_while_validating")
        if _descriptor_has_extended_acl(descriptor, label=label):
            raise RuntimeError(f"{label}_must_not_have_extended_acl")
        return resolved
    finally:
        os.close(descriptor)


def atomic_replace_owner_only_bytes(
    path: Path,
    payload: bytes,
    *,
    label: str,
) -> None:
    """Atomically replace a private file without inheriting an unsafe old ACL."""
    if os.path.lexists(path) and path.is_symlink():
        raise RuntimeError(f"{label}_target_is_symlink")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        if _descriptor_has_extended_acl(descriptor, label=label):
            raise RuntimeError(f"{label}_temporary_has_extended_acl")
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if os.path.lexists(path) and path.is_symlink():
            raise RuntimeError(f"{label}_target_is_symlink")
        os.replace(temporary, path)
        directory = os.open(
            path.parent,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        if read_stable_regular_bytes(
            path, label=label, owner_only_mode=0o600
        ) != payload:
            raise RuntimeError(f"{label}_verification_failed")
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)


def stable_regular_file_evidence(
    path: Path, *, label: str = "sha256_source"
) -> Mapping[str, object]:
    digest = hashlib.sha256()
    size_bytes = 0
    with _stable_regular_descriptor(path, label=label) as descriptor:
        opened = os.fstat(descriptor)
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
            size_bytes += len(chunk)
    return {
        "sha256": digest.hexdigest(),
        "size_bytes": size_bytes,
        "device": opened.st_dev,
        "inode": opened.st_ino,
        "mtime_ns": opened.st_mtime_ns,
    }
