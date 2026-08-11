#!/usr/bin/env python3
"""Relocate persisted Mango Calls paths inside an offline pipeline copy.

The script deliberately knows only the path fields read by the Calls runtime.
It is not a recursive JSON/SQLite string replacement utility.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import secrets
import sqlite3
import stat
import struct
import subprocess
import sys
import tempfile
from contextlib import ExitStack, closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterator, Mapping, MutableMapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mango_mvp.productization.capture_staging import (  # noqa: E402
    entry_from_json,
    load_capture_recovery,
    recoverable_json_tail,
    recoverable_utf8_tail,
)
from mango_mvp.productization.mango_calls_service_contract import (  # noqa: E402
    ready_row_is_complete,
)


INVENTORY_SCHEMA = "mango_calls_pipeline_inventory_v1"
RELOCATION_SCHEMA = "mango_calls_pipeline_relocation_v1"
CONFIRM_ENV = "CONFIRM_MANGO_CALLS_RELOCATION"
CONFIRM_VALUE = "RELOCATE_MANGO_CALLS_PIPELINE"
CAPTURE_REL = "capture/capture_manifest.jsonl"
WORKING_DB_REL = "working/mango_calls_pipeline.sqlite"
READY_DB_REL = "drop/mango_calls_ready.sqlite"
READY_MANIFEST_REL = "drop/mango_calls_ready.manifest.json"
CURSOR_REL = "state/mango_api_freshness.json"
ARTIFACT_ORDER = (CAPTURE_REL, WORKING_DB_REL, READY_DB_REL, READY_MANIFEST_REL)
REQUIRED_TRANSFER_FILES = (*ARTIFACT_ORDER, CURSOR_REL)
CAPTURE_PATH_FIELDS = ("local_audio_path", "canonical_audio_path")
SQLITE_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")
FORBIDDEN_PATH_MARKERS = (
    "stable_runtime",
    "yandex.disk",
    "yandexdisk",
    "cloudstorage",
    "mobile documents",
    "dropbox",
    "onedrive",
    "google drive",
)


class RelocationError(RuntimeError):
    """Fail-closed operator error."""


def canonical_json(payload: Mapping[str, Any], *, indent: int | None = 2) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=indent,
            separators=None if indent is not None else (",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def path_exists(path: Path) -> bool:
    return os.path.lexists(path)


def absolute_path(value: Path, *, label: str) -> Path:
    expanded = value.expanduser()
    if not expanded.is_absolute() or ".." in value.parts:
        raise RelocationError(f"{label} must be an explicit absolute path without '..'")
    normalized = Path(os.path.normpath(str(expanded)))
    if str(normalized) == "/":
        raise RelocationError(f"{label} must not be filesystem root")
    return normalized


def path_has_forbidden_marker(path: Path) -> bool:
    folded = "/".join(path.parts).casefold()
    return any(marker in folded for marker in FORBIDDEN_PATH_MARKERS)


def assert_no_symlink_components(path: Path, *, allow_missing: bool) -> None:
    current = Path(path.anchor)
    missing_seen = False
    for part in path.parts[1:]:
        current /= part
        if missing_seen:
            continue
        try:
            current_stat = os.lstat(current)
        except FileNotFoundError:
            if not allow_missing:
                raise RelocationError(f"path component is missing: {current}") from None
            missing_seen = True
            continue
        if stat.S_ISLNK(current_stat.st_mode):
            raise RelocationError(f"path contains a symlink component: {current}")


def assert_owner(path_stat: os.stat_result, *, label: str) -> None:
    if path_stat.st_uid != os.getuid():
        raise RelocationError(f"{label} is not owned by the current user")


def assert_not_in_git(path: Path) -> None:
    candidate = path if path_exists(path) and path.is_dir() else path.parent
    while True:
        if path_exists(candidate / ".git"):
            raise RelocationError("runtime path must stay outside Git")
        if candidate == candidate.parent:
            return
        candidate = candidate.parent


def relative_to_root(path: Path, root: Path, *, label: str) -> Path:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise RelocationError(f"{label} is outside its allowed root") from exc
    if relative == Path(".") or not relative.parts or ".." in relative.parts:
        raise RelocationError(f"{label} does not name a file below its root")
    return relative


def roots_overlap(first: Path, second: Path) -> bool:
    try:
        first.relative_to(second)
    except ValueError:
        pass
    else:
        return True
    try:
        second.relative_to(first)
    except ValueError:
        return False
    return True


def validate_relocation_roots(
    pipeline_root: Path,
    old_root: Path,
    new_root: Path,
    *,
    execute: bool,
) -> tuple[Path, Path, Path, Path]:
    pipeline = absolute_path(pipeline_root, label="pipeline_root")
    old = absolute_path(old_root, label="old_root")
    new = absolute_path(new_root, label="new_root")
    if roots_overlap(old, new):
        raise RelocationError("old_root and new_root must be distinct and non-overlapping")

    home_value = os.environ.get("HOME")
    if not home_value:
        raise RelocationError("HOME is required")
    home = absolute_path(Path(home_value), label="HOME")
    local_root = home / ".mango_local"
    assert_no_symlink_components(home, allow_missing=False)
    assert_no_symlink_components(local_root, allow_missing=False)
    local_stat = os.lstat(local_root)
    if not stat.S_ISDIR(local_stat.st_mode):
        raise RelocationError("$HOME/.mango_local must be a real directory")
    assert_owner(local_stat, label="$HOME/.mango_local")
    if stat.S_IMODE(local_stat.st_mode) != 0o700:
        raise RelocationError("$HOME/.mango_local must have owner-only permissions")

    relative_to_root(new, local_root, label="new_root")
    if path_has_forbidden_marker(new):
        raise RelocationError("new_root points at a forbidden runtime location")
    assert_no_symlink_components(new, allow_missing=True)
    assert_not_in_git(new)

    try:
        pipeline.relative_to(local_root)
    except ValueError:
        if execute or pipeline != old:
            raise RelocationError("pipeline_root is outside its allowed root")
        assert_no_symlink_components(pipeline, allow_missing=False)
    else:
        if path_has_forbidden_marker(pipeline):
            raise RelocationError("pipeline_root points at a forbidden runtime location")
        assert_no_symlink_components(pipeline, allow_missing=False)
        assert_not_in_git(pipeline)

    pipeline_stat = os.lstat(pipeline)
    if not stat.S_ISDIR(pipeline_stat.st_mode):
        raise RelocationError("pipeline_root must be a real directory")
    assert_owner(pipeline_stat, label="pipeline_root")
    if execute and (not new.parent.is_dir() or new.parent.is_symlink()):
        raise RelocationError("new_root parent must be an existing real directory")
    if execute and pipeline_stat.st_dev != os.stat(new.parent, follow_symlinks=False).st_dev:
        raise RelocationError("pipeline_root and new_root must support an atomic same-filesystem rename")
    if execute and path_exists(new) and new != pipeline:
        new_stat = os.lstat(new)
        if not stat.S_ISDIR(new_stat.st_mode):
            raise RelocationError("existing new_root must be a real directory")
        if any(new.iterdir()):
            raise RelocationError("existing new_root must be empty")
    return pipeline, old, new, local_root


def private_inventory_document_path(
    value: Path,
    *,
    label: str,
    allow_missing: bool,
) -> Path:
    path = absolute_path(value, label=label)
    home_value = os.environ.get("HOME")
    if not home_value:
        raise RelocationError("HOME is required")
    home = absolute_path(Path(home_value), label="HOME")
    local_root = home / ".mango_local"
    assert_no_symlink_components(home, allow_missing=False)
    assert_no_symlink_components(local_root, allow_missing=False)
    local_stat = os.lstat(local_root)
    if not stat.S_ISDIR(local_stat.st_mode):
        raise RelocationError("$HOME/.mango_local must be a real directory")
    assert_owner(local_stat, label="$HOME/.mango_local")
    if stat.S_IMODE(local_stat.st_mode) != 0o700:
        raise RelocationError("$HOME/.mango_local must have owner-only permissions")
    relative_to_root(path, local_root, label=label)
    if path_has_forbidden_marker(path):
        raise RelocationError(f"{label} points at a forbidden metadata location")
    assert_no_symlink_components(path, allow_missing=allow_missing)
    assert_not_in_git(path)
    if path_exists(path):
        path_stat = os.lstat(path)
        if not stat.S_ISREG(path_stat.st_mode) or path_stat.st_nlink != 1:
            raise RelocationError(f"{label} must be a single-link regular file")
        assert_owner(path_stat, label=label)
    return path


def _mode(value: os.stat_result) -> int:
    return stat.S_IMODE(value.st_mode)


def _entry_identity(value: os.stat_result) -> tuple[int, int, int]:
    return value.st_dev, value.st_ino, stat.S_IFMT(value.st_mode)


def _hash_open_file(parent_fd: int, name: str, expected: os.stat_result) -> str:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(name, flags, dir_fd=parent_fd)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or _entry_identity(opened) != _entry_identity(expected):
            raise RelocationError("tree entry changed while opening")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
        current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            _entry_identity(after) != _entry_identity(expected)
            or _entry_identity(current) != _entry_identity(expected)
            or after.st_size != expected.st_size
            or after.st_mtime_ns != expected.st_mtime_ns
        ):
            raise RelocationError("tree entry changed while hashing")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def build_inventory(root: Path) -> Mapping[str, Any]:
    root = absolute_path(root, label="inventory_root")
    assert_no_symlink_components(root, allow_missing=False)
    root_stat = os.lstat(root)
    if not stat.S_ISDIR(root_stat.st_mode):
        raise RelocationError("inventory_root must be a real directory")

    directories: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    symlinks: list[dict[str, str]] = []
    special: list[dict[str, str]] = []
    inode_paths: dict[tuple[int, int], list[str]] = {}
    inode_links: dict[tuple[int, int], int] = {}

    def walk(directory_fd: int, relative: PurePosixPath, directory_stat: os.stat_result) -> None:
        directories.append(
            {
                "relative_path": "." if relative == PurePosixPath(".") else relative.as_posix(),
                "mode": _mode(directory_stat),
                "mtime_ns": directory_stat.st_mtime_ns,
            }
        )
        with os.scandir(directory_fd) as iterator:
            entries = sorted(iterator, key=lambda item: item.name)
        for entry in entries:
            entry_relative = PurePosixPath(entry.name) if relative == PurePosixPath(".") else relative / entry.name
            entry_stat = entry.stat(follow_symlinks=False)
            if stat.S_ISLNK(entry_stat.st_mode):
                symlinks.append(
                    {
                        "relative_path": entry_relative.as_posix(),
                        "target": os.readlink(entry.name, dir_fd=directory_fd),
                    }
                )
                continue
            if stat.S_ISDIR(entry_stat.st_mode):
                flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
                child_fd = os.open(entry.name, flags, dir_fd=directory_fd)
                try:
                    opened = os.fstat(child_fd)
                    if _entry_identity(opened) != _entry_identity(entry_stat):
                        raise RelocationError("directory changed during inventory")
                    walk(child_fd, entry_relative, opened)
                    current = os.stat(entry.name, dir_fd=directory_fd, follow_symlinks=False)
                    if _entry_identity(current) != _entry_identity(entry_stat):
                        raise RelocationError("directory changed during inventory")
                finally:
                    os.close(child_fd)
                continue
            if stat.S_ISREG(entry_stat.st_mode):
                digest = _hash_open_file(directory_fd, entry.name, entry_stat)
                relative_text = entry_relative.as_posix()
                files.append(
                    {
                        "relative_path": relative_text,
                        "size_bytes": entry_stat.st_size,
                        "sha256": digest,
                        "mode": _mode(entry_stat),
                        "mtime_ns": entry_stat.st_mtime_ns,
                    }
                )
                inode_key = (entry_stat.st_dev, entry_stat.st_ino)
                inode_paths.setdefault(inode_key, []).append(relative_text)
                inode_links[inode_key] = entry_stat.st_nlink
                continue
            special.append(
                {
                    "relative_path": entry_relative.as_posix(),
                    "type": oct(stat.S_IFMT(entry_stat.st_mode)),
                }
            )
        after_directory = os.fstat(directory_fd)
        if (
            _entry_identity(after_directory) != _entry_identity(directory_stat)
            or _mode(after_directory) != _mode(directory_stat)
            or after_directory.st_mtime_ns != directory_stat.st_mtime_ns
            or after_directory.st_ctime_ns != directory_stat.st_ctime_ns
        ):
            raise RelocationError("directory changed during inventory")

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    root_fd = os.open(root, flags)
    try:
        opened_root = os.fstat(root_fd)
        if _entry_identity(opened_root) != _entry_identity(root_stat):
            raise RelocationError("inventory_root changed while opening")
        walk(root_fd, PurePosixPath("."), opened_root)
        current_root = os.lstat(root)
        if (
            _entry_identity(current_root) != _entry_identity(root_stat)
            or _mode(current_root) != _mode(root_stat)
            or current_root.st_mtime_ns != root_stat.st_mtime_ns
            or current_root.st_ctime_ns != root_stat.st_ctime_ns
        ):
            raise RelocationError("inventory_root changed during inventory")
    finally:
        os.close(root_fd)

    if symlinks or special:
        details = {"symlinks": symlinks, "special": special}
        raise RelocationError(f"inventory tree contains unsupported entries: {json.dumps(details, ensure_ascii=False, sort_keys=True)}")
    external_hardlinks = [
        paths
        for inode, paths in inode_paths.items()
        if inode_links[inode] > len(paths)
    ]
    if external_hardlinks:
        raise RelocationError("inventory tree contains hard links with aliases outside the tree")

    directories.sort(key=lambda item: item["relative_path"])
    files.sort(key=lambda item: item["relative_path"])
    return {
        "schema_version": INVENTORY_SCHEMA,
        "source_root": str(root),
        "directories": directories,
        "files": files,
        "symlinks": [],
        "special": [],
        "totals": {
            "directories": len(directories),
            "files": len(files),
            "size_bytes": sum(int(item["size_bytes"]) for item in files),
        },
    }


def _safe_relative(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise RelocationError(f"{label} is invalid")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or value in {".", ".."}:
        raise RelocationError(f"{label} escapes the inventory root")
    normalized = pure.as_posix()
    if normalized != value:
        raise RelocationError(f"{label} is not normalized")
    return normalized


def validate_inventory(payload: Any) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping) or payload.get("schema_version") != INVENTORY_SCHEMA:
        raise RelocationError("inventory schema is invalid")
    source_root = absolute_path(Path(str(payload.get("source_root") or "")), label="inventory source_root")
    directories = payload.get("directories")
    files = payload.get("files")
    if not isinstance(directories, list) or not isinstance(files, list):
        raise RelocationError("inventory entries are invalid")
    seen: set[str] = set()
    for item in directories:
        if not isinstance(item, Mapping):
            raise RelocationError("inventory directory entry is invalid")
        relative = str(item.get("relative_path"))
        if relative != ".":
            relative = _safe_relative(relative, label="inventory directory path")
        if relative in seen:
            raise RelocationError("inventory contains duplicate paths")
        seen.add(relative)
        if not isinstance(item.get("mode"), int) or not isinstance(item.get("mtime_ns"), int):
            raise RelocationError("inventory directory metadata is invalid")
    for item in files:
        if not isinstance(item, Mapping):
            raise RelocationError("inventory file entry is invalid")
        relative = _safe_relative(item.get("relative_path"), label="inventory file path")
        if relative in seen:
            raise RelocationError("inventory contains duplicate paths")
        seen.add(relative)
        if (
            not isinstance(item.get("size_bytes"), int)
            or not isinstance(item.get("mode"), int)
            or not isinstance(item.get("mtime_ns"), int)
            or not isinstance(item.get("sha256"), str)
            or len(str(item["sha256"])) != 64
            or any(character not in "0123456789abcdef" for character in str(item["sha256"]))
        ):
            raise RelocationError("inventory file metadata is invalid")
    if payload.get("symlinks") != [] or payload.get("special") != []:
        raise RelocationError("inventory records unsupported entries")
    totals = payload.get("totals")
    if not isinstance(totals, Mapping) or totals != {
        "directories": len(directories),
        "files": len(files),
        "size_bytes": sum(int(item["size_bytes"]) for item in files),
    }:
        raise RelocationError("inventory totals are invalid")
    selection = payload.get("selection")
    if selection is not None:
        if (
            not isinstance(selection, Mapping)
            or selection.get("schema_version")
            != "mango_calls_selective_transfer_v1"
            or selection.get("policy")
            != "all_non_audio_plus_unfinished_and_all_multi_audio"
        ):
            raise RelocationError("selective inventory contract is invalid")
        selected_paths = [str(item["relative_path"]) for item in files]
        expected_files_from = _nul_files_from(selected_paths)
        if selection.get("files_from_sha256") != sha256_bytes(expected_files_from):
            raise RelocationError("selective files-from digest is invalid")
        omitted = selection.get("omitted_audio")
        if not isinstance(omitted, list):
            raise RelocationError("selective omitted-audio evidence is invalid")
        omitted_seen: set[str] = set()
        for item in omitted:
            if not isinstance(item, Mapping):
                raise RelocationError("selective omitted-audio entry is invalid")
            relative = _safe_relative(
                item.get("relative_path"), label="omitted audio path"
            )
            if relative in seen or relative in omitted_seen:
                raise RelocationError("selective audio sets overlap or contain duplicates")
            omitted_seen.add(relative)
            if (
                not isinstance(item.get("size_bytes"), int)
                or int(item["size_bytes"]) <= 0
                or not isinstance(item.get("sha256"), str)
                or len(str(item["sha256"])) != 64
            ):
                raise RelocationError("selective omitted-audio metadata is invalid")
        if selection.get("omitted_audio_sha256") != sha256_bytes(
            canonical_json({"files": omitted}, indent=None)
        ):
            raise RelocationError("selective omitted-audio digest is invalid")
        for key in (
            "full_inventory_sha256",
            "completed_call_ids_sha256",
        ):
            value = selection.get(key)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise RelocationError(f"selective {key} is invalid")
        if (
            not isinstance(selection.get("completed_call_ids_count"), int)
            or int(selection["completed_call_ids_count"]) < 0
        ):
            raise RelocationError("selective completed-call count is invalid")
    result = dict(payload)
    result["source_root"] = str(source_root)
    return result


def read_private_json(path: Path, *, label: str, require_private: bool = True) -> Mapping[str, Any]:
    assert_no_symlink_components(path, allow_missing=False)
    path_stat = os.lstat(path)
    if not stat.S_ISREG(path_stat.st_mode):
        raise RelocationError(f"{label} must be a regular file")
    assert_owner(path_stat, label=label)
    if require_private and _mode(path_stat) & 0o077:
        raise RelocationError(f"{label} must have owner-only permissions")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if _entry_identity(opened) != _entry_identity(path_stat):
            raise RelocationError(f"{label} changed while opening")
        with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
            descriptor = -1
            payload = json.load(handle)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if not isinstance(payload, Mapping):
        raise RelocationError(f"{label} must contain a JSON object")
    return payload


def atomic_write_bytes(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
    if path_exists(path):
        path_stat = os.lstat(path)
        if not stat.S_ISREG(path_stat.st_mode):
            raise RelocationError(f"refusing to replace non-regular file: {path}")
        with path.open("rb") as handle:
            if handle.read() == payload and _mode(path_stat) == mode:
                return
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def secure_output_parent(path: Path) -> None:
    parent = path.parent
    assert_no_symlink_components(parent, allow_missing=True)
    parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    assert_no_symlink_components(parent, allow_missing=False)
    parent_stat = os.lstat(parent)
    if not stat.S_ISDIR(parent_stat.st_mode):
        raise RelocationError("output parent must be a real directory")
    assert_owner(parent_stat, label="output parent")
    if _mode(parent_stat) & 0o077:
        raise RelocationError("output parent must have owner-only permissions")
    if path_exists(path) and stat.S_ISLNK(os.lstat(path).st_mode):
        raise RelocationError("output path must not be a symlink")


def write_inventory(root: Path, output: Path) -> Mapping[str, Any]:
    root = absolute_path(root, label="inventory_root")
    output = private_inventory_document_path(
        output,
        label="inventory_out",
        allow_missing=True,
    )
    try:
        output.relative_to(root)
    except ValueError:
        pass
    else:
        raise RelocationError("inventory output must stay outside inventory_root")
    inventory = build_inventory(root)
    secure_output_parent(output)
    atomic_write_bytes(output, canonical_json(inventory))
    return inventory


def inventory_projection(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "directories": payload["directories"],
        "files": payload["files"],
        "symlinks": payload["symlinks"],
        "special": payload["special"],
        "totals": payload["totals"],
    }


def transfer_inventory_projection(
    payload: Mapping[str, Any], *, selective: bool | None = None
) -> Mapping[str, Any]:
    if selective is None:
        selective = isinstance(payload.get("selection"), Mapping)
    return {
        "directories": [
            (
                {"relative_path": item["relative_path"]}
                if selective
                else {"relative_path": item["relative_path"], "mode": item["mode"]}
            )
            for item in payload["directories"]
        ],
        "files": [
            {
                "relative_path": item["relative_path"],
                "size_bytes": item["size_bytes"],
                "sha256": item["sha256"],
                "mode": item["mode"],
            }
            for item in payload["files"]
        ],
        "symlinks": payload["symlinks"],
        "special": payload["special"],
        "totals": payload["totals"],
    }


def _nul_files_from(relative_paths: Sequence[str]) -> bytes:
    normalized = sorted(set(relative_paths))
    for relative in normalized:
        _safe_relative(relative, label="rsync files-from path")
        if "\n" in relative or "\r" in relative or relative == ".":
            raise RelocationError("rsync files-from contains an unsafe path")
    return b"".join(value.encode("utf-8") + b"\0" for value in normalized)


def _inventory_subset(
    full: Mapping[str, Any],
    selected_paths: set[str],
    *,
    files_from_sha256: str,
    omitted_audio: Sequence[Mapping[str, Any]],
    completed_call_ids: set[str],
) -> Mapping[str, Any]:
    files = [
        dict(item)
        for item in full["files"]
        if str(item["relative_path"]) in selected_paths
    ]
    directory_paths = {"."}
    for relative in selected_paths:
        parent = PurePosixPath(relative).parent
        while parent.as_posix() != ".":
            directory_paths.add(parent.as_posix())
            parent = parent.parent
    directories = [
        dict(item)
        for item in full["directories"]
        if str(item["relative_path"]) in directory_paths
    ]
    omitted_payload = [dict(item) for item in omitted_audio]
    omitted_bytes = canonical_json({"files": omitted_payload}, indent=None)
    completed_digest = hashlib.sha256(
        "\n".join(sorted(completed_call_ids)).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": INVENTORY_SCHEMA,
        "source_root": full["source_root"],
        "directories": directories,
        "files": files,
        "symlinks": [],
        "special": [],
        "totals": {
            "directories": len(directories),
            "files": len(files),
            "size_bytes": sum(int(item["size_bytes"]) for item in files),
        },
        "selection": {
            "schema_version": "mango_calls_selective_transfer_v1",
            "policy": "all_non_audio_plus_unfinished_and_all_multi_audio",
            "full_inventory_sha256": sha256_bytes(canonical_json(full)),
            "files_from_sha256": files_from_sha256,
            "omitted_audio": omitted_payload,
            "omitted_audio_sha256": sha256_bytes(omitted_bytes),
            "completed_call_ids_count": len(completed_call_ids),
            "completed_call_ids_sha256": completed_digest,
        },
    }


def _embedded_relative(value: Any, root: Path, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RelocationError(f"{label} must be a non-empty absolute path")
    relative, _was_old = classify_embedded_path(value, root, root, label=label)
    return relative.as_posix()


def _source_row_call_ids(path: Path) -> Mapping[Any, str]:
    with closing(sqlite3.connect(immutable_sqlite_uri(path), uri=True, timeout=30)) as connection:
        return {
            row[0]: str(row[1] or "").strip()
            for row in connection.execute(
                "SELECT id, source_call_id FROM call_records"
            )
        }


def build_selective_inventory(root: Path) -> tuple[Mapping[str, Any], bytes]:
    """Build the exact transfer set while retaining completed historical audio."""
    root = absolute_path(root, label="selective_inventory_root")
    full = validate_inventory(build_inventory(root))
    full_files, _directories = inventory_sets(full)
    for relative in REQUIRED_TRANSFER_FILES:
        if relative not in full_files:
            raise RelocationError(f"required pipeline artifact is missing: {relative}")
    validate_transfer_cursor(root)

    ready_manifest = validate_ready_manifest(
        root / READY_MANIFEST_REL,
        root / READY_DB_REL,
        root / WORKING_DB_REL,
        expected_ready_mtime_ns=int(full_files[READY_DB_REL]["mtime_ns"]),
        expected_source_storage=inventory_sqlite_storage_signature(
            full_files, WORKING_DB_REL
        ),
    )
    if ready_manifest.get("status") != "ready":
        raise RelocationError("sealed ready manifest is not ready")
    completed_call_ids = read_completed_call_ids(
        root / WORKING_DB_REL
    ) & read_completed_call_ids(root / READY_DB_REL)
    _capture_bytes, _capture_report = plan_capture(
        root / CAPTURE_REL,
        old_root=root,
        new_root=root,
        files=full_files,
        directories={str(item["relative_path"]) for item in full["directories"]},
        completed_call_ids=completed_call_ids,
    )

    all_audio: set[str] = {
        relative
        for relative in full_files
        if relative.startswith("capture/recordings/")
        or relative.startswith("working/audio/")
    }
    required_audio: set[str] = set()
    omittable_audio: set[str] = set()

    def is_transfer_audio(relative: str) -> bool:
        return relative.startswith("capture/recordings/") or relative.startswith(
            "working/audio/"
        )
    capture_raw = _read_regular_bytes(root / CAPTURE_REL, label="capture manifest")
    capture_rows, _tail = split_capture_rows(capture_raw)
    latest: dict[str, Mapping[str, Any]] = {}
    for _raw, _newline, payload in capture_rows:
        latest[str(payload.get("event_key") or "")] = payload
    for payload in latest.values():
        call_id = str(payload.get("provider_call_id") or "").strip()
        status_value = str(payload.get("status") or "")
        paths: list[tuple[str, Any]] = []
        if payload.get("local_audio_path"):
            paths.append(("local_audio_path", payload["local_audio_path"]))
        if payload.get("canonical_audio_path"):
            paths.append(("canonical_audio_path", payload["canonical_audio_path"]))
        recording_paths = payload.get("recording_paths")
        if recording_paths not in (None, []):
            if not isinstance(recording_paths, list):
                raise RelocationError("capture.recording_paths must be a list")
            paths.extend(("recording_paths", value) for value in recording_paths)
        relative_paths = [
            _embedded_relative(value, root, label=f"capture.{field}")
            for field, value in paths
        ]
        all_audio.update(relative for relative in relative_paths if is_transfer_audio(relative))
        is_multi = status_value == "multiple_recordings_needs_review" or bool(
            recording_paths
        )
        if is_multi:
            required_audio.update(relative_paths)
        elif call_id in completed_call_ids and status_value == "downloaded":
            omittable_audio.update(
                relative for relative in relative_paths if is_transfer_audio(relative)
            )
        elif call_id not in completed_call_ids:
            if status_value == "downloaded" and payload.get("local_audio_path"):
                required_audio.add(
                    _embedded_relative(
                        payload["local_audio_path"], root, label="capture.local_audio_path"
                    )
                )
            if status_value == "duplicate_recording" and payload.get(
                "canonical_audio_path"
            ):
                required_audio.add(
                    _embedded_relative(
                        payload["canonical_audio_path"],
                        root,
                        label="capture.canonical_audio_path",
                    )
                )

    for database in (root / WORKING_DB_REL, root / READY_DB_REL):
        locally_complete, complete_rows = read_completed_rows(database)
        row_call_ids = _source_row_call_ids(database)
        for row_id, source_file in read_source_rows(database):
            relative = _embedded_relative(
                source_file, root, label=f"{database.name}.source_file"
            )
            if is_transfer_audio(relative):
                all_audio.add(relative)
            row_is_complete = bool(
                row_id in complete_rows
                and row_call_ids.get(row_id) in completed_call_ids
                and row_call_ids.get(row_id) in locally_complete
            )
            if row_is_complete and is_transfer_audio(relative):
                omittable_audio.add(relative)
            elif not row_is_complete:
                required_audio.add(relative)

    missing_required = sorted(
        relative
        for relative in required_audio
        if relative not in full_files or int(full_files[relative]["size_bytes"]) <= 0
    )
    if missing_required:
        raise RelocationError("unfinished or multi audio is missing from source inventory")
    # Retain every unreferenced/orphan audio by default.  An audio file may be
    # omitted only when an exact capture/SQLite reference proves that every
    # owner is in the strict working ∩ sealed-ready completion set.
    omitted_paths = omittable_audio - required_audio
    selected_paths = set(full_files) - omitted_paths
    for relative in REQUIRED_TRANSFER_FILES:
        selected_paths.add(relative)
    omitted_audio = [
        dict(full_files[relative])
        for relative in sorted(omitted_paths)
        if relative in full_files
    ]
    files_from = _nul_files_from(sorted(selected_paths))
    return (
        _inventory_subset(
            full,
            selected_paths,
            files_from_sha256=sha256_bytes(files_from),
            omitted_audio=omitted_audio,
            completed_call_ids=completed_call_ids,
        ),
        files_from,
    )


def write_selective_inventory(
    root: Path,
    inventory_output: Path,
    files_from_output: Path,
) -> Mapping[str, Any]:
    inventory_output = private_inventory_document_path(
        inventory_output, label="inventory_out", allow_missing=True
    )
    files_from_output = private_inventory_document_path(
        files_from_output, label="files_from_out", allow_missing=True
    )
    if inventory_output == files_from_output:
        raise RelocationError("inventory and files-from outputs must differ")
    for output in (inventory_output, files_from_output):
        secure_output_parent(output)
    root = absolute_path(root, label="selective_inventory_root")
    probe_rsync_from0(root)
    with optional_process_locks(root):
        inventory, files_from = build_selective_inventory(root)
    atomic_write_bytes(files_from_output, files_from)
    atomic_write_bytes(inventory_output, canonical_json(inventory))
    return inventory


def probe_rsync_from0(source_root: Path, binary: Path = Path("/usr/bin/rsync")) -> None:
    if not binary.is_file():
        raise RelocationError("rsync binary is missing")
    with tempfile.TemporaryDirectory(prefix="mango-rsync-from0-") as temporary:
        result = subprocess.run(
            [
                str(binary),
                "-anR",
                "--from0",
                "--files-from=-",
                f"{source_root}/",
                f"{temporary}/",
            ],
            input=b"",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
        )
    if result.returncode != 0:
        raise RelocationError("rsync does not pass the required --from0 probe")


def verify_selective_source(
    expected_path: Path,
    source_root: Path,
    files_from_path: Path,
) -> Mapping[str, Any]:
    expected_path = private_inventory_document_path(
        expected_path, label="verify_selective_source", allow_missing=False
    )
    files_from_path = private_inventory_document_path(
        files_from_path, label="files_from", allow_missing=False
    )
    expected = validate_inventory(read_private_json(expected_path, label="inventory"))
    if not isinstance(expected.get("selection"), Mapping):
        raise RelocationError("source verification requires a selective inventory")
    source_root = absolute_path(source_root, label="selective_inventory_root")
    if expected["source_root"] != str(source_root):
        raise RelocationError("selective inventory source_root mismatch")
    files_from = _read_regular_bytes(files_from_path, label="files-from")
    wanted_files_from = _nul_files_from(
        [str(item["relative_path"]) for item in expected["files"]]
    )
    if files_from != wanted_files_from or sha256_bytes(files_from) != expected[
        "selection"
    ].get("files_from_sha256"):
        raise RelocationError("selective files-from no longer matches inventory")
    with optional_process_locks(source_root):
        full = validate_inventory(build_inventory(source_root))
        if sha256_bytes(canonical_json(full)) != expected["selection"].get(
            "full_inventory_sha256"
        ):
            raise RelocationError("source tree changed after selective inventory")
    full_files, _directories = inventory_sets(full)
    selected_and_omitted = [
        *expected["files"],
        *expected["selection"].get("omitted_audio", ()),
    ]
    for item in selected_and_omitted:
        current = full_files.get(str(item["relative_path"]))
        if current is None or not _inventory_content_matches(current, item):
            raise RelocationError("selective source evidence no longer matches a file")
    return {
        "status": "source_verified",
        "selected_files": len(expected["files"]),
        "omitted_historical_audio": len(
            expected["selection"].get("omitted_audio", ())
        ),
        "files_from_sha256": sha256_bytes(files_from),
    }


def validate_transfer_cursor(root: Path) -> Mapping[str, Any]:
    cursor = read_private_json(
        root / CURSOR_REL, label="Mango freshness cursor", require_private=False
    )
    if (
        cursor.get("schema_version") != "mango_api_freshness_v1"
        or cursor.get("mango_enumeration_complete") is not True
    ):
        raise RelocationError("Mango freshness cursor is incomplete")
    try:
        parsed_until = datetime.fromisoformat(
            str(cursor.get("until") or "").replace("Z", "+00:00")
        )
        if parsed_until.tzinfo is None or parsed_until.utcoffset() is None:
            raise ValueError
        end_offset = int(cursor.get("manifest_end_offset"))
    except (TypeError, ValueError):
        raise RelocationError("Mango freshness cursor identity is invalid") from None
    manifest_raw = _read_regular_bytes(root / CAPTURE_REL, label="capture manifest")
    if end_offset <= 0 or end_offset > len(manifest_raw):
        raise RelocationError("Mango freshness cursor offset is invalid")
    expected_sha = cursor.get("manifest_snapshot_sha256")
    if (
        not isinstance(expected_sha, str)
        or expected_sha != sha256_bytes(manifest_raw[:end_offset])
    ):
        raise RelocationError("Mango freshness cursor capture digest is invalid")
    return cursor


def verify_inventory(expected_path: Path, target_root: Path) -> Mapping[str, Any]:
    expected_path = private_inventory_document_path(
        expected_path,
        label="verify_inventory",
        allow_missing=False,
    )
    expected = validate_inventory(read_private_json(expected_path, label="inventory"))
    actual = validate_inventory(build_inventory(absolute_path(target_root, label="inventory_root")))
    selective = isinstance(expected.get("selection"), Mapping)
    if transfer_inventory_projection(expected, selective=selective) != transfer_inventory_projection(
        actual, selective=selective
    ):
        raise RelocationError("source and target inventories differ")
    return {
        "status": "verified",
        "source_root": expected["source_root"],
        "target_root": actual["source_root"],
        "files": actual["totals"]["files"],
        "size_bytes": actual["totals"]["size_bytes"],
    }


def _read_regular_bytes(path: Path, *, label: str) -> bytes:
    path_stat = os.lstat(path)
    if not stat.S_ISREG(path_stat.st_mode):
        raise RelocationError(f"{label} must be a regular file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if _entry_identity(opened) != _entry_identity(path_stat):
            raise RelocationError(f"{label} changed while opening")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        current = os.stat(path, follow_symlinks=False)
        if (
            _entry_identity(after) != _entry_identity(path_stat)
            or _entry_identity(current) != _entry_identity(path_stat)
            or after.st_size != path_stat.st_size
            or after.st_mtime_ns != path_stat.st_mtime_ns
        ):
            raise RelocationError(f"{label} changed while reading")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def split_capture_rows(raw: bytes) -> tuple[list[tuple[bytes, bool, MutableMapping[str, Any]]], bytes]:
    rows: list[tuple[bytes, bool, MutableMapping[str, Any]]] = []
    parts = raw.splitlines(keepends=True)
    if not parts and raw:
        parts = [raw]
    offset = 0
    for index, part in enumerate(parts):
        has_newline = part.endswith(b"\n")
        body = part[:-1] if has_newline else part
        is_final = index == len(parts) - 1
        try:
            text = body.decode("utf-8")
        except UnicodeDecodeError as exc:
            if is_final and not has_newline and recoverable_utf8_tail(body, exc):
                return rows, raw[offset:]
            raise RelocationError("capture manifest contains invalid UTF-8") from exc
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            if is_final and not has_newline and recoverable_json_tail(text, exc):
                return rows, raw[offset:]
            raise RelocationError("capture manifest contains non-final or unrecoverable JSON corruption") from exc
        if not isinstance(payload, MutableMapping):
            raise RelocationError("capture manifest row must be a JSON object")
        try:
            entry_from_json(payload)
        except (TypeError, ValueError) as exc:
            raise RelocationError("capture manifest row violates the runtime schema") from exc
        rows.append((part, has_newline, payload))
        offset += len(part)
    return rows, b""


def classify_embedded_path(value: str, old_root: Path, new_root: Path, *, label: str) -> tuple[Path, bool]:
    if "\x00" in value:
        raise RelocationError(f"{label} contains NUL")
    raw = Path(value)
    if not raw.is_absolute() or ".." in raw.parts:
        raise RelocationError(f"{label} must be an absolute normalized path")
    normalized = Path(os.path.normpath(value))
    if str(normalized) != value:
        raise RelocationError(f"{label} is not normalized")
    try:
        return relative_to_root(normalized, old_root, label=label), True
    except RelocationError as exc:
        raise RelocationError(f"{label} must point below old_root before the first durable plan") from exc


def inventory_sets(inventory: Mapping[str, Any]) -> tuple[dict[str, Mapping[str, Any]], set[str]]:
    files = {str(item["relative_path"]): item for item in inventory["files"]}
    directories = {str(item["relative_path"]) for item in inventory["directories"]}
    return files, directories


def map_embedded_path(
    value: str,
    *,
    old_root: Path,
    new_root: Path,
    files: Mapping[str, Mapping[str, Any]],
    directories: set[str],
    required: bool,
    label: str,
) -> tuple[str, bool]:
    relative, was_old = classify_embedded_path(value, old_root, new_root, label=label)
    relative_text = relative.as_posix()
    if required:
        file_entry = files.get(relative_text)
        if file_entry is None or int(file_entry["size_bytes"]) <= 0:
            raise RelocationError(f"{label} target asset is missing or empty")
    elif relative.parent != Path(".") and relative.parent.as_posix() not in directories:
        raise RelocationError(f"{label} target parent is missing")
    mapped = str(new_root / relative)
    return mapped, was_old


def embedded_asset_present(
    value: str,
    *,
    old_root: Path,
    new_root: Path,
    files: Mapping[str, Mapping[str, Any]],
    label: str,
) -> bool:
    relative, _was_old = classify_embedded_path(
        value,
        old_root,
        new_root,
        label=label,
    )
    entry = files.get(relative.as_posix())
    return entry is not None and int(entry["size_bytes"]) > 0


def _relocation_row_is_complete(row: Mapping[str, Any]) -> bool:
    return ready_row_is_complete(row)


def read_completed_rows(path: Path) -> tuple[set[str], set[Any]]:
    """Return unambiguous durable call IDs and row IDs safe to leave audio behind."""
    checks = sqlite_checks(path)
    if checks != {"quick_check": "ok", "integrity_check": "ok"}:
        raise RelocationError(f"SQLite checks failed: {path.name}")
    with closing(sqlite3.connect(immutable_sqlite_uri(path), uri=True, timeout=30)) as connection:
        configure_sqlite_memory_temp(connection)
        connection.execute("PRAGMA query_only=ON")
        columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(call_records)")}
        required = {
            "id",
            "source_call_id",
            "transcription_status",
            "resolve_status",
            "analysis_status",
            "analysis_json",
            "transcript_variants_json",
        }
        if not required.issubset(columns):
            return set(), set()
        connection.row_factory = sqlite3.Row
        rows = [dict(row) for row in connection.execute("SELECT * FROM call_records")]
    counts: dict[str, int] = {}
    for row in rows:
        call_id = str(row.get("source_call_id") or "").strip()
        if call_id:
            counts[call_id] = counts.get(call_id, 0) + 1
    complete_rows = {
        row["id"]
        for row in rows
        if _relocation_row_is_complete(row)
        and str(row.get("source_call_id") or "").strip()
        and counts[str(row.get("source_call_id") or "").strip()] == 1
    }
    complete_calls = {
        str(row.get("source_call_id") or "").strip()
        for row in rows
        if row["id"] in complete_rows
    }
    return complete_calls, complete_rows


def read_completed_call_ids(path: Path) -> set[str]:
    return read_completed_rows(path)[0]


def plan_capture(
    path: Path,
    *,
    old_root: Path,
    new_root: Path,
    files: Mapping[str, Mapping[str, Any]],
    directories: set[str],
    completed_call_ids: set[str],
) -> tuple[bytes, Mapping[str, Any]]:
    recovery_path = path.with_name(f".{path.name}.recovery.json")
    if path_exists(recovery_path):
        try:
            recovery = load_capture_recovery(recovery_path)
        except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
            raise RelocationError("capture recovery ledger is invalid or busy") from exc
        if recovery.get("status") != "resolved" or int(recovery.get("unresolved_count") or 0):
            raise RelocationError("unresolved capture recovery must be acknowledged before relocation")
    raw = _read_regular_bytes(path, label="capture manifest")
    rows, tail = split_capture_rows(raw)
    output = bytearray()
    changed_rows = 0
    changed_paths = 0
    omitted_historical_ready_assets = 0
    completed_event_keys = {
        str(payload.get("event_key") or "").strip()
        for _original, _has_newline, payload in rows
        if str(payload.get("provider_call_id") or "").strip() in completed_call_ids
    }
    for original, has_newline, payload in rows:
        changed = False
        status_value = str(payload.get("status") or "")
        provider_call_id = str(payload.get("provider_call_id") or "").strip()
        canonical_event_key = str(payload.get("canonical_event_key") or "").strip()
        historical_ready = provider_call_id in completed_call_ids or (
            status_value == "duplicate_recording"
            and canonical_event_key in completed_event_keys
        )
        for field in CAPTURE_PATH_FIELDS:
            value = payload.get(field)
            if value in (None, ""):
                continue
            if not isinstance(value, str):
                raise RelocationError(f"capture field {field} must be a string")
            normally_required = (
                field == "local_audio_path" and status_value == "downloaded"
            ) or (
                field == "canonical_audio_path" and status_value == "duplicate_recording"
            )
            required = normally_required and not historical_ready
            present = embedded_asset_present(
                value,
                old_root=old_root,
                new_root=new_root,
                files=files,
                label=f"capture.{field}",
            )
            mapped, was_old = map_embedded_path(
                value,
                old_root=old_root,
                new_root=new_root,
                files=files,
                directories=directories,
                required=required,
                label=f"capture.{field}",
            )
            if normally_required and historical_ready and not present:
                omitted_historical_ready_assets += 1
            if was_old:
                payload[field] = mapped
                changed = True
                changed_paths += 1
        recording_paths = payload.get("recording_paths")
        if recording_paths not in (None, []):
            if not isinstance(recording_paths, list) or any(not isinstance(item, str) for item in recording_paths):
                raise RelocationError("capture.recording_paths must be a list of strings")
            mapped_paths: list[str] = []
            for item in recording_paths:
                present = embedded_asset_present(
                    item,
                    old_root=old_root,
                    new_root=new_root,
                    files=files,
                    label="capture.recording_paths",
                )
                mapped, was_old = map_embedded_path(
                    item,
                    old_root=old_root,
                    new_root=new_root,
                    files=files,
                    directories=directories,
                    required=True,
                    label="capture.recording_paths",
                )
                mapped_paths.append(mapped)
                changed = changed or was_old
                changed_paths += int(was_old)
            if changed:
                payload["recording_paths"] = mapped_paths
        recording_assets = payload.get("recording_assets")
        if status_value == "multiple_recordings_needs_review" and not recording_assets:
            raise RelocationError(
                "multiple recording capture row requires recording_assets integrity metadata"
            )
        if recording_assets not in (None, []):
            if not isinstance(recording_assets, list) or any(
                not isinstance(item, MutableMapping) for item in recording_assets
            ):
                raise RelocationError("capture.recording_assets must be a list of objects")
            recording_ids = payload.get("recording_ids")
            if (
                not isinstance(recording_ids, list)
                or not isinstance(recording_paths, list)
                or len(recording_assets) != len(recording_ids)
                or len(recording_assets) != len(recording_paths)
            ):
                raise RelocationError(
                    "capture.recording_assets must align with recording_ids and recording_paths"
                )
            mapped_assets: list[MutableMapping[str, Any]] = []
            for index, raw_asset in enumerate(recording_assets):
                asset = dict(raw_asset)
                value = asset.get("path")
                if not isinstance(value, str) or not value:
                    raise RelocationError("capture.recording_assets.path must be a string")
                if (
                    str(asset.get("recording_id") or "") != str(recording_ids[index])
                    or value != recording_paths[index]
                ):
                    raise RelocationError(
                        "capture.recording_assets order or identity is invalid"
                    )
                relative_asset, _was_old = classify_embedded_path(
                    value,
                    old_root,
                    new_root,
                    label="capture.recording_assets.path",
                )
                inventory_asset = files.get(relative_asset.as_posix())
                asset_size = asset.get("size_bytes")
                asset_sha = asset.get("checksum_sha256")
                if (
                    inventory_asset is None
                    or not isinstance(asset_size, int)
                    or isinstance(asset_size, bool)
                    or asset_size
                    != int(inventory_asset["size_bytes"])
                    or not isinstance(asset_sha, str)
                    or asset_sha
                    != str(inventory_asset["sha256"])
                ):
                    raise RelocationError(
                        "capture.recording_assets size or checksum_sha256 is invalid"
                    )
                present = embedded_asset_present(
                    value,
                    old_root=old_root,
                    new_root=new_root,
                    files=files,
                    label="capture.recording_assets.path",
                )
                mapped, was_old = map_embedded_path(
                    value,
                    old_root=old_root,
                    new_root=new_root,
                    files=files,
                    directories=directories,
                    required=True,
                    label="capture.recording_assets.path",
                )
                asset["path"] = mapped
                mapped_assets.append(asset)
                changed = changed or was_old
                changed_paths += int(was_old)
            if changed:
                payload["recording_assets"] = mapped_assets
        if changed:
            changed_rows += 1
            encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
            output.extend(encoded)
            if has_newline:
                output.extend(b"\n")
        else:
            output.extend(original)
    output.extend(tail)
    result = bytes(output)
    result_rows, result_tail = split_capture_rows(result)
    if result_tail != tail or len(result_rows) != len(rows):
        raise RelocationError("capture tail/row identity changed while planning")
    return result, {
        "rows": len(rows),
        "changed_rows": changed_rows,
        "changed_paths": changed_paths,
        "omitted_historical_ready_assets": omitted_historical_ready_assets,
        "incomplete_tail_preserved": bool(tail),
        "tail_size_bytes": len(tail),
        "tail_sha256": sha256_bytes(tail) if tail else None,
    }


def immutable_sqlite_uri(path: Path) -> str:
    return f"{path.as_uri()}?mode=ro&immutable=1"


def sqlite_checks(path: Path) -> Mapping[str, str]:
    uri = immutable_sqlite_uri(path)
    with closing(sqlite3.connect(uri, uri=True, timeout=30)) as connection:
        configure_sqlite_memory_temp(connection)
        connection.execute("PRAGMA query_only=ON")
        return {
            "quick_check": str(connection.execute("PRAGMA quick_check").fetchone()[0]),
            "integrity_check": str(connection.execute("PRAGMA integrity_check").fetchone()[0]),
        }


def sqlite_sidecar_snapshot(path: Path) -> Mapping[str, tuple[Any, ...]]:
    result: dict[str, tuple[Any, ...]] = {}
    for suffix in SQLITE_SIDECAR_SUFFIXES:
        candidate = Path(f"{path}{suffix}")
        if not path_exists(candidate):
            continue
        assert_no_symlink_components(candidate, allow_missing=False)
        candidate_stat = os.lstat(candidate)
        if (
            not stat.S_ISREG(candidate_stat.st_mode)
            or candidate_stat.st_nlink != 1
            or candidate_stat.st_uid != os.getuid()
        ):
            raise RelocationError("SQLite sidecar must be an owner single-link regular file")
        if suffix == "-wal" and candidate_stat.st_size != 0:
            raise RelocationError("SQLite has an active WAL sidecar")
        if suffix == "-journal":
            raise RelocationError("SQLite has a rollback journal sidecar")
        if suffix == "-shm" and candidate_stat.st_size != 32768:
            raise RelocationError("SQLite has an unexpected SHM sidecar")
        result[suffix] = (
            _entry_identity(candidate_stat),
            candidate_stat.st_size,
            candidate_stat.st_mtime_ns,
            _mode(candidate_stat),
            sha256_bytes(_read_regular_bytes(candidate, label=f"SQLite {suffix} sidecar")),
        )
    return result


def check_sqlite_files(paths: Sequence[Path]) -> Mapping[str, Any]:
    expected = {"quick_check": "ok", "integrity_check": "ok"}
    for value in paths:
        path = absolute_path(value, label="check_sqlite")
        assert_no_symlink_components(path, allow_missing=False)
        before = os.lstat(path)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != os.getuid()
            or before.st_size <= 0
        ):
            raise RelocationError("checked SQLite must be a non-empty owner single-link regular file")
        database_sha256 = sha256_bytes(_read_regular_bytes(path, label="checked SQLite"))
        sidecars_before = sqlite_sidecar_snapshot(path)
        if sqlite_checks(path) != expected:
            raise RelocationError("SQLite checks failed")
        database_sha256_after = sha256_bytes(_read_regular_bytes(path, label="checked SQLite"))
        sidecars_after = sqlite_sidecar_snapshot(path)
        after = os.lstat(path)
        if (
            _entry_identity(after) != _entry_identity(before)
            or after.st_nlink != before.st_nlink
            or after.st_uid != before.st_uid
            or _mode(after) != _mode(before)
            or after.st_size != before.st_size
            or after.st_mtime_ns != before.st_mtime_ns
            or database_sha256_after != database_sha256
            or sidecars_after != sidecars_before
        ):
            raise RelocationError("SQLite changed during immutable checks")
    return {
        "status": "sqlite_checks_ok",
        "databases": len(paths),
        **expected,
    }


def sqlite_journal_mode(path: Path) -> str:
    header = _read_regular_bytes(path, label="SQLite header")[:100]
    if len(header) < 20 or not header.startswith(b"SQLite format 3\x00"):
        raise RelocationError("SQLite header is invalid")
    write_version, read_version = header[18], header[19]
    if (write_version, read_version) == (2, 2):
        return "wal"
    if (write_version, read_version) == (1, 1):
        return "delete"
    raise RelocationError("SQLite header has an unsupported journal format")


def configure_sqlite_memory_temp(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA temp_store=MEMORY")
    observed = connection.execute("PRAGMA temp_store").fetchone()
    if observed is None or int(observed[0]) != 2:
        raise RelocationError("SQLite connection could not force in-memory temp storage")


def read_source_rows(path: Path) -> list[tuple[Any, str]]:
    checks = sqlite_checks(path)
    if checks != {"quick_check": "ok", "integrity_check": "ok"}:
        raise RelocationError(f"SQLite checks failed: {path.name}")
    with closing(sqlite3.connect(immutable_sqlite_uri(path), uri=True, timeout=30)) as connection:
        configure_sqlite_memory_temp(connection)
        connection.execute("PRAGMA query_only=ON")
        columns = {str(row[1]): row for row in connection.execute("PRAGMA table_info(call_records)")}
        if "id" not in columns or "source_file" not in columns:
            raise RelocationError(f"call_records(id, source_file) is required: {path.name}")
        rows = [(row[0], str(row[1])) for row in connection.execute("SELECT id, source_file FROM call_records ORDER BY id")]
    if len({row[0] for row in rows}) != len(rows):
        raise RelocationError(f"call_records.id must be unique: {path.name}")
    return rows


def plan_sqlite_rows(
    path: Path,
    *,
    old_root: Path,
    new_root: Path,
    files: Mapping[str, Mapping[str, Any]],
    directories: set[str],
    completed_call_ids: set[str],
) -> tuple[list[tuple[Any, str, str]], list[tuple[Any, str]], Mapping[str, Any]]:
    updates: list[tuple[Any, str, str]] = []
    expected_rows: list[tuple[Any, str]] = []
    rows = read_source_rows(path)
    _locally_completed_call_ids, completed_rows = read_completed_rows(path)
    with closing(sqlite3.connect(immutable_sqlite_uri(path), uri=True, timeout=30)) as connection:
        row_call_ids = {
            row[0]: str(row[1] or "").strip()
            for row in connection.execute(
                "SELECT id, source_call_id FROM call_records"
            )
        }
    final_values: set[str] = set()
    omitted_historical_ready_assets = 0
    for row_id, source_file in rows:
        historical_ready = (
            row_id in completed_rows
            and row_call_ids.get(row_id) in completed_call_ids
        )
        present = embedded_asset_present(
            source_file,
            old_root=old_root,
            new_root=new_root,
            files=files,
            label=f"{path.name}.call_records.source_file",
        )
        mapped, was_old = map_embedded_path(
            source_file,
            old_root=old_root,
            new_root=new_root,
            files=files,
            directories=directories,
            required=not historical_ready,
            label=f"{path.name}.call_records.source_file",
        )
        if historical_ready and not present:
            omitted_historical_ready_assets += 1
        if mapped in final_values:
            raise RelocationError(f"relocation would create duplicate source_file values: {path.name}")
        final_values.add(mapped)
        expected_rows.append((row_id, mapped))
        if was_old:
            updates.append((row_id, source_file, mapped))
    return updates, expected_rows, {
        "rows": len(rows),
        "updates": len(updates),
        "completed_call_ids": len(completed_call_ids),
        "omitted_historical_ready_assets": omitted_historical_ready_assets,
    }


def sqlite_storage_signature(
    path: Path,
    *,
    wal_path: Path | None = None,
) -> Mapping[str, Mapping[str, int]]:
    result: dict[str, Mapping[str, int]] = {}
    for label, candidate in (
        ("db", path),
        ("wal", wal_path or Path(str(path) + "-wal")),
    ):
        if path_exists(candidate):
            candidate_stat = os.lstat(candidate)
            if not stat.S_ISREG(candidate_stat.st_mode):
                raise RelocationError("SQLite storage signature contains a non-regular file")
            result[label] = {
                "size_bytes": candidate_stat.st_size,
                "mtime_ns": candidate_stat.st_mtime_ns,
            }
    return result


def inventory_sqlite_storage_signature(
    files: Mapping[str, Mapping[str, Any]],
    relative: str,
) -> Mapping[str, Mapping[str, int]]:
    result: dict[str, Mapping[str, int]] = {}
    for label, candidate in (("db", relative), ("wal", f"{relative}-wal")):
        entry = files.get(candidate)
        if entry is not None:
            result[label] = {
                "size_bytes": int(entry["size_bytes"]),
                "mtime_ns": int(entry["mtime_ns"]),
            }
    return result


def validate_ready_manifest(
    path: Path,
    ready_db: Path,
    working_db: Path,
    *,
    expected_ready_mtime_ns: int | None = None,
    expected_source_storage: Mapping[str, Mapping[str, int]] | None = None,
) -> MutableMapping[str, Any]:
    payload = dict(read_private_json(path, label="ready manifest", require_private=False))
    ready_checks = sqlite_checks(ready_db)
    ready_bytes = _read_regular_bytes(ready_db, label="ready DB")
    ready_stat = ready_db.stat()
    if (
        payload.get("status") != "ready"
        or payload.get("sha256") != sha256_bytes(ready_bytes)
        or payload.get("size_bytes") != ready_stat.st_size
        or payload.get("ready_mtime_ns") != (
            ready_stat.st_mtime_ns if expected_ready_mtime_ns is None else expected_ready_mtime_ns
        )
        or payload.get("quick_check") != "ok"
        or ready_checks != {"quick_check": "ok", "integrity_check": "ok"}
        or payload.get("source_storage") != (
            sqlite_storage_signature(working_db)
            if expected_source_storage is None
            else expected_source_storage
        )
    ):
        raise RelocationError("ready DB manifest seal is invalid")
    return payload


def artifact_metadata(path: Path) -> Mapping[str, Any]:
    value = _read_regular_bytes(path, label=path.name)
    path_stat = path.stat()
    return {
        "sha256": sha256_bytes(value),
        "size_bytes": len(value),
        "mtime_ns": path_stat.st_mtime_ns,
        "mode": _mode(path_stat),
    }


def _metadata_matches(path: Path, expected: Mapping[str, Any]) -> bool:
    if not path_exists(path):
        return False
    path_stat = os.lstat(path)
    if (
        not stat.S_ISREG(path_stat.st_mode)
        or path_stat.st_nlink != 1
        or path_stat.st_uid != os.getuid()
    ):
        return False
    if (
        path_stat.st_size != expected.get("size_bytes")
        or path_stat.st_mtime_ns != expected.get("mtime_ns")
        or _mode(path_stat) != expected.get("mode")
    ):
        return False
    return sha256_bytes(_read_regular_bytes(path, label=path.name)) == expected.get("sha256")


def _metadata_content_matches(path: Path, expected: Mapping[str, Any]) -> bool:
    if not path_exists(path):
        return False
    path_stat = os.lstat(path)
    if not stat.S_ISREG(path_stat.st_mode):
        return False
    if (
        path_stat.st_size != expected.get("size_bytes")
        or path_stat.st_mtime_ns != expected.get("mtime_ns")
    ):
        return False
    return sha256_bytes(_read_regular_bytes(path, label=path.name)) == expected.get("sha256")


@contextmanager
def optional_process_locks(pipeline_root: Path) -> Iterator[None]:
    with ExitStack() as stack:
        for relative in (
            "locks/process_a.lock",
            "locks/process_b.lock",
            "locks/capture.lock",
            "locks/pipeline.lock",
        ):
            path = pipeline_root / relative
            if not path_exists(path):
                continue
            path_stat = os.lstat(path)
            if not stat.S_ISREG(path_stat.st_mode):
                raise RelocationError(f"process lock is not a regular file: {relative}")
            handle = stack.enter_context(path.open("rb"))
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RelocationError(f"process lock is busy: {relative}") from exc
        yield


def state_key(old_root: Path, new_root: Path) -> str:
    contract = json.dumps([str(old_root), str(new_root)], separators=(",", ":"))
    return hashlib.sha256(contract.encode("utf-8")).hexdigest()


class PinnedDirectory:
    def __init__(
        self,
        path: Path,
        descriptor: int,
        *,
        label: str,
        parent: PinnedDirectory | None = None,
        name: str | None = None,
        private: bool,
    ) -> None:
        self.path = path
        self.descriptor = descriptor
        self.label = label
        self.parent = parent
        self.name = name
        self.private = private
        self.identity = _entry_identity(os.fstat(descriptor))

    def verify_bound(self) -> None:
        if self.parent is None:
            current = os.lstat(self.path)
        else:
            self.parent.verify_bound()
            current = os.stat(
                str(self.name),
                dir_fd=self.parent.descriptor,
                follow_symlinks=False,
            )
        opened = os.fstat(self.descriptor)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or not stat.S_ISDIR(current.st_mode)
            or _entry_identity(opened) != self.identity
            or _entry_identity(current) != self.identity
        ):
            raise RelocationError(f"{self.label} is no longer bound to its verified directory")
        assert_owner(opened, label=self.label)
        if self.private and _mode(opened) != 0o700:
            raise RelocationError(f"{self.label} is not owner-only")

    def close(self) -> None:
        os.close(self.descriptor)


@contextmanager
def inside_pinned_directory(directory: PinnedDirectory) -> Iterator[None]:
    # This CLI is single-threaded: a pinned cwd lets SQLite resolve its relative
    # main/journal names against the opened directory inode, not a replaceable path.
    directory.verify_bound()
    previous = os.open(
        ".",
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fchdir(directory.descriptor)
        yield
    finally:
        try:
            os.fchdir(previous)
        finally:
            os.close(previous)
            directory.verify_bound()


def _pin_directory(
    path: Path,
    *,
    label: str,
    private: bool,
    parent: PinnedDirectory | None = None,
    name: str | None = None,
) -> PinnedDirectory:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(
            str(name) if parent is not None else path,
            flags,
            dir_fd=parent.descriptor if parent is not None else None,
        )
    except OSError as exc:
        raise RelocationError(f"{label} is unsafe") from exc
    pinned = PinnedDirectory(
        path,
        descriptor,
        label=label,
        parent=parent,
        name=name,
        private=private,
    )
    try:
        pinned.verify_bound()
    except Exception:
        pinned.close()
        raise
    return pinned


def _open_child_directory(
    parent: PinnedDirectory,
    name: str,
    *,
    label: str,
    create: bool,
) -> PinnedDirectory:
    if not name or Path(name).name != name:
        raise RelocationError(f"{label} name is invalid")
    parent.verify_bound()
    if create:
        try:
            os.mkdir(name, 0o700, dir_fd=parent.descriptor)
            os.fsync(parent.descriptor)
        except FileExistsError:
            pass
    child = _pin_directory(
        parent.path / name,
        label=label,
        private=False,
        parent=parent,
        name=name,
    )
    opened = os.fstat(child.descriptor)
    assert_owner(opened, label=label)
    if _mode(opened) != 0o700:
        os.fchmod(child.descriptor, 0o700)
        os.fsync(child.descriptor)
    child.private = True
    child.verify_bound()
    return child


@contextmanager
def secure_state_store(local_root: Path, key: str) -> Iterator[PinnedDirectory]:
    if len(key) != 64 or any(character not in "0123456789abcdef" for character in key):
        raise RelocationError("relocation state key is invalid")
    local = _pin_directory(
        local_root,
        label="$HOME/.mango_local",
        private=False,
    )
    state_parent: PinnedDirectory | None = None
    state: PinnedDirectory | None = None
    try:
        state_parent = _open_child_directory(
            local,
            "mango_calls_relocation_state",
            label="relocation state parent",
            create=True,
        )
        state = _open_child_directory(
            state_parent,
            key,
            label="relocation state",
            create=True,
        )
        yield state
        state.verify_bound()
    finally:
        if state is not None:
            state.close()
        if state_parent is not None:
            state_parent.close()
        local.close()


@contextmanager
def relocation_lock(state: PinnedDirectory) -> Iterator[None]:
    state.verify_bound()
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open("relocation.lock", flags, 0o600, dir_fd=state.descriptor)
    except OSError as exc:
        raise RelocationError("relocation lock is unsafe") from exc
    try:
        opened = os.fstat(descriptor)
        current = os.stat("relocation.lock", dir_fd=state.descriptor, follow_symlinks=False)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or _entry_identity(opened) != _entry_identity(current)
        ):
            raise RelocationError("relocation lock is unsafe")
        assert_owner(opened, label="relocation lock")
        if _mode(opened) != 0o600:
            os.fchmod(descriptor, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RelocationError("another relocation is already running") from exc
        state.verify_bound()
        yield
        state.verify_bound()
    finally:
        os.close(descriptor)


@contextmanager
def secure_staging_dir(
    state: PinnedDirectory,
    *,
    reset: bool,
) -> Iterator[PinnedDirectory]:
    staging = _open_child_directory(
        state,
        "staging",
        label="staging directory",
        create=reset,
    )
    try:
        if reset:
            with os.scandir(staging.descriptor) as iterator:
                entries = list(iterator)
            planned_names = {"capture.jsonl", "working.sqlite", "ready.sqlite", "ready.manifest.json"}
            cleanup_names = planned_names | {
                f"{name}{suffix}"
                for name in ("working.sqlite", "ready.sqlite")
                for suffix in SQLITE_SIDECAR_SUFFIXES
            }
            for entry in entries:
                if entry.name not in cleanup_names:
                    raise RelocationError("staging directory contains an unexpected entry")
                entry_stat = entry.stat(follow_symlinks=False)
                if not stat.S_ISREG(entry_stat.st_mode) or entry_stat.st_nlink != 1:
                    raise RelocationError("staging directory contains an unsafe planned artifact")
                assert_owner(entry_stat, label="staged artifact")
                os.unlink(entry.name, dir_fd=staging.descriptor)
            os.fsync(staging.descriptor)
        staging.verify_bound()
        yield staging
        staging.verify_bound()
    finally:
        staging.close()


def _read_regular_bytes_at(
    directory: PinnedDirectory,
    name: str,
    *,
    label: str,
    require_private: bool,
) -> bytes:
    if not name or Path(name).name != name:
        raise RelocationError(f"{label} name is invalid")
    directory.verify_bound()
    try:
        path_stat = os.stat(name, dir_fd=directory.descriptor, follow_symlinks=False)
    except FileNotFoundError:
        raise
    if (
        not stat.S_ISREG(path_stat.st_mode)
        or path_stat.st_nlink != 1
        or (require_private and _mode(path_stat) & 0o077)
    ):
        raise RelocationError(f"{label} must be an owner-only regular file")
    assert_owner(path_stat, label=label)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    descriptor = os.open(name, flags, dir_fd=directory.descriptor)
    try:
        opened = os.fstat(descriptor)
        if _entry_identity(opened) != _entry_identity(path_stat):
            raise RelocationError(f"{label} changed while opening")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        current = os.stat(name, dir_fd=directory.descriptor, follow_symlinks=False)
        if (
            _entry_identity(after) != _entry_identity(path_stat)
            or _entry_identity(current) != _entry_identity(path_stat)
            or after.st_nlink != 1
            or after.st_size != path_stat.st_size
            or after.st_mtime_ns != path_stat.st_mtime_ns
        ):
            raise RelocationError(f"{label} changed while reading")
        return b"".join(chunks)
    finally:
        os.close(descriptor)
        directory.verify_bound()


def read_private_json_at(
    directory: PinnedDirectory,
    name: str,
    *,
    label: str,
) -> Mapping[str, Any]:
    raw = _read_regular_bytes_at(
        directory,
        name,
        label=label,
        require_private=True,
    )
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RelocationError(f"{label} is invalid JSON") from exc
    if not isinstance(payload, Mapping):
        raise RelocationError(f"{label} must contain a JSON object")
    return payload


def atomic_write_bytes_at(
    directory: PinnedDirectory,
    name: str,
    payload: bytes,
    *,
    mode: int = 0o600,
) -> None:
    if not name or Path(name).name != name:
        raise RelocationError("state output name is invalid")
    directory.verify_bound()
    try:
        current = _read_regular_bytes_at(
            directory,
            name,
            label=name,
            require_private=True,
        )
    except FileNotFoundError:
        current = None
    if current == payload:
        current_stat = os.stat(name, dir_fd=directory.descriptor, follow_symlinks=False)
        if _mode(current_stat) == mode:
            return
    temporary = f".{name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(temporary, flags, mode, dir_fd=directory.descriptor)
    try:
        os.fchmod(descriptor, mode)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        directory.verify_bound()
        os.replace(
            temporary,
            name,
            src_dir_fd=directory.descriptor,
            dst_dir_fd=directory.descriptor,
        )
        os.fsync(directory.descriptor)
        directory.verify_bound()
    finally:
        try:
            os.unlink(temporary, dir_fd=directory.descriptor)
        except FileNotFoundError:
            pass


def source_inventory_contract(source_inventory_path: Path, old_root: Path) -> tuple[Mapping[str, Any], str]:
    source_stat = os.lstat(source_inventory_path)
    if not stat.S_ISREG(source_stat.st_mode) or _mode(source_stat) & 0o077:
        raise RelocationError("source inventory must be a private regular file")
    assert_owner(source_stat, label="source inventory")
    raw = _read_regular_bytes(source_inventory_path, label="source inventory")
    payload = validate_inventory(json.loads(raw.decode("utf-8")))
    if payload["source_root"] != str(old_root):
        raise RelocationError("source inventory source_root does not match old_root")
    return payload, sha256_bytes(raw)


def preflight_plan(
    pipeline_root: Path,
    old_root: Path,
    new_root: Path,
    inventory: Mapping[str, Any],
    source_inventory: Mapping[str, Any],
) -> Mapping[str, Any]:
    files, directories = inventory_sets(inventory)
    source_files, _source_directories = inventory_sets(source_inventory)
    for relative in REQUIRED_TRANSFER_FILES:
        if relative not in files:
            raise RelocationError(f"required pipeline artifact is missing: {relative}")
    working_journal_mode = sqlite_journal_mode(pipeline_root / WORKING_DB_REL)
    ready_journal_mode = sqlite_journal_mode(pipeline_root / READY_DB_REL)
    if working_journal_mode != "wal" or ready_journal_mode != "delete":
        raise RelocationError("working/ready SQLite journal modes must be WAL/DELETE")
    for relative, label in ((WORKING_DB_REL, "working WAL"), (READY_DB_REL, "ready DELETE")):
        wal = files.get(f"{relative}-wal")
        shm = files.get(f"{relative}-shm")
        journal = files.get(f"{relative}-journal")
        if journal is not None or (wal is not None and int(wal["size_bytes"]) != 0):
            raise RelocationError(f"{label} SQLite has an active sidecar")
        if shm is not None and int(shm["size_bytes"]) != 32768:
            raise RelocationError(f"{label} SQLite has an unexpected SHM sidecar")

    completed_call_ids = read_completed_call_ids(
        pipeline_root / READY_DB_REL
    ) & read_completed_call_ids(pipeline_root / WORKING_DB_REL)
    capture_target, capture_report = plan_capture(
        pipeline_root / CAPTURE_REL,
        old_root=old_root,
        new_root=new_root,
        files=files,
        directories=directories,
        completed_call_ids=completed_call_ids,
    )
    working_updates, working_expected_rows, working_report = plan_sqlite_rows(
        pipeline_root / WORKING_DB_REL,
        old_root=old_root,
        new_root=new_root,
        files=files,
        directories=directories,
        completed_call_ids=completed_call_ids,
    )
    ready_updates, ready_expected_rows, ready_report = plan_sqlite_rows(
        pipeline_root / READY_DB_REL,
        old_root=old_root,
        new_root=new_root,
        files=files,
        directories=directories,
        completed_call_ids=completed_call_ids,
    )
    ready_manifest = validate_ready_manifest(
        pipeline_root / READY_MANIFEST_REL,
        pipeline_root / READY_DB_REL,
        pipeline_root / WORKING_DB_REL,
        expected_ready_mtime_ns=int(source_files[READY_DB_REL]["mtime_ns"]),
        expected_source_storage=inventory_sqlite_storage_signature(source_files, WORKING_DB_REL),
    )
    ready_db_value = ready_manifest.get("ready_db")
    if not isinstance(ready_db_value, str):
        raise RelocationError("ready manifest ready_db is invalid")
    mapped_ready_db, _ = map_embedded_path(
        ready_db_value,
        old_root=old_root,
        new_root=new_root,
        files=files,
        directories=directories,
        required=True,
        label="ready manifest ready_db",
    )
    if mapped_ready_db != str(new_root / READY_DB_REL):
        raise RelocationError("ready manifest ready_db does not map to the ready DB artifact")
    return {
        "capture_bytes": capture_target,
        "capture": capture_report,
        "working_updates": working_updates,
        "working_expected_rows": working_expected_rows,
        "working": working_report,
        "ready_updates": ready_updates,
        "ready_expected_rows": ready_expected_rows,
        "ready": ready_report,
        "ready_manifest": ready_manifest,
        "working_journal_mode": working_journal_mode,
        "ready_journal_mode": ready_journal_mode,
    }


def _create_staged_file_at(
    staging: PinnedDirectory,
    name: str,
    payload: bytes,
) -> None:
    if not name or Path(name).name != name:
        raise RelocationError("staged file name is invalid")
    staging.verify_bound()
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(name, flags, 0o600, dir_fd=staging.descriptor)
    keep_output = False
    try:
        os.fchmod(descriptor, 0o600)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_uid != os.getuid()
            or _mode(opened) != 0o600
            or opened.st_size != len(payload)
        ):
            raise RelocationError("staged file metadata is invalid")
        keep_output = True
    finally:
        os.close(descriptor)
        if not keep_output:
            os.unlink(name, dir_fd=staging.descriptor)
    os.fsync(staging.descriptor)
    staging.verify_bound()


def artifact_metadata_at(
    directory: PinnedDirectory,
    name: str,
) -> Mapping[str, Any]:
    value = _read_regular_bytes_at(
        directory,
        name,
        label=name,
        require_private=True,
    )
    path_stat = os.stat(name, dir_fd=directory.descriptor, follow_symlinks=False)
    return {
        "sha256": sha256_bytes(value),
        "size_bytes": len(value),
        "mtime_ns": path_stat.st_mtime_ns,
        "mode": _mode(path_stat),
    }


def sqlite_checks_at(
    directory: PinnedDirectory,
    name: str,
) -> Mapping[str, str]:
    directory.verify_bound()
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(name, flags, dir_fd=directory.descriptor)
    try:
        opened = os.fstat(descriptor)
        current = os.stat(name, dir_fd=directory.descriptor, follow_symlinks=False)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_uid != os.getuid()
            or _entry_identity(opened) != _entry_identity(current)
        ):
            raise RelocationError("staged SQLite entry is unsafe")
        with closing(
            sqlite3.connect(
                f"file:/dev/fd/{descriptor}?mode=ro&immutable=1",
                uri=True,
                timeout=30,
            )
        ) as connection:
            configure_sqlite_memory_temp(connection)
            connection.execute("PRAGMA query_only=ON")
            result = {
                "quick_check": str(connection.execute("PRAGMA quick_check").fetchone()[0]),
                "integrity_check": str(connection.execute("PRAGMA integrity_check").fetchone()[0]),
            }
        after = os.fstat(descriptor)
        current = os.stat(name, dir_fd=directory.descriptor, follow_symlinks=False)
        if (
            _entry_identity(after) != _entry_identity(opened)
            or _entry_identity(current) != _entry_identity(opened)
            or after.st_nlink != 1
        ):
            raise RelocationError("staged SQLite entry changed during validation")
        return result
    finally:
        os.close(descriptor)
        directory.verify_bound()


def _digest_sqlite_value(digest: Any, value: Any) -> None:
    if value is None:
        payload = b""
        marker = b"n"
    elif isinstance(value, bytes):
        payload = value
        marker = b"b"
    elif isinstance(value, str):
        payload = value.encode("utf-8")
        marker = b"s"
    elif isinstance(value, int):
        payload = str(value).encode("ascii")
        marker = b"i"
    elif isinstance(value, float):
        payload = struct.pack(">d", value)
        marker = b"f"
    else:
        raise RelocationError("SQLite contains an unsupported value type")
    digest.update(marker)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def sqlite_business_digest(
    connection: sqlite3.Connection,
    *,
    ignored_columns: Mapping[str, set[str]],
) -> str:
    digest = hashlib.sha256()
    schema_rows = connection.execute(
        """
        SELECT type, name, tbl_name, COALESCE(sql, '')
        FROM sqlite_master
        WHERE name NOT LIKE 'sqlite_%'
        ORDER BY type, name
        """
    ).fetchall()
    for schema_row in schema_rows:
        for value in schema_row:
            _digest_sqlite_value(digest, value)
    table_names = [
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
    ]
    for table_name in table_names:
        quoted = '"' + table_name.replace('"', '""') + '"'
        cursor = connection.execute(f"SELECT * FROM {quoted}")
        columns = [str(item[0]) for item in cursor.description or ()]
        ignored = ignored_columns.get(table_name, set())
        if not ignored.issubset(columns):
            raise RelocationError("SQLite business digest ignore-list does not match schema")
        kept_indexes = [index for index, name in enumerate(columns) if name not in ignored]
        row_digests: list[bytes] = []
        for row in cursor:
            row_digest = hashlib.sha256()
            for index in kept_indexes:
                _digest_sqlite_value(row_digest, row[index])
            row_digests.append(row_digest.digest())
        digest.update(table_name.encode("utf-8"))
        digest.update(len(row_digests).to_bytes(8, "big"))
        for row_digest in sorted(row_digests):
            digest.update(row_digest)
    return digest.hexdigest()


def _stage_sqlite(
    source: Path,
    staging: PinnedDirectory,
    staged_name: str,
    updates: Sequence[tuple[Any, str, str]],
    expected_rows: Sequence[tuple[Any, str]],
    *,
    journal_mode: str,
) -> None:
    if journal_mode not in {"wal", "delete"}:
        raise RelocationError("unsupported staged SQLite journal mode")
    if not staged_name or Path(staged_name).name != staged_name:
        raise RelocationError("staged SQLite name is invalid")
    expected_header = {"wal": b"\x02\x02", "delete": b"\x01\x01"}[journal_mode]
    staging.verify_bound()
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(staged_name, flags, 0o600, dir_fd=staging.descriptor)
    os.fchmod(descriptor, 0o600)
    os.close(descriptor)
    keep_output = False
    try:
        with inside_pinned_directory(staging):
            with closing(
                sqlite3.connect(
                    immutable_sqlite_uri(source),
                    uri=True,
                    timeout=30,
                )
            ) as source_db, closing(sqlite3.connect(staged_name, timeout=30)) as target_db:
                configure_sqlite_memory_temp(source_db)
                configure_sqlite_memory_temp(target_db)
                source_db.execute("PRAGMA query_only=ON")
                source_business_digest = sqlite_business_digest(
                    source_db,
                    ignored_columns={"call_records": {"source_file"}},
                )
                source_db.backup(target_db)
                target_db.execute("PRAGMA journal_mode=DELETE")
                target_db.execute("BEGIN IMMEDIATE")
                for row_id, before, after in updates:
                    cursor = target_db.execute(
                        "UPDATE call_records SET source_file=? WHERE id=? AND source_file=?",
                        (after, row_id, before),
                    )
                    if cursor.rowcount != 1:
                        raise RelocationError("SQLite row changed while staging relocation")
                target_db.commit()
                observed_rows = [
                    (row[0], str(row[1]))
                    for row in target_db.execute(
                        "SELECT id, source_file FROM call_records ORDER BY id"
                    )
                ]
                if observed_rows != list(expected_rows):
                    raise RelocationError("SQLite relocated paths failed full-table exact readback")
                if sqlite_business_digest(
                    target_db,
                    ignored_columns={"call_records": {"source_file"}},
                ) != source_business_digest:
                    raise RelocationError("SQLite business fields changed during relocation")
                quick = str(target_db.execute("PRAGMA quick_check").fetchone()[0])
                integrity = str(target_db.execute("PRAGMA integrity_check").fetchone()[0])
                if quick != "ok" or integrity != "ok":
                    raise RelocationError("staged SQLite failed validation")
            with closing(sqlite3.connect(staged_name, timeout=30)) as target_db:
                configure_sqlite_memory_temp(target_db)
                observed_mode = str(
                    target_db.execute(f"PRAGMA journal_mode={journal_mode}").fetchone()[0]
                ).casefold()
                if observed_mode != journal_mode:
                    raise RelocationError("staged SQLite journal mode could not be preserved")
                target_db.commit()
            if journal_mode == "wal":
                with closing(sqlite3.connect(staged_name, timeout=30)) as target_db:
                    configure_sqlite_memory_temp(target_db)
                    checkpoint_result = target_db.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
                    if checkpoint_result is None or int(checkpoint_result[0]) != 0:
                        raise RelocationError("staged WAL SQLite checkpoint failed")

        for suffix in SQLITE_SIDECAR_SUFFIXES:
            sidecar_name = f"{staged_name}{suffix}"
            try:
                sidecar_stat = os.stat(
                    sidecar_name,
                    dir_fd=staging.descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                continue
            if (
                not stat.S_ISREG(sidecar_stat.st_mode)
                or sidecar_stat.st_nlink != 1
                or sidecar_stat.st_uid != os.getuid()
            ):
                raise RelocationError("staged SQLite sidecar is unsafe")
            os.unlink(sidecar_name, dir_fd=staging.descriptor)

        descriptor = os.open(
            staged_name,
            os.O_RDWR
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=staging.descriptor,
        )
        try:
            os.fchmod(descriptor, 0o600)
            os.fsync(descriptor)
            opened = os.fstat(descriptor)
            current = os.stat(
                staged_name,
                dir_fd=staging.descriptor,
                follow_symlinks=False,
            )
            header = os.pread(descriptor, 20, 0)
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_nlink != 1
                or opened.st_uid != os.getuid()
                or _mode(opened) != 0o600
                or _entry_identity(opened) != _entry_identity(current)
                or not header.startswith(b"SQLite format 3\x00")
                or header[18:20] != expected_header
            ):
                raise RelocationError("staged SQLite metadata is invalid")
            with closing(
                sqlite3.connect(
                    f"file:/dev/fd/{descriptor}?mode=ro&immutable=1",
                    uri=True,
                    timeout=30,
                )
            ) as check_db:
                configure_sqlite_memory_temp(check_db)
                check_db.execute("PRAGMA query_only=ON")
                copied_rows = [
                    (row[0], str(row[1]))
                    for row in check_db.execute(
                        "SELECT id, source_file FROM call_records ORDER BY id"
                    )
                ]
                if copied_rows != list(expected_rows):
                    raise RelocationError("copied staged SQLite paths failed exact readback")
                if (
                    str(check_db.execute("PRAGMA quick_check").fetchone()[0]) != "ok"
                    or str(check_db.execute("PRAGMA integrity_check").fetchone()[0]) != "ok"
                ):
                    raise RelocationError("copied staged SQLite failed validation")
        finally:
            os.close(descriptor)
        keep_output = True
    finally:
        if not keep_output:
            for candidate in (
                staged_name,
                *(f"{staged_name}{suffix}" for suffix in SQLITE_SIDECAR_SUFFIXES),
            ):
                try:
                    candidate_stat = os.stat(
                        candidate,
                        dir_fd=staging.descriptor,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    continue
                if (
                    stat.S_ISREG(candidate_stat.st_mode)
                    and candidate_stat.st_nlink == 1
                    and candidate_stat.st_uid == os.getuid()
                ):
                    os.unlink(candidate, dir_fd=staging.descriptor)
        os.fsync(staging.descriptor)
        staging.verify_bound()


def build_staged_artifacts(
    pipeline_root: Path,
    new_root: Path,
    staging: PinnedDirectory,
    planned: Mapping[str, Any],
    *,
    relocated_at: str,
) -> Mapping[str, Mapping[str, Any]]:
    staged_names: dict[str, str] = {}
    capture_source = pipeline_root / CAPTURE_REL
    if planned["capture_bytes"] != _read_regular_bytes(capture_source, label="capture manifest"):
        capture_name = "capture.jsonl"
        _create_staged_file_at(staging, capture_name, planned["capture_bytes"])
        staged_names[CAPTURE_REL] = capture_name

    for relative, updates, expected_rows, name, journal_mode in (
        (
            WORKING_DB_REL,
            planned["working_updates"],
            planned["working_expected_rows"],
            "working.sqlite",
            planned["working_journal_mode"],
        ),
        (
            READY_DB_REL,
            planned["ready_updates"],
            planned["ready_expected_rows"],
            "ready.sqlite",
            planned["ready_journal_mode"],
        ),
    ):
        if updates:
            _stage_sqlite(
                pipeline_root / relative,
                staging,
                name,
                updates,
                expected_rows,
                journal_mode=str(journal_mode),
            )
            staged_names[relative] = name

    ready_name = staged_names.get(READY_DB_REL)
    ready_checks = (
        sqlite_checks_at(staging, ready_name)
        if ready_name is not None
        else sqlite_checks(pipeline_root / READY_DB_REL)
    )
    if ready_checks != {"quick_check": "ok", "integrity_check": "ok"}:
        raise RelocationError("staged ready DB failed checks")
    ready_metadata = (
        artifact_metadata_at(staging, ready_name)
        if ready_name is not None
        else artifact_metadata(pipeline_root / READY_DB_REL)
    )
    working_name = staged_names.get(WORKING_DB_REL)
    if working_name is None:
        working_storage = dict(sqlite_storage_signature(pipeline_root / WORKING_DB_REL))
    else:
        working_metadata = artifact_metadata_at(staging, working_name)
        working_storage = {
            "db": {
                "size_bytes": int(working_metadata["size_bytes"]),
                "mtime_ns": int(working_metadata["mtime_ns"]),
            }
        }
        wal_path = Path(str(pipeline_root / WORKING_DB_REL) + "-wal")
        if path_exists(wal_path):
            wal_stat = os.lstat(wal_path)
            if not stat.S_ISREG(wal_stat.st_mode):
                raise RelocationError("working SQLite WAL is not a regular file")
            working_storage["wal"] = {
                "size_bytes": wal_stat.st_size,
                "mtime_ns": wal_stat.st_mtime_ns,
            }
    manifest = dict(planned["ready_manifest"])
    manifest.update(
        {
            "status": "ready",
            "ready_db": str(new_root / READY_DB_REL),
            "sha256": ready_metadata["sha256"],
            "size_bytes": ready_metadata["size_bytes"],
            "ready_mtime_ns": ready_metadata["mtime_ns"],
            "quick_check": "ok",
            "integrity_check": "ok",
            "source_storage": working_storage,
            "relocated_at": relocated_at,
        }
    )
    manifest_bytes = canonical_json(manifest)
    current_manifest = _read_regular_bytes(pipeline_root / READY_MANIFEST_REL, label="ready manifest")
    if manifest_bytes != current_manifest:
        manifest_name = "ready.manifest.json"
        _create_staged_file_at(staging, manifest_name, manifest_bytes)
        staged_names[READY_MANIFEST_REL] = manifest_name

    result: dict[str, Mapping[str, Any]] = {}
    for relative in ARTIFACT_ORDER:
        source = pipeline_root / relative
        before = artifact_metadata(source)
        staged_name = staged_names.get(relative)
        after = artifact_metadata_at(staging, staged_name) if staged_name is not None else before
        result[relative] = {
            "before": before,
            "after": after,
            "staged_name": staged_name,
        }
    return result


def _load_relocation_file_at(
    state: PinnedDirectory,
    name: str,
    *,
    label: str,
) -> Mapping[str, Any] | None:
    try:
        payload = read_private_json_at(state, name, label=label)
    except FileNotFoundError:
        return None
    if payload.get("schema_version") != RELOCATION_SCHEMA:
        raise RelocationError(f"{label} schema is invalid")
    return payload


def _contract_matches(
    payload: Mapping[str, Any],
    *,
    old_root: Path,
    new_root: Path,
    source_inventory_sha256: str,
) -> bool:
    return (
        payload.get("old_root") == str(old_root)
        and payload.get("new_root") == str(new_root)
        and payload.get("source_inventory_sha256") == source_inventory_sha256
    )


def _compare_inventory_to_source(current: Mapping[str, Any], source: Mapping[str, Any]) -> None:
    selective = isinstance(source.get("selection"), Mapping)
    if transfer_inventory_projection(current, selective=selective) != transfer_inventory_projection(
        source, selective=selective
    ):
        raise RelocationError("pipeline_root no longer matches the verified source inventory")


def _inventory_content_matches(
    observed: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> bool:
    return all(
        observed.get(key) == expected.get(key)
        for key in ("size_bytes", "mtime_ns", "sha256")
    )


def permission_normalized_projection(inventory: Mapping[str, Any]) -> Mapping[str, Any]:
    normalized = dict(inventory)
    normalized["directories"] = [
        {**item, "mode": 0o700}
        for item in inventory["directories"]
    ]
    normalized["files"] = [
        {**item, "mode": 0o600}
        for item in inventory["files"]
    ]
    return inventory_projection(normalized)


def _validate_staging_dir_for_plan(
    staging: PinnedDirectory,
    plan: Mapping[str, Any],
) -> None:
    artifacts = plan.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise RelocationError("relocation plan artifacts are invalid")
    allowed_names: set[str] = set()
    for artifact in artifacts.values():
        if not isinstance(artifact, Mapping):
            raise RelocationError("relocation plan artifact is invalid")
        staged_name = artifact.get("staged_name")
        if staged_name is None:
            continue
        if not isinstance(staged_name, str) or not staged_name or Path(staged_name).name != staged_name:
            raise RelocationError("relocation staged name is invalid")
        allowed_names.add(staged_name)

    staging.verify_bound()
    with os.scandir(staging.descriptor) as iterator:
        entries = list(iterator)
    for entry in entries:
        if entry.name not in allowed_names:
            raise RelocationError("staging directory contains an unexpected entry")
        entry_stat = entry.stat(follow_symlinks=False)
        if (
            not stat.S_ISREG(entry_stat.st_mode)
            or entry_stat.st_nlink != 1
            or _mode(entry_stat) != 0o600
        ):
            raise RelocationError("staging directory contains an unsafe planned artifact")
        assert_owner(entry_stat, label="staged artifact")
    staging.verify_bound()


def _validate_resume_tree(
    pipeline_root: Path,
    state: PinnedDirectory,
    plan: Mapping[str, Any],
) -> Mapping[str, Any]:
    before_raw = _read_regular_bytes_at(
        state,
        "before_inventory.json",
        label="before inventory",
        require_private=True,
    )
    if sha256_bytes(before_raw) != plan.get("before_inventory_sha256"):
        raise RelocationError("relocation before-inventory hash is invalid")
    before = validate_inventory(json.loads(before_raw.decode("utf-8")))
    current = validate_inventory(build_inventory(pipeline_root))
    before_files = {str(item["relative_path"]): item for item in before["files"]}
    current_files = {str(item["relative_path"]): item for item in current["files"]}
    before_dirs = {str(item["relative_path"]): item for item in before["directories"]}
    current_dirs = {str(item["relative_path"]): item for item in current["directories"]}
    if set(before_files) != set(current_files) or set(before_dirs) != set(current_dirs):
        raise RelocationError("relocation tree gained or lost entries during resume")
    for relative, before_entry in before_files.items():
        current_entry = current_files[relative]
        current_mode = int(current_files[relative]["mode"])
        if relative in ARTIFACT_ORDER:
            artifact = plan.get("artifacts", {}).get(relative)
            if not isinstance(artifact, Mapping):
                raise RelocationError("relocation plan artifact is missing")
            after_entry = artifact.get("after")
            if not isinstance(after_entry, Mapping):
                raise RelocationError("relocation plan artifact target is invalid")
            if not (
                _inventory_content_matches(current_entry, before_entry)
                or _inventory_content_matches(current_entry, after_entry)
            ):
                raise RelocationError(f"relocation artifact drifted during resume: {relative}")
            if current_mode not in {int(before_entry["mode"]), int(after_entry["mode"]), 0o600}:
                raise RelocationError(f"relocation artifact mode drifted during resume: {relative}")
            continue
        if not _inventory_content_matches(current_entry, before_entry):
            raise RelocationError(f"unplanned file changed during relocation: {relative}")
        if current_mode not in {int(before_entry["mode"]), 0o600}:
            raise RelocationError(f"unplanned file mode changed during relocation: {relative}")
    for relative, before_entry in before_dirs.items():
        current_mode = int(current_dirs[relative]["mode"])
        if current_mode not in {int(before_entry["mode"]), 0o700}:
            raise RelocationError(f"directory mode changed unexpectedly during relocation: {relative}")
    return current


def _apply_permissions(root: Path, inventory: Mapping[str, Any]) -> int:
    file_groups: dict[tuple[int, int], list[tuple[Path, Mapping[str, Any], os.stat_result]]] = {}
    for item in inventory["files"]:
        relative = str(item["relative_path"])
        path = root / relative
        path_stat = os.lstat(path)
        if not stat.S_ISREG(path_stat.st_mode):
            raise RelocationError("file changed before permission repair")
        assert_owner(path_stat, label=relative)
        file_groups.setdefault((path_stat.st_dev, path_stat.st_ino), []).append(
            (path, item, path_stat)
        )
    for group in file_groups.values():
        expected_links = len(group)
        if any(path_stat.st_nlink != expected_links for _path, _item, path_stat in group):
            raise RelocationError("tree contains hard links with aliases outside the tree")
        for _path, item, path_stat in group:
            if (
                path_stat.st_size != int(item["size_bytes"])
                or path_stat.st_mtime_ns != int(item["mtime_ns"])
                or _mode(path_stat) != int(item["mode"])
            ):
                raise RelocationError("file metadata changed before permission repair")

    changed = 0
    directories = sorted((str(item["relative_path"]) for item in inventory["directories"]), key=lambda value: (value.count("/"), value))
    for relative in directories:
        path = root if relative == "." else root / relative
        path_stat = os.lstat(path)
        if not stat.S_ISDIR(path_stat.st_mode):
            raise RelocationError("directory changed before permission repair")
        assert_owner(path_stat, label=relative)
        flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            descriptor = os.open(path, flags)
        except OSError as exc:
            raise RelocationError("directory is unsafe during permission repair") from exc
        try:
            opened = os.fstat(descriptor)
            current_path = os.lstat(path)
            if (
                _entry_identity(opened) != _entry_identity(path_stat)
                or _entry_identity(current_path) != _entry_identity(path_stat)
            ):
                raise RelocationError("directory changed during permission repair")
            if _mode(opened) != 0o700:
                os.fchmod(descriptor, 0o700)
                changed += 1
            after = os.fstat(descriptor)
            current_path = os.lstat(path)
            if (
                _entry_identity(after) != _entry_identity(path_stat)
                or _entry_identity(current_path) != _entry_identity(path_stat)
                or _mode(after) != 0o700
            ):
                raise RelocationError("directory changed during permission repair")
        finally:
            os.close(descriptor)

    for group in file_groups.values():
        path, item, path_stat = group[0]
        expected_links = len(group)
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        try:
            descriptor = os.open(path, flags)
        except OSError as exc:
            raise RelocationError("file is unsafe during permission repair") from exc
        try:
            opened = os.fstat(descriptor)
            current_path = os.lstat(path)
            if (
                not stat.S_ISREG(opened.st_mode)
                or _entry_identity(opened) != _entry_identity(path_stat)
                or _entry_identity(current_path) != _entry_identity(path_stat)
                or opened.st_nlink != expected_links
            ):
                raise RelocationError("file changed during permission repair")
            assert_owner(opened, label=str(item["relative_path"]))
            digest = hashlib.sha256()
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
            after_read = os.fstat(descriptor)
            if (
                _entry_identity(after_read) != _entry_identity(path_stat)
                or after_read.st_nlink != expected_links
                or after_read.st_size != int(item["size_bytes"])
                or after_read.st_mtime_ns != int(item["mtime_ns"])
                or digest.hexdigest() != item["sha256"]
            ):
                raise RelocationError("file changed during permission repair")
            if _mode(after_read) != 0o600:
                os.fchmod(descriptor, 0o600)
                changed += len(group)
            after_chmod = os.fstat(descriptor)
            if (
                _entry_identity(after_chmod) != _entry_identity(path_stat)
                or after_chmod.st_nlink != expected_links
                or _mode(after_chmod) != 0o600
            ):
                raise RelocationError("file changed during permission repair")
            for alias_path, _alias_item, alias_stat in group:
                current_alias = os.lstat(alias_path)
                if _entry_identity(current_alias) != _entry_identity(alias_stat):
                    raise RelocationError("hard-link alias changed during permission repair")
        finally:
            os.close(descriptor)
    return changed


def _commit_plan(
    pipeline_root: Path,
    staging: PinnedDirectory,
    plan: Mapping[str, Any],
    *,
    checkpoint: Callable[[str], None],
) -> None:
    artifacts = plan.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise RelocationError("relocation plan artifacts are invalid")
    for relative in ARTIFACT_ORDER:
        artifact = artifacts.get(relative)
        if not isinstance(artifact, Mapping):
            raise RelocationError("relocation plan is incomplete")
        target = pipeline_root / relative
        before = artifact.get("before")
        after = artifact.get("after")
        if not isinstance(before, Mapping) or not isinstance(after, Mapping):
            raise RelocationError("relocation artifact metadata is invalid")
        if _metadata_content_matches(target, after):
            continue
        if not _metadata_content_matches(target, before):
            raise RelocationError(f"artifact is neither planned before nor after generation: {relative}")
        staged_name = artifact.get("staged_name")
        if not isinstance(staged_name, str) or not staged_name:
            if (
                before.get("sha256") == after.get("sha256")
                and before.get("size_bytes") == after.get("size_bytes")
                and before.get("mtime_ns") == after.get("mtime_ns")
            ):
                continue
            raise RelocationError(f"staged artifact is missing from plan: {relative}")
        try:
            staged_metadata = artifact_metadata_at(staging, staged_name)
        except FileNotFoundError:
            staged_metadata = None
        if staged_metadata != dict(after):
            raise RelocationError(f"staged artifact is missing or changed: {relative}")
        staging.verify_bound()
        os.replace(staged_name, target, src_dir_fd=staging.descriptor)
        os.fsync(staging.descriptor)
        fsync_directory(target.parent)
        checkpoint(f"after_replace:{relative}")
        if not _metadata_matches(target, after):
            raise RelocationError(f"committed artifact failed readback: {relative}")


def _write_complete(
    state: PinnedDirectory,
    *,
    contract: Mapping[str, Any],
    after_inventory: Mapping[str, Any],
) -> Mapping[str, Any]:
    after_bytes = canonical_json(after_inventory)
    atomic_write_bytes_at(state, "after_inventory.json", after_bytes)
    complete = {
        "schema_version": RELOCATION_SCHEMA,
        "status": "complete",
        "old_root": contract["old_root"],
        "new_root": contract["new_root"],
        "source_inventory_sha256": contract["source_inventory_sha256"],
        "after_inventory_sha256": sha256_bytes(after_bytes),
    }
    atomic_write_bytes_at(state, "complete.json", canonical_json(complete))
    return complete


def relocate_pipeline(
    pipeline_root: Path,
    old_root: Path,
    new_root: Path,
    source_inventory_path: Path,
    *,
    execute: bool,
    confirmation: str,
    checkpoint: Callable[[str], None] = lambda _name: None,
) -> Mapping[str, Any]:
    pipeline, old, new, local_root = validate_relocation_roots(
        pipeline_root,
        old_root,
        new_root,
        execute=execute,
    )
    source_inventory_path = private_inventory_document_path(
        source_inventory_path,
        label="source_inventory",
        allow_missing=False,
    )
    try:
        source_inventory_path.relative_to(pipeline)
    except ValueError:
        pass
    else:
        raise RelocationError("source inventory must stay outside pipeline_root")
    source_inventory, source_inventory_sha = source_inventory_contract(source_inventory_path, old)
    state_dir_path = local_root / "mango_calls_relocation_state" / state_key(old, new)
    assert_no_symlink_components(state_dir_path, allow_missing=True)

    with optional_process_locks(pipeline):
        if not execute:
            if path_exists(state_dir_path / "plan.json") and not path_exists(state_dir_path / "complete.json"):
                return {
                    "schema_version": RELOCATION_SCHEMA,
                    "status": "resume_required",
                    "pipeline_root": str(pipeline),
                    "old_root": str(old),
                    "new_root": str(new),
                }
            current = validate_inventory(build_inventory(pipeline))
            _compare_inventory_to_source(current, source_inventory)
            planned = preflight_plan(pipeline, old, new, current, source_inventory)
            return {
                "schema_version": RELOCATION_SCHEMA,
                "status": "dry_run",
                "pipeline_root": str(pipeline),
                "old_root": str(old),
                "new_root": str(new),
                "capture": planned["capture"],
                "working": planned["working"],
                "ready": planned["ready"],
                "permissions_to_change": sum(
                    int(int(item["mode"]) != 0o700) for item in current["directories"]
                ) + sum(int(int(item["mode"]) != 0o600) for item in current["files"]),
            }

        if confirmation != CONFIRM_VALUE:
            raise RelocationError(f"execute requires {CONFIRM_ENV}={CONFIRM_VALUE}")
        with secure_state_store(local_root, state_key(old, new)) as state:
            checkpoint("after_state_open")
            state.verify_bound()
            with relocation_lock(state):
                complete = _load_relocation_file_at(
                    state,
                    "complete.json",
                    label="completion marker",
                )
                if complete is not None:
                    if not _contract_matches(
                        complete,
                        old_root=old,
                        new_root=new,
                        source_inventory_sha256=source_inventory_sha,
                    ):
                        raise RelocationError("completed relocation contract differs from requested roots/inventory")
                    after_payload = validate_inventory(
                        read_private_json_at(
                            state,
                            "after_inventory.json",
                            label="after inventory",
                        )
                    )
                    after_bytes = canonical_json(after_payload)
                    if sha256_bytes(after_bytes) != complete.get("after_inventory_sha256"):
                        raise RelocationError("completed relocation after-inventory hash is invalid")
                    current = validate_inventory(build_inventory(pipeline))
                    if inventory_projection(current) != inventory_projection(after_payload):
                        raise RelocationError("completed relocation tree has changed")
                    return {
                        "schema_version": RELOCATION_SCHEMA,
                        "status": "already_relocated",
                        "pipeline_root": str(pipeline),
                        "old_root": str(old),
                        "new_root": str(new),
                        "changes": 0,
                    }

                plan = _load_relocation_file_at(
                    state,
                    "plan.json",
                    label="relocation plan",
                )
                reset_staging = plan is None
                if plan is None:
                    current = validate_inventory(build_inventory(pipeline))
                    _compare_inventory_to_source(current, source_inventory)
                    planned = preflight_plan(pipeline, old, new, current, source_inventory)
                    started_at = datetime.now(timezone.utc).isoformat()
                    before_bytes = canonical_json(current)
                    atomic_write_bytes_at(state, "before_inventory.json", before_bytes)
                else:
                    if not _contract_matches(
                        plan,
                        old_root=old,
                        new_root=new,
                        source_inventory_sha256=source_inventory_sha,
                    ):
                        raise RelocationError("prepared relocation contract differs from requested roots/inventory")
                    original = plan.get("original_pipeline_root")
                    if str(pipeline) not in {str(original), str(new)}:
                        raise RelocationError("resume pipeline_root is not the original transfer or new_root")

                with secure_staging_dir(state, reset=reset_staging) as staging:
                    if plan is None:
                        artifacts = build_staged_artifacts(
                            pipeline,
                            new,
                            staging,
                            planned,
                            relocated_at=started_at,
                        )
                        plan = {
                            "schema_version": RELOCATION_SCHEMA,
                            "status": "prepared",
                            "created_at": started_at,
                            "original_pipeline_root": str(pipeline),
                            "old_root": str(old),
                            "new_root": str(new),
                            "source_inventory_sha256": source_inventory_sha,
                            "before_inventory_sha256": sha256_bytes(before_bytes),
                            "artifacts": artifacts,
                            "capture": planned["capture"],
                            "working": planned["working"],
                            "working_expected_rows": planned["working_expected_rows"],
                            "ready": planned["ready"],
                            "ready_expected_rows": planned["ready_expected_rows"],
                        }
                        atomic_write_bytes_at(state, "plan.json", canonical_json(plan))
                        checkpoint("after_plan")

                    _validate_resume_tree(pipeline, state, plan)
                    _validate_staging_dir_for_plan(staging, plan)
                    _commit_plan(pipeline, staging, plan, checkpoint=checkpoint)
                    current_before_permissions = _validate_resume_tree(pipeline, state, plan)
                    checkpoint("before_permissions")
                    permission_changes = _apply_permissions(pipeline, current_before_permissions)
                    checkpoint("after_permissions")
                    after_inventory = validate_inventory(build_inventory(pipeline))
                    if inventory_projection(after_inventory) != permission_normalized_projection(
                        current_before_permissions
                    ):
                        raise RelocationError("tree changed across permission repair")
                    for relative in ARTIFACT_ORDER:
                        artifact = plan["artifacts"][relative]
                        expected = {**artifact["after"], "mode": 0o600}
                        if not _metadata_matches(pipeline / relative, expected):
                            raise RelocationError(f"final artifact verification failed: {relative}")
                    ready_manifest = validate_ready_manifest(
                        pipeline / READY_MANIFEST_REL,
                        pipeline / READY_DB_REL,
                        pipeline / WORKING_DB_REL,
                    )
                    if ready_manifest.get("ready_db") != str(new / READY_DB_REL):
                        raise RelocationError("final ready manifest points at the wrong root")
                    for relative, expected_key in (
                        (WORKING_DB_REL, "working_expected_rows"),
                        (READY_DB_REL, "ready_expected_rows"),
                    ):
                        expected_rows = plan.get(expected_key)
                        if not isinstance(expected_rows, list):
                            raise RelocationError("relocation plan SQLite expectations are invalid")
                        observed_rows = read_source_rows(pipeline / relative)
                        if observed_rows != [tuple(row) for row in expected_rows]:
                            raise RelocationError(
                                f"committed SQLite paths failed exact validation: {relative}"
                            )
                    complete = _write_complete(
                        state,
                        contract=plan,
                        after_inventory=after_inventory,
                    )
                    checkpoint("after_complete")
                    return {
                        "schema_version": RELOCATION_SCHEMA,
                        "status": "relocated",
                        "pipeline_root": str(pipeline),
                        "old_root": str(old),
                        "new_root": str(new),
                        "capture": plan["capture"],
                        "working": plan["working"],
                        "ready": plan["ready"],
                        "permission_changes": permission_changes,
                        "after_inventory_sha256": complete["after_inventory_sha256"],
                        "state_dir": str(state.path),
                    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inventory or safely relocate an offline Mango Calls pipeline copy.")
    parser.add_argument("--inventory-root", type=Path)
    parser.add_argument("--selective-inventory-root", type=Path)
    parser.add_argument("--inventory-out", type=Path)
    parser.add_argument("--files-from-out", type=Path)
    parser.add_argument("--verify-inventory", type=Path)
    parser.add_argument("--verify-selective-source", type=Path)
    parser.add_argument("--pipeline-root", type=Path)
    parser.add_argument("--old-root", type=Path)
    parser.add_argument("--new-root", type=Path)
    parser.add_argument("--source-inventory", type=Path)
    parser.add_argument("--check-sqlite", type=Path, nargs="+")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    inventory_build = args.inventory_root is not None and args.inventory_out is not None and args.verify_inventory is None
    inventory_verify = args.verify_inventory is not None and args.inventory_root is not None and args.inventory_out is None
    selective_build = (
        args.selective_inventory_root is not None
        and args.inventory_out is not None
        and args.files_from_out is not None
        and args.verify_selective_source is None
    )
    selective_verify = (
        args.selective_inventory_root is not None
        and args.verify_selective_source is not None
        and args.files_from_out is not None
        and args.inventory_out is None
    )
    relocation = all(value is not None for value in (args.pipeline_root, args.old_root, args.new_root, args.source_inventory)) and (args.dry_run or args.execute)
    sqlite_check = args.check_sqlite is not None
    if sqlite_check and (
        any(
            value is not None
            for value in (
                args.inventory_root,
                args.selective_inventory_root,
                args.inventory_out,
                args.files_from_out,
                args.verify_inventory,
                args.verify_selective_source,
                args.pipeline_root,
                args.old_root,
                args.new_root,
                args.source_inventory,
            )
        )
        or args.dry_run
        or args.execute
    ):
        parser.error("SQLite check mode cannot be combined with inventory or relocation options")
    if sum((inventory_build, inventory_verify, selective_build, selective_verify, relocation, sqlite_check)) != 1:
        parser.error("choose exactly one complete mode: inventory, selective inventory, verify, SQLite check, or relocation")
    if (inventory_build or inventory_verify or selective_build or selective_verify) and any(value is not None for value in (args.pipeline_root, args.old_root, args.new_root, args.source_inventory)):
        parser.error("inventory modes cannot be combined with relocation roots")
    if (inventory_build or inventory_verify or selective_build or selective_verify) and (args.dry_run or args.execute):
        parser.error("inventory modes cannot be combined with dry-run/execute")
    if relocation and any(
        value is not None
        for value in (
            args.inventory_root,
            args.selective_inventory_root,
            args.inventory_out,
            args.files_from_out,
            args.verify_inventory,
            args.verify_selective_source,
        )
    ):
        parser.error("relocation mode cannot be combined with inventory options")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.check_sqlite is not None:
            result = check_sqlite_files(args.check_sqlite)
        elif args.selective_inventory_root is not None and args.inventory_out is not None:
            inventory = write_selective_inventory(
                args.selective_inventory_root,
                args.inventory_out,
                args.files_from_out,
            )
            result = {
                "status": "selective_inventory_written",
                "source_root": inventory["source_root"],
                "inventory_out": str(
                    absolute_path(args.inventory_out, label="inventory_out")
                ),
                "files_from_out": str(
                    absolute_path(args.files_from_out, label="files_from_out")
                ),
                "selected_files": inventory["totals"]["files"],
                "omitted_historical_audio": len(
                    inventory["selection"]["omitted_audio"]
                ),
            }
        elif args.verify_selective_source is not None:
            result = verify_selective_source(
                args.verify_selective_source,
                args.selective_inventory_root,
                args.files_from_out,
            )
        elif args.inventory_out is not None:
            inventory = write_inventory(args.inventory_root, args.inventory_out)
            result: Mapping[str, Any] = {
                "status": "inventory_written",
                "source_root": inventory["source_root"],
                "inventory_out": str(absolute_path(args.inventory_out, label="inventory_out")),
                "files": inventory["totals"]["files"],
                "size_bytes": inventory["totals"]["size_bytes"],
            }
        elif args.verify_inventory is not None:
            result = verify_inventory(args.verify_inventory, args.inventory_root)
        else:
            result = relocate_pipeline(
                args.pipeline_root,
                args.old_root,
                args.new_root,
                args.source_inventory,
                execute=bool(args.execute),
                confirmation=os.environ.get(CONFIRM_ENV, ""),
            )
    except (OSError, ValueError, sqlite3.DatabaseError, RelocationError, json.JSONDecodeError) as exc:
        print(
            json.dumps(
                {
                    "schema_version": RELOCATION_SCHEMA,
                    "status": "failed",
                    "error": type(exc).__name__,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
