#!/usr/bin/env python3
"""Pull one sealed Mango calls drop from an M1 worker over read-only SSH."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Iterator, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mango_mvp.customer_timeline.calls_two_processes import (  # noqa: E402
    CallsTwoProcessesConfig,
)
from mango_mvp.productization.owner_only_io import (  # noqa: E402
    atomic_replace_owner_only_bytes,
    path_has_cloud_marker,
    read_stable_regular_bytes,
    read_stable_regular_bytes_with_path,
    validate_owner_only_directory,
)

from scripts.receive_mango_calls_drop import (  # noqa: E402
    CONFIRMATION as RECEIVER_CONFIRMATION,
    DB_NAME,
    MANIFEST_NAME,
    SCHEMA,
    accept_drop,
    handoff_lock,
    load_package,
    secure_directory,
)

CONFIRMATION = "PULL_MANGO_CALLS_REMOTE_DROP"
TRANSFER_SCHEMA = "mango_calls_remote_transfer_v1"
TRANSFER_MARKER_SUFFIX = ".owner.json"
CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


def fsync_directory(path: Path) -> None:
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


def safe_remote_host(value: str) -> str:
    host = value.rsplit("@", 1)[-1]
    if value.startswith("-") or host.startswith("-") or not re.fullmatch(r"(?:[A-Za-z0-9_.-]+@)?[A-Za-z0-9_.-]+", value):
        raise RuntimeError("remote SSH host is invalid")
    return value


def safe_remote_path(value: str) -> str:
    if (not value.startswith("/") or not re.fullmatch(r"[A-Za-z0-9_./-]+", value)
            or ".." in PurePosixPath(value).parts):
        raise RuntimeError("remote drop path is unsafe")
    return value.rstrip("/")


def checked_run(command: Sequence[str], runner: CommandRunner, label: str) -> None:
    if runner(command).returncode != 0:
        raise RuntimeError(f"remote pull step failed:{label}")


def transfer_provenance(incoming: Path, pipeline_root: Path) -> Mapping[str, str]:
    resolved_pipeline = pipeline_root.expanduser().resolve(strict=False)
    pipeline_key = hashlib.sha256(os.fsencode(str(resolved_pipeline))).hexdigest()[:16]
    return {
        "schema_version": TRANSFER_SCHEMA,
        "incoming_root": str(incoming),
        "pipeline_root": str(resolved_pipeline),
        "transfer_name": f".transfer-{pipeline_key}",
    }


def transfer_marker_path(transfer: Path) -> Path:
    return transfer.with_name(f"{transfer.name}{TRANSFER_MARKER_SUFFIX}")


def cleanup_transfer_marker_temps(marker_path: Path) -> None:
    pattern = re.compile(
        rf"\.{re.escape(marker_path.name)}\.[A-Za-z0-9_-]+\.tmp"
    )
    removed = False
    for entry in marker_path.parent.iterdir():
        if not pattern.fullmatch(entry.name):
            continue
        read_stable_regular_bytes(
            entry,
            label="orphan_ssh_transfer_marker_temp",
            owner_only_mode=0o600,
        )
        entry.unlink()
        removed = True
    if removed:
        fsync_directory(marker_path.parent)


def validate_stale_transfer_inventory(
    transfer: Path,
    expected: Mapping[str, str],
) -> None:
    raw = read_stable_regular_bytes(
        transfer_marker_path(transfer),
        label="ssh_transfer_marker",
        owner_only_mode=0o600,
    )
    try:
        marker = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("stale SSH transfer marker is invalid") from exc
    if marker != dict(expected):
        raise RuntimeError("stale SSH transfer marker does not match this pipeline")
    allowed_files = {
        "manifest.before.json",
        "manifest.after.json",
        MANIFEST_NAME,
        DB_NAME,
    }
    for entry in transfer.iterdir():
        info = os.lstat(entry)
        if stat.S_ISDIR(info.st_mode) and entry.name.startswith(".ssh-auth-"):
            validate_owner_only_directory(
                entry,
                label="stale_ssh_material_directory",
                owner_only_mode=0o700,
            )
            for secret in entry.iterdir():
                if not (
                    secret.name in {"identity", "known_hosts"}
                    or re.fullmatch(
                        r"\.(?:identity|known_hosts)\.[A-Za-z0-9_-]+\.tmp",
                        secret.name,
                    )
                ):
                    raise RuntimeError("stale SSH material inventory is unknown")
                read_stable_regular_bytes(
                    secret,
                    label="stale_ssh_material",
                    owner_only_mode=0o600,
                )
            continue
        if (
            entry.name not in allowed_files
            or not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
        ):
            raise RuntimeError("stale SSH transfer inventory is unknown")


def prepare_transfer_directory(incoming: Path, pipeline_root: Path) -> Path:
    """Replace a proven stale transfer for this pipeline before network I/O."""
    expected = transfer_provenance(incoming, pipeline_root)
    transfer = incoming / expected["transfer_name"]
    marker_path = transfer_marker_path(transfer)
    cleanup_transfer_marker_temps(marker_path)
    if os.path.lexists(transfer):
        validate_owner_only_directory(
            transfer,
            label="stale_ssh_transfer_directory",
            owner_only_mode=0o700,
        )
        if not shutil.rmtree.avoids_symlink_attacks:
            raise RuntimeError("safe stale SSH transfer cleanup is unavailable")
        validate_stale_transfer_inventory(transfer, expected)
        shutil.rmtree(transfer)
        marker_path.unlink()
        fsync_directory(incoming)
    elif os.path.lexists(marker_path):
        raw = read_stable_regular_bytes(
            marker_path,
            label="orphan_ssh_transfer_marker",
            owner_only_mode=0o600,
        )
        try:
            marker = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("orphan SSH transfer marker is invalid") from exc
        if marker != dict(expected):
            raise RuntimeError("orphan SSH transfer marker is not owned by this pipeline")
        marker_path.unlink()
        fsync_directory(incoming)
    try:
        atomic_replace_owner_only_bytes(
            marker_path,
            (json.dumps(expected, ensure_ascii=False, sort_keys=True) + "\n").encode(
                "utf-8"
            ),
            label="ssh_transfer_marker",
        )
        fsync_directory(incoming)
        transfer.mkdir(mode=0o700)
        transfer.chmod(0o700)
        resolved = validate_owner_only_directory(
            transfer,
            label="ssh_transfer_directory",
            owner_only_mode=0o700,
        )
        fsync_directory(incoming)
        return resolved
    except Exception:
        shutil.rmtree(transfer, ignore_errors=True)
        marker_path.unlink(missing_ok=True)
        raise


@dataclass(frozen=True)
class _SshMaterial:
    identity_file: Path
    known_hosts: Path


@contextmanager
def materialized_ssh_files(
    identity_file: Path,
    known_hosts: Path,
    *,
    directory: Path,
) -> Iterator[_SshMaterial]:
    """Bind checked SSH bytes to private paths for the whole transfer."""
    try:
        identity_raw, identity_source = read_stable_regular_bytes_with_path(
            identity_file.expanduser(),
            label="ssh_identity",
            owner_only_mode=0o600,
        )
        known_hosts_raw, known_hosts_source = read_stable_regular_bytes_with_path(
            known_hosts.expanduser(),
            label="ssh_known_hosts",
            owner_only_mode=0o600,
        )
    except RuntimeError as exc:
        raise RuntimeError("SSH files must be owner-only regular files") from exc
    for source in (identity_source, known_hosts_source):
        if source == ROOT or ROOT in source.parents or path_has_cloud_marker(source):
            raise RuntimeError("SSH files must stay outside repository and cloud folders")
    auth_root = Path(tempfile.mkdtemp(prefix=".ssh-auth-", dir=directory))
    auth_root.chmod(0o700)
    try:
        validate_owner_only_directory(
            auth_root,
            label="ssh_material_directory",
            owner_only_mode=0o700,
        )
        identity = auth_root / "identity"
        hosts = auth_root / "known_hosts"
        atomic_replace_owner_only_bytes(
            identity,
            identity_raw,
            label="ssh_material_identity",
        )
        atomic_replace_owner_only_bytes(
            hosts,
            known_hosts_raw,
            label="ssh_material_known_hosts",
        )
        yield _SshMaterial(identity_file=identity, known_hosts=hosts)
    finally:
        shutil.rmtree(auth_root, ignore_errors=True)


def rsync_command(host: str, source: str, destination: Path, *,
                  ssh_material: _SshMaterial) -> list[str]:
    ssh = "/usr/bin/ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o IdentitiesOnly=yes -o ConnectTimeout=15"
    ssh += (
        f" -i {shlex.quote(str(ssh_material.identity_file))}"
        " -o UserKnownHostsFile="
        f"{shlex.quote(str(ssh_material.known_hosts))}"
    )
    return [
        "/usr/bin/rsync", "-a", "--partial", "-e",
        ssh,
        f"{host}:{source}", str(destination),
    ]


def manifest_sha(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("remote sealed manifest must be a regular file")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(payload, Mapping) or payload.get("schema_version") != SCHEMA
            or payload.get("status") != "ready" or not re.fullmatch(r"[0-9a-f]{64}", str(payload.get("sha256") or ""))):
        raise RuntimeError("remote sealed manifest is invalid")
    return str(payload["sha256"])


def final_json(text: str) -> Mapping[str, object]:
    decoder, result = json.JSONDecoder(), None
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping) and not text[index + end:].strip():
            result = value
    if result is None:
        raise RuntimeError("Process B returned no final JSON")
    return result


def pull_drop(*, remote_host: str, remote_drop_root: str, incoming_root: Path,
              pipeline_root: Path, execute: bool, confirmation: str,
              identity_file: Path | None = None, known_hosts: Path | None = None,
              runner: CommandRunner | None = None, lock_held: bool = False) -> Mapping[str, object]:
    host, remote_root = safe_remote_host(remote_host), safe_remote_path(remote_drop_root)
    local_incoming = incoming_root.expanduser()
    if not local_incoming.is_absolute():
        raise RuntimeError("local incoming path is unsafe")
    if not execute:
        return {"status": "dry_run", "mode": "dry_run", "transport": "read_only_ssh_pull"}
    if confirmation != CONFIRMATION:
        raise RuntimeError("explicit pull confirmation is required")
    if identity_file is None or known_hosts is None:
        raise RuntimeError("dedicated SSH identity and known_hosts are required")
    local_incoming = secure_directory(local_incoming, create=True)
    local_incoming = validate_owner_only_directory(
        local_incoming,
        label="ssh_incoming_directory",
        owner_only_mode=0o700,
    )
    if not lock_held:
        with handoff_lock(pipeline_root):
            return pull_drop(
                remote_host=host, remote_drop_root=remote_root, incoming_root=local_incoming,
                pipeline_root=pipeline_root, execute=True, confirmation=CONFIRMATION,
                identity_file=identity_file, known_hosts=known_hosts,
                runner=runner, lock_held=True,
            )
    runner = runner or (lambda command: subprocess.run(command, capture_output=True, text=True, check=False))
    transfer = prepare_transfer_directory(local_incoming, pipeline_root)
    before, after = transfer / "manifest.before.json", transfer / "manifest.after.json"
    remote_manifest, remote_db = f"{remote_root}/{MANIFEST_NAME}", f"{remote_root}/{DB_NAME}"
    started, delta_seeded = time.monotonic(), False
    try:
        with materialized_ssh_files(
            identity_file,
            known_hosts,
            directory=transfer,
        ) as ssh_material:
            checked_run(
                rsync_command(
                    host,
                    remote_manifest,
                    before,
                    ssh_material=ssh_material,
                ),
                runner,
                "manifest_before",
            )
            expected_sha = manifest_sha(before)
            try:
                local_db, _ = load_package(
                    pipeline_root.expanduser().resolve() / "drop",
                    expected_sha,
                    exact=False,
                )
            except (OSError, RuntimeError):
                local_db = None
            if local_db is not None:
                return {
                    "status": "reused",
                    "sha256": expected_sha,
                    "size_bytes": local_db.stat().st_size,
                    "mode": "manifest_only",
                    "remote_files_received": 1,
                    "elapsed_sec": round(time.monotonic() - started, 3),
                    "transport": "read_only_ssh_manifest_only",
                }
            try:
                seed_db, _ = load_package(
                    pipeline_root.expanduser().resolve() / "drop", exact=False
                )
                cloned = subprocess.run(
                    ["/bin/cp", "-c", str(seed_db), str(transfer / DB_NAME)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
                delta_seeded = cloned.returncode == 0
            except (OSError, RuntimeError):
                delta_seeded = False
            checked_run(
                rsync_command(
                    host,
                    remote_db,
                    transfer / DB_NAME,
                    ssh_material=ssh_material,
                ),
                runner,
                "database",
            )
            checked_run(
                rsync_command(
                    host,
                    remote_manifest,
                    after,
                    ssh_material=ssh_material,
                ),
                runner,
                "manifest_after",
            )
        if before.read_bytes() != after.read_bytes() or manifest_sha(after) != expected_sha:
            raise RuntimeError("remote manifest changed during transfer")
        before.replace(transfer / MANIFEST_NAME)
        after.unlink()
        result = accept_drop(
            transfer, pipeline_root, expected_sha, execute=True,
            confirmation=RECEIVER_CONFIRMATION, cleanup=True, lock_held=True,
        )
        return {**result, "mode": "delta" if delta_seeded else "full", "delta_seeded": delta_seeded,
                "remote_files_received": 3, "elapsed_sec": round(time.monotonic() - started, 3),
                "transport": "read_only_ssh_pull"}
    finally:
        if transfer.exists():
            shutil.rmtree(transfer)
        transfer_marker_path(transfer).unlink(missing_ok=True)
        fsync_directory(local_incoming)


def pull_then_process_b(*, remote_host: str, remote_drop_root: str, incoming_root: Path,
                        pipeline_root: Path, config: Path, execute: bool, confirmation: str,
                        identity_file: Path | None = None, known_hosts: Path | None = None,
                        transfer_runner: CommandRunner | None = None,
                        process_runner: CommandRunner | None = None) -> Mapping[str, object]:
    if not execute:
        return pull_drop(
            remote_host=remote_host, remote_drop_root=remote_drop_root, incoming_root=incoming_root,
            pipeline_root=pipeline_root, execute=False, confirmation=confirmation,
            identity_file=identity_file, known_hosts=known_hosts, runner=transfer_runner,
        )
    parsed_config = CallsTwoProcessesConfig.from_json(config)
    configured_pipeline = parsed_config.pipeline_root.expanduser().resolve(
        strict=False
    )
    requested_pipeline = pipeline_root.expanduser().resolve(strict=False)
    if requested_pipeline != configured_pipeline:
        raise RuntimeError("pull pipeline_root does not match runtime config")
    with handoff_lock(pipeline_root):
        transfer = pull_drop(
            remote_host=remote_host, remote_drop_root=remote_drop_root, incoming_root=incoming_root,
            pipeline_root=pipeline_root, execute=True, confirmation=confirmation,
            identity_file=identity_file, known_hosts=known_hosts,
            runner=transfer_runner, lock_held=True,
        )
        runner = process_runner or (lambda command: subprocess.run(command, capture_output=True, text=True, check=False))
        completed = runner([sys.executable, str(ROOT / "scripts" / "run_mango_calls_pipeline.py"),
                            "--config", str(config.expanduser().resolve()), "process-b"])
        payload = final_json(completed.stdout or "")
        status, reason = str(payload.get("status") or ""), str(payload.get("stop_reason") or "")
        if completed.returncode != 0 or not (status == "ok" or (status == "idle" and reason == "drop_unchanged")):
            raise RuntimeError("Process B did not complete the accepted drop")
        return {**transfer, "process_b_status": status, "process_b_stop_reason": reason}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pull a sealed Mango calls drop over read-only SSH.")
    parser.add_argument("--remote-host", required=True)
    parser.add_argument("--remote-drop-root", required=True)
    parser.add_argument("--incoming-root", type=Path, required=True)
    parser.add_argument("--pipeline-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--identity-file", type=Path)
    parser.add_argument("--known-hosts", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirmation", default="")
    args = parser.parse_args(argv)
    try:
        if args.execute and (args.identity_file is None or args.known_hosts is None):
            raise RuntimeError("dedicated SSH identity and known_hosts are required")
        result = pull_then_process_b(
            remote_host=args.remote_host, remote_drop_root=args.remote_drop_root,
            incoming_root=args.incoming_root, pipeline_root=args.pipeline_root,
            config=args.config, execute=args.execute, confirmation=args.confirmation,
            identity_file=args.identity_file, known_hosts=args.known_hosts,
        )
    except Exception as exc:
        result = {"status": "failed", "stop_reason": f"pull_exception:{type(exc).__name__}"}
        print(json.dumps(result, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
