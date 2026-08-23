#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mango_mvp.productization.mango_calls_config import (  # noqa: E402
    load_owner_only_runtime_config,
)
from mango_mvp.productization.mango_calls_service_contract import (  # noqa: E402
    current_git_sha,
    git_worktree_is_clean,
    read_host_id,
)
from mango_mvp.productization.owner_only_io import (  # noqa: E402
    atomic_replace_owner_only_bytes,
    path_has_cloud_marker,
    read_stable_regular_bytes,
    validate_owner_only_directory,
)
from mango_mvp.services.controlled_call_scope import (  # noqa: E402
    CONTROLLED_CAPTURE_REQUEST_SCHEMA,
    load_controlled_capture_request,
)


def _boundary(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RuntimeError("window boundary must be an ISO datetime") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise RuntimeError("window boundary must be timezone-aware")
    return parsed.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def _source_call_id(value: str) -> str:
    if (
        value != value.strip()
        or not value
        or len(value) > 256
        or any(ord(char) < 32 for char in value)
    ):
        raise RuntimeError("source_call_id must be one canonical string")
    return value


def _repair_published_request_hardlinks(out: Path) -> bool:
    """Finish an interrupted same-inode publish without trusting its content."""

    if not os.path.lexists(out):
        return False
    published = os.lstat(out)
    if (
        not stat.S_ISREG(published.st_mode)
        or stat.S_ISLNK(published.st_mode)
        or published.st_uid != os.getuid()
        or stat.S_IMODE(published.st_mode) != 0o600
    ):
        raise RuntimeError("published controlled request is unsafe")
    if published.st_nlink == 1:
        return False
    pattern = re.compile(
        rf"^\.{re.escape(out.name)}\.\d+\.[0-9a-f]{{32}}\.pending$"
    )
    matching: list[Path] = []
    for candidate in out.parent.iterdir():
        if not pattern.fullmatch(candidate.name):
            continue
        current = os.lstat(candidate)
        if (
            stat.S_ISREG(current.st_mode)
            and not stat.S_ISLNK(current.st_mode)
            and current.st_uid == os.getuid()
            and (current.st_dev, current.st_ino)
            == (published.st_dev, published.st_ino)
        ):
            matching.append(candidate)
    if published.st_nlink != len(matching) + 1:
        raise RuntimeError("published controlled request has unknown hardlinks")
    for candidate in matching:
        candidate.unlink()
    repaired = os.lstat(out)
    if (
        (repaired.st_dev, repaired.st_ino)
        != (published.st_dev, published.st_ino)
        or repaired.st_nlink != 1
    ):
        raise RuntimeError("published controlled request repair is incomplete")
    descriptor = os.open(
        out.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return True


def _remove_abandoned_unpublished_requests(out: Path) -> None:
    """Delete only old, owner-only pending files that were never published."""

    pattern = re.compile(
        rf"^\.{re.escape(out.name)}\.(\d+)\.[0-9a-f]{{32}}\.pending$"
    )
    for candidate in out.parent.iterdir():
        match = pattern.fullmatch(candidate.name)
        if not match:
            continue
        current = os.lstat(candidate)
        pid = int(match.group(1))
        if (
            not stat.S_ISREG(current.st_mode)
            or stat.S_ISLNK(current.st_mode)
            or current.st_uid != os.getuid()
            or stat.S_IMODE(current.st_mode) != 0o600
            or current.st_nlink != 1
            or _pid_exists(pid)
        ):
            raise RuntimeError("abandoned controlled request pending is unsafe")
        candidate.unlink()


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return pid > 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create an immutable owner-only request for one isolated Mango call. "
            "Does not call Mango, models, or external systems."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--source-call-id", required=True)
    parser.add_argument("--since", required=True)
    parser.add_argument("--until", required=True)
    parser.add_argument("--expected-count", required=True, type=int)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    previous_umask = os.umask(0o077)
    try:
        # This is intentionally a bootstrap read.  The fully validated runtime
        # config cannot exist until this immutable request and its digest have
        # been created, so using CallsTwoProcessesConfig.from_json() here would
        # introduce an impossible request -> config -> request cycle.
        config = load_owner_only_runtime_config(args.config)
        if str(config.get("processing_scope") or "").strip().lower() != (
            "controlled_1_prepare"
        ):
            raise RuntimeError("controlled request requires a preparation config")
        if str(config.get("runtime_authority_mode") or "").strip().lower() != (
            "isolated_controlled"
        ):
            raise RuntimeError("controlled request requires isolated authority")
        if config.get("require_cutover_authority") is not False or config.get(
            "strict_ready_provenance"
        ) is not True:
            raise RuntimeError("controlled request authority flags are invalid")
        if int(config.get("stage_limit", 0)) != 1:
            raise RuntimeError("controlled request requires stage_limit=1")
        if args.expected_count != 1:
            raise RuntimeError("controlled request expected_count must equal one")
        expected_code_sha = str(config.get("expected_code_sha") or "").strip()
        if not re.fullmatch(r"[0-9a-f]{40}", expected_code_sha):
            raise RuntimeError("expected_code_sha is missing or invalid")
        actual_sha = current_git_sha(ROOT)
        if actual_sha != expected_code_sha or not git_worktree_is_clean(ROOT):
            raise RuntimeError("Git worktree must be clean at expected_code_sha")
        pipeline_root = Path(str(config.get("pipeline_root") or "")).expanduser()
        if not pipeline_root.is_absolute():
            raise RuntimeError("controlled pipeline_root must be absolute")
        pipeline_root = pipeline_root.resolve(strict=False)
        owner_local = (Path.home() / ".mango_local").resolve(strict=True)
        if (
            pipeline_root == owner_local
            or owner_local not in pipeline_root.parents
            or path_has_cloud_marker(pipeline_root)
            or not pipeline_root.name.startswith("controlled-")
        ):
            raise RuntimeError("controlled pipeline_root is not isolated owner-local")
        tenant_id = str(config.get("tenant_id") or "").strip()
        if not tenant_id or len(tenant_id) > 128:
            raise RuntimeError("tenant_id is missing or invalid")
        expected_host_id = str(
            config.get("expected_active_host_id") or ""
        ).strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{2,127}", expected_host_id):
            raise RuntimeError("expected_active_host_id is missing or invalid")
        configured_host_path = str(config.get("host_id_path") or "").strip()
        host_id_path = (
            Path(configured_host_path).expanduser()
            if configured_host_path
            else pipeline_root / "state" / "host_id"
        )
        if not host_id_path.is_absolute():
            raise RuntimeError("host_id_path must be absolute")
        resolved_host_path = host_id_path.resolve(strict=False)
        if (
            path_has_cloud_marker(resolved_host_path)
            or pipeline_root not in resolved_host_path.parents
        ):
            raise RuntimeError("host_id_path must stay below controlled pipeline_root")
        host_id = read_host_id(host_id_path)
        if host_id != expected_host_id:
            raise RuntimeError("active host_id does not match config")
        out = args.out.expanduser()
        if not out.is_absolute():
            raise RuntimeError("request output path must be absolute")
        parent = out.parent.resolve(strict=True)
        if (
            path_has_cloud_marker(parent)
            or (parent != owner_local and owner_local not in parent.parents)
        ):
            raise RuntimeError("request output must stay below $HOME/.mango_local")
        validate_owner_only_directory(
            parent,
            label="controlled_capture_request_parent",
            owner_only_mode=0o700,
        )
        production_cursor = Path(
            str(config.get("production_cursor_guard_path") or "")
        ).expanduser()
        if not production_cursor.is_absolute():
            raise RuntimeError("production_cursor_guard_path must be absolute")
        resolved_production_cursor = production_cursor.resolve(strict=False)
        if (
            path_has_cloud_marker(resolved_production_cursor)
            or owner_local not in resolved_production_cursor.parents
            or pipeline_root in resolved_production_cursor.parents
        ):
            raise RuntimeError("production cursor guard is not isolated")
        payload = {
            "schema_version": CONTROLLED_CAPTURE_REQUEST_SCHEMA,
            "source_call_ids": [_source_call_id(args.source_call_id)],
            "expected_count": 1,
            "since": _boundary(args.since),
            "until": _boundary(args.until),
            "pipeline_root": str(pipeline_root),
            "tenant_id": tenant_id,
            "code_sha": actual_sha,
            "host_id": host_id,
        }
        raw = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest = hashlib.sha256(raw).hexdigest()
        _repair_published_request_hardlinks(out)
        if not os.path.lexists(out):
            _remove_abandoned_unpublished_requests(out)
        pending = parent / f".{out.name}.{os.getpid()}.{uuid.uuid4().hex}.pending"
        created = False
        reused = False
        try:
            atomic_replace_owner_only_bytes(
                pending,
                raw,
                label="controlled_capture_request_pending",
            )
            # Validate the complete authority before publishing its final name.
            load_controlled_capture_request(
                path=pending,
                expected_sha256=digest,
                expected_tenant_id=tenant_id,
                expected_code_sha=actual_sha,
                expected_host_id=host_id,
                host_id_path=host_id_path,
                project_root=ROOT,
                expected_pipeline_root=pipeline_root,
            )
            try:
                os.link(pending, out, follow_symlinks=False)
                created = True
            except FileExistsError:
                existing = read_stable_regular_bytes(
                    out,
                    label="controlled_capture_request_existing",
                    owner_only_mode=0o600,
                )
                if existing != raw:
                    raise RuntimeError(
                        "controlled capture request already exists with different content"
                    )
                reused = True
        finally:
            pending.unlink(missing_ok=True)
        loaded = load_controlled_capture_request(
            path=out,
            expected_sha256=digest,
            expected_tenant_id=tenant_id,
            expected_code_sha=actual_sha,
            expected_host_id=host_id,
            host_id_path=host_id_path,
            project_root=ROOT,
            expected_pipeline_root=pipeline_root,
        )
        print(
            json.dumps(
                {
                    "status": "ok",
                    "path": str(loaded.request_path),
                    "sha256": loaded.request_sha256,
                    "source_call_id_sha256": hashlib.sha256(
                        loaded.source_call_id.encode("utf-8")
                    ).hexdigest(),
                    "expected_count": loaded.expected_count,
                    "created": created,
                    "reused": reused,
                    "runs_mango_api": False,
                    "runs_models": False,
                    "writes_external_systems": False,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - do not echo a private identifier.
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error": type(exc).__name__,
                    "runs_mango_api": False,
                    "runs_models": False,
                    "writes_external_systems": False,
                },
                sort_keys=True,
            )
        )
        return 1
    finally:
        os.umask(previous_umask)


if __name__ == "__main__":
    raise SystemExit(main())
