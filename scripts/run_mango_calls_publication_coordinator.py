#!/usr/bin/env python3
"""Local, fail-closed publication coordinator for Mango Calls.

Phase A only prepares owner-only artifacts.  It never writes Google, Yandex
Disk, CRM, Timeline or any other external system.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import stat
import sys
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mango_mvp.productization.mango_calls_service_contract import (  # noqa: E402
    read_host_id,
    read_stable_regular_bytes,
    safe_alert_payload,
    sha256_file,
    validate_ready_manifest_payload,
)
from scripts.export_daily_mango_calls_resolve import export_day  # noqa: E402
from scripts.publish_current_mango_calls_google import (  # noqa: E402
    atomic_owner_json,
    build_safe_plan,
    load_manager_rows,
    owner_json,
    stable_json_object,
)


MOSCOW = ZoneInfo("Europe/Moscow")
SCHEMA = "mango_calls_publication_coordinator_v1"
COMMANDS = ("current-plan", "daily-close", "daily-alert", "daily-status")


def _read_config(path: Path) -> Mapping[str, Any]:
    try:
        raw = read_stable_regular_bytes(
            path,
            label="publication_coordinator_config",
            owner_only_mode=0o600,
        )
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("publication coordinator config is invalid") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError("publication coordinator config must be an object")
    return payload


def _absolute_without_symlink_components(path: Path, *, label: str) -> Path:
    expanded = path.expanduser()
    absolute = expanded if expanded.is_absolute() else Path.cwd() / expanded
    absolute = absolute.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            break
        if stat.S_ISLNK(info.st_mode):
            raise RuntimeError(f"{label} contains a symbolic link")
    return absolute


def _local_root(config: Mapping[str, Any]) -> Path:
    home_local_raw = _absolute_without_symlink_components(
        Path.home() / ".mango_local", label="$HOME/.mango_local"
    )
    home_local = home_local_raw.resolve(strict=False)
    value = config.get("publication_root") or (
        home_local_raw / "mango_calls_publication"
    )
    root_raw = _absolute_without_symlink_components(
        Path(str(value)), label="publication_root"
    )
    root = root_raw.resolve(strict=False)
    try:
        root.relative_to(home_local)
    except ValueError:
        raise RuntimeError("publication_root must stay below $HOME/.mango_local") from None
    if root == home_local:
        raise RuntimeError("publication_root is unsafe")
    return root


@contextmanager
def _lock(path: Path):
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    parent_info = os.lstat(path.parent)
    if (
        not stat.S_ISDIR(parent_info.st_mode)
        or stat.S_ISLNK(parent_info.st_mode)
        or parent_info.st_uid != os.getuid()
    ):
        raise RuntimeError("publication coordinator lock directory is unsafe")
    path.parent.chmod(0o700)
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise RuntimeError("publication coordinator lock is unsafe") from exc
    with os.fdopen(descriptor, "a+", encoding="utf-8") as handle:
        opened = os.fstat(handle.fileno())
        current = os.lstat(path)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_uid != os.getuid()
            or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)
        ):
            raise RuntimeError("publication coordinator lock is unsafe")
        if stat.S_IMODE(opened.st_mode) != 0o600:
            os.fchmod(handle.fileno(), 0o600)
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("publication coordinator is already running") from exc
        yield


def _ready_paths(config: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    pipeline = Path(str(config["pipeline_root"])).expanduser().resolve(strict=False)
    ready = pipeline / "drop" / "mango_calls_ready.sqlite"
    working = pipeline / "working" / "mango_calls_pipeline.sqlite"
    return pipeline, ready, working


def _ready_manifest(
    config: Mapping[str, Any], ready: Path
) -> Mapping[str, Any]:
    manifest = stable_json_object(
        ready.with_suffix(".manifest.json"), label="ready manifest"
    )
    expected_code_sha = str(config.get("expected_code_sha") or "").strip()
    host_id_path = Path(str(config.get("host_id_path") or "")).expanduser()
    if len(expected_code_sha) != 40 or any(
        value not in "0123456789abcdef" for value in expected_code_sha
    ):
        raise RuntimeError("publication expected_code_sha is invalid")
    expected_host_id = read_host_id(host_id_path)
    errors = validate_ready_manifest_payload(
        manifest,
        require_closure=False,
        expected_code_sha=expected_code_sha,
        expected_host_id=expected_host_id,
    )
    if errors:
        raise RuntimeError("ready manifest rejected: " + ",".join(errors))
    before = os.lstat(ready)
    if not stat.S_ISREG(before.st_mode) or ready.is_symlink():
        raise RuntimeError("ready SQLite must be a regular file")
    actual_sha = sha256_file(ready)
    after = os.lstat(ready)
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or actual_sha != manifest.get("sha256")
        or after.st_size != manifest.get("size_bytes")
    ):
        raise RuntimeError("ready SQLite does not match its sealed manifest")
    return manifest


def _day_verdict(manifest: Mapping[str, Any], day: date) -> Mapping[str, Any]:
    verdicts = manifest.get("daily_verdicts")
    verdict = verdicts.get(day.isoformat()) if isinstance(verdicts, Mapping) else None
    return verdict if isinstance(verdict, Mapping) else {}


def _current_plan(
    config: Mapping[str, Any], root: Path, day: date
) -> Mapping[str, Any]:
    _pipeline, ready, _working = _ready_paths(config)
    manifest_path = ready.with_suffix(".manifest.json")
    manifest = _ready_manifest(config, ready)
    google = config.get("current_google")
    if not isinstance(google, Mapping):
        raise RuntimeError("current_google local plan config is missing")
    owner_email = str(google.get("owner_email") or "").strip()
    allowed = google.get("allowed_emails")
    if not owner_email or not isinstance(allowed, list):
        raise RuntimeError("current_google owner/allowlist is incomplete")
    link_path = google.get("link_evidence")
    links = owner_json(Path(str(link_path)).expanduser()) if link_path else {}
    rows = load_manager_rows(
        ready_db=ready,
        ready_manifest=manifest_path,
        ready_manifest_payload=manifest,
        day=day,
        owner_email=owner_email,
        allowed_emails=[str(value) for value in allowed],
        link_evidence=links,
    )
    plan = build_safe_plan(day=day, rows=rows, ready_manifest=manifest)
    output = root / "current" / f"{day.isoformat()}.safe-plan.json"
    atomic_owner_json(output, plan)
    return {
        "status": "planned",
        "day": day.isoformat(),
        "rows": len(rows),
        "plan": str(output),
        "external_write": False,
    }


def _daily_export(
    config: Mapping[str, Any], root: Path, day: date, *, sealed_only: bool
) -> Mapping[str, Any]:
    _pipeline, ready, working = _ready_paths(config)
    daily = config.get("daily_export")
    daily = daily if isinstance(daily, Mapping) else {}
    manager_users = (
        Path(str(daily["manager_users"])).expanduser()
        if daily.get("manager_users")
        else None
    )
    kwargs: dict[str, Any] = {"sealed_only": sealed_only}
    for key, argument in (
        ("tallanto_export", "tallanto_export"),
        ("tallanto_env", "tallanto_env"),
    ):
        if daily.get(key):
            kwargs[argument] = Path(str(daily[key])).expanduser()
    if daily.get("tallanto_snapshot_as_of"):
        kwargs["tallanto_snapshot_as_of"] = datetime.fromisoformat(
            str(daily["tallanto_snapshot_as_of"]).replace("Z", "+00:00")
        )
    return export_day(
        ready,
        working,
        root / "daily",
        day,
        manager_users,
        **kwargs,
    )


def _daily_close(
    config: Mapping[str, Any], root: Path, day: date
) -> Mapping[str, Any]:
    _pipeline, ready, _working = _ready_paths(config)
    manifest = _ready_manifest(config, ready)
    attempts: list[Mapping[str, Any]] = []
    verdicts = manifest.get("daily_verdicts")
    catch_up_days = max(3, int(config.get("max_catch_up_days") or 7))
    earliest = day - timedelta(days=catch_up_days)
    candidates = {day}
    if isinstance(verdicts, Mapping):
        for raw_day in verdicts:
            try:
                candidate = date.fromisoformat(str(raw_day))
            except ValueError:
                continue
            if earliest <= candidate <= day:
                candidates.add(candidate)
    for candidate in sorted(candidates):
        verdict = _day_verdict(manifest, candidate)
        if verdict.get("closure_ok") is not True:
            attempts.append(
                {"day": candidate.isoformat(), "status": "pending_closure"}
            )
            continue
        try:
            result = _daily_export(config, root, candidate, sealed_only=True)
            attempts.append(
                {
                    "day": candidate.isoformat(),
                    "status": "closed",
                    "reused": bool(result.get("reused")),
                    "package_status": result.get("package_status"),
                }
            )
        except Exception as exc:
            attempts.append(
                {
                    "day": candidate.isoformat(),
                    "status": "failed",
                    "error_type": type(exc).__name__,
                }
            )
    target = next(item for item in attempts if item["day"] == day.isoformat())
    return {
        "status": target["status"],
        "day": day.isoformat(),
        "attempts": attempts,
        "external_write": False,
    }


def _daily_alert(
    config: Mapping[str, Any], root: Path, day: date
) -> Mapping[str, Any]:
    pipeline, ready, _working = _ready_paths(config)
    manifest_error = False
    try:
        manifest = _ready_manifest(config, ready)
        verdict = _day_verdict(manifest, day)
    except Exception:
        manifest_error = True
        verdict = {}
    try:
        free_bytes = shutil.disk_usage(pipeline).free
    except OSError:
        free_bytes = 0
    required_free = int(float(config.get("min_free_gib", 40)) * 1024**3)
    pending = int(verdict.get("pending_unique") or 0)
    mango = int(verdict.get("mango_unique") or 0)
    oldest = float(verdict.get("oldest_pending_age_minutes") or 0)
    reasons = []
    if manifest_error:
        reasons.append("ready_manifest_unavailable")
    if verdict.get("closure_ok") is not True:
        reasons.append("day_not_closed")
    if int(verdict.get("pending_over_sla") or 0) > 0:
        reasons.append("pending_over_sla")
    if mango > 0 and pending == mango and oldest > 60:
        reasons.append("all_calls_pending_over_60_minutes")
    if free_bytes < required_free:
        reasons.append("disk_below_threshold")
    payload = safe_alert_payload(
        {
            "schema_version": SCHEMA,
            "status": "alert" if reasons else "ok",
            "stop_reason": ",".join(reasons),
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "closure_ok": verdict.get("closure_ok") is True,
            "mango_unique": mango,
            "pending_unique": pending,
            "pending_over_sla": int(verdict.get("pending_over_sla") or 0),
            "oldest_pending_age_minutes": oldest,
            "free_bytes": free_bytes,
            "required_free_bytes": required_free,
        }
    )
    alert_path = root / "alerts" / f"{day.isoformat()}.json"
    atomic_owner_json(alert_path, payload)
    return {**payload, "alert": str(alert_path), "external_write": False}


def _daily_status(
    config: Mapping[str, Any], root: Path, day: date
) -> Mapping[str, Any]:
    _pipeline, ready, _working = _ready_paths(config)
    closed = False
    try:
        manifest = _ready_manifest(config, ready)
        closed = _day_verdict(manifest, day).get("closure_ok") is True
        package = _daily_export(config, root, day, sealed_only=closed)
        result: dict[str, Any] = {
            "status": "final" if closed else "incomplete",
            "package_status": package.get("package_status"),
            "reused": bool(package.get("reused")),
        }
    except Exception as exc:
        result = {
            "status": "failed",
            "package_status": (
                "FINAL_EXPORT_FAILED" if closed else "INCOMPLETE_EXPORT_FAILED"
            ),
            "error_type": type(exc).__name__,
        }
    result.update(
        schema_version=SCHEMA,
        day=day.isoformat(),
        closure_ok=closed,
        generated_at=datetime.now(timezone.utc).isoformat(),
        external_write=False,
    )
    status_path = root / "status" / f"{day.isoformat()}.json"
    atomic_owner_json(status_path, result)
    return {**result, "status_file": str(status_path)}


def run(
    config_path: Path,
    command: str,
    *,
    day: Optional[date] = None,
) -> Mapping[str, Any]:
    config = _read_config(config_path)
    root = _local_root(config)
    root.mkdir(parents=True, mode=0o700, exist_ok=True)
    root.chmod(0o700)
    target_day = day or (
        datetime.now(MOSCOW).date()
        if command == "current-plan"
        else datetime.now(MOSCOW).date() - timedelta(days=1)
    )
    with _lock(root / "locks" / "coordinator.lock"):
        if command == "current-plan":
            result = _current_plan(config, root, target_day)
        elif command == "daily-close":
            result = _daily_close(config, root, target_day)
        elif command == "daily-alert":
            result = _daily_alert(config, root, target_day)
        elif command == "daily-status":
            result = _daily_status(config, root, target_day)
        else:
            raise ValueError("unsupported coordinator command")
    report = {
        "schema_version": SCHEMA,
        "command": command,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        **result,
    }
    atomic_owner_json(root / "state" / f"{command}.json", report)
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build local Mango Calls publication artifacts without external writes."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--day", type=date.fromisoformat)
    args = parser.parse_args(argv)
    result = run(args.config, args.command, day=args.day)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("status") not in {"failed", "alert"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
