#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
import os
import plistlib
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Sequence


OLD_LABEL = "com.mango.calls-two-processes"
LABEL_A = "com.mango.calls-process-a"
LABEL_B = "com.mango.calls-process-b"
LABEL_CAPTURE = "com.mango.calls-capture"
LABEL_PIPELINE = "com.mango.calls-pipeline"
LABEL_WATCHDOG = "com.mango.calls-watchdog"
LABEL_DAILY_0600 = "com.mango.calls-publication-close-0600"
LABEL_DAILY_0700 = "com.mango.calls-publication-close-0700"
LABEL_DAILY_0800 = "com.mango.calls-publication-close-0800"
LABEL_DAILY_ALERT = "com.mango.calls-publication-alert-0830"
LABEL_DAILY_STATUS = "com.mango.calls-publication-status-0850"
MIN_INTERVAL_SECONDS = 300
DEFAULT_INTERVAL_SECONDS = 900
ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render or install separate launchd triggers for Mango calls A/B.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--env-file", required=True)
    parser.add_argument("--interval-seconds", type=int, help="Legacy default for both process intervals.")
    parser.add_argument("--process-a-interval-seconds", type=int)
    parser.add_argument("--process-b-interval-seconds", type=int)
    parser.add_argument("--process-a-only", action="store_true", help="Render/install only Process A for an M1 worker host.")
    parser.add_argument("--process-b-only", action="store_true", help="Render/install only Process B for the Timeline host.")
    parser.add_argument(
        "--fast-service",
        action="store_true",
        help="Render split capture/pipeline/watchdog plus demand-only Process B.",
    )
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--out", help="Legacy primary plist path; process B is written next to it.")
    parser.add_argument("--out-dir", help="Directory for rendered plists.")
    return parser


def _interval(value: Optional[int], fallback: int) -> int:
    interval = fallback if value is None else value
    if interval < MIN_INTERVAL_SECONDS:
        raise ValueError("interval must be at least 300 seconds")
    return interval


def _plist_paths(args: argparse.Namespace) -> dict[str, Path]:
    if args.out and args.out_dir:
        raise ValueError("--out and --out-dir are mutually exclusive")
    if args.out and args.fast_service:
        raise ValueError("--fast-service requires --out-dir or the standard LaunchAgents directory")
    if args.out:
        primary = Path(args.out).expanduser().resolve()
        return {
            LABEL_A: primary,
            LABEL_B: primary.with_name(f"{LABEL_B}.plist"),
        }
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else Path.home() / "Library" / "LaunchAgents"
    )
    labels = (
        (
            LABEL_CAPTURE,
            LABEL_PIPELINE,
            LABEL_B,
            LABEL_WATCHDOG,
            LABEL_DAILY_0600,
            LABEL_DAILY_0700,
            LABEL_DAILY_0800,
            LABEL_DAILY_ALERT,
            LABEL_DAILY_STATUS,
        )
        if args.fast_service
        else (LABEL_A, LABEL_B)
    )
    return {label: out_dir / f"{label}.plist" for label in labels}


def _payload(
    *,
    label: str,
    command: str,
    interval_seconds: Optional[int],
    config_path: Path,
    env_path: Path,
    log_dir: Path,
    calendar_minutes: Optional[Sequence[int]] = None,
    calendar_schedule: Optional[Sequence[dict[str, int]]] = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "Label": label,
        "ProgramArguments": [
            "/bin/zsh",
            str(ROOT / "scripts" / "run_mango_calls_process.sh"),
            str(config_path),
            str(env_path),
            command,
        ],
        "WorkingDirectory": str(ROOT),
        "RunAtLoad": False,
        "ProcessType": "Background",
        "ThrottleInterval": 60,
        "StandardOutPath": str(log_dir / f"{command}.stdout.log"),
        "StandardErrorPath": str(log_dir / f"{command}.stderr.log"),
    }
    if interval_seconds is not None:
        payload["StartInterval"] = interval_seconds
    if calendar_minutes is not None:
        payload["StartCalendarInterval"] = [
            {"Minute": int(minute)} for minute in calendar_minutes
        ]
    if calendar_schedule is not None:
        payload["StartCalendarInterval"] = [dict(value) for value in calendar_schedule]
    return payload


def _validate_fast_install_authority(
    config_path: Path, config: dict[str, object]
) -> None:
    if config.get("require_cutover_authority") is not True:
        raise RuntimeError("fast service install requires cutover authority")
    expected = str(config.get("expected_code_sha") or "")
    host = Path(str(config.get("host_id_path") or "")).expanduser()
    cutover = Path(str(config.get("cutover_manifest_path") or "")).expanduser()
    if not expected or not host.is_file() or not cutover.is_file():
        raise RuntimeError("fast service cutover proof is incomplete")
    import sys

    src = ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from mango_mvp.customer_timeline.calls_two_processes import CallsTwoProcessesConfig
    from mango_mvp.productization.mango_calls_service_contract import (
        sha256_file,
        verify_cutover_authority,
    )

    parsed = CallsTwoProcessesConfig.from_json(config_path)
    if not parsed.require_cutover_authority or not parsed.strict_ready_provenance:
        raise RuntimeError("fast service install requires the complete strict config")
    cursor = parsed.cursor_path
    if not cursor.is_file() or cursor.is_symlink():
        raise RuntimeError("fast service transferred cursor is missing")

    report = verify_cutover_authority(
        cutover_manifest_path=cutover,
        host_id_path=host,
        expected_code_sha=expected,
        project_root=ROOT,
        expected_source_cursor_sha256=sha256_file(cursor),
        proof_max_age_minutes=int(config.get("cutover_proof_max_age_minutes") or 90),
        require_fresh_previous_host_proof=True,
    )
    if report.get("ok") is not True:
        raise RuntimeError("fast service cutover authority validation failed")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_private_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(path):
        current = os.lstat(path)
        if not stat.S_ISREG(current.st_mode) or path.is_symlink():
            raise RuntimeError(f"refusing to replace unsafe plist path: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            os.fchmod(handle.fileno(), 0o600)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if os.path.lexists(path):
            current = os.lstat(path)
            if not stat.S_ISREG(current.st_mode) or path.is_symlink():
                raise RuntimeError(f"refusing to replace unsafe plist path: {path}")
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _write_plist(path: Path, payload: dict[str, object]) -> None:
    _atomic_private_bytes(path, plistlib.dumps(payload, sort_keys=True))


def _bootout_if_loaded(domain: str, label: str) -> bool:
    target = f"{domain}/{label}"
    probe = subprocess.run(
        ["launchctl", "print", target],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if probe.returncode != 0:
        return False
    result = subprocess.run(
        ["launchctl", "bootout", target],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"launchctl_bootout_failed:{label}")
    return True


def _is_loaded(domain: str, label: str) -> bool:
    return subprocess.run(
        ["launchctl", "print", f"{domain}/{label}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def _validate_local_file(path: Path, *, secret: bool) -> None:
    info = path.lstat()
    mode = stat.S_IMODE(info.st_mode)
    if (
        path.is_symlink()
        or not stat.S_ISREG(info.st_mode)
        or info.st_uid != os.getuid()
        or mode != 0o600
    ):
        kind = "env" if secret else "config"
        raise RuntimeError(f"{kind} file permissions are unsafe")


def _live_process_lock(pipeline_root: Path, name: str) -> bool:
    path = pipeline_root / "locks" / f"{name}.lock"
    try:
        handle = path.open("r+", encoding="utf-8")
    except FileNotFoundError:
        return False
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return False
    finally:
        handle.close()


def _standard_plist(label: str) -> Path:
    return (Path.home() / "Library" / "LaunchAgents" / f"{label}.plist").resolve()


def _conflicting_plist_exists(labels: Sequence[str]) -> bool:
    return any(_standard_plist(label).exists() for label in labels)


def _bootstrap(domain: str, path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["launchctl", "bootstrap", domain, str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )


def _restore_incumbents(
    *,
    domain: str,
    paths: dict[str, Path],
    previous_files: dict[str, Optional[bytes]],
    loaded_before: dict[str, bool],
) -> list[str]:
    errors: list[str] = []
    for label in paths:
        try:
            _bootout_if_loaded(domain, label)
        except RuntimeError:
            errors.append(f"rollback_bootout_failed:{label}")
    for label, path in paths.items():
        previous = previous_files[label]
        try:
            if previous is None:
                path.unlink(missing_ok=True)
            else:
                _atomic_private_bytes(path, previous)
        except OSError:
            errors.append(f"rollback_file_restore_failed:{label}")
    for label in reversed(tuple(paths)):
        if not loaded_before.get(label):
            continue
        restored = _bootstrap(domain, paths[label])
        if restored.returncode != 0:
            errors.append(f"rollback_bootstrap_failed:{label}")
    return errors


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.process_a_only and args.process_b_only:
        raise ValueError("--process-a-only and --process-b-only are mutually exclusive")
    if args.fast_service and (args.process_a_only or args.process_b_only):
        raise ValueError("--fast-service is mutually exclusive with single-role modes")
    if args.process_b_interval_seconds is not None and not args.process_b_only:
        raise ValueError("process B is demand-only and must not have an interval")
    if not args.install and not args.out and not args.out_dir:
        raise ValueError("render-only mode requires an explicit --out or --out-dir")
    config_path = Path(args.config).expanduser().absolute()
    env_path = Path(args.env_file).expanduser().absolute()
    if not config_path.is_file() or not env_path.is_file():
        raise FileNotFoundError("config and env file must exist")
    _validate_local_file(config_path, secret=False)
    _validate_local_file(env_path, secret=True)
    fallback_interval = _interval(args.interval_seconds, DEFAULT_INTERVAL_SECONDS)
    intervals: dict[str, Optional[int]] = {
        LABEL_A: _interval(args.process_a_interval_seconds, fallback_interval),
        # Process B is demand-only. It is kicked off by the Process A wrapper
        # only after Process A returns an explicit successful status.
        LABEL_B: _interval(args.process_b_interval_seconds, fallback_interval) if args.process_b_only else None,
    }
    config = json.loads(config_path.read_text(encoding="utf-8"))
    pipeline_root = Path(str(config["pipeline_root"])).expanduser().resolve()
    log_dir = pipeline_root / "logs"
    paths = _plist_paths(args)
    if args.install and any(path != _standard_plist(label) for label, path in paths.items()):
        raise RuntimeError("install mode must use the standard LaunchAgents paths")
    domain = f"gui/{os.getuid()}"
    if args.fast_service:
        if args.install:
            if stat.S_IMODE(config_path.stat().st_mode) != 0o600:
                raise RuntimeError("fast service config must be owner-only 0600")
            _validate_fast_install_authority(config_path, config)
            conflicts = (
                OLD_LABEL,
                LABEL_A,
                LABEL_CAPTURE,
                LABEL_PIPELINE,
                LABEL_B,
                LABEL_WATCHDOG,
                LABEL_DAILY_0600,
                LABEL_DAILY_0700,
                LABEL_DAILY_0800,
                LABEL_DAILY_ALERT,
                LABEL_DAILY_STATUS,
            )
            if any(_is_loaded(domain, label) for label in conflicts) or any(
                _live_process_lock(pipeline_root, name)
                for name in ("process_a", "capture", "pipeline", "process_b")
            ):
                raise RuntimeError("fast service install refuses while an incumbent Calls process is active")
            if _conflicting_plist_exists(conflicts):
                raise RuntimeError("fast service install refuses while incumbent plists exist")
    elif args.process_a_only:
        paths = {LABEL_A: paths[LABEL_A]}
        if args.install and (_is_loaded(domain, LABEL_B) or _is_loaded(domain, OLD_LABEL)
                             or _live_process_lock(pipeline_root, "process_b")):
            raise RuntimeError("process-a-only install refuses while local Process B or legacy service is active")
        if args.install and _conflicting_plist_exists((LABEL_B, OLD_LABEL)):
            raise RuntimeError("process-a-only install refuses while conflicting LaunchAgent plist exists")
    elif args.process_b_only:
        paths = {LABEL_B: paths[LABEL_B]}
        if args.install and (_is_loaded(domain, LABEL_A) or _is_loaded(domain, OLD_LABEL)
                             or _live_process_lock(pipeline_root, "process_a")):
            raise RuntimeError("process-b-only install refuses while local Process A or legacy service is active")
        if args.install and _conflicting_plist_exists((LABEL_A, OLD_LABEL)):
            raise RuntimeError("process-b-only install refuses while conflicting LaunchAgent plist exists")
    if args.install and (args.process_a_only or args.process_b_only):
        label = next(iter(paths))
        if paths[label] != _standard_plist(label):
            raise RuntimeError("single-role install must use the standard LaunchAgents path")
    loaded_before = (
        {label: _is_loaded(domain, label) for label in paths}
        if args.install
        else {}
    )
    previous_files = {
        label: path.read_bytes() if path.is_file() else None
        for label, path in paths.items()
    }
    payloads = {}
    if args.fast_service:
        payloads[LABEL_CAPTURE] = _payload(
            label=LABEL_CAPTURE,
            command="capture-worker",
            interval_seconds=None,
            calendar_minutes=(0, 15, 30, 45),
            config_path=config_path,
            env_path=env_path,
            log_dir=log_dir,
        )
        payloads[LABEL_PIPELINE] = _payload(
            label=LABEL_PIPELINE,
            command="pipeline-worker",
            interval_seconds=None,
            calendar_minutes=(7, 37),
            config_path=config_path,
            env_path=env_path,
            log_dir=log_dir,
        )
        payloads[LABEL_B] = _payload(
            label=LABEL_B,
            command="process-b-worker",
            interval_seconds=None,
            config_path=config_path,
            env_path=env_path,
            log_dir=log_dir,
        )
        payloads[LABEL_WATCHDOG] = _payload(
            label=LABEL_WATCHDOG,
            command="watchdog-worker",
            interval_seconds=None,
            calendar_minutes=(12, 27, 42, 57),
            config_path=config_path,
            env_path=env_path,
            log_dir=log_dir,
        )
        for label, command, hour, minute in (
            (LABEL_DAILY_0600, "publication-close", 6, 0),
            (LABEL_DAILY_0700, "publication-close", 7, 0),
            (LABEL_DAILY_0800, "publication-close", 8, 0),
            (LABEL_DAILY_ALERT, "publication-alert", 8, 30),
            (LABEL_DAILY_STATUS, "publication-status", 8, 50),
        ):
            payloads[label] = _payload(
                label=label,
                command=command,
                interval_seconds=None,
                calendar_schedule=({"Hour": hour, "Minute": minute},),
                config_path=config_path,
                env_path=env_path,
                log_dir=log_dir,
            )
    else:
        if LABEL_A in paths:
            payloads[LABEL_A] = _payload(
                label=LABEL_A,
                command="process-a-worker" if args.process_a_only else "process-a",
                interval_seconds=intervals[LABEL_A],
                config_path=config_path,
                env_path=env_path,
                log_dir=log_dir,
            )
        if LABEL_B in paths:
            payloads[LABEL_B] = _payload(
                label=LABEL_B,
                command="process-b-pull" if args.process_b_only else "process-b",
                interval_seconds=intervals[LABEL_B],
                config_path=config_path,
                env_path=env_path,
                log_dir=log_dir,
            )
    if args.install:
        log_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
        log_dir.chmod(0o700)
    for label, payload in payloads.items():
        _write_plist(paths[label], payload)
    result = {
        "status": "rendered",
        "labels": list(paths),
        "plist_paths": {label: str(path) for label, path in paths.items()},
        "interval_seconds": {
            label: intervals.get(label) for label in paths
        },
        "calendar_minutes": {
            label: list(payload.get("StartCalendarInterval", []))
            for label, payload in payloads.items()
        },
        "installed": False,
    }
    if args.install:
        installed_labels: list[str] = []
        # B must be loaded before scheduled A can ever kick it off.
        install_order = tuple(
            label
            for label in (
                LABEL_B,
                LABEL_DAILY_STATUS,
                LABEL_DAILY_ALERT,
                LABEL_DAILY_0800,
                LABEL_DAILY_0700,
                LABEL_DAILY_0600,
                LABEL_WATCHDOG,
                LABEL_PIPELINE,
                LABEL_CAPTURE,
                LABEL_A,
            )
            if label in paths
        )
        for label in install_order:
            plist_path = paths[label]
            try:
                _bootout_if_loaded(domain, label)
            except RuntimeError:
                rollback_errors = _restore_incumbents(
                    domain=domain,
                    paths=paths,
                    previous_files=previous_files,
                    loaded_before=loaded_before,
                )
                result.update(
                    status="failed",
                    stop_reason=f"launchctl_bootout_failed:{label}",
                    rollback_errors=rollback_errors,
                )
                print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
                return 1
            install = _bootstrap(domain, plist_path)
            if install.returncode != 0:
                rollback_errors = _restore_incumbents(
                    domain=domain,
                    paths=paths,
                    previous_files=previous_files,
                    loaded_before=loaded_before,
                )
                result.update(
                    status="failed",
                    stop_reason=f"launchctl_bootstrap_failed:{label}",
                    rollback_errors=rollback_errors,
                )
                print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
                return 1
            installed_labels.append(label)
        try:
            result["old_label_booted_out"] = _bootout_if_loaded(domain, OLD_LABEL)
        except RuntimeError:
            rollback_errors = _restore_incumbents(
                domain=domain,
                paths=paths,
                previous_files=previous_files,
                loaded_before=loaded_before,
            )
            result.update(
                status="failed",
                stop_reason="legacy_bootout_failed",
                rollback_errors=rollback_errors,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
            return 1
        result.update(status="installed", installed=True)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
