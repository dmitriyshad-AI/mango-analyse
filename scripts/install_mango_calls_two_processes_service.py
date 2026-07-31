#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
import os
import plistlib
import stat
import subprocess
from pathlib import Path
from typing import Optional, Sequence


OLD_LABEL = "com.mango.calls-two-processes"
LABEL_A = "com.mango.calls-process-a"
LABEL_B = "com.mango.calls-process-b"
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
    return {
        LABEL_A: out_dir / f"{LABEL_A}.plist",
        LABEL_B: out_dir / f"{LABEL_B}.plist",
    }


def _payload(
    *,
    label: str,
    command: str,
    interval_seconds: Optional[int],
    config_path: Path,
    env_path: Path,
    log_dir: Path,
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
    return payload


def _write_plist(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".plist.tmp")
    with temp.open("wb") as handle:
        plistlib.dump(payload, handle, sort_keys=True)
    temp.replace(path)


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
    mode = stat.S_IMODE(path.stat().st_mode)
    if path.stat().st_uid != os.getuid() or (mode != 0o600 if secret else bool(mode & 0o022)):
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
                temp = path.with_suffix(".plist.rollback.tmp")
                temp.write_bytes(previous)
                temp.replace(path)
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
    if args.process_b_interval_seconds is not None and not args.process_b_only:
        raise ValueError("process B is demand-only and must not have an interval")
    config_path = Path(args.config).expanduser().resolve()
    env_path = Path(args.env_file).expanduser().resolve()
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
    log_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
    log_dir.chmod(0o700)
    paths = _plist_paths(args)
    domain = f"gui/{os.getuid()}"
    if args.process_a_only:
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
    for label, payload in payloads.items():
        _write_plist(paths[label], payload)
    result = {
        "status": "rendered",
        "labels": list(paths),
        "plist_paths": {label: str(path) for label, path in paths.items()},
        "interval_seconds": {label: intervals[label] for label in paths},
        "installed": False,
    }
    if args.install:
        installed_labels: list[str] = []
        # B must be loaded before scheduled A can ever kick it off.
        install_order = tuple(label for label in (LABEL_B, LABEL_A) if label in paths)
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
