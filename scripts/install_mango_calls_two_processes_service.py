#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import plistlib
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
    interval_seconds: int,
    config_path: Path,
    env_path: Path,
    log_dir: Path,
) -> dict[str, object]:
    return {
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
        "StartInterval": interval_seconds,
        "ProcessType": "Background",
        "ThrottleInterval": 60,
        "StandardOutPath": str(log_dir / f"{command}.stdout.log"),
        "StandardErrorPath": str(log_dir / f"{command}.stderr.log"),
    }


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


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    env_path = Path(args.env_file).expanduser().resolve()
    if not config_path.is_file() or not env_path.is_file():
        raise FileNotFoundError("config and env file must exist")
    fallback_interval = _interval(args.interval_seconds, DEFAULT_INTERVAL_SECONDS)
    intervals = {
        LABEL_A: _interval(args.process_a_interval_seconds, fallback_interval),
        LABEL_B: _interval(args.process_b_interval_seconds, fallback_interval),
    }
    config = json.loads(config_path.read_text(encoding="utf-8"))
    pipeline_root = Path(str(config["pipeline_root"])).expanduser().resolve()
    log_dir = pipeline_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    paths = _plist_paths(args)
    payloads = {
        LABEL_A: _payload(
            label=LABEL_A,
            command="process-a",
            interval_seconds=intervals[LABEL_A],
            config_path=config_path,
            env_path=env_path,
            log_dir=log_dir,
        ),
        LABEL_B: _payload(
            label=LABEL_B,
            command="process-b",
            interval_seconds=intervals[LABEL_B],
            config_path=config_path,
            env_path=env_path,
            log_dir=log_dir,
        ),
    }
    for label, payload in payloads.items():
        _write_plist(paths[label], payload)
    result = {
        "status": "rendered",
        "labels": [LABEL_A, LABEL_B],
        "plist_paths": {label: str(path) for label, path in paths.items()},
        "interval_seconds": intervals,
        "installed": False,
    }
    if args.install:
        domain = f"gui/{os.getuid()}"
        installed_labels: list[str] = []
        for label, plist_path in paths.items():
            _bootout_if_loaded(domain, label)
            install = subprocess.run(
                ["launchctl", "bootstrap", domain, str(plist_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if install.returncode != 0:
                for installed_label in installed_labels:
                    _bootout_if_loaded(domain, installed_label)
                result.update(status="failed", stop_reason=f"launchctl_bootstrap_failed:{label}")
                print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
                return 1
            installed_labels.append(label)
        try:
            result["old_label_booted_out"] = _bootout_if_loaded(domain, OLD_LABEL)
        except RuntimeError:
            for installed_label in installed_labels:
                _bootout_if_loaded(domain, installed_label)
            result.update(status="failed", stop_reason="legacy_bootout_failed")
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
            return 1
        result.update(status="installed", installed=True)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
