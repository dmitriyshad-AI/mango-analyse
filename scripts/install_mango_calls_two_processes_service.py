#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import plistlib
import subprocess
from pathlib import Path
from typing import Optional, Sequence


LABEL = "com.mango.calls-two-processes"
ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Install one launchd trigger for Mango calls process A -> B.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--env-file", required=True)
    parser.add_argument("--interval-seconds", type=int, default=900)
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--out")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    env_path = Path(args.env_file).expanduser().resolve()
    if not config_path.is_file() or not env_path.is_file():
        raise FileNotFoundError("config and env file must exist")
    if args.interval_seconds < 300:
        raise ValueError("interval must be at least 300 seconds")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    pipeline_root = Path(str(config["pipeline_root"])).expanduser().resolve()
    log_dir = pipeline_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    plist_path = (
        Path(args.out).expanduser().resolve()
        if args.out
        else Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"
    )
    payload = {
        "Label": LABEL,
        "ProgramArguments": [
            "/bin/zsh",
            str(ROOT / "scripts" / "run_mango_calls_cycle.sh"),
            str(config_path),
            str(env_path),
        ],
        "WorkingDirectory": str(ROOT),
        "RunAtLoad": False,
        "StartInterval": int(args.interval_seconds),
        "ProcessType": "Background",
        "ThrottleInterval": 60,
        "StandardOutPath": str(log_dir / "launchd.stdout.log"),
        "StandardErrorPath": str(log_dir / "launchd.stderr.log"),
    }
    plist_path.parent.mkdir(parents=True, exist_ok=True)
    temp = plist_path.with_suffix(".plist.tmp")
    with temp.open("wb") as handle:
        plistlib.dump(payload, handle, sort_keys=True)
    temp.replace(plist_path)
    result = {
        "status": "rendered",
        "label": LABEL,
        "plist_path": str(plist_path),
        "interval_seconds": args.interval_seconds,
        "installed": False,
    }
    if args.install:
        domain = f"gui/{__import__('os').getuid()}"
        probe = subprocess.run(
            ["launchctl", "print", f"{domain}/{LABEL}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if probe.returncode == 0:
            subprocess.run(
                ["launchctl", "bootout", domain, str(plist_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        install = subprocess.run(
            ["launchctl", "bootstrap", domain, str(plist_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if install.returncode != 0:
            result.update(status="failed", stop_reason="launchctl_bootstrap_failed")
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
            return 1
        result.update(status="installed", installed=True)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
