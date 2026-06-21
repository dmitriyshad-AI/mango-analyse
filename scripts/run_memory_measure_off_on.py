#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence


DEFAULT_SCENARIOS = Path("product_data/telegram_dynamic_test_sets/memory_rich_2026-06-21.jsonl")
DEFAULT_SNAPSHOT = Path("product_data/knowledge_base/kb_release_20260612_v6_7_staging_r4_1/kb_release_v3_snapshot.json")
REPO_RELATIVE_TIMELINE_DB = Path("product_data/customer_timeline/customer_timeline_prod_20260621/customer_timeline.sqlite")
MAIN_FOLDER_TIMELINE_DB = Path(
    "/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_timeline/"
    "customer_timeline_prod_20260621/customer_timeline.sqlite"
)
READY_ENV = "MEMORY_MEASURE_STREAMS_1_2_READY"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare or run OFF/ON M1 memory measurement pair.")
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--timeline-db", type=Path, default=_default_timeline_db())
    parser.add_argument("--out-root", type=Path, default=_default_out_root())
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--judge-prompt-version", default="v9.1")
    parser.add_argument("--execute", action="store_true", help="Actually run the two simulator commands.")
    parser.add_argument(
        "--streams-ready",
        action="store_true",
        help=f"Confirm D8/D3 streams are integrated; alternative to {READY_ENV}=1.",
    )
    args = parser.parse_args(argv)

    if args.parallel < 1:
        raise ValueError("--parallel must be >= 1")
    if args.execute and not (args.streams_ready or os.getenv(READY_ENV) in {"1", "true", "yes", "да"}):
        raise RuntimeError(
            f"Measurement launch is blocked until Streams 1-2 are integrated. "
            f"Pass --streams-ready or set {READY_ENV}=1 only after Dmitry's go."
        )

    commands = build_commands(
        scenarios=args.scenarios,
        snapshot=args.snapshot,
        timeline_db=args.timeline_db,
        out_root=args.out_root,
        parallel=args.parallel,
        judge_prompt_version=args.judge_prompt_version,
    )
    args.out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_root / "memory_measure_off_on_manifest.json"
    commands_path = args.out_root / "run_off_on_commands.sh"
    manifest = {
        "schema_version": "memory_measure_off_on_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "execute": bool(args.execute),
        "precondition": "Run only after Streams 1-2 are integrated and Dmitry gives go.",
        "scenarios": str(args.scenarios),
        "snapshot": str(args.snapshot),
        "timeline_db": str(args.timeline_db),
        "parallel": args.parallel,
        "judge_prompt_version": args.judge_prompt_version,
        "commands": commands,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    commands_path.write_text(render_shell(commands), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "commands": str(commands_path), "execute": bool(args.execute)}, ensure_ascii=False))

    if not args.execute:
        return 0
    for arm in ("off", "on"):
        command = commands[arm]
        env = os.environ.copy()
        env.update(command["env"])
        subprocess.run(command["argv"], check=True, env=env)
    return 0


def build_commands(
    *,
    scenarios: Path,
    snapshot: Path,
    timeline_db: Path,
    out_root: Path,
    parallel: int,
    judge_prompt_version: str,
) -> Mapping[str, Mapping[str, object]]:
    base = [
        "python3",
        "scripts/run_telegram_dynamic_client_sim.py",
        "--scenarios",
        str(scenarios),
        "--snapshot",
        str(snapshot),
        "--brand",
        "all",
        "--parallel",
        str(parallel),
        "--judge-prompt-version",
        str(judge_prompt_version),
    ]
    common_env = {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": "src",
    }
    return {
        "off": {
            "label": "memory_off",
            "out_dir": str(out_root / "memory_off"),
            "env": {**common_env, "TELEGRAM_BOT_SAFE_CRM_CONTEXT": "0"},
            "argv": [*base, "--out-dir", str(out_root / "memory_off")],
        },
        "on": {
            "label": "memory_on",
            "out_dir": str(out_root / "memory_on"),
            "env": {
                **common_env,
                "TELEGRAM_BOT_SAFE_CRM_CONTEXT": "1",
                "TELEGRAM_BOT_SAFE_CRM_CONTEXT_DB": str(timeline_db),
            },
            "argv": [*base, "--out-dir", str(out_root / "memory_on")],
        },
    }


def render_shell(commands: Mapping[str, Mapping[str, object]]) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "# Do not run before Streams 1-2 are integrated and Dmitry gives go.",
    ]
    for arm in ("off", "on"):
        command = commands[arm]
        env = command["env"]
        argv = command["argv"]
        assert isinstance(env, Mapping)
        assert isinstance(argv, Sequence)
        env_prefix = " ".join(f"{key}={shlex.quote(str(value))}" for key, value in sorted(env.items()))
        lines.append("")
        lines.append(f"# {command['label']}")
        lines.append(f"{env_prefix} {shlex.join([str(value) for value in argv])}")
    return "\n".join(lines) + "\n"


def _default_timeline_db() -> Path:
    return REPO_RELATIVE_TIMELINE_DB if REPO_RELATIVE_TIMELINE_DB.exists() else MAIN_FOLDER_TIMELINE_DB


def _default_out_root() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Path("audits/_inbox") / f"memory_measure_off_on_{stamp}"


if __name__ == "__main__":
    raise SystemExit(main())
