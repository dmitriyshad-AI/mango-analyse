#!/usr/bin/env python3
"""Read-only sentinel for the actual live bot / loop runtime."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import scripts.wappi_draft_loop_ops as wappi_ops


PROCESS_MARKERS = (
    "run_telegram_public_pilot_bots.py",
    "run_amo_wappi_draft_loop.py",
    "m1_watcher.py",
)
DEFAULT_DAILY_DIR = Path.home() / "Claude Projects" / "Foton" / "_daily"


@dataclass(frozen=True)
class LiveProcessRow:
    kind: str
    pid: int
    worktree: str
    head: str
    command: str
    env: dict[str, str]
    db_paths: list[str]
    warnings: list[str]


@dataclass(frozen=True)
class LiveTruthSnapshot:
    schema_version: str
    generated_at: str
    repo: dict[str, str]
    processes: list[LiveProcessRow]
    status: str


def _git_value(root: Path, *args: str) -> str:
    completed = subprocess.run(["git", "-C", str(root), *args], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _extract_cwd_from_command(command: str, fallback: Path) -> Path:
    for token in str(command or "").split():
        if token.endswith(".py") and "/" in token:
            path = Path(token)
            if path.is_absolute():
                parts = path.parts
                if "scripts" in parts:
                    idx = parts.index("scripts")
                    return Path(*parts[:idx])
    return fallback


def _process_cwd(pid: int) -> Path | None:
    completed = subprocess.run(
        ["lsof", "-a", "-p", str(pid), "-d", "cwd", "-Fn"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if completed.returncode != 0:
        return None
    for line in completed.stdout.splitlines():
        if line.startswith("n") and len(line) > 1:
            return Path(line[1:])
    return None


def _process_marker(command: str) -> str:
    try:
        tokens = shlex.split(str(command or ""))
    except ValueError:
        tokens = str(command or "").split()
    token_names = {Path(token).name for token in tokens}
    return next((marker for marker in PROCESS_MARKERS if marker in token_names), "")


def _lsof_db_paths(pid: int) -> list[str]:
    completed = subprocess.run(["lsof", "-p", str(pid)], text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
    if completed.returncode != 0:
        return []
    paths: set[str] = set()
    for line in completed.stdout.splitlines():
        if any(marker in line for marker in (".sqlite", ".sqlite3", ".db", ".sqlite-wal", ".sqlite-shm")):
            parts = line.split()
            if parts:
                paths.add(parts[-1])
    return sorted(paths)


def build_snapshot(
    *,
    repo_root: Path,
    processes: Sequence[wappi_ops.ProcessInfo] | None = None,
    env_reader=wappi_ops.read_process_environ,
    lsof_reader=_lsof_db_paths,
    cwd_reader=_process_cwd,
    expected_heads: Mapping[str, str] | None = None,
) -> LiveTruthSnapshot:
    process_rows: list[LiveProcessRow] = []
    expected_heads = expected_heads or {}
    for process in processes if processes is not None else wappi_ops.list_processes():
        marker = _process_marker(process.command)
        if not marker:
            continue
        env, _source = env_reader(process.pid)
        process_cwd = cwd_reader(process.pid)
        worktree = process_cwd or _extract_cwd_from_command(process.command, repo_root)
        head = _git_value(worktree, "rev-parse", "--short", "HEAD")
        warnings: list[str] = []
        if process_cwd is None:
            warnings.append(f"cwd_unavailable command_path_fallback={worktree}")
        expected = expected_heads.get(marker) or expected_heads.get(str(worktree))
        if expected and not head:
            warnings.append(f"head_unavailable expected={expected}")
        elif expected and not head.startswith(expected):
            warnings.append(f"head_drift expected={expected} actual={head}")
        process_rows.append(
            LiveProcessRow(
                kind=marker,
                pid=process.pid,
                worktree=str(worktree),
                head=head,
                command=process.command,
                env=wappi_ops.filter_runtime_env(env),
                db_paths=lsof_reader(process.pid),
                warnings=warnings,
            )
        )
    status = "PASS" if not any(row.warnings for row in process_rows) else "WARN"
    return LiveTruthSnapshot(
        schema_version="live_truth_sentinel_v1_2026_07_08",
        generated_at=datetime.now().isoformat(timespec="seconds"),
        repo={"path": str(repo_root), "head": _git_value(repo_root, "rev-parse", "--short", "HEAD")},
        processes=process_rows,
        status=status,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only live runtime truth sentinel.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--daily-dir", type=Path, default=DEFAULT_DAILY_DIR)
    parser.add_argument("--expect-head", action="append", default=[], help="marker_or_worktree=sha")
    parser.add_argument("--no-write", action="store_true", help="Do not write Foton/_daily snapshot.")
    args = parser.parse_args(argv)

    expected = dict(item.split("=", 1) for item in args.expect_head if "=" in item)
    snapshot = build_snapshot(repo_root=args.root.resolve(), expected_heads=expected)
    text = json.dumps(asdict(snapshot), ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if not args.no_write:
        args.daily_dir.expanduser().mkdir(parents=True, exist_ok=True)
        out = args.daily_dir.expanduser() / f"live_truth_{datetime.now():%Y%m%d_%H%M%S}.json"
        out.write_text(text + "\n", encoding="utf-8")
    return 0 if snapshot.status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
