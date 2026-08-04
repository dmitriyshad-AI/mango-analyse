#!/usr/bin/env python3
"""Read-only sentinel for the actual live bot / loop runtime."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import scripts.wappi_draft_loop_ops as wappi_ops


FORBIDDEN_PROCESS_MARKERS = ("run_telegram_public_pilot_bots.py",)
PROCESS_MARKERS = (
    "run_amo_wappi_draft_loop.py",
    "m1_watcher.py",
    *FORBIDDEN_PROCESS_MARKERS,
)
DEFAULT_DAILY_DIR = Path.home() / "Claude Projects" / "Foton" / "_daily"
DEFAULT_WAPPI_RUNTIME_DIR = Path.home() / ".mango_local" / "draft_loop"
DEFAULT_WAPPI_MANIFEST = DEFAULT_WAPPI_RUNTIME_DIR / "phase1b_startup_manifest.json"
DEFAULT_WAPPI_HEARTBEAT = DEFAULT_WAPPI_RUNTIME_DIR / "heartbeat.json"
WAPPI_PROCESS_MARKER = "run_amo_wappi_draft_loop.py"
START_TIME_TOLERANCE_SECONDS = 10
WAPPI_LAUNCHD_LABEL = "com.mango.wappi-draft-loop"


@dataclass(frozen=True)
class LiveProcessRow:
    kind: str
    pid: int
    worktree: str
    head: str
    worktree_head: str
    head_source: str
    process_started_at: str
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


def _process_started_at(pid: int) -> datetime | None:
    completed = subprocess.run(
        ["ps", "-p", str(pid), "-o", "lstart="],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    text = completed.stdout.strip()
    if completed.returncode != 0 or not text:
        return None
    try:
        return datetime.strptime(text, "%a %b %d %H:%M:%S %Y").astimezone(timezone.utc)
    except ValueError:
        return None


def _wappi_launchd_pid() -> int | None:
    completed = subprocess.run(
        ["launchctl", "print", f"gui/{os.getuid()}/{WAPPI_LAUNCHD_LABEL}"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    match = re.search(r"^\s*pid\s*=\s*(\d+)\s*$", completed.stdout, flags=re.MULTILINE)
    return int(match.group(1)) if completed.returncode == 0 and match else None


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _parse_timestamp(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _path_matches(left: Any, right: Path) -> bool:
    text = str(left or "").strip()
    return bool(text) and Path(text).expanduser().resolve() == right.expanduser().resolve()


def _wappi_loaded_head(
    *,
    process_started_at: datetime | None,
    worktree: Path,
    manifest_path: Path,
    heartbeat_path: Path,
) -> tuple[str, str, list[str]]:
    manifest = _read_json_object(manifest_path)
    heartbeat = _read_json_object(heartbeat_path)
    warnings: list[str] = []

    manifest_head = str(manifest.get("head") or "").strip()
    manifest_started = _parse_timestamp(manifest.get("started_at"))
    manifest_ready = str(manifest.get("status") or "").strip().casefold() == "ready"
    manifest_valid = bool(manifest_ready and manifest_head and manifest_started and process_started_at)
    if manifest and not manifest_ready:
        warnings.append(f"startup_manifest_not_ready status={manifest.get('status', '')}")
    if manifest_valid:
        delta = abs((manifest_started - process_started_at).total_seconds())
        manifest_valid = delta <= START_TIME_TOLERANCE_SECONDS
        if not manifest_valid:
            warnings.append(f"startup_manifest_pid_mismatch delta_seconds={delta:.0f}")
    elif manifest:
        warnings.append("startup_manifest_unverified")
    if manifest and not _path_matches(manifest.get("cwd"), worktree):
        manifest_valid = False
        warnings.append(f"startup_manifest_cwd_mismatch manifest={manifest.get('cwd', '')} process={worktree}")

    heartbeat_warnings: list[str] = []
    heartbeat_head = str(heartbeat.get("git_sha") or "").strip()
    heartbeat_at = _parse_timestamp(heartbeat.get("last_cycle_at"))
    heartbeat_valid = bool(heartbeat_head and heartbeat_at and process_started_at)
    if heartbeat_valid and heartbeat_at < process_started_at:
        heartbeat_valid = False
        heartbeat_warnings.append("heartbeat_predates_process")
    if heartbeat and not _path_matches(heartbeat.get("code_root"), worktree):
        heartbeat_valid = False
        heartbeat_warnings.append(f"heartbeat_cwd_mismatch heartbeat={heartbeat.get('code_root', '')} process={worktree}")

    if manifest_valid and heartbeat_valid and not (
        manifest_head.startswith(heartbeat_head) or heartbeat_head.startswith(manifest_head)
    ):
        warnings.append(f"runtime_head_disagreement manifest={manifest_head} heartbeat={heartbeat_head}")
        return "", "unverified", warnings
    if manifest_valid:
        return manifest_head, "startup_manifest", warnings
    warnings.extend(heartbeat_warnings)
    if heartbeat_valid:
        return heartbeat_head, "heartbeat", warnings
    warnings.append("loaded_head_unverified")
    return "", "unverified", warnings


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
    process_started_reader=_process_started_at,
    wappi_pid_reader=_wappi_launchd_pid,
    expected_heads: Mapping[str, str] | None = None,
    wappi_manifest_path: Path = DEFAULT_WAPPI_MANIFEST,
    wappi_heartbeat_path: Path = DEFAULT_WAPPI_HEARTBEAT,
) -> LiveTruthSnapshot:
    process_rows: list[LiveProcessRow] = []
    expected_heads = expected_heads or {}
    selected_processes = [
        (process, marker)
        for process in (processes if processes is not None else wappi_ops.list_processes())
        if (marker := _process_marker(process.command))
    ]
    wappi_process_count = sum(marker == WAPPI_PROCESS_MARKER for _process, marker in selected_processes)
    for process, marker in selected_processes:
        env, _source = env_reader(process.pid)
        process_cwd = cwd_reader(process.pid)
        worktree = process_cwd or _extract_cwd_from_command(process.command, repo_root)
        worktree_head = _git_value(worktree, "rev-parse", "HEAD")
        process_started = process_started_reader(process.pid)
        warnings: list[str] = []
        if marker in FORBIDDEN_PROCESS_MARKERS:
            warnings.append(f"forbidden_process marker={marker}")
        if process_cwd is None:
            warnings.append(f"cwd_unavailable command_path_fallback={worktree}")
        head = worktree_head
        head_source = "worktree_unverified"
        if marker == WAPPI_PROCESS_MARKER:
            launchd_pid = wappi_pid_reader()
            if launchd_pid is None:
                warnings.append("wappi_launchd_pid_unavailable")
            elif launchd_pid != process.pid:
                warnings.append(f"wappi_launchd_pid_mismatch launchd={launchd_pid} process={process.pid}")
            if wappi_process_count != 1:
                warnings.append(f"wappi_process_count expected=1 actual={wappi_process_count}")
            head, head_source, runtime_warnings = _wappi_loaded_head(
                process_started_at=process_started,
                worktree=worktree,
                manifest_path=wappi_manifest_path,
                heartbeat_path=wappi_heartbeat_path,
            )
            warnings.extend(runtime_warnings)
        if not worktree_head:
            warnings.append("worktree_head_unavailable")
        expected = expected_heads.get(marker) or expected_heads.get(str(worktree)) or worktree_head
        if expected and not head:
            warnings.append(f"head_unavailable expected={expected}")
        elif expected and not (head.startswith(expected) or expected.startswith(head)):
            warnings.append(f"head_drift expected={expected} actual={head}")
        process_rows.append(
            LiveProcessRow(
                kind=marker,
                pid=process.pid,
                worktree=str(worktree),
                head=head,
                worktree_head=worktree_head,
                head_source=head_source,
                process_started_at=process_started.isoformat() if process_started else "",
                command=process.command,
                env=wappi_ops.filter_runtime_env(env),
                db_paths=lsof_reader(process.pid),
                warnings=warnings,
            )
        )
    status = "NO_PROCESS" if not process_rows else "PASS" if not any(row.warnings for row in process_rows) else "WARN"
    return LiveTruthSnapshot(
        schema_version="live_truth_sentinel_v2_2026_07_19",
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
