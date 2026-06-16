#!/usr/bin/env python3
"""Preflight checks for a TZ before implementation."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path


DEFAULT_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_LOCAL_OUTPUTS = ("docs/PROJECT_NOW.md", "docs/_automation_status/", "tasks/_")
FORBIDDEN_ZONE_PREFIXES = (
    "~/.codex",
    ".codex/",
    ".codex_local/",
    "stable_runtime/",
    "graphify-out/",
    "runs/",
    "transcripts/",
    "audits/_results/",
)


@dataclass
class TzHeader:
    branch: str | None
    zones: list[str]
    test_cmd: str | None
    semantic: str | None = None


@dataclass
class WorktreeEntry:
    path: str
    branch: str | None
    detached: bool = False
    locked: bool = False
    prunable: bool = False

    @property
    def ignored(self) -> bool:
        return self.detached or self.locked or self.prunable


def _run_git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-c", "core.quotepath=off", *args],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout if result.returncode == 0 else result.stderr


def parse_tz_header(text: str) -> TzHeader:
    head = text[:6000]

    def grab_plain(key: str) -> str | None:
        match = re.search(rf"^{re.escape(key)}:\s*(.+)$", head, re.M)
        return match.group(1).strip() if match else None

    def grab_bold(key: str) -> str | None:
        match = re.search(rf"\*\*{re.escape(key)}:\*\*\s*(.+?)(?=\s+\*\*[^*]+:\*\*|$)", head, re.S)
        if not match:
            return None
        return " ".join(match.group(1).strip().split())

    branch = grab_plain("Ветка") or grab_bold("Ветка")
    zones_text = grab_plain("Зоны") or grab_bold("Зоны") or ""
    test_cmd = grab_plain("Тест-команда") or grab_bold("Тест-команда")
    semantic = grab_plain("Семантический-аудит") or grab_bold("Семантический-аудит")
    zones = []
    for raw_zone in re.split(r"[,;]", zones_text):
        zone = raw_zone.strip().replace("`", "").rstrip(".").strip()
        zone = re.sub(r"^(репо-корневой|root)\s+", "", zone, flags=re.I).strip()
        if zone:
            zones.append(zone)
    if test_cmd and "`" in test_cmd:
        parts = test_cmd.split("`")
        if len(parts) >= 3:
            test_cmd = parts[1].strip()
    return TzHeader(branch=branch, zones=zones, test_cmd=test_cmd, semantic=semantic)


def parse_worktrees_porcelain(text: str) -> list[WorktreeEntry]:
    entries: list[WorktreeEntry] = []
    current: dict[str, object] | None = None
    for line in text.splitlines():
        if line.startswith("worktree "):
            if current:
                entries.append(
                    WorktreeEntry(
                        path=str(current["path"]),
                        branch=current.get("branch") if isinstance(current.get("branch"), str) else None,
                        detached=bool(current.get("detached")),
                        locked=bool(current.get("locked")),
                        prunable=bool(current.get("prunable")),
                    )
                )
            current = {"path": line[len("worktree ") :]}
        elif current is not None and line.startswith("branch "):
            current["branch"] = line[len("branch ") :]
        elif current is not None and line == "detached":
            current["detached"] = True
        elif current is not None and line.startswith("locked"):
            current["locked"] = True
        elif current is not None and line == "prunable":
            current["prunable"] = True
    if current:
        entries.append(
            WorktreeEntry(
                path=str(current["path"]),
                branch=current.get("branch") if isinstance(current.get("branch"), str) else None,
                detached=bool(current.get("detached")),
                locked=bool(current.get("locked")),
                prunable=bool(current.get("prunable")),
            )
        )
    return entries


def _dirty_paths(root: Path) -> list[str]:
    out: list[str] = []
    for line in _run_git(root, "status", "--porcelain").splitlines():
        if not line.strip():
            continue
        path = line[3:].strip().strip('"')
        if " -> " in path:
            path = path.split(" -> ", 1)[1].strip().strip('"')
        out.append(path)
    return out


def _allowed_by_zone(path: str, zones: list[str]) -> bool:
    normalized = path.replace("\\", "/")
    allowed = list(zones) + list(ALLOWED_LOCAL_OUTPUTS)
    return any(normalized == zone.rstrip("/") or normalized.startswith(zone.rstrip("/") + "/") for zone in allowed)


def _forbidden_zone(zone: str) -> bool:
    normalized = zone.strip().replace("\\", "/")
    return any(normalized == prefix.rstrip("/") or normalized.startswith(prefix.rstrip("/") + "/") for prefix in FORBIDDEN_ZONE_PREFIXES)


def collect_only_command(test_cmd: str) -> list[str]:
    tokens = shlex.split(test_cmd)
    env_prefix: list[str] = []
    while tokens and "=" in tokens[0] and not tokens[0].startswith("-"):
        env_prefix.append(tokens.pop(0))
    if "pytest" not in " ".join(tokens):
        return env_prefix + tokens
    filtered: list[str] = []
    skip_next = False
    for token in tokens:
        if skip_next:
            skip_next = False
            continue
        if token == "--maxfail":
            skip_next = True
            continue
        if token.startswith("--maxfail="):
            continue
        if token == "--collect-only":
            continue
        filtered.append(token)
    try:
        pytest_index = next(i for i, token in enumerate(filtered) if token.endswith("pytest") or token == "pytest")
    except StopIteration:
        pytest_index = len(filtered) - 1
    insert_at = pytest_index + 1
    return env_prefix + filtered[:insert_at] + ["--collect-only", "-q"] + [
        token for token in filtered[insert_at:] if token != "-q"
    ]


def _run_collect_only(root: Path, test_cmd: str) -> tuple[int, str]:
    command = collect_only_command(test_cmd)
    env = os.environ.copy()
    plain_command: list[str] = []
    for token in command:
        if "=" in token and not token.startswith("-") and not plain_command:
            key, value = token.split("=", 1)
            env[key] = value
        else:
            plain_command.append(token)
    if not plain_command:
        return 0, ""
    result = subprocess.run(plain_command, cwd=root, env=env, capture_output=True, text=True, timeout=120)
    return result.returncode, result.stdout + result.stderr


def run_preflight(root: Path, tz_path: Path, *, run_collect: bool = True) -> tuple[bool, list[str]]:
    failures: list[str] = []
    root = root.resolve()
    tz_path = tz_path.resolve()
    if not tz_path.exists():
        return False, [f"ТЗ не найден: {tz_path}"]
    try:
        rel_tz = tz_path.relative_to(root)
    except ValueError:
        return False, [f"ТЗ вне репозитория: {tz_path}"]
    if not str(rel_tz).startswith("tasks/_running/"):
        failures.append(f"ТЗ должен лежать в tasks/_running, сейчас: {rel_tz}")
    header = parse_tz_header(tz_path.read_text(encoding="utf-8", errors="ignore"))
    branch = _run_git(root, "rev-parse", "--abbrev-ref", "HEAD").strip()
    if header.branch and header.branch != branch:
        failures.append(f"ветка {branch} != заявленной в ТЗ {header.branch}")
    for zone in header.zones:
        if _forbidden_zone(zone):
            failures.append(f"зона ТЗ пересекает запретный путь: {zone}")
    dirty_outside = [path for path in _dirty_paths(root) if not _allowed_by_zone(path, header.zones)]
    if dirty_outside:
        failures.append("грязь вне зон ТЗ: " + ", ".join(sorted(dirty_outside)[:20]))
    project_now = root / "docs" / "PROJECT_NOW.md"
    if not project_now.exists():
        failures.append("docs/PROJECT_NOW.md отсутствует")
    elif datetime.now() - datetime.fromtimestamp(project_now.stat().st_mtime) > timedelta(hours=24):
        failures.append("docs/PROJECT_NOW.md старше 24 часов")

    registry = root / "docs" / "worktrees_registry.md"
    registry_text = registry.read_text(encoding="utf-8", errors="ignore") if registry.exists() else ""
    for entry in parse_worktrees_porcelain(_run_git(root, "worktree", "list", "--porcelain")):
        if entry.path == str(root) or entry.ignored:
            continue
        branch_name = (entry.branch or "").removeprefix("refs/heads/")
        if entry.path not in registry_text and (not branch_name or branch_name not in registry_text):
            failures.append(f"worktree вне реестра: {entry.path}")

    if run_collect and header.test_cmd and not failures:
        rc, output = _run_collect_only(root, header.test_cmd)
        if rc:
            failures.append("test collect-only failed: " + output[-1200:])
    return not failures, failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--tz", required=True, type=Path)
    parser.add_argument("--skip-collect-only", action="store_true")
    args = parser.parse_args(argv)
    ok, failures = run_preflight(args.root, args.tz, run_collect=not args.skip_collect_only)
    if not ok:
        print("PREFLIGHT: СТОП")
        for failure in failures:
            print(f" - {failure}")
        return 1
    print("PREFLIGHT: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
