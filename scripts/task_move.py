#!/usr/bin/env python3
"""Move a TZ task between inbox/running/done/failed with an audit stamp."""

from __future__ import annotations

import argparse
import subprocess
from datetime import datetime
from pathlib import Path


DEFAULT_ROOT = Path(__file__).resolve().parents[1]
ACTION_DIRS = {"take": "_running", "done": "_done", "fail": "_failed"}


def _branch(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=20,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _resolve_task(root: Path, task: str) -> Path:
    candidate = Path(task)
    if candidate.is_absolute():
        return candidate
    direct = root / candidate
    if direct.exists():
        return direct
    for subdir in ("_inbox_codex", "_running", "_done", "_failed"):
        path = root / "tasks" / subdir / task
        if path.exists():
            return path
    return direct


def move_task(root: Path, task: str, action: str, reason: str | None = None) -> Path:
    src = _resolve_task(root, task).resolve()
    if not src.exists():
        raise FileNotFoundError(f"СТОП: нет файла {src}")
    if src.suffix != ".md":
        raise ValueError(f"СТОП: ожидается .md ТЗ, получен {src.name}")
    dst_dir = root / "tasks" / ACTION_DIRS[action]
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if dst.exists():
        raise FileExistsError(f"СТОП: {dst} уже существует")
    stamp = f"> {action.upper()} {datetime.now():%Y-%m-%d %H:%M} | ветка {_branch(root)} | codex"
    if action == "fail":
        clean_reason = (reason or "").strip()
        if not clean_reason:
            raise ValueError("СТОП: --fail требует непустую причину")
        stamp += f" | причина: {clean_reason}"
    body = src.read_text(encoding="utf-8")
    dst.write_text(stamp + "\n\n" + body, encoding="utf-8")
    src.unlink()
    return dst


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--take", action="store_true")
    group.add_argument("--done", action="store_true")
    group.add_argument("--fail")
    args = parser.parse_args(argv)
    action = "take" if args.take else "done" if args.done else "fail"
    dst = move_task(args.root.resolve(), args.task, action, args.fail)
    print(f"OK: {dst.relative_to(args.root.resolve())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
