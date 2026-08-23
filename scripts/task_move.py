#!/usr/bin/env python3
"""Move a TZ task between inbox/running/done/failed with an audit stamp."""

from __future__ import annotations

import argparse
import re
import subprocess
from datetime import datetime
from pathlib import Path


DEFAULT_ROOT = Path(__file__).resolve().parents[1]
ACTION_DIRS = {"take": "_running", "done": "_done", "fail": "_failed"}
OUTCOMES = {"attempt_complete", "problem_closed", "blocked", "superseded"}


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


def _field(body: str, name: str) -> str:
    match = re.search(rf"^{re.escape(name)}:\s*(.*?)\s*$", body.split("\n## ", 1)[0], re.M)
    return match.group(1) if match else ""


def _set_outcome(body: str, outcome: str) -> str:
    header, marker, tail = body.partition("\n## ")
    lines = [line for line in header.splitlines() if not re.match(r"^Исход:\s*", line)]
    anchor = next((index + 1 for index, line in enumerate(lines) if line.startswith("Problem-ID:")), 0)
    lines.insert(anchor, f"Исход: {outcome}")
    rendered = "\n".join(lines) + (marker + tail if marker else "")
    return rendered + ("\n" if body.endswith("\n") and not rendered.endswith("\n") else "")


def move_task(
    root: Path, task: str, action: str, reason: str | None = None, *, outcome: str | None = None,
) -> Path:
    src = _resolve_task(root, task).resolve()
    if not src.exists():
        raise FileNotFoundError(f"СТОП: нет файла {src}")
    if src.suffix != ".md":
        raise ValueError(f"СТОП: ожидается .md ТЗ, получен {src.name}")
    tasks_root = (root / "tasks").resolve()
    try:
        src.relative_to(tasks_root)
        source_is_internal = True
    except ValueError:
        source_is_internal = False
    if action != "take" and not source_is_internal:
        raise ValueError("СТОП: --done/--fail разрешены только для ТЗ внутри tasks/")
    dst_dir = root / "tasks" / ACTION_DIRS[action]
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if dst.exists():
        raise FileExistsError(f"СТОП: {dst} уже существует")
    body = src.read_text(encoding="utf-8")
    problem_id = _field(body, "Problem-ID")
    if action == "take" and outcome:
        raise ValueError("СТОП: --outcome указывается при завершении попытки")
    if action != "take":
        if problem_id:
            if outcome not in OUTCOMES:
                raise ValueError("СТОП: ТЗ с Problem-ID требует --outcome")
            if (action == "fail" and outcome != "blocked") or (action == "done" and outcome == "blocked"):
                raise ValueError("СТОП: blocked соответствует --fail, остальные исходы — --done")
        else:
            outcome = "legacy_unknown"
        if outcome == "problem_closed" and not _field(body, "Closure-evidence"):
            raise ValueError("СТОП: problem_closed требует непустой Closure-evidence")
        if outcome == "superseded" and not _field(body, "Следующий шаг"):
            raise ValueError("СТОП: superseded требует поле Следующий шаг")
        body = _set_outcome(body, outcome)
    stamp = f"> {action.upper()} {datetime.now():%Y-%m-%d %H:%M} | ветка {_branch(root)} | codex"
    if action == "fail":
        clean_reason = (reason or "").strip()
        if not clean_reason:
            raise ValueError("СТОП: --fail требует непустую причину")
        stamp += f" | причина: {clean_reason}"
    dst.write_text(stamp + "\n\n" + body, encoding="utf-8")
    if source_is_internal:
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
    parser.add_argument("--outcome", choices=sorted(OUTCOMES))
    args = parser.parse_args(argv)
    action = "take" if args.take else "done" if args.done else "fail"
    dst = move_task(args.root.resolve(), args.task, action, args.fail, outcome=args.outcome)
    print(f"OK: {dst.relative_to(args.root.resolve())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
