#!/usr/bin/env python3
"""Create a read-only stale task report for the local queue."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path


DEFAULT_ROOT = Path(__file__).resolve().parents[1]
LIMITS_DAYS = {"_inbox_codex": 7, "_running": 2}


def build_stale_report(root: Path, now: datetime | None = None) -> str:
    current = now or datetime.now()
    lines = [f"# Стейл-отчёт очереди — {current:%Y-%m-%d %H:%M}", ""]
    total = 0
    for subdir, days in LIMITS_DAYS.items():
        path = root / "tasks" / subdir
        stale: list[tuple[str, int]] = []
        if path.exists():
            threshold = current - timedelta(days=days)
            for file in sorted(path.glob("*.md")):
                mtime = datetime.fromtimestamp(file.stat().st_mtime)
                if mtime < threshold:
                    stale.append((file.name, (current - mtime).days))
        lines.append(f"## tasks/{subdir} (старше {days} дн.): {len(stale)}")
        for name, age in stale:
            lines.append(f"- `{name}` — {age} дн.")
        if not stale:
            lines.append("- нет")
        lines.append("")
        total += len(stale)
    lines.append(f"Всего stale: {total}")
    lines.append("")
    return "\n".join(lines)


def write_stale_report(root: Path, out: Path | None = None) -> Path:
    target = out or root / "docs" / "_automation_status" / "stale_tasks.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(build_stale_report(root), encoding="utf-8")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    out = args.out.resolve() if args.out else None
    target = write_stale_report(root, out)
    print(f"OK: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
