#!/usr/bin/env python3
"""Generate a short local project passport for the current Mango checkout."""

from __future__ import annotations

import argparse
import re
import subprocess
from datetime import datetime
from pathlib import Path


DEFAULT_ROOT = Path(__file__).resolve().parents[1]
QUEUE_DIRS = ("_running", "_inbox_codex", "_done", "_failed")


def _run_git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-c", "core.quotepath=off", *args],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=20,
    )
    if result.returncode:
        return result.stderr.strip()
    return result.stdout.strip()


def _queue_files(root: Path, subdir: str, limit: int = 12) -> list[str]:
    path = root / "tasks" / subdir
    if not path.exists():
        return []
    files = sorted(path.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [file.name for file in files[:limit]]


def _recent_audits(root: Path, limit: int = 10) -> list[str]:
    path = root / "audits" / "_inbox"
    if not path.exists():
        return []
    entries = [p for p in path.iterdir() if p.is_dir()]
    entries.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [entry.name for entry in entries[:limit]]


def _extract_blockers(root: Path) -> list[str]:
    path = root / "docs" / "BLOCKERS.yaml"
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="ignore")
    if re.search(r"^\s*blockers:\s*\[\s*\]\s*$", text, re.M):
        return []
    blockers: list[str] = []
    current: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            if current:
                blockers.append(_format_blocker(current))
            current = {}
            stripped = stripped[2:].strip()
            if ":" in stripped:
                key, value = stripped.split(":", 1)
                current[key.strip()] = value.strip().strip('"')
        elif ":" in stripped and current is not None:
            key, value = stripped.split(":", 1)
            current[key.strip()] = value.strip().strip('"')
    if current:
        blockers.append(_format_blocker(current))
    return [item for item in blockers if item.strip()]


def _format_blocker(item: dict[str, str]) -> str:
    what = item.get("what") or item.get("title") or "без описания"
    owner = item.get("owner")
    since = item.get("since")
    tail = " ".join(part for part in (f"owner={owner}" if owner else "", f"since={since}" if since else "") if part)
    return f"{what} ({tail})" if tail else what


def _active_kb_mentions(root: Path) -> list[str]:
    mentions: set[str] = set()
    for rel in (
        "src/mango_mvp/channels/subscription_llm_parts/config.py",
        "src/mango_mvp/channels/subscription_llm_parts/provider.py",
        "src/mango_mvp/integrations/draft_loop.py",
        "scripts/build_mango_clean_bundle.py",
        "scripts/run_telegram_dynamic_client_sim.py",
        "scripts/run_telegram_public_pilot_bots.py",
    ):
        path = root / rel
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in re.findall(r"kb_release_[A-Za-z0-9_./-]+", text):
            mentions.add(match.rstrip('",)'))
    return sorted(mentions)


def build_project_now(root: Path) -> str:
    now = datetime.now().isoformat(timespec="seconds")
    branch = _run_git(root, "rev-parse", "--abbrev-ref", "HEAD")
    head = _run_git(root, "rev-parse", "--short", "HEAD")
    status = _run_git(root, "status", "--short")
    dirty_count = len([line for line in status.splitlines() if line.strip()])
    lines = [
        "# PROJECT_NOW",
        "",
        f"Сгенерирован: {now}",
        f"Ветка: `{branch}`",
        f"HEAD: `{head}`",
        f"Грязных файлов: {dirty_count}",
        "",
        "## Очередь",
    ]
    for subdir in QUEUE_DIRS:
        items = _queue_files(root, subdir)
        lines.append(f"### tasks/{subdir}: {len(items)} показано")
        lines.extend(f"- `{item}`" for item in items)
        if not items:
            lines.append("- нет")
        lines.append("")
    lines.append("## Блокеры")
    blockers = _extract_blockers(root)
    lines.extend(f"- {item}" for item in blockers) if blockers else lines.append("- нет")
    lines.extend(["", "## Свежие audit packs"])
    audits = _recent_audits(root)
    lines.extend(f"- `{item}`" for item in audits) if audits else lines.append("- нет")
    lines.extend(["", "## KB-релизы, найденные в runtime-коде"])
    kb_mentions = _active_kb_mentions(root)
    lines.extend(f"- `{item}`" for item in kb_mentions[-12:]) if kb_mentions else lines.append("- не найдено")
    lines.extend(["", "## Git status", "```", status or "clean", "```", ""])
    return "\n".join(lines)


def write_project_now(root: Path, out: Path | None = None) -> Path:
    target = out or root / "docs" / "PROJECT_NOW.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(build_project_now(root), encoding="utf-8")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    out = args.out.resolve() if args.out else None
    target = write_project_now(root, out)
    print(f"OK: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
