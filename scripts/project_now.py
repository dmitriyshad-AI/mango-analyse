#!/usr/bin/env python3
"""Generate a short local project passport for the current Mango checkout."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path


DEFAULT_ROOT = Path(__file__).resolve().parents[1]
if str(DEFAULT_ROOT) not in sys.path:
    sys.path.insert(0, str(DEFAULT_ROOT))
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


def _live_snapshot(root: Path) -> dict[str, object]:
    try:
        from scripts.skills.live_truth import build_snapshot

        return asdict(build_snapshot(repo_root=root))
    except Exception as exc:  # fail-soft: passport generation must remain usable.
        return {"status": "UNAVAILABLE", "processes": [], "error": type(exc).__name__}


def _live_lines(root: Path) -> list[str]:
    snapshot = _live_snapshot(root)
    lines = ["## Клиентские live-каналы", f"- Статус проверки клиентских каналов: `{snapshot.get('status') or 'UNKNOWN'}`"]
    processes = snapshot.get("processes")
    if not isinstance(processes, list) or not processes:
        lines.append("- Процессы клиентских каналов: не найдены; calls и Timeline здесь не проверяются")
    else:
        for raw in processes:
            if not isinstance(raw, dict):
                continue
            lines.append(
                "- "
                f"`{raw.get('kind') or 'unknown'}` PID `{raw.get('pid')}`: "
                f"loaded `{raw.get('head') or 'unverified'}` ({raw.get('head_source') or 'unknown'}), "
                f"worktree `{raw.get('worktree') or 'unknown'}` @ `{raw.get('worktree_head') or 'unknown'}`"
            )
            env = raw.get("env")
            if isinstance(env, dict) and env:
                safe_values = {"DRAFT_LOOP_EXPECTED_HEAD", "TELEGRAM_DIRECT_PATH_PILOT_CONFIG"}
                rendered = ", ".join(
                    f"{key}={value if key in safe_values or str(value).casefold() in {'0', '1', 'true', 'false'} else '[set]'}"
                    for key, value in sorted(env.items())
                )
                lines.append(f"  - Эффективные runtime-настройки: `{rendered}`")
            warnings = raw.get("warnings")
            if isinstance(warnings, list) and warnings:
                lines.append("  - Предупреждения: " + "; ".join(str(item) for item in warnings))
    if snapshot.get("error"):
        lines.append(f"- Ошибка чтения: `{snapshot['error']}`")
    return lines


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
    ]
    lines.extend(_live_lines(root))
    lines.extend(["", "## Очередь"])
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
