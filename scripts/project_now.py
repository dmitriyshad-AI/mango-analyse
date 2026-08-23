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


def _task_field(text: str, name: str) -> str:
    match = re.search(rf"^{re.escape(name)}:\s*(.*?)\s*$", text.split("\n## ", 1)[0], re.M)
    return match.group(1) if match else ""


def _task_records(root: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for state in QUEUE_DIRS:
        for path in (root / "tasks" / state).glob("*.md"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            stamp = re.search(r"^> (?:TAKE|DONE|FAIL) (\d{4}-\d{2}-\d{2} \d{2}:\d{2})", text, re.M)
            problem = _task_field(text, "Problem-ID")
            records.append({
                "path": path.name, "state": state, "stamp": stamp.group(1) if stamp else "",
                "problem": problem, "feature": _task_field(text, "Feature-ID"),
                "outcome": _task_field(text, "Исход") or ("outcome_missing" if problem else "legacy_unknown"),
                "next": _task_field(text, "Следующий шаг"), "branch": _task_field(text, "Ветка"),
            })
    return records


def _worktree_heads(root: Path) -> dict[str, tuple[str, str]]:
    result: dict[str, tuple[str, str]] = {}
    for block in _run_git(root, "worktree", "list", "--porcelain").split("\n\n"):
        fields = dict(line.split(" ", 1) for line in block.splitlines() if " " in line)
        branch = fields.get("branch", "").removeprefix("refs/heads/")
        if branch:
            result[branch] = (Path(fields.get("worktree", "unknown")).name, fields.get("HEAD", "unknown")[:12])
    return result


def _problem_lifecycle_lines(root: Path) -> list[str]:
    records, worktrees = _task_records(root), _worktree_heads(root)
    grouped: dict[str, list[dict[str, str]]] = {}
    for record in records:
        if record["problem"]:
            grouped.setdefault(record["problem"], []).append(record)
    lines = ["", "## Жизненный цикл проблем", "### Открытые Problem-ID"]
    open_count = 0
    for problem, attempts in sorted(grouped.items()):
        active = [item for item in attempts if item["state"] == "_running"]
        pending = [item for item in attempts if item["state"] == "_inbox_codex"]
        completed = sorted(
            (item for item in attempts if item["state"] in {"_done", "_failed"}),
            key=lambda item: (item["stamp"], item["path"]),
        )
        latest = completed[-1] if completed else None
        if not active and not pending and latest and latest["outcome"] == "problem_closed":
            continue
        open_count += 1
        active_text = ", ".join(
            f"{item['path']} | {item['branch'] or 'branch?'} | "
            f"{worktrees.get(item['branch'], ('worktree?', 'HEAD?'))[0]}@{worktrees.get(item['branch'], ('worktree?', 'HEAD?'))[1]}"
            for item in active
        ) or "нет"
        latest_text = f"{latest['path']} ({latest['outcome']})" if latest else "нет"
        next_step = next((item["next"] for item in [*active, *pending, *(completed[-1:] or [])] if item["next"]), "не указан")
        lines.append(f"- `{problem}`: active={active_text}; latest={latest_text}; next={next_step}")
    if not open_count:
        lines.append("- нет")
    conflicts: dict[str, list[str]] = {}
    for record in (item for item in records if item["state"] == "_running" and item["feature"]):
        conflicts.setdefault(record["feature"], []).append(record["path"])
    lines.append("### Конфликты Feature-ID")
    found = {feature: paths for feature, paths in conflicts.items() if len(paths) > 1}
    lines.extend(f"- `{feature}`: {', '.join(sorted(paths))}" for feature, paths in sorted(found.items()))
    if not found:
        lines.append("- нет")
    lines.append("")
    return lines


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
    lines.extend(_problem_lifecycle_lines(root))
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
