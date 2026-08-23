#!/usr/bin/env python3
"""Preflight checks for a TZ before implementation."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
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
SAFE_TEST_ENV_KEYS = {"PYTHONDONTWRITEBYTECODE", "PYTHONPATH", "PYTEST_DISABLE_PLUGIN_AUTOLOAD"}
SAFE_PYTEST_FLAGS = frozenset(
    "-q --quiet --collect-only -v -vv -s -x --exitfirst --disable-warnings --strict-markers --strict-config".split()
)
SAFE_PYTEST_OPTION_PREFIXES = ("--tb=", "--color=", "--maxfail=")
INVENTORY_COVERAGE = frozenset({"graphify", "worktrees", "raw_rg", "git_refs", "tasks", "audits", "decisions"})
NON_CODE_ZONES = ("docs/", "tasks/", "audits/", "product_data/", "D1_audit_backlog/")


@dataclass
class TzHeader:
    branch: str | None
    zones: list[str]
    test_cmd: str | None
    semantic: str | None = None
    feature_id: str | None = None
    problem_id: str | None = None
    change: str | None = None
    symbols: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()


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
    split = lambda value: tuple(item.strip() for item in re.split(r"[,;]", value or "") if item.strip())
    return TzHeader(
        branch, zones, test_cmd, semantic,
        grab_plain("Feature-ID") or grab_bold("Feature-ID"),
        grab_plain("Problem-ID") or grab_bold("Problem-ID"), grab_plain("Изменение") or grab_bold("Изменение"),
        split(grab_plain("Ключевые-символы") or grab_bold("Ключевые-символы")),
        split(grab_plain("Ключевые-слова") or grab_bold("Ключевые-слова")),
    )


def is_code_task(header: TzHeader, text: str = "") -> bool:
    def non_code(zone: str) -> bool:
        normalized = zone.replace("\\", "/").removeprefix("./")
        if Path(normalized).suffix.casefold() in {".py", ".sh", ".js", ".ts", ".tsx", ".go", ".rs", ".java"}:
            return False
        return normalized in {".gitignore", "README.md", "ARCHITECTURE.md"} or any(
            normalized == prefix.rstrip("/") or normalized.startswith(prefix) for prefix in NON_CODE_ZONES
        )
    body = re.sub(r"^(?:Зоны|Тест-команда|Ключевые-(?:символы|слова)):\s*.*$", "", text, flags=re.M)
    code_paths = re.findall(r"(?<![\w/])((?:[\w.-]+/)+[\w.-]+\.(?:py|sh|js|ts|tsx|go|rs|java))\b", body)
    return any(not non_code(path) for path in code_paths) or not header.zones or not all(non_code(zone) for zone in header.zones)


def required_roles(header: TzHeader, text: str) -> list[dict[str, str]]:
    haystack = " ".join(filter(None, [text, header.feature_id, header.problem_id, header.change, *header.symbols, *header.keywords, *header.zones])).casefold()
    roles: list[dict[str, str]] = []
    def add(role: str, model: str, reason: str) -> None:
        if not any(item["role"] == role for item in roles):
            roles.append({"role": role, "model": model, "reasoning": "xhigh", "reason": reason})
    if is_code_task(header, text):
        add("claude-code", "claude-opus", "independent code context review")
        add("architect-auditor", "gpt-5.5", "architecture and reuse owner")
        add("breaker", "gpt-5.5", "adversarial verification")
    if any(marker in haystack for marker in ("рефактор", "refactor", "удален", "удалить", "cleanup", "уборк", "дубл")):
        add("cleaner", "gpt-5.5", "cleanup and duplication scope")
    if (header.semantic or "").casefold() in {"да", "yes", "true"} or any(
        marker in haystack for marker in ("knowledge_base", "база знаний", "клиент", "crm", "amo", "tallanto", "wappi", "telegram", "email")
    ):
        add("business-auditor", "gpt-5.5", "business semantic review")
    return roles


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
    for line in _run_git(root, "status", "--porcelain", "--untracked-files=all").splitlines():
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
    if not tokens:
        raise ValueError("empty test command")
    if "/" in tokens[0] or "\\" in tokens[0]:
        raise ValueError("Тест-команда preflight не принимает внешний путь к исполняемому файлу")
    executable = tokens[0]
    direct_pytest = executable in {"pytest", "py.test"}
    module_pytest = bool(
        re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", executable)
        and len(tokens) >= 3
        and tokens[1:3] == ["-m", "pytest"]
    )
    if not (direct_pytest or module_pytest):
        raise ValueError("Тест-команда preflight должна запускать только pytest")
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


def _validate_pytest_targets(root: Path, command: list[str]) -> str | None:
    plain = list(command)
    while plain and "=" in plain[0] and not plain[0].startswith("-"):
        plain.pop(0)
    executable = Path(plain[0]).name
    args = plain[3:] if re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", executable) else plain[1:]
    skip_value = False
    for token in args:
        if skip_value:
            skip_value = False
            continue
        if token in {"-k", "-m"}:
            skip_value = True
            continue
        if token.startswith("-") and not (
            token in SAFE_PYTEST_FLAGS or token.startswith(SAFE_PYTEST_OPTION_PREFIXES)
        ):
            return f"unsafe pytest option: {token}"
        if token.startswith("-"):
            continue
        target_text = token.split("::", 1)[0]
        target = (root / target_text).resolve()
        tests_root = (root / "tests").resolve()
        try:
            target.relative_to(tests_root)
        except ValueError:
            return f"pytest target must be inside tests/: {token}"
    return None


def _run_collect_only(root: Path, test_cmd: str) -> tuple[int, str]:
    try:
        command = collect_only_command(test_cmd)
    except ValueError as exc:
        return 2, f"unsafe test command: {exc}"
    env = os.environ.copy()
    plain_command: list[str] = []
    for token in command:
        if "=" in token and not token.startswith("-") and not plain_command:
            key, value = token.split("=", 1)
            if key not in SAFE_TEST_ENV_KEYS:
                return 2, f"unsafe test environment key: {key}"
            if key == "PYTHONPATH":
                for part in value.split(os.pathsep):
                    if not part:
                        continue
                    try:
                        (root / part).resolve().relative_to(root.resolve())
                    except ValueError:
                        return 2, f"unsafe PYTHONPATH entry: {part}"
            env[key] = value
        else:
            plain_command.append(token)
    if not plain_command:
        return 0, ""
    unsafe_target = _validate_pytest_targets(root, command)
    if unsafe_target:
        return 2, unsafe_target
    result = subprocess.run(plain_command, cwd=root, env=env, capture_output=True, text=True, timeout=120)
    return result.returncode, result.stdout + result.stderr


def _refresh_inventory(root: Path, header: TzHeader) -> tuple[dict[str, object] | None, str | None]:
    command = [
        sys.executable, str(root / "scripts/skills/inventory_before_build.py"), "--root", str(root),
        "--feature-id", header.feature_id or "", "--problem-id", header.problem_id or "",
        "--change", header.change or "", "--symbols", ",".join(header.symbols),
        "--keywords", ",".join(header.keywords), "--json",
    ]
    result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=120)
    if result.returncode not in {0, 1}:
        return None, "inventory helper завершился с ошибкой: " + result.stderr[-500:]
    try:
        payload = json.loads(result.stdout)
    except (json.JSONDecodeError, TypeError):
        return None, "inventory helper не вернул валидный JSON: " + result.stderr[-500:]
    return payload if isinstance(payload, dict) else None, None if isinstance(payload, dict) else "inventory JSON должен быть object"

def _validate_inventory(root: Path, header: TzHeader, path: Path) -> list[str]:
    try:
        supplied = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"inventory не читается: {exc}"]
    if not isinstance(supplied, dict):
        return ["inventory JSON должен быть object"]
    failures: list[str] = []
    if supplied.get("schema_version") != "mango_prebuild_inventory_v1":
        failures.append("неверная schema inventory")
    if supplied.get("feature_id") != header.feature_id or supplied.get("problem_id") != header.problem_id:
        failures.append("inventory относится к другому Feature-ID/Problem-ID")
    if not isinstance(supplied.get("coverage"), dict) or set(supplied["coverage"]) != INVENTORY_COVERAGE or not supplied.get("queries") or not supplied.get("generator_command_sha256"):
        failures.append("inventory coverage/queries/generator hash неполны")
    candidates = supplied.get("candidates")
    if not isinstance(candidates, list):
        failures.append("inventory не содержит evidence candidates")
        candidates = []
    decision = supplied.get("decision")
    owner = supplied.get("selected_owner")
    if decision == "stop" or supplied.get("unresolved"):
        failures.append("inventory требует STOP")
    if decision in {"reuse", "extend", "port"}:
        wanted = {"DONOR_REF"} if decision == "port" else {"ACTIVE_REUSE", "ACTIVE_EXTEND"}
        if not isinstance(owner, dict) or not owner.get("path") or not any(
            item.get("classification") in wanted
            and item.get("verified_in_raw_source") is True
            and item.get("path") == owner.get("path")
            and item.get("symbol") == owner.get("symbol")
            for item in candidates if isinstance(item, dict)
        ):
            failures.append(f"decision={decision} без raw-подтверждённого owner")
    elif decision == "new":
        if supplied.get("graph_matches_head") is not True or not any(
            isinstance(item, dict) and item.get("classification") == "ABSENT_PROVEN" for item in candidates
        ):
            failures.append("decision=new без ABSENT_PROVEN на свежем Graphify")
    else:
        failures.append("неизвестный inventory decision")
    current, error = _refresh_inventory(root, header)
    if error:
        failures.append(error)
    elif current is not None:
        normalized = lambda value: {key: item for key, item in value.items() if key != "generated_at"}
        if normalized(supplied) != normalized(current):
            failures.append("inventory протух или подделан: повторный scan отличается")
    return failures


def _validate_claude_receipt(root: Path, receipt: Path, task: Path, inventory: Path) -> list[str]:
    result = subprocess.run(
        [sys.executable, str(root / "scripts/make_audit_pack.py"), "--root", str(root),
         "--verify-receipt", str(receipt), "--expected-task", str(task),
         "--expected-inventory", str(inventory)],
        cwd=root, capture_output=True, text=True, timeout=30,
    )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return ["Claude receipt verifier не вернул JSON: " + (result.stderr or result.stdout)[-500:]]
    return [] if result.returncode == 0 and payload.get("ok") is True else [
        "Claude receipt невалиден: " + "; ".join(payload.get("errors") or ["unknown error"])
    ]

def run_preflight(
    root: Path, tz_path: Path, *, inventory_path: Path | None = None,
    claude_receipt: Path | None = None, run_collect: bool = True,
) -> tuple[bool, list[str]]:
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
    tz_text = tz_path.read_text(encoding="utf-8", errors="ignore")
    header = parse_tz_header(tz_text)
    if is_code_task(header, tz_text):
        missing = [
            name for name, value in (
                ("Feature-ID", header.feature_id), ("Problem-ID", header.problem_id),
                ("Изменение", header.change), ("Ключевые-символы/слова", header.symbols or header.keywords),
            ) if not value
        ]
        if missing:
            failures.append("code-ТЗ без обязательных полей: " + ", ".join(missing))
        elif header.change not in {"new", "extend", "fix", "remove"}:
            failures.append(f"неверное Изменение: {header.change}")
        if inventory_path is None:
            failures.append("code-ТЗ требует --inventory")
        elif not missing:
            failures.extend(_validate_inventory(root, header, inventory_path.resolve()))
        if claude_receipt is None:
            failures.append("code-ТЗ требует --claude-receipt")
        elif inventory_path is not None and not missing:
            failures.extend(_validate_claude_receipt(root, claude_receipt.resolve(), tz_path, inventory_path.resolve()))
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
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--claude-receipt", type=Path)
    parser.add_argument("--skip-collect-only", action="store_true")
    args = parser.parse_args(argv)
    ok, failures = run_preflight(
        args.root, args.tz, inventory_path=args.inventory, claude_receipt=args.claude_receipt,
        run_collect=not args.skip_collect_only,
    )
    if not ok:
        print("PREFLIGHT: СТОП")
        for failure in failures:
            print(f" - {failure}")
        return 1
    print("PREFLIGHT: OK")
    header = parse_tz_header(args.tz.read_text(encoding="utf-8", errors="ignore"))
    print("REQUIRED_ROLES: " + json.dumps(required_roles(header, args.tz.read_text(encoding="utf-8", errors="ignore")), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
