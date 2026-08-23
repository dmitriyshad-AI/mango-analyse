#!/usr/bin/env python3
"""Create a local audit pack with PII redaction and a final manifest."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4


DEFAULT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = DEFAULT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mango_mvp.replay_exam.pseudonymizer import EMAIL_RE, LABELED_BARE_PHONE_RE, PHONE_RE

DEFAULT_CLIENT_PATHS = (
    "product_data/knowledge_base/",
    "src/mango_mvp/channels/",
    "src/mango_mvp/integrations/draft_loop.py",
    "scripts/run_amo_wappi_draft_loop.py",
    "scripts/run_telegram_public_pilot_bots.py",
    "product_data/telegram_dynamic_test_sets/",
)
TEMPLATES = {
    "implementation_notes.md": "# Что сделано\n\n# Как проверялось\n\n# Что осталось\n",
    "risk_review.md": "# Риски\n\n- Клиентский риск:\n- Данные/записи:\n- Откат:\n",
    "backward_compatibility.md": "# Обратная совместимость\n\n- Форматы:\n- Потребители:\n",
}
AMO_TEMPLATES = {
    "dry_run_report.md": "# Dry-run\n\n",
    "rollback_contract.md": "# Snapshot / rollback\n\n",
    "readback_plan.md": "# Readback plan\n\n",
    "not_written_live.md": "# Что не было записано live\n\n",
}
CONTEXT_PREFIXES = ("src", "scripts", "tests", ".agents", ".claude")
CONTEXT_EXACT = frozenset({
    "AGENTS.md", "CLAUDE.md", "README.md", "ARCHITECTURE.md",
    "docs/PROJECT_NOW.md", "docs/RUNBOOK.md", "docs/DECISIONS_LOG.md",
    "scripts/skills/inventory_before_build.py",
})
CONTEXT_BLOCKED = frozenset({
    ".codex", ".codex_local", "product_data", "stable_runtime", "runtime", "runs",
    "transcripts", "audio", "mail", "calls", "graphify-out",
})
SURFACE_IGNORED_PREFIXES = ("audits/_inbox/",)
CLAUDE_PACK_FILES = frozenset({
    "task.md", "prebuild_inventory.json", "git_context.txt", "context_files.json", "review_prompt.md",
})
SECRET_RE = re.compile(
    r"(?ix)(?:\b[A-Z][A-Z0-9_]*(?:TOKEN|SECRET|API_KEY|PASSWORD)\b|"
    r"[\"']?(?:token|secret|api_key|password|authorization)[\"']?)[ \t]*[:=][ \t]*[\"']?"
    r"(?!\[|<|redacted|required\b)[^\s\"',#}]{8,}|Authorization\s*:\s*Bearer\s+[A-Za-z0-9._-]{12,}|"
    r"\bsk-(?:proj-)?[A-Za-z0-9_-]{12,}|\b\d{6,12}:[A-Za-z0-9_-]{20,}\b"
)


def _run_git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-c", "core.quotepath=off", *args],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout if result.returncode == 0 else result.stderr


def mask_pii(text: str) -> str:
    text = PHONE_RE.sub("[redacted_phone]", text)
    text = LABELED_BARE_PHONE_RE.sub(
        lambda match: f"{match.group('label')}{match.group('separator')}[redacted_phone]",
        text,
    )
    return EMAIL_RE.sub("[redacted_email]", text)


def _assert_safe_output_path(root: Path, path: Path) -> None:
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    parts = set(resolved_path.parts)
    if ".codex" in parts or ".codex_local" in parts:
        raise ValueError("output path points to codex home/local state")
    try:
        rel = resolved_path.relative_to(resolved_root)
    except ValueError:
        return
    if rel.parts and rel.parts[0] == "stable_runtime":
        raise ValueError("output path points to stable_runtime")


def _write_text(path: Path, text: str, written: list[str]) -> None:
    path.write_text(mask_pii(text), encoding="utf-8")
    written.append(path.name)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_required(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-c", "core.quotepath=off", *args], cwd=root,
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode:
        raise ValueError(f"git {' '.join(args)} failed: {result.stderr[-300:]}")
    return result.stdout


def _forbidden_context_path(rel: Path) -> bool:
    parts = tuple(part.casefold() for part in rel.parts)
    name = rel.name.casefold()
    envish = any(part == ".env" or part.startswith(".env.") for part in parts) or name.endswith((".env", ".env.example"))
    return envish or any(part in CONTEXT_BLOCKED for part in parts)


def _repo_file(root: Path, value: Path, kind: str = "context") -> tuple[Path, str]:
    raw = value.expanduser()
    lexical = Path(os.path.abspath(raw if raw.is_absolute() else root / raw))
    try:
        rel = lexical.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{kind} path outside repository") from exc
    cursor = root
    for part in rel.parts:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError(f"{kind} path contains symlink: {rel}")
    if not lexical.is_file() or _forbidden_context_path(rel):
        raise ValueError(f"unsafe {kind} path: {rel}")
    posix = rel.as_posix()
    if mask_pii(posix) != posix:
        raise ValueError(f"{kind} path contains PII-like value: {rel}")
    if kind == "task" and not (posix.startswith("tasks/_running/") and rel.suffix == ".md"):
        raise ValueError("task must be a markdown file in tasks/_running")
    if kind == "inventory" and not (posix.startswith("audits/_inbox/") and rel.name == "prebuild_inventory.json"):
        raise ValueError("inventory must be audits/_inbox/.../prebuild_inventory.json")
    if kind == "context" and not (
        posix in CONTEXT_EXACT or rel.parts[0] in CONTEXT_PREFIXES
    ):
        raise ValueError(f"context path is not allowlisted: {rel}")
    if kind == "context" and (rel.name == "graph.json" or rel.suffix.casefold() in {".db", ".sqlite", ".jsonl", ".mp3", ".wav", ".m4a"}):
        raise ValueError(f"unsafe context file type: {rel}")
    return lexical, posix


def _assert_no_secret(name: str, data: bytes) -> None:
    if SECRET_RE.search(data.decode("utf-8", errors="ignore")):
        raise ValueError(f"secret-like value in {name}; package blocked")


def _assert_no_pii(name: str, data: bytes) -> None:
    text = data.decode("utf-8", errors="ignore")
    if mask_pii(text) != text:
        raise ValueError(f"PII-like value in {name}; package blocked")


def _audit_evidence_path(root: Path, value: Path) -> Path:
    lexical = Path(os.path.abspath(value if value.is_absolute() else root / value))
    try:
        rel = lexical.relative_to(root)
    except ValueError as exc:
        raise ValueError("Claude evidence outside repository") from exc
    cursor = root
    linked = False
    for part in rel.parts:
        cursor /= part
        linked = linked or cursor.is_symlink()
    if rel.parts[:2] != ("audits", "_inbox") or linked:
        raise ValueError("Claude evidence must be under audits/_inbox")
    return lexical


def _field(text: str, name: str) -> str:
    match = re.search(rf"^{re.escape(name)}:\s*([^\n]+)$", text, re.M)
    value = match.group(1).strip() if match else ""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError(f"missing or unsafe {name}")
    return value


def _task_context_paths(root: Path, text: str) -> tuple[Path, ...]:
    found: list[Path] = []
    pattern = r"(?<![\w/])((?:[\w.-]+/)+[\w.-]+\.(?:py|sh|md|json|yaml|yml))\b"
    for value in re.findall(pattern, text):
        path = Path(value)
        try:
            _repo_file(root, path)
        except ValueError:
            continue
        if path not in found:
            found.append(path)
    return tuple(found)


def _local_import_paths(root: Path, paths: list[Path]) -> tuple[Path, ...]:
    found: list[Path] = []
    for path in paths:
        if path.suffix != ".py":
            continue
        try:
            tree = ast.parse((root / path).read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeError):
            continue
        candidates: list[Path] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = Path(*alias.name.split("."))
                    candidates.extend((module, Path("src") / module))
            elif isinstance(node, ast.ImportFrom):
                module = Path(*(node.module or "").split(".")) if node.module else Path()
                if node.level:
                    base = path.parent
                    for _ in range(node.level - 1):
                        base = base.parent
                    candidates.append(base / module)
                    candidates.extend(base / module / alias.name for alias in node.names if alias.name != "*")
                else:
                    candidates.extend((module, Path("src") / module))
                    candidates.extend(module / alias.name for alias in node.names if alias.name != "*")
                    candidates.extend(Path("src") / module / alias.name for alias in node.names if alias.name != "*")
        for stem in candidates:
            for candidate in (stem.with_suffix(".py"), stem / "__init__.py"):
                try:
                    _, safe_rel = _repo_file(root, candidate)
                except ValueError:
                    continue
                safe = Path(safe_rel)
                if safe not in found:
                    found.append(safe)
                break
    return tuple(found)


def _safe_context_name(rel: str) -> bool:
    path = Path(rel)
    return bool(
        (rel in CONTEXT_EXACT or (path.parts and path.parts[0] in CONTEXT_PREFIXES))
        and not _forbidden_context_path(path) and mask_pii(rel) == rel
    )


def _branch_diff(root: Path, base: str = "main") -> tuple[str, tuple[str, ...], tuple[dict[str, str], ...]]:
    raw = _git_required(root, "diff", "--name-status", f"{base}...HEAD")
    safe, numstat = [], []
    for line in raw.splitlines():
        rel = line.split("\t")[-1].strip()
        if _safe_context_name(rel):
            safe.append(rel)
    for line in _git_required(root, "diff", "--numstat", f"{base}...HEAD").splitlines():
        added, deleted, rel = line.split("\t", 2)
        if _safe_context_name(rel):
            numstat.append({"path": rel, "added": added, "deleted": deleted})
    return _sha(raw.encode()), tuple(sorted(set(safe))), tuple(sorted(numstat, key=lambda item: item["path"]))


def _review_prompt(head: str, pack_rel: str, nonce: str) -> bytes:
    manifest_rel = f"{pack_rel}/manifest.json"
    return (
        "Проведи независимый read-only аудит задачи. Не меняй файлы и внешние системы.\n"
        f"PACK_DIR: {pack_rel}\nMANIFEST: {manifest_rel}\nNONCE: {nonce}\n"
        "Файлы пакета читай только из PACK_DIR: task.md, prebuild_inventory.json, git_context.txt, "
        "context_files.json и manifest.json. Исходники читай только по путям из context_files.json.\n"
        "Начни ответ отдельными строками: `MODE: READ_ONLY`, точные `PACK_DIR`, `MANIFEST`, `NONCE`, "
        "`CONTEXT_READ: task.md, prebuild_inventory.json, git_context.txt, context_files.json, "
        "manifest.json`, затем `HEAD: ...`, `FILES_HASH: ...` из manifest, "
        "`SELECTED_OWNER: ...` из inventory и `VERDICT: PASS|PASS_WITH_FIXES|STOP`.\n"
        "Ставь PASS, если приёмка выполнена, даже при наличии неблокирующих нот. "
        "PASS_WITH_FIXES означает обязательную правку до preflight; STOP — сработавшее STOP-условие ТЗ.\n"
        f"Назови полный HEAD {head} и дай конкретные замечания минимум в 200 символах. "
        "Не повторяй значения или синтетические примеры токенов/ключей.\n"
    ).encode()


def _valid_review_result(
    text: str, head: str, pack_rel: str, nonce: str, files_hash: str, owner_path: str,
) -> bool:
    folded = text.casefold()
    return (
        len(text) >= 200 and f"head: {head}" in folded and "mode: read_only" in folded
        and re.search(rf"^PACK_DIR:\s*{re.escape(pack_rel)}\s*$", text, re.M)
        and re.search(rf"^MANIFEST:\s*{re.escape(pack_rel)}/manifest\.json\s*$", text, re.M)
        and re.search(rf"^NONCE:\s*{re.escape(nonce)}\s*$", text, re.M)
        and re.search(rf"^FILES_HASH:\s*{re.escape(files_hash)}\s*$", text, re.M)
        and re.search(rf"^SELECTED_OWNER:\s*{re.escape(owner_path)}\s*$", text, re.M)
        and _review_verdict(text) in {"PASS", "PASS_WITH_FIXES", "STOP"}
        and all(name in folded for name in CLAUDE_PACK_FILES - {"review_prompt.md"})
    )


def _review_verdict(text: str) -> str:
    matches = re.findall(r"^VERDICT:\s*(PASS|PASS_WITH_FIXES|STOP)\s*$", text, re.M | re.I)
    return matches[0].upper() if len(matches) == 1 else ""


def _code_surface(root: Path, head: str | None = None) -> tuple[str, str, tuple[Path, ...]]:
    status = _git_required(root, "status", "--porcelain", "--untracked-files=all").splitlines()
    hashes, safe_status, safe_files, blocked = [f"head:{head or _git_required(root, 'rev-parse', 'HEAD').strip()}"], [], [], 0
    for line in status:
        rel = line[3:].split(" -> ")[-1].strip().strip('"')
        if any(rel.startswith(prefix) for prefix in SURFACE_IGNORED_PREFIXES):
            continue
        rel_path = Path(rel)
        allowed = rel in CONTEXT_EXACT or (rel_path.parts and rel_path.parts[0] in CONTEXT_PREFIXES)
        if not allowed or _forbidden_context_path(rel_path):
            blocked += 1
            hashes.append(f"blocked:{line[:2]}:{_sha(rel.encode())}")
            continue
        try:
            source, safe_rel = _repo_file(root, rel_path)
            hashes.append(f"{line[:2]}:{safe_rel}:{_sha(source.read_bytes())}")
            safe_files.append(Path(safe_rel))
        except ValueError:
            if (root / rel_path).exists() or (root / rel_path).is_symlink():
                raise
            hashes.append(f"{line[:2]}:{rel}:deleted")
        safe_status.append(mask_pii(line))
    surface = _sha("\n".join(sorted(hashes)).encode())
    return surface, "\n".join([*safe_status, f"blocked_paths: {blocked}"]), tuple(safe_files)


def _changed_files(root: Path, base: str) -> str:
    diff = _run_git(root, "diff", "--name-status", f"{base}...HEAD")
    status = _run_git(root, "status", "--porcelain")
    return (diff + "\n" + status).strip() + "\n"


def _semantic_required(changed_text: str, client_paths: tuple[str, ...] = DEFAULT_CLIENT_PATHS) -> bool:
    return any(marker in changed_text for marker in client_paths)


def create_audit_pack(
    root: Path,
    slug: str,
    *,
    out_root: Path | None = None,
    tests_file: Path | None = None,
    review_file: Path | None = None,
    base: str = "main",
    semantic: bool = False,
    amo: bool = False,
) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", slug):
        raise ValueError("slug may contain only ASCII letters, digits, dots, underscores and dashes")
    root = root.resolve()
    base_dir = out_root or root / "audits" / "_inbox"
    pack = base_dir / f"{slug}_{datetime.now():%Y%m%d%H%M%S}"
    _assert_safe_output_path(root, pack)
    pack.mkdir(parents=True, exist_ok=False)
    written: list[str] = []

    changed = _changed_files(root, base)
    _write_text(pack / "changed_files.txt", changed, written)
    git_context = (
        f"branch: {_run_git(root, 'rev-parse', '--abbrev-ref', 'HEAD').strip()}\n"
        f"rev: {_run_git(root, 'rev-parse', '--short', 'HEAD').strip()}\n\n"
        f"{_run_git(root, 'status', '--short')}"
    )
    _write_text(pack / "git_context.txt", git_context, written)
    if tests_file and tests_file.exists():
        _write_text(pack / "test_output.txt", tests_file.read_text(encoding="utf-8"), written)
    else:
        _write_text(pack / "test_output.txt", "# Тесты\n\nНе приложены.\n", written)
    for name, body in TEMPLATES.items():
        _write_text(pack / name, body, written)
    semantic_needed = semantic or _semantic_required(changed)
    if semantic_needed:
        _write_text(
            pack / "semantic_review.md",
            "# Semantic review\n\n- Бренды раздельны?\n- Цены/даты/условия из KB?\n- P0 и ПДн не ослаблены?\n",
            written,
        )
    if amo:
        for name, body in AMO_TEMPLATES.items():
            _write_text(pack / name, body, written)
    if review_file and review_file.exists():
        _write_text(pack / "review_self.txt", review_file.read_text(encoding="utf-8"), written)

    manifest = {
        "slug": slug,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "repo_head": _run_git(root, "rev-parse", "--short", "HEAD").strip(),
        "semantic_required": semantic_needed,
        "client_paths": list(DEFAULT_CLIENT_PATHS),
        "files_written_before_manifest": written,
        "pii_redaction": ["phone", "email"],
    }
    (pack / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return pack


def create_claude_context_pack(
    root: Path, slug: str, task_path: Path, inventory_path: Path, *,
    context_files: tuple[Path, ...] = (), out_root: Path | None = None,
) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", slug):
        raise ValueError("unsafe slug")
    root = root.resolve()
    task, task_rel = _repo_file(root, task_path, "task")
    inventory, inventory_rel = _repo_file(root, inventory_path, "inventory")
    task_raw, inventory_raw = task.read_bytes(), inventory.read_bytes()
    _assert_no_secret(task_rel, task_raw)
    _assert_no_secret(inventory_rel, inventory_raw)
    _assert_no_pii(inventory_rel, inventory_raw)
    inventory_json = json.loads(inventory_raw)
    owner = inventory_json.get("selected_owner") or {}
    head = _git_required(root, "rev-parse", "HEAD").strip()
    branch = _git_required(root, "rev-parse", "--abbrev-ref", "HEAD").strip()
    surface_hash, safe_status, dirty_context = _code_surface(root, head)
    branch_diff_hash, branch_changed, branch_numstat = _branch_diff(root)
    defaults = [Path(name) for name in CONTEXT_EXACT if (root / name).is_file()]
    task_context = _task_context_paths(root, task_raw.decode("utf-8", errors="ignore"))
    requested = [*defaults, *context_files, *task_context, *dirty_context]
    if owner.get("path"):
        requested.append(Path(owner["path"]))
    requested.extend(_local_import_paths(root, requested))
    sources: dict[str, str] = {}
    source_scan_warnings: list[str] = []
    dirty_rel = {path.as_posix() for path in dirty_context}
    for requested_path in dict.fromkeys(requested):
        source, rel = _repo_file(root, requested_path)
        raw = source.read_bytes()
        if rel in dirty_rel:
            _assert_no_secret(rel, raw)
            _assert_no_pii(rel, raw)
        raw_text = raw.decode("utf-8", errors="ignore")
        if SECRET_RE.search(raw_text) or mask_pii(raw_text) != raw_text:
            source_scan_warnings.append(rel)
        sources[rel] = _sha(raw)
    pack = (out_root or root / "audits/_inbox") / f"{slug}_{datetime.now():%Y%m%d%H%M%S}"
    _assert_safe_output_path(root, pack)
    pack = _audit_evidence_path(root, pack)
    pack_rel = pack.relative_to(root).as_posix()
    context_json = json.dumps(
        {
            "schema_version": "mango_claude_context_files_v1",
            "files": dict(sorted(sources.items())),
            "changed_vs_main": [path for path in branch_changed if path in sources],
            "branch_numstat": [item for item in branch_numstat if item["path"] in sources],
            "source_scan_warnings": sorted(set(source_scan_warnings)),
        },
        ensure_ascii=False, indent=2,
    ).encode() + b"\n"
    git_context = mask_pii(
        f"head: {head}\nbranch: {branch}\nworktree_label: {root.name}\n"
        f"worktree_path_sha256: {_sha(str(root).encode())}\nstatus:\n"
        + safe_status
    ).encode()
    task_copy = mask_pii(task_raw.decode("utf-8", errors="replace")).encode()
    evidence_hashes = {"task.md": _sha(task_raw), "prebuild_inventory.json": _sha(inventory_raw), **sources}
    files_hash = _sha("\n".join(f"{name}:{sha}" for name, sha in sorted(evidence_hashes.items())).encode())
    prompt_template_hash = _sha(_review_prompt(head, "<PACK_DIR>", "<NONCE>"))
    nonce = _sha(f"{head}\n{files_hash}\n{prompt_template_hash}".encode())[:32]
    prompt = _review_prompt(head, pack_rel, nonce)
    files = {
        "task.md": task_copy, "prebuild_inventory.json": inventory_raw,
        "git_context.txt": git_context, "context_files.json": context_json,
        "review_prompt.md": prompt,
    }
    for name, data in files.items():
        _assert_no_secret(name, data)
    prompt_hash = _sha(prompt)
    manifest = {
        "schema_version": "mango_claude_context_pack_v1", "slug": slug,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "head": head, "branch": branch, "worktree_label": root.name,
        "worktree_path_sha256": _sha(str(root).encode()),
        "pack_path": pack_rel, "review_nonce": nonce,
        "review_nonce_kind": "deterministic_binding_not_freshness",
        "feature_id": _field(task_raw.decode(errors="ignore"), "Feature-ID"),
        "problem_id": _field(task_raw.decode(errors="ignore"), "Problem-ID"),
        "task_source": {"path": task_rel, "sha256": _sha(task_raw)},
        "inventory_source": {"path": inventory_rel, "sha256": _sha(inventory_raw)},
        "files": {name: _sha(data) for name, data in files.items()},
        "prompt_sha256": prompt_hash, "prompt_template_sha256": prompt_template_hash,
        "files_hash": files_hash,
        "code_surface_sha256": surface_hash,
        "branch_diff_base": "main", "branch_diff_sha256": branch_diff_hash,
        "dedupe_key": _sha(f"{head}\n{prompt_template_hash}\n{files_hash}".encode()),
        "pii_redaction": ["phone", "email"],
        "secret_handling": "pack inputs blocked; allowlisted source contents referenced by path/hash, not copied",
    }
    pack.mkdir(parents=True, exist_ok=False)
    for name, data in files.items():
        (pack / name).write_bytes(data)
    (pack / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return pack


def verify_claude_context(
    root: Path, pack: Path, receipt: Path | None = None, *,
    expected_task: Path | None = None, expected_inventory: Path | None = None,
) -> list[str]:
    root, pack = root.resolve(), _audit_evidence_path(root.resolve(), pack).resolve()
    errors: list[str] = []
    try:
        manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("schema_version") != "mango_claude_context_pack_v1":
            errors.append("invalid context pack schema")
        pack_rel = pack.relative_to(root).as_posix()
        if manifest.get("pack_path") != pack_rel:
            errors.append("context pack path mismatch")
        nonce = str(manifest.get("review_nonce", ""))
        if not re.fullmatch(r"[0-9a-f]{32}", nonce):
            errors.append("invalid review nonce")
        expected_files = manifest.get("files") or {}
        actual_files = {item.name for item in pack.iterdir() if item.is_file() and item.name != "manifest.json"}
        if actual_files != set(expected_files) or actual_files != CLAUDE_PACK_FILES:
            errors.append("context pack file set mismatch")
        for name, wanted in expected_files.items():
            if not (pack / name).is_file() or _sha((pack / name).read_bytes()) != wanted:
                errors.append(f"context pack byte mismatch: {name}")
        context_payload = json.loads((pack / "context_files.json").read_text(encoding="utf-8"))
        context = context_payload["files"]
        for rel, wanted in context.items():
            source, _ = _repo_file(root, Path(rel))
            if _sha(source.read_bytes()) != wanted:
                errors.append(f"context source drift: {rel}")
        evidence_hashes = {
            "task.md": manifest["task_source"]["sha256"],
            "prebuild_inventory.json": manifest["inventory_source"]["sha256"], **context,
        }
        files_hash = _sha("\n".join(f"{name}:{sha}" for name, sha in sorted(evidence_hashes.items())).encode())
        prompt_raw = (pack / "review_prompt.md").read_bytes()
        template_hash = _sha(_review_prompt(manifest["head"], "<PACK_DIR>", "<NONCE>"))
        expected_nonce = _sha(f"{manifest['head']}\n{files_hash}\n{template_hash}".encode())[:32]
        expected_dedupe = _sha(f"{manifest['head']}\n{template_hash}\n{files_hash}".encode())
        if (
            files_hash != manifest.get("files_hash") or _sha(prompt_raw) != manifest.get("prompt_sha256")
            or prompt_raw != _review_prompt(manifest["head"], pack_rel, nonce)
            or template_hash != manifest.get("prompt_template_sha256") or nonce != expected_nonce
            or manifest.get("dedupe_key") != expected_dedupe
        ):
            errors.append("prompt/files hash mismatch")
        if _git_required(root, "rev-parse", "HEAD").strip() != manifest.get("head"):
            errors.append("HEAD mismatch")
        if (
            _git_required(root, "rev-parse", "--abbrev-ref", "HEAD").strip() != manifest.get("branch")
            or root.name != manifest.get("worktree_label")
            or _sha(str(root).encode()) != manifest.get("worktree_path_sha256")
        ):
            errors.append("branch/worktree mismatch")
        if _code_surface(root, manifest["head"])[0] != manifest.get("code_surface_sha256"):
            errors.append("code surface mismatch")
        diff_hash, changed, numstat = _branch_diff(root, str(manifest.get("branch_diff_base", "main")))
        if (
            diff_hash != manifest.get("branch_diff_sha256")
            or context_payload.get("changed_vs_main") != [path for path in changed if path in context]
            or context_payload.get("branch_numstat") != [item for item in numstat if item["path"] in context]
        ):
            errors.append("branch diff mismatch")
        for label, expected, key in (("task", expected_task, "task_source"), ("inventory", expected_inventory, "inventory_source")):
            source, rel = _repo_file(root, Path(manifest[key]["path"]), label)
            if _sha(source.read_bytes()) != manifest[key]["sha256"]:
                errors.append(f"{label} source drift")
            if expected is not None and _repo_file(root, expected, label)[1] != rel:
                errors.append(f"{label} binding mismatch")
        if receipt is not None:
            receipt = _audit_evidence_path(root, receipt)
            expected_receipt = pack.with_name(pack.name + "_claude_receipt.json")
            expected_output = pack.with_name(pack.name + "_claude_cli.json")
            if receipt.resolve() != expected_receipt.resolve():
                errors.append("receipt path is not canonical")
            data = json.loads(receipt.read_text(encoding="utf-8"))
            required = {
                "schema_version": "mango_claude_receipt_v1", "created_by": "run_claude_review_v1",
                "command_profile": "claude_readonly_v1", "receipt_kind": "local_unsigned",
                "status": "completed_read_only", "mode": "read_only",
            }
            if (
                any(data.get(key) != value for key, value in required.items())
                or data.get("claude_exit_code") != 0 or data.get("permission_mode") != "plan"
                or data.get("allowed_tools") != ["Read", "Glob", "Grep"] or data.get("safe_mode") is not True
                or not isinstance(data.get("output_secret_like_redactions"), int)
            ):
                errors.append("receipt status/mode invalid")
            if data.get("verdict") != "PASS":
                errors.append(f"Claude verdict is not PASS: {data.get('verdict') or 'missing'}")
            if data.get("head") != manifest.get("head") or data.get("manifest_sha256") != _sha((pack / "manifest.json").read_bytes()):
                errors.append("receipt HEAD/manifest mismatch")
            if data.get("prompt_sha256") != manifest.get("prompt_sha256") or data.get("files_hash") != manifest.get("files_hash"):
                errors.append("receipt prompt/files mismatch")
            if data.get("dedupe_key") != manifest.get("dedupe_key") or data.get("manifest_path") != str((pack / "manifest.json").relative_to(root)):
                errors.append("receipt pack binding mismatch")
            if data.get("pack_path") != pack_rel or data.get("review_nonce") != nonce:
                errors.append("receipt nonce/path mismatch")
            command = data.get("command")
            command_tail = [
                "--safe-mode", "--permission-mode", "plan", "--tools", "Read,Glob,Grep",
                "--strict-mcp-config", "--no-chrome", "--disable-slash-commands", "--no-session-persistence",
                "--effort", "high", "--model", data.get("model"), "--session-id", data.get("session"),
                "--output-format", "json", "-p", "<review_prompt.md>",
            ]
            if not isinstance(command, list) or len(command) < 2 or command[1:] != command_tail:
                errors.append("receipt command mismatch")
            output = _audit_evidence_path(root, Path(str(data.get("output_path", ""))))
            if output.resolve() != expected_output.resolve():
                errors.append("Claude output path is not canonical")
            output_json = json.loads(output.read_text(encoding="utf-8")) if output.is_file() else {}
            inventory_payload = json.loads((pack / "prebuild_inventory.json").read_text(encoding="utf-8"))
            owner_path = str((inventory_payload.get("selected_owner") or {}).get("path") or "NONE")
            if output.is_file():
                _assert_no_secret("Claude output", output.read_bytes())
            if (
                not output.is_file() or _sha(output.read_bytes()) != data.get("output_sha256")
                or output_json.get("session_id") != data.get("session")
                or not _valid_review_result(
                    str(output_json.get("result", "")), manifest["head"], pack_rel, nonce,
                    manifest["files_hash"], owner_path,
                )
            ):
                errors.append("Claude output missing, changed or does not name HEAD")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        errors.append(f"invalid Claude context evidence: {exc}")
    return errors


def run_claude_review(
    root: Path, pack: Path, *, model: str = "opus", repeat_reason: str = "",
    claude_bin: Path | None = None, timeout: int = 1800,
) -> Path:
    root, pack = root.resolve(), _audit_evidence_path(root.resolve(), pack).resolve()
    errors = verify_claude_context(root, pack)
    if errors:
        raise ValueError("; ".join(errors))
    manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
    for old in pack.parent.glob("*_claude_receipt.json"):
        try:
            if json.loads(old.read_text(encoding="utf-8")).get("dedupe_key") == manifest["dedupe_key"] and not repeat_reason:
                raise ValueError("duplicate Claude review requires --repeat-reason")
        except json.JSONDecodeError:
            continue
    binary = (claude_bin or Path.home() / ".local/bin/claude").resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise ValueError("Claude CLI executable not found")
    session = str(uuid4())
    command = [
        str(binary), "--safe-mode", "--permission-mode", "plan", "--tools", "Read,Glob,Grep",
        "--strict-mcp-config", "--no-chrome", "--disable-slash-commands", "--no-session-persistence",
        "--effort", "high", "--model", model, "--session-id", session,
        "--output-format", "json", "-p", (pack / "review_prompt.md").read_text(encoding="utf-8"),
    ]
    result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        detail = SECRET_RE.sub(
            "[redacted_secret_like]", mask_pii(result.stderr[-500:]).replace(str(root), "[redacted_worktree_path]"),
        )
        raise ValueError(f"Claude CLI failed rc={result.returncode}: {detail}")
    failed = pack.with_name(f"{pack.name}_claude_cli_failed_{session}.json")
    try:
        output_json = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        sanitized = SECRET_RE.sub(
            "[redacted_secret_like]", mask_pii(result.stdout).replace(str(root), "[redacted_worktree_path]"),
        )
        failed.write_text(sanitized, encoding="utf-8")
        raise ValueError("Claude CLI returned invalid JSON; sanitized output preserved") from exc
    raw_review = str(output_json.get("result", ""))
    masked_review = mask_pii(raw_review).replace(str(root), "[redacted_worktree_path]")
    review, redactions = SECRET_RE.subn("[redacted_secret_like]", masked_review)
    output_json = {"session_id": output_json.get("session_id"), "result": review}
    output_raw = (json.dumps(output_json, ensure_ascii=False) + "\n").encode()
    _assert_no_secret("sanitized Claude output", output_raw)
    inventory_payload = json.loads((pack / "prebuild_inventory.json").read_text(encoding="utf-8"))
    owner_path = str((inventory_payload.get("selected_owner") or {}).get("path") or "NONE")
    if output_json.get("session_id") != session or not _valid_review_result(
        review, manifest["head"], manifest["pack_path"], manifest["review_nonce"],
        manifest["files_hash"], owner_path,
    ):
        failed.write_bytes(output_raw)
        raise ValueError("Claude JSON lacks session_id or required structured review")
    if verify_claude_context(root, pack):
        raise ValueError("Claude review changed or invalidated the reviewed surface")
    stored = pack.with_name(pack.name + "_claude_cli.json")
    receipt = pack.with_name(pack.name + "_claude_receipt.json")
    stored.write_bytes(output_raw)
    payload = {
        "schema_version": "mango_claude_receipt_v1", "created_by": "run_claude_review_v1",
        "command_profile": "claude_readonly_v1", "receipt_kind": "local_unsigned",
        "proves": "local pack integrity + HEAD binding under the cooperative workflow",
        "does_not_prove": "Claude execution or reviewer identity against a malicious local writer",
        "status": "completed_read_only", "mode": "read_only", "head": manifest["head"],
        "manifest_path": str((pack / "manifest.json").relative_to(root)),
        "pack_path": manifest["pack_path"], "review_nonce": manifest["review_nonce"],
        "manifest_sha256": _sha((pack / "manifest.json").read_bytes()),
        "prompt_sha256": manifest["prompt_sha256"], "files_hash": manifest["files_hash"],
        "dedupe_key": manifest["dedupe_key"], "model": model, "session": session,
        "verdict": _review_verdict(review),
        "claude_exit_code": result.returncode, "permission_mode": "plan",
        "allowed_tools": ["Read", "Glob", "Grep"], "safe_mode": True,
        "command": [*command[:-1], "<review_prompt.md>"],
        "output_secret_like_redactions": redactions,
        "output_pii_redacted": masked_review != raw_review,
        "output_path": str(stored.relative_to(root)), "output_sha256": _sha(output_raw),
        "repeat_reason": repeat_reason,
    }
    receipt.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("slug", nargs="?")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out-root", type=Path)
    parser.add_argument("--tests", type=Path)
    parser.add_argument("--review-file", type=Path)
    parser.add_argument("--base", default="main")
    parser.add_argument("--semantic", action="store_true")
    parser.add_argument("--amo", action="store_true")
    parser.add_argument("--claude-task", type=Path)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--context-file", action="append", type=Path, default=[])
    parser.add_argument("--run-claude", type=Path)
    parser.add_argument("--model", default="opus")
    parser.add_argument("--repeat-reason", default="")
    parser.add_argument("--verify-receipt", type=Path)
    parser.add_argument("--expected-task", type=Path)
    parser.add_argument("--expected-inventory", type=Path)
    args = parser.parse_args(argv)
    if args.verify_receipt:
        args.verify_receipt = _audit_evidence_path(args.root.resolve(), args.verify_receipt)
        data = json.loads(args.verify_receipt.read_text(encoding="utf-8"))
        pack = (args.root / data["manifest_path"]).resolve().parent
        errors = verify_claude_context(
            args.root, pack, args.verify_receipt,
            expected_task=args.expected_task, expected_inventory=args.expected_inventory,
        )
        print(json.dumps({"ok": not errors, "errors": errors}, ensure_ascii=False))
        return int(bool(errors))
    if args.run_claude:
        print(run_claude_review(
            args.root, args.run_claude, model=args.model, repeat_reason=args.repeat_reason,
        ))
        return 0
    if args.claude_task:
        if not args.slug or not args.inventory:
            parser.error("context mode requires slug and --inventory")
        print(create_claude_context_pack(
            args.root, args.slug, args.claude_task, args.inventory,
            context_files=tuple(args.context_file), out_root=args.out_root,
        ))
        return 0
    if not args.slug:
        parser.error("slug is required")
    pack = create_audit_pack(
        args.root,
        args.slug,
        out_root=args.out_root,
        tests_file=args.tests,
        review_file=args.review_file,
        base=args.base,
        semantic=args.semantic,
        amo=args.amo,
    )
    print(f"OK: {pack}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
