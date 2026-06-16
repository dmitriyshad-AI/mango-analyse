#!/usr/bin/env python3
"""Create a local audit pack with PII redaction and a final manifest."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path


DEFAULT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLIENT_PATHS = (
    "product_data/knowledge_base/",
    "src/mango_mvp/channels/",
    "src/mango_mvp/integrations/draft_loop.py",
    "scripts/run_amo_wappi_draft_loop.py",
    "scripts/run_telegram_public_pilot_bots.py",
    "product_data/telegram_dynamic_test_sets/",
)
PHONE_RE = re.compile(r"(?<!\d)(?:\+7|8)[\s\-(]*\d{3}[\s\-)]*\d{3}[\s\-]*\d{2}[\s\-]*\d{2}(?!\d)")
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
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
        "pii_redaction": ["ru_phone", "email"],
    }
    (pack / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return pack


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("slug")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out-root", type=Path)
    parser.add_argument("--tests", type=Path)
    parser.add_argument("--review-file", type=Path)
    parser.add_argument("--base", default="main")
    parser.add_argument("--semantic", action="store_true")
    parser.add_argument("--amo", action="store_true")
    args = parser.parse_args(argv)
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
