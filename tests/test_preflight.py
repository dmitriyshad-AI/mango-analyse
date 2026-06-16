from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

from scripts import preflight


def _write(path: Path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _tz(root: Path, zones: str = "scripts/, tests/, docs/, tasks/, AGENTS.md, .gitignore") -> Path:
    path = root / "tasks/_running/TZ.md"
    _write(
        path,
        "\n".join(
            [
                "Ветка: main",
                f"Зоны: {zones}",
                "Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_project_now.py tests/test_audit_pack_pii.py tests/test_preflight.py",
                "",
            ]
        ),
    )
    return path


def _prepare_root(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    _write(root / "docs/PROJECT_NOW.md", "fresh")
    _write(root / "docs/worktrees_registry.md", "/tmp/registered\n")
    _write(root / "AGENTS.md", "rules")
    return root


def test_parse_worktrees_ignores_prunable_detached_locked():
    text = """worktree /repo
HEAD abc
branch refs/heads/main

worktree /tmp/detached
HEAD def
detached

worktree /tmp/locked
HEAD ghi
branch refs/heads/feature
locked reason

worktree /tmp/prunable
HEAD jkl
branch refs/heads/old
prunable
"""
    entries = preflight.parse_worktrees_porcelain(text)

    assert entries[0].ignored is False
    assert all(entry.ignored for entry in entries[1:])


def test_preflight_passes_dirty_files_inside_tz_zones_and_collect_only_is_safe(tmp_path, monkeypatch):
    root = _prepare_root(tmp_path)
    tz = _tz(root)
    calls: dict[str, object] = {}

    def fake_git(_root: Path, *args: str) -> str:
        joined = " ".join(args)
        if joined == "rev-parse --abbrev-ref HEAD":
            return "main\n"
        if joined == "status --porcelain":
            return " M scripts/project_now.py\n?? tests/test_project_now.py\n"
        if joined == "worktree list --porcelain":
            return f"worktree {root}\nHEAD abc\nbranch refs/heads/main\n\nworktree /tmp/detached\nHEAD def\ndetached\n"
        return ""

    def fake_collect(_root: Path, test_cmd: str):
        calls["cmd"] = preflight.collect_only_command(test_cmd)
        return 0, "collected"

    monkeypatch.setattr(preflight, "_run_git", fake_git)
    monkeypatch.setattr(preflight, "_run_collect_only", fake_collect)

    ok, failures = preflight.run_preflight(root, tz)

    assert ok, failures
    assert "PYTHONDONTWRITEBYTECODE=1" in calls["cmd"]
    assert "PYTHONPATH=src" in calls["cmd"]
    assert "--collect-only" in calls["cmd"]


def test_preflight_blocks_dirty_file_outside_tz_zones(tmp_path, monkeypatch):
    root = _prepare_root(tmp_path)
    tz = _tz(root)
    monkeypatch.setattr(preflight, "_run_collect_only", lambda *_args: (_ for _ in ()).throw(AssertionError("no collect")))
    monkeypatch.setattr(
        preflight,
        "_run_git",
        lambda _root, *args: "main\n"
        if " ".join(args) == "rev-parse --abbrev-ref HEAD"
        else " M src/mango_mvp/channels/provider.py\n"
        if " ".join(args) == "status --porcelain"
        else f"worktree {root}\nbranch refs/heads/main\n"
        if " ".join(args) == "worktree list --porcelain"
        else "",
    )

    ok, failures = preflight.run_preflight(root, tz)

    assert not ok
    assert any("грязь вне зон" in failure for failure in failures)


def test_preflight_blocks_forbidden_zone_in_tz(tmp_path, monkeypatch):
    root = _prepare_root(tmp_path)
    tz = _tz(root, zones="scripts/, stable_runtime/")
    monkeypatch.setattr(preflight, "_run_git", lambda _root, *args: "main\n" if " ".join(args) == "rev-parse --abbrev-ref HEAD" else "")

    ok, failures = preflight.run_preflight(root, tz, run_collect=False)

    assert not ok
    assert any("запретный путь" in failure for failure in failures)


def test_preflight_blocks_unregistered_active_worktree(tmp_path, monkeypatch):
    root = _prepare_root(tmp_path)
    tz = _tz(root)
    monkeypatch.setattr(
        preflight,
        "_run_git",
        lambda _root, *args: "main\n"
        if " ".join(args) == "rev-parse --abbrev-ref HEAD"
        else ""
        if " ".join(args) == "status --porcelain"
        else f"worktree {root}\nbranch refs/heads/main\n\nworktree /tmp/unregistered\nbranch refs/heads/feature\n"
        if " ".join(args) == "worktree list --porcelain"
        else "",
    )

    ok, failures = preflight.run_preflight(root, tz, run_collect=False)

    assert not ok
    assert any("worktree вне реестра" in failure for failure in failures)


def test_preflight_blocks_missing_or_stale_project_now(tmp_path, monkeypatch):
    root = _prepare_root(tmp_path)
    tz = _tz(root)
    stale = root / "docs/PROJECT_NOW.md"
    old = (datetime.now() - timedelta(days=2)).timestamp()
    os.utime(stale, (old, old))
    monkeypatch.setattr(preflight, "_run_git", lambda _root, *args: "main\n" if " ".join(args) == "rev-parse --abbrev-ref HEAD" else "")

    ok, failures = preflight.run_preflight(root, tz, run_collect=False)

    assert not ok
    assert any("PROJECT_NOW" in failure for failure in failures)
