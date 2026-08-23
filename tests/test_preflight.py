from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

from scripts import preflight


def _write(path: Path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _tz(root: Path, zones: str = "docs/, tasks/") -> Path:
    path = root / "tasks/_running/TZ.md"
    _write(
        path,
        "\n".join(
            [
                "Ветка: main",
                f"Зоны: {zones}",
                "Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_project_now.py tests/test_audit_pack_pii.py tests/test_preflight.py",
                "Семантический-аудит: нет",
                "",
            ]
        ),
    )
    return path


def _code_tz(root: Path, zones: str = "scripts/preflight.py, tests/test_preflight.py") -> Path:
    path = _tz(root, zones)
    path.write_text(
        path.read_text(encoding="utf-8")
        + "Feature-ID: feature.preflight\nProblem-ID: problem.context\nИзменение: extend\n"
        + "Ключевые-символы: run_preflight\nКлючевые-слова: hard gate\n",
        encoding="utf-8",
    )
    return path


def _inventory_payload(*, decision: str = "extend") -> dict[str, object]:
    classification = "DONOR_REF" if decision == "port" else "ACTIVE_EXTEND"
    candidates: list[dict[str, object]] = [{
        "classification": classification,
        "source": "raw_symbol",
        "path": "scripts/preflight.py",
        "line": 1,
        "sha": "abc",
        "symbol": "run_preflight",
        "why_relevant": "owner",
        "verified_in_raw_source": True,
        "worktree": "/repo",
    }]
    owner: dict[str, str] = {"path": "scripts/preflight.py", "symbol": "run_preflight", "sha": "abc"}
    if decision == "new":
        candidates = [{"classification": "ABSENT_PROVEN", "verified_in_raw_source": True}]
        owner = {}
    return {
        "schema_version": "mango_prebuild_inventory_v1",
        "feature_id": "feature.preflight",
        "problem_id": "problem.context",
        "repo_head": "abc",
        "branch": "main",
        "worktree": "/repo",
        "status_fingerprint": "sha256:fingerprint",
        "graph_revision": "abc",
        "graph_matches_head": True,
        "generator_version": "mango_inventory_before_build_v2",
        "generator_command_sha256": "sha256:generator",
        "queries": ["feature.preflight", "problem.context", "run_preflight", "hard gate"],
        "coverage": {key: "completed" for key in preflight.INVENTORY_COVERAGE},
        "candidates": candidates,
        "decision": decision,
        "selected_owner": owner,
        "unresolved": [],
        "generated_at": "now",
    }


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
        if joined == "status --porcelain --untracked-files=all":
            return " M docs/PROJECT_NOW.md\n?? tasks/_running/note.md\n"
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


def test_preflight_rejects_non_pytest_command_without_running_it(tmp_path, monkeypatch):
    called = False

    def fake_run(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("unsafe command must not run")

    monkeypatch.setattr(preflight.subprocess, "run", fake_run)

    code, output = preflight._run_collect_only(tmp_path, "bash -c 'touch owned'")

    assert code == 2
    assert "unsafe test command" in output
    assert called is False

    external_code, external_output = preflight._run_collect_only(tmp_path, "/tmp/python3 -m pytest tests/")
    assert external_code == 2
    assert "внешний путь" in external_output
    assert called is False


def test_preflight_rejects_pythonpath_outside_project(tmp_path, monkeypatch):
    monkeypatch.setattr(
        preflight.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unsafe command must not run")),
    )

    code, output = preflight._run_collect_only(tmp_path, "PYTHONPATH=/tmp python3 -m pytest -q")

    assert code == 2
    assert "unsafe PYTHONPATH" in output


def test_preflight_rejects_external_pytest_target_and_plugin(tmp_path, monkeypatch):
    monkeypatch.setattr(
        preflight.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unsafe command must not run")),
    )

    external_code, external_output = preflight._run_collect_only(tmp_path, "python3 -m pytest /tmp/evil.py")
    plugin_code, plugin_output = preflight._run_collect_only(tmp_path, "python3 -m pytest -p evil tests/")

    assert external_code == 2
    assert "inside tests" in external_output
    assert plugin_code == 2
    assert "unsafe pytest option" in plugin_output

    equals_code, equals_output = preflight._run_collect_only(tmp_path, "python3 -m pytest /tmp/evil=1.py")
    assert equals_code == 2
    assert "inside tests" in equals_output

    write_code, write_output = preflight._run_collect_only(
        tmp_path,
        "python3 -m pytest --basetemp=/tmp/replace-me tests/",
    )
    assert write_code == 2
    assert "unsafe pytest option" in write_output


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
        if " ".join(args) == "status --porcelain --untracked-files=all"
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
        if " ".join(args) == "status --porcelain --untracked-files=all"
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


def test_preflight_blocks_old_or_exact_file_code_tz_without_inventory(tmp_path, monkeypatch):
    root = _prepare_root(tmp_path)
    tz = _code_tz(root, zones="scripts/preflight.py")
    monkeypatch.setattr(
        preflight,
        "_run_git",
        lambda _root, *args: "main\n" if " ".join(args) == "rev-parse --abbrev-ref HEAD" else f"worktree {root}\nbranch refs/heads/main\n" if " ".join(args) == "worktree list --porcelain" else "",
    )

    ok, failures = preflight.run_preflight(root, tz, run_collect=False)
    assert not ok
    assert "code-ТЗ требует --inventory" in failures

    legacy = _tz(root, zones="scripts/preflight.py")
    ok, failures = preflight.run_preflight(root, legacy, run_collect=False)
    assert not ok
    assert any("code-ТЗ без обязательных полей" in item for item in failures)

    hidden = _tz(root, zones="docs/")
    hidden.write_text(hidden.read_text(encoding="utf-8") + "\nИзменить `scripts/preflight.py`.\n", encoding="utf-8")
    ok, failures = preflight.run_preflight(root, hidden, run_collect=False)
    assert not ok
    assert any("code-ТЗ без обязательных полей" in item for item in failures)


def test_preflight_allows_docs_only_legacy_without_inventory(tmp_path, monkeypatch):
    root = _prepare_root(tmp_path)
    tz = _tz(root, zones="docs/, tasks/")
    monkeypatch.setattr(
        preflight,
        "_run_git",
        lambda _root, *args: "main\n" if " ".join(args) == "rev-parse --abbrev-ref HEAD" else f"worktree {root}\nbranch refs/heads/main\n" if " ".join(args) == "worktree list --porcelain" else "",
    )

    ok, failures = preflight.run_preflight(root, tz, run_collect=False)
    assert ok, failures


def test_inventory_validation_rejects_forgery_staleness_and_missing_owner(tmp_path, monkeypatch):
    header = preflight.parse_tz_header(_code_tz(tmp_path).read_text(encoding="utf-8"))
    supplied = _inventory_payload()
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps(supplied), encoding="utf-8")
    monkeypatch.setattr(preflight, "_refresh_inventory", lambda *_args: (supplied, None))
    assert preflight._validate_inventory(tmp_path, header, path) == []

    forged = dict(supplied)
    forged.pop("coverage")
    path.write_text(json.dumps(forged), encoding="utf-8")
    failures = preflight._validate_inventory(tmp_path, header, path)
    assert any("coverage" in item for item in failures)

    stale = dict(supplied)
    stale["status_fingerprint"] = "sha256:old"
    path.write_text(json.dumps(stale), encoding="utf-8")
    failures = preflight._validate_inventory(tmp_path, header, path)
    assert any("протух или подделан" in item for item in failures)

    no_owner = dict(supplied)
    no_owner["selected_owner"] = {}
    monkeypatch.setattr(preflight, "_refresh_inventory", lambda *_args: (no_owner, None))
    path.write_text(json.dumps(no_owner), encoding="utf-8")
    failures = preflight._validate_inventory(tmp_path, header, path)
    assert any("raw-подтверждённого owner" in item for item in failures)


def test_inventory_validation_requires_fresh_absence_for_new(tmp_path, monkeypatch):
    header = preflight.parse_tz_header(_code_tz(tmp_path).read_text(encoding="utf-8"))
    supplied = _inventory_payload(decision="new")
    supplied["graph_matches_head"] = False
    supplied["candidates"] = []
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps(supplied), encoding="utf-8")
    monkeypatch.setattr(preflight, "_refresh_inventory", lambda *_args: (supplied, None))

    failures = preflight._validate_inventory(tmp_path, header, path)
    assert any("ABSENT_PROVEN" in item for item in failures)


def test_inventory_refresh_rejects_tool_error_even_with_json_stdout(tmp_path, monkeypatch):
    header = preflight.parse_tz_header(_code_tz(tmp_path).read_text(encoding="utf-8"))
    monkeypatch.setattr(
        preflight.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 2, json.dumps(_inventory_payload()), "fatal"),
    )
    payload, error = preflight._refresh_inventory(tmp_path, header)
    assert payload is None
    assert "завершился с ошибкой" in (error or "")


def test_required_roles_are_computed_from_task_risk(tmp_path):
    text = _code_tz(tmp_path).read_text(encoding="utf-8")
    header = preflight.parse_tz_header(text)
    roles = {item["role"] for item in preflight.required_roles(header, "Рефакторинг Wappi клиентского черновика")}
    assert {"claude-code", "architect-auditor", "breaker", "cleaner", "business-auditor"} <= roles

    docs_header = preflight.parse_tz_header(_tz(tmp_path, zones="docs/").read_text(encoding="utf-8"))
    assert preflight.required_roles(docs_header, "Обновить внутренний runbook") == []
