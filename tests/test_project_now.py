from __future__ import annotations

from pathlib import Path

from scripts import project_now


def _touch(path: Path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_project_now_writes_passport_from_tmp_project(tmp_path, monkeypatch):
    root = tmp_path / "project"
    _touch(root / "tasks/_running/TZ_running.md")
    _touch(root / "tasks/_inbox_codex/TZ_inbox.md")
    _touch(root / "tasks/_done/TZ_done.md")
    _touch(root / "tasks/_failed/TZ_failed.md")
    _touch(root / "audits/_inbox/audit_pack/implementation_notes.md")
    _touch(
        root / "docs/BLOCKERS.yaml",
        "blockers:\n  - what: wait for owner\n    owner: Дмитрий\n    since: 2026-06-16\n",
    )

    def fake_git(_root: Path, *args: str) -> str:
        joined = " ".join(args)
        if joined == "rev-parse --abbrev-ref HEAD":
            return "main"
        if joined == "rev-parse --short HEAD":
            return "abc1234"
        if joined == "status --short":
            return " M scripts/project_now.py"
        return ""

    monkeypatch.setattr(project_now, "_run_git", fake_git)
    monkeypatch.setattr(
        project_now,
        "_live_snapshot",
        lambda _root: {
            "status": "WARN",
            "processes": [
                {
                    "kind": "run_amo_wappi_draft_loop.py",
                    "pid": 42,
                    "head": "loaded123",
                    "head_source": "startup_manifest",
                    "worktree": str(root),
                    "worktree_head": "abc1234",
                    "env": {
                        "DRAFT_LOOP_AUTO_RESOLVER": "0",
                        "TELEGRAM_PILOT_MANAGER_CHAT_IDS": "123,456",
                    },
                    "warnings": ["head_drift"],
                }
            ],
        },
    )
    out = project_now.write_project_now(root)

    text = out.read_text(encoding="utf-8")
    assert "Сгенерирован:" in text
    assert "Ветка: `main`" in text
    assert "`TZ_running.md`" in text
    assert "`TZ_inbox.md`" in text
    assert "wait for owner" in text
    assert "## Клиентские live-каналы" in text
    assert "Статус проверки клиентских каналов" in text
    assert "loaded123" in text
    assert "head_drift" in text
    assert "TELEGRAM_PILOT_MANAGER_CHAT_IDS=[set]" in text
    assert "123,456" not in text


def test_project_now_does_not_touch_home_codex_or_stable_runtime(tmp_path, monkeypatch):
    root = tmp_path / "project"
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("CODEX_HOME", str(home / ".codex"))
    monkeypatch.setattr(project_now, "_run_git", lambda *_args: "")

    project_now.write_project_now(root)

    assert not (home / ".codex").exists()
    assert not (root / "stable_runtime").exists()


def test_project_now_cli_accepts_explicit_root_and_out(tmp_path, monkeypatch):
    root = tmp_path / "project"
    out = tmp_path / "PROJECT_NOW.md"
    monkeypatch.setattr(project_now, "_run_git", lambda *_args: "")

    assert project_now.main(["--root", str(root), "--out", str(out)]) == 0

    assert out.exists()
    assert "PROJECT_NOW" in out.read_text(encoding="utf-8")
