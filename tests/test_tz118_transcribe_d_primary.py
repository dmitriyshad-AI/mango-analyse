from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import mango_mvp.services.transcribe as transcribe_module
from mango_mvp.config import get_settings
from mango_mvp.services.transcribe import TranscribeService


def _settings(tmp_path: Path, **overrides):
    get_settings.cache_clear()
    base = get_settings()
    values = {
        "database_url": f"sqlite:///{tmp_path / 'test.db'}",
        "llm_cache_enabled": False,
        "llm_cache_dir": str(tmp_path / "llm_cache"),
        "openai_api_key": None,
        "mono_role_assignment_mode": "off",
        "mono_role_assignment_min_confidence": 0.62,
        "mono_role_assignment_llm_threshold": 0.72,
        "codex_cli_command": "codex",
        "codex_cli_timeout_sec": 30,
        "codex_reasoning_effort": "medium",
    }
    values.update(overrides)
    return replace(base, **values)


def _turns() -> list[dict[str, object]]:
    return [
        {"start": 0.0, "text": "Алло."},
        {"start": 2.0, "text": "Сколько стоит курс по физике?"},
    ]


def test_d_primary_default_off_keeps_existing_mono_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    service = TranscribeService(_settings(tmp_path))

    def fail_codex(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("codex must not be called when role assignment is off")

    monkeypatch.setattr(service, "_assign_roles_with_codex", fail_codex)
    warnings: list[str] = []

    result = service._assign_roles_for_mono(_turns(), "Иван", warnings)

    assert result is None
    assert warnings == []


def test_d_primary_codex_selective_keeps_high_confidence_rule(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    service = TranscribeService(
        _settings(tmp_path, mono_role_assignment_mode="codex_selective")
    )

    def fail_codex(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("codex must not be called for high-confidence rule result")

    monkeypatch.setattr(service, "_assign_roles_with_codex", fail_codex)
    turns = [
        {
            "start": 0.0,
            "text": "Добрый день, учебный центр Фотон, меня зовут Иван.",
        },
        {"start": 4.0, "text": "Сколько стоит курс по физике?"},
    ]
    warnings: list[str] = []

    result = service._assign_roles_for_mono(turns, "Иван", warnings)

    assert result is not None
    assert result["meta"]["provider"] == "rule_high_conf"
    assert warnings == []


def test_d_primary_codex_selective_uses_model_only_for_low_confidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = TranscribeService(
        _settings(tmp_path, mono_role_assignment_mode="codex_selective")
    )
    turns = _turns()

    def fake_rule(self, turns, manager_name):  # noqa: ANN001
        return self._normalize_role_assignment_payload(  # noqa: SLF001
            {
                "roles": ["manager", "client"],
                "confidence": 0.50,
                "notes": "low rule confidence",
            },
            turns=turns,
            manager_name=manager_name,
            provider="rule",
        )

    def fake_codex(self, turns, manager_name):  # noqa: ANN001
        return self._normalize_role_assignment_payload(  # noqa: SLF001
            {
                "roles": ["manager", "client"],
                "confidence": 0.93,
                "notes": "synthetic",
                "rationale": "Первый ход похож на менеджера, второй на клиента.",
            },
            turns=turns,
            manager_name=manager_name,
            provider="codex_cli",
        )

    monkeypatch.setattr(TranscribeService, "_assign_roles_rule_based", fake_rule)
    monkeypatch.setattr(TranscribeService, "_assign_roles_with_codex", fake_codex)
    warnings: list[str] = []

    result = service._assign_roles_for_mono(turns, "Иван", warnings)

    assert result is not None
    assert result["meta"]["provider"] == "codex_cli"
    assert result["meta"]["low_info_filter_applied"] is True
    assert result["meta"]["low_info_filter_mode"] == "mark"
    assert result["meta"]["low_info_turn_indexes"] == [1]
    assert "segment_guard_applied" not in result["meta"]
    assert warnings == []


def test_d_primary_codex_cli_role_assignment_drops_openai_api_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = TranscribeService(_settings(tmp_path, openai_api_key="must_not_be_used"))
    turns = _turns()
    captured: dict[str, object] = {}

    monkeypatch.setenv("OPENAI_API_KEY", "secret-from-shell")
    monkeypatch.setenv("OPENAI_ORG_ID", "org-from-shell")
    monkeypatch.setenv("OPENAI_PROJECT", "project-from-shell")
    monkeypatch.setattr(transcribe_module.shutil, "which", lambda _cmd: "/usr/bin/codex")
    monkeypatch.setattr(
        TranscribeService,
        "_prepare_role_assignment_codex_home",
        lambda _self: str(tmp_path),
    )

    def fake_run(cmd, *, input, capture_output, text, check, timeout, env, cwd):  # noqa: ANN001
        captured["cmd"] = cmd
        captured["input"] = input
        captured["env"] = dict(env)
        captured["cwd"] = cwd
        out_path = Path(cmd[cmd.index("--output-last-message") + 1])
        out_path.write_text(
            json.dumps(
                {
                    "roles": ["manager", "client"],
                    "confidence": 0.91,
                    "notes": "synthetic",
                    "rationale": "synthetic role split",
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="tokens used 123")

    monkeypatch.setattr(transcribe_module.subprocess, "run", fake_run)

    result = service._assign_roles_with_codex(turns, "Иван")

    env = captured["env"]
    assert "OPENAI_API_KEY" not in env
    assert "OPENAI_ORG_ID" not in env
    assert "OPENAI_PROJECT" not in env
    assert captured["cwd"] == str(tmp_path)
    assert "--sandbox" in captured["cmd"]
    assert "read-only" in captured["cmd"]
    assert result["meta"]["provider"] == "codex_cli"
    assert result["meta"]["tokens_used_actual"] == 123
