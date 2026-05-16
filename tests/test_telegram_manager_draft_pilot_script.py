from __future__ import annotations

import json

import pytest

from scripts.telegram_manager_draft_pilot import build_preview_service_from_env, run_dry_run, run_long_polling


def test_telegram_manager_draft_pilot_dry_run_builds_manager_payload(capsys) -> None:
    exit_code = run_dry_run(manager_chat_id="700100")
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["mode"] == "dry_run"
    assert payload["inbound_result"]["status"] == "accepted"
    assert payload["manager_deliveries"][0]["status"] == "ready_for_manager_chat"
    assert payload["manager_deliveries"][0]["telegram_api_called"] is False
    assert payload["safety"]["bot_polling"]["client_send"] is False
    assert payload["safety"]["manager_inbox"]["client_send"] is False


def test_telegram_manager_draft_pilot_long_polling_requires_explicit_confirmation() -> None:
    with pytest.raises(SystemExit, match="Long polling не запущен"):
        run_long_polling("")


def test_preview_service_from_env_uses_llm_when_enabled(monkeypatch) -> None:
    monkeypatch.setenv("TELEGRAM_PILOT_LLM_ENABLED", "1")
    monkeypatch.setenv("TELEGRAM_PILOT_CODEX_REASONING_EFFORT", "xhigh")

    service = build_preview_service_from_env()

    assert service is not None
    assert service.draft_provider.reasoning_effort == "xhigh"
