from __future__ import annotations

import subprocess
from pathlib import Path

from mango_mvp.channels.subscription_llm import (
    CodexExecConfig,
    CodexExecDraftProvider,
    DraftGenerationResult,
    FakeDraftProvider,
    contains_bot_identity_disclosure,
    parse_llm_json,
)


def test_codex_exec_provider_builds_command_without_openai_key(tmp_path: Path) -> None:
    command = CodexExecConfig(model="gpt-5.5", reasoning_effort="medium").build_command(tmp_path / "out.txt")

    assert "OPENAI_API_KEY" not in " ".join(command)
    assert command[:2] == ["codex", "exec"]
    assert "--sandbox" in command
    assert "read-only" in command


def test_provider_parses_valid_json() -> None:
    result = parse_llm_json('{"route":"draft_for_manager","draft_text":"Здравствуйте! Уточним детали.","topic_id":"theme:001_pricing","topic_confidence":0.7}')

    assert result.route == "draft_for_manager"
    assert result.topic_id == "theme:001_pricing"


def test_provider_falls_back_on_invalid_json() -> None:
    result = parse_llm_json("not json")

    assert result.route == "manager_only"
    assert "llm_fallback" in result.safety_flags


def test_provider_timeout_returns_safe_fallback() -> None:
    def runner(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=1)

    result = CodexExecDraftProvider(runner=runner).generate("prompt")

    assert result.route == "manager_only"
    assert "codex_exec_timeout" in result.safety_flags


def test_draft_text_does_not_disclose_bot_identity() -> None:
    result = parse_llm_json('{"route":"draft_for_manager","draft_text":"Как ИИ я могу подсказать."}')

    assert result.route == "manager_only"
    assert "bot_identity_disclosure" in result.safety_flags
    assert contains_bot_identity_disclosure("Я бот и нейросеть")
    for phrase in ("я бот", "как ИИ", "нейросеть", "искусственный интеллект", "GPT", "Claude", "Codex"):
        assert contains_bot_identity_disclosure(f"Тест: {phrase}")


def test_fake_provider_records_prompt() -> None:
    provider = FakeDraftProvider(DraftGenerationResult(route="draft_for_manager", draft_text="Здравствуйте!"))

    result = provider.generate("prompt")

    assert result.draft_text == "Здравствуйте!"
    assert provider.prompts == ["prompt"]
