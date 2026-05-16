from __future__ import annotations

import json
from pathlib import Path

import pytest

from mango_mvp.question_catalog.classifier import QuestionClassifierConfig, classify_question
from mango_mvp.question_catalog.theme_assigner_llm import (
    ThemeAssignmentResult,
    ThemeAssignerConfig,
    ThemeAssignerError,
    assign_theme_llm,
)
from mango_mvp.services.llm_response_cache import LLMResponseCache


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeChoice:
    def __init__(self, content: str):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, payloads: list[dict[str, object]]):
        self.payloads = list(payloads)
        self.calls = 0

    def create(self, **_kwargs: object) -> _FakeResponse:
        self.calls += 1
        if not self.payloads:
            raise AssertionError("unexpected LLM call")
        payload = self.payloads.pop(0)
        return _FakeResponse(json.dumps(payload, ensure_ascii=False))


class _FakeChat:
    def __init__(self, completions: _FakeCompletions):
        self.completions = completions


class _FakeClient:
    def __init__(self, payloads: list[dict[str, object]]):
        self.completions = _FakeCompletions(payloads)
        self.chat = _FakeChat(self.completions)


def test_llm_returns_valid_theme_id_only(tmp_path: Path) -> None:
    client = _FakeClient(
        [
            {
                "theme_id": "theme:001_pricing",
                "confidence": 0.95,
                "reasoning": "Клиент явно спрашивает стоимость.",
            }
        ]
    )
    cache = LLMResponseCache(enabled=True, root_dir=tmp_path / "cache")

    result = assign_theme_llm(
        "Сколько стоит ЕГЭ по математике?",
        {"product": "регулярный_курс", "subject": "математика"},
        config=ThemeAssignerConfig(cache_root_dir=tmp_path / "cache", enable_escalation=False),
        client=client,
        cache=cache,
    )

    assert result.theme_id == "theme:001_pricing"
    assert result.confidence == 0.95
    assert result.model == "gpt-4o-mini"
    assert result.cache_hit is False
    assert client.completions.calls == 1


def test_llm_rejects_unknown_theme_id(tmp_path: Path) -> None:
    client = _FakeClient([{"theme_id": "theme:999_fake", "confidence": 0.95, "reasoning": "bad"}])

    with pytest.raises(ThemeAssignerError):
        assign_theme_llm(
            "Сколько стоит?",
            {},
            config=ThemeAssignerConfig(cache_root_dir=tmp_path / "cache", enable_escalation=False),
            client=client,
            cache=LLMResponseCache(enabled=True, root_dir=tmp_path / "cache"),
        )


def test_llm_low_confidence_falls_back_to_rule() -> None:
    def low_confidence_assigner(*_args: object, **_kwargs: object) -> ThemeAssignmentResult:
        return ThemeAssignmentResult(
            theme_id="theme:012_certificates",
            confidence=0.2,
            reasoning="Не уверен.",
            model="fake-low",
        )

    result = classify_question(
        "Сколько стоит подготовка к ЕГЭ?",
        config=QuestionClassifierConfig(llm_enabled=True),
        llm_assigner=low_confidence_assigner,
    )

    assert result.theme_id == "theme:001_pricing"
    assert result.classification_method == "llm_low_confidence_rule_fallback"
    assert result.llm_model == "fake-low"


def test_llm_error_falls_back_to_rule() -> None:
    def failing_assigner(*_args: object, **_kwargs: object) -> ThemeAssignmentResult:
        raise RuntimeError("network timeout")

    result = classify_question(
        "Не пришла ссылка на личный кабинет",
        config=QuestionClassifierConfig(llm_enabled=True),
        llm_assigner=failing_assigner,
    )

    assert result.theme_id == "theme:025_missing_links_access"
    assert result.classification_method == "llm_error_rule_fallback"
    assert "network timeout" in result.reasoning


def test_cache_hit_skips_llm_call(tmp_path: Path) -> None:
    cache = LLMResponseCache(enabled=True, root_dir=tmp_path / "cache")
    config = ThemeAssignerConfig(cache_root_dir=tmp_path / "cache", enable_escalation=False)
    first_client = _FakeClient(
        [
            {
                "theme_id": "theme:013_schedule",
                "confidence": 0.91,
                "reasoning": "Клиент спрашивает расписание.",
            }
        ]
    )

    first = assign_theme_llm(
        "Когда занятия?",
        {},
        config=config,
        client=first_client,
        cache=cache,
    )
    second_client = _FakeClient([])
    second = assign_theme_llm(
        "Когда занятия?",
        {},
        config=config,
        client=second_client,
        cache=cache,
    )

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.theme_id == "theme:013_schedule"
    assert first_client.completions.calls == 1
    assert second_client.completions.calls == 0
