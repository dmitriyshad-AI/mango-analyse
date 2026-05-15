from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from mango_mvp.services import transcribe
from mango_mvp.services.transcribe import TranscribeService
from tests.test_dialogue_format import make_settings


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "transcribe_merge_corpus_z1"
ZERO_DROPS = {"count": 0, "total_chars": 0, "samples": []}


def _service(provider: str = "rule") -> TranscribeService:
    return TranscribeService(
        replace(
            make_settings(),
            dual_merge_provider=provider,
            dual_merge_similarity_threshold=1.1,
            llm_cache_enabled=False,
        )
    )


def _load_cases() -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for path in sorted(FIXTURE_DIR.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(payload, list), path
        cases.extend(payload)
    return cases


def test_merge_variant_pair_reports_suspicious_drops_without_changing_text() -> None:
    service = _service()
    primary = "Здравствуйте мусор мусор мусор мусор мусор мусор мусор мусор мусор стоимость курса"
    secondary = "Здравствуйте стоимость курса"

    result = service._merge_variant_pair(primary, secondary, speaker_label="Менеджер")

    assert result["text"] == service._merge_texts(primary, secondary)
    assert result["suspicious_drops"]["count"] > 0
    assert result["suspicious_drops"]["total_chars"] > 0
    assert result["suspicious_drops"]["samples"]


def test_suspicious_drops_counter_uses_normalized_tokens_and_autojunk_false(monkeypatch: pytest.MonkeyPatch) -> None:
    service = _service()
    original_sequence_matcher = transcribe.difflib.SequenceMatcher
    calls: list[dict[str, object]] = []

    def spy_sequence_matcher(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        calls.append(dict(kwargs))
        return original_sequence_matcher(*args, **kwargs)

    monkeypatch.setattr(transcribe.difflib, "SequenceMatcher", spy_sequence_matcher)

    result = service._count_suspicious_drops_in_merge(
        "Здравствуйте МУСОР мусор мусор мусор мусор мусор мусор мусор мусор стоимость курса",
        "Здравствуйте стоимость курса",
    )

    assert result["count"] == 1
    assert calls
    assert calls[0]["autojunk"] is False


def test_clean_merge_has_zero_suspicious_drops() -> None:
    result = _service()._merge_variant_pair(
        "Добрый день, подскажите класс ученика",
        "Добрый день, подскажите класс ребенка",
        speaker_label="Клиент",
    )

    assert result["suspicious_drops"] == ZERO_DROPS


def test_notes_remains_string_when_suspicious_drops_present() -> None:
    result = _service()._merge_variant_pair(
        "Здравствуйте мусор мусор мусор мусор мусор мусор мусор мусор мусор стоимость курса",
        "Здравствуйте стоимость курса",
        speaker_label="Менеджер",
    )

    assert isinstance(result["notes"], str)
    assert result["suspicious_drops"]["count"] > 0


@pytest.mark.parametrize(
    ("provider", "method"),
    [
        ("openai", "_merge_with_openai"),
        ("ollama", "_merge_with_ollama"),
        ("codex_cli", "_merge_with_codex_cli"),
    ],
)
def test_llm_success_paths_add_zero_suspicious_drops(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    method: str,
) -> None:
    service = _service(provider)

    def fake_merge(*, primary_text: str, secondary_text: str, speaker_label: str) -> dict[str, object]:
        return {
            "text": "llm merged",
            "selection": "MIX",
            "confidence": 0.9,
            "notes": "ok",
            "provider": provider,
        }

    monkeypatch.setattr(service, method, fake_merge)

    result = service._merge_variant_pair(
        "вариант A с мусором мусором мусором мусором мусором мусором мусором мусором",
        "вариант B нормальный",
        speaker_label="Менеджер",
    )

    assert result["provider"] == provider
    assert result["suspicious_drops"] == ZERO_DROPS
    assert isinstance(result["notes"], str)


def test_merge_meta_with_suspicious_drops_is_json_serializable() -> None:
    result = _service()._merge_variant_pair(
        "Здравствуйте мусор мусор мусор мусор мусор мусор мусор мусор мусор стоимость курса",
        "Здравствуйте стоимость курса",
        speaker_label="Менеджер",
    )

    json.dumps(result, ensure_ascii=False)


def test_merge_corpus_text_is_unchanged_by_suspicious_drop_diagnostics() -> None:
    service = _service()
    cases = _load_cases()

    assert len(cases) >= 20
    for case in cases:
        primary = str(case["primary"])
        secondary = str(case["secondary"])
        result = service._merge_variant_pair(primary, secondary, speaker_label="Менеджер")

        assert result["text"] == service._merge_texts(primary, secondary), case["case_id"]
        if case["expect_suspicious_drops"]:
            assert result["suspicious_drops"]["count"] > 0, case["case_id"]
        else:
            assert result["suspicious_drops"]["count"] == 0, case["case_id"]
