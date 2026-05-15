from __future__ import annotations

from dataclasses import replace

from mango_mvp.services.analyze import (
    ANALYZE_PROMPT_VERSION_COMPACT,
    ANALYZE_PROMPT_VERSION_FULL,
    AnalyzeService,
    SYSTEM_PROMPT_FULL,
)
from tests.test_dialogue_format import make_settings


class DummyCall:
    source_file = "dummy.mp3"
    source_filename = "dummy.mp3"
    started_at = None
    manager_name = "Иван"
    phone = "+79990000000"
    direction = "outbound"
    duration_sec = 180
    transcript_variants_json = None


def _service() -> AnalyzeService:
    return AnalyzeService(replace(make_settings(), analyze_provider="mock"))


def _structured_fields(
    *,
    budget: str | None = "50000",
    price_sensitivity: str | None = "high",
    discount_interest: bool | None = True,
    school: str | None = "школа №16",
    lead_priority: str | None = "hot",
) -> dict:
    return {
        "people": {"parent_fio": None, "child_fio": None},
        "contacts": {"email": None, "preferred_channel": None},
        "student": {"grade_current": "9 класс", "school": school},
        "interests": {
            "products": ["летний лагерь"],
            "format": [],
            "subjects": ["математика"],
            "exam_targets": [],
        },
        "commercial": {
            "price_sensitivity": price_sensitivity,
            "budget": budget,
            "discount_interest": discount_interest,
        },
        "objections": [],
        "next_step": {"action": None, "due": None},
        "lead_priority": lead_priority,
    }


def test_full_profile_user_prompt_includes_hints_section() -> None:
    service = _service()
    text = "MANAGER:\nРасскажите про летний лагерь.\n\nCLIENT:\nИнтересует математика для 9 класса."

    context = service._analysis_prompt_context(DummyCall(), text, "full")

    assert context["profile"] == "full"
    assert "Deterministic hints JSON" in context["user_prompt"]
    assert "subject_candidates" in context["user_prompt"]
    assert "target_product_candidates" in context["user_prompt"]


def test_system_prompt_full_v7_mentions_hints() -> None:
    assert ANALYZE_PROMPT_VERSION_FULL == "v7"
    assert ANALYZE_PROMPT_VERSION_COMPACT == "v6"
    assert "deterministic hints" in SYSTEM_PROMPT_FULL.lower()
    assert "Never invent facts from hints" in SYSTEM_PROMPT_FULL


def test_compose_history_summary_adds_commercial_school_priority_with_draft() -> None:
    service = _service()

    result = service._compose_history_summary(
        DummyCall(),
        draft_history_summary="Клиент интересуется летним лагерем по математике.",
        summary="Нужна программа лагеря.",
        structured_fields=_structured_fields(),
        objections=[],
        next_step_action=None,
        due=None,
        follow_up_reason=None,
    )

    assert "Школа: школа №16." in result
    assert "Коммерческий контекст: чувствительность к цене: высокая; бюджет: 50000; интересуется скидками." in result
    assert "Приоритет лида: горячий." in result


def test_compose_history_summary_adds_commercial_school_priority_without_draft() -> None:
    service = _service()

    result = service._compose_history_summary(
        DummyCall(),
        draft_history_summary=None,
        summary="Клиент спрашивал про летний лагерь.",
        structured_fields=_structured_fields(price_sensitivity="medium", lead_priority="warm"),
        objections=[],
        next_step_action=None,
        due=None,
        follow_up_reason=None,
    )

    assert "Школа: школа №16." in result
    assert "Коммерческий контекст: чувствительность к цене: средняя; бюджет: 50000; интересуется скидками." in result
    assert "Приоритет лида: теплый." in result


def test_compose_history_summary_skips_empty_budget_and_cold_priority() -> None:
    service = _service()

    result = service._compose_history_summary(
        DummyCall(),
        draft_history_summary=None,
        summary="Клиент спрашивал про курс.",
        structured_fields=_structured_fields(
            budget="не указан",
            price_sensitivity=None,
            discount_interest=False,
            school=None,
            lead_priority="cold",
        ),
        objections=[],
        next_step_action=None,
        due=None,
        follow_up_reason=None,
    )

    assert "Коммерческий контекст" not in result
    assert "бюджет: не указан" not in result.lower()
    assert "Приоритет лида" not in result
    assert "Школа:" not in result


def test_consecutive_client_yes_lines_are_preserved() -> None:
    service = _service()
    transcript = (
        "[00:00.1] Клиент: Да, да, да.\n"
        "[00:00.6] Клиент: Да, да, да.\n"
        "[00:01.0] Менеджер: Хорошо, хорошо.\n"
    )

    result = service._compact_transcript_for_prompt(transcript, "compact")

    assert result["transcript"].count("Клиент: Да") == 2
    assert "Менеджер: Хорошо" in result["transcript"]
    assert result["transcript_compaction_removed_lines"] == 0
    assert result["transcript_compaction_shortened_lines"] >= 2
