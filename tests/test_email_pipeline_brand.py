import json
from pathlib import Path

import pytest

from scripts.email_pipeline.brand import infer_email_brand
from scripts.email_pipeline.pilot_100 import ensure_local_output_dir_allowed
from scripts.email_pipeline import summary as summary_module
from scripts.email_pipeline.summary import SummaryItem, mask_pii, summarize_items


def test_brand_uses_explicit_content_words() -> None:
    assert infer_email_brand("ЦДПО Фотон: вопрос", "").brand == "foton"
    assert infer_email_brand("УНПК МФТИ", "").brand == "unpk"


def test_brand_conflict_returns_none() -> None:
    result = infer_email_brand("Фотон и УНПК", "")
    assert result.brand == "none"
    assert result.brand_source == "none"


def test_kmipt_email_is_not_brand_signal() -> None:
    result = infer_email_brand("Вопрос", "Напишите на edu@kmipt.ru")
    assert result.brand == "none"
    assert result.brand_source == "none"


def test_course_links_are_content_signals() -> None:
    assert infer_email_brand("", "https://cdpofoton.ru/program").brand == "foton"
    assert infer_email_brand("", "https://kmipt.ru/courses/math").brand == "unpk"


def test_dates_are_last_resort_signal() -> None:
    assert infer_email_brand("", "смена 20-28 июня").brand == "foton"
    assert infer_email_brand("", "период 15-25 августа").brand == "unpk"


def test_mask_pii_handles_html_phone_fragments() -> None:
    masked = mask_pii("8 (800)&nbsp;123-45-67 hello@example.com Иванов И.И. Мария Иванова")
    assert "123-45-67" not in masked
    assert "hello@example.com" not in masked
    assert "Иванов И.И." not in masked
    assert "Мария Иванова" not in masked
    assert "[phone]" in masked
    assert "[email]" in masked
    assert "[name]" in masked


def test_mask_pii_preserves_business_numbers() -> None:
    text = (
        "Воскресенье 10:50-12:30, 2025-2026 уч.г., 8 класс, стоимость 126 000 руб. "
        "ИНН 7713010010, КПП 771301001, счет 40817810600010833998."
    )
    masked = mask_pii(text)
    assert "10:50-12:30" in masked
    assert "2025-2026" in masked
    assert "8 класс" in masked
    assert "126 000 руб" in masked
    assert "7713010010" in masked
    assert "771301001" in masked
    assert "40817810600010833998" in masked
    assert "[phone]" not in masked
    assert "[number]" not in masked


def test_mask_pii_still_masks_real_phones() -> None:
    masked = mask_pii("Телефон +7 (916) 123-45-67, запасной 8 916 765-43-21, хвост 123-45-67")
    assert "+7 (916) 123-45-67" not in masked
    assert "8 916 765-43-21" not in masked
    assert "123-45-67" not in masked
    assert masked.count("[phone]") == 3


def test_local_raw_output_must_be_git_ignored() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    ensure_local_output_dir_allowed(repo_root=repo_root, local_output_dir=repo_root / ".codex_local/email_pipeline")
    with pytest.raises(RuntimeError, match="not git-ignored"):
        ensure_local_output_dir_allowed(repo_root=repo_root, local_output_dir=repo_root / "email_pipeline_raw_outputs")
    with pytest.raises(RuntimeError, match="must not be written to Foton"):
        ensure_local_output_dir_allowed(
            repo_root=repo_root,
            local_output_dir=Path("/Users/dmitrijfabarisov/Claude Projects/Foton/email_pipeline_raw_outputs"),
        )


def test_summarize_items_repairs_missing_batch_row(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_call_codex_json(prompt: str, **_: object) -> dict[str, object]:
        payload = json.loads(prompt[prompt.index('{"emails":') :])
        shas = [row["message_sha256"] for row in payload["emails"]]
        calls.append(shas)
        if len(shas) == 3:
            shas = shas[:2]
        return {
            "summaries": [
                {
                    "message_sha256": sha,
                    "summary": f"Сводка {sha}",
                    "topic": "тест",
                    "next_step": None,
                    "confidence": 1.0,
                }
                for sha in shas
            ]
        }

    monkeypatch.setattr(summary_module, "_call_codex_json", fake_call_codex_json)
    items = [
        SummaryItem(
            message_sha256=f"sha-{index}",
            direction="inbound",
            brand="none",
            brand_source="none",
            subject="Тема",
            body="Текст",
        )
        for index in range(3)
    ]
    result = summarize_items(
        items,
        provider="codex_cli",
        model="gpt-5.5",
        reasoning="medium",
        batch_size=3,
        max_llm_calls=4,
        project_root=Path(__file__).resolve().parents[1],
    )

    assert result.llm_calls_total == 2
    assert set(result.summaries) == {"sha-0", "sha-1", "sha-2"}
    assert calls == [["sha-0", "sha-1", "sha-2"], ["sha-2"]]
