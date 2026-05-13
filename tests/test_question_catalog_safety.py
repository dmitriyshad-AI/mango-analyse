from __future__ import annotations

import pytest

from mango_mvp.question_catalog.safety import assert_public_text_safe, guard_question_catalog_output_path, redact_public_text


def test_redact_public_text_removes_contacts_names_and_prices() -> None:
    text, flags = redact_public_text("Мария, цена 50 000 рублей, пишите на test@example.com или +7 900 123-45-67")

    assert "Мария" not in text
    assert "50 000" not in text
    assert "test@example.com" not in text
    assert "+7 900" not in text
    assert "актуальную стоимость" in text
    assert "price_redacted" in flags


def test_public_text_assertion_blocks_raw_contacts() -> None:
    with pytest.raises(ValueError, match="unsafe"):
        assert_public_text_safe("Позвоните +7 900 123-45-67")


def test_redact_public_text_removes_urls_and_long_service_numbers() -> None:
    text, flags = redact_public_text(
        "Посмотрите https://example.com/ticket?fn=7381440901039584 и счет 30101810400000000225"
    )

    assert "https" not in text
    assert "7381440901039584" not in text
    assert "30101810400000000225" not in text
    assert "url_redacted" in flags
    assert "long_number_redacted" in flags


def test_output_guard_rejects_stable_runtime(tmp_path) -> None:
    project = tmp_path / "project"
    stable = project / "stable_runtime" / "question_catalog"

    with pytest.raises(ValueError, match="stable_runtime"):
        guard_question_catalog_output_path(stable, project_root=project)

    assert guard_question_catalog_output_path(project / "product_data" / "question_catalog", project_root=project)
