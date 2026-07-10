from __future__ import annotations

import json

from scripts.email_pipeline.summary import SummaryItem, build_summary_prompt


def test_summary_prompt_exposes_only_whitelisted_record_fields() -> None:
    prompt = build_summary_prompt(
        [
            SummaryItem(
                message_sha256="sha-1",
                direction="inbound",
                brand="foton",
                brand_source="internal_signal_should_not_leak",
                subject="Оплата",
                body="Клиент просит расписание.",
            )
        ]
    )

    payload = json.loads(prompt.rsplit("\n\n", 1)[1])
    record = payload["emails"][0]

    assert set(record) == {"message_sha256", "direction", "brand", "subject", "body"}
    assert "brand_source" not in prompt
    assert "_detected" not in prompt
    assert "_status" not in prompt


def test_summary_prompt_warns_not_to_turn_masks_or_subject_into_facts() -> None:
    prompt = build_summary_prompt(
        [
            SummaryItem(
                message_sha256="sha-1",
                direction="inbound",
                brand="foton",
                brand_source="content",
                subject="Оплата",
                body="Здравствуйте.",
            )
        ]
    )

    assert "не пиши 'данные скрыты'" in prompt
    assert "Тема письма (`subject`) — вспомогательный источник" in prompt
