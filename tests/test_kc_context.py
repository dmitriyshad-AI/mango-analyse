from __future__ import annotations

from datetime import datetime, timezone

from mango_mvp.knowledge_base.fact_registry import FRESHNESS_METADATA_ONLY, FRESHNESS_UNKNOWN, FactSource, KnowledgeChunk
from mango_mvp.knowledge_base.kc_context import (
    SCHEDULE_SAFE_TEMPLATE,
    build_kc_context,
    build_kc_context_from_snapshot,
    limit_context_chunks,
    render_prompt_context,
)


def test_context_builder_limits_chunks() -> None:
    chunks = [
        KnowledgeChunk(
            chunk_id=f"kc_chunk:{index}",
            source_id="source:kc",
            title=f"Раздел {index}",
            text=("Стоимость и расписание курса. " * 40) + str(index),
            fact_types=("price", "schedule"),
        )
        for index in range(10)
    ]

    selected = limit_context_chunks(
        chunks,
        query="Какая стоимость и расписание?",
        required_fact_keys=("prices.current", "schedule.current"),
        max_chunks=3,
        max_chunk_chars=180,
        total_char_limit=420,
    )

    assert len(selected) == 3
    assert sum(len(chunk.text) for chunk in selected) <= 420
    assert all(len(chunk.text) <= 180 for chunk in selected)
    assert all(chunk.metadata.get("trimmed_for_prompt") is True for chunk in selected)


def test_schedule_missing_uses_safe_schedule_template() -> None:
    received_at = datetime(2026, 5, 16, 9, 30, tzinfo=timezone.utc)
    context = build_kc_context(
        message_text="Можно расписание на субботу или воскресенье?",
        chunks=[
            KnowledgeChunk(
                chunk_id="kc_chunk:schedule",
                source_id="source:kc",
                title="Расписание",
                text="Расписание пока уточняет менеджер.",
                fact_types=("schedule",),
            )
        ],
        sources=[
            FactSource(
                source_id="source:schedule_unknown",
                title="Расписание из базы знаний КЦ",
                source_kind="local_docx",
                fact_types=("schedule",),
                path="База знаний КЦ.docx",
                freshness_status=FRESHNESS_UNKNOWN,
            )
        ],
        received_at=received_at,
    )

    assert context.precise_answers_allowed is False
    assert context.safe_templates["schedule"] == SCHEDULE_SAFE_TEMPLATE
    assert context.manager_followup_required is True
    assert context.manager_followup_deadline == "2026-05-17T09:30:00+00:00"
    assert context.freshness_blocks[0]["reason"] == "schedule_source_not_ready"


def test_context_can_be_built_from_snapshot_and_renders_source_freshness() -> None:
    source = FactSource(
        source_id="source:drive_price",
        title="Drive price doc",
        source_kind="google_drive_doc",
        fact_types=("price",),
        google_drive_url="google-drive://price",
        freshness_status=FRESHNESS_METADATA_ONLY,
        read_status="metadata_only",
    )
    chunk = KnowledgeChunk(
        chunk_id="kc_chunk:price",
        source_id=source.source_id,
        title="Стоимость",
        text="Точную стоимость можно назвать только после проверки свежего прайс-документа.",
        fact_types=("price",),
        freshness_status=FRESHNESS_METADATA_ONLY,
        metadata={"source_title": source.title},
    )
    snapshot = {"sources": [source.to_json_dict()], "chunks": [chunk.to_json_dict()]}

    context = build_kc_context_from_snapshot(
        message_text="Сколько стоит обучение?",
        snapshot=snapshot,
        required_fact_keys=("prices.current",),
        max_chunks=3,
    )
    rendered = render_prompt_context(context)

    assert context.precise_answers_allowed is False
    assert context.freshness_blocks[0]["candidate_source_ids"] == ["source:drive_price"]
    assert "source=Drive price doc" in rendered
    assert "freshness=metadata_only" in rendered
