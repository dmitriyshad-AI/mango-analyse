from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from mango_mvp.customer_timeline import BotContextChunk, CustomerIdentity, CustomerTimelineSQLiteStore, TimelineEvent
from mango_mvp.customer_timeline.stage4b_bot_opening import (
    STAGE4B_OPENING_POLICY_VERSION,
    Stage4BBotOpeningConfig,
    run_stage4b_bot_opening,
)


NOW = datetime(2026, 7, 3, 12, 0, tzinfo=timezone.utc)


def test_stage4b_opens_only_linked_non_empty_mail_chunks_and_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        customer = CustomerIdentity(
            tenant_id="foton",
            identity_status="strong",
            customer_id="customer:known",
            primary_email="client@example.com",
            created_at=NOW,
            updated_at=NOW,
        )
        store.upsert_customer(customer)
        open_event = _mail_event(customer, "open", "Клиент спрашивает расписание и стоимость Фотона.")
        empty_event = _mail_event(customer, "empty", "Пустое письмо.")
        hidden_event = _mail_event(customer, "hidden", "Старое письмо.")
        store.upsert_event(open_event)
        store.upsert_event(empty_event)
        store.upsert_event(hidden_event)
        store.upsert_bot_context_chunk(_mail_chunk(open_event, text="Фотон. Расписание: суббота 12.15-14.15, цена 59 000 руб."))
        store.upsert_bot_context_chunk(_mail_chunk(empty_event, text="Временно непустой текст."))
        store.upsert_bot_context_chunk(_mail_chunk(hidden_event, text="Этот чанк будет superseded."))
        store._con.execute(  # noqa: SLF001 - test fixture creates historical empty text.
            "UPDATE bot_context_chunks SET record_json = json_set(record_json, '$.text', '') WHERE event_id = ?",
            (empty_event.event_id,),
        )
        store._con.execute(  # noqa: SLF001 - test fixture creates historical superseded chunk.
            "UPDATE bot_context_chunks SET superseded_by = ? WHERE event_id = ?",
            (open_event.event_id, hidden_event.event_id),
        )
        store._con.commit()  # noqa: SLF001

    config = Stage4BBotOpeningConfig(
        timeline_db_path=db_path,
        allowed_root=tmp_path,
        out_dir=tmp_path / "out",
        apply=True,
        allow_test_paths=True,
    )
    first = run_stage4b_bot_opening(config)
    second = run_stage4b_bot_opening(config)

    assert first["plan"]["candidate_chunks"] == 1
    assert first["apply"]["chunks_updated"] == 1
    assert first["after"]["mail_stage2_chunks_bot_visible"] == 1
    assert first["final_checks"]["mail_stage2_review_violations_after"] == 0
    assert second["plan"]["already_open"] == 1
    assert second["apply"]["chunks_updated"] == 0

    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        rows = {
            row["event_id"]: row
            for row in con.execute(
                "SELECT event_id, allowed_for_bot, requires_manager_review, superseded_by, record_json FROM bot_context_chunks"
            )
        }
    opened = rows[open_event.event_id]
    payload = json.loads(opened["record_json"])
    assert opened["allowed_for_bot"] == 1
    assert opened["requires_manager_review"] == 0
    assert payload["metadata"]["memory_status"] == "usable_memory"
    assert payload["metadata"]["client_safe"] is False
    assert payload["metadata"]["bot_memory_allowed"] is True
    assert payload["metadata"]["bot_memory_policy_version"] == STAGE4B_OPENING_POLICY_VERSION
    assert "foton" in payload["metadata"]["sensitivity_tags"]
    assert "money" in payload["metadata"]["sensitivity_tags"]
    assert "schedule" in payload["metadata"]["sensitivity_tags"]
    assert rows[empty_event.event_id]["allowed_for_bot"] == 0
    assert rows[empty_event.event_id]["requires_manager_review"] == 1
    assert rows[hidden_event.event_id]["allowed_for_bot"] == 0
    assert rows[hidden_event.event_id]["requires_manager_review"] == 1


def test_stage4b_refuses_non_staging_path_without_test_override(tmp_path: Path) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path).close()

    config = Stage4BBotOpeningConfig(
        timeline_db_path=db_path,
        allowed_root=tmp_path,
        out_dir=tmp_path / "out",
        apply=True,
    )

    try:
        run_stage4b_bot_opening(config)
    except ValueError as exc:
        assert ".codex_local/staging" in str(exc)
    else:  # pragma: no cover - defensive assertion.
        raise AssertionError("stage4b opening accepted a non-staging path")


def test_stage4b_refuses_nested_fake_staging_path(tmp_path: Path) -> None:
    db_path = tmp_path / ".codex_local" / "foo" / "staging" / "customer_timeline.sqlite"
    CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path).close()

    config = Stage4BBotOpeningConfig(
        timeline_db_path=db_path,
        allowed_root=tmp_path,
        out_dir=tmp_path / "out",
        apply=True,
    )

    try:
        run_stage4b_bot_opening(config)
    except ValueError as exc:
        assert ".codex_local/staging" in str(exc)
    else:  # pragma: no cover - defensive assertion.
        raise AssertionError("stage4b opening accepted a nested fake staging path")


def _mail_event(customer: CustomerIdentity, suffix: str, summary: str) -> TimelineEvent:
    return TimelineEvent(
        tenant_id="foton",
        customer_id=customer.customer_id,
        event_type="email_message",
        event_at=NOW,
        source_system="mail_archive_stage2",
        source_id=f"{suffix:0<64}"[:64],
        direction="inbound",
        summary=summary,
        text_preview=summary,
        match_status="strong_unique",
        created_at=NOW,
        record={"message_sha256": f"{suffix:0<64}"[:64]},
    )


def _mail_chunk(event: TimelineEvent, *, text: str) -> BotContextChunk:
    return BotContextChunk(
        tenant_id=event.tenant_id,
        customer_id=event.customer_id or "",
        event_id=event.event_id,
        source_ref=event.source_ref,
        source_system=event.source_system,
        chunk_type="email_message",
        text=text,
        summary=event.summary or "",
        event_at=event.event_at,
        allowed_for_bot=False,
        requires_manager_review=True,
        created_at=event.created_at,
    )
