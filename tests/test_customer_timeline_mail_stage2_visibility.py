from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mango_mvp.customer_timeline import BotContextChunk, CustomerIdentity, CustomerTimelineSQLiteStore, IdentityStatus
from mango_mvp.customer_timeline.ids import stable_digest
from mango_mvp.customer_timeline.store import json_dumps, json_loads, scrub_timeline_persisted_json
from mango_mvp.customer_timeline.mail_stage2_visibility import (
    assert_mail_stage2_visibility_gate,
    harden_mail_stage2_bot_visibility,
    mail_stage2_visibility_gate,
)


NOW = datetime(2026, 7, 3, 12, 0, tzinfo=timezone.utc)


def test_harden_mail_stage2_visibility_closes_chunks_and_preserves_diagnostics(tmp_path: Path) -> None:
    db = tmp_path / "staging" / "customer_timeline_staging.sqlite"
    db.parent.mkdir()
    store = CustomerTimelineSQLiteStore(db, allowed_root=tmp_path)
    try:
        store.upsert_customer(CustomerIdentity(tenant_id="foton", customer_id="customer:1", identity_status=IdentityStatus.STRONG))
        unsafe_seed = BotContextChunk(
            tenant_id="foton",
            customer_id="customer:1",
            source_system="mail_archive_stage2",
            source_ref="mail:a",
            chunk_type="email_message",
            text="Полный текст письма.",
            summary="Сводка письма.",
            event_at=NOW,
            allowed_for_bot=False,
            requires_manager_review=True,
            metadata={"bot_eligible_candidate": True, "bot_gate_reason": "usable_linked_qualified"},
            created_at=NOW,
        )
        store.upsert_bot_context_chunk(unsafe_seed)
        store.upsert_bot_context_chunk(
            BotContextChunk(
                tenant_id="foton",
                customer_id="customer:1",
                source_system="mail_archive_stage2",
                source_ref="mail:b",
                chunk_type="email_message",
                text="Уже закрыто.",
                summary="Закрытая сводка.",
                event_at=NOW,
                allowed_for_bot=False,
                requires_manager_review=True,
                metadata={"bot_eligible_candidate": False},
                created_at=NOW,
            )
        )
    finally:
        store.close()
    _force_legacy_unsafe_chunk(db, source_ref="mail:a")

    dry = harden_mail_stage2_bot_visibility(db, allowed_root=tmp_path, apply=False, allow_test_paths=True)
    assert dry["updated_chunks"] == 0
    assert dry["before"]["unsafe_active_chunks"] == 1
    assert dry["diagnostic_bot_eligible_candidates"] == 1

    report = harden_mail_stage2_bot_visibility(db, allowed_root=tmp_path, apply=True, allow_test_paths=True)
    assert report["updated_chunks"] == 1
    assert report["before"]["unsafe_active_chunks"] == 1
    assert report["after"]["unsafe_active_chunks"] == 0
    assert report["gate_passed"] is True
    assert_mail_stage2_visibility_gate(db, allowed_root=tmp_path, allow_test_paths=True)

    repeat = harden_mail_stage2_bot_visibility(db, allowed_root=tmp_path, apply=True, allow_test_paths=True)
    assert repeat["updated_chunks"] == 0
    assert repeat["before"]["unsafe_active_chunks"] == 0

    with sqlite3.connect(db) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            """
            SELECT allowed_for_bot, requires_manager_review, record_json
            FROM bot_context_chunks
            WHERE source_system = 'mail_archive_stage2'
            ORDER BY source_ref
            """
        ).fetchall()
    assert [(row["allowed_for_bot"], row["requires_manager_review"]) for row in rows] == [(0, 1), (0, 1)]
    payload = json.loads(rows[0]["record_json"])
    assert payload["allowed_for_bot"] is False
    assert payload["requires_manager_review"] is True
    assert payload["metadata"]["bot_eligible_candidate"] is True
    assert payload["metadata"]["bot_gate_reason"] == "usable_linked_qualified"

    counts = mail_stage2_visibility_gate(db, allowed_root=tmp_path, allow_test_paths=True)
    assert counts["flag_counts"] == {"allowed_0_review_1": 2}


def test_harden_mail_stage2_visibility_rejects_prod_like_path(tmp_path: Path) -> None:
    prod_like = tmp_path / "customer_timeline_prod_20260621" / "customer_timeline.sqlite"
    prod_like.parent.mkdir()
    prod_like.write_bytes(b"not sqlite")

    with pytest.raises(ValueError, match="refusing to harden prod"):
        harden_mail_stage2_bot_visibility(prod_like, allowed_root=tmp_path, apply=False, allow_test_paths=True)


def test_harden_mail_stage2_visibility_rejects_non_staging_path(tmp_path: Path) -> None:
    db = tmp_path / "regular" / "customer_timeline.sqlite"
    db.parent.mkdir()
    CustomerTimelineSQLiteStore(db, allowed_root=tmp_path).close()

    with pytest.raises(ValueError, match=r"\.codex_local/staging"):
        harden_mail_stage2_bot_visibility(db, allowed_root=tmp_path, apply=False)


def test_harden_mail_stage2_visibility_rejects_other_source_system(tmp_path: Path) -> None:
    db = tmp_path / "staging" / "customer_timeline_staging.sqlite"
    db.parent.mkdir()
    CustomerTimelineSQLiteStore(db, allowed_root=tmp_path).close()

    with pytest.raises(ValueError, match="only supports mail_archive_stage2"):
        harden_mail_stage2_bot_visibility(
            db,
            allowed_root=tmp_path,
            source_system="customer_timeline_bot_safe_summary",
            apply=False,
            allow_test_paths=True,
        )


def _force_legacy_unsafe_chunk(db: Path, *, source_ref: str) -> None:
    with sqlite3.connect(db) as con:
        con.row_factory = sqlite3.Row
        row = con.execute(
            "SELECT chunk_id, record_json FROM bot_context_chunks WHERE source_ref = ?",
            (source_ref,),
        ).fetchone()
        payload = json_loads(row["record_json"])
        payload["allowed_for_bot"] = True
        payload["requires_manager_review"] = False
        record_hash = stable_digest(scrub_timeline_persisted_json(payload))
        con.execute(
            """
            UPDATE bot_context_chunks
            SET allowed_for_bot = 1,
                requires_manager_review = 0,
                record_json = ?,
                record_hash = ?
            WHERE chunk_id = ?
            """,
            (json_dumps(payload), record_hash, row["chunk_id"]),
        )
        con.commit()
