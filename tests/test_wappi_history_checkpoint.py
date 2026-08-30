"""Wappi large-history fetch checkpoint: resume from the last CONFIRMED chat.

Every test drives the production entry point ``run_wappi_history_import`` with a
fake read-only Wappi client. No live Wappi call, no real message body.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

import pytest

import mango_mvp.customer_timeline.wappi_history_import as wappi_history_module
from mango_mvp.customer_timeline.contracts import (
    CustomerIdentity,
    IdentityLink,
    IdentityMatchClass,
    IdentityStatus,
)
from mango_mvp.customer_timeline.store import CustomerTimelineSQLiteStore
from mango_mvp.customer_timeline.wappi_history_import import (
    WAPPI_HISTORY_CHECKPOINT_SCHEMA_VERSION,
    WappiFetchLimits,
    WappiHistoryImportConfig,
    WappiProfileSpec,
    load_wappi_history_checkpoint,
    run_wappi_history_import,
    usable_wappi_checkpoint_profiles,
    wappi_checkpoint_anchor,
    wappi_checkpoint_token,
    wappi_fetch_universe_fingerprint,
    wappi_history_checkpoint_path,
    wappi_timeline_state,
)
from mango_mvp.integrations.amo_wappi_phase1 import AmoWappiHttpError
from mango_mvp.integrations.amo_wappi_transport import DefaultDenyTransport, SafeTransportPolicy


class CheckpointFakeClient:
    """Minimal WappiHistoryClient: offset/limit slicing plus injectable failures."""

    def __init__(
        self,
        chats: Mapping[str, list[Mapping[str, Any]]],
        messages: Mapping[tuple[str, str, str], list[Mapping[str, Any]]],
    ) -> None:
        self.transport = DefaultDenyTransport(
            lambda **_kwargs: {"ok": True},
            policy=SafeTransportPolicy.wappi_read_only(),
        )
        self.chats = {key: list(value) for key, value in chats.items()}
        self.messages = {key: list(value) for key, value in messages.items()}
        self.chat_calls: list[tuple[str, int, int]] = []
        self.message_calls: list[tuple[str, str, int]] = []
        self.message_orders: list[str] = []
        self.message_request_calls: list[tuple[str, str, int, int, str]] = []
        self.fail_catalog_from_offset: Optional[int] = None
        self.fail_message_at: Optional[tuple[str, int]] = None

    def list_chats(
        self, *, channel: str, profile_id: str, limit: int = 50, offset: int = 0,
        order: str = "desc", show_all: bool = False,
    ) -> Mapping[str, Any]:
        self.chat_calls.append((profile_id, offset, limit))
        if self.fail_catalog_from_offset is not None and offset >= self.fail_catalog_from_offset:
            raise AmoWappiHttpError("HTTP 502: upstream unavailable")
        items = self.chats.get(profile_id, [])
        return {"dialogs": items[offset : offset + limit], "total_count": len(items)}

    def get_chat_messages(
        self, *, channel: str, profile_id: str, chat_id: str, limit: int = 50, offset: int = 0,
        order: str = "desc", mark_all: bool = False,
    ) -> Mapping[str, Any]:
        self.message_calls.append((profile_id, chat_id, offset))
        self.message_orders.append(order)
        self.message_request_calls.append((profile_id, chat_id, offset, limit, order))
        if self.fail_message_at is not None and (chat_id, offset) == self.fail_message_at:
            raise AmoWappiHttpError("HTTP 503: service unavailable")
        items = self.messages.get((channel, profile_id, chat_id), [])
        if order == "desc":
            items = list(reversed(items))
        return {"messages": items[offset : offset + limit]}


def build_universe(
    total_chats: int, *, messages_per_chat: int = 1, profile_id: str = "p-tg",
    channel: str = "telegram", body: str = "Здравствуйте",
) -> tuple[list[Mapping[str, Any]], dict[tuple[str, str, str], list[Mapping[str, Any]]]]:
    chats = [
        {"id": f"c{index:04d}", "type": "user", "last_timestamp": 1_753_000_000 + messages_per_chat - 1}
        for index in range(total_chats)
    ]
    messages: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for chat in chats:
        chat_id = str(chat["id"])
        messages[(channel, profile_id, chat_id)] = [
            {
                "id": f"{chat_id}-m{number:03d}",
                "chat_id": chat_id,
                "type": "text",
                "body": f"{body} {number}",
                "time": 1_753_000_000 + number,
            }
            for number in range(messages_per_chat)
        ]
    return chats, messages


def write_phase1_config(tmp_path: Path) -> Path:
    path = tmp_path / "amo_wappi_phase1.json"
    path.write_text(
        json.dumps(
            {
                "profiles": {
                    "p-tg": {"brand": "foton", "channel": "telegram", "label": "Foton Telegram"},
                    "p-max": {"brand": "unpk", "channel": "max", "label": "UNPK Max"},
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def make_config(
    tmp_path: Path, *, db_path: Path, phase1: Path, checkpoint_dir: Optional[Path],
    request_limit_total: int = 100_000, page_size: int = 10, apply: bool = True,
) -> WappiHistoryImportConfig:
    return WappiHistoryImportConfig(
        timeline_db=db_path,
        allowed_root=tmp_path,
        phase1_config=phase1,
        pairs_file=None,
        auto_pairs_file=None,
        apply=apply,
        checkpoint_dir=checkpoint_dir,
        limits=WappiFetchLimits(
            page_size=page_size,
            request_limit_total=request_limit_total,
            complete_message_history=True,
            sleep_seconds=0,
        ),
    )


def install_exact_widget_identity(
    tmp_path: Path,
    *,
    db_path: Path,
    contact_id: str,
    customer_id: str,
    link_db: Optional[Path] = None,
) -> Path:
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        store.upsert_customer(
            CustomerIdentity(
                tenant_id="foton",
                customer_id=customer_id,
                identity_status=IdentityStatus.STRONG,
                source_ref=f"amocrm:contact:{contact_id}",
            ),
            actor="test",
        )
        store.upsert_identity_link(
            IdentityLink(
                tenant_id="foton",
                customer_id=customer_id,
                link_type="amo_contact_id",
                link_value=contact_id,
                source_system="amocrm_snapshot",
                source_ref=f"amocrm:contact:{contact_id}",
                match_class=IdentityMatchClass.STRONG_UNIQUE,
                confidence=1.0,
            ),
            actor="test",
        )
    resolved_link_db = link_db or tmp_path / "wappi_amo_links.sqlite"
    with sqlite3.connect(resolved_link_db) as con:
        wappi_history_module._ensure_wappi_widget_link_schema(con)
        con.execute(
            "INSERT INTO wappi_amo_links "
            "(channel,profile_id,chat_id,contact_id,lead_ids_json,status,checked_at,"
            "response_sha256,resolution_source,last_timestamp,matched_points) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(channel,profile_id,chat_id) DO UPDATE SET "
            "contact_id=excluded.contact_id,status=excluded.status,"
            "response_sha256=excluded.response_sha256,"
            "resolution_source=excluded.resolution_source",
            (
                "telegram", "p-tg", "c0000", contact_id, "[]", "resolved",
                "2026-08-29T00:00:00+00:00", f"proof-{contact_id}",
                "wappi_widget", 1_753_000_000, 0,
            ),
        )
        con.commit()
    return resolved_link_db


def wappi_row_count(db_path: Path) -> int:
    with sqlite3.connect(db_path) as con:
        return int(
            con.execute(
                "SELECT COUNT(*) FROM timeline_events WHERE source_system LIKE 'wappi_%'"
            ).fetchone()[0]
        )


def active_wappi_row_count(db_path: Path) -> int:
    with sqlite3.connect(db_path) as con:
        return int(
            con.execute(
                "SELECT COUNT(*) FROM timeline_events "
                "WHERE source_system LIKE 'wappi_%' AND superseded_by IS NULL"
            ).fetchone()[0]
        )


def read_checkpoint(checkpoint_dir: Path) -> Mapping[str, Any]:
    return load_wappi_history_checkpoint(checkpoint_dir)


def prepare(tmp_path: Path) -> tuple[Path, Path, Path]:
    db_path = tmp_path / "customer_timeline.sqlite"
    CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path).close()
    checkpoint_dir = tmp_path / "checkpoints"
    return db_path, write_phase1_config(tmp_path), checkpoint_dir


def test_source_lifecycle_rejects_non_wappi_sources(tmp_path: Path) -> None:
    db_path = tmp_path / "timeline.sqlite"
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        with pytest.raises(ValueError, match="restricted to Wappi"):
            store.set_timeline_source_records_active(
                "foton",
                source_records={"amocrm_event": ("1",)},
                active=False,
                retirement_marker="retired:wappi_expected_excluded:0123456789abcdef",
                retirement_reason="non_personal_chat",
            )


def test_source_lifecycle_does_not_restore_similar_foreign_marker(tmp_path: Path) -> None:
    db_path, _phase1, _checkpoint_dir = prepare(tmp_path)
    with sqlite3.connect(db_path) as con:
        con.execute(
            "INSERT INTO timeline_events "
            "(event_id,dedupe_key,tenant_id,event_type,source_system,source_id,direction,"
            "match_status,event_at,importance,created_at,record_json,record_hash,superseded_by) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "event:foreign-marker", "dedupe:foreign-marker", "foton", "message",
                "wappi_telegram", "message:1", "system", "strong_unique",
                "2026-08-27T00:00:00+00:00", 0, "2026-08-27T00:00:00+00:00", "{}", "hash",
                "retired:wappi_expected_excluded:not-hex",
            ),
        )
        con.commit()
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        report = store.set_timeline_source_records_active(
            "foton",
            source_records={"wappi_telegram": ("message:1",)},
            active=True,
            retirement_marker="retired:wappi_expected_excluded:0123456789abcdef",
            retirement_reason="reclassified_personal",
        )
    with sqlite3.connect(db_path) as con:
        marker = con.execute(
            "SELECT superseded_by FROM timeline_events WHERE event_id='event:foreign-marker'"
        ).fetchone()[0]

    assert report["changed_events"] == 0
    assert marker == "retired:wappi_expected_excluded:not-hex"


def test_delta_catalog_retires_and_restores_all_chat_history_without_message_reads(
    tmp_path: Path,
) -> None:
    db_path, phase1, checkpoint_dir = prepare(tmp_path)
    chats, messages = build_universe(1, messages_per_chat=2)
    config = make_config(
        tmp_path,
        db_path=db_path,
        phase1=phase1,
        checkpoint_dir=checkpoint_dir,
    )

    initial = run_wappi_history_import(
        config,
        client=CheckpointFakeClient({"p-tg": chats, "p-max": []}, messages),
    )

    assert initial["validation_ok"] is True
    assert wappi_row_count(db_path) == 2
    assert active_wappi_row_count(db_path) == 2

    non_personal = [{**chats[0], "type": "group"}]
    excluded_client = CheckpointFakeClient({"p-tg": non_personal, "p-max": []}, messages)
    excluded = run_wappi_history_import(config, client=excluded_client)

    assert excluded_client.message_calls == []
    assert wappi_row_count(db_path) == 2
    assert active_wappi_row_count(db_path) == 0
    assert excluded["source_lifecycle"]["current_ledger_expected_excluded_records"] == 0
    assert excluded["source_lifecycle"]["catalog_non_personal_source_records"] == 2
    assert excluded["source_lifecycle"]["retired_events"] == 2
    assert excluded["source_lifecycle"]["non_personal_active_after"] == 0
    assert excluded["source_lifecycle"]["complete"] is True
    with sqlite3.connect(db_path) as con:
        markers = {str(row[0]) for row in con.execute("SELECT superseded_by FROM timeline_events")}
    assert len(markers) == 1
    assert next(iter(markers)).startswith("retired:wappi_expected_excluded:")

    restored_client = CheckpointFakeClient({"p-tg": chats, "p-max": []}, messages)
    restored = run_wappi_history_import(config, client=restored_client)

    assert restored_client.message_calls == []
    assert restored["source_lifecycle"]["restored_events"] == 2
    assert restored["history_validation"]["mode"] == "incremental_catalog"
    assert restored["history_validation"]["full_audit_completed_this_run"] is False
    assert active_wappi_row_count(db_path) == 2
    with sqlite3.connect(db_path) as con:
        restored_rows = con.execute(
            "SELECT customer_id,match_status,"
            "json_extract(record_json,'$.resolution_reason'),"
            "json_extract(record_json,'$.metadata.identity_authority'),"
            "json_extract(record_json,'$.metadata.pending_attribution') "
            "FROM timeline_events ORDER BY source_id"
        ).fetchall()
        active_bot_chunks = con.execute(
            "SELECT count(*) FROM bot_context_chunks "
            "WHERE source_system IN ('wappi_telegram','wappi_max') "
            "AND coalesce(superseded_by,'')=''"
        ).fetchone()[0]
    assert restored_rows == [
        (
            None,
            "unmatched",
            "wappi_catalog_reclassified_personal_pending_attribution",
            "pending_attribution",
            1,
        ),
        (
            None,
            "unmatched",
            "wappi_catalog_reclassified_personal_pending_attribution",
            "pending_attribution",
            1,
        ),
    ]
    assert active_bot_chunks == 0

    repeated_client = CheckpointFakeClient({"p-tg": chats, "p-max": []}, messages)
    repeated = run_wappi_history_import(config, client=repeated_client)
    assert repeated_client.message_calls == []
    assert repeated["source_lifecycle"]["restored_events"] == 0
    assert repeated["source_lifecycle"]["retired_events"] == 0
    assert wappi_row_count(db_path) == active_wappi_row_count(db_path) == 2

    changed_non_personal = [
        {
            **chats[0],
            "type": "group",
            "last_timestamp": int(chats[0]["last_timestamp"]) + 10,
        }
    ]
    changed_excluded = run_wappi_history_import(
        config,
        client=CheckpointFakeClient(
            {"p-tg": changed_non_personal, "p-max": []}, messages
        ),
    )
    assert changed_excluded["source_lifecycle"]["retired_events"] == 2
    assert active_wappi_row_count(db_path) == 0

    messages[("telegram", "p-tg", "c0000")].append(
        {
            "id": "c0000-m002",
            "chat_id": "c0000",
            "type": "text",
            "body": "Снова личный",
            "time": 1_753_000_020,
        }
    )
    changed_personal = [
        {**chats[0], "last_timestamp": int(chats[0]["last_timestamp"]) + 20}
    ]
    changed_client = CheckpointFakeClient(
        {"p-tg": changed_personal, "p-max": []}, messages
    )
    changed_restored = run_wappi_history_import(config, client=changed_client)

    assert changed_restored["publish_ready"] is True
    assert {
        order
        for profile, chat, _offset, _limit, order in changed_client.message_request_calls
        if profile == "p-tg" and chat == "c0000"
    } == {"desc"}
    assert wappi_row_count(db_path) == active_wappi_row_count(db_path) == 3


def test_missing_catalog_chat_blocks_lifecycle_pass_without_retiring_history(
    tmp_path: Path,
) -> None:
    db_path, phase1, checkpoint_dir = prepare(tmp_path)
    chats, messages = build_universe(2)
    config = make_config(
        tmp_path,
        db_path=db_path,
        phase1=phase1,
        checkpoint_dir=checkpoint_dir,
    )
    initial = run_wappi_history_import(
        config,
        client=CheckpointFakeClient({"p-tg": chats, "p-max": []}, messages),
    )
    assert initial["publish_ready"] is True

    checkpoint = dict(read_checkpoint(checkpoint_dir))
    profiles = {key: dict(value) for key, value in checkpoint["profiles"].items()}
    tg_state = profiles["wappi_telegram:p-tg"]
    tg_state["complete"] = False
    tg_state["incremental_cycle"] = True
    tg_state["active_chat"] = {
        "chat": wappi_history_module.wappi_checkpoint_token("c0000"),
        "message_offset": 1,
        "page_anchor": "saved-page",
        "page_offset": 0,
    }
    wappi_history_checkpoint_path(checkpoint_dir).write_text(
        json.dumps(
            {
                "schema_version": WAPPI_HISTORY_CHECKPOINT_SCHEMA_VERSION,
                "profiles": profiles,
            }
        ),
        encoding="utf-8",
    )

    missing_one = run_wappi_history_import(
        config,
        client=CheckpointFakeClient({"p-tg": chats[1:], "p-max": []}, messages),
    )

    assert missing_one["source_lifecycle"]["unmatched_rows"] == 1
    assert missing_one["source_lifecycle"]["complete"] is False
    assert missing_one["publish_ready"] is False
    assert active_wappi_row_count(db_path) == 2
    saved = read_checkpoint(checkpoint_dir)["profiles"]["wappi_telegram:p-tg"]
    assert saved["active_chat"] is None
    assert saved["reset_reason"] == "active_chat_missing_from_catalog"


def test_source_absent_message_reappears_in_current_chat_with_unchanged_marker(
    tmp_path: Path,
) -> None:
    db_path, phase1, checkpoint_dir = prepare(tmp_path)
    chats, messages = build_universe(2)
    link_db = install_exact_widget_identity(
        tmp_path,
        db_path=db_path,
        contact_id="2002",
        customer_id="customer:source-absent",
    )
    config = replace(
        make_config(
            tmp_path,
            db_path=db_path,
            phase1=phase1,
            checkpoint_dir=checkpoint_dir,
        ),
        widget_link_db=link_db,
        refresh_widget_links=False,
    )
    baseline = run_wappi_history_import(
        config,
        client=CheckpointFakeClient({"p-tg": chats, "p-max": []}, messages),
    )
    assert baseline["publish_ready"] is True

    empty_messages = dict(messages)
    empty_messages[("telegram", "p-tg", "c0000")] = []
    retired = run_wappi_history_import(
        config,
        client=CheckpointFakeClient(
            {"p-tg": chats[1:], "p-max": []},
            empty_messages,
        ),
    )
    assert retired["source_lifecycle"]["source_absent_retired_events"] == 1
    assert active_wappi_row_count(db_path) == 1

    reappeared_client = CheckpointFakeClient(
        {"p-tg": chats, "p-max": []},
        messages,
    )
    reappeared = run_wappi_history_import(config, client=reappeared_client)

    assert reappeared["publish_ready"] is True
    assert [
        request
        for request in reappeared_client.message_request_calls
        if request[1] == "c0000"
    ] == [
        ("p-tg", "c0000", 0, 10, "asc"),
        ("p-tg", "c0000", 0, 10, "asc"),
    ]
    assert reappeared["source_lifecycle"]["restored_events"] == 1
    assert active_wappi_row_count(db_path) == 2
    with sqlite3.connect(db_path) as con:
        marker = con.execute(
            "SELECT superseded_by FROM timeline_events "
            "WHERE json_extract(record_json,'$.metadata.chat_id')='c0000'"
        ).fetchone()[0]
    assert marker is None

    disjoint_messages = [
        {
            "id": f"c0000-replacement-{index:03d}",
            "chat_id": "c0000",
            "type": "text",
            "body": f"Стабильная замена {index}",
            "time": 1_753_100_000 + index,
        }
        for index in range(35)
    ]
    replaced_source = dict(messages)
    replaced_source[("telegram", "p-tg", "c0000")] = disjoint_messages
    replaced_while_absent = run_wappi_history_import(
        config,
        client=CheckpointFakeClient(
            {"p-tg": chats[1:], "p-max": []},
            replaced_source,
        ),
    )
    assert replaced_while_absent["publish_ready"] is True
    assert replaced_while_absent["source_lifecycle"]["source_absent_retired_events"] == 1

    changed_catalog = [
        {**dict(chats[0]), "last_timestamp": 1_753_100_034},
        dict(chats[1]),
    ]
    disjoint_client = CheckpointFakeClient(
        {"p-tg": changed_catalog, "p-max": []},
        replaced_source,
    )
    disjoint_reappeared = run_wappi_history_import(config, client=disjoint_client)

    assert disjoint_reappeared["publish_ready"] is True
    disjoint_calls = [
        request
        for request in disjoint_client.message_request_calls
        if request[1] == "c0000"
    ]
    assert [request[2] for request in disjoint_calls] == [0, 10, 20, 30] * 2
    assert {request[4] for request in disjoint_calls} == {"asc"}
    checkpoint = read_checkpoint(checkpoint_dir)["profiles"]["wappi_telegram:p-tg"]
    cursor = checkpoint["chat_cursors"][wappi_checkpoint_token("c0000")]
    assert cursor["message_digest"] == wappi_history_module.wappi_message_checkpoint_token(
        "p-tg",
        "c0000",
        "c0000-replacement-034",
    )

    overdue_checkpoint = dict(read_checkpoint(checkpoint_dir))
    overdue_profiles = {
        key: dict(value)
        for key, value in overdue_checkpoint["profiles"].items()
    }
    for profile_state in overdue_profiles.values():
        profile_state["full_audit_at"] = (
            datetime.now(timezone.utc) - timedelta(days=8)
        ).isoformat()
    wappi_history_checkpoint_path(checkpoint_dir).write_text(
        json.dumps(
            {
                "schema_version": WAPPI_HISTORY_CHECKPOINT_SCHEMA_VERSION,
                "profiles": overdue_profiles,
            }
        ),
        encoding="utf-8",
    )
    overdue_client = CheckpointFakeClient(
        {"p-tg": changed_catalog, "p-max": []},
        replaced_source,
    )
    overdue = run_wappi_history_import(config, client=overdue_client)

    assert overdue["publish_ready"] is True
    assert overdue["history_validation"]["full_audit_passed"] is True
    overdue_disjoint_calls = [
        request
        for request in overdue_client.message_request_calls
        if request[1] == "c0000"
    ]
    assert [request[2] for request in overdue_disjoint_calls] == [0, 10, 20, 30] * 2
    assert {request[4] for request in overdue_disjoint_calls} == {"asc"}


def test_source_absent_exact_owner_conflict_does_not_reassign_event(
    tmp_path: Path,
) -> None:
    db_path, phase1, checkpoint_dir = prepare(tmp_path)
    chats, messages = build_universe(2)
    first_customer = "customer:source-absent-first"
    link_db = install_exact_widget_identity(
        tmp_path,
        db_path=db_path,
        contact_id="2002",
        customer_id=first_customer,
    )
    config = replace(
        make_config(
            tmp_path,
            db_path=db_path,
            phase1=phase1,
            checkpoint_dir=checkpoint_dir,
        ),
        widget_link_db=link_db,
        refresh_widget_links=False,
    )
    baseline = run_wappi_history_import(
        config,
        client=CheckpointFakeClient({"p-tg": chats, "p-max": []}, messages),
    )
    assert baseline["publish_ready"] is True

    empty_messages = dict(messages)
    empty_messages[("telegram", "p-tg", "c0000")] = []
    retired = run_wappi_history_import(
        config,
        client=CheckpointFakeClient(
            {"p-tg": chats[1:], "p-max": []},
            empty_messages,
        ),
    )
    assert retired["source_lifecycle"]["source_absent_retired_events"] == 1

    install_exact_widget_identity(
        tmp_path,
        db_path=db_path,
        contact_id="3003",
        customer_id="customer:source-absent-second",
        link_db=link_db,
    )
    checkpoint_path = wappi_history_checkpoint_path(checkpoint_dir)
    checkpoint_before = checkpoint_path.read_bytes()
    with sqlite3.connect(db_path) as con:
        event_before = con.execute(
            "SELECT customer_id,superseded_by,record_hash FROM timeline_events "
            "WHERE json_extract(record_json,'$.metadata.chat_id')='c0000'"
        ).fetchone()

    conflict = run_wappi_history_import(
        config,
        client=CheckpointFakeClient(
            {"p-tg": chats[1:], "p-max": []},
            messages,
        ),
    )

    assert conflict["publish_ready"] is False
    assert conflict["mode"] == "apply_blocked"
    assert conflict["writes"]["applied"] is False
    assert conflict["checkpoint"]["committed"] is False
    assert conflict["profiles"]["p-tg"]["message_page_drift_reason"] == (
        "historical_chat_identity_unproven"
    )
    assert checkpoint_path.read_bytes() == checkpoint_before
    with sqlite3.connect(db_path) as con:
        event_after = con.execute(
            "SELECT customer_id,superseded_by,record_hash FROM timeline_events "
            "WHERE json_extract(record_json,'$.metadata.chat_id')='c0000'"
        ).fetchone()
    assert event_after == event_before
    assert event_after[0] == first_customer


def test_resolved_historical_personal_chat_requires_exact_snapshot_and_active_history(
    tmp_path: Path,
) -> None:
    db_path, phase1, checkpoint_dir = prepare(tmp_path)
    chats, messages = build_universe(2)
    base_config = make_config(
        tmp_path,
        db_path=db_path,
        phase1=phase1,
        checkpoint_dir=checkpoint_dir,
    )
    baseline = run_wappi_history_import(
        base_config,
        client=CheckpointFakeClient({"p-tg": chats, "p-max": []}, messages),
    )
    assert baseline["publish_ready"] is True
    customer_id = "customer:historical-widget"
    link_db = install_exact_widget_identity(
        tmp_path,
        db_path=db_path,
        contact_id="2002",
        customer_id=customer_id,
    )
    config = replace(
        base_config,
        widget_link_db=link_db,
        refresh_widget_links=False,
    )

    absent_client = CheckpointFakeClient(
        {"p-tg": chats[1:], "p-max": []}, messages
    )
    absent = run_wappi_history_import(
        config,
        client=absent_client,
    )

    assert absent["publish_ready"] is True
    assert absent["source_lifecycle"]["unmatched_rows"] == 0
    assert absent["source_lifecycle"]["verified_historical_personal_chats"] == 1
    assert absent["source_lifecycle"]["verified_historical_personal_source_records"] == 1
    assert absent["profiles"]["p-tg"]["historical_snapshot_checks"] == 1
    assert absent["profiles"]["p-tg"]["historical_snapshot_verified"] == 1
    assert [
        request
        for request in absent_client.message_request_calls
        if request[1] == "c0000"
    ] == [
        ("p-tg", "c0000", 0, 10, "asc"),
        ("p-tg", "c0000", 0, 10, "asc"),
    ]
    assert absent["source_lifecycle"]["restored_events"] == 0
    assert absent["source_lifecycle"]["retired_events"] == 0
    assert active_wappi_row_count(db_path) == 2

    checkpoint_path = wappi_history_checkpoint_path(checkpoint_dir)
    checkpoint_before = checkpoint_path.read_bytes()

    class GrowingHistoricalSnapshotClient(CheckpointFakeClient):
        historical_calls = 0

        def get_chat_messages(self, **kwargs: Any) -> Mapping[str, Any]:
            if kwargs["chat_id"] == "c0000":
                self.historical_calls += 1
                if self.historical_calls == 2:
                    self.messages[("telegram", "p-tg", "c0000")].append(
                        {
                            "id": "c0000-race",
                            "chat_id": "c0000",
                            "type": "text",
                            "body": "Появилось между снимком и проверкой",
                            "time": 1_752_999_998,
                        }
                    )
            return super().get_chat_messages(**kwargs)

    raced_absent = run_wappi_history_import(
        config,
        client=GrowingHistoricalSnapshotClient(
            {"p-tg": chats[1:], "p-max": []}, messages
        ),
    )

    assert raced_absent["publish_ready"] is False
    assert raced_absent["writes"]["applied"] is False
    assert raced_absent["checkpoint"]["committed"] is False
    assert raced_absent["profiles"]["p-tg"]["message_page_drift_reason"] == (
        "full_history_pagination_drift"
    )
    assert checkpoint_path.read_bytes() == checkpoint_before
    assert active_wappi_row_count(db_path) == 2

    original_head = dict(messages[("telegram", "p-tg", "c0000")][0])
    messages[("telegram", "p-tg", "c0000")][0] = {
        **original_head,
        "body": "Отредактированное сообщение с прежним ID",
        "time": 1_753_000_010,
    }
    rewritten_absent = run_wappi_history_import(
        config,
        client=CheckpointFakeClient({"p-tg": chats[1:], "p-max": []}, messages),
    )

    assert rewritten_absent["publish_ready"] is True
    assert rewritten_absent["mode"] == "apply"
    assert rewritten_absent["writes"]["applied"] is True
    assert rewritten_absent["checkpoint"]["committed"] is True
    assert rewritten_absent["profiles"]["p-tg"]["historical_snapshot_reconciled"] == 1
    assert rewritten_absent["profiles"]["p-tg"]["historical_snapshot_conflicting"] == 1
    with sqlite3.connect(db_path) as con:
        rewritten_payload = json.loads(
            con.execute(
                "SELECT record_json FROM timeline_events "
                "WHERE json_extract(record_json,'$.metadata.message_id')='c0000-m000'"
            ).fetchone()[0]
        )
    assert rewritten_payload["record"]["message"]["text"] == (
        "Отредактированное сообщение с прежним ID"
    )
    assert active_wappi_row_count(db_path) == 2

    messages[("telegram", "p-tg", "c0000")][0] = original_head
    new_message = {
        "id": "c0000-new",
        "chat_id": "c0000",
        "type": "text",
        "body": "Новое сообщение задним числом в отсутствующем чате",
        "time": 1_752_999_999,
    }
    messages[("telegram", "p-tg", "c0000")].append(new_message)
    changed_absent = run_wappi_history_import(
        config,
        client=CheckpointFakeClient({"p-tg": chats[1:], "p-max": []}, messages),
    )

    assert changed_absent["publish_ready"] is True
    assert changed_absent["mode"] == "apply"
    assert changed_absent["writes"]["applied"] is True
    assert changed_absent["checkpoint"]["committed"] is True
    assert changed_absent["profiles"]["p-tg"]["historical_snapshot_reconciled"] == 1
    assert changed_absent["profiles"]["p-tg"]["historical_snapshot_source_new"] == 1
    assert changed_absent["profiles"]["p-tg"]["historical_snapshot_conflicting"] == 1
    assert changed_absent["source_lifecycle"]["verified_historical_personal_chats"] == 1
    assert active_wappi_row_count(db_path) == 3

    messages[("telegram", "p-tg", "c0000")] = [new_message]
    partially_deleted = run_wappi_history_import(
        config,
        client=CheckpointFakeClient({"p-tg": chats[1:], "p-max": []}, messages),
    )

    assert partially_deleted["publish_ready"] is True
    assert partially_deleted["profiles"]["p-tg"]["historical_snapshot_source_absent"] == 1
    assert partially_deleted["source_lifecycle"]["source_absent_retired_events"] == 1
    assert partially_deleted["source_lifecycle"]["source_absent_active_after"] == 0
    assert active_wappi_row_count(db_path) == 2

    messages[("telegram", "p-tg", "c0000")] = []
    deleted_to_empty = run_wappi_history_import(
        config,
        client=CheckpointFakeClient({"p-tg": chats[1:], "p-max": []}, messages),
    )

    assert deleted_to_empty["publish_ready"] is True
    assert deleted_to_empty["profiles"]["p-tg"]["historical_snapshot_source_records"] == 0
    assert deleted_to_empty["source_lifecycle"]["source_absent_retired_events"] == 1
    assert deleted_to_empty["source_lifecycle"]["verified_historical_expected_active"] == 0
    assert active_wappi_row_count(db_path) == 1

    repeated_empty = run_wappi_history_import(
        config,
        client=CheckpointFakeClient({"p-tg": chats[1:], "p-max": []}, messages),
    )

    assert repeated_empty["publish_ready"] is True
    assert repeated_empty["source_lifecycle"]["source_absent_retired_events"] == 0
    assert repeated_empty["source_lifecycle"]["source_absent_active_after"] == 0
    assert active_wappi_row_count(db_path) == 1

    messages[("telegram", "p-tg", "c0000")] = [original_head, new_message]
    reappeared = run_wappi_history_import(
        config,
        client=CheckpointFakeClient({"p-tg": chats[1:], "p-max": []}, messages),
    )

    assert reappeared["publish_ready"] is True
    assert reappeared["profiles"]["p-tg"]["historical_snapshot_source_new"] == 2
    assert reappeared["source_lifecycle"]["restored_events"] == 2
    assert reappeared["source_lifecycle"]["verified_historical_present_active"] == 2
    assert active_wappi_row_count(db_path) == 3

    grouped = [{**dict(chats[0]), "type": "group"}, dict(chats[1])]
    grouped_client = CheckpointFakeClient({"p-tg": grouped, "p-max": []}, messages)
    excluded = run_wappi_history_import(
        config,
        client=grouped_client,
    )

    assert excluded["publish_ready"] is True
    assert excluded["source_lifecycle"]["verified_historical_personal_source_records"] == 0
    assert not any(request[1] == "c0000" for request in grouped_client.message_request_calls)
    assert excluded["source_lifecycle"]["retired_events"] == 2
    assert active_wappi_row_count(db_path) == 1

    absent_after_exclusion = run_wappi_history_import(
        config,
        client=CheckpointFakeClient({"p-tg": chats[1:], "p-max": []}, messages),
    )

    assert absent_after_exclusion["publish_ready"] is True
    assert absent_after_exclusion["source_lifecycle"]["unmatched_rows"] == 0
    assert absent_after_exclusion["source_lifecycle"]["verified_historical_personal_source_records"] == 0
    assert absent_after_exclusion["source_lifecycle"]["catalog_non_personal_source_records"] == 2
    assert active_wappi_row_count(db_path) == 1


def test_current_non_personal_records_are_retired_and_excluded_from_persistence_gate(
    tmp_path: Path,
) -> None:
    db_path, phase1, checkpoint_dir = prepare(tmp_path)
    chats, messages = build_universe(1, messages_per_chat=2)
    non_personal = [{**chats[0], "type": "group"}]
    config = make_config(
        tmp_path,
        db_path=db_path,
        phase1=phase1,
        checkpoint_dir=checkpoint_dir,
    )

    report = run_wappi_history_import(
        config,
        client=CheckpointFakeClient({"p-tg": non_personal, "p-max": []}, messages),
    )

    assert report["source_persistence_complete"] is True
    assert report["checkpoint"]["committed"] is True
    assert report["summary"]["messages_expected_in_timeline"] == 0
    assert report["summary"]["messages_missing_from_timeline"] == 0
    assert report["source_lifecycle"]["current_ledger_expected_excluded_records"] == 2
    assert report["source_lifecycle"]["retired_events"] == 2
    assert report["source_lifecycle"]["non_personal_active_after"] == 0
    assert wappi_row_count(db_path) == 2
    assert active_wappi_row_count(db_path) == 0


def test_new_non_personal_chat_is_not_confirmed_before_personal_flip(
    tmp_path: Path,
) -> None:
    db_path, phase1, checkpoint_dir = prepare(tmp_path)
    config = make_config(
        tmp_path,
        db_path=db_path,
        phase1=phase1,
        checkpoint_dir=checkpoint_dir,
    )
    run_wappi_history_import(
        config,
        client=CheckpointFakeClient({"p-tg": [], "p-max": []}, {}),
    )
    chats, messages = build_universe(1)
    group = [{**chats[0], "type": "group"}]
    group_client = CheckpointFakeClient({"p-tg": group, "p-max": []}, messages)
    group_report = run_wappi_history_import(config, client=group_client)

    token = wappi_history_module.wappi_checkpoint_token("c0000")
    assert group_report["checkpoint"]["committed"] is True
    group_state = read_checkpoint(checkpoint_dir)["profiles"]["wappi_telegram:p-tg"]
    assert group_client.message_calls == []
    assert token not in group_state["chats_done"]
    assert token not in group_state["chat_markers"]

    personal_client = CheckpointFakeClient({"p-tg": chats, "p-max": []}, messages)
    personal = run_wappi_history_import(config, client=personal_client)

    assert personal["publish_ready"] is True
    assert {
        order
        for profile, chat, _offset, _limit, order in personal_client.message_request_calls
        if profile == "p-tg" and chat == "c0000"
    } == {"asc"}
    assert active_wappi_row_count(db_path) == 1


def test_checkpoint_requires_complete_message_history(tmp_path: Path) -> None:
    db_path, phase1, checkpoint_dir = prepare(tmp_path)
    with pytest.raises(ValueError, match="complete_message_history"):
        WappiHistoryImportConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            phase1_config=phase1,
            checkpoint_dir=checkpoint_dir,
            limits=WappiFetchLimits(complete_message_history=False, sleep_seconds=0),
        )


def test_checkpoint_dir_must_stay_under_allowed_root(tmp_path: Path) -> None:
    db_path, phase1, _ = prepare(tmp_path)
    with pytest.raises(ValueError, match="allowed root"):
        WappiHistoryImportConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            phase1_config=phase1,
            checkpoint_dir=tmp_path.parent / "outside_checkpoints",
            limits=WappiFetchLimits(complete_message_history=True, sleep_seconds=0),
        )


def test_catalog_failure_on_page_13_writes_nothing_until_full_catalog_is_proven(
    tmp_path: Path,
) -> None:
    """An unproven catalogue cannot authorize message reads or partial writes."""
    db_path, phase1, checkpoint_dir = prepare(tmp_path)
    chats, messages = build_universe(130)
    first = CheckpointFakeClient({"p-tg": chats, "p-max": []}, messages)
    first.fail_catalog_from_offset = 120

    report_one = run_wappi_history_import(
        make_config(tmp_path, db_path=db_path, phase1=phase1, checkpoint_dir=checkpoint_dir),
        client=first,
    )

    assert report_one["mode"] == "apply_blocked"
    assert report_one["validation_ok"] is False
    assert report_one["checkpoint"]["complete"] is False
    assert report_one["checkpoint"]["committed"] is False
    tg_state = report_one["checkpoint"]["profiles"]["wappi_telegram:p-tg"]
    assert tg_state["stop_reason"] == "network_error"
    assert tg_state["chats_done"] == 0
    assert first.message_calls == []
    assert wappi_row_count(db_path) == 0

    second = CheckpointFakeClient({"p-tg": chats, "p-max": []}, messages)
    report_two = run_wappi_history_import(
        make_config(tmp_path, db_path=db_path, phase1=phase1, checkpoint_dir=checkpoint_dir),
        client=second,
    )

    refetched = {chat_id for _profile, chat_id, _offset in second.message_calls}
    assert {chat["id"] for chat in chats}.issubset(refetched)
    assert wappi_row_count(db_path) == 130
    assert report_two["checkpoint"]["complete"] is True
    assert report_two["validation_ok"] is True
    assert wappi_history_checkpoint_path(checkpoint_dir).exists()


def test_message_page_failure_resumes_inside_long_chat(tmp_path: Path) -> None:
    """A long chat dies on its second message page and resumes from that offset."""
    db_path, phase1, checkpoint_dir = prepare(tmp_path)
    chats, messages = build_universe(1, messages_per_chat=30)
    first = CheckpointFakeClient({"p-tg": chats, "p-max": []}, messages)
    # Catalogue page + first message page fit; the second message page does not.
    report_one = run_wappi_history_import(
        make_config(tmp_path, db_path=db_path, phase1=phase1, checkpoint_dir=checkpoint_dir, request_limit_total=7),
        client=first,
    )

    tg_state = report_one["checkpoint"]["profiles"]["wappi_telegram:p-tg"]
    assert report_one["checkpoint"]["complete"] is False
    assert tg_state["stop_reason"] == "request_budget"
    assert tg_state["chats_done"] == 0
    assert tg_state["active_chat_message_offset"] == 10
    assert wappi_row_count(db_path) == 10

    second = CheckpointFakeClient({"p-tg": chats, "p-max": []}, messages)
    run_wappi_history_import(
        make_config(tmp_path, db_path=db_path, phase1=phase1, checkpoint_dir=checkpoint_dir),
        client=second,
    )

    # The resume re-reads the last confirmed page as an anchor, then continues.
    assert second.message_calls[0][2] == 0
    assert 10 in [offset for _profile, _chat, offset in second.message_calls]
    assert wappi_row_count(db_path) == 30
    assert wappi_history_checkpoint_path(checkpoint_dir).exists()


def test_request_limit_pause_writes_and_next_run_continues(tmp_path: Path) -> None:
    db_path, phase1, checkpoint_dir = prepare(tmp_path)
    chats, messages = build_universe(20)
    first = CheckpointFakeClient({"p-tg": chats, "p-max": []}, messages)

    report_one = run_wappi_history_import(
        make_config(
            tmp_path,
            db_path=db_path,
            phase1=phase1,
            checkpoint_dir=checkpoint_dir,
            request_limit_total=12,
        ),
        client=first,
    )

    assert report_one["mode"] == "apply"  # a paused run still writes what it confirmed
    assert report_one["validation_ok"] is False  # ... but never claims freshness
    assert any(marker.endswith("request_limit_hit") for marker in report_one["limit_hits"])
    assert report_one["checkpoint"]["deferred_limit_hits"]
    written_after_first = wappi_row_count(db_path)
    assert 0 < written_after_first < 20

    second = CheckpointFakeClient({"p-tg": chats, "p-max": []}, messages)
    report_two = run_wappi_history_import(
        make_config(tmp_path, db_path=db_path, phase1=phase1, checkpoint_dir=checkpoint_dir),
        client=second,
    )

    assert report_two["checkpoint"]["complete"] is True
    assert report_two["validation_ok"] is True
    assert wappi_row_count(db_path) == 20
    assert report_two["summary"]["duplicate_source_ids_before_import"] == 0


def test_new_chat_between_runs_is_picked_up_without_refetching_confirmed(tmp_path: Path) -> None:
    db_path, phase1, checkpoint_dir = prepare(tmp_path)
    chats, messages = build_universe(20)
    first = CheckpointFakeClient({"p-tg": chats, "p-max": []}, messages)
    run_wappi_history_import(
        make_config(tmp_path, db_path=db_path, phase1=phase1, checkpoint_dir=checkpoint_dir, request_limit_total=12),
        client=first,
    )
    confirmed_before = read_checkpoint(checkpoint_dir)["profiles"]["wappi_telegram:p-tg"]["chats_done"]
    assert confirmed_before

    # A brand new chat lands at the FRONT of the catalogue: every positional index
    # shifts, so an index-based checkpoint would silently skip a chat here.
    new_chat = {"id": "c9999", "type": "user"}
    reordered = [new_chat, *chats]
    messages[("telegram", "p-tg", "c9999")] = [
        {"id": "c9999-m000", "chat_id": "c9999", "type": "text", "body": "Новый чат", "time": 1_753_000_500}
    ]
    second = CheckpointFakeClient({"p-tg": reordered, "p-max": []}, messages)
    run_wappi_history_import(
        make_config(tmp_path, db_path=db_path, phase1=phase1, checkpoint_dir=checkpoint_dir),
        client=second,
    )

    refetched = {chat_id for _profile, chat_id, _offset in second.message_calls}
    already_done = {
        chat["id"] for chat in chats if wappi_checkpoint_token(str(chat["id"])) in set(confirmed_before)
    }
    assert already_done and not refetched & already_done
    assert "c9999" in refetched
    assert wappi_row_count(db_path) == 21


def test_new_message_in_confirmed_chat_is_loaded_before_terminal_complete(tmp_path: Path) -> None:
    db_path, phase1, checkpoint_dir = prepare(tmp_path)
    chats, messages = build_universe(20)
    run_wappi_history_import(
        make_config(tmp_path, db_path=db_path, phase1=phase1, checkpoint_dir=checkpoint_dir, request_limit_total=12),
        client=CheckpointFakeClient({"p-tg": chats, "p-max": []}, messages),
    )
    confirmed = set(read_checkpoint(checkpoint_dir)["profiles"]["wappi_telegram:p-tg"]["chats_done"])
    confirmed_chat = next(chat for chat in chats if wappi_checkpoint_token(str(chat["id"])) in confirmed)
    chat_id = str(confirmed_chat["id"])
    messages[("telegram", "p-tg", chat_id)].append(
        {"id": f"{chat_id}-new", "chat_id": chat_id, "type": "text", "body": "Новое", "time": 1_753_000_999}
    )
    confirmed_chat["last_timestamp"] = 1_753_000_999

    report = run_wappi_history_import(
        make_config(tmp_path, db_path=db_path, phase1=phase1, checkpoint_dir=checkpoint_dir),
        client=CheckpointFakeClient({"p-tg": chats, "p-max": []}, messages),
    )

    assert report["validation_ok"] is True
    assert wappi_row_count(db_path) == 21


def test_saved_tail_marker_cannot_hide_a_later_message(tmp_path: Path) -> None:
    db_path, phase1, checkpoint_dir = prepare(tmp_path)
    chats, messages = build_universe(2)
    config = make_config(tmp_path, db_path=db_path, phase1=phase1, checkpoint_dir=checkpoint_dir)
    run_wappi_history_import(
        config, client=CheckpointFakeClient({"p-tg": chats, "p-max": []}, messages)
    )
    profile = WappiProfileSpec(profile_id="p-tg", brand="foton", channel="telegram")
    state = wappi_timeline_state(db_path, tenant_id="foton", profiles=(profile,))[
        "wappi_telegram:p-tg"
    ]
    tokens = [wappi_checkpoint_token(str(chat["id"])) for chat in chats]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    wappi_history_checkpoint_path(checkpoint_dir).write_text(
        json.dumps(
            {
                "schema_version": WAPPI_HISTORY_CHECKPOINT_SCHEMA_VERSION,
                "profiles": {
                    "wappi_telegram:p-tg": {
                        "fingerprint": wappi_fetch_universe_fingerprint(
                            profile, config.limits, tenant_id="foton"
                        ),
                        "complete": False,
                        "catalog_next_offset": 2,
                        "catalog_page_anchor": wappi_checkpoint_anchor(tuple(tokens)),
                        "chats_done": tokens,
                        "tail_checked": tokens,
                        "active_chat": None,
                        "timeline_rows": state["rows"],
                        "timeline_source_digest": state["source_digest"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    chat_id = str(chats[0]["id"])
    messages[("telegram", "p-tg", chat_id)].append(
        {"id": f"{chat_id}-new", "chat_id": chat_id, "type": "text", "body": "Новое", "time": 1_753_001_000}
    )

    report = run_wappi_history_import(
        config, client=CheckpointFakeClient({"p-tg": chats, "p-max": []}, messages)
    )

    assert report["validation_ok"] is True
    assert report["checkpoint"]["profiles"]["wappi_telegram:p-tg"]["reset_reason"] is None
    assert wappi_row_count(db_path) == 3


def test_message_page_drift_restarts_that_chat_from_zero(tmp_path: Path) -> None:
    db_path, phase1, checkpoint_dir = prepare(tmp_path)
    chats, messages = build_universe(1, messages_per_chat=30)
    first = CheckpointFakeClient({"p-tg": chats, "p-max": []}, messages)
    run_wappi_history_import(
        make_config(tmp_path, db_path=db_path, phase1=phase1, checkpoint_dir=checkpoint_dir, request_limit_total=7),
        client=first,
    )
    saved = read_checkpoint(checkpoint_dir)["profiles"]["wappi_telegram:p-tg"]["active_chat"]
    assert saved["message_offset"] == 10

    # The confirmed first page changes underneath us: the saved offset is no longer
    # trustworthy, so the chat must restart at zero rather than skip messages.
    drifted = dict(messages)
    drifted[("telegram", "p-tg", "c0000")] = [
        {**dict(item), "id": f"shifted-{item['id']}"} for item in messages[("telegram", "p-tg", "c0000")]
    ]
    second = CheckpointFakeClient({"p-tg": chats, "p-max": []}, drifted)
    run_wappi_history_import(
        make_config(tmp_path, db_path=db_path, phase1=phase1, checkpoint_dir=checkpoint_dir),
        client=second,
    )

    offsets = [offset for _profile, _chat, offset in second.message_calls]
    assert offsets[0] == 0 and offsets[1] == 0
    assert wappi_row_count(db_path) == 40  # 10 original + 30 renamed, no lost page


def test_repeat_run_creates_no_duplicates(tmp_path: Path) -> None:
    db_path, phase1, checkpoint_dir = prepare(tmp_path)
    chats, messages = build_universe(6, messages_per_chat=2)
    config = make_config(tmp_path, db_path=db_path, phase1=phase1, checkpoint_dir=checkpoint_dir)

    run_wappi_history_import(config, client=CheckpointFakeClient({"p-tg": chats, "p-max": []}, messages))
    first_rows = wappi_row_count(db_path)
    second = CheckpointFakeClient({"p-tg": chats, "p-max": []}, messages)
    report_two = run_wappi_history_import(config, client=second)

    assert first_rows == 12
    assert wappi_row_count(db_path) == 12
    assert second.message_calls == []
    assert report_two["validation_ok"] is True
    assert "p-tg" not in report_two["summary"]["empty_profiles"]
    assert report_two["profiles"]["p-tg"]["records_built"] == 0
    assert report_two["profiles"]["p-tg"]["catalog_passes"] == 2
    assert report_two["profiles"]["p-tg"]["incremental_chats_skipped"] == 6


def test_incremental_run_fetches_only_changed_chat(tmp_path: Path) -> None:
    db_path, phase1, checkpoint_dir = prepare(tmp_path)
    chats, messages = build_universe(4)
    config = make_config(tmp_path, db_path=db_path, phase1=phase1, checkpoint_dir=checkpoint_dir)
    run_wappi_history_import(config, client=CheckpointFakeClient({"p-tg": chats, "p-max": []}, messages))

    changed = [dict(chat) for chat in chats]
    changed[2]["last_timestamp"] = int(changed[2]["last_timestamp"]) + 1
    messages[("telegram", "p-tg", "c0002")].append(
        {"id": "c0002-m001", "chat_id": "c0002", "type": "text", "body": "Новое", "time": 1_753_000_001}
    )
    second = CheckpointFakeClient({"p-tg": changed, "p-max": []}, messages)
    report = run_wappi_history_import(config, client=second)

    assert {chat_id for _profile, chat_id, _offset in second.message_calls} == {"c0002"}
    assert wappi_row_count(db_path) == 5
    assert report["profiles"]["p-tg"]["incremental_chats_changed"] == 1
    assert report["profiles"]["p-tg"]["incremental_chats_skipped"] == 3


def test_failed_changed_chat_does_not_force_full_refetch(tmp_path: Path) -> None:
    db_path, phase1, checkpoint_dir = prepare(tmp_path)
    chats, messages = build_universe(3, messages_per_chat=30)
    config = make_config(tmp_path, db_path=db_path, phase1=phase1, checkpoint_dir=checkpoint_dir)
    run_wappi_history_import(config, client=CheckpointFakeClient({"p-tg": chats, "p-max": []}, messages))

    changed = [dict(chat) for chat in chats]
    changed[1]["last_timestamp"] = 1_753_000_030
    messages[("telegram", "p-tg", "c0001")].append(
        {"id": "c0001-m030", "chat_id": "c0001", "type": "text", "body": "Позднее", "time": 1_753_000_030}
    )
    failing = CheckpointFakeClient({"p-tg": changed, "p-max": []}, messages)
    failing.fail_message_at = ("c0001", 0)
    failed = run_wappi_history_import(config, client=failing)

    state = read_checkpoint(checkpoint_dir)["profiles"]["wappi_telegram:p-tg"]
    assert failed["checkpoint"]["complete"] is False
    assert state["incremental_cycle"] is True

    finisher = CheckpointFakeClient({"p-tg": changed, "p-max": []}, messages)
    report = run_wappi_history_import(config, client=finisher)

    assert {chat_id for _profile, chat_id, _offset in finisher.message_calls} == {"c0001"}
    assert report["checkpoint"]["complete"] is True
    assert report["profiles"]["p-tg"]["incremental_chats_skipped"] == 2
    assert wappi_row_count(db_path) == 91


def test_long_changed_chat_eventually_completes_after_budget_stops(tmp_path: Path) -> None:
    db_path, phase1, checkpoint_dir = prepare(tmp_path)
    chats, messages = build_universe(1, messages_per_chat=50)
    config = make_config(
        tmp_path,
        db_path=db_path,
        phase1=phase1,
        checkpoint_dir=checkpoint_dir,
        request_limit_total=8,
    )

    reports = []
    for _ in range(8):
        reports.append(
            run_wappi_history_import(
                config,
                client=CheckpointFakeClient({"p-tg": chats, "p-max": []}, messages),
            )
        )
        if reports[-1]["checkpoint"]["complete"]:
            break

    assert reports[-1]["checkpoint"]["complete"] is True
    assert wappi_row_count(db_path) == 50
    assert len(reports) < 8


def test_periodic_full_audit_recovers_message_hidden_by_unchanged_marker(tmp_path: Path) -> None:
    db_path, phase1, checkpoint_dir = prepare(tmp_path)
    chats, messages = build_universe(1)
    config = make_config(tmp_path, db_path=db_path, phase1=phase1, checkpoint_dir=checkpoint_dir)
    run_wappi_history_import(config, client=CheckpointFakeClient({"p-tg": chats, "p-max": []}, messages))

    messages[("telegram", "p-tg", "c0000")].append(
        {"id": "c0000-late", "chat_id": "c0000", "type": "text", "body": "Позднее", "time": 1_753_000_000}
    )
    daily = CheckpointFakeClient({"p-tg": chats, "p-max": []}, messages)
    daily_report = run_wappi_history_import(config, client=daily)
    assert daily.message_calls == []
    assert wappi_row_count(db_path) == 1
    assert daily_report["history_validation"] == {
        "mode": "incremental_catalog",
        "catalog_incremental_passed": True,
        "full_audit_completed_this_run": False,
        "full_audit_fresh_for_all_profiles": True,
        "full_audit_passed": True,
        "full_audit_interval_days": 7,
    }

    checkpoint = dict(read_checkpoint(checkpoint_dir))
    profiles = {key: dict(value) for key, value in checkpoint["profiles"].items()}
    old_audit_at = (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()
    for profile_state in profiles.values():
        profile_state["full_audit_at"] = old_audit_at
    wappi_history_checkpoint_path(checkpoint_dir).write_text(
        json.dumps({"schema_version": WAPPI_HISTORY_CHECKPOINT_SCHEMA_VERSION, "profiles": profiles}),
        encoding="utf-8",
    )
    audit = CheckpointFakeClient({"p-tg": chats, "p-max": []}, messages)
    audit_report = run_wappi_history_import(config, client=audit)

    assert {chat_id for _profile, chat_id, _offset in audit.message_calls} == {"c0000"}
    assert wappi_row_count(db_path) == 2
    assert audit_report["history_validation"]["full_audit_passed"] is True


def test_periodic_full_audit_resumes_and_reaches_last_chat_with_small_budget(tmp_path: Path) -> None:
    db_path, phase1, checkpoint_dir = prepare(tmp_path)
    chats, messages = build_universe(20)
    config = make_config(tmp_path, db_path=db_path, phase1=phase1, checkpoint_dir=checkpoint_dir)
    run_wappi_history_import(config, client=CheckpointFakeClient({"p-tg": chats, "p-max": []}, messages))
    messages[("telegram", "p-tg", "c0019")].append(
        {"id": "c0019-late", "chat_id": "c0019", "type": "text", "body": "Позднее", "time": 1_753_000_000}
    )
    checkpoint = dict(read_checkpoint(checkpoint_dir))
    profiles = {key: dict(value) for key, value in checkpoint["profiles"].items()}
    for state in profiles.values():
        state["full_audit_at"] = (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()
    wappi_history_checkpoint_path(checkpoint_dir).write_text(
        json.dumps({"schema_version": WAPPI_HISTORY_CHECKPOINT_SCHEMA_VERSION, "profiles": profiles}),
        encoding="utf-8",
    )

    seen_calls: list[str] = []
    for _ in range(12):
        client = CheckpointFakeClient({"p-tg": chats, "p-max": []}, messages)
        report = run_wappi_history_import(
            make_config(
                tmp_path, db_path=db_path, phase1=phase1, checkpoint_dir=checkpoint_dir,
                request_limit_total=12,
            ),
            client=client,
        )
        seen_calls.extend(chat_id for _profile, chat_id, _offset in client.message_calls)
        if report["checkpoint"]["complete"]:
            break

    assert report["checkpoint"]["complete"] is True
    assert report["history_validation"]["full_audit_fresh_for_all_profiles"] is True
    assert report["history_validation"]["full_audit_passed"] is True
    assert wappi_row_count(db_path) == 21
    assert "c0019" in seen_calls
    assert {seen_calls.count(str(chat["id"])) for chat in chats} == {2}
    state = read_checkpoint(checkpoint_dir)["profiles"]["wappi_telegram:p-tg"]
    assert state["full_audit_markers"] == {}
    assert state["full_audit_started_at"] == ""

    unchanged = CheckpointFakeClient({"p-tg": chats, "p-max": []}, messages)
    daily = run_wappi_history_import(config, client=unchanged)
    assert daily["history_validation"]["mode"] == "incremental_catalog"
    assert unchanged.message_calls == []
    assert daily["summary"]["incremental_chats_skipped"] == 20


def test_full_audit_rechecks_markerless_cursor_before_pass(tmp_path: Path) -> None:
    db_path, phase1, checkpoint_dir = prepare(tmp_path)
    chats, messages = build_universe(1)
    chats[0].pop("last_timestamp", None)
    config = make_config(tmp_path, db_path=db_path, phase1=phase1, checkpoint_dir=checkpoint_dir)
    initial = run_wappi_history_import(
        config,
        client=CheckpointFakeClient({"p-tg": chats, "p-max": []}, messages),
    )
    assert initial["publish_ready"] is True

    checkpoint = dict(read_checkpoint(checkpoint_dir))
    profiles = {key: dict(value) for key, value in checkpoint["profiles"].items()}
    tg_state = profiles["wappi_telegram:p-tg"]
    chat_token = wappi_history_module.wappi_checkpoint_token("c0000")
    tg_state["full_audit_at"] = (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()
    tg_state["full_audit_started_at"] = datetime.now(timezone.utc).isoformat()
    tg_state["full_audit_markers"] = {chat_token: 0}
    tg_state["complete"] = False
    tg_state["incremental_cycle"] = False
    wappi_history_checkpoint_path(checkpoint_dir).write_text(
        json.dumps(
            {
                "schema_version": WAPPI_HISTORY_CHECKPOINT_SCHEMA_VERSION,
                "profiles": profiles,
            }
        ),
        encoding="utf-8",
    )
    messages[("telegram", "p-tg", "c0000")].append(
        {
            "id": "c0000-late",
            "chat_id": "c0000",
            "type": "text",
            "body": "Позднее",
            "time": 1_753_000_100,
        }
    )

    client = CheckpointFakeClient({"p-tg": chats, "p-max": []}, messages)
    report = run_wappi_history_import(config, client=client)

    assert report["history_validation"]["full_audit_passed"] is True
    assert wappi_row_count(db_path) == 2
    assert {
        order
        for profile, chat, _offset, _limit, order in client.message_request_calls
        if profile == "p-tg" and chat == "c0000"
    } == {"desc"}


def test_full_audit_network_failure_keeps_checked_chat_progress(tmp_path: Path) -> None:
    db_path, phase1, checkpoint_dir = prepare(tmp_path)
    chats, messages = build_universe(10)
    config = make_config(tmp_path, db_path=db_path, phase1=phase1, checkpoint_dir=checkpoint_dir)
    run_wappi_history_import(config, client=CheckpointFakeClient({"p-tg": chats, "p-max": []}, messages))
    checkpoint = dict(read_checkpoint(checkpoint_dir))
    profiles = {key: dict(value) for key, value in checkpoint["profiles"].items()}
    for state in profiles.values():
        state["full_audit_at"] = (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()
    wappi_history_checkpoint_path(checkpoint_dir).write_text(
        json.dumps({"schema_version": WAPPI_HISTORY_CHECKPOINT_SCHEMA_VERSION, "profiles": profiles}),
        encoding="utf-8",
    )
    failed = CheckpointFakeClient({"p-tg": chats, "p-max": []}, messages)
    failed.fail_message_at = ("c0005", 0)
    run_wappi_history_import(config, client=failed)
    saved = read_checkpoint(checkpoint_dir)["profiles"]["wappi_telegram:p-tg"]
    assert len(saved["full_audit_markers"]) == 5

    resumed = CheckpointFakeClient({"p-tg": chats, "p-max": []}, messages)
    report = run_wappi_history_import(config, client=resumed)
    resumed_chats = {chat_id for _profile, chat_id, _offset in resumed.message_calls}
    assert not resumed_chats & {f"c{index:04d}" for index in range(5)}
    assert report["checkpoint"]["complete"] is True


def test_future_full_audit_timestamp_is_treated_as_clock_anomaly(tmp_path: Path) -> None:
    db_path, phase1, checkpoint_dir = prepare(tmp_path)
    chats, messages = build_universe(1)
    config = make_config(tmp_path, db_path=db_path, phase1=phase1, checkpoint_dir=checkpoint_dir)
    run_wappi_history_import(config, client=CheckpointFakeClient({"p-tg": chats, "p-max": []}, messages))
    messages[("telegram", "p-tg", "c0000")].append(
        {"id": "c0000-late", "chat_id": "c0000", "type": "text", "body": "Позднее", "time": 1_753_000_000}
    )
    checkpoint = dict(read_checkpoint(checkpoint_dir))
    profiles = {key: dict(value) for key, value in checkpoint["profiles"].items()}
    profiles["wappi_telegram:p-tg"]["full_audit_at"] = "2099-01-01T00:00:00+00:00"
    wappi_history_checkpoint_path(checkpoint_dir).write_text(
        json.dumps({"schema_version": WAPPI_HISTORY_CHECKPOINT_SCHEMA_VERSION, "profiles": profiles}),
        encoding="utf-8",
    )

    report = run_wappi_history_import(
        config, client=CheckpointFakeClient({"p-tg": chats, "p-max": []}, messages)
    )

    assert report["profiles"]["p-tg"]["full_audit_clock_anomaly"] is True
    assert wappi_row_count(db_path) == 2


def test_incremental_run_fetches_only_new_chat(tmp_path: Path) -> None:
    db_path, phase1, checkpoint_dir = prepare(tmp_path)
    chats, messages = build_universe(3)
    config = make_config(tmp_path, db_path=db_path, phase1=phase1, checkpoint_dir=checkpoint_dir)
    run_wappi_history_import(config, client=CheckpointFakeClient({"p-tg": chats, "p-max": []}, messages))

    new_chat = {"id": "c9999", "type": "user", "last_timestamp": 1_753_000_900}
    messages[("telegram", "p-tg", "c9999")] = [
        {"id": "c9999-m000", "chat_id": "c9999", "type": "text", "body": "Новый чат", "time": 1_753_000_900}
    ]
    second = CheckpointFakeClient({"p-tg": [*chats, new_chat], "p-max": []}, messages)
    report = run_wappi_history_import(config, client=second)

    assert {chat_id for _profile, chat_id, _offset in second.message_calls} == {"c9999"}
    assert wappi_row_count(db_path) == 4
    assert report["profiles"]["p-tg"]["incremental_chats_new"] == 1
    assert report["profiles"]["p-tg"]["incremental_chats_skipped"] == 3


def test_regressed_catalog_marker_is_refetched_and_reported(tmp_path: Path) -> None:
    db_path, phase1, checkpoint_dir = prepare(tmp_path)
    chats, messages = build_universe(2)
    config = make_config(tmp_path, db_path=db_path, phase1=phase1, checkpoint_dir=checkpoint_dir)
    run_wappi_history_import(config, client=CheckpointFakeClient({"p-tg": chats, "p-max": []}, messages))
    regressed = [dict(chat) for chat in chats]
    regressed[0]["last_timestamp"] = int(regressed[0]["last_timestamp"]) - 1
    second = CheckpointFakeClient({"p-tg": regressed, "p-max": []}, messages)

    report = run_wappi_history_import(config, client=second)

    assert {chat_id for _profile, chat_id, _offset in second.message_calls} == {"c0000"}
    assert report["profiles"]["p-tg"]["incremental_chats_marker_regressed"] == 1


def test_complete_legacy_checkpoint_without_markers_refetches_once(tmp_path: Path) -> None:
    db_path, phase1, checkpoint_dir = prepare(tmp_path)
    chats, messages = build_universe(3)
    config = make_config(tmp_path, db_path=db_path, phase1=phase1, checkpoint_dir=checkpoint_dir)
    run_wappi_history_import(config, client=CheckpointFakeClient({"p-tg": chats, "p-max": []}, messages))
    checkpoint = dict(read_checkpoint(checkpoint_dir))
    profiles = {key: dict(value) for key, value in checkpoint["profiles"].items()}
    profiles["wappi_telegram:p-tg"].pop("chat_markers", None)
    save_payload = {"schema_version": WAPPI_HISTORY_CHECKPOINT_SCHEMA_VERSION, "profiles": profiles}
    wappi_history_checkpoint_path(checkpoint_dir).write_text(json.dumps(save_payload), encoding="utf-8")

    second = CheckpointFakeClient({"p-tg": chats, "p-max": []}, messages)
    report = run_wappi_history_import(config, client=second)

    assert {chat_id for _profile, chat_id, _offset in second.message_calls} == {"c0000", "c0001", "c0002"}
    assert report["profiles"]["p-tg"]["incremental_chats_without_marker"] == 3
    saved = read_checkpoint(checkpoint_dir)["profiles"]["wappi_telegram:p-tg"]
    assert len(saved["chat_markers"]) == 3


def test_telegram_and_max_profiles_do_not_share_checkpoint(tmp_path: Path) -> None:
    db_path, phase1, checkpoint_dir = prepare(tmp_path)
    tg_chats, tg_messages = build_universe(20, profile_id="p-tg", channel="telegram")
    max_chats, max_messages = build_universe(4, profile_id="p-max", channel="max")
    max_chats = [{**dict(chat), "id": str(chat["id"]), "type": "DIALOG"} for chat in max_chats]
    client = CheckpointFakeClient({"p-tg": tg_chats, "p-max": max_chats}, {**tg_messages, **max_messages})

    run_wappi_history_import(
        make_config(tmp_path, db_path=db_path, phase1=phase1, checkpoint_dir=checkpoint_dir),
        client=client,
    )

    state = read_checkpoint(checkpoint_dir)["profiles"]
    assert set(state) == {"wappi_telegram:p-tg", "wappi_max:p-max"}
    # Same chat ids on both channels must NOT be confused for one another.
    assert state["wappi_telegram:p-tg"]["fingerprint"] != state["wappi_max:p-max"]["fingerprint"]
    assert state["wappi_max:p-max"]["catalog_chats_seen"] == 4
    assert state["wappi_telegram:p-tg"]["catalog_chats_seen"] == 20
    assert wappi_fetch_universe_fingerprint(
        WappiProfileSpec(profile_id="p-max", brand="unpk", channel="max"),
        WappiFetchLimits(page_size=10, complete_message_history=True, sleep_seconds=0),
        tenant_id="foton",
    ) == state["wappi_max:p-max"]["fingerprint"]


def test_first_profile_cannot_starve_the_second_one(tmp_path: Path) -> None:
    """Regression: with a global budget the alphabetically first profile ate
    everything every night and the second channel never loaded at all."""
    db_path, phase1, checkpoint_dir = prepare(tmp_path)
    tg_chats, tg_messages = build_universe(30, profile_id="p-tg", channel="telegram")
    max_chats, max_messages = build_universe(3, profile_id="p-max", channel="max")
    max_chats = [{"id": str(chat["id"]), "type": "DIALOG"} for chat in max_chats]
    universe = {**tg_messages, **max_messages}

    max_rows = 0
    for _ in range(6):
        run_wappi_history_import(
            make_config(
                tmp_path, db_path=db_path, phase1=phase1,
                checkpoint_dir=checkpoint_dir, request_limit_total=17,
            ),
            client=CheckpointFakeClient({"p-tg": tg_chats, "p-max": max_chats}, universe),
        )
        with sqlite3.connect(db_path) as con:
            max_rows = int(
                con.execute(
                    "SELECT COUNT(*) FROM timeline_events WHERE source_system = 'wappi_max'"
                ).fetchone()[0]
            )
        if max_rows >= 3:
            break

    assert max_rows == 3, "the MAX profile must make progress, not starve behind Telegram"


def test_corrupted_entry_types_are_ignored_not_crashed(tmp_path: Path) -> None:
    """A checkpoint with a valid schema but garbage field types must degrade like a
    corrupted file, otherwise the nightly step fails the same way every night."""
    db_path, phase1, checkpoint_dir = prepare(tmp_path)
    chats, messages = build_universe(4)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    wappi_history_checkpoint_path(checkpoint_dir).write_text(
        json.dumps(
            {
                "schema_version": WAPPI_HISTORY_CHECKPOINT_SCHEMA_VERSION,
                "profiles": {
                    "wappi_telegram:p-tg": {
                        "fingerprint": "x", "chats_done": [], "tail_checked": [],
                        "timeline_rows": 0, "catalog_next_offset": 0,
                        "active_chat": {"chat": "digest", "message_offset": "не число", "page_anchor": "x"},
                    },
                    "wappi_max:p-max": {"fingerprint": "y", "chats_done": ["ok"], "catalog_next_offset": "мусор"},
                    "broken-list": {"chats_done": [], "full_audit_markers": ["broken"]},
                    "broken-string": {"chats_done": [], "full_audit_markers": "broken"},
                    "broken-null": {"chats_done": [], "full_audit_markers": None},
                    "broken-values": {"chats_done": [], "full_audit_markers": {"x": "broken"}},
                    "broken-complete": {"chats_done": [], "complete": 1},
                    "broken-cycle": {"chats_done": [], "incremental_cycle": "true"},
                    "broken-infinity": {"chats_done": [], "chat_markers": {"x": float("inf")}},
                    "broken-negative-infinity": {"chats_done": [], "chat_markers": {"x": float("-inf")}},
                    "broken-nan": {"chats_done": [], "chat_markers": {"x": float("nan")}},
                    "broken-bool-true": {"chats_done": [], "chat_markers": {"x": True}},
                    "broken-bool-false": {"chats_done": [], "chat_markers": {"x": False}},
                    "broken-float": {"chats_done": [], "chat_markers": {"x": 1.5}},
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    assert not load_wappi_history_checkpoint(checkpoint_dir).get("profiles")

    report = run_wappi_history_import(
        make_config(tmp_path, db_path=db_path, phase1=phase1, checkpoint_dir=checkpoint_dir),
        client=CheckpointFakeClient({"p-tg": chats, "p-max": []}, messages),
    )

    assert report["checkpoint"]["complete"] is True
    assert wappi_row_count(db_path) == 4


def test_checkpoint_integer_string_marker_remains_compatible(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    wappi_history_checkpoint_path(checkpoint_dir).write_text(
        json.dumps({
            "schema_version": WAPPI_HISTORY_CHECKPOINT_SCHEMA_VERSION,
            "profiles": {
                "wappi_telegram:p-tg": {
                    "chats_done": [], "chat_markers": {"digest": "1753000000"},
                    "full_audit_markers": {"digest": "0"}, "active_chat": None,
                    "timeline_rows": 0, "catalog_next_offset": 0,
                }
            },
        }),
        encoding="utf-8",
    )

    state = load_wappi_history_checkpoint(checkpoint_dir)["profiles"]["wappi_telegram:p-tg"]
    assert state["chat_markers"]["digest"] == "1753000000"


def test_checkpoint_with_truncated_utf8_is_ignored(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    wappi_history_checkpoint_path(checkpoint_dir).write_bytes(b'{"schema_version":"\xff')

    assert load_wappi_history_checkpoint(checkpoint_dir) == {}


def test_checkpoint_with_unknown_schema_is_ignored(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    wappi_history_checkpoint_path(checkpoint_dir).write_text(
        json.dumps({"schema_version": "future", "profiles": {}}),
        encoding="utf-8",
    )

    assert load_wappi_history_checkpoint(checkpoint_dir) == {}


def test_untouched_profile_keeps_its_progress_on_save(tmp_path: Path) -> None:
    """Saving must merge, not overwrite: a profile missing from this run's config
    must not lose the progress a previous run confirmed."""
    db_path, phase1, checkpoint_dir = prepare(tmp_path)
    chats, messages = build_universe(20)
    run_wappi_history_import(
        make_config(tmp_path, db_path=db_path, phase1=phase1, checkpoint_dir=checkpoint_dir, request_limit_total=12),
        client=CheckpointFakeClient({"p-tg": chats, "p-max": []}, messages),
    )
    state = json.loads(wappi_history_checkpoint_path(checkpoint_dir).read_text(encoding="utf-8"))
    state["profiles"]["wappi_telegram:p-gone"] = {
        "fingerprint": "legacy", "chats_done": ["deadbeef"], "timeline_rows": 0, "complete": False,
    }
    wappi_history_checkpoint_path(checkpoint_dir).write_text(
        json.dumps(state, ensure_ascii=False), encoding="utf-8"
    )

    run_wappi_history_import(
        make_config(tmp_path, db_path=db_path, phase1=phase1, checkpoint_dir=checkpoint_dir, request_limit_total=20),
        client=CheckpointFakeClient({"p-tg": chats, "p-max": []}, messages),
    )

    survivors = read_checkpoint(checkpoint_dir).get("profiles") or {}
    assert "wappi_telegram:p-gone" in survivors


def test_page_size_change_resets_incompatible_checkpoint(tmp_path: Path) -> None:
    db_path, phase1, checkpoint_dir = prepare(tmp_path)
    chats, messages = build_universe(20)
    run_wappi_history_import(
        make_config(tmp_path, db_path=db_path, phase1=phase1, checkpoint_dir=checkpoint_dir, request_limit_total=12),
        client=CheckpointFakeClient({"p-tg": chats, "p-max": []}, messages),
    )
    confirmed_before = read_checkpoint(checkpoint_dir)["profiles"]["wappi_telegram:p-tg"]["chats_done"]
    assert confirmed_before

    resized = CheckpointFakeClient({"p-tg": chats, "p-max": []}, messages)
    report = run_wappi_history_import(
        make_config(
            tmp_path, db_path=db_path, phase1=phase1, checkpoint_dir=checkpoint_dir,
            request_limit_total=12, page_size=20,
        ),
        client=resized,
    )

    tg_state = report["checkpoint"]["profiles"]["wappi_telegram:p-tg"]
    assert tg_state["reset_reason"] == "fingerprint_changed"
    assert tg_state["resumed_from"] == 0
    refetched = {chat_id for _profile, chat_id, _offset in resized.message_calls}
    assert refetched & {
        chat["id"] for chat in chats if wappi_checkpoint_token(str(chat["id"])) in set(confirmed_before)
    }


def test_checkpoint_file_contains_no_personal_data(tmp_path: Path) -> None:
    db_path, phase1, checkpoint_dir = prepare(tmp_path)
    chats, messages = build_universe(20, body="Иван Петров +7 900 000-00-00 ivan@example.invalid")
    run_wappi_history_import(
        make_config(tmp_path, db_path=db_path, phase1=phase1, checkpoint_dir=checkpoint_dir, request_limit_total=12),
        client=CheckpointFakeClient({"p-tg": chats, "p-max": []}, messages),
    )

    raw = wappi_history_checkpoint_path(checkpoint_dir).read_text(encoding="utf-8")
    for forbidden in ("Иван", "Петров", "900 000", "@example.invalid", "token", "secret", "Здравствуйте"):
        assert forbidden not in raw
    for chat in chats:
        assert str(chat["id"]) not in raw  # raw chat ids are personal identifiers
    payload = json.loads(raw)
    assert payload["schema_version"] == WAPPI_HISTORY_CHECKPOINT_SCHEMA_VERSION
    assert wappi_history_checkpoint_path(checkpoint_dir).stat().st_mode & 0o077 == 0


def test_terminal_complete_keeps_incremental_state(tmp_path: Path) -> None:
    db_path, phase1, checkpoint_dir = prepare(tmp_path)
    chats, messages = build_universe(20)
    run_wappi_history_import(
        make_config(tmp_path, db_path=db_path, phase1=phase1, checkpoint_dir=checkpoint_dir, request_limit_total=12),
        client=CheckpointFakeClient({"p-tg": chats, "p-max": []}, messages),
    )
    assert wappi_history_checkpoint_path(checkpoint_dir).exists()

    report = run_wappi_history_import(
        make_config(tmp_path, db_path=db_path, phase1=phase1, checkpoint_dir=checkpoint_dir),
        client=CheckpointFakeClient({"p-tg": chats, "p-max": []}, messages),
    )

    assert report["checkpoint"]["complete"] is True
    assert report["checkpoint"]["profiles"]["wappi_telegram:p-tg"]["stop_reason"] == "source_exhausted"
    assert report["validation_ok"] is True
    assert wappi_history_checkpoint_path(checkpoint_dir).exists()
    saved = read_checkpoint(checkpoint_dir)["profiles"]["wappi_telegram:p-tg"]
    assert saved["complete"] is True
    assert len(saved["chat_markers"]) == 20


def test_checkpoint_disabled_keeps_current_fail_closed_behaviour(tmp_path: Path) -> None:
    db_path, phase1, _ = prepare(tmp_path)
    chats, messages = build_universe(20)
    client = CheckpointFakeClient({"p-tg": chats, "p-max": []}, messages)

    report = run_wappi_history_import(
        make_config(tmp_path, db_path=db_path, phase1=phase1, checkpoint_dir=None, request_limit_total=12),
        client=client,
    )

    assert report["checkpoint"]["enabled"] is False
    assert report["mode"] == "apply_blocked"
    assert report["validation_ok"] is False
    assert wappi_row_count(db_path) == 0


def test_pagination_drift_blocks_write_and_checkpoint(tmp_path: Path) -> None:
    db_path, phase1, checkpoint_dir = prepare(tmp_path)
    chats, messages = build_universe(2, messages_per_chat=20)

    class DriftingClient(CheckpointFakeClient):
        def get_chat_messages(self, **kwargs: Any) -> Mapping[str, Any]:
            payload = super().get_chat_messages(**kwargs)
            items = list(payload.get("messages") or ())
            if kwargs.get("chat_id") == "c0000" and kwargs.get("offset") == 0 and items:
                self.message_calls.append(("drift", "c0000", -1))
                if len([call for call in self.message_calls if call[0] == "drift"]) > 1:
                    return {"messages": [{**dict(items[0]), "id": "drifted"}, *items[1:]]}
            return payload

    report = run_wappi_history_import(
        make_config(tmp_path, db_path=db_path, phase1=phase1, checkpoint_dir=checkpoint_dir),
        client=DriftingClient({"p-tg": chats, "p-max": []}, messages),
    )

    assert any("pagination_drift_detected" in marker for marker in report["limit_hits"])
    assert report["mode"] == "apply_blocked"  # drift is never deferred
    assert report["validation_ok"] is False
    assert report["checkpoint"]["committed"] is False
    assert not wappi_history_checkpoint_path(checkpoint_dir).exists()
    assert wappi_row_count(db_path) == 0


def test_checkpoint_dropped_when_confirmed_rows_disappear(tmp_path: Path) -> None:
    """The staging DB was rebuilt: rows the checkpoint vouches for are gone."""
    checkpoint = {
        "schema_version": WAPPI_HISTORY_CHECKPOINT_SCHEMA_VERSION,
        "profiles": {
            "wappi_telegram:p-tg": {"fingerprint": "x", "chats_done": ["a"], "timeline_rows": 500},
            "wappi_max:p-max": {"fingerprint": "y", "chats_done": ["b"], "timeline_rows": 2},
        },
    }

    usable = usable_wappi_checkpoint_profiles(
        checkpoint, db_row_counts={"wappi_telegram:p-tg": 10, "wappi_max:p-max": 7}
    )

    assert "wappi_telegram:p-tg" not in usable
    assert "wappi_max:p-max" in usable


def test_wappi_timeline_state_on_missing_db(tmp_path: Path) -> None:
    profiles = (
        WappiProfileSpec(profile_id="p-tg", brand="foton", channel="telegram"),
        WappiProfileSpec(profile_id="p-max", brand="unpk", channel="max"),
    )
    state = wappi_timeline_state(tmp_path / "nope.sqlite", tenant_id="foton", profiles=profiles)
    assert {key: value["rows"] for key, value in state.items()} == {
        "wappi_telegram:p-tg": 0,
        "wappi_max:p-max": 0,
    }


def test_anchor_probe_that_eats_the_last_request_never_confirms_the_chat(tmp_path: Path) -> None:
    """Regression: the resume anchor probe must not spend the last request and then
    let the caller mark a chat done with zero new messages read."""
    db_path, phase1, checkpoint_dir = prepare(tmp_path)
    chats, messages = build_universe(1, messages_per_chat=30)
    run_wappi_history_import(
        make_config(tmp_path, db_path=db_path, phase1=phase1, checkpoint_dir=checkpoint_dir, request_limit_total=7),
        client=CheckpointFakeClient({"p-tg": chats, "p-max": []}, messages),
    )
    assert wappi_row_count(db_path) == 10

    # Budget for run 2: p-max 2 + two proved catalogue passes (4) + exactly one
    # message request, which the anchor probe consumes.
    starved = CheckpointFakeClient({"p-tg": chats, "p-max": []}, messages)
    report = run_wappi_history_import(
        make_config(tmp_path, db_path=db_path, phase1=phase1, checkpoint_dir=checkpoint_dir, request_limit_total=7),
        client=starved,
    )

    tg_state = report["checkpoint"]["profiles"]["wappi_telegram:p-tg"]
    assert tg_state["chats_done"] == 0, "chat confirmed without reading a single new message"
    assert tg_state["complete"] is False
    assert report["validation_ok"] is False
    assert wappi_history_checkpoint_path(checkpoint_dir).exists()

    finisher = CheckpointFakeClient({"p-tg": chats, "p-max": []}, messages)
    run_wappi_history_import(
        make_config(tmp_path, db_path=db_path, phase1=phase1, checkpoint_dir=checkpoint_dir),
        client=finisher,
    )
    assert wappi_row_count(db_path) == 30
    assert wappi_history_checkpoint_path(checkpoint_dir).exists()


def test_sibling_profile_growth_cannot_mask_another_profiles_lost_rows(tmp_path: Path) -> None:
    """Regression: confirmed-row accounting is per profile, not per channel."""
    checkpoint = {
        "schema_version": WAPPI_HISTORY_CHECKPOINT_SCHEMA_VERSION,
        "profiles": {
            "wappi_telegram:tg-a": {"fingerprint": "a", "chats_done": ["x"], "timeline_rows": 5},
            "wappi_telegram:tg-b": {"fingerprint": "b", "chats_done": ["y"], "timeline_rows": 7},
        },
    }

    usable = usable_wappi_checkpoint_profiles(
        checkpoint,
        # tg-a grew, tg-b lost rows: the channel total would still look healthy.
        db_row_counts={"wappi_telegram:tg-a": 40, "wappi_telegram:tg-b": 3},
    )

    assert "wappi_telegram:tg-a" in usable
    assert "wappi_telegram:tg-b" not in usable


def test_same_row_count_with_different_source_ids_drops_checkpoint() -> None:
    checkpoint = {
        "schema_version": WAPPI_HISTORY_CHECKPOINT_SCHEMA_VERSION,
        "profiles": {
            "wappi_telegram:p-tg": {
                "fingerprint": "x", "chats_done": ["a"], "timeline_rows": 10,
                "timeline_source_digest": "old",
            }
        },
    }

    usable = usable_wappi_checkpoint_profiles(
        checkpoint,
        db_row_counts={"wappi_telegram:p-tg": 10},
        db_source_digests={"wappi_telegram:p-tg": "different"},
    )

    assert not usable


def test_timeline_state_digest_changes_when_source_ids_change(tmp_path: Path) -> None:
    db_path, phase1, checkpoint_dir = prepare(tmp_path)
    chats, messages = build_universe(2)
    run_wappi_history_import(
        make_config(tmp_path, db_path=db_path, phase1=phase1, checkpoint_dir=checkpoint_dir),
        client=CheckpointFakeClient({"p-tg": chats, "p-max": []}, messages),
    )
    profiles = (WappiProfileSpec(profile_id="p-tg", brand="foton", channel="telegram"),)
    before = wappi_timeline_state(db_path, tenant_id="foton", profiles=profiles)
    with sqlite3.connect(db_path) as con:
        con.execute(
            "UPDATE timeline_events SET source_id = ? WHERE rowid = ("
            "SELECT rowid FROM timeline_events WHERE source_system = 'wappi_telegram' LIMIT 1)",
            ("p-tg:changed:source",),
        )
    after = wappi_timeline_state(db_path, tenant_id="foton", profiles=profiles)

    assert before["wappi_telegram:p-tg"]["rows"] == after["wappi_telegram:p-tg"]["rows"]
    assert before["wappi_telegram:p-tg"]["source_digest"] != after["wappi_telegram:p-tg"]["source_digest"]


@pytest.mark.parametrize("request_limit", (10, 22))
def test_catalog_larger_than_budget_fails_loud_instead_of_zero_progress_loop(
    tmp_path: Path, request_limit: int
) -> None:
    db_path, phase1, checkpoint_dir = prepare(tmp_path)
    chats, messages = build_universe(100)

    report = run_wappi_history_import(
        make_config(
            tmp_path, db_path=db_path, phase1=phase1,
            checkpoint_dir=checkpoint_dir, request_limit_total=request_limit,
        ),
        client=CheckpointFakeClient({"p-tg": chats, "p-max": []}, messages),
    )

    assert report["mode"] == "apply_blocked"
    assert any(marker.endswith("checkpoint_no_progress") for marker in report["limit_hits"])
    assert not wappi_history_checkpoint_path(checkpoint_dir).exists()
    assert wappi_row_count(db_path) == 0


def test_short_catalog_pages_continue_until_reported_total(tmp_path: Path) -> None:
    db_path, phase1, checkpoint_dir = prepare(tmp_path)
    chats, messages = build_universe(7)

    class ShortPageClient(CheckpointFakeClient):
        def list_chats(self, **kwargs: Any) -> Mapping[str, Any]:
            kwargs = dict(kwargs)
            kwargs["limit"] = min(3, int(kwargs.get("limit") or 3))
            payload = dict(super().list_chats(**kwargs))
            payload["total_count"] = len(self.chats.get(str(kwargs["profile_id"]), []))
            return payload

    report = run_wappi_history_import(
        make_config(tmp_path, db_path=db_path, phase1=phase1, checkpoint_dir=checkpoint_dir),
        client=ShortPageClient({"p-tg": chats, "p-max": []}, messages),
    )

    assert report["validation_ok"] is True
    assert wappi_row_count(db_path) == 7


def test_overlapping_catalog_pages_cannot_claim_complete(tmp_path: Path) -> None:
    db_path, phase1, checkpoint_dir = prepare(tmp_path)
    chats, messages = build_universe(7)

    class OverlapClient(CheckpointFakeClient):
        def list_chats(self, **kwargs: Any) -> Mapping[str, Any]:
            profile_id = str(kwargs["profile_id"])
            offset = int(kwargs.get("offset") or 0)
            self.chat_calls.append((profile_id, offset, int(kwargs.get("limit") or 0)))
            items = self.chats.get(profile_id, [])
            pages = {0: items[0:3], 3: items[2:5], 6: items[5:6]}
            return {"dialogs": pages.get(offset, []), "total_count": len(items)}

    report = run_wappi_history_import(
        make_config(tmp_path, db_path=db_path, phase1=phase1, checkpoint_dir=checkpoint_dir),
        client=OverlapClient({"p-tg": chats, "p-max": []}, messages),
    )

    assert report["validation_ok"] is False
    assert any(marker.endswith("pagination_drift_detected") for marker in report["limit_hits"])
    assert wappi_row_count(db_path) == 0


def test_complete_history_requires_catalog_total_count(tmp_path: Path) -> None:
    db_path, phase1, checkpoint_dir = prepare(tmp_path)
    chats, messages = build_universe(2)

    class NoTotalClient(CheckpointFakeClient):
        def list_chats(self, **kwargs: Any) -> Mapping[str, Any]:
            payload = dict(super().list_chats(**kwargs))
            payload.pop("total_count", None)
            return payload

    report = run_wappi_history_import(
        make_config(tmp_path, db_path=db_path, phase1=phase1, checkpoint_dir=checkpoint_dir),
        client=NoTotalClient({"p-tg": chats, "p-max": []}, messages),
    )

    assert report["validation_ok"] is False
    assert any(marker.endswith("pagination_drift_detected") for marker in report["limit_hits"])
    assert wappi_row_count(db_path) == 0


def test_fingerprint_changes_with_brand_or_tenant() -> None:
    limits = WappiFetchLimits(complete_message_history=True)
    foton = WappiProfileSpec(profile_id="p", brand="foton", channel="telegram")
    unpk = WappiProfileSpec(profile_id="p", brand="unpk", channel="telegram")

    assert wappi_fetch_universe_fingerprint(foton, limits, tenant_id="foton") != wappi_fetch_universe_fingerprint(
        unpk, limits, tenant_id="foton"
    )
    assert wappi_fetch_universe_fingerprint(foton, limits, tenant_id="foton") != wappi_fetch_universe_fingerprint(
        foton, limits, tenant_id="other"
    )


def test_starved_budget_accumulates_history_then_larger_final_pass_confirms_it(tmp_path: Path) -> None:
    """A small budget may accumulate rows, but a sufficiently large final pass is
    required to recheck every confirmed chat before claiming completeness."""
    db_path, phase1, checkpoint_dir = prepare(tmp_path)
    chats, messages = build_universe(40, messages_per_chat=3)
    expected = sum(len(items) for items in messages.values())

    runs = 0
    progress: list[int] = []
    while runs < 40:
        runs += 1
        report = run_wappi_history_import(
            make_config(
                tmp_path, db_path=db_path, phase1=phase1,
                checkpoint_dir=checkpoint_dir, request_limit_total=16,
            ),
            client=CheckpointFakeClient({"p-tg": chats, "p-max": []}, messages),
        )
        progress.append(wappi_row_count(db_path))
        if progress[-1] == expected:
            break
        assert report["validation_ok"] is False  # never claims freshness mid-way
        assert wappi_history_checkpoint_path(checkpoint_dir).exists()

    assert wappi_row_count(db_path) == expected
    assert runs > 1, "the budget was supposed to force several runs"
    assert progress == sorted(progress), "every run must move forward, never backwards"

    report = run_wappi_history_import(
        make_config(tmp_path, db_path=db_path, phase1=phase1, checkpoint_dir=checkpoint_dir),
        client=CheckpointFakeClient({"p-tg": chats, "p-max": []}, messages),
    )

    assert report["checkpoint"]["complete"] is True
    assert report["validation_ok"] is True
    assert wappi_history_checkpoint_path(checkpoint_dir).exists()


def test_confirmed_catalog_page_drift_is_reported_honestly(tmp_path: Path) -> None:
    """The page a previous run confirmed changed between runs: say so, but keep the
    identity-based progress (a reordered catalogue must not cost a full re-read)."""
    db_path, phase1, checkpoint_dir = prepare(tmp_path)
    chats, messages = build_universe(20)
    run_wappi_history_import(
        make_config(tmp_path, db_path=db_path, phase1=phase1, checkpoint_dir=checkpoint_dir, request_limit_total=12),
        client=CheckpointFakeClient({"p-tg": chats, "p-max": []}, messages),
    )
    saved = read_checkpoint(checkpoint_dir)["profiles"]["wappi_telegram:p-tg"]
    assert saved["catalog_page_anchor"] and saved["catalog_next_offset"] > 0
    confirmed_before = saved["chats_done"]

    # Rewrite the very first catalogue page under the checkpoint's feet.
    shuffled = [{"id": f"n{index:04d}", "type": "user"} for index in range(10)] + chats[10:]
    for chat in shuffled[:10]:
        messages[("telegram", "p-tg", str(chat["id"]))] = [
            {"id": f"{chat['id']}-m000", "chat_id": str(chat["id"]), "type": "text",
             "body": "Сдвиг", "time": 1_753_000_700}
        ]
    report = run_wappi_history_import(
        make_config(tmp_path, db_path=db_path, phase1=phase1, checkpoint_dir=checkpoint_dir),
        client=CheckpointFakeClient({"p-tg": shuffled, "p-max": []}, messages),
    )

    tg_state = report["checkpoint"]["profiles"]["wappi_telegram:p-tg"]
    assert tg_state["reset_reason"] == "catalog_page_drift"
    assert tg_state["resumed_from"] == len(confirmed_before)  # progress kept, not thrown away


def test_dry_run_never_commits_a_checkpoint(tmp_path: Path) -> None:
    db_path, phase1, checkpoint_dir = prepare(tmp_path)
    chats, messages = build_universe(20)

    report = run_wappi_history_import(
        make_config(
            tmp_path, db_path=db_path, phase1=phase1, checkpoint_dir=checkpoint_dir,
            request_limit_total=12, apply=False,
        ),
        client=CheckpointFakeClient({"p-tg": chats, "p-max": []}, messages),
    )

    assert report["mode"] == "dry_run_preview"
    assert report["checkpoint"]["committed"] is False
    assert not wappi_history_checkpoint_path(checkpoint_dir).exists()
    assert wappi_row_count(db_path) == 0


def test_checkpoint_does_not_advance_when_current_source_id_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path, phase1, checkpoint_dir = prepare(tmp_path)
    chats, messages = build_universe(2)
    monkeypatch.setattr(
        wappi_history_module,
        "load_existing_wappi_source_ids",
        lambda *_args, **_kwargs: set(),
    )

    report = run_wappi_history_import(
        make_config(tmp_path, db_path=db_path, phase1=phase1, checkpoint_dir=checkpoint_dir),
        client=CheckpointFakeClient({"p-tg": chats, "p-max": []}, messages),
    )

    assert report["source_persistence_complete"] is False
    assert report["publish_ready"] is False
    assert report["summary"]["messages_missing_from_timeline"] > 0
    assert report["checkpoint"]["committed"] is False
    assert not wappi_history_checkpoint_path(checkpoint_dir).exists()


def test_catalog_snapshot_drift_blocks_terminal_complete(tmp_path: Path) -> None:
    """Regression: catalogue drift must not be reported as a finished profile."""
    db_path, phase1, checkpoint_dir = prepare(tmp_path)
    chats, messages = build_universe(2)

    class ReorderingClient(CheckpointFakeClient):
        def __init__(self, chats_map: Any, messages_map: Any) -> None:
            super().__init__(chats_map, messages_map)
            self.second_page_calls = 0

        def list_chats(self, **kwargs: Any) -> Mapping[str, Any]:
            if kwargs.get("profile_id") == "p-tg" and kwargs.get("offset") == 0:
                self.second_page_calls += 1
                if self.second_page_calls >= 2:
                    self.chats["p-tg"] = [{"id": "c9999", "type": "user"}, *self.chats["p-tg"][1:]]
            return super().list_chats(**kwargs)

    messages[("telegram", "p-tg", "c9999")] = [
        {"id": "c9999-m000", "chat_id": "c9999", "type": "text", "body": "Появился", "time": 1_753_000_900}
    ]
    report = run_wappi_history_import(
        make_config(tmp_path, db_path=db_path, phase1=phase1, checkpoint_dir=checkpoint_dir),
        client=ReorderingClient({"p-tg": chats, "p-max": []}, messages),
    )

    tg_state = report["checkpoint"]["profiles"]["wappi_telegram:p-tg"]
    assert tg_state["complete"] is False
    assert tg_state["stop_reason"] == "catalog_drift"
    assert report["validation_ok"] is False
    assert not wappi_history_checkpoint_path(checkpoint_dir).exists()


def test_catalog_class_drift_stops_before_message_reads(tmp_path: Path) -> None:
    db_path, phase1, checkpoint_dir = prepare(tmp_path)
    chats, messages = build_universe(1)

    class ClassDriftClient(CheckpointFakeClient):
        p_tg_passes = 0

        def list_chats(self, **kwargs: Any) -> Mapping[str, Any]:
            if kwargs["profile_id"] == "p-tg" and int(kwargs.get("offset") or 0) == 0:
                self.p_tg_passes += 1
                if self.p_tg_passes == 2:
                    self.chats["p-tg"][0] = {**self.chats["p-tg"][0], "type": "group"}
            return super().list_chats(**kwargs)

    client = ClassDriftClient({"p-tg": chats, "p-max": []}, messages)
    report = run_wappi_history_import(
        make_config(tmp_path, db_path=db_path, phase1=phase1, checkpoint_dir=checkpoint_dir),
        client=client,
    )

    stats = report["profiles"]["p-tg"]
    assert stats["catalog_passes"] == 2
    assert stats["chat_snapshot_drift_detected"] is True
    assert stats["pagination_drift_detected"] is True
    assert client.message_calls == []
    assert report["validation_ok"] is False


def test_oversized_whole_catalog_is_verified_by_two_independent_passes(
    tmp_path: Path,
) -> None:
    db_path, phase1, checkpoint_dir = prepare(tmp_path)
    chats, messages = build_universe(2)

    class WholeSnapshotClient(CheckpointFakeClient):
        def list_chats(self, **kwargs: Any) -> Mapping[str, Any]:
            profile_id = str(kwargs["profile_id"])
            self.chat_calls.append(
                (profile_id, int(kwargs.get("offset") or 0), int(kwargs.get("limit") or 0))
            )
            items = self.chats.get(profile_id, [])
            return {"dialogs": items, "total_count": len(items)}

    client = WholeSnapshotClient({"p-tg": chats, "p-max": []}, messages)
    report = run_wappi_history_import(
        make_config(
            tmp_path,
            db_path=db_path,
            phase1=phase1,
            checkpoint_dir=checkpoint_dir,
            page_size=1,
        ),
        client=client,
    )

    stats = report["profiles"]["p-tg"]
    assert [offset for profile, offset, _limit in client.chat_calls if profile == "p-tg"] == [0, 0]
    assert stats["catalog_passes"] == 2
    assert stats["catalog_boundary_mode"] == "oversized_whole_snapshot"
    assert stats["catalog_boundary_proven"] is True
    assert stats["chat_snapshot_drift_detected"] is False
    assert report["validation_ok"] is True


def test_malformed_catalog_payload_is_not_treated_as_empty(tmp_path: Path) -> None:
    db_path, phase1, checkpoint_dir = prepare(tmp_path)

    class MalformedCatalogClient(CheckpointFakeClient):
        def list_chats(self, **kwargs: Any) -> Mapping[str, Any]:
            if kwargs["profile_id"] == "p-tg":
                return {"dialogs": "not-a-list", "total_count": 0}
            return super().list_chats(**kwargs)

    client = MalformedCatalogClient({"p-tg": [], "p-max": []}, {})
    report = run_wappi_history_import(
        make_config(tmp_path, db_path=db_path, phase1=phase1, checkpoint_dir=checkpoint_dir),
        client=client,
    )

    assert report["profiles"]["p-tg"]["pagination_drift_detected"] is True
    assert client.message_calls == []
    assert report["validation_ok"] is False


def test_full_history_duplicate_across_pages_is_pagination_drift() -> None:
    class CrossPageDuplicateClient:
        def __init__(self) -> None:
            self.offsets: list[int] = []

        def get_chat_messages(self, **kwargs: Any) -> Mapping[str, Any]:
            offset = int(kwargs["offset"])
            self.offsets.append(offset)
            ids = {0: ("m1", "m2"), 2: ("m2", "m3")}.get(offset, ())
            return {
                "messages": [
                    {
                        "id": message_id,
                        "chat_id": "chat",
                        "type": "text",
                        "body": message_id,
                        "time": index + 1,
                    }
                    for index, message_id in enumerate(ids)
                ]
            }

    client = CrossPageDuplicateClient()
    stats = wappi_history_module.WappiFetchStats()
    wappi_history_module.fetch_chat_messages(
        client,
        profile=WappiProfileSpec(profile_id="p-tg", brand="foton", channel="telegram"),
        chat_id="chat",
        limits=WappiFetchLimits(page_size=2, complete_message_history=True, sleep_seconds=0),
        request_counter=stats,
        request_budget=4,
    )

    assert client.offsets == [0, 2]
    assert wappi_history_module.fetch_chat_messages.last_pagination_drift_detected is True


def test_delta_tail_rejects_more_than_exactly_one_overlap() -> None:
    class ExtraOverlapClient:
        def __init__(self) -> None:
            self.offsets: list[int] = []

        def get_chat_messages(self, **kwargs: Any) -> Mapping[str, Any]:
            offset = int(kwargs["offset"])
            self.offsets.append(offset)
            ids = {0: ("m5", "m4", "m3"), 2: ("m4", "m3", "m2")}.get(offset, ())
            return {
                "messages": [
                    {
                        "id": message_id,
                        "chat_id": "chat",
                        "type": "text",
                        "body": message_id,
                        "time": 100 - index,
                    }
                    for index, message_id in enumerate(ids)
                ]
            }

    client = ExtraOverlapClient()
    stats = wappi_history_module.WappiFetchStats()
    wappi_history_module.fetch_chat_messages(
        client,
        profile=WappiProfileSpec(profile_id="p-tg", brand="foton", channel="telegram"),
        chat_id="chat",
        limits=WappiFetchLimits(page_size=3, complete_message_history=True, sleep_seconds=0),
        request_counter=stats,
        request_budget=3,
        stop_after_message_token=wappi_history_module.wappi_message_checkpoint_token(
            "p-tg", "chat", "m0"
        ),
    )

    assert client.offsets == [0, 2]
    assert wappi_history_module.fetch_chat_messages.last_pagination_drift_detected is True
    assert wappi_history_module.fetch_chat_messages.last_tail_drift_reason == "unexpected_overlap"


def test_delta_tail_accepts_metadata_only_append_after_matching_head_proof() -> None:
    message = {
        "id": "m1",
        "chat_id": "chat",
        "type": "text",
        "body": "Без изменений",
        "time": 100,
    }

    class StableHeadClient:
        def __init__(self) -> None:
            self.offsets: list[int] = []

        def get_chat_messages(self, **kwargs: Any) -> Mapping[str, Any]:
            self.offsets.append(int(kwargs["offset"]))
            return {"messages": [message]}

    client = StableHeadClient()
    rows = wappi_history_module.fetch_chat_messages(
        client,
        profile=WappiProfileSpec(profile_id="p-tg", brand="foton", channel="telegram"),
        chat_id="chat",
        limits=WappiFetchLimits(page_size=10, complete_message_history=True, sleep_seconds=0),
        request_counter=wappi_history_module.WappiFetchStats(),
        request_budget=2,
        stop_after_message_token=wappi_history_module.wappi_message_checkpoint_token(
            "p-tg", "chat", "m1"
        ),
    )

    assert rows == ()
    assert client.offsets == [0, 0]
    assert wappi_history_module.fetch_chat_messages.last_boundary_found is True
    assert wappi_history_module.fetch_chat_messages.last_pagination_drift_detected is False
    first_signature = wappi_history_module.fetch_chat_messages.last_tail_first_signature
    head_signature = wappi_history_module.fetch_chat_messages.last_tail_head_signature
    assert len(first_signature) == 64
    assert head_signature == first_signature


def test_delta_tail_empty_append_requires_budget_for_head_proof() -> None:
    class BoundaryHeadClient:
        def __init__(self) -> None:
            self.calls = 0

        def get_chat_messages(self, **_kwargs: Any) -> Mapping[str, Any]:
            self.calls += 1
            return {
                "messages": [
                    {
                        "id": "m1",
                        "chat_id": "chat",
                        "type": "text",
                        "body": "Сохранённая граница",
                        "time": 100,
                    }
                ]
            }

    client = BoundaryHeadClient()
    rows = wappi_history_module.fetch_chat_messages(
        client,
        profile=WappiProfileSpec(profile_id="p-tg", brand="foton", channel="telegram"),
        chat_id="chat",
        limits=WappiFetchLimits(page_size=10, complete_message_history=True, sleep_seconds=0),
        request_counter=wappi_history_module.WappiFetchStats(),
        request_budget=1,
        stop_after_message_token=wappi_history_module.wappi_message_checkpoint_token(
            "p-tg", "chat", "m1"
        ),
    )

    assert rows == ()
    assert client.calls == 1
    assert wappi_history_module.fetch_chat_messages.last_pagination_drift_detected is True
    assert wappi_history_module.fetch_chat_messages.last_tail_drift_reason == "head_proof_budget"


def test_delta_tail_blocks_empty_append_when_head_changes_during_proof() -> None:
    boundary = {
        "id": "m1",
        "chat_id": "chat",
        "type": "text",
        "body": "Сохранённая граница",
        "time": 100,
    }
    new_message = {
        "id": "m2",
        "chat_id": "chat",
        "type": "text",
        "body": "Новое сообщение",
        "time": 101,
    }

    class MovingHeadClient:
        def __init__(self) -> None:
            self.offsets: list[int] = []

        def get_chat_messages(self, **kwargs: Any) -> Mapping[str, Any]:
            self.offsets.append(int(kwargs["offset"]))
            rows = [boundary] if len(self.offsets) == 1 else [new_message, boundary]
            return {"messages": rows}

    client = MovingHeadClient()
    rows = wappi_history_module.fetch_chat_messages(
        client,
        profile=WappiProfileSpec(profile_id="p-tg", brand="foton", channel="telegram"),
        chat_id="chat",
        limits=WappiFetchLimits(page_size=10, complete_message_history=True, sleep_seconds=0),
        request_counter=wappi_history_module.WappiFetchStats(),
        request_budget=2,
        stop_after_message_token=wappi_history_module.wappi_message_checkpoint_token(
            "p-tg", "chat", "m1"
        ),
    )

    assert rows == ()
    assert client.offsets == [0, 0]
    assert wappi_history_module.fetch_chat_messages.last_pagination_drift_detected is True
    assert wappi_history_module.fetch_chat_messages.last_tail_drift_reason == "head_changed"
    assert (
        wappi_history_module.fetch_chat_messages.last_tail_head_signature
        != wappi_history_module.fetch_chat_messages.last_tail_first_signature
    )


def test_delta_tail_rebases_stable_terminal_snapshot_when_boundary_disappears() -> None:
    visible = [
        {
            "id": "m2",
            "chat_id": "chat",
            "type": "text",
            "body": "Новое видимое сообщение",
            "time": 102,
        },
        {
            "id": "m1",
            "chat_id": "chat",
            "type": "text",
            "body": "Старое видимое сообщение",
            "time": 101,
        },
    ]

    class StableTerminalClient:
        def __init__(self) -> None:
            self.offsets: list[int] = []

        def get_chat_messages(self, **kwargs: Any) -> Mapping[str, Any]:
            self.offsets.append(int(kwargs["offset"]))
            return {"status": "done", "has_more": False, "messages": visible}

    client = StableTerminalClient()
    rows = wappi_history_module.fetch_chat_messages(
        client,
        profile=WappiProfileSpec(profile_id="p-tg", brand="foton", channel="telegram"),
        chat_id="chat",
        limits=WappiFetchLimits(page_size=10, complete_message_history=True, sleep_seconds=0),
        request_counter=wappi_history_module.WappiFetchStats(),
        request_budget=2,
        stop_after_message_token=wappi_history_module.wappi_message_checkpoint_token(
            "p-tg", "chat", "deleted-boundary"
        ),
    )

    assert [row.message_id for row in rows] == ["m1", "m2"]
    assert client.offsets == [0, 0]
    assert wappi_history_module.fetch_chat_messages.last_boundary_found is True
    assert wappi_history_module.fetch_chat_messages.last_pagination_drift_detected is False
    assert wappi_history_module.fetch_chat_messages.last_head_message_token == (
        wappi_history_module.wappi_message_checkpoint_token("p-tg", "chat", "m2")
    )


def test_delta_tail_terminal_rebase_fails_closed_when_head_changes() -> None:
    first = {
        "id": "m1", "chat_id": "chat", "type": "text", "body": "Первый", "time": 101,
    }
    changed = {
        "id": "m2", "chat_id": "chat", "type": "text", "body": "Новый", "time": 102,
    }

    class MovingTerminalClient:
        calls = 0

        def get_chat_messages(self, **_kwargs: Any) -> Mapping[str, Any]:
            self.calls += 1
            rows = [first] if self.calls == 1 else [changed, first]
            return {"status": "done", "has_more": False, "messages": rows}

    client = MovingTerminalClient()
    rows = wappi_history_module.fetch_chat_messages(
        client,
        profile=WappiProfileSpec(profile_id="p-tg", brand="foton", channel="telegram"),
        chat_id="chat",
        limits=WappiFetchLimits(page_size=10, complete_message_history=True, sleep_seconds=0),
        request_counter=wappi_history_module.WappiFetchStats(),
        request_budget=2,
        stop_after_message_token=wappi_history_module.wappi_message_checkpoint_token(
            "p-tg", "chat", "deleted-boundary"
        ),
    )

    assert rows == ()
    assert client.calls == 2
    assert wappi_history_module.fetch_chat_messages.last_pagination_drift_detected is True
    assert wappi_history_module.fetch_chat_messages.last_tail_drift_reason == "head_changed"


@pytest.mark.parametrize(
    ("second_payload", "reason"),
    (
        ({}, "malformed_head"),
        (
            {
                "status": "queued",
                "has_more": False,
                "messages": [
                    {"id": "m1", "chat_id": "chat", "type": "text", "body": "x", "time": 1}
                ],
            },
            "head_not_terminal",
        ),
    ),
)
def test_delta_tail_terminal_rebase_requires_valid_terminal_head(
    second_payload: Mapping[str, Any],
    reason: str,
) -> None:
    first_payload = {
        "status": "done",
        "has_more": False,
        "messages": [
            {"id": "m1", "chat_id": "chat", "type": "text", "body": "x", "time": 1}
        ],
    }

    class InvalidProofClient:
        calls = 0

        def get_chat_messages(self, **_kwargs: Any) -> Mapping[str, Any]:
            self.calls += 1
            return first_payload if self.calls == 1 else second_payload

    client = InvalidProofClient()
    rows = wappi_history_module.fetch_chat_messages(
        client,
        profile=WappiProfileSpec(profile_id="p-tg", brand="foton", channel="telegram"),
        chat_id="chat",
        limits=WappiFetchLimits(page_size=10, complete_message_history=True, sleep_seconds=0),
        request_counter=wappi_history_module.WappiFetchStats(),
        request_budget=2,
        stop_after_message_token=wappi_history_module.wappi_message_checkpoint_token(
            "p-tg", "chat", "deleted-boundary"
        ),
    )

    assert rows == ()
    assert client.calls == 2
    assert wappi_history_module.fetch_chat_messages.last_pagination_drift_detected is True
    assert wappi_history_module.fetch_chat_messages.last_tail_drift_reason == reason


@pytest.mark.parametrize(
    "payload",
    (
        {"messages": [{"id": "m1", "chat_id": "chat", "type": "text", "body": "x", "time": 1}]},
        {"status": "queued", "has_more": False, "messages": [{"id": "m1", "chat_id": "chat", "type": "text", "body": "x", "time": 1}]},
        {"status": "done", "has_more": True, "messages": [{"id": "m1", "chat_id": "chat", "type": "text", "body": "x", "time": 1}]},
    ),
)
def test_delta_tail_missing_boundary_requires_proven_terminal_page(
    payload: Mapping[str, Any],
) -> None:
    class UnprovenClient:
        calls = 0

        def get_chat_messages(self, **_kwargs: Any) -> Mapping[str, Any]:
            self.calls += 1
            return payload

    client = UnprovenClient()
    rows = wappi_history_module.fetch_chat_messages(
        client,
        profile=WappiProfileSpec(profile_id="p-tg", brand="foton", channel="telegram"),
        chat_id="chat",
        limits=WappiFetchLimits(page_size=10, complete_message_history=True, sleep_seconds=0),
        request_counter=wappi_history_module.WappiFetchStats(),
        request_budget=2,
        stop_after_message_token=wappi_history_module.wappi_message_checkpoint_token(
            "p-tg", "chat", "deleted-boundary"
        ),
    )

    assert rows == ()
    assert client.calls == 1
    assert wappi_history_module.fetch_chat_messages.last_pagination_drift_detected is True
    assert (
        wappi_history_module.fetch_chat_messages.last_tail_drift_reason
        == "boundary_not_found_short_page"
    )


def test_delta_tail_terminal_rebase_requires_head_proof_budget() -> None:
    class TerminalClient:
        calls = 0

        def get_chat_messages(self, **_kwargs: Any) -> Mapping[str, Any]:
            self.calls += 1
            return {
                "status": "done",
                "has_more": False,
                "messages": [
                    {"id": "m1", "chat_id": "chat", "type": "text", "body": "x", "time": 1}
                ],
            }

    client = TerminalClient()
    rows = wappi_history_module.fetch_chat_messages(
        client,
        profile=WappiProfileSpec(profile_id="p-tg", brand="foton", channel="telegram"),
        chat_id="chat",
        limits=WappiFetchLimits(page_size=10, complete_message_history=True, sleep_seconds=0),
        request_counter=wappi_history_module.WappiFetchStats(),
        request_budget=1,
        stop_after_message_token=wappi_history_module.wappi_message_checkpoint_token(
            "p-tg", "chat", "deleted-boundary"
        ),
    )

    assert rows == ()
    assert client.calls == 1
    assert wappi_history_module.fetch_chat_messages.last_pagination_drift_detected is True
    assert wappi_history_module.fetch_chat_messages.last_tail_drift_reason == "head_proof_budget"


@pytest.mark.parametrize(
    "malformed_payload",
    ({}, {"messages": "not-a-list"}, {"messages": ["not-an-object"]}, {"data": {}}),
)
def test_empty_baseline_malformed_payload_is_blocking(
    malformed_payload: Mapping[str, Any],
) -> None:
    class MalformedClient(CheckpointFakeClient):
        def get_chat_messages(self, **kwargs: Any) -> Mapping[str, Any]:
            self.message_calls.append(
                (kwargs["profile_id"], kwargs["chat_id"], kwargs["offset"])
            )
            self.message_request_calls.append(
                (
                    kwargs["profile_id"], kwargs["chat_id"], kwargs["offset"],
                    kwargs["limit"], kwargs["order"],
                )
            )
            return malformed_payload

    client = MalformedClient({"p-tg": []}, {})
    stats = wappi_history_module.WappiFetchStats()
    rows = wappi_history_module.fetch_chat_messages(
        client,
        profile=WappiProfileSpec(profile_id="p-tg", brand="foton", channel="telegram"),
        chat_id="empty-chat",
        limits=WappiFetchLimits(page_size=10, complete_message_history=True, sleep_seconds=0),
        request_counter=stats,
        request_budget=2,
        allow_empty_tail=True,
        empty_baseline_tail=True,
    )

    assert rows == ()
    assert len(client.message_request_calls) == 1
    assert wappi_history_module.fetch_chat_messages.last_boundary_found is False
    assert wappi_history_module.fetch_chat_messages.last_pagination_drift_detected is True
    assert wappi_history_module.fetch_chat_messages.last_tail_drift_reason == "malformed_page"


def test_empty_baseline_cursor_sentinel_is_strict() -> None:
    assert wappi_history_module._wappi_chat_cursor_is_valid(
        {"empty_baseline": True, "timestamp": 0}
    )
    for invalid in (
        {"empty_baseline": True, "timestamp": 1},
        {"empty_baseline": True, "timestamp": 0, "message_digest": "x" * 64},
        {"empty_baseline": True},
        {"empty_baseline": True, "timestamp": "0"},
        {"empty_baseline": True, "timestamp": False},
        {"empty_baseline": True, "timestamp": 0, "message_digest": ""},
    ):
        assert not wappi_history_module._wappi_chat_cursor_is_valid(invalid)


def test_full_history_malformed_message_payload_is_blocking() -> None:
    class MalformedClient(CheckpointFakeClient):
        def get_chat_messages(self, **_kwargs: Any) -> Mapping[str, Any]:
            return {"messages": "not-a-list"}

    rows = wappi_history_module.fetch_chat_messages(
        MalformedClient({"p-tg": []}, {}),
        profile=WappiProfileSpec(profile_id="p-tg", brand="foton", channel="telegram"),
        chat_id="malformed-chat",
        limits=WappiFetchLimits(page_size=10, complete_message_history=True, sleep_seconds=0),
        request_counter=wappi_history_module.WappiFetchStats(),
        request_budget=2,
    )

    assert rows == ()
    assert wappi_history_module.fetch_chat_messages.last_pagination_drift_detected is True


def test_strict_full_snapshot_rechecks_exact_page_terminal() -> None:
    messages = {
        ("telegram", "p-tg", "chat"): [
            {
                "id": "m-1",
                "chat_id": "chat",
                "type": "text",
                "body": "Исходное",
                "time": 1,
            }
        ]
    }

    class TerminalGrowthClient(CheckpointFakeClient):
        calls = 0

        def get_chat_messages(self, **kwargs: Any) -> Mapping[str, Any]:
            self.calls += 1
            if self.calls == 4:
                self.messages[("telegram", "p-tg", "chat")].append(
                    {
                        "id": "m-2",
                        "chat_id": "chat",
                        "type": "text",
                        "body": "Появилось на прежней пустой границе",
                        "time": 2,
                    }
                )
            return super().get_chat_messages(**kwargs)

    client = TerminalGrowthClient({"p-tg": []}, messages)
    rows = wappi_history_module.fetch_chat_messages(
        client,
        profile=WappiProfileSpec(profile_id="p-tg", brand="foton", channel="telegram"),
        chat_id="chat",
        limits=WappiFetchLimits(
            page_size=1,
            complete_message_history=True,
            sleep_seconds=0,
        ),
        request_counter=wappi_history_module.WappiFetchStats(),
        request_budget=4,
        strict_snapshot_verification=True,
    )

    assert [row.message_id for row in rows] == ["m-1"]
    assert client.message_request_calls == [
        ("p-tg", "chat", 0, 1, "asc"),
        ("p-tg", "chat", 1, 1, "asc"),
        ("p-tg", "chat", 0, 1, "asc"),
        ("p-tg", "chat", 1, 1, "asc"),
    ]
    assert wappi_history_module.fetch_chat_messages.last_pagination_drift_detected is True


@pytest.mark.parametrize(
    "envelope_shape",
    ["direct", "nested", "sibling_pagination", "nested_meta"],
)
def test_strict_full_snapshot_never_accepts_short_page_with_more_hidden(
    envelope_shape: str,
) -> None:
    message = {
        "id": "m-1",
        "chat_id": "chat",
        "type": "text",
        "body": "Видимое",
        "time": 1,
    }

    class IncompletePageClient(CheckpointFakeClient):
        def get_chat_messages(self, **kwargs: Any) -> Mapping[str, Any]:
            self.message_request_calls.append(
                (
                    kwargs["profile_id"],
                    kwargs["chat_id"],
                    kwargs["offset"],
                    kwargs["limit"],
                    kwargs["order"],
                )
            )
            page = {
                "messages": [message] if kwargs["offset"] == 0 else [],
                "has_more": True,
            }
            if envelope_shape == "nested":
                return {
                    "data": {
                        "status": "ok",
                        "has_more": True,
                        "data": {"messages": page["messages"]},
                    }
                }
            if envelope_shape == "sibling_pagination":
                return {
                    "data": {"messages": page["messages"]},
                    "pagination": {"has_more": True},
                }
            if envelope_shape == "nested_meta":
                return {
                    "data": {
                        "messages": page["messages"],
                        "meta": {"has_more": True},
                    }
                }
            return page

    client = IncompletePageClient({"p-tg": []}, {})
    rows = wappi_history_module.fetch_chat_messages(
        client,
        profile=WappiProfileSpec(profile_id="p-tg", brand="foton", channel="telegram"),
        chat_id="chat",
        limits=WappiFetchLimits(
            page_size=10,
            complete_message_history=True,
            sleep_seconds=0,
        ),
        request_counter=wappi_history_module.WappiFetchStats(),
        request_budget=3,
        strict_snapshot_verification=True,
    )

    assert [row.message_id for row in rows] == ["m-1"]
    assert [request[2] for request in client.message_request_calls] == [0, 1]
    assert wappi_history_module.fetch_chat_messages.last_pagination_drift_detected is True


def test_full_history_accepts_proven_terminal_null_message_page() -> None:
    class TerminalNullClient(CheckpointFakeClient):
        def get_chat_messages(self, **_kwargs: Any) -> Mapping[str, Any]:
            return {"status": "done", "has_more": False, "messages": None}

    rows = wappi_history_module.fetch_chat_messages(
        TerminalNullClient({"p-max": []}, {}),
        profile=WappiProfileSpec(profile_id="p-max", brand="unpk", channel="max"),
        chat_id="empty-chat",
        limits=WappiFetchLimits(page_size=10, complete_message_history=True, sleep_seconds=0),
        request_counter=wappi_history_module.WappiFetchStats(),
        request_budget=2,
    )

    assert rows == ()
    assert wappi_history_module.fetch_chat_messages.last_pagination_drift_detected is False


def test_full_history_semantically_deduplicates_identical_ids() -> None:
    message = {
        "id": "m-1",
        "chat_id": "chat",
        "type": "text",
        "body": "Текст",
        "time": 1,
    }

    class DuplicateClient(CheckpointFakeClient):
        def get_chat_messages(self, **_kwargs: Any) -> Mapping[str, Any]:
            return {"messages": [message, {**message, "transport_only": "ignored"}]}

    rows = wappi_history_module.fetch_chat_messages(
        DuplicateClient({"p-tg": []}, {}),
        profile=WappiProfileSpec(profile_id="p-tg", brand="foton", channel="telegram"),
        chat_id="chat",
        limits=WappiFetchLimits(page_size=10, complete_message_history=True, sleep_seconds=0),
        request_counter=wappi_history_module.WappiFetchStats(),
        request_budget=3,
    )

    assert [row.message_id for row in rows] == ["m-1"]
    assert wappi_history_module.fetch_chat_messages.last_pagination_drift_detected is False


def test_empty_baseline_tail_accepts_proven_terminal_null_with_head_proof() -> None:
    class TerminalNullClient(CheckpointFakeClient):
        calls = 0

        def get_chat_messages(self, **_kwargs: Any) -> Mapping[str, Any]:
            self.calls += 1
            return {"status": "done", "has_more": False, "messages": None}

    client = TerminalNullClient({"p-max": []}, {})
    rows = wappi_history_module.fetch_chat_messages(
        client,
        profile=WappiProfileSpec(profile_id="p-max", brand="unpk", channel="max"),
        chat_id="empty-chat",
        limits=WappiFetchLimits(page_size=10, complete_message_history=True, sleep_seconds=0),
        request_counter=wappi_history_module.WappiFetchStats(),
        request_budget=2,
        allow_empty_tail=True,
        empty_baseline_tail=True,
    )

    assert rows == ()
    assert client.calls == 2
    assert wappi_history_module.fetch_chat_messages.last_boundary_found is True
    assert wappi_history_module.fetch_chat_messages.last_pagination_drift_detected is False


def test_regressed_marker_missing_boundary_stops_after_three_tail_pages(
    tmp_path: Path,
) -> None:
    db_path, phase1, checkpoint_dir = prepare(tmp_path)
    chats, messages = build_universe(1, messages_per_chat=500)
    config = make_config(
        tmp_path,
        db_path=db_path,
        phase1=phase1,
        checkpoint_dir=checkpoint_dir,
        page_size=100,
    )
    run_wappi_history_import(
        config,
        client=CheckpointFakeClient({"p-tg": chats, "p-max": []}, messages),
    )
    original_rows = wappi_row_count(db_path)
    messages[("telegram", "p-tg", "c0000")] = [
        {**dict(row), "id": f"replaced-{row['id']}"}
        for row in messages[("telegram", "p-tg", "c0000")]
    ]
    regressed = [{**dict(chats[0]), "last_timestamp": int(chats[0]["last_timestamp"]) - 1}]
    client = CheckpointFakeClient({"p-tg": regressed, "p-max": []}, messages)

    report = run_wappi_history_import(config, client=client)

    stats = report["profiles"]["p-tg"]
    assert report["validation_ok"] is False
    assert stats["message_page_drift_reason"] == "boundary_not_found_max_pages"
    assert stats["message_page_drift_pages"] == 3
    assert stats["message_page_drift_offset"] == 198
    assert stats["message_page_drift_cursor_kind"] == "message_digest"
    assert stats["message_page_drift_marker_relation"] == "regressed"
    assert len(stats["message_page_drift_first_signature"]) == 64
    assert stats["message_page_drift_head_signature"] == ""
    assert client.message_request_calls == [
        ("p-tg", "c0000", 0, 100, "desc"),
        ("p-tg", "c0000", 99, 100, "desc"),
        ("p-tg", "c0000", 198, 100, "desc"),
    ]
    assert wappi_row_count(db_path) == original_rows


def test_metadata_only_marker_append_commits_after_head_proof(tmp_path: Path) -> None:
    db_path, phase1, checkpoint_dir = prepare(tmp_path)
    chats, messages = build_universe(2)
    config = make_config(tmp_path, db_path=db_path, phase1=phase1, checkpoint_dir=checkpoint_dir)
    baseline = run_wappi_history_import(
        config,
        client=CheckpointFakeClient({"p-tg": chats, "p-max": []}, messages),
    )
    assert baseline["validation_ok"] is True
    before = read_checkpoint(checkpoint_dir)["profiles"]["wappi_telegram:p-tg"]
    safe_token = wappi_checkpoint_token("c0000")
    gap_token = wappi_checkpoint_token("c0001")
    gap_marker = before["chat_markers"][gap_token]
    gap_cursor = before["chat_cursors"][gap_token]
    messages[("telegram", "p-tg", "c0000")].append(
        {
            "id": "c0000-new", "chat_id": "c0000", "type": "text",
            "body": "Доказанное новое сообщение", "time": 1_753_000_001,
        }
    )
    chats[0] = {**dict(chats[0]), "last_timestamp": 1_753_000_001}
    chats[1] = {**dict(chats[1]), "last_timestamp": int(chats[1]["last_timestamp"]) + 1}

    client = CheckpointFakeClient({"p-tg": chats, "p-max": []}, messages)
    report = run_wappi_history_import(config, client=client)

    assert report["validation_ok"] is True
    assert report["mode"] == "apply"
    assert report["checkpoint"]["committed"] is True
    assert report["checkpoint"]["complete"] is True
    assert report["checkpoint"]["deferred_limit_hits"] == []
    assert wappi_row_count(db_path) == 3
    after = read_checkpoint(checkpoint_dir)["profiles"]["wappi_telegram:p-tg"]
    assert after["chat_markers"][safe_token] == 1_753_000_001
    assert after["chat_markers"][gap_token] == gap_marker + 1
    assert after["chat_cursors"][gap_token] == gap_cursor
    assert [
        request
        for request in client.message_request_calls
        if request[1] == "c0001"
    ] == [
        ("p-tg", "c0001", 0, 10, "desc"),
        ("p-tg", "c0001", 0, 10, "desc"),
    ]


def test_deleted_boundary_rebases_terminal_chat_without_losing_history(
    tmp_path: Path,
) -> None:
    db_path, phase1, checkpoint_dir = prepare(tmp_path)
    chats, messages = build_universe(1, messages_per_chat=2)
    config = make_config(
        tmp_path,
        db_path=db_path,
        phase1=phase1,
        checkpoint_dir=checkpoint_dir,
    )
    baseline = run_wappi_history_import(
        config,
        client=CheckpointFakeClient({"p-tg": chats, "p-max": []}, messages),
    )
    assert baseline["validation_ok"] is True
    assert wappi_row_count(db_path) == 2

    source_key = ("telegram", "p-tg", "c0000")
    messages[source_key] = [
        dict(messages[source_key][0]),
        {
            "id": "c0000-new",
            "chat_id": "c0000",
            "type": "text",
            "body": "Новое после удаления прежней границы",
            "time": 1_753_000_002,
        },
    ]
    chats[0] = {**dict(chats[0]), "last_timestamp": 1_753_000_002}

    class TerminalSnapshotClient(CheckpointFakeClient):
        def get_chat_messages(self, **kwargs: Any) -> Mapping[str, Any]:
            payload = super().get_chat_messages(**kwargs)
            return {**payload, "status": "done", "has_more": False}

    client = TerminalSnapshotClient({"p-tg": chats, "p-max": []}, messages)
    report = run_wappi_history_import(config, client=client)

    assert report["validation_ok"] is True
    assert report["publish_ready"] is True
    assert report["checkpoint"]["committed"] is True
    assert report["checkpoint"]["complete"] is True
    assert "p-tg" not in report["summary"]["empty_profiles"]
    assert report["profiles"]["p-tg"]["incremental_tail_fallbacks"] == 1
    assert wappi_row_count(db_path) == 3
    assert [
        request
        for request in client.message_request_calls
        if request[1] == "c0000"
    ] == [
        ("p-tg", "c0000", 0, 10, "desc"),
        ("p-tg", "c0000", 0, 10, "desc"),
    ]
    checkpoint = read_checkpoint(checkpoint_dir)["profiles"]["wappi_telegram:p-tg"]
    cursor = checkpoint["chat_cursors"][wappi_checkpoint_token("c0000")]
    assert cursor["message_digest"] == wappi_history_module.wappi_message_checkpoint_token(
        "p-tg", "c0000", "c0000-new"
    )


def test_moving_head_blocks_all_writes_and_checkpoint_progress(tmp_path: Path) -> None:
    db_path, phase1, checkpoint_dir = prepare(tmp_path)
    chats, messages = build_universe(2)
    config = make_config(tmp_path, db_path=db_path, phase1=phase1, checkpoint_dir=checkpoint_dir)
    baseline = run_wappi_history_import(
        config,
        client=CheckpointFakeClient({"p-tg": chats, "p-max": []}, messages),
    )
    assert baseline["validation_ok"] is True
    checkpoint_path = wappi_history_checkpoint_path(checkpoint_dir)
    checkpoint_bytes_before = checkpoint_path.read_bytes()
    checkpoint_before = read_checkpoint(checkpoint_dir)
    rows_before = wappi_row_count(db_path)

    messages[("telegram", "p-tg", "c0000")].append(
        {
            "id": "c0000-new", "chat_id": "c0000", "type": "text",
            "body": "Доказанное новое сообщение", "time": 1_753_000_001,
        }
    )
    chats[0] = {**dict(chats[0]), "last_timestamp": 1_753_000_001}
    chats[1] = {**dict(chats[1]), "last_timestamp": 1_753_000_001}
    moving_message = {
        "id": "c0001-new", "chat_id": "c0001", "type": "text",
        "body": "Появилось между проверками", "time": 1_753_000_001,
    }

    class MovingGapHeadClient(CheckpointFakeClient):
        gap_head_calls = 0

        def get_chat_messages(self, **kwargs: Any) -> Mapping[str, Any]:
            payload = super().get_chat_messages(**kwargs)
            if kwargs["chat_id"] == "c0001" and int(kwargs["offset"]) == 0:
                self.gap_head_calls += 1
                if self.gap_head_calls == 2:
                    return {"messages": [moving_message, *(payload.get("messages") or ())]}
            return payload

    client = MovingGapHeadClient({"p-tg": chats, "p-max": []}, messages)
    report = run_wappi_history_import(config, client=client)

    stats = report["profiles"]["p-tg"]
    assert report["validation_ok"] is False
    assert report["mode"] == "apply_blocked"
    assert report["writes"]["applied"] is False
    assert report["summary"]["write_applied"] is False
    assert report["checkpoint"]["committed"] is False
    assert report["checkpoint"]["deferred_limit_hits"] == []
    assert "p-tg:pagination_drift_detected" in report["limit_hits"]
    assert stats["message_page_drift_reason"] == "head_changed"
    assert stats["message_page_drift_marker_relation"] == "append"
    assert stats["message_page_drift_offset"] == 0
    assert stats["message_page_drift_pages"] == 1
    assert stats["message_page_drift_first_signature"]
    assert stats["message_page_drift_head_signature"]
    assert stats["message_page_drift_head_signature"] != stats["message_page_drift_first_signature"]
    assert any(request[1] == "c0000" for request in client.message_request_calls)
    assert wappi_row_count(db_path) == rows_before
    assert checkpoint_path.read_bytes() == checkpoint_bytes_before
    assert read_checkpoint(checkpoint_dir) == checkpoint_before
