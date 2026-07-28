"""Wappi large-history fetch checkpoint: resume from the last CONFIRMED chat.

Every test drives the production entry point ``run_wappi_history_import`` with a
fake read-only Wappi client. No live Wappi call, no real message body.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Mapping, Optional

import pytest

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
        if self.fail_message_at is not None and (chat_id, offset) == self.fail_message_at:
            raise AmoWappiHttpError("HTTP 503: service unavailable")
        items = self.messages.get((channel, profile_id, chat_id), [])
        return {"messages": items[offset : offset + limit]}


def build_universe(
    total_chats: int, *, messages_per_chat: int = 1, profile_id: str = "p-tg",
    channel: str = "telegram", body: str = "Здравствуйте",
) -> tuple[list[Mapping[str, Any]], dict[tuple[str, str, str], list[Mapping[str, Any]]]]:
    chats = [{"id": f"c{index:04d}", "type": "user"} for index in range(total_chats)]
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


def wappi_row_count(db_path: Path) -> int:
    with sqlite3.connect(db_path) as con:
        return int(
            con.execute(
                "SELECT COUNT(*) FROM timeline_events WHERE source_system LIKE 'wappi_%'"
            ).fetchone()[0]
        )


def read_checkpoint(checkpoint_dir: Path) -> Mapping[str, Any]:
    return load_wappi_history_checkpoint(checkpoint_dir)


def prepare(tmp_path: Path) -> tuple[Path, Path, Path]:
    db_path = tmp_path / "customer_timeline.sqlite"
    CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path).close()
    checkpoint_dir = tmp_path / "checkpoints"
    return db_path, write_phase1_config(tmp_path), checkpoint_dir


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


def test_catalog_failure_on_page_13_resumes_from_that_page(tmp_path: Path) -> None:
    """Catalogue dies on page 13 of 13: run 1 confirms 120 chats, run 2 finishes."""
    db_path, phase1, checkpoint_dir = prepare(tmp_path)
    chats, messages = build_universe(130)
    first = CheckpointFakeClient({"p-tg": chats, "p-max": []}, messages)
    first.fail_catalog_from_offset = 120

    report_one = run_wappi_history_import(
        make_config(tmp_path, db_path=db_path, phase1=phase1, checkpoint_dir=checkpoint_dir),
        client=first,
    )

    assert report_one["mode"] == "apply"
    assert report_one["validation_ok"] is False
    assert report_one["checkpoint"]["complete"] is False
    assert report_one["checkpoint"]["committed"] is True
    tg_state = report_one["checkpoint"]["profiles"]["wappi_telegram:p-tg"]
    assert tg_state["stop_reason"] == "network_error"
    assert tg_state["chats_done"] == 120
    assert wappi_row_count(db_path) == 120

    second = CheckpointFakeClient({"p-tg": chats, "p-max": []}, messages)
    report_two = run_wappi_history_import(
        make_config(tmp_path, db_path=db_path, phase1=phase1, checkpoint_dir=checkpoint_dir),
        client=second,
    )

    refetched = {chat_id for _profile, chat_id, _offset in second.message_calls}
    assert {chat["id"] for chat in chats[120:]}.issubset(refetched)
    assert wappi_row_count(db_path) == 130
    assert report_two["checkpoint"]["complete"] is True
    assert report_two["validation_ok"] is True
    assert not wappi_history_checkpoint_path(checkpoint_dir).exists()


def test_message_page_failure_resumes_inside_long_chat(tmp_path: Path) -> None:
    """A long chat dies on its second message page and resumes from that offset."""
    db_path, phase1, checkpoint_dir = prepare(tmp_path)
    chats, messages = build_universe(1, messages_per_chat=30)
    first = CheckpointFakeClient({"p-tg": chats, "p-max": []}, messages)
    # Catalogue page + first message page fit; the second message page does not.
    report_one = run_wappi_history_import(
        make_config(tmp_path, db_path=db_path, phase1=phase1, checkpoint_dir=checkpoint_dir, request_limit_total=5),
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
    assert not wappi_history_checkpoint_path(checkpoint_dir).exists()


def test_request_limit_pause_writes_and_next_run_continues(tmp_path: Path) -> None:
    db_path, phase1, checkpoint_dir = prepare(tmp_path)
    chats, messages = build_universe(20)
    first = CheckpointFakeClient({"p-tg": chats, "p-max": []}, messages)

    report_one = run_wappi_history_import(
        make_config(tmp_path, db_path=db_path, phase1=phase1, checkpoint_dir=checkpoint_dir, request_limit_total=12),
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
    assert report_two["summary"]["duplicate_source_ids_before_import"] > 0


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
    assert already_done and refetched & already_done  # final tail check prevents missing new messages
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
        make_config(tmp_path, db_path=db_path, phase1=phase1, checkpoint_dir=checkpoint_dir, request_limit_total=5),
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
    report_two = run_wappi_history_import(
        config, client=CheckpointFakeClient({"p-tg": chats, "p-max": []}, messages)
    )

    assert first_rows == 12
    assert wappi_row_count(db_path) == 12
    assert report_two["summary"]["duplicate_source_ids_before_import"] == 12


def test_telegram_and_max_profiles_do_not_share_checkpoint(tmp_path: Path) -> None:
    db_path, phase1, checkpoint_dir = prepare(tmp_path)
    tg_chats, tg_messages = build_universe(20, profile_id="p-tg", channel="telegram")
    max_chats, max_messages = build_universe(4, profile_id="p-max", channel="max")
    max_chats = [{"id": str(chat["id"]), "type": "DIALOG"} for chat in max_chats]
    client = CheckpointFakeClient({"p-tg": tg_chats, "p-max": max_chats}, {**tg_messages, **max_messages})

    run_wappi_history_import(
        make_config(tmp_path, db_path=db_path, phase1=phase1, checkpoint_dir=checkpoint_dir, request_limit_total=12),
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
                checkpoint_dir=checkpoint_dir, request_limit_total=14,
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


def test_terminal_complete_clears_partial_state(tmp_path: Path) -> None:
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
    assert not wappi_history_checkpoint_path(checkpoint_dir).exists()


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
        make_config(tmp_path, db_path=db_path, phase1=phase1, checkpoint_dir=checkpoint_dir, request_limit_total=5),
        client=CheckpointFakeClient({"p-tg": chats, "p-max": []}, messages),
    )
    assert wappi_row_count(db_path) == 10

    # Budget for run 2: p-max 2 + catalogue 1 + verification 1 + exactly one message
    # request, which the anchor probe consumes.
    starved = CheckpointFakeClient({"p-tg": chats, "p-max": []}, messages)
    report = run_wappi_history_import(
        make_config(tmp_path, db_path=db_path, phase1=phase1, checkpoint_dir=checkpoint_dir, request_limit_total=5),
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
    assert not wappi_history_checkpoint_path(checkpoint_dir).exists()


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
    assert not wappi_history_checkpoint_path(checkpoint_dir).exists()


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
    assert wappi_history_checkpoint_path(checkpoint_dir).exists()
