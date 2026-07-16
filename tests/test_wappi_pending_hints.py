from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from mango_mvp.customer_timeline.wappi_history_import import (
    WappiChatResolution,
    WappiProfileSpec,
)
from mango_mvp.customer_timeline.wappi_pending_hints import (
    PendingWappiChat,
    _hint_row,
    build_pending_hints,
    load_pending_wappi_chats,
    validate_human_decisions,
    write_hint_pack,
)
from mango_mvp.integrations.amo_wappi_transport import DefaultDenyTransport, SafeTransportPolicy


def test_load_pending_wappi_chats_dedups_open_wappi_conflicts_and_counts_messages(tmp_path: Path) -> None:
    db_path = tmp_path / "timeline.sqlite"
    seed_conflicts(
        db_path,
        [
            conflict("c-1", "open", "wappi_telegram", "p-tg", "chat-1", reason="pair_missing"),
            conflict("c-2", "open", "wappi_telegram", "p-tg", "chat-1", reason="pair_missing"),
            conflict("c-3", "open", "wappi_telegram", "p-tg", "chat-1", reason="amo_no_contact"),
            conflict("c-4", "open", "wappi_max", "p-max", "chat-2", brand="mango", reason="pair_missing"),
            conflict("c-5", "resolved", "wappi_telegram", "p-tg", "closed-chat"),
            conflict("c-6", "open", "telegram_native", "p-tg", "native-chat"),
            conflict("c-7", "open", "wappi_telegram", "", "missing-profile"),
            conflict("c-8", "open", "wappi_telegram", "p-other", "other-tenant", tenant_id="other"),
        ],
    )

    rows = load_pending_wappi_chats(db_path, tenant_id="foton")

    assert [(item.profile_id, item.chat_id) for item in rows] == [("p-max", "chat-2"), ("p-tg", "chat-1")]
    by_key = {(item.profile_id, item.chat_id): item for item in rows}
    assert by_key[("p-tg", "chat-1")].pending_message_count == 3
    assert by_key[("p-tg", "chat-1")].previous_reason == "pair_missing"
    assert by_key[("p-tg", "chat-1")].channel == "telegram"
    assert by_key[("p-max", "chat-2")].pending_message_count == 1
    assert by_key[("p-max", "chat-2")].channel == "max"


def test_load_pending_wappi_chats_rejects_inconsistent_brand_for_same_chat(tmp_path: Path) -> None:
    db_path = tmp_path / "timeline.sqlite"
    seed_conflicts(
        db_path,
        [
            conflict("c-1", "open", "wappi_telegram", "p-tg", "chat-1", brand="foton"),
            conflict("c-2", "open", "wappi_telegram", "p-tg", "chat-1", brand="unpk"),
        ],
    )

    with pytest.raises(ValueError, match="inconsistent brand/source"):
        load_pending_wappi_chats(db_path, tenant_id="foton")


def test_write_hint_pack_is_restricted_to_codex_local_and_writes_review_files(tmp_path: Path) -> None:
    row = sample_hint("p-tg", "chat-1", WappiChatResolution(status="pending_attribution", reason="missing"))

    with pytest.raises(ValueError, match=r"\.codex_local"):
        write_hint_pack(tmp_path / "exports", [row], {"writes": 0})

    paths = write_hint_pack(tmp_path / ".codex_local" / "wappi_hints", [row], {"writes": 0})

    jsonl_rows = read_jsonl(Path(paths["jsonl"]))
    review_rows = list(csv.DictReader(Path(paths["review_csv"]).open("r", encoding="utf-8-sig", newline="")))
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    assert jsonl_rows == [row]
    assert review_rows[0]["decision"] == "pending"
    assert review_rows[0]["hint_id"] == row["hint_id"]
    assert summary["policy"]["read_only"] is True
    assert summary["policy"]["automatic_binding"] is False
    assert summary["policy"]["chunk_visibility_changed"] is False
    assert summary["review_sample"] == {
        "rows": 1,
        "proposed": 0,
        "sha256": summary["review_sample"]["sha256"],
    }


def test_write_hint_pack_freezes_deterministic_proposal_first_review_sample(tmp_path: Path) -> None:
    rows = [
        sample_hint("p-tg", "chat-no-proposal", WappiChatResolution(status="pending_attribution", reason="missing")),
        sample_hint("p-tg", "chat-proposal-1", WappiChatResolution(status="resolved", customer_id="customer-1")),
        sample_hint("p-tg", "chat-proposal-2", WappiChatResolution(status="resolved", customer_id="customer-2")),
    ]

    first = write_hint_pack(tmp_path / ".codex_local" / "first", rows, {"writes": 0}, review_limit=30)
    second = write_hint_pack(tmp_path / ".codex_local" / "second", list(reversed(rows)), {"writes": 0}, review_limit=30)

    first_rows = read_jsonl(Path(first["review_sample_jsonl"]))
    second_rows = read_jsonl(Path(second["review_sample_jsonl"]))
    assert first_rows == second_rows
    assert len(first_rows) == 3
    assert [row["proposal_status"] for row in first_rows[:2]] == ["proposed", "proposed"]


def test_write_hint_pack_rejects_review_sample_outside_30_to_50(tmp_path: Path) -> None:
    row = sample_hint("p-tg", "chat-1", WappiChatResolution(status="pending_attribution", reason="missing"))

    with pytest.raises(ValueError, match="30-50"):
        write_hint_pack(tmp_path / ".codex_local" / "invalid", [row], {"writes": 0}, review_limit=2)


def test_existing_pair_hint_does_not_claim_current_amo_evidence() -> None:
    row = sample_hint(
        "p-tg",
        "chat-1",
        WappiChatResolution(
            status="resolved",
            customer_id="customer-1",
            pair_source="manual",
            resolution_source="draft_loop_pair",
        ),
    )

    assert row["rationale"] == "existing_pair:manual:current_amo_not_rechecked"
    assert row["rationale_ru"] == "В текущем файле уже есть явная пара; её актуальность в AMO этим прогоном ещё не подтверждена."
    assert row["single_active_lead"] is False
    assert row["evidence_complete"] is False
    assert row["review_gate"] == "verify_existing_pair"


def test_assert_readonly_wappi_client_rejects_transport_with_write_capability() -> None:
    client = FakeReadonlyWappiClient(chats={}, messages={})
    client.transport = DefaultDenyTransport(lambda **_kwargs: {"ok": True})

    with pytest.raises(RuntimeError, match="Wappi-only"):
        from mango_mvp.customer_timeline.wappi_history_import import assert_readonly_wappi_client

        assert_readonly_wappi_client(client)


def test_hint_fingerprint_is_frozen_for_same_proposal_basis() -> None:
    row = sample_hint(
        "p-tg",
        "chat-1",
        WappiChatResolution(
            status="resolved",
            customer_id="customer-1",
            lead_id="1001",
            contact_id="2002",
            expected_brand="foton",
            resolution_source="amo_auto_resolver",
            match_key="telegram_id",
            evidence={
                "exact_match_kind": "telegram_id",
                "single_active_lead": True,
                "organization_brand": "foton",
                "organization_value_count": 1,
                "timeline_identity_sources": ("amo_contact_id",),
            },
        ),
    )

    assert row["hint_id"] == "451b5d28075b971a1652"
    assert row["proposal_fingerprint"] == "e59b52d092339d62a264ee587a00801fe27987e0451a3b5784ae795e71c0df06"
    assert row["proposal_status"] == "proposed"
    assert row["rationale"] == "amo_auto_resolver:telegram_id:one_active_lead:org_foton:timeline_amo_contact_id"
    assert row["evidence_complete"] is True
    assert row["rationale_ru"] == (
        "Точное совпадение по Telegram ID; в AMO один активный лид; "
        "организация Фотон; клиент подтверждён по контакту AMO."
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda rows: rows[:1], "frozen hint set"),
        (lambda rows: [rows[0], rows[0]], "unique non-empty hint_id"),
        (lambda rows: [{**rows[0], "proposal_fingerprint": "changed"}, rows[1]], "proposal fingerprint changed"),
        (lambda rows: [{**rows[0], "decision": "approve"}, rows[1]], "cannot approve a row without a proposal"),
    ],
)
def test_validate_human_decisions_rejects_missing_duplicate_changed_fingerprint_and_no_proposal_approve(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    hints = [
        sample_hint("p-tg", "chat-1", WappiChatResolution(status="pending_attribution", reason="missing")),
        sample_hint("p-tg", "chat-2", WappiChatResolution(status="resolved", customer_id="customer-2")),
    ]
    hints_jsonl = write_jsonl(tmp_path / "hints.jsonl", hints)
    decisions_csv = write_decisions_csv(tmp_path / "decisions.csv", mutate([{**row, "decision": "reject"} for row in hints]))

    with pytest.raises(ValueError, match=message):
        validate_human_decisions(hints_jsonl, decisions_csv)


def test_validate_human_decisions_accepts_approve_reject_and_reports_precision(tmp_path: Path) -> None:
    proposed = sample_hint("p-tg", "chat-1", WappiChatResolution(status="resolved", customer_id="customer-1"))
    no_proposal = sample_hint("p-tg", "chat-2", WappiChatResolution(status="pending_attribution", reason="missing"))
    hints_jsonl = write_jsonl(tmp_path / "hints.jsonl", [proposed, no_proposal])
    decisions_csv = write_decisions_csv(
        tmp_path / "decisions.csv",
        [{**proposed, "decision": "approve"}, {**no_proposal, "decision": "reject"}],
    )

    report = validate_human_decisions(hints_jsonl, decisions_csv)

    assert report["counts"] == {"approve": 1, "reject": 1}
    assert report["review_complete"] is True
    assert report["proposed_reviewed"] == 1
    assert report["approved_proposals"] == 1
    assert report["precision"] == 1.0
    assert report["binding_executed"] is False
    assert report["chunk_visibility_changed"] is False


def test_non_proposal_rows_may_remain_pending_after_all_proposals_reviewed(tmp_path: Path) -> None:
    proposed = sample_hint("p-tg", "chat-1", WappiChatResolution(status="resolved", customer_id="customer-1"))
    no_proposal = sample_hint("p-tg", "chat-2", WappiChatResolution(status="pending_attribution", reason="missing"))
    hints_jsonl = write_jsonl(tmp_path / "hints.jsonl", [proposed, no_proposal])
    decisions_csv = write_decisions_csv(
        tmp_path / "decisions.csv",
        [{**proposed, "decision": "approve"}, {**no_proposal, "decision": "pending"}],
    )

    report = validate_human_decisions(hints_jsonl, decisions_csv)

    assert report["review_complete"] is True
    assert report["proposed_pending"] == 0
    assert report["counts"]["pending"] == 1


def test_build_pending_hints_is_readonly_and_returns_proposed_and_no_proposal_rows() -> None:
    client = FakeReadonlyWappiClient(
        chats={
            "p-tg": [{"id": "chat-1"}, {"id": "chat-2"}],
        },
        messages={
            ("telegram", "p-tg", "chat-1"): [raw_message("m-1", "chat-1", "Добрый день")],
            ("telegram", "p-tg", "chat-2"): [raw_message("m-2", "chat-2", "Нужна цена")],
        },
    )
    resolver = FakeResolver(
        {
            "chat-1": WappiChatResolution(
                status="resolved",
                customer_id="customer-1",
                lead_id="1001",
                contact_id="2002",
                expected_brand="foton",
                reason="",
                resolution_source="amo_auto_resolver",
                match_key="telegram_id",
                evidence={
                    "exact_match_kind": "telegram_id",
                    "single_active_lead": True,
                    "organization_brand": "foton",
                    "organization_value_count": 1,
                    "timeline_identity_sources": ("amo_contact_id",),
                },
            ),
            "chat-2": WappiChatResolution(
                status="pending_attribution",
                expected_brand="foton",
                reason="amo_no_exact_contact",
                resolution_source="amo_auto_resolver",
            ),
        }
    )
    pending = [
        PendingWappiChat("p-tg", "chat-1", "wappi_telegram", "foton", 2, "pair_missing"),
        PendingWappiChat("p-tg", "chat-2", "wappi_telegram", "foton", 1, "amo_no_contact"),
    ]

    rows, summary = build_pending_hints(
        client=client,
        profiles=[WappiProfileSpec("p-tg", "foton", "telegram")],
        resolver=resolver,
        pending_chats=pending,
        page_size=10,
        list_request_limit=10,
        messages_per_chat=2,
        sleep_seconds=0,
        amo_pause_seconds_per_call=0,
    )

    assert [row["proposal_status"] for row in rows] == ["proposed", "no_proposal"]
    assert rows[0]["proposed_customer_id"] == "customer-1"
    assert rows[0]["rationale"] == "amo_auto_resolver:telegram_id:one_active_lead:org_foton:timeline_amo_contact_id"
    assert rows[1]["proposed_customer_id"] == ""
    assert rows[1]["rationale"] == "no_proposal:amo_no_exact_contact"
    assert summary["writes"] == 0
    assert summary["human_decisions_applied"] == 0
    assert summary["proposal_status_counts"] == {"proposed": 1, "no_proposal": 1}
    assert client.calls == [
        {"kind": "list", "channel": "telegram", "profile_id": "p-tg", "limit": 10, "offset": 0, "show_all": False},
        {"kind": "messages", "channel": "telegram", "profile_id": "p-tg", "chat_id": "chat-1", "limit": 2, "offset": 0, "mark_all": False},
        {"kind": "messages", "channel": "telegram", "profile_id": "p-tg", "chat_id": "chat-2", "limit": 2, "offset": 0, "mark_all": False},
    ]
    assert resolver.calls == [
        {"profile_id": "p-tg", "chat_id": "chat-1", "message_count": 1},
        {"profile_id": "p-tg", "chat_id": "chat-2", "message_count": 1},
    ]


def test_build_pending_hints_does_not_call_resolver_on_profile_brand_mismatch() -> None:
    client = FakeReadonlyWappiClient(
        chats={"p-tg": [{"id": "chat-1"}]},
        messages={("telegram", "p-tg", "chat-1"): [raw_message("m-1", "chat-1", "Добрый день")]},
    )
    resolver = FakeResolver({})

    rows, summary = build_pending_hints(
        client=client,
        profiles=[WappiProfileSpec("p-tg", "foton", "telegram")],
        resolver=resolver,
        pending_chats=[PendingWappiChat("p-tg", "chat-1", "wappi_telegram", "unpk", 1, "pair_missing")],
        sleep_seconds=0,
        amo_pause_seconds_per_call=0,
    )

    assert rows[0]["proposal_status"] == "no_proposal"
    assert rows[0]["rationale"] == "no_proposal:pending_profile_brand_or_channel_mismatch"
    assert resolver.calls == []
    assert summary["writes"] == 0


def test_risky_previous_resolution_requires_current_amo_recheck() -> None:
    row = _hint_row(
        PendingWappiChat("p-tg", "chat-1", "wappi_telegram", "foton", 1, "multi_active_lead"),
        WappiChatResolution(status="resolved", customer_id="customer-1"),
    )

    assert row["review_gate"] == "recheck_current_amo_lead_state"


class FakeReadonlyWappiClient:
    def __init__(
        self,
        *,
        chats: Mapping[str, Sequence[Mapping[str, Any]]],
        messages: Mapping[tuple[str, str, str], Sequence[Mapping[str, Any]]],
    ) -> None:
        self.transport = DefaultDenyTransport(
            lambda **_kwargs: {"ok": True},
            policy=SafeTransportPolicy.wappi_read_only(),
        )
        self.chats = {key: list(value) for key, value in chats.items()}
        self.messages = {key: list(value) for key, value in messages.items()}
        self.calls: list[dict[str, Any]] = []

    def list_chats(
        self,
        *,
        channel: str,
        profile_id: str,
        limit: int = 50,
        offset: int = 0,
        order: str = "desc",
        show_all: bool = False,
    ) -> Mapping[str, Any]:
        self.calls.append(
            {
                "kind": "list",
                "channel": channel,
                "profile_id": profile_id,
                "limit": limit,
                "offset": offset,
                "show_all": show_all,
            }
        )
        items = self.chats.get(profile_id, [])
        return {"dialogs": items[offset : offset + limit]}

    def get_chat_messages(
        self,
        *,
        channel: str,
        profile_id: str,
        chat_id: str,
        limit: int = 50,
        offset: int = 0,
        order: str = "desc",
        mark_all: bool = False,
    ) -> Mapping[str, Any]:
        self.calls.append(
            {
                "kind": "messages",
                "channel": channel,
                "profile_id": profile_id,
                "chat_id": chat_id,
                "limit": limit,
                "offset": offset,
                "mark_all": mark_all,
            }
        )
        items = self.messages.get((channel, profile_id, chat_id), [])
        return {"messages": items[offset : offset + limit]}


class FakeResolver:
    def __init__(self, resolutions: Mapping[str, WappiChatResolution]) -> None:
        self.resolutions = dict(resolutions)
        self.amo_auto_calls = 0
        self.calls: list[dict[str, Any]] = []

    def resolve_chat(
        self,
        *,
        profile: WappiProfileSpec,
        dialog: Mapping[str, Any],
        messages: Sequence[Any],
    ) -> WappiChatResolution:
        chat_id = str(dialog["id"])
        self.amo_auto_calls += 1
        self.calls.append({"profile_id": profile.profile_id, "chat_id": chat_id, "message_count": len(messages)})
        return self.resolutions[chat_id]


def sample_hint(profile_id: str, chat_id: str, resolution: WappiChatResolution) -> dict[str, Any]:
    return _hint_row(PendingWappiChat(profile_id, chat_id, "wappi_telegram", "foton", 1, "pair_missing"), resolution)


def seed_conflicts(db_path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with sqlite3.connect(db_path) as con:
        con.execute(
            """
            CREATE TABLE timeline_conflicts (
              conflict_id TEXT PRIMARY KEY,
              tenant_id TEXT NOT NULL,
              conflict_type TEXT NOT NULL,
              severity TEXT NOT NULL,
              status TEXT NOT NULL,
              created_at TEXT NOT NULL,
              resolved_at TEXT,
              record_hash TEXT NOT NULL,
              record_json TEXT NOT NULL
            )
            """
        )
        con.executemany(
            """
            INSERT INTO timeline_conflicts (
              conflict_id, tenant_id, conflict_type, severity, status, created_at, resolved_at, record_hash, record_json
            )
            VALUES (:conflict_id, :tenant_id, 'pending_attribution', 'medium', :status, '2026-07-12T00:00:00+00:00', NULL, :conflict_id, :record_json)
            """,
            rows,
        )


def conflict(
    conflict_id: str,
    status: str,
    source_system: str,
    profile_id: str,
    chat_id: str,
    *,
    brand: str = "foton",
    reason: str = "unknown",
    tenant_id: str = "foton",
) -> dict[str, Any]:
    return {
        "conflict_id": conflict_id,
        "tenant_id": tenant_id,
        "status": status,
        "record_json": json.dumps(
            {
                "metadata": {
                    "source_system": source_system,
                    "profile_id": profile_id,
                    "chat_id": chat_id,
                    "brand": brand,
                    "resolution_reason": reason,
                }
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
    }


def raw_message(message_id: str, chat_id: str, body: str) -> dict[str, Any]:
    return {"id": message_id, "chat_id": chat_id, "type": "text", "body": body, "time": 1_753_000_000}


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_decisions_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "hint_id",
                "proposal_fingerprint",
                "decision",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
    return path
