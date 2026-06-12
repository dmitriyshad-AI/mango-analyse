from __future__ import annotations

import csv
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mango_mvp.existing_clients import amo_step2_scan as step2


NOW = datetime(2026, 6, 12, 12, 0, tzinfo=timezone.utc)


class FakeMcpClient:
    def __init__(self, leads: list[dict], contacts: dict[str, dict]) -> None:
        self.leads = leads
        self.contacts = contacts
        self.calls = 0

    def amo_api_get(self, *, path: str, params: dict | None = None, limit: int = 50) -> dict:
        self.calls += 1
        if path == "leads":
            return {"_embedded": {"leads": self.leads}}
        if path.startswith("contacts/"):
            contact_id = path.split("/", 1)[1]
            return self.contacts[contact_id]
        raise AssertionError(path)


def test_step2_builds_known_family_note_and_is_idempotent(tmp_path: Path) -> None:
    profiles_db = make_profiles_db(tmp_path / "customer_profiles.sqlite")
    project = tmp_path / "project"
    out = project / "product_data" / "customer_profiles" / "step2"
    lead = lead_payload(501, [101])
    contact = contact_payload(101, "+7 999 000 00 00")

    summary = step2.build_step2_scan(
        step2.NewLeadScanOptions(
            project_root=project,
            out_root=out,
            profiles_db=profiles_db,
            since=NOW,
            client=FakeMcpClient([lead], {"101": contact}),
            page_limit=10,
            sleep_sec=0.5,
            generated_at=NOW,
        )
    )

    assert summary["read_only"] is True
    assert summary["write_crm"] is False
    assert summary["counts"]["known_family_notes"] == 1
    notes = rows(out / "family_note_drafts.csv")
    assert notes[0]["review_class"] == "known_family"
    assert "Телефон известен" in notes[0]["note_text"]
    assert "Уточните, о ком разговор" in notes[0]["note_text"]
    assert "8 класс" in notes[0]["note_text"]

    second = step2.build_step2_scan(
        step2.NewLeadScanOptions(
            project_root=project,
            out_root=out,
            profiles_db=profiles_db,
            since=NOW,
            client=FakeMcpClient([lead], {"101": contact}),
            page_limit=10,
            sleep_sec=0.5,
            generated_at=NOW,
        )
    )
    assert second["counts"]["family_note_drafts"] == 0
    assert second["counts"]["note_drafts_skipped_existing"] == 1


def test_step2_common_phone_note_is_neutral(tmp_path: Path) -> None:
    profiles_db = make_profiles_db(tmp_path / "customer_profiles.sqlite", duplicate_phone=True)
    project = tmp_path / "project"
    out = project / "product_data" / "customer_profiles" / "step2"

    step2.build_step2_scan(
        step2.NewLeadScanOptions(
            project_root=project,
            out_root=out,
            profiles_db=profiles_db,
            since=NOW,
            client=FakeMcpClient([lead_payload(501, [101])], {"101": contact_payload(101, "+7 999 000 00 00")}),
            page_limit=10,
            sleep_sec=0.5,
            generated_at=NOW,
        )
    )

    notes = rows(out / "family_note_drafts.csv")
    assert notes[0]["review_class"] == "common_phone"
    assert "Общий телефон" in notes[0]["note_text"]
    assert "Родитель" not in notes[0]["note_text"]
    assert "Ученик" not in notes[0]["note_text"]


def test_step2_skips_leads_without_phone_or_profile(tmp_path: Path) -> None:
    profiles_db = make_profiles_db(tmp_path / "customer_profiles.sqlite")
    project = tmp_path / "project"
    out = project / "product_data" / "customer_profiles" / "step2"
    leads = [lead_payload(501, [101]), lead_payload(502, [102])]
    contacts = {"101": contact_payload(101, ""), "102": contact_payload(102, "+7 999 111 22 33")}

    summary = step2.build_step2_scan(
        step2.NewLeadScanOptions(
            project_root=project,
            out_root=out,
            profiles_db=profiles_db,
            since=NOW,
            client=FakeMcpClient(leads, contacts),
            page_limit=10,
            sleep_sec=0.5,
            generated_at=NOW,
        )
    )

    assert summary["counts"]["leads_without_phone"] == 1
    assert summary["counts"]["leads_without_profile"] == 1
    assert summary["counts"]["family_note_drafts"] == 0


def test_step2_no_profile_skip_does_not_block_future_match(tmp_path: Path) -> None:
    profiles_db = make_profiles_db(tmp_path / "customer_profiles.sqlite")
    project = tmp_path / "project"
    out = project / "product_data" / "customer_profiles" / "step2"
    lead = lead_payload(501, [101])
    contact = contact_payload(101, "+7 999 333 44 55")

    first = step2.build_step2_scan(
        step2.NewLeadScanOptions(
            project_root=project,
            out_root=out,
            profiles_db=profiles_db,
            since=NOW,
            client=FakeMcpClient([lead], {"101": contact}),
            page_limit=10,
            sleep_sec=0.5,
            generated_at=NOW,
        )
    )
    assert first["counts"]["leads_without_profile"] == 1
    assert first["counts"]["family_note_drafts"] == 0

    con = sqlite3.connect(profiles_db)
    try:
        insert_profile(con, "p-new", "+79993334455", "Новый Родитель", "Новый Ученик")
        con.commit()
    finally:
        con.close()

    second = step2.build_step2_scan(
        step2.NewLeadScanOptions(
            project_root=project,
            out_root=out,
            profiles_db=profiles_db,
            since=NOW,
            client=FakeMcpClient([lead], {"101": contact}),
            page_limit=10,
            sleep_sec=0.5,
            generated_at=NOW,
        )
    )
    assert second["counts"]["known_family_notes"] == 1
    assert second["counts"]["note_drafts_skipped_existing"] == 0


def test_callback_task_drafts_dedupe_bad_json_and_deadlines(tmp_path: Path) -> None:
    requests = tmp_path / "callbacks.jsonl"
    requests.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "chat_id": "chat-1",
                        "message_id": "m1",
                        "text": "Пожалуйста, перезвоните завтра утром",
                        "brand": "unpk",
                        "created_at": "2026-06-12T20:30:00+00:00",
                        "lead_id": "501",
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "chat_id": "chat-1",
                        "message_id": "m2",
                        "text": "Можно перезвонить после 15 часов?",
                        "brand": "unpk",
                        "created_at": "2026-06-12T20:40:00+00:00",
                        "lead_id": "501",
                    },
                    ensure_ascii=False,
                ),
                "{bad json",
                json.dumps({"chat_id": "chat-2", "text": "Спасибо, я сам перезвоню"}, ensure_ascii=False),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    project = tmp_path / "project"
    out = project / "product_data" / "customer_profiles" / "step2"

    summary = step2.build_step2_scan(
        step2.NewLeadScanOptions(
            project_root=project,
            out_root=out,
            profiles_db=make_profiles_db(tmp_path / "customer_profiles.sqlite"),
            since=NOW,
            client=FakeMcpClient([], {}),
            page_limit=10,
            sleep_sec=0.5,
            callback_requests_path=requests,
            generated_at=NOW,
        )
    )

    assert summary["counts"]["callback_requests_seen"] == 3
    assert summary["counts"]["callback_requests_malformed"] == 1
    assert summary["counts"]["callback_requests_without_intent"] == 1
    assert summary["counts"]["callback_requests_deduped"] == 1
    assert summary["counts"]["callback_task_drafts"] == 1
    tasks = rows(out / "callback_task_drafts.csv")
    assert tasks[0]["task_type"] == "call"
    assert tasks[0]["complete_till_iso"].startswith("2026-06-13T16:00:00")
    assert "+7" not in tasks[0]["text"]


def test_step2_live_write_flags_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="live AMO writes are disabled"):
        step2.build_step2_scan(
            step2.NewLeadScanOptions(
                project_root=tmp_path / "project",
                out_root=tmp_path / "project" / "product_data" / "customer_profiles" / "step2",
                profiles_db=make_profiles_db(tmp_path / "customer_profiles.sqlite"),
                since=NOW,
                client=FakeMcpClient([], {}),
                enable_amo_notes=True,
            )
        )


def test_step2_grade_text_does_not_duplicate_class_suffix() -> None:
    fields = [
        field_row("child_name", "Аня"),
        field_row("grade", "7 класс"),
        field_row("subject", "математика"),
    ]

    text = step2._children_note_text(fields)

    assert "7 класс" in text
    assert "класс кл" not in text
    assert "класс класс" not in text


def make_profiles_db(path: Path, *, duplicate_phone: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    try:
        con.executescript(
            """
            CREATE TABLE customer_profiles (
              profile_id TEXT PRIMARY KEY,
              tenant_id TEXT NOT NULL,
              primary_phone TEXT,
              display_name TEXT,
              built_at TEXT NOT NULL,
              build_id TEXT NOT NULL,
              source_event_count INTEGER NOT NULL,
              last_event_at TEXT
            );
            CREATE TABLE profile_fields (
              field_id TEXT PRIMARY KEY,
              profile_id TEXT NOT NULL,
              field TEXT NOT NULL,
              value TEXT NOT NULL,
              child_key TEXT NOT NULL DEFAULT '',
              brand TEXT NOT NULL DEFAULT 'unknown',
              source_system TEXT NOT NULL,
              source_ref TEXT NOT NULL,
              event_at TEXT NOT NULL,
              quote TEXT NOT NULL DEFAULT '',
              superseded_by TEXT NOT NULL DEFAULT ''
            );
            """
        )
        insert_profile(con, "p1", "+79990000000", "Родитель Один", "Ученик Один")
        if duplicate_phone:
            insert_profile(con, "p2", "+79990000000", "Родитель Два", "Ученик Два")
        con.commit()
    finally:
        con.close()
    return path


def insert_profile(con: sqlite3.Connection, profile_id: str, phone: str, parent: str, child: str) -> None:
    con.execute(
        "INSERT INTO customer_profiles VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (profile_id, "default", phone, parent, NOW.isoformat(), "build", 3, NOW.isoformat()),
    )
    for idx, (field, value, brand) in enumerate(
        [
            ("parent_name", parent, "unknown"),
            ("child_name", child, "unpk"),
            ("grade", "8", "unpk"),
            ("subject", "физика", "unpk"),
        ],
        start=1,
    ):
        con.execute(
            "INSERT INTO profile_fields VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (f"{profile_id}-{idx}", profile_id, field, value, "child_1", brand, "test", "fixture", NOW.isoformat(), "", ""),
        )


def field_row(field: str, value: str, *, child_key: str = "child_1", brand: str = "unpk") -> dict[str, str]:
    return {
        "field": field,
        "value": value,
        "child_key": child_key,
        "brand": brand,
        "event_at": NOW.isoformat(),
    }


def lead_payload(lead_id: int, contacts: list[int]) -> dict:
    return {"id": lead_id, "created_at": int(NOW.timestamp()), "_embedded": {"contacts": [{"id": item} for item in contacts]}}


def contact_payload(contact_id: int, phone: str) -> dict:
    values = [{"value": phone}] if phone else []
    return {
        "id": contact_id,
        "custom_fields_values": [
            {"field_name": "Телефон", "field_code": "PHONE", "values": values},
        ],
    }


def rows(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open(encoding="utf-8-sig")))
