from __future__ import annotations

import csv
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from mango_mvp.existing_clients import amo_step3_contact_cards as step3


NOW = datetime(2026, 6, 12, 15, 0, tzinfo=timezone.utc)


class FakeFieldClient:
    calls = 0

    def amo_api_get(self, *, path: str, params: dict | None = None, limit: int = 50) -> dict:
        self.calls += 1
        assert path == "contacts/custom_fields"
        return {
            "_embedded": {
                "custom_fields": [
                    {"id": 1, "name": "Телефон", "type": "multitext"},
                    {"id": 99, "name": "ИИ: профиль клиента", "type": "textarea", "is_api_only": False},
                ]
            }
        }


def test_step3_builds_stage_a_dry_run_cards(tmp_path: Path) -> None:
    profiles_db = make_profiles_db(tmp_path / "profiles.sqlite")
    amo_db = make_amo_snapshot_db(tmp_path / "amo.sqlite")
    project = tmp_path / "project"
    out = project / "product_data" / "customer_profiles" / "step3"

    summary = step3.build_contact_card_stage_a(
        step3.ContactCardOptions(
            project_root=project,
            out_root=out,
            profiles_db=profiles_db,
            amo_snapshot_db=amo_db,
            client=FakeFieldClient(),
            stage_a_families=2,
            generated_at=NOW,
        )
    )

    assert summary["read_only"] is True
    assert summary["write_crm"] is False
    assert summary["field_check"]["status"] == "ok"
    assert summary["stage_a_families_selected"] == 1
    rows = csv_rows(out / "contact_card_dry_run.csv")
    assert rows[0]["field_name"] == "ИИ: профиль клиента"
    assert rows[0]["status"] == "dry_run"
    assert "Ученик:" in rows[0]["card_text"]
    assert "Семья:" in rows[0]["card_text"]
    assert "Договоренность семьи:" in rows[0]["card_text"]
    assert "Возражения:" in rows[0]["card_text"]
    assert "Tallanto-статус:" in rows[0]["card_text"]
    assert "+7" not in rows[0]["card_text"]


def test_step3_skips_phone_with_multiple_profiles(tmp_path: Path) -> None:
    profiles_db = make_profiles_db(tmp_path / "profiles.sqlite", duplicate_phone=True)
    amo_db = make_amo_snapshot_db(tmp_path / "amo.sqlite")

    summary = step3.build_contact_card_stage_a(
        step3.ContactCardOptions(
            project_root=tmp_path / "project",
            out_root=tmp_path / "project" / "product_data" / "customer_profiles" / "step3",
            profiles_db=profiles_db,
            amo_snapshot_db=amo_db,
            stage_a_families=5,
            generated_at=NOW,
        )
    )

    assert summary["stage_a_families_selected"] == 0
    assert summary["skip_counts"]["phone_2plus_profiles"] == 2


def test_contact_card_quality_blocks_raw_phone_and_mixed_brands() -> None:
    findings = step3.contact_card_findings("Ученик: Иван [Фотон] [УНПК]\nСемья: +79990000000")

    assert "raw_phone_in_card" in findings
    assert "mixed_brand_markers" in findings


def test_contact_card_field_wrong_type_is_reported() -> None:
    class WrongTypeClient:
        def amo_api_get(self, *, path: str, params: dict | None = None, limit: int = 50) -> dict:
            return {"_embedded": {"custom_fields": [{"id": 99, "name": "ИИ: профиль клиента", "type": "numeric"}]}}

    assert step3.check_contact_card_field(WrongTypeClient())["status"] == "wrong_type"


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
        insert_profile(con, "p1", "+79990000000", "Родитель", "Ученик", brand="unpk")
        if duplicate_phone:
            insert_profile(con, "p2", "+79990000000", "Другой", "Другой ученик", brand="unpk")
        con.commit()
    finally:
        con.close()
    return path


def insert_profile(con: sqlite3.Connection, profile_id: str, phone: str, parent: str, child: str, *, brand: str) -> None:
    con.execute(
        "INSERT INTO customer_profiles VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (profile_id, "default", phone, parent, NOW.isoformat(), "build", 4, NOW.isoformat()),
    )
    fields = [
        ("parent_name", parent, "child_1", "unknown"),
        ("child_name", child, "child_1", brand),
        ("grade", "8", "child_1", brand),
        ("subject", "физика", "child_1", brand),
        ("next_step", "уточнить расписание", "", brand),
        ("objection", "сомнение по времени", "", brand),
    ]
    for idx, (field, value, child_key, field_brand) in enumerate(fields, start=1):
        con.execute(
            "INSERT INTO profile_fields VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (f"{profile_id}-{idx}", profile_id, field, value, child_key, field_brand, "test", "fixture", NOW.isoformat(), "", ""),
        )


def make_amo_snapshot_db(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    try:
        con.execute(
            """
            CREATE TABLE contacts (
              contact_id TEXT PRIMARY KEY,
              name TEXT,
              name_key TEXT,
              parent_name TEXT,
              parent_key TEXT,
              phones_json TEXT,
              tallanto_ids_json TEXT,
              lead_ids_json TEXT,
              active_lead_ids_json TEXT,
              has_active_lead INTEGER,
              has_tallanto_link INTEGER,
              created_at TEXT,
              updated_at TEXT
            )
            """
        )
        con.execute(
            "INSERT INTO contacts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "101",
                "Ученик",
                "ученик",
                "Родитель",
                "родитель",
                json.dumps(["+79990000000"]),
                json.dumps(["t1"]),
                json.dumps(["501"]),
                json.dumps(["501"]),
                1,
                1,
                NOW.isoformat(),
                NOW.isoformat(),
            ),
        )
        con.commit()
    finally:
        con.close()
    return path


def csv_rows(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open(encoding="utf-8-sig")))
