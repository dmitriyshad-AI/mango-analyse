from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from mango_mvp.customer_profile.contracts import ProfileFieldCandidate, ProfileSnapshot
from mango_mvp.customer_profile.crm_summary import CRM_SUMMARY_MAX_CHARS, main, render_crm_summary_from_db
from mango_mvp.customer_profile.store import CustomerProfileSQLiteStore


NOW = datetime(2026, 6, 10, 12, 0, tzinfo=timezone.utc)


def _profiles_db(tmp_path: Path, *, fields: list[ProfileFieldCandidate]) -> Path:
    db = tmp_path / "customer_profiles.sqlite"
    with CustomerProfileSQLiteStore(db) as store:
        store.replace_profiles(
            build_id="test-build",
            built_at=NOW,
            timeline_db_path=tmp_path / "customer_timeline.sqlite",
            timeline_db_sha256="0" * 64,
            profiles=[
                ProfileSnapshot(
                    profile_id="cust-1",
                    tenant_id="foton",
                    primary_phone="+79991234567",
                    display_name="Мария",
                    source_event_count=5,
                    last_event_at=NOW,
                )
            ],
            fields=fields,
        )
    return db


def _field(
    field: str,
    value: str,
    *,
    field_id: str | None = None,
    child_key: str = "",
    brand: str = "foton",
    event_at: datetime = NOW,
    superseded_by: str = "",
) -> ProfileFieldCandidate:
    return ProfileFieldCandidate(
        profile_id="cust-1",
        field=field,
        value=value,
        child_key=child_key,
        brand=brand,
        source_system="test",
        source_ref=f"test:{field}:{field_id or value}",
        event_at=event_at,
        field_id=field_id,
        superseded_by=superseded_by,
    )


def test_crm_summary_masks_phone_and_renders_required_blocks(tmp_path: Path) -> None:
    db = _profiles_db(
        tmp_path,
        fields=[
            _field("parent_name", "Мария Иванова"),
            _field("child_name", "Аня", child_key="child_1"),
            _field("grade", "8", child_key="child_1"),
            _field("subject", "математика; физика", child_key="child_1"),
            _field("tallanto_group", "Фотон-8"),
            _field("tallanto_balance", "1500"),
            _field("payment_fact", "2026-06-01: 12000"),
            _field("next_step", "Созвониться после оплаты"),
            _field(
                "brand_touch",
                '{"brand":"foton","channel":"telegram","count":3,"first_at":"2026-05-01","last_at":"2026-06-01"}',
            ),
        ],
    )

    text = render_crm_summary_from_db(db, phone="+7 999 123-45-67")

    assert len(text) <= CRM_SUMMARY_MAX_CHARS
    assert "+***4567" in text
    assert "+79991234567" not in text
    assert "Мария Иванова" in text
    assert "Аня - 8 класс - математика; физика" in text
    assert "Tallanto:" in text
    assert "Договоренности: 2026-06-10 - Созвониться после оплаты" in text
    assert "Касания: foton/telegram: 3 (2026-05-01..2026-06-01)" in text


def test_crm_summary_does_not_include_superseded_or_service_data(tmp_path: Path) -> None:
    db = _profiles_db(
        tmp_path,
        fields=[
            _field("child_name", "Аня", child_key="child_1"),
            _field("grade", "7", field_id="old-grade", child_key="child_1", superseded_by="new-grade"),
            _field("grade", "8", field_id="new-grade", child_key="child_1"),
            _field("next_step", "Позвонить завтра source_id raw-123 profile_id cust-1 ИНН 1234567890"),
            _field("source_id", "raw-123"),
            _field("legal_requisites", "ИНН 1234567890 КПП 123456789"),
        ],
    )

    text = render_crm_summary_from_db(db, profile_id="cust-1")

    assert "8 класс" in text
    assert "7 класс" not in text
    assert "source_id" not in text
    assert "profile_id" not in text
    assert "raw-123" not in text
    assert "cust-1" not in text
    assert "ИНН" not in text
    assert "КПП" not in text


def test_crm_summary_enforces_length_limit(tmp_path: Path) -> None:
    long_step = " ".join(["длинная договоренность"] * 120)
    db = _profiles_db(
        tmp_path,
        fields=[
            _field("child_name", "Аня", child_key="child_1"),
            _field("grade", "8", child_key="child_1"),
            _field("subject", "математика", child_key="child_1"),
            _field("next_step", long_step),
            _field("brand_touch", "telegram=120, whatsapp=80, звонки=30, период 2025-01-01..2026-06-10"),
        ],
    )

    text = render_crm_summary_from_db(db, profile_id="cust-1")

    assert len(text) <= CRM_SUMMARY_MAX_CHARS


def test_empty_profile_gives_clear_placeholder(tmp_path: Path) -> None:
    db = _profiles_db(tmp_path, fields=[])

    text = render_crm_summary_from_db(db, phone="+79991234567")

    assert "данных пока недостаточно" in text
    assert "Активных полей профиля нет" in text
    assert "+***4567" in text


def test_profile_phone_index_is_absent_by_default(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("PROFILE_PHONE_INDEX", raising=False)

    db = _profiles_db(tmp_path, fields=[])
    with sqlite3.connect(db) as con:
        columns = {row[1] for row in con.execute("PRAGMA table_info(customer_profiles)").fetchall()}

    assert "primary_phone_norm" not in columns
    assert "данных пока недостаточно" in render_crm_summary_from_db(db, phone="9991234567")


def test_profile_phone_index_flag_adds_norm_column_and_preserves_lookup(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PROFILE_PHONE_INDEX", "1")

    db = _profiles_db(tmp_path, fields=[])
    with sqlite3.connect(db) as con:
        columns = {row[1] for row in con.execute("PRAGMA table_info(customer_profiles)").fetchall()}
        indexes = {row[1] for row in con.execute("PRAGMA index_list(customer_profiles)").fetchall()}
        row = con.execute("SELECT primary_phone_norm FROM customer_profiles WHERE profile_id = 'cust-1'").fetchone()

    assert "primary_phone_norm" in columns
    assert "idx_customer_profiles_phone_norm" in indexes
    assert row[0] == "+79991234567"
    assert "данных пока недостаточно" in render_crm_summary_from_db(db, phone="9991234567")


def test_phone_with_multiple_profiles_requires_manual_selection_without_first_preview(tmp_path: Path) -> None:
    db = tmp_path / "customer_profiles.sqlite"
    with CustomerProfileSQLiteStore(db) as store:
        store.replace_profiles(
            build_id="test-build",
            built_at=NOW,
            timeline_db_path=tmp_path / "customer_timeline.sqlite",
            timeline_db_sha256="0" * 64,
            profiles=[
                ProfileSnapshot(
                    profile_id="cust-1",
                    tenant_id="foton",
                    primary_phone="+79991234567",
                    display_name="Мария",
                    source_event_count=1,
                    last_event_at=NOW,
                ),
                ProfileSnapshot(
                    profile_id="cust-2",
                    tenant_id="foton",
                    primary_phone="+7 999 123-45-67",
                    display_name="Анна",
                    source_event_count=1,
                    last_event_at=NOW,
                ),
            ],
            fields=[
                _field("next_step", "Показать нельзя без выбора профиля"),
            ],
        )

    text = render_crm_summary_from_db(db, phone="+79991234567")

    assert "Найдено несколько профилей" in text
    assert "выберите профиль вручную" in text
    assert "+***4567" in text
    assert "Мария" not in text
    assert "Анна" not in text
    assert "Показать нельзя" not in text


def test_preview_cli_writes_optional_out_file(tmp_path: Path, capsys) -> None:
    db = _profiles_db(tmp_path, fields=[_field("next_step", "Написать менеджеру")])
    out = tmp_path / "summary.txt"

    exit_code = main(["--profiles-db", str(db), "--profile-id", "cust-1", "--out", str(out)])

    assert exit_code == 0
    assert "Написать менеджеру" in out.read_text(encoding="utf-8")
    assert "Написать менеджеру" in capsys.readouterr().out
