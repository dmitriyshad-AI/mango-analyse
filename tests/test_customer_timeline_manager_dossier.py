from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest
from openpyxl import load_workbook

from mango_mvp.customer_timeline.contracts import (
    CustomerIdentity,
    CustomerOpportunity,
    IdentityStatus,
    OpportunityType,
    TimelineDirection,
    TimelineEvent,
    TimelineEventType,
)
from mango_mvp.customer_timeline.manager_dossier import (
    build_customer_dossier,
    build_manager_dossier_workbook,
    load_canonical_call_client_texts,
)
from mango_mvp.customer_timeline.store import CustomerTimelineSQLiteStore


NOW = datetime(2026, 7, 3, 12, 0, tzinfo=timezone.utc)


def test_manager_dossier_extracts_interests_and_pains_from_client_text_only(tmp_path: Path) -> None:
    db = _timeline_db(tmp_path)
    _seed_customer_with_call_and_opportunity(db, tmp_path)
    canonical_db = _canonical_calls_db(
        tmp_path,
        {
            "call-client": "Нас интересует летняя школа, телефон +7 916 123-45-67, но переживаем, что не успеваем по математике.",
            "call-manager-only": "",
        },
    )

    with sqlite3.connect(db) as con:
        canonical_calls = load_canonical_call_client_texts(canonical_db)
        dossier = build_customer_dossier(
            con,
            tenant_id="foton",
            customer_id="customer:1",
            canonical_calls=canonical_calls,
        )

    interest_text = "\n".join(item.text for item in dossier.interests)
    pain_text = "\n".join(item.text for item in dossier.pains)
    assert "Из данных: Летняя школа по математике" in interest_text
    assert "Служебная акция из title" not in interest_text
    assert "Нас интересует летняя школа" in interest_text
    assert "916" not in interest_text
    assert "123-45-67" not in interest_text
    assert "[contact]" in interest_text
    assert "переживаем" in pain_text
    assert "не успеваем" in pain_text
    assert "сложно оплатить" not in pain_text


def test_manager_dossier_workbook_stays_under_allowed_root_and_writes_summary(tmp_path: Path) -> None:
    db = _timeline_db(tmp_path)
    _seed_customer_with_call_and_opportunity(db, tmp_path)
    canonical_db = _canonical_calls_db(tmp_path, {"call-client": "Хотим рассмотреть курс, но сложно по времени."})
    out = tmp_path / ".codex_local" / "review" / "dossier.xlsx"

    summary = build_manager_dossier_workbook(
        timeline_db=db,
        allowed_root=tmp_path,
        out_xlsx=out,
        customer_ids=("customer:1",),
        canonical_calls_db=canonical_db,
    )

    assert summary["customers"] == 1
    assert summary["interests_total"] >= 2
    assert summary["pains_total"] == 1
    assert summary["safety"]["write_crm"] is False
    assert out.exists()
    summary_path = out.with_suffix(".summary.json")
    assert json.loads(summary_path.read_text(encoding="utf-8"))["customers"] == 1
    wb = load_workbook(out, read_only=True)
    assert "Оглавление" in wb.sheetnames
    assert "Клиент 1" in wb.sheetnames
    values = [row for row in wb["Клиент 1"].iter_rows(values_only=True)]
    assert ("Интересы", "Из данных: Летняя школа по математике", "products_of_interest") in values
    assert any(row[0] == "Боли" and "сложно по времени" in str(row[1]) for row in values)

    with pytest.raises(ValueError, match=".codex_local"):
        build_manager_dossier_workbook(
            timeline_db=db,
            allowed_root=tmp_path,
            out_xlsx=tmp_path / "dossier.xlsx",
            customer_ids=("customer:1",),
        )

    with pytest.raises(ValueError, match="allowed root"):
        build_manager_dossier_workbook(
            timeline_db=db,
            allowed_root=tmp_path / "inside",
            out_xlsx=tmp_path / "outside.xlsx",
            customer_ids=("customer:1",),
        )


def test_manager_dossier_matches_canonical_call_id_and_prefixed_source_id(tmp_path: Path) -> None:
    db = _timeline_db(tmp_path)
    _seed_customer_with_call_and_opportunity(db, tmp_path)
    store = CustomerTimelineSQLiteStore(db, allowed_root=tmp_path)
    try:
        store.upsert_event(
            TimelineEvent(
                tenant_id="foton",
                customer_id="customer:1",
                event_type=TimelineEventType.MANGO_CALL,
                event_at=NOW,
                source_system="mango_call",
                source_id="source-without-canonical-match",
                direction=TimelineDirection.INBOUND,
                summary="Summary не должен использоваться.",
                match_status="strong_unique",
                record={"canonical_call_id": "canonical-from-record"},
                created_at=NOW,
            )
        )
        store.upsert_event(
            TimelineEvent(
                tenant_id="foton",
                customer_id="customer:1",
                event_type=TimelineEventType.MANGO_CALL,
                event_at=NOW,
                source_system="mango_call",
                source_id="call:prefixed-id",
                direction=TimelineDirection.INBOUND,
                summary="Summary не должен использоваться.",
                match_status="strong_unique",
                created_at=NOW,
            )
        )
    finally:
        store.close()
    canonical_db = _canonical_calls_db(
        tmp_path,
        {
            "canonical-from-record": "Рассматриваем курс по программированию.",
            "prefixed-id": "Хотим интенсив, но очень сложно с расписанием.",
        },
    )

    with sqlite3.connect(db) as con:
        dossier = build_customer_dossier(
            con,
            tenant_id="foton",
            customer_id="customer:1",
            canonical_calls=load_canonical_call_client_texts(canonical_db),
        )

    interest_text = "\n".join(item.text for item in dossier.interests)
    pain_text = "\n".join(item.text for item in dossier.pains)
    assert "Рассматриваем курс по программированию" in interest_text
    assert "Хотим интенсив" in interest_text
    assert "сложно с расписанием" in pain_text


def _timeline_db(tmp_path: Path) -> Path:
    db = tmp_path / "timeline.sqlite"
    store = CustomerTimelineSQLiteStore(db, allowed_root=tmp_path)
    store.close()
    return db


def _seed_customer_with_call_and_opportunity(db: Path, tmp_path: Path) -> None:
    store = CustomerTimelineSQLiteStore(db, allowed_root=tmp_path)
    try:
        store.upsert_customer(
            CustomerIdentity(
                tenant_id="foton",
                customer_id="customer:1",
                identity_status=IdentityStatus.STRONG,
                display_name="Тестовый клиент",
                primary_phone="+79000000000",
                primary_email="parent@example.com",
                summary={"products_of_interest": ["ОГЭ по физике"]},
            )
        )
        store.upsert_opportunity(
            CustomerOpportunity(
                tenant_id="foton",
                customer_id="customer:1",
                opportunity_type=OpportunityType.AMO_DEAL,
                source_system="amo",
                source_id="lead-1",
                title="Летняя школа по математике",
                product_context={"products_of_interest": ["Летняя школа по математике"]},
                opened_at=NOW,
            )
        )
        store.upsert_opportunity(
            CustomerOpportunity(
                tenant_id="foton",
                customer_id="customer:1",
                opportunity_type=OpportunityType.AMO_DEAL,
                source_system="amo",
                source_id="lead-title-only",
                title="Служебная акция из title",
                product_context={},
                opened_at=NOW,
            )
        )
        store.upsert_event(
            TimelineEvent(
                tenant_id="foton",
                customer_id="customer:1",
                event_type=TimelineEventType.MANGO_CALL,
                event_at=NOW,
                source_system="mango_call",
                source_id="call-client",
                direction=TimelineDirection.INBOUND,
                summary="Клиент интересуется летней школой.",
                match_status="strong_unique",
                created_at=NOW,
            )
        )
        store.upsert_event(
            TimelineEvent(
                tenant_id="foton",
                customer_id="customer:1",
                event_type=TimelineEventType.MANGO_CALL,
                event_at=NOW,
                source_system="mango_call",
                source_id="call-manager-only",
                direction=TimelineDirection.INBOUND,
                summary="Менеджер говорит: клиенту сложно оплатить.",
                match_status="strong_unique",
                created_at=NOW,
            )
        )
        store.upsert_event(
            TimelineEvent(
                tenant_id="foton",
                customer_id="customer:1",
                event_type=TimelineEventType.EMAIL_MESSAGE,
                event_at=NOW,
                source_system="mail_archive_stage2",
                source_id="mail-1",
                direction=TimelineDirection.INBOUND,
                summary="Письмо для попадания в сегмент звонки+письма.",
                match_status="strong_unique",
                created_at=NOW,
            )
        )
    finally:
        store.close()


def _canonical_calls_db(tmp_path: Path, transcripts: dict[str, str]) -> Path:
    db = tmp_path / "canonical_calls.sqlite"
    with sqlite3.connect(db) as con:
        con.execute("CREATE TABLE canonical_calls (canonical_call_id TEXT PRIMARY KEY, transcript_client TEXT)")
        con.executemany(
            "INSERT INTO canonical_calls (canonical_call_id, transcript_client) VALUES (?, ?)",
            sorted(transcripts.items()),
        )
        con.commit()
    return db
