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
    assert "переживаем" in pain_text.casefold()
    assert "не успеваем" in pain_text.casefold()
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
    assert ("Интересы", "Из данных: Летняя школа по математике", "Данные клиента/сделки") in values
    assert any(row[0] == "Боли" and "сложно по времени" in str(row[1]).casefold() for row in values)

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


def test_manager_dossier_workbook_includes_full_manager_sections(tmp_path: Path) -> None:
    db = _timeline_db(tmp_path)
    _seed_customer_with_call_and_opportunity(db, tmp_path)
    _seed_full_dossier_tables(db)
    reconcile = tmp_path / ".codex_local" / "reconcile.json"
    reconcile.parent.mkdir(parents=True)
    reconcile.write_text(
        json.dumps(
            {
                "status": "checked",
                "generated_at": "2026-07-03T12:00:00+00:00",
                "customers_checked": 42,
                "customers_changed": 0,
                "snapshot_stale": False,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    out = tmp_path / ".codex_local" / "review" / "dossier_full.xlsx"

    summary = build_manager_dossier_workbook(
        timeline_db=db,
        allowed_root=tmp_path,
        out_xlsx=out,
        customer_ids=("customer:1",),
        canonical_calls_db=tmp_path / "missing.sqlite",
        reconcile_json=reconcile,
    )

    assert summary["canonical_calls_loaded"] == 0
    assert "not found" in summary["canonical_calls_warning"]
    assert summary["family_rows_total"] == 1
    assert summary["money_rows_total"] == 2
    assert summary["signals_total"] == 1
    assert summary["objections_total"] == 1
    assert summary["next_step_rows_total"] == 1
    wb = load_workbook(out, read_only=True)
    values = [row for row in wb["Клиент 1"].iter_rows(values_only=True)]
    joined = "\n".join(str(cell) for row in values for cell in row if cell)
    assert "0 расхождений из 42" in joined
    assert "Иван" in joined and "класс: 8" in joined and "предметы: математика" in joined
    assert "факт оплат" in joined and "120 000 руб." in joined
    assert "план сделок" in joined and "80 000 руб." in joined
    assert "сделка зависла" in joined
    assert ("Следующий шаг", "Позвонить в понедельник по оплате", "Сигнал Customer Timeline") in values
    assert values[0] == ("Раздел", "Значение", "Откуда")
    assert "family_links_v1" not in joined
    assert "derived_signals" not in joined
    assert "price: Дорого, просит рассрочку." in joined
    assert "Письмо «Расписание»: полный текст в базе." in joined
    assert "Требуется ручная проверка модельной выжимки" not in joined


def test_manager_dossier_skips_missing_customer_ids(tmp_path: Path) -> None:
    db = _timeline_db(tmp_path)
    _seed_customer_with_call_and_opportunity(db, tmp_path)
    out = tmp_path / ".codex_local" / "review" / "dossier_missing.xlsx"

    summary = build_manager_dossier_workbook(
        timeline_db=db,
        allowed_root=tmp_path,
        out_xlsx=out,
        customer_ids=("customer:1", "customer:missing"),
    )

    assert summary["requested_customers"] == 2
    assert summary["customers"] == 1
    assert summary["missing_customer_ids_count"] == 1
    assert summary["missing_customer_ids_sample"] == ["customer:missing"]
    assert out.exists()


def test_manager_dossier_omits_generic_history_next_step(tmp_path: Path) -> None:
    db = _timeline_db(tmp_path)
    _seed_customer_with_call_and_opportunity(db, tmp_path)
    _seed_full_dossier_tables(db, signal_action="Посмотреть историю и ответить с учётом прошлого запроса клиента.")
    out = tmp_path / ".codex_local" / "review" / "dossier_no_generic_next.xlsx"

    summary = build_manager_dossier_workbook(
        timeline_db=db,
        allowed_root=tmp_path,
        out_xlsx=out,
        customer_ids=("customer:1",),
    )

    assert summary["signals_total"] == 1
    assert summary["next_step_rows_total"] == 0
    wb = load_workbook(out, read_only=True)
    values = [row for row in wb["Клиент 1"].iter_rows(values_only=True)]
    assert not any(row[0] == "Следующий шаг" for row in values)
    joined = "\n".join(str(cell) for row in values for cell in row if cell is not None)
    assert "Посмотреть историю" not in joined


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
    assert "сложно с расписанием" in pain_text.casefold()


def test_manager_dossier_cleans_speech_fillers_and_repeated_words(tmp_path: Path) -> None:
    db = _timeline_db(tmp_path)
    _seed_customer_with_call_and_opportunity(db, tmp_path)
    canonical_db = _canonical_calls_db(
        tmp_path,
        {
            "call-client": (
                "Ну эээ вот нас нас интересует олимпиадная математика онлайн, "
                "и хотим попробовать, потому что очень переживаем переживаем из-за экзамена"
            )
        },
    )

    with sqlite3.connect(db) as con:
        dossier = build_customer_dossier(
            con,
            tenant_id="foton",
            customer_id="customer:1",
            canonical_calls=load_canonical_call_client_texts(canonical_db),
        )

    interest_from_call = next(item.text for item in dossier.interests if item.source == "mango_call:call-client")
    pain_from_call = next(item.text for item in dossier.pains if item.source == "mango_call:call-client")

    assert "эээ" not in interest_from_call.casefold()
    assert "вот нас нас" not in interest_from_call.casefold()
    assert "Нас интересует олимпиадная математика онлайн" in interest_from_call
    assert interest_from_call.endswith(".")
    assert "переживаем переживаем" not in pain_from_call.casefold()
    assert pain_from_call.endswith(".")


def test_manager_dossier_interest_quote_is_sentence_not_raw_transcript(tmp_path: Path) -> None:
    db = _timeline_db(tmp_path)
    _seed_customer_with_call_and_opportunity(db, tmp_path)
    canonical_db = _canonical_calls_db(
        tmp_path,
        {
            "call-client": (
                "Да я как раз хотела сама вам звонить Нас интересует математика очная "
                "Можете там счет скинуть или как там сколько будет полгод"
            )
        },
    )

    with sqlite3.connect(db) as con:
        dossier = build_customer_dossier(
            con,
            tenant_id="foton",
            customer_id="customer:1",
            canonical_calls=load_canonical_call_client_texts(canonical_db),
        )

    interest_from_call = next(item.text for item in dossier.interests if item.source == "mango_call:call-client")

    assert interest_from_call == "Интерес из звонка: Нас интересует математика очная."
    assert "сколько будет полгод" not in interest_from_call


def test_manager_dossier_ignores_generic_wanted_phrases_without_product_context(tmp_path: Path) -> None:
    db = _timeline_db(tmp_path)
    _seed_customer_with_call_and_opportunity(db, tmp_path)
    canonical_db = _canonical_calls_db(
        tmp_path,
        {
            "call-client": (
                "Хотела спросить, как дела. "
                "Мы хотим уже бы хотим, чтобы он начал учиться. "
                "Нас интересует математика онлайн."
            )
        },
    )

    with sqlite3.connect(db) as con:
        dossier = build_customer_dossier(
            con,
            tenant_id="foton",
            customer_id="customer:1",
            canonical_calls=load_canonical_call_client_texts(canonical_db),
        )

    interest_text = "\n".join(item.text for item in dossier.interests if item.source == "mango_call:call-client")

    assert "Хотела спросить" not in interest_text
    assert "Мы хотим уже бы хотим" not in interest_text
    assert "Нас интересует математика онлайн" in interest_text


def test_manager_dossier_pain_quote_trims_adjacent_asr_tail(tmp_path: Path) -> None:
    db = _timeline_db(tmp_path)
    _seed_customer_with_call_and_opportunity(db, tmp_path)
    canonical_db = _canonical_calls_db(
        tmp_path,
        {"call-client": "Переживаю, у вас же мест мало Сейчас Катя у вас рисунок карандаши."},
    )

    with sqlite3.connect(db) as con:
        dossier = build_customer_dossier(
            con,
            tenant_id="foton",
            customer_id="customer:1",
            canonical_calls=load_canonical_call_client_texts(canonical_db),
        )

    pain_text = "\n".join(item.text for item in dossier.pains if item.source == "mango_call:call-client")

    assert "Переживаю, у вас же мест мало." in pain_text
    assert "рисунок" not in pain_text.casefold()


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


def _seed_full_dossier_tables(db: Path, *, signal_action: str = "Позвонить в понедельник по оплате") -> None:
    with sqlite3.connect(db) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS family_links_v1 (
              tenant_id TEXT NOT NULL,
              family_id TEXT NOT NULL,
              customer_id TEXT NOT NULL,
              child_key TEXT NOT NULL,
              canonical_name TEXT NOT NULL,
              name_variants_json TEXT NOT NULL,
              grades_json TEXT NOT NULL,
              subjects_json TEXT NOT NULL,
              brand TEXT NOT NULL,
              status TEXT NOT NULL,
              confidence TEXT NOT NULL,
              reason TEXT NOT NULL,
              source_refs_json TEXT NOT NULL,
              evidence_count INTEGER NOT NULL,
              created_at TEXT NOT NULL,
              record_hash TEXT NOT NULL,
              record_json TEXT NOT NULL,
              PRIMARY KEY (tenant_id, family_id, customer_id, child_key)
            )
            """
        )
        con.execute(
            """
            INSERT OR REPLACE INTO family_links_v1
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "foton",
                "family:1",
                "customer:1",
                "child:1",
                "Иван",
                json.dumps(["Иван"], ensure_ascii=False),
                json.dumps(["8"], ensure_ascii=False),
                json.dumps(["математика"], ensure_ascii=False),
                "foton",
                "confident",
                "high",
                "single_child_family",
                json.dumps(["test"], ensure_ascii=False),
                1,
                NOW.isoformat(),
                "hash-family",
                "{}",
            ),
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS customer_purchases_v1 (
              tenant_id TEXT NOT NULL,
              customer_id TEXT NOT NULL,
              period TEXT NOT NULL,
              money_kind TEXT NOT NULL,
              total_in REAL,
              total_out REAL,
              deals_cnt INTEGER NOT NULL DEFAULT 0,
              last_purchase_at TEXT,
              sources_json TEXT NOT NULL,
              computability TEXT NOT NULL,
              code_version TEXT NOT NULL,
              PRIMARY KEY (tenant_id, customer_id, period, money_kind)
            )
            """
        )
        con.executemany(
            """
            INSERT OR REPLACE INTO customer_purchases_v1
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                ("foton", "customer:1", "all_time", "fact", 120000, 0, 1, "2026-06-01", "[]", "ok", "test"),
                ("foton", "customer:1", "all_time", "plan", 80000, 0, 1, None, "[]", "ok", "test"),
            ],
        )
        con.execute(
            """
            INSERT OR REPLACE INTO derived_signals
            (signal_id, tenant_id, customer_id, opportunity_id, event_id, signal_type, severity, status, expires_at,
             confidence, requires_manager_review, created_at, record_hash, record_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "signal:1",
                "foton",
                "customer:1",
                None,
                None,
                "deal_stalling",
                "high",
                "active",
                "2026-07-10T00:00:00+00:00",
                0.9,
                1,
                NOW.isoformat(),
                "hash-signal",
                json.dumps({"recommended_action": signal_action, "evidence_text": "нет движения 7 дней"}, ensure_ascii=False),
            ),
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS customer_objections_v1 (
              tenant_id TEXT NOT NULL,
              customer_id TEXT NOT NULL,
              source_event_id TEXT NOT NULL,
              source_channel TEXT NOT NULL,
              objection_type TEXT NOT NULL,
              quote_preview TEXT NOT NULL,
              budget_hint_rub INTEGER,
              price_sensitivity TEXT NOT NULL,
              extracted_at TEXT NOT NULL,
              extractor_version TEXT NOT NULL,
              speaker TEXT NOT NULL DEFAULT 'unknown',
              direction TEXT NOT NULL DEFAULT 'unknown',
              confidence TEXT NOT NULL DEFAULT 'low',
              PRIMARY KEY (tenant_id, customer_id, source_event_id, objection_type)
            )
            """
        )
        con.execute(
            """
            INSERT OR REPLACE INTO customer_objections_v1
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "foton",
                "customer:1",
                "event:email",
                "email",
                "price",
                "дорого, просит рассрочку",
                60000,
                "high",
                NOW.isoformat(),
                "test",
                "client",
                "inbound",
                "high",
            ),
        )
        con.execute(
            """
            UPDATE timeline_events
            SET subject = 'Расписание',
                summary = 'Требуется ручная проверка модельной выжимки: текст короткий'
            WHERE tenant_id = 'foton' AND customer_id = 'customer:1' AND event_type = 'email_message'
            """
        )
        con.commit()
