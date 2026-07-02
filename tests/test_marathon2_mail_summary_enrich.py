from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

from mango_mvp.customer_timeline import CustomerIdentity, CustomerTimelineSQLiteStore, IdentityLink
from mango_mvp.customer_timeline.contracts import TimelineEvent, TimelineEventType
from scripts.run_marathon2_mail_summary_enrich import (
    EnrichConfig,
    PROMPT_VERSION,
    _anti_hallucination_reasons,
    _ensure_local_staging_out_dir,
    _load_crm_customer_ids,
    _load_target_mail_rows,
    _prepare_rows_with_summaries,
    _text_hash,
)


NOW = datetime(2026, 7, 2, 12, 0, tzinfo=timezone.utc)


def test_marathon2_mail_summary_target_selection_uses_only_crm_customers(tmp_path: Path) -> None:
    db = _seed_staging(tmp_path)
    export_dir = tmp_path / ".codex_local" / "staging" / "e5_crm_export"
    export_dir.mkdir(parents=True)
    (export_dir / "pilot_20_crm_card_candidates.jsonl").write_text(
        json.dumps({"customer_id": "customer:target"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    ids = _load_crm_customer_ids(export_dir)

    with sqlite3.connect(db) as con:
        con.row_factory = sqlite3.Row
        rows = _load_target_mail_rows(con, tenant_id="foton", customer_ids=sorted(ids))

    assert ids == {"customer:target"}
    assert [row["customer_id"] for row in rows] == ["customer:target"]
    assert rows[0]["contact_phone"] == "+79161234567"
    assert rows[0]["message_sha256"] == "sha-target"


def test_marathon2_mail_summary_short_text_is_cached_without_llm(tmp_path: Path) -> None:
    db = _seed_staging(tmp_path)
    config = _config(tmp_path, db)
    with sqlite3.connect(db) as con:
        con.row_factory = sqlite3.Row
        con.execute(
            """
            CREATE TABLE email_summary_cache_v1 (
              message_sha256 TEXT PRIMARY KEY,
              text_sha256 TEXT NOT NULL,
              prompt_version TEXT NOT NULL,
              provider TEXT NOT NULL,
              model TEXT NOT NULL,
              reasoning TEXT NOT NULL,
              source_kind TEXT NOT NULL,
              summary_text TEXT NOT NULL,
              summary_payload_json TEXT NOT NULL,
              created_at TEXT NOT NULL
            )
            """
        )
        rows = _load_target_mail_rows(con, tenant_id="foton", customer_ids=["customer:target"])
        prepared, stats = _prepare_rows_with_summaries(con, rows, config=config)
        repeat, repeat_stats = _prepare_rows_with_summaries(con, rows, config=config)

    assert stats["llm_calls_total"] == 0
    assert stats["cache_misses_short"] == 1
    assert repeat_stats["cache_hits"] == 1
    assert repeat_stats["llm_calls_total"] == 0
    assert prepared[0]["summary_payload"]["summary"] == "Короткое письмо про расписание."
    assert repeat[0]["summary_payload"] == prepared[0]["summary_payload"]


def test_marathon2_mail_summary_cache_uses_email_brand_not_customer_brand(tmp_path: Path) -> None:
    db = _seed_staging(tmp_path)
    config = _config(tmp_path, db)
    with sqlite3.connect(db) as con:
        con.row_factory = sqlite3.Row
        con.execute(
            """
            CREATE TABLE email_summary_cache_v1 (
              message_sha256 TEXT PRIMARY KEY,
              text_sha256 TEXT NOT NULL,
              prompt_version TEXT NOT NULL,
              provider TEXT NOT NULL,
              model TEXT NOT NULL,
              reasoning TEXT NOT NULL,
              source_kind TEXT NOT NULL,
              summary_text TEXT NOT NULL,
              summary_payload_json TEXT NOT NULL,
              created_at TEXT NOT NULL
            )
            """
        )
        con.execute(
            """
            UPDATE timeline_events
            SET record_json = json_set(record_json, '$.record.brand', 'foton')
            WHERE source_id = 'sha-target'
            """
        )
        rows = _load_target_mail_rows(con, tenant_id="foton", customer_ids=["customer:target"])
        _, stats = _prepare_rows_with_summaries(con, rows, config=config)
        con.execute(
            """
            UPDATE timeline_events
            SET record_json = json_set(record_json, '$.record.customer_brand', 'unpk')
            WHERE source_id = 'sha-target'
            """
        )
        rows_after_enrich = _load_target_mail_rows(con, tenant_id="foton", customer_ids=["customer:target"])
        _, repeat_stats = _prepare_rows_with_summaries(con, rows_after_enrich, config=config)

    assert rows[0]["brand"] == "foton"
    assert rows_after_enrich[0]["brand"] == "foton"
    assert stats["cache_misses_short"] == 1
    assert repeat_stats["cache_hits"] == 1
    assert repeat_stats["llm_calls_total"] == 0


def test_marathon2_mail_summary_cache_migrates_legacy_brand_hash(tmp_path: Path) -> None:
    db = _seed_staging(tmp_path)
    config = _config(tmp_path, db)
    with sqlite3.connect(db) as con:
        con.row_factory = sqlite3.Row
        con.execute(
            """
            CREATE TABLE email_summary_cache_v1 (
              message_sha256 TEXT PRIMARY KEY,
              text_sha256 TEXT NOT NULL,
              prompt_version TEXT NOT NULL,
              provider TEXT NOT NULL,
              model TEXT NOT NULL,
              reasoning TEXT NOT NULL,
              source_kind TEXT NOT NULL,
              summary_text TEXT NOT NULL,
              summary_payload_json TEXT NOT NULL,
              created_at TEXT NOT NULL
            )
            """
        )
        con.execute(
            """
            UPDATE timeline_events
            SET record_json = json_set(record_json, '$.record.brand', 'foton')
            WHERE source_id = 'sha-target'
            """
        )
        row = _load_target_mail_rows(con, tenant_id="foton", customer_ids=["customer:target"])[0]
        legacy_hash = sha256(
            "\n".join(
                [
                    str(row.get("subject_full") or ""),
                    str(row.get("direction") or ""),
                    str(row.get("brand") or ""),
                    str(row.get("full_clean_text") or ""),
                ]
            ).encode("utf-8")
        ).hexdigest()
        payload = {"summary": "Старый кэш", "topic": "Расписание", "next_step": None}
        con.execute(
            """
            INSERT INTO email_summary_cache_v1 (
              message_sha256, text_sha256, prompt_version, provider, model, reasoning,
              source_kind, summary_text, summary_payload_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "sha-target",
                legacy_hash,
                PROMPT_VERSION,
                "stub",
                "stub",
                "none",
                "llm",
                "Старый кэш",
                json.dumps(payload, ensure_ascii=False),
                NOW.isoformat(),
            ),
        )
        prepared, stats = _prepare_rows_with_summaries(con, [row], config=config)
        new_hash = con.execute(
            "SELECT text_sha256 FROM email_summary_cache_v1 WHERE message_sha256='sha-target'"
        ).fetchone()[0]

    assert legacy_hash != _text_hash(row)
    assert stats["cache_hits"] == 1
    assert stats["llm_calls_total"] == 0
    assert prepared[0]["summary_payload"]["summary"] == "Старый кэш"
    assert new_hash == _text_hash(row)


def test_marathon2_mail_summary_cache_does_not_hide_changed_text(tmp_path: Path) -> None:
    db = _seed_staging(tmp_path)
    config = _config(tmp_path, db)
    with sqlite3.connect(db) as con:
        con.row_factory = sqlite3.Row
        con.execute(
            """
            CREATE TABLE email_summary_cache_v1 (
              message_sha256 TEXT PRIMARY KEY,
              text_sha256 TEXT NOT NULL,
              prompt_version TEXT NOT NULL,
              provider TEXT NOT NULL,
              model TEXT NOT NULL,
              reasoning TEXT NOT NULL,
              source_kind TEXT NOT NULL,
              summary_text TEXT NOT NULL,
              summary_payload_json TEXT NOT NULL,
              created_at TEXT NOT NULL
            )
            """
        )
        rows = _load_target_mail_rows(con, tenant_id="foton", customer_ids=["customer:target"])
        _, stats = _prepare_rows_with_summaries(con, rows, config=config)
        con.execute(
            """
            UPDATE timeline_events
            SET record_json = json_set(record_json, '$.record.full_clean_text', ?)
            WHERE source_id = 'sha-target'
            """,
            ("Изменённое длинное письмо. " * 40,),
        )
        changed_rows = _load_target_mail_rows(con, tenant_id="foton", customer_ids=["customer:target"])
        _, repeat_stats = _prepare_rows_with_summaries(con, changed_rows, config=config)

    assert stats["cache_misses_short"] == 1
    assert repeat_stats["cache_hits"] == 0
    assert repeat_stats["cache_misses_long"] == 1
    assert repeat_stats["missing_long_requires_summary"] == 1


def test_marathon2_mail_summary_out_dir_must_stay_under_local_staging(tmp_path: Path) -> None:
    _ensure_local_staging_out_dir(tmp_path / ".codex_local" / "staging" / "ok", allowed_root=tmp_path)

    try:
        _ensure_local_staging_out_dir(tmp_path / "Foton" / "bad", allowed_root=tmp_path)
    except ValueError as exc:
        assert ".codex_local/staging" in str(exc)
    else:
        raise AssertionError("out_dir outside .codex_local/staging must fail")


def test_marathon2_mail_summary_hallucination_gate_blocks_new_money() -> None:
    row = {
        "subject_full": "Расписание",
        "full_clean_text": "Занятие в воскресенье 10:50-12:30.",
    }
    payload = {
        "summary": "Занятие в воскресенье 10:50-12:30, стоимость 126 000 руб.",
        "topic": "Расписание",
        "next_step": None,
        "amount_rub": 126000,
    }

    reasons = _anti_hallucination_reasons(row, payload)

    assert "amount_rub_not_in_source" in reasons


def _seed_staging(tmp_path: Path) -> Path:
    stage = tmp_path / ".codex_local" / "staging"
    stage.mkdir(parents=True)
    db = stage / "customer_timeline_staging.sqlite"
    with CustomerTimelineSQLiteStore(db, allowed_root=tmp_path) as store:
        for customer_id, phone in (("customer:target", "+79161234567"), ("customer:other", "+79160000000")):
            store.upsert_customer(
                CustomerIdentity(tenant_id="foton", customer_id=customer_id, identity_status="strong", created_at=NOW)
            )
            store.upsert_identity_link(
                IdentityLink(
                    tenant_id="foton",
                    customer_id=customer_id,
                    link_type="phone",
                    link_value=phone,
                    source_system="fixture",
                    source_ref=f"phone:{customer_id}",
                    match_class="strong_unique",
                    first_seen_at=NOW,
                    last_seen_at=NOW,
                )
            )
        for customer_id, source_id in (("customer:target", "sha-target"), ("customer:other", "sha-other")):
            store.upsert_event(
                TimelineEvent(
                    tenant_id="foton",
                    customer_id=customer_id,
                    event_type=TimelineEventType.EMAIL_MESSAGE,
                    event_at=NOW,
                    source_system="mail_archive_stage2",
                    source_id=source_id,
                    direction="inbound",
                    subject="Расписание",
                    summary="Расписание",
                    match_status="strong_unique",
                    record={"full_clean_text": "Короткое письмо про расписание."},
                    created_at=NOW,
                )
            )
    return db


def _config(tmp_path: Path, db: Path) -> EnrichConfig:
    return EnrichConfig(
        timeline_db=db,
        prod_timeline_db=db,
        allowed_root=tmp_path,
        out_dir=tmp_path / ".codex_local" / "staging" / "block1_mail_summary",
        crm_export_dir=tmp_path / ".codex_local" / "staging" / "e5_crm_export",
        review_workbook=None,
        tenant_id="foton",
        provider="stub",
        model="stub",
        reasoning="none",
        batch_size=10,
        max_llm_calls=0,
        codex_home=None,
        summarize=False,
        apply=False,
    )
