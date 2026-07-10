from __future__ import annotations

import json
import sqlite3
from types import SimpleNamespace
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

from mango_mvp.customer_timeline import CustomerIdentity, CustomerTimelineSQLiteStore, IdentityLink
from mango_mvp.customer_timeline.contracts import TimelineEvent, TimelineEventType
import scripts.run_marathon2_mail_summary_enrich as enrich_module
from scripts.run_marathon2_mail_summary_enrich import (
    EnrichConfig,
    PROMPT_VERSION,
    _anti_hallucination_reasons,
    _ensure_cache_table,
    _ensure_local_staging_out_dir,
    _load_crm_customer_ids,
    _load_review_customer_ids,
    _load_target_mail_rows,
    _prepare_rows_with_summaries,
    _sanitize_summary_payload_for_stage2,
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


def test_marathon2_mail_summary_prefers_batch_ready_over_pilot_customers(tmp_path: Path) -> None:
    export_dir = tmp_path / ".codex_local" / "staging" / "e5_crm_export"
    export_dir.mkdir(parents=True)
    (export_dir / "pilot_20_crm_card_candidates.jsonl").write_text(
        json.dumps({"customer_id": "customer:pilot"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (export_dir / "batch_ready_crm_card_candidates.jsonl").write_text(
        json.dumps({"customer_id": "customer:ready"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    assert _load_crm_customer_ids(export_dir) == {"customer:ready"}


def test_marathon2_mail_summary_prefers_all_candidates_over_ready_and_pilot(tmp_path: Path) -> None:
    export_dir = tmp_path / ".codex_local" / "staging" / "e5_crm_export"
    export_dir.mkdir(parents=True)
    (export_dir / "pilot_20_crm_card_candidates.jsonl").write_text(
        json.dumps({"customer_id": "customer:pilot"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (export_dir / "batch_ready_crm_card_candidates.jsonl").write_text(
        json.dumps({"customer_id": "customer:ready"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (export_dir / "all_candidates_crm_card_candidates.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"customer_id": "customer:one"}, ensure_ascii=False),
                json.dumps({"customer_id": "customer:two"}, ensure_ascii=False),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert _load_crm_customer_ids(export_dir) == {"customer:one", "customer:two"}


def test_marathon2_mail_summary_loads_review_customer_ids_from_workbook(tmp_path: Path) -> None:
    from openpyxl import Workbook

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Обзор клиентов"
    sheet.append(["name", "customer_id"])
    sheet.append(["one", "customer:review"])
    sheet.append(["empty", None])
    path = tmp_path / "review.xlsx"
    workbook.save(path)

    assert _load_review_customer_ids(path) == {"customer:review"}


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


def test_marathon2_mail_summary_cache_accepts_legacy_brand_hash_without_writeback(tmp_path: Path) -> None:
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
        stored_hash = con.execute(
            "SELECT text_sha256 FROM email_summary_cache_v1 WHERE message_sha256='sha-target'"
        ).fetchone()[0]

    assert legacy_hash != _text_hash(row)
    assert stats["cache_hits"] == 1
    assert stats["llm_calls_total"] == 0
    assert prepared[0]["summary_payload"]["summary"] == "Старый кэш"
    assert stored_hash == legacy_hash


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


def test_marathon2_mail_summary_sanitizes_cached_payload_in_memory_without_writeback(tmp_path: Path) -> None:
    db = _seed_staging(tmp_path)
    config = _config(tmp_path, db)
    with sqlite3.connect(db) as con:
        con.row_factory = sqlite3.Row
        _ensure_cache_table(con)
        con.execute(
            """
            UPDATE timeline_events
            SET record_json = json_set(record_json, '$.record.full_clean_text', ?)
            WHERE source_id = 'sha-target'
            """,
            ("Стоимость курса 2 000 000 руб.",),
        )
        row = _load_target_mail_rows(con, tenant_id="foton", customer_ids=["customer:target"])[0]
        payload = {
            "message_sha256": "sha-target",
            "summary": "Стоимость курса 2 000 000 руб.",
            "topic": "Стоимость 2 000 000 руб.",
            "next_step": None,
            "confidence": 0.9,
            "extraction_source": "model",
            "event_type": "other",
            "money_direction": "none",
            "student_name": None,
            "payer_name": None,
            "contact_name": None,
            "grade": None,
            "subject_area": None,
            "amount_rub": 2_000_000,
            "amount_kind": "quote",
            "amount_is_total": True,
            "amount_items": [{"amount_rub": 2_000_000, "amount_kind": "quote", "description": "2 000 000 руб.", "is_total": True}],
            "amount_uncertain": False,
            "deadline_date": None,
            "contract_no": None,
            "document_no": None,
            "requisites": [],
            "has_attachment": False,
            "is_plain_acknowledgement": False,
        }
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
                _text_hash(row),
                PROMPT_VERSION,
                "stub",
                "stub",
                "none",
                "llm",
                payload["summary"],
                json.dumps(payload, ensure_ascii=False),
                NOW.isoformat(),
            ),
        )
        prepared, stats = _prepare_rows_with_summaries(con, [row], config=config)
        cached = json.loads(
            con.execute("SELECT summary_payload_json FROM email_summary_cache_v1 WHERE message_sha256='sha-target'").fetchone()[0]
        )

    assert stats["cache_hits"] == 1
    assert stats["sanitized_payloads"] == 1
    assert prepared[0]["summary_payload"]["amount_rub"] is None
    assert cached["amount_rub"] == 2_000_000
    assert "2 000 000" in cached["summary"]


def test_marathon2_mail_summary_missing_full_text_is_review_not_prefix_summary(tmp_path: Path) -> None:
    db = _seed_staging(tmp_path)
    config = _config(tmp_path, db)
    with sqlite3.connect(db) as con:
        con.row_factory = sqlite3.Row
        _ensure_cache_table(con)
        con.execute(
            """
            UPDATE timeline_events
            SET summary = 'Тема-подобный префикс не является полным текстом',
                text_preview = 'Превью тоже не является полным текстом',
                record_json = json_set(record_json, '$.record.full_clean_text', NULL)
            WHERE source_id = 'sha-target'
            """
        )
        rows = _load_target_mail_rows(con, tenant_id="foton", customer_ids=["customer:target"])
        prepared, stats = _prepare_rows_with_summaries(con, rows, config=config)

    payload = prepared[0]["summary_payload"]
    assert rows[0]["body_missing"] is True
    assert prepared[0]["full_clean_text"] == ""
    assert payload["summary_review_needed"] is True
    assert "missing_full_clean_text" in payload["summary_review_reasons"]
    assert payload["summary"] != "Тема-подобный префикс не является полным текстом"
    assert stats["missing_full_text_rows"] == 1
    assert stats["fallback_rows"] == 1
    assert stats["llm_calls_total"] == 0


def test_marathon2_mail_summary_internal_payload_key_becomes_review_before_sanitize(
    tmp_path: Path,
    monkeypatch,
) -> None:
    db = _seed_staging(tmp_path)
    config = _config(tmp_path, db)
    config = EnrichConfig(
        **{
            **config.__dict__,
            "summarize": True,
            "batch_size": 10,
            "max_llm_calls": 1,
        }
    )

    def fake_summarize_items(*args, **kwargs):
        return SimpleNamespace(
            summaries={
                "sha-target": {
                    "message_sha256": "sha-target",
                    "summary": "Клиент просит расписание.",
                    "topic": "Расписание",
                    "next_step": None,
                    "confidence": 0.9,
                    "extraction_source": "model",
                    "event_type": "other",
                    "money_direction": "none",
                    "brand_source": "content",
                }
            },
            llm_calls_total=1,
        )

    monkeypatch.setattr(enrich_module, "summarize_items", fake_summarize_items)

    with sqlite3.connect(db) as con:
        con.row_factory = sqlite3.Row
        _ensure_cache_table(con)
        con.execute(
            """
            UPDATE timeline_events
            SET record_json = json_set(record_json, '$.record.full_clean_text', ?)
            WHERE source_id = 'sha-target'
            """,
            ("Длинное письмо про расписание. " * 40,),
        )
        row = _load_target_mail_rows(con, tenant_id="foton", customer_ids=["customer:target"])[0]
        prepared, stats = _prepare_rows_with_summaries(con, [row], config=config)

    payload = prepared[0]["summary_payload"]
    assert stats["summary_review_needed"] == 1
    assert "internal_marker_leak" in payload["summary_review_reasons"]
    assert "brand_source" not in payload


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


def test_marathon2_mail_summary_hallucination_gate_blocks_missing_business_specifics() -> None:
    row = {
        "subject_full": "Расписание и стоимость",
        "full_clean_text": "Занятия по воскресеньям 10:50-12:30. Стоимость 126 000 руб. Нужно заполнить форму.",
    }
    payload = {
        "summary": "Клиент спрашивает про занятия и оформление.",
        "topic": "Занятия",
        "next_step": None,
        "extraction_source": "model",
        "event_type": "other",
        "money_direction": "none",
    }

    reasons = _anti_hallucination_reasons(row, payload)

    assert "missing_business_specifics" in reasons


def test_marathon2_mail_summary_hallucination_gate_blocks_one_missing_amount_even_if_time_kept() -> None:
    row = {
        "subject_full": "Расписание и стоимость",
        "full_clean_text": "Занятия по воскресеньям 10:50-12:30. Стоимость 126 000 руб. Нужно заполнить форму.",
    }
    payload = {
        "summary": "Занятия проходят по воскресеньям 10:50-12:30, клиенту нужно заполнить форму.",
        "topic": "Расписание",
        "next_step": None,
        "extraction_source": "model",
        "event_type": "other",
        "money_direction": "none",
    }

    reasons = _anti_hallucination_reasons(row, payload)

    assert "missing_business_specifics" in reasons


def test_marathon2_mail_summary_hallucination_gate_does_not_block_subject_only_date() -> None:
    row = {
        "subject_full": "Расписание 10.07",
        "full_clean_text": "Клиент просит расписание занятий.",
    }
    payload = {
        "summary": "Клиент просит расписание занятий.",
        "topic": "Расписание",
        "next_step": None,
        "extraction_source": "model",
        "event_type": "other",
        "money_direction": "none",
    }

    reasons = _anti_hallucination_reasons(row, payload)

    assert "missing_business_specifics" not in reasons


def test_marathon2_mail_summary_hallucination_gate_blocks_internal_marker_text() -> None:
    row = {
        "subject_full": "Письмо",
        "full_clean_text": "Клиент просит расписание.",
    }
    payload = {
        "summary": "brand_source=content, клиент просит расписание.",
        "topic": "Расписание",
        "next_step": None,
        "extraction_source": "model",
        "event_type": "other",
        "money_direction": "none",
    }

    reasons = _anti_hallucination_reasons(row, payload)

    assert "internal_marker_leak" in reasons


def test_marathon2_mail_summary_hallucination_gate_blocks_false_hidden_claim() -> None:
    row = {
        "subject_full": "Письмо",
        "full_clean_text": "Клиент просит расписание.",
    }
    payload = {
        "summary": "Клиент просит расписание, контактные данные скрыты.",
        "topic": "Расписание",
        "next_step": None,
        "extraction_source": "model",
        "event_type": "other",
        "money_direction": "none",
    }

    reasons = _anti_hallucination_reasons(row, payload)

    assert "false_hidden_marker" in reasons


def test_marathon2_mail_summary_hallucination_gate_blocks_hidden_claim_from_mask_token_source() -> None:
    row = {
        "subject_full": "Письмо",
        "full_clean_text": "Телефон клиента [phone], просит расписание.",
    }
    payload = {
        "summary": "Клиент просит расписание, телефон замаскирован.",
        "topic": "Расписание",
        "next_step": None,
        "extraction_source": "model",
        "event_type": "other",
        "money_direction": "none",
    }

    reasons = _anti_hallucination_reasons(row, payload)

    assert "false_hidden_marker" in reasons


def test_marathon2_mail_summary_sanitizer_drops_internal_payload_keys() -> None:
    payload = {
        "message_sha256": "sha",
        "summary": "Клиент просит расписание.",
        "brand_source": "content",
        "brand_mixing_detected": True,
        "memory_status": "usable_memory",
        "event_type": "other",
    }

    sanitized = _sanitize_summary_payload_for_stage2(payload)

    assert "brand_source" not in sanitized
    assert "brand_mixing_detected" not in sanitized
    assert "memory_status" not in sanitized
    assert sanitized["summary"] == "Клиент просит расписание."


def test_marathon2_mail_summary_sanitizer_preserves_m1_quality_flags() -> None:
    payload = {
        "message_sha256": "sha",
        "summary": "Клиент прислал документ.",
        "quality_flags": ["contains_password", "contains_personal_data"],
    }

    sanitized = _sanitize_summary_payload_for_stage2(payload)

    assert sanitized["quality_flags"] == ["contains_password", "contains_personal_data"]


def test_marathon2_mail_summary_hallucination_gate_blocks_new_model_facts() -> None:
    row = {
        "subject_full": "Расписание",
        "full_clean_text": "Здравствуйте, подскажите варианты занятий на выходных.",
    }
    payload = {
        "summary": "Ученик Иван интересуется математикой и оплатил курс.",
        "topic": "Оплата курса",
        "next_step": "Подготовить договор.",
        "event_type": "payment",
        "money_direction": "in",
        "student_name": "Иван",
        "grade": "8 класс",
        "subject_area": "математика",
        "amount_rub": None,
        "amount_kind": "actual_payment",
    }

    reasons = _anti_hallucination_reasons(row, payload)

    assert "student_name_not_in_source" in reasons
    assert "grade_not_in_source" in reasons
    assert "subject_area_not_in_source" in reasons
    assert "actual_payment_not_supported_by_source" in reasons


def test_marathon2_mail_summary_hallucination_gate_blocks_fallback_and_empty_summary() -> None:
    row = {
        "subject_full": "Расписание",
        "full_clean_text": "Клиент просит расписание.",
    }
    payload = {
        "summary": "",
        "topic": "Расписание",
        "next_step": None,
        "extraction_source": "fallback",
        "event_type": "other",
        "money_direction": "none",
    }

    reasons = _anti_hallucination_reasons(row, payload)

    assert "extraction_source_not_model" in reasons
    assert "empty_summary" in reasons


def test_marathon2_mail_summary_hallucination_gate_blocks_unsupported_next_step() -> None:
    row = {
        "subject_full": "Расписание",
        "full_clean_text": "Клиент просит расписание занятий на выходных.",
    }
    payload = {
        "summary": "Клиент просит расписание занятий на выходных.",
        "topic": "Расписание",
        "next_step": "Подготовить договор на обучение.",
        "extraction_source": "model",
        "event_type": "other",
        "money_direction": "none",
    }

    reasons = _anti_hallucination_reasons(row, payload)

    assert "next_step_not_supported_by_source" in reasons


def test_marathon2_mail_summary_hallucination_gate_blocks_subject_only_next_step() -> None:
    row = {
        "subject_full": "Договор на обучение",
        "full_clean_text": "Здравствуйте, пришлите расписание занятий.",
    }
    payload = {
        "summary": "Клиент просит расписание занятий.",
        "topic": "Договор",
        "next_step": "Подготовить договор на обучение.",
        "extraction_source": "model",
        "event_type": "other",
        "money_direction": "none",
    }

    reasons = _anti_hallucination_reasons(row, payload)

    assert "next_step_not_supported_by_source" in reasons


def test_marathon2_mail_summary_hallucination_gate_allows_supported_next_step() -> None:
    row = {
        "subject_full": "Расписание",
        "full_clean_text": "Клиент просит расписание занятий на выходных.",
    }
    payload = {
        "summary": "Клиент просит расписание занятий на выходных.",
        "topic": "Расписание",
        "next_step": "Отправить расписание занятий на выходных.",
        "extraction_source": "model",
        "event_type": "other",
        "money_direction": "none",
    }

    reasons = _anti_hallucination_reasons(row, payload)

    assert "next_step_not_supported_by_source" not in reasons


def test_marathon2_mail_summary_hallucination_gate_blocks_free_text_payment() -> None:
    row = {
        "subject_full": "Расписание",
        "full_clean_text": "Клиент просит расписание занятий на выходных.",
    }
    payload = {
        "summary": "Клиент оплатил курс и просит расписание.",
        "topic": "Оплата",
        "next_step": None,
        "extraction_source": "model",
        "event_type": "other",
        "money_direction": "none",
    }

    reasons = _anti_hallucination_reasons(row, payload)

    assert "payment_text_not_supported_by_source" in reasons


def test_marathon2_mail_summary_hallucination_gate_blocks_subject_only_payment() -> None:
    row = {
        "subject_full": "Оплата курса",
        "full_clean_text": "Здравствуйте, отправьте расписание занятий.",
    }
    payload = {
        "summary": "Клиент оплатил курс и просит расписание.",
        "topic": "Оплата курса",
        "next_step": None,
        "extraction_source": "model",
        "event_type": "payment",
        "money_direction": "in",
        "amount_kind": "actual_payment",
    }

    reasons = _anti_hallucination_reasons(row, payload)

    assert "payment_text_not_supported_by_source" in reasons
    assert "actual_payment_not_supported_by_source" in reasons


def test_marathon2_mail_summary_hallucination_gate_blocks_new_plain_number() -> None:
    row = {
        "subject_full": "Расписание",
        "full_clean_text": "Клиент просит расписание занятий.",
    }
    payload = {
        "summary": "Клиент просит расписание для 8 класса.",
        "topic": "Расписание",
        "next_step": None,
        "extraction_source": "model",
        "event_type": "other",
        "money_direction": "none",
    }

    reasons = _anti_hallucination_reasons(row, payload)

    assert "new_numeric_token:8" in reasons


def test_marathon2_mail_summary_hallucination_gate_blocks_requisites_and_amount_items() -> None:
    row = {
        "subject_full": "Стоимость",
        "full_clean_text": "Стоимость курса 50 000 руб.",
    }
    payload = {
        "summary": "Стоимость курса 50 000 руб., оплатить по реквизитам банка.",
        "topic": "Стоимость",
        "next_step": None,
        "event_type": "other",
        "money_direction": "none",
        "requisites": ["БИК 044525225"],
        "amount_rub": 50000,
        "amount_kind": "quote",
        "amount_items": [{"amount_rub": 126000, "amount_kind": "quote", "description": "полная цена", "is_total": True}],
    }

    reasons = _anti_hallucination_reasons(row, payload)

    assert "requisites_in_summary_payload" in reasons
    assert "forbidden_requisite_or_document_term" in reasons
    assert "amount_item_not_in_source" in reasons


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
