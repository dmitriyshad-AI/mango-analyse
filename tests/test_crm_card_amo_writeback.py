from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from mango_mvp.crm_card_amo_writeback import (
    build_crm_card_amo_payloads,
    deal_guard_reasons,
    dry_run_entity,
    field_catalog_guard,
    open_amo_opportunities,
    raw_amo_opportunities_for_customer,
    select_customer_ids_for_amo_dry_run_from_db,
)


def _entity_with_values(values: dict[str, str]) -> dict[str, object]:
    return {
        "id": 123,
        "custom_fields_values": [
            {"field_id": index, "field_name": field, "values": [{"value": value}]}
            for index, (field, value) in enumerate(values.items(), start=1000)
        ],
    }


def test_crm_card_payloads_map_manager_card_to_contact_and_deal_fields() -> None:
    projection = {
        "contact_card": {
            "fields": {
                "Последняя сводка": "Сводка по клиенту",
                "История общения": "Сжатая история",
            }
        },
        "deal_card": {
            "fields": {
                "Следующий шаг": "Позвонить и согласовать оплату",
            }
        },
    }

    contact_payload, deal_payload = build_crm_card_amo_payloads(projection)

    assert contact_payload == {
        "AI-рекомендованный следующий шаг": "Позвонить и согласовать оплату",
        "Последняя AI-сводка": "Сводка по клиенту",
        "Авто история общения": "Сжатая история",
    }
    assert deal_payload == {
        "AI-рекомендованный следующий шаг": "Позвонить и согласовать оплату",
        "AI-сводка по сделке": "Сводка по клиенту",
        "AI-история по сделке": "Сжатая история",
    }


def test_deal_guard_requires_brand_and_open_deal_count_before_write() -> None:
    assert deal_guard_reasons({"deal_brand": "foton", "selected_deal_id": "123"}) == [
        "missing_active_brand",
        "missing_open_deal_count",
    ]


def test_deal_guard_blocks_brand_conflict_and_multiple_open_deals() -> None:
    reasons = deal_guard_reasons(
        {
            "active_brand": "foton",
            "deal_brand": "unpk",
            "open_deal_count": "2",
            "selected_deal_id": "123",
        }
    )

    assert "brand_conflict_channel_deal" in reasons
    assert "multiple_open_deals" in reasons


def test_field_catalog_guard_requires_textarea_target_fields() -> None:
    reasons = field_catalog_guard(
        [{"id": 1, "name": "AI-рекомендованный следующий шаг", "type": "text"}],
        ("AI-рекомендованный следующий шаг", "Авто история общения"),
    )

    assert "amo_field_not_textarea:AI-рекомендованный следующий шаг:text" in reasons
    assert "missing_amo_field:Авто история общения" in reasons


def test_dry_run_entity_saves_snapshot_journal_and_blocks_clobber(tmp_path: Path) -> None:
    input_manifest = tmp_path / "dry_run_input_manifest.json"
    input_manifest.write_text("{}", encoding="utf-8")
    payload = {"Авто история общения": "Новая история"}

    row, findings = dry_run_entity(
        entity_type="contact",
        entity_id="777",
        customer_id="customer:1",
        payload=payload,
        current_entity=_entity_with_values({"Авто история общения": "Старая история"}),
        fresh_entity=_entity_with_values({"Авто история общения": "Ручная правка менеджера"}),
        field_catalog=[{"id": 1000, "name": "Авто история общения", "type": "textarea"}],
        catalog_guard=[],
        row_index=1,
        batch_id="batch",
        input_manifest=input_manifest,
        input_sha256="abc",
        out_dir=tmp_path,
        journal_path=tmp_path / "journal.jsonl",
        guard_input={},
    )

    assert row["status"] == "blocked"
    assert row["pre_patch_status"] == "blocked"
    assert "clobber_protected" in row["reason"]
    assert findings[0]["risk_type"].startswith("clobber_protected")
    assert (tmp_path / "pre_write_snapshot.jsonl").exists()
    journal_lines = (tmp_path / "journal.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(journal_lines[0])["action"] == "clobber_protected"


def test_raw_opportunity_lookup_and_sample_selection_are_read_only(tmp_path: Path) -> None:
    db = tmp_path / "customer_timeline.sqlite"
    con = sqlite3.connect(db)
    con.executescript(
        """
        CREATE TABLE customer_opportunities (
          tenant_id TEXT,
          customer_id TEXT,
          source_system TEXT,
          source_id TEXT,
          closed_at TEXT,
          opened_at TEXT,
          opportunity_id TEXT,
          record_json TEXT
        );
        CREATE TABLE identity_links (
          tenant_id TEXT,
          customer_id TEXT,
          link_type TEXT,
          link_value TEXT,
          match_class TEXT
        );
        """
    )
    con.execute(
        "INSERT INTO customer_opportunities VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "foton",
            "customer:ok",
            "amocrm_snapshot",
            "456",
            "",
            "2026-06-21T10:00:00+00:00",
            "opp1",
            json.dumps(
                {
                    "tenant_id": "foton",
                    "customer_id": "customer:ok",
                    "source_system": "amocrm_snapshot",
                    "source_id": "456",
                    "opened_at": "2026-06-21T10:00:00+00:00",
                    "product_context": {"brand": "foton"},
                }
            ),
        ),
    )
    con.execute(
        "INSERT INTO identity_links VALUES (?, ?, ?, ?, ?)",
        ("foton", "customer:ok", "amo_contact_id", "123", "strong_unique"),
    )
    con.commit()
    con.close()

    selected = select_customer_ids_for_amo_dry_run_from_db(db, tenant_id="foton", sample_size=5)
    raw = raw_amo_opportunities_for_customer(db, tenant_id="foton", customer_id="customer:ok")

    assert selected == ["customer:ok"]
    assert open_amo_opportunities({}, raw_opportunities=raw)[0]["source_id"] == "456"
    assert raw[0]["product_context"]["brand"] == "foton"
