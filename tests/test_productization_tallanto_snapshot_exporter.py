from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from mango_mvp.productization.product_db import initialize_product_db
from mango_mvp.productization.tallanto_snapshot_exporter import collect_candidate_phones, export_tallanto_snapshot

class FakeTallantoClient:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def search_contacts_by_phone(self, phone: str, *, max_records: int = 20):
        self.queries.append(phone)
        return [
            {
                "id": "t-1",
                "name": "Tallanto Contact",
                "phone_mobile": phone,
                "assigned_user_id": "77",
                "assigned_user_name": "Owner",
            }
        ]


def test_tallanto_snapshot_exporter_reads_product_db_phones_and_writes_snapshot(tmp_path: Path) -> None:
    product_root, product_db = _product_db_with_phone(tmp_path)
    out = product_root / "crm_snapshots" / "tallanto_entities.json"
    client = FakeTallantoClient()

    report = export_tallanto_snapshot(product_root, product_db, out, client=client)

    assert report["summary"]["phones_seen"] == 1
    assert report["summary"]["entities_exported"] == 1
    assert report["entities"][0]["entity_id"] == "t-1"
    assert report["safety"]["write_tallanto"] is False
    assert json.loads(out.read_text(encoding="utf-8"))["summary"]["validation_ok"] is True
    assert client.queries == ["+79990000000"]


def test_tallanto_snapshot_exporter_refuses_outside_output(tmp_path: Path) -> None:
    product_root, product_db = _product_db_with_phone(tmp_path)

    with pytest.raises(ValueError, match="Tallanto snapshot output"):
        export_tallanto_snapshot(product_root, product_db, tmp_path / "outside.json", client=FakeTallantoClient())


def test_collect_candidate_phones_returns_normalized_unique_values(tmp_path: Path) -> None:
    _product_root, product_db = _product_db_with_phone(tmp_path)

    assert collect_candidate_phones(product_db, limit=10) == ["+79990000000"]


def _product_db_with_phone(tmp_path: Path) -> tuple[Path, Path]:
    product_root = tmp_path / "product_appliance"
    product_db = product_root / "mango_product_appliance.sqlite"
    initialize_product_db(product_db, product_root)
    now = "2026-05-09T10:00:00+00:00"
    with sqlite3.connect(str(product_db)) as con:
        con.execute(
            """
            INSERT INTO capture_inbox_items (
              tenant_id, provider, event_key, provider_call_id, status,
              started_at, client_phone, first_seen_at, last_seen_at
            ) VALUES (
              'foton', 'mango', 'foton:mango:CALL-1', 'CALL-1', 'ready_for_capture',
              ?, '8 999 000 00 00', ?, ?
            )
            """,
            (now, now, now),
        )
        con.commit()
    return product_root, product_db
