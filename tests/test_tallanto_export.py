from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from mango_mvp.amocrm_runtime.deal_dossier import build_deal_dossier
from mango_mvp.amocrm_runtime.phone_context import PhoneContext
from mango_mvp.amocrm_runtime.tallanto_export import (
    discover_tallanto_schema,
    export_module_snapshot,
    export_tallanto_schema_bundle,
)


class FakeTallantoClient:
    def __init__(self):
        self.config = type("Cfg", (), {"base_url": "https://kmipt.tallanto.com"})()

    def list_possible_modules(self):
        return {"Ученики": {"module": "Contact"}}

    def list_possible_fields(self, module):
        return {"id": {"label": f"{module} id"}}

    def list_enum_values(self, options):
        return {option: ["one", "two"] for option in options}

    def iter_entry_list(self, **kwargs):
        return [{"id": "1", "name": kwargs["module"]}]


def test_tallanto_export_writes_bundle_and_snapshot(tmp_path):
    client = FakeTallantoClient()
    bundle = discover_tallanto_schema(client, modules=["Contact"], enum_options=["filial_list"])
    assert bundle["modules"] == ["Contact"]
    written = export_tallanto_schema_bundle(client, output_dir=tmp_path, modules=["Contact"], enum_options=["filial_list"])
    for path in written.values():
        assert Path(path).exists()
    snapshot_path = export_module_snapshot(client, module="Contact", output_path=tmp_path / "contacts.json")
    payload = json.loads(Path(snapshot_path).read_text())
    assert payload["record_count"] == 1
    assert payload["records"][0]["name"] == "Contact"


def test_build_deal_dossier_includes_live_tallanto_context():
    with tempfile.TemporaryDirectory(prefix="mango_tallanto_dossier_") as td:
        phone_context = PhoneContext(
            phone="+79990001122",
            source_dir=td,
            contact_row={
                "Всего звонков в истории": "1",
                "Звонков с полным анализом": "1",
                "Незакрытых звонков в истории": "0",
                "Полная история проанализирована": "Да",
                "Краткая история общения": "Клиент думает.",
                "Хронология общения (последние 5 касаний)": "2026-04-10 — думает",
                "ID Tallanto": "123",
                "Статус матчинга Tallanto": "exact_phone_single",
            },
            call_rows=[],
            call_ids=[],
            first_call_at="2026-04-10 10:00:00",
            last_call_at="2026-04-10 10:00:00",
            manager_history=[],
            interest_summary="",
            objections_summary="",
            current_sales_temperature="warm",
            recommended_next_step="Перезвонить",
            follow_up_due_at="2026-04-20",
            history_summary="Клиент думает.",
            chronology="2026-04-10 — думает",
            tallanto_id="123",
            tallanto_match_status="exact_phone_single",
        )

        with patch(
            "mango_mvp.amocrm_runtime.deal_dossier.build_live_tallanto_context",
            return_value={"enabled": True, "status": "ok", "matched_via": "tallanto_id", "contacts_found": 1},
        ):
            dossier = build_deal_dossier(
                phone_context=phone_context,
                contact={"id": 1, "name": "Ивановы"},
                lead={"id": 10, "name": "Сделка", "pipeline_id": 100, "status_id": 143},
                notes=[],
                tasks=[],
                pipeline_name="Сделки B2C",
                status_name="Закрыто и не реализовано",
                user_map={},
            )

    assert dossier["tallanto_live"]["status"] == "ok"
    assert dossier["tallanto_live"]["matched_via"] == "tallanto_id"
