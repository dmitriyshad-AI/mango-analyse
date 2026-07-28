from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from mango_mvp.amocrm_runtime import amo_integration
from mango_mvp.channels import night_funnel_shadow


def test_amo_contact_helpers_delegate_to_one_implementation(monkeypatch) -> None:
    monkeypatch.setattr(amo_integration, "_contact_update_endpoint", lambda base, contact_id: f"{base}/{contact_id}")
    assert amo_integration._contact_entity_endpoint("https://example.test", 17) == "https://example.test/17"

    marker = {"canonical": True}
    monkeypatch.setattr(amo_integration, "_flatten_contact_field_item", lambda item: marker)
    assert amo_integration._flatten_lead_field_item({"id": 17}) is marker


def test_night_funnel_jsonl_writers_delegate_to_one_implementation(monkeypatch) -> None:
    calls: list[tuple[Path, Mapping[str, Any]]] = []

    def canonical_writer(path: Path, record: Mapping[str, Any]) -> Path:
        calls.append((path, record))
        return path

    monkeypatch.setattr(night_funnel_shadow, "append_shadow_log", canonical_writer)
    lead_path = Path("lead.jsonl")
    tee_path = Path("tee.jsonl")
    lead = {"lead": 1}
    tee = {"tee": 2}

    assert night_funnel_shadow.append_lead_card(lead_path, lead) == lead_path
    assert night_funnel_shadow.append_inbound_tee_record(tee_path, tee) == tee_path
    assert calls == [(lead_path, lead), (tee_path, tee)]
