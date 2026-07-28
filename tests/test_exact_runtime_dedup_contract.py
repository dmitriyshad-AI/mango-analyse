from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pytest

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


def test_channel_brand_normalizers_are_aliases_of_one_owner() -> None:
    from mango_mvp.channels import answer_contract, conversation_intent_plan, draft_prompt_builder
    from mango_mvp.channels import telegram_pilot_context_builder
    from mango_mvp.channels.pilot_context import normalize_active_brand

    aliases = (
        answer_contract._normalize_brand,
        conversation_intent_plan._normalize_brand,
        draft_prompt_builder._normalize_active_brand,
        telegram_pilot_context_builder._normalize_active_brand,
    )
    assert all(alias is normalize_active_brand for alias in aliases)
    assert [normalize_active_brand(value) for value in (None, "", "ФОТОН", "унпк ", "other")] == [
        "unknown", "unknown", "foton", "unpk", "unknown",
    ]


def test_telegram_context_truthy_is_alias_of_one_owner() -> None:
    from mango_mvp.channels import telegram_pilot_context_builder
    from mango_mvp.channels.pilot_context import truthy

    assert telegram_pilot_context_builder._truthy is truthy
    assert [truthy(value) for value in (None, "", 0, 1, False, True, "есть", "on", "нет")] == [
        False, False, False, True, False, True, True, False, False,
    ]


def test_customer_context_int_or_zero_is_alias_of_one_owner() -> None:
    from mango_mvp.channels import customer_context_for_draft
    from mango_mvp.channels.pilot_context import int_or_zero

    assert customer_context_for_draft.int_or_zero is int_or_zero
    huge = 10**400
    assert [int_or_zero(value) for value in (None, "", "12", True, False, huge, float("nan"))] == [
        0, 0, 12, 1, 0, huge, 0,
    ]
    with pytest.raises(OverflowError):
        int_or_zero(float("inf"))
