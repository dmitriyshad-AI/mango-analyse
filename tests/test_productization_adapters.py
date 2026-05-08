from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Sequence

import pytest

from mango_mvp.productization.adapters import AdapterRegistry
from mango_mvp.productization.contracts import RecordingAsset, TelephonyCallEvent, TenantRef


@dataclass
class FakeTelephonyAdapter:
    provider: str = "Mango"

    def poll_calls(
        self,
        tenant: TenantRef,
        since: datetime,
        until: datetime,
    ) -> Sequence[TelephonyCallEvent]:
        return ()

    def get_recording(self, event: TelephonyCallEvent) -> Optional[RecordingAsset]:
        return None


def test_adapter_registry_normalizes_provider_names() -> None:
    registry = AdapterRegistry()
    adapter = FakeTelephonyAdapter()

    registry.register_telephony(adapter)

    assert registry.telephony("mango") is adapter
    assert registry.telephony(" MANGO ") is adapter
    assert registry.telephony_providers() == ("mango",)


def test_adapter_registry_rejects_duplicate_provider() -> None:
    registry = AdapterRegistry()
    registry.register_telephony(FakeTelephonyAdapter(provider="mango"))

    with pytest.raises(ValueError):
        registry.register_telephony(FakeTelephonyAdapter(provider="MANGO"))


def test_adapter_registry_raises_for_unknown_provider() -> None:
    registry = AdapterRegistry()

    with pytest.raises(KeyError):
        registry.telephony("missing")
