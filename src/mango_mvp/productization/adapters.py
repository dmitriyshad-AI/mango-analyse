from __future__ import annotations

from datetime import datetime
from typing import Dict, Iterable, Optional, Protocol, Sequence

from mango_mvp.productization.contracts import (
    CrmContactSnapshot,
    CrmOutcomeSnapshot,
    RecordingAsset,
    TelephonyCallEvent,
    TenantRef,
)


class TelephonyAdapter(Protocol):
    provider: str

    def poll_calls(
        self,
        tenant: TenantRef,
        since: datetime,
        until: datetime,
    ) -> Sequence[TelephonyCallEvent]:
        """Return normalized call events without triggering downstream processing."""

    def get_recording(self, event: TelephonyCallEvent) -> Optional[RecordingAsset]:
        """Resolve a recording reference or storage URI for a normalized event."""


class CrmAdapter(Protocol):
    provider: str

    def find_contact_by_phone(
        self,
        tenant: TenantRef,
        phone: str,
    ) -> Optional[CrmContactSnapshot]:
        """Read contact context only. Writeback is intentionally out of this contract."""

    def find_outcomes_for_call(
        self,
        event: TelephonyCallEvent,
    ) -> Iterable[CrmOutcomeSnapshot]:
        """Return candidate deal/outcome links for insight analytics."""


class AdapterRegistry:
    def __init__(self) -> None:
        self._telephony: Dict[str, TelephonyAdapter] = {}
        self._crm: Dict[str, CrmAdapter] = {}

    def register_telephony(self, adapter: TelephonyAdapter) -> None:
        provider = _normalize_provider(adapter.provider)
        if provider in self._telephony:
            raise ValueError(f"Telephony adapter already registered: {provider}")
        self._telephony[provider] = adapter

    def register_crm(self, adapter: CrmAdapter) -> None:
        provider = _normalize_provider(adapter.provider)
        if provider in self._crm:
            raise ValueError(f"CRM adapter already registered: {provider}")
        self._crm[provider] = adapter

    def telephony(self, provider: str) -> TelephonyAdapter:
        key = _normalize_provider(provider)
        try:
            return self._telephony[key]
        except KeyError as exc:
            raise KeyError(f"Unknown telephony adapter: {key}") from exc

    def crm(self, provider: str) -> CrmAdapter:
        key = _normalize_provider(provider)
        try:
            return self._crm[key]
        except KeyError as exc:
            raise KeyError(f"Unknown CRM adapter: {key}") from exc

    def telephony_providers(self) -> Sequence[str]:
        return tuple(sorted(self._telephony))

    def crm_providers(self) -> Sequence[str]:
        return tuple(sorted(self._crm))


def _normalize_provider(provider: str) -> str:
    normalized = provider.strip().lower()
    if not normalized:
        raise ValueError("Provider name must not be empty")
    return normalized
