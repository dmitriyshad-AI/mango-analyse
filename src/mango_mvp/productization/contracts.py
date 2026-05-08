from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Mapping, Optional


class Direction(str, Enum):
    INBOUND = "inbound"
    OUTBOUND = "outbound"
    INTERNAL = "internal"
    UNKNOWN = "unknown"


class OutcomeStatus(str, Enum):
    WON = "won"
    LOST = "lost"
    PENDING = "pending"
    NO_DEAL_FOUND = "no_deal_found"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class TenantRef:
    tenant_id: str
    display_name: str = ""

    def __post_init__(self) -> None:
        tenant_id = self.tenant_id.strip()
        if not tenant_id:
            raise ValueError("tenant_id must not be empty")
        object.__setattr__(self, "tenant_id", tenant_id)
        object.__setattr__(self, "display_name", self.display_name.strip())


@dataclass(frozen=True)
class TelephonyCallEvent:
    tenant: TenantRef
    provider: str
    provider_call_id: str
    started_at: datetime
    ended_at: Optional[datetime]
    direction: Direction = Direction.UNKNOWN
    client_phone: Optional[str] = None
    manager_ref: Optional[str] = None
    recording_ref: Optional[str] = None
    recording_url: Optional[str] = None
    raw_payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        provider = _normalize_key_part(self.provider, "provider")
        provider_call_id = _normalize_id_part(self.provider_call_id, "provider_call_id")
        _require_timezone(self.started_at, "started_at")
        if self.ended_at is not None:
            _require_timezone(self.ended_at, "ended_at")
            if self.ended_at < self.started_at:
                raise ValueError("ended_at must not be earlier than started_at")
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "provider_call_id", provider_call_id)
        object.__setattr__(self, "raw_payload", dict(self.raw_payload))

    @property
    def event_key(self) -> str:
        return stable_event_key(
            tenant_id=self.tenant.tenant_id,
            provider=self.provider,
            provider_call_id=self.provider_call_id,
        )

    @property
    def duration_seconds(self) -> Optional[int]:
        if self.ended_at is None:
            return None
        return int((self.ended_at - self.started_at).total_seconds())


@dataclass(frozen=True)
class RecordingAsset:
    event_key: str
    uri: str
    content_type: Optional[str] = None
    checksum_sha256: Optional[str] = None
    size_bytes: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.event_key.strip():
            raise ValueError("event_key must not be empty")
        if not self.uri.strip():
            raise ValueError("uri must not be empty")


@dataclass(frozen=True)
class CaptureIngestCandidate:
    event_key: str
    tenant_id: str
    provider: str
    provider_call_id: str
    started_at: datetime
    direction: Direction
    audio_ref: Optional[str]
    client_phone: Optional[str] = None
    manager_ref: Optional[str] = None
    raw_payload: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_event(
        cls,
        event: TelephonyCallEvent,
        audio_ref: Optional[str],
    ) -> "CaptureIngestCandidate":
        return cls(
            event_key=event.event_key,
            tenant_id=event.tenant.tenant_id,
            provider=event.provider,
            provider_call_id=event.provider_call_id,
            started_at=event.started_at,
            direction=event.direction,
            audio_ref=audio_ref,
            client_phone=event.client_phone,
            manager_ref=event.manager_ref,
            raw_payload=event.raw_payload,
        )

    def __post_init__(self) -> None:
        _require_timezone(self.started_at, "started_at")
        object.__setattr__(self, "raw_payload", dict(self.raw_payload))


@dataclass(frozen=True)
class CrmContactSnapshot:
    tenant: TenantRef
    provider: str
    provider_contact_id: str
    phone: str
    raw_payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider", _normalize_key_part(self.provider, "provider"))
        object.__setattr__(
            self,
            "provider_contact_id",
            _normalize_id_part(self.provider_contact_id, "provider_contact_id"),
        )
        object.__setattr__(self, "raw_payload", dict(self.raw_payload))


@dataclass(frozen=True)
class CrmOutcomeSnapshot:
    tenant: TenantRef
    provider: str
    provider_deal_id: str
    status: OutcomeStatus
    amount: Optional[float] = None
    closed_at: Optional[datetime] = None
    raw_payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider", _normalize_key_part(self.provider, "provider"))
        object.__setattr__(
            self,
            "provider_deal_id",
            _normalize_id_part(self.provider_deal_id, "provider_deal_id"),
        )
        if self.closed_at is not None:
            _require_timezone(self.closed_at, "closed_at")
        object.__setattr__(self, "raw_payload", dict(self.raw_payload))


def stable_event_key(tenant_id: str, provider: str, provider_call_id: str) -> str:
    return ":".join(
        (
            _normalize_key_part(tenant_id, "tenant_id"),
            _normalize_key_part(provider, "provider"),
            _normalize_id_part(provider_call_id, "provider_call_id"),
        )
    )


def _normalize_key_part(value: str, field_name: str) -> str:
    normalized = value.strip().lower()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


def _normalize_id_part(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


def _require_timezone(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
