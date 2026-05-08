from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Optional, Sequence

from mango_mvp.productization.contracts import Direction, TelephonyCallEvent, TenantRef


@dataclass(frozen=True)
class MangoOfficeFieldAliases:
    call_id: Sequence[str] = ("call_id", "entry_id", "id")
    started_at: Sequence[str] = ("started_at", "start_time", "timestamp", "date", "start")
    ended_at: Sequence[str] = ("ended_at", "finish_time", "end_time", "finish")
    direction: Sequence[str] = ("direction", "call_direction", "type")
    client_phone: Sequence[str] = ("client_phone", "phone", "ani")
    manager_ref: Sequence[str] = ("manager_ref", "employee_id", "extension", "manager_extension")
    recording_ref: Sequence[str] = ("recording_ref", "recording_id", "record_id", "records")
    recording_url: Sequence[str] = ("recording_url", "record_url", "recording_link")


class MangoOfficePayloadMapper:
    provider = "mango"

    def __init__(self, aliases: Optional[MangoOfficeFieldAliases] = None) -> None:
        self.aliases = aliases or MangoOfficeFieldAliases()

    def from_payload(self, tenant: TenantRef, payload: Mapping[str, Any]) -> TelephonyCallEvent:
        call_id = _first_present(payload, self.aliases.call_id)
        if call_id is None:
            raise ValueError("Mango payload does not contain call id")

        started_at_raw = _first_present(payload, self.aliases.started_at)
        if started_at_raw is None:
            raise ValueError("Mango payload does not contain start time")

        ended_at_raw = _first_present(payload, self.aliases.ended_at)
        direction_raw = _first_present(payload, self.aliases.direction)
        direction = _parse_direction(direction_raw)
        if direction == Direction.UNKNOWN:
            direction = _infer_direction(payload)

        return TelephonyCallEvent(
            tenant=tenant,
            provider=self.provider,
            provider_call_id=str(call_id),
            started_at=_parse_datetime(started_at_raw),
            ended_at=_parse_datetime(ended_at_raw) if ended_at_raw is not None else None,
            direction=direction,
            client_phone=_extract_client_phone(payload, direction, self.aliases),
            manager_ref=_extract_manager_ref(payload, direction, self.aliases),
            recording_ref=_extract_recording_ref(payload, self.aliases),
            recording_url=_optional_str(_first_present(payload, self.aliases.recording_url)),
            raw_payload=payload,
        )

    def from_payloads(
        self,
        tenant: TenantRef,
        payloads: Iterable[Mapping[str, Any]],
    ) -> Sequence[TelephonyCallEvent]:
        return tuple(self.from_payload(tenant=tenant, payload=payload) for payload in payloads)


def _first_present(payload: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in payload and payload[name] not in (None, ""):
            return payload[name]
    return None


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _extract_client_phone(
    payload: Mapping[str, Any],
    direction: Direction,
    aliases: MangoOfficeFieldAliases,
) -> Optional[str]:
    explicit = _optional_str(_first_present(payload, aliases.client_phone))
    if explicit:
        return explicit
    if direction == Direction.INBOUND:
        return _optional_str(payload.get("from_number"))
    if direction == Direction.OUTBOUND:
        return _optional_str(payload.get("to_number"))
    return _optional_str(payload.get("from_number") or payload.get("to_number"))


def _extract_manager_ref(
    payload: Mapping[str, Any],
    direction: Direction,
    aliases: MangoOfficeFieldAliases,
) -> Optional[str]:
    explicit = _optional_str(_first_present(payload, aliases.manager_ref))
    if explicit:
        return explicit
    if direction == Direction.INBOUND:
        return _optional_str(payload.get("to_extension"))
    if direction == Direction.OUTBOUND:
        return _optional_str(payload.get("from_extension"))
    return _optional_str(payload.get("from_extension") or payload.get("to_extension"))


def _extract_recording_ref(
    payload: Mapping[str, Any],
    aliases: MangoOfficeFieldAliases,
) -> Optional[str]:
    value = _first_present(payload, aliases.recording_ref)
    if isinstance(value, list):
        if not value:
            return None
        return _optional_str(value[0])
    if isinstance(value, Mapping):
        for key in ("url", "link", "id", "recording_id", "record_id"):
            if key in value:
                return _optional_str(value[key])
        return None
    if isinstance(value, str):
        text = value.strip()
        if text in {"", "[]"}:
            return None
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1].strip()
            if not text:
                return None
            return text.split(",", 1)[0].strip() or None
        return text
    return _optional_str(value)


def _parse_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, (int, float)):
        parsed = datetime.fromtimestamp(value, tz=timezone.utc)
    else:
        text = str(value).strip()
        if text.isdigit():
            parsed = datetime.fromtimestamp(int(text), tz=timezone.utc)
            return parsed
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        parsed = datetime.fromisoformat(text)

    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _parse_direction(value: Any) -> Direction:
    if value is None:
        return Direction.UNKNOWN
    normalized = str(value).strip().lower()
    inbound = {"in", "inbound", "incoming", "входящий"}
    outbound = {"out", "outbound", "outgoing", "исходящий"}
    internal = {"internal", "inner", "внутренний"}
    if normalized in inbound:
        return Direction.INBOUND
    if normalized in outbound:
        return Direction.OUTBOUND
    if normalized in internal:
        return Direction.INTERNAL
    return Direction.UNKNOWN


def _infer_direction(payload: Mapping[str, Any]) -> Direction:
    from_extension = _optional_str(payload.get("from_extension"))
    to_extension = _optional_str(payload.get("to_extension"))
    from_number = _optional_str(payload.get("from_number"))
    to_number = _optional_str(payload.get("to_number"))

    if from_extension and to_extension and not from_number and not to_number:
        return Direction.INTERNAL
    if to_extension and from_number:
        return Direction.INBOUND
    if from_extension and to_number:
        return Direction.OUTBOUND
    return Direction.UNKNOWN
