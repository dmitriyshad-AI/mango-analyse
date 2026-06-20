from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from typing import Any, Mapping, Optional

from mango_mvp.utils.phone import normalize_phone


_KEY_RE = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,79}$")
_EMAIL_RE = re.compile(r"(?i)^[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}$")


def stable_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def stable_prefixed_id(prefix: str, payload: Mapping[str, Any], *, length: int = 32) -> str:
    normalized_prefix = normalize_key(prefix, "id prefix")
    return f"{normalized_prefix}:{stable_digest(payload)[:length]}"


def normalize_key(value: Any, field_name: str) -> str:
    normalized = str(value or "").strip().lower()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    if not _KEY_RE.match(normalized):
        raise ValueError(f"{field_name} contains unsupported characters: {value!r}")
    return normalized


def require_text(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} must not be empty")
    return text


def optional_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def require_timezone(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")


def require_confidence(value: Optional[float], field_name: str = "confidence") -> Optional[float]:
    if value is None:
        return None
    confidence = float(value)
    if not 0 <= confidence <= 1:
        raise ValueError(f"{field_name} must be between 0 and 1")
    return confidence


def normalize_email(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text.startswith("mailto:"):
        text = text[7:].strip()
    return text if _EMAIL_RE.match(text) else ""


def normalize_identity_value(link_type: str, value: Any) -> str:
    kind = normalize_key(link_type, "link_type")
    text = require_text(value, "link_value")
    if kind in {"email", "primary_email"}:
        normalized = normalize_email(text)
        if not normalized:
            raise ValueError(f"invalid email identity value: {value!r}")
        return normalized
    if kind in {"phone", "mango_client_phone", "whatsapp_phone", "primary_phone"}:
        normalized_phone = normalize_phone(text)
        if not normalized_phone:
            raise ValueError(f"invalid phone identity value: {value!r}")
        return normalized_phone
    return text


def require_ordered_datetimes(
    start: Optional[datetime],
    end: Optional[datetime],
    *,
    start_name: str,
    end_name: str,
) -> None:
    if start is not None:
        require_timezone(start, start_name)
    if end is not None:
        require_timezone(end, end_name)
    if start is not None and end is not None and end < start:
        raise ValueError(f"{end_name} must be greater than or equal to {start_name}")


def stable_customer_id(
    *,
    tenant_id: str,
    primary_phone: Optional[str] = None,
    primary_email: Optional[str] = None,
    source_ref: Optional[str] = None,
) -> str:
    phone = normalize_identity_value("phone", primary_phone) if primary_phone else None
    email = normalize_identity_value("email", primary_email) if primary_email else None
    ref = optional_text(source_ref)
    if not any((phone, email, ref)):
        raise ValueError("customer_id requires primary_phone, primary_email, source_ref, or explicit id")
    return stable_prefixed_id(
        "customer",
        {
            "tenant_id": normalize_key(tenant_id, "tenant_id"),
            "primary_phone": phone,
            "primary_email": email,
            "source_ref": ref,
        },
    )


def stable_identity_link_id(
    *,
    tenant_id: str,
    link_type: str,
    link_value: str,
    source_system: str,
    source_ref: str,
) -> str:
    return stable_prefixed_id(
        "identity_link",
        {
            "tenant_id": normalize_key(tenant_id, "tenant_id"),
            "link_type": normalize_key(link_type, "link_type"),
            "link_value": normalize_identity_value(link_type, link_value),
            "source_system": normalize_key(source_system, "source_system"),
            "source_ref": require_text(source_ref, "source_ref"),
        },
    )


def stable_opportunity_id(
    *,
    tenant_id: str,
    customer_id: str,
    opportunity_type: str,
    source_system: str,
    source_id: str,
) -> str:
    return stable_prefixed_id(
        "opportunity",
        {
            "tenant_id": normalize_key(tenant_id, "tenant_id"),
            "customer_id": require_text(customer_id, "customer_id"),
            "opportunity_type": normalize_key(opportunity_type, "opportunity_type"),
            "source_system": normalize_key(source_system, "source_system"),
            "source_id": require_text(source_id, "source_id"),
        },
    )


def stable_event_id(
    *,
    tenant_id: str,
    source_system: str,
    source_id: str,
    event_type: str,
) -> str:
    return stable_prefixed_id(
        "timeline_event",
        {
            "tenant_id": normalize_key(tenant_id, "tenant_id"),
            "source_system": normalize_key(source_system, "source_system"),
            "source_id": require_text(source_id, "source_id"),
            "event_type": normalize_key(event_type, "event_type"),
        },
    )


def stable_artifact_id(
    *,
    tenant_id: str,
    event_id: str,
    artifact_type: str,
    path: str,
    sha256: Optional[str] = None,
) -> str:
    return stable_prefixed_id(
        "event_artifact",
        {
            "tenant_id": normalize_key(tenant_id, "tenant_id"),
            "event_id": require_text(event_id, "event_id"),
            "artifact_type": normalize_key(artifact_type, "artifact_type"),
            "path": require_text(path, "path"),
            "sha256": optional_text(sha256),
        },
    )


def stable_signal_id(
    *,
    tenant_id: str,
    customer_id: Optional[str],
    signal_type: str,
    source_event_ids: tuple[str, ...],
    evidence_text: Optional[str] = None,
) -> str:
    return stable_prefixed_id(
        "derived_signal",
        {
            "tenant_id": normalize_key(tenant_id, "tenant_id"),
            "customer_id": optional_text(customer_id),
            "signal_type": normalize_key(signal_type, "signal_type"),
            "source_event_ids": sorted(require_text(item, "source_event_id") for item in source_event_ids),
            "evidence_text": optional_text(evidence_text),
        },
    )


def stable_chunk_id(
    *,
    tenant_id: str,
    customer_id: str,
    chunk_type: str,
    event_id: Optional[str] = None,
    source_ref: Optional[str] = None,
    ordinal: int = 0,
) -> str:
    if ordinal < 0:
        raise ValueError("ordinal must not be negative")
    if not event_id and not source_ref:
        raise ValueError("chunk_id requires event_id or source_ref")
    return stable_prefixed_id(
        "bot_context_chunk",
        {
            "tenant_id": normalize_key(tenant_id, "tenant_id"),
            "customer_id": require_text(customer_id, "customer_id"),
            "chunk_type": normalize_key(chunk_type, "chunk_type"),
            "event_id": optional_text(event_id),
            "source_ref": optional_text(source_ref),
            "ordinal": ordinal,
        },
    )
