from __future__ import annotations

from datetime import datetime, timezone

import pytest

from mango_mvp.productization.contracts import (
    Direction,
    TelephonyCallEvent,
    TenantRef,
    stable_event_key,
)


def test_stable_event_key_is_normalized() -> None:
    assert stable_event_key(" Foton ", " Mango ", " CALL-42 ") == "foton:mango:CALL-42"


def test_telephony_call_event_requires_timezone_aware_started_at() -> None:
    with pytest.raises(ValueError):
        TelephonyCallEvent(
            tenant=TenantRef("foton"),
            provider="mango",
            provider_call_id="call-1",
            started_at=datetime(2026, 5, 7, 9, 0),
            ended_at=None,
        )


def test_telephony_call_event_rejects_negative_duration() -> None:
    with pytest.raises(ValueError):
        TelephonyCallEvent(
            tenant=TenantRef("foton"),
            provider="mango",
            provider_call_id="call-1",
            started_at=datetime(2026, 5, 7, 9, 5, tzinfo=timezone.utc),
            ended_at=datetime(2026, 5, 7, 9, 0, tzinfo=timezone.utc),
        )


def test_telephony_call_event_exposes_duration_and_event_key() -> None:
    event = TelephonyCallEvent(
        tenant=TenantRef("foton"),
        provider="mango",
        provider_call_id="CALL-99",
        started_at=datetime(2026, 5, 7, 9, 0, tzinfo=timezone.utc),
        ended_at=datetime(2026, 5, 7, 9, 3, 30, tzinfo=timezone.utc),
        direction=Direction.OUTBOUND,
    )

    assert event.event_key == "foton:mango:CALL-99"
    assert event.duration_seconds == 210


def test_tenant_id_must_not_be_empty() -> None:
    with pytest.raises(ValueError):
        TenantRef(" ")
