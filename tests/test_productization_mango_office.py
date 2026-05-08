from __future__ import annotations

from datetime import datetime, timezone

from mango_mvp.productization.contracts import Direction, TenantRef
from mango_mvp.productization.mango_office import MangoOfficePayloadMapper


def test_mango_office_payload_mapper_parses_poll_payload() -> None:
    mapper = MangoOfficePayloadMapper()
    tenant = TenantRef("foton")

    event = mapper.from_payload(
        tenant,
        {
            "call_id": "MANGO-100",
            "start_time": "2026-05-07T09:00:00+03:00",
            "end_time": "2026-05-07T09:04:00+03:00",
            "direction": "incoming",
            "from_number": "+79990000000",
            "extension": "101",
            "recording_link": "https://records.example/mango-100.mp3",
        },
    )

    assert event.event_key == "foton:mango:MANGO-100"
    assert event.started_at == datetime(2026, 5, 7, 6, 0, tzinfo=timezone.utc)
    assert event.ended_at == datetime(2026, 5, 7, 6, 4, tzinfo=timezone.utc)
    assert event.direction == Direction.INBOUND
    assert event.client_phone == "+79990000000"
    assert event.manager_ref == "101"
    assert event.recording_url == "https://records.example/mango-100.mp3"


def test_mango_office_payload_mapper_parses_unix_timestamp() -> None:
    mapper = MangoOfficePayloadMapper()
    tenant = TenantRef("foton")

    event = mapper.from_payload(
        tenant,
        {
            "id": "MANGO-101",
            "timestamp": 1778144400,
            "type": "outbound",
            "phone": "+79990000001",
            "recording_id": "rec-101",
        },
    )

    assert event.provider_call_id == "MANGO-101"
    assert event.started_at == datetime.fromtimestamp(1778144400, tz=timezone.utc)
    assert event.direction == Direction.OUTBOUND
    assert event.recording_ref == "rec-101"


def test_mango_office_payload_mapper_treats_empty_records_as_missing() -> None:
    mapper = MangoOfficePayloadMapper()
    tenant = TenantRef("foton")

    event = mapper.from_payload(
        tenant,
        {
            "entry_id": "MANGO-102",
            "start": "1778144400",
            "finish": "1778144460",
            "from_number": "+79990000002",
            "to_extension": "101",
            "records": "[]",
        },
    )

    assert event.direction == Direction.INBOUND
    assert event.client_phone == "+79990000002"
    assert event.manager_ref == "101"
    assert event.recording_ref is None


def test_mango_office_payload_mapper_extracts_record_id_from_brackets() -> None:
    mapper = MangoOfficePayloadMapper()
    tenant = TenantRef("foton")

    event = mapper.from_payload(
        tenant,
        {
            "entry_id": "MANGO-103",
            "start": "1778144400",
            "finish": "1778144460",
            "from_extension": "101",
            "to_number": "+79990000003",
            "records": "[MToxMDA3MjE5OToyNjY5NDEwNTgwMzow]",
        },
    )

    assert event.direction == Direction.OUTBOUND
    assert event.client_phone == "+79990000003"
    assert event.manager_ref == "101"
    assert event.recording_ref == "MToxMDA3MjE5OToyNjY5NDEwNTgwMzow"
