from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Mapping

import pytest

from mango_mvp.productization.mango_office_client import (
    MangoOfficeApiError,
    MangoOfficeClient,
    MangoOfficeCredentials,
    build_stats_request_payload,
    extract_stats_rows,
    parse_stats_csv,
)


class FakeResponse:
    def __init__(self, status_code: int, payload: Any, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text or json.dumps(payload)

    def json(self) -> Any:
        return self._payload


class FakeSession:
    def __init__(self) -> None:
        self.calls = []

    def post(self, url: str, data: Mapping[str, str], timeout: int) -> FakeResponse:
        self.calls.append({"url": url, "data": dict(data), "timeout": timeout})
        if url.endswith("/vpbx/stats/request"):
            return FakeResponse(200, {"request_id": "REQ-1"})
        if url.endswith("/vpbx/stats/result"):
            return FakeResponse(200, {"data": [{"entry_id": "CALL-1", "start": "1778144400"}]})
        raise AssertionError(f"Unexpected URL: {url}")


def test_build_signed_form_uses_mango_sha256_formula() -> None:
    client = MangoOfficeClient(
        credentials=MangoOfficeCredentials(api_key="key", api_salt="salt"),
        session=FakeSession(),
    )
    payload = {"date_from": "1", "date_to": "2"}

    signed = client.build_signed_form(payload)

    expected_json = '{"date_from":"1","date_to":"2"}'
    expected_sign = hashlib.sha256(f"key{expected_json}salt".encode("utf-8")).hexdigest()
    assert signed.json_body == expected_json
    assert signed.sign == expected_sign
    assert signed.form_data == {
        "vpbx_api_key": "key",
        "sign": expected_sign,
        "json": expected_json,
    }


def test_build_stats_request_payload_uses_unix_seconds() -> None:
    payload = build_stats_request_payload(
        since=datetime(2026, 5, 7, 6, 0, tzinfo=timezone.utc),
        until=datetime(2026, 5, 7, 7, 0, tzinfo=timezone.utc),
    )

    assert payload["date_from"] == "1778133600"
    assert payload["date_to"] == "1778137200"
    assert payload["from"] == {"extension": "", "number": ""}
    assert payload["to"] == {"extension": "", "number": ""}
    assert "entry_id" in payload["fields"]


def test_poll_call_history_performs_request_then_result() -> None:
    session = FakeSession()
    client = MangoOfficeClient(
        credentials=MangoOfficeCredentials(api_key="key", api_salt="salt"),
        base_url="https://example.test",
        session=session,
    )

    rows = client.poll_call_history(
        since=datetime(2026, 5, 7, 6, 0, tzinfo=timezone.utc),
        until=datetime(2026, 5, 7, 7, 0, tzinfo=timezone.utc),
    )

    assert rows == ({"entry_id": "CALL-1", "start": "1778144400"},)
    assert [call["url"] for call in session.calls] == [
        "https://example.test/vpbx/stats/request",
        "https://example.test/vpbx/stats/result",
    ]
    second_payload = json.loads(session.calls[1]["data"]["json"])
    assert second_payload == {"request_id": "REQ-1"}


def test_extract_stats_rows_accepts_common_result_shapes() -> None:
    assert extract_stats_rows([{"entry_id": "1"}]) == ({"entry_id": "1"},)
    assert extract_stats_rows({"result": [{"entry_id": "2"}]}) == ({"entry_id": "2"},)
    assert extract_stats_rows({"entry_id": "3", "start": 1}) == ({"entry_id": "3", "start": 1},)


def test_extract_stats_rows_accepts_mango_csv_result() -> None:
    rows = extract_stats_rows(
        "[];1778150907;1778150929;0;202;sip:user4@example;;79037901748;"
        "1121;74951508151;abonent;MjY2OTQwODU4NjA=\r\n"
    )

    assert rows == (
        {
            "records": "[]",
            "start": "1778150907",
            "finish": "1778150929",
            "answer": "0",
            "from_extension": "202",
            "from_number": "sip:user4@example",
            "to_extension": "",
            "to_number": "79037901748",
            "disconnect_reason": "1121",
            "line_number": "74951508151",
            "location": "abonent",
            "entry_id": "MjY2OTQwODU4NjA=",
        },
    )


def test_parse_stats_csv_rejects_unexpected_field_count() -> None:
    with pytest.raises(MangoOfficeApiError):
        parse_stats_csv("a;b;c", ("a", "b"))


def test_extract_stats_rows_rejects_unknown_shape() -> None:
    with pytest.raises(MangoOfficeApiError):
        extract_stats_rows({"status": "pending"})


def test_post_command_raises_on_http_error() -> None:
    class ErrorSession:
        def post(self, url: str, data: Mapping[str, str], timeout: int) -> FakeResponse:
            return FakeResponse(500, {"error": "boom"}, text="boom")

    client = MangoOfficeClient(
        credentials=MangoOfficeCredentials(api_key="key", api_salt="salt"),
        session=ErrorSession(),
    )

    with pytest.raises(MangoOfficeApiError):
        client.post_command("/vpbx/stats/request", {"x": "y"})
