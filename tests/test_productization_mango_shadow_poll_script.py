from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from scripts import mango_office_shadow_poll


def test_shadow_poll_report_counts_actions_without_runtime_side_effects() -> None:
    args = SimpleNamespace(
        tenant="foton",
        hours=2,
        since="2026-05-07T06:00:00+00:00",
        until="2026-05-07T07:00:00+00:00",
        base_url="https://example.test",
        api_key="key",
        api_salt="salt",
        seen_keys=None,
        allow_metadata_only=False,
        out=None,
    )

    rows = [
        {
            "entry_id": "CALL-1",
            "start": "1778133600",
            "finish": "1778133900",
            "from_number": "+79990000000",
            "to_extension": "101",
            "records": "rec-1",
        },
        {
            "entry_id": "CALL-2",
            "start": "1778134200",
            "finish": "1778134500",
            "from_number": "+79990000001",
            "to_extension": "102",
        },
    ]

    with patch.object(mango_office_shadow_poll.MangoOfficeClient, "poll_call_history", return_value=rows):
        report = mango_office_shadow_poll.build_report(args)

    assert report["counts"] == {
        "source_rows": 2,
        "normalized_events": 2,
        "normalization_errors": 0,
        "enqueue_shadow_capture": 1,
        "skip_duplicate": 0,
        "skip_no_recording": 1,
    }
    assert report["decisions"][0]["action"] == "enqueue_shadow_capture"
    assert report["decisions"][1]["action"] == "skip_no_recording"


def test_shadow_poll_can_archive_raw_payload_rows(tmp_path) -> None:
    raw_out = tmp_path / "raw.jsonl"
    args = SimpleNamespace(
        tenant="foton",
        hours=2,
        since="2026-05-07T06:00:00+00:00",
        until="2026-05-07T07:00:00+00:00",
        base_url="https://example.test",
        api_key="key",
        api_salt="salt",
        seen_keys=None,
        allow_metadata_only=False,
        raw_payload_jsonl=str(raw_out),
        out=None,
    )
    rows = [
        {
            "entry_id": "CALL-1",
            "start": "1778133600",
            "finish": "1778133900",
            "from_number": "+79990000000",
            "to_extension": "101",
            "records": "rec-1",
        }
    ]

    with patch.object(mango_office_shadow_poll.MangoOfficeClient, "poll_call_history", return_value=rows):
        report = mango_office_shadow_poll.build_report(args)

    assert report["raw_payload_archive"] == {"enabled": True, "path": str(raw_out), "rows": 1}
    payload = raw_out.read_text(encoding="utf-8").strip()
    assert "CALL-1" in payload
    assert "raw_payload" in payload
