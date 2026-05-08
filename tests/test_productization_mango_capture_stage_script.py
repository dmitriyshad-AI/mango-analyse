from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import mango_office_capture_stage


def test_event_from_report_row_maps_missing_report_shape() -> None:
    tenant = mango_office_capture_stage.TenantRef("foton")
    event = mango_office_capture_stage.event_from_report_row(
        tenant,
        {
            "provider_call_id": "CALL-1",
            "recording_ref": "rec-1",
            "started_at_utc": "2026-05-07T06:00:00+00:00",
            "duration_seconds": 60,
            "direction": "outbound",
            "client_phone": "+79990000000",
            "manager_ref": "101",
        },
    )

    assert event.event_key == "foton:mango:CALL-1"
    assert event.recording_ref == "rec-1"
    assert event.started_at == datetime(2026, 5, 7, 6, 0, tzinfo=timezone.utc)
    assert event.ended_at == datetime(2026, 5, 7, 6, 1, tzinfo=timezone.utc)
    assert event.direction.value == "outbound"


def test_run_capture_stage_can_dry_run_from_report(tmp_path: Path) -> None:
    report = tmp_path / "missing.json"
    report.write_text(
        json.dumps(
            {
                "missing": [
                    {
                        "provider_call_id": "CALL-1",
                        "recording_ref": "rec-1",
                        "started_at_utc": "2026-05-07T06:00:00+00:00",
                        "duration_seconds": 60,
                        "direction": "outbound",
                        "client_phone": "+79990000000",
                        "manager_ref": "101",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        tenant="foton",
        from_report=str(report),
        hours=2,
        since=None,
        until=None,
        out_root=str(tmp_path / "stage"),
        out_dir=None,
        manifest=None,
        audit_out=None,
        base_url="https://example.test",
        api_key=None,
        api_salt=None,
        limit=None,
        dry_run=True,
        sleep_sec=0,
        link_retries=0,
        rate_limit_sleep_sec=0,
        timeout_sec=1,
    )

    summary = mango_office_capture_stage.run_capture_stage(args)

    assert summary["stage"]["dry_run_download"] == 1
    assert summary["audit"]["manifest_rows"] == 1
    assert Path(summary["audit_path"]).exists()


def test_run_capture_stage_poll_mode_uses_mapper_and_stage_layer(tmp_path: Path) -> None:
    args = SimpleNamespace(
        tenant="foton",
        from_report=None,
        hours=2,
        since="2026-05-07T06:00:00+00:00",
        until="2026-05-07T07:00:00+00:00",
        out_root=str(tmp_path / "stage"),
        out_dir=None,
        manifest=None,
        audit_out=None,
        base_url="https://example.test",
        api_key="key",
        api_salt="salt",
        limit=None,
        dry_run=True,
        sleep_sec=0,
        link_retries=0,
        rate_limit_sleep_sec=0,
        timeout_sec=1,
    )
    rows = [
        {
            "entry_id": "CALL-1",
            "start": "1778133600",
            "finish": "1778133660",
            "from_number": "+79990000000",
            "to_extension": "101",
            "records": "[rec-1]",
        }
    ]

    with patch.object(mango_office_capture_stage.MangoOfficeClient, "poll_call_history", return_value=rows):
        summary = mango_office_capture_stage.run_capture_stage(args)

    assert summary["stage"]["dry_run_download"] == 1
    assert summary["audit"]["latest_status_counts"] == {"dry_run_download": 1}
