from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mango_mvp.productization.mango_live_shadow_poll import (
    build_mango_live_shadow_poll_report,
    event_keys_from_job_result,
    read_seen_event_keys,
)
from tests.test_productization_product_db import bootstrap_sample_product_db


class FakeMangoPollClient:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows
        self.calls = []

    def poll_call_history(self, since: datetime, until: datetime) -> list[dict]:
        self.calls.append((since, until))
        return self.rows


def mango_row(call_id: str, recording: str | None = "rec-1") -> dict:
    row = {
        "entry_id": call_id,
        "start": "1778133600",
        "finish": "1778133900",
        "from_number": "+79990000000",
        "to_extension": "101",
    }
    if recording is not None:
        row["records"] = recording
    return row


def test_live_shadow_poll_reports_live_actions_and_archives_payload(tmp_path: Path) -> None:
    product_root, product_db, _tenant_config = bootstrap_sample_product_db(tmp_path)
    raw_payload = product_root / "raw_payload_archive" / "live" / "poll.jsonl"
    client = FakeMangoPollClient(
        [
            mango_row("CALL-1", "rec-known"),
            mango_row("CALL-4", "rec-new"),
            mango_row("CALL-5", None),
            {"entry_id": "BROKEN"},
        ]
    )

    report = build_mango_live_shadow_poll_report(
        product_db_path=product_db,
        product_root=product_root,
        tenant_id="foton",
        since=datetime(2026, 5, 7, 6, 0, tzinfo=timezone.utc),
        until=datetime(2026, 5, 7, 7, 0, tzinfo=timezone.utc),
        raw_payload_path=raw_payload,
        client=client,
    )

    assert report["summary"]["source_rows"] == 4
    assert report["summary"]["raw_payload_rows"] == 4
    assert report["summary"]["normalized_events"] == 3
    assert report["summary"]["normalization_errors"] == 1
    assert report["summary"]["enqueue_shadow_capture"] == 1
    assert report["summary"]["skip_duplicate"] == 1
    assert report["summary"]["skip_no_recording"] == 1
    assert report["summary"]["validation_ok"] is False
    assert report["validation_ok"] is False
    assert report["action_counts"] == {
        "ENQUEUE_SHADOW_CAPTURE": 1,
        "SKIP_DUPLICATE": 1,
        "SKIP_NO_RECORDING": 1,
    }
    assert raw_payload.exists()
    assert len(raw_payload.read_text(encoding="utf-8").splitlines()) == 4
    assert report["safety"]["download_audio"] is False
    assert report["safety"]["write_runtime_db"] is False
    assert client.calls


def test_live_shadow_poll_can_allow_metadata_only(tmp_path: Path) -> None:
    product_root, product_db, _tenant_config = bootstrap_sample_product_db(tmp_path)
    raw_payload = product_root / "raw_payload_archive" / "live" / "poll.jsonl"
    client = FakeMangoPollClient([mango_row("CALL-5", None)])

    report = build_mango_live_shadow_poll_report(
        product_db_path=product_db,
        product_root=product_root,
        tenant_id="foton",
        since=datetime(2026, 5, 7, 6, 0, tzinfo=timezone.utc),
        until=datetime(2026, 5, 7, 7, 0, tzinfo=timezone.utc),
        raw_payload_path=raw_payload,
        allow_metadata_only=True,
        client=client,
    )

    assert report["summary"]["validation_ok"] is True
    assert report["validation_ok"] is True
    assert report["summary"]["enqueue_shadow_capture"] == 1
    assert report["summary"]["skip_no_recording"] == 0


def test_live_shadow_poll_seen_keys_include_previous_job_results(tmp_path: Path) -> None:
    product_root, product_db, _tenant_config = bootstrap_sample_product_db(tmp_path)
    result = {
        "decisions": [
            {
                "action_code": "ENQUEUE_SHADOW_CAPTURE",
                "event": {"event_key": "foton:mango:CALL-99"},
            },
            {
                "action_code": "SKIP_NO_RECORDING",
                "event": {"event_key": "foton:mango:CALL-100"},
            }
        ]
    }
    with sqlite3.connect(product_db) as con:
        con.execute(
            """
            INSERT INTO job_runs (job_type, tenant_id, status, planned_at, input_ref, result_json)
            VALUES ('shadow_poll', 'foton', 'succeeded', '2026-05-07T00:00:00+00:00', '{}', ?)
            """,
            (json.dumps(result),),
        )
        con.commit()

    seen = read_seen_event_keys(product_db)

    assert "foton:mango:CALL-1" in seen
    assert "foton:mango:CALL-99" in seen
    assert "foton:mango:CALL-100" not in seen
    assert event_keys_from_job_result(json.dumps(result)) == {"foton:mango:CALL-99"}


def test_live_shadow_poll_refuses_raw_archive_outside_product_root(tmp_path: Path) -> None:
    product_root, product_db, _tenant_config = bootstrap_sample_product_db(tmp_path)

    with pytest.raises(ValueError, match="raw payload"):
        build_mango_live_shadow_poll_report(
            product_db_path=product_db,
            product_root=product_root,
            tenant_id="foton",
            since=datetime(2026, 5, 7, 6, 0, tzinfo=timezone.utc),
            until=datetime(2026, 5, 7, 7, 0, tzinfo=timezone.utc),
            raw_payload_path=tmp_path / "outside.jsonl",
            client=FakeMangoPollClient([]),
        )
