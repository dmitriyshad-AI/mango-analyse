from __future__ import annotations

import hashlib
import json
import os
import plistlib
import signal
import shlex
import subprocess
import sqlite3
import sys
import time
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Mapping, Sequence
from zoneinfo import ZoneInfo

import pytest

from mango_mvp import cli as mango_cli
import mango_mvp.customer_timeline.calls_two_processes as calls_runtime
from mango_mvp.config import get_settings
from mango_mvp.customer_timeline.calls_two_processes import (
    CallsTwoProcessesConfig,
    LockBusy,
    SEQUENTIAL_PIPELINE_STAGES,
    assert_no_pdn,
    call_db_has_open_work,
    call_event_source_systems,
    command_path,
    capture_mango_window,
    codex_network_available,
    dead_letter_total,
    dead_letter_mass_failure,
    environment_preflight,
    ensure_codex_runtime_anchor,
    finalize_report,
    hardlink_or_copy,
    module_probe_command,
    missing_capture_recovery_events,
    new_calls_run_id,
    normalize_unambiguous_legacy_asr_topologies,
    prepare_ingest_inputs,
    prepare_codex_home,
    process_lease,
    pipeline_stages,
    publish_ready_db,
    publish_ready_db_if_changed,
    parent_lifeline_subprocess_command,
    read_fully_ready_call_ids,
    read_json,
    read_known_processed_ids,
    run_capture,
    run_sequential_pipeline_workers,
    run_pipeline,
    run_increment_producer,
    run_local_watchdog,
    run_process_a,
    run_process_b,
    run_cycle,
    run_command,
    run_controlled_local_previews,
    compact_command_reports,
    pipeline_freshness,
    safe_daily_payload,
    sha256_file,
    sqlite_check,
    stage_timeout_deadline,
    terminate_process_group,
    worker_command,
    write_json,
)
from mango_mvp.customer_timeline.store import CustomerTimelineSQLiteStore
from mango_mvp.productization.contracts import Direction, TelephonyCallEvent, TenantRef
from mango_mvp.services.controlled_call_scope import (
    ControlledCallScope,
    ControlledCaptureRequest,
)
from mango_mvp.productization.ready_publication import (
    commit_ready_generation,
    inspect_ready_publication,
    recover_ready_generation,
)
from mango_mvp.productization.owner_only_io import read_stable_regular_bytes


def config_for(tmp_path: Path, *, timeline_name: str = "customer_timeline_staging.sqlite") -> CallsTwoProcessesConfig:
    allowed = tmp_path / "staging"
    allowed.mkdir(parents=True, exist_ok=True)
    return CallsTwoProcessesConfig(
        pipeline_root=tmp_path / "pipeline",
        timeline_db=allowed / timeline_name,
        timeline_allowed_root=allowed,
        python_executable=Path(sys.executable),
        codex_binary=Path(sys.executable),
        codex_home_root=tmp_path / "codex_home",
        expected_active_host_id="m1-host",
        min_free_gib=1,
    )


_REAL_POLL_MANGO_OFFICIAL_LIST_PAGES = (
    calls_runtime.poll_mango_official_list_pages
)


@pytest.fixture(autouse=True)
def _synthetic_official_mango_list(monkeypatch: pytest.MonkeyPatch) -> None:
    def prove(
        _client: object,
        *,
        since: datetime,
        until: datetime,
        expected_call_ids: Sequence[str],
    ) -> Mapping[str, object]:
        call_ids = sorted(expected_call_ids)
        pages = []
        if call_ids:
            pages.append(
                {
                    "offset": 0,
                    "rows": len(call_ids),
                    "entry_ids": call_ids,
                    "entry_ids_sha256": calls_runtime._canonical_json_sha256(call_ids),
                    "buckets": [
                        {
                            "period": "2026-08-12",
                            "declared_total_calls_count": len(call_ids),
                            "rows": len(call_ids),
                            "entry_ids": call_ids,
                            "entry_ids_sha256": calls_runtime._canonical_json_sha256(call_ids),
                        }
                    ],
                    "status": "complete",
                }
            )
        pages.append(
            {
                "offset": len(call_ids),
                "rows": 0,
                "entry_ids": [],
                "entry_ids_sha256": calls_runtime._canonical_json_sha256([]),
                "buckets": [],
                "status": "complete",
            }
        )
        return calls_runtime.build_mango_official_list_proof(
            call_ids=call_ids,
            page_receipts=pages,
            since=since,
            until=until,
        )

    monkeypatch.setattr(calls_runtime, "poll_mango_official_list_pages", prove)


def create_empty_capture_manifest(config: CallsTwoProcessesConfig) -> None:
    from mango_mvp.productization.capture_staging import CaptureManifestStore

    CaptureManifestStore(config.capture_manifest).ensure_exists()


def test_official_mango_list_uses_documented_dates_and_ignores_daily_total() -> None:
    requests: list[Mapping[str, object]] = []
    sleeps: list[float] = []

    class Client:
        stats_result_poll_attempts = 3
        stats_result_poll_interval_sec = 0.25
        sleeper = sleeps.append

        def __init__(self) -> None:
            self.offset = 0
            self.result_calls = 0

        def post_command(
            self, path: str, payload: Mapping[str, object]
        ) -> Mapping[str, object]:
            if path.endswith("/request"):
                requests.append(dict(payload))
                self.offset = int(payload["offset"])
                self.result_calls = 0
                return {"key": f"page-{self.offset}"}
            self.result_calls += 1
            if self.offset == 0 and self.result_calls == 1:
                return {"result": 1000, "status": "work"}
            ids = ["call-a"] if self.offset == 0 else []
            return {
                "result": 1000,
                "status": "complete",
                "data": [
                    {
                        "period": "2026-08-12",
                        "total_calls_count": 582,
                        "list": [{"entry_id": value} for value in ids],
                    }
                ],
            }

    proof = _REAL_POLL_MANGO_OFFICIAL_LIST_PAGES(
        Client(),
        since=datetime(2026, 8, 12, 10, 0, 0, 999999, tzinfo=timezone.utc),
        until=datetime(2026, 8, 12, 11, tzinfo=timezone.utc),
        expected_call_ids=["call-a"],
    )

    assert proof["complete"] is True
    assert proof["observed_count"] == 1
    assert proof["terminal_empty_page"] is True
    assert [page["offset"] for page in proof["pages"]] == [0, 1]
    assert [request["offset"] for request in requests] == [0, 1]
    assert requests[0]["limit"] == 5000
    assert requests[0]["start_date"] == "12.08.2026 13:00:00"
    assert requests[0]["end_date"] == "12.08.2026 14:00:00"
    assert "T" not in str(requests[0]["start_date"])
    assert proof["pages"][0]["buckets"][0][
        "declared_total_calls_count"
    ] == 582
    assert sleeps == [0.25]


def test_official_mango_list_accepts_proven_empty_window() -> None:
    class Client:
        stats_result_poll_attempts = 1
        stats_result_poll_interval_sec = 0
        sleeper = staticmethod(lambda _seconds: None)

        def post_command(
            self, path: str, payload: Mapping[str, object]
        ) -> Mapping[str, object]:
            if path.endswith("/request"):
                return {"key": "empty"}
            return {
                "result": 1000,
                "status": "complete",
                "data": [
                    {
                        "period": "2026-08-12",
                        "total_calls_count": 0,
                        "list": [],
                    }
                ],
            }

    proof = _REAL_POLL_MANGO_OFFICIAL_LIST_PAGES(
        Client(),
        since=datetime(2026, 8, 12, 10, tzinfo=timezone.utc),
        until=datetime(2026, 8, 12, 11, tzinfo=timezone.utc),
        expected_call_ids=[],
    )
    assert proof["complete"] is True
    assert proof["observed_count"] == 0
    assert proof["pages"][0]["rows"] == 0


def test_official_mango_extended_datetime_rejects_naive_boundary() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        calls_runtime._mango_extended_wire_datetime(datetime(2026, 8, 12, 10))


def test_official_mango_extended_datetime_uses_moscow_day_boundary() -> None:
    assert calls_runtime._mango_extended_wire_datetime(
        datetime(2026, 8, 12, 21, tzinfo=timezone.utc)
    ) == "13.08.2026 00:00:00"


def test_official_mango_list_accepts_documented_multiblock_shape() -> None:
    call_ids = [f"call-{index}" for index in range(5)]

    class Client:
        stats_result_poll_attempts = 1
        stats_result_poll_interval_sec = 0
        sleeper = staticmethod(lambda _seconds: None)

        def __init__(self) -> None:
            self.offset = 0

        def post_command(
            self, path: str, payload: Mapping[str, object]
        ) -> Mapping[str, object]:
            if path.endswith("/request"):
                self.offset = int(payload["offset"])
                return {"key": "documented-shape"}
            if self.offset:
                return {"result": 1000, "status": "complete", "data": []}
            return {
                "result": 1000,
                "status": "complete",
                "data": [
                    {
                        "period": "2026-08-12",
                        "total_calls_count": 5,
                        "list": [
                            {"entry_id": value} for value in call_ids[:3]
                        ],
                    },
                    {
                        "period": "2026-08-11",
                        "list": [
                            {"entry_id": value} for value in call_ids[3:]
                        ],
                    },
                ],
            }

    proof = _REAL_POLL_MANGO_OFFICIAL_LIST_PAGES(
        Client(),
        since=datetime(2026, 8, 11, tzinfo=timezone.utc),
        until=datetime(2026, 8, 13, tzinfo=timezone.utc),
        expected_call_ids=call_ids,
    )
    assert proof["complete"] is True
    assert proof["observed_count"] == 5
    assert proof["pages"][0]["rows"] == 5
    assert proof["pages"][1]["rows"] == 0


@pytest.mark.parametrize("total", [5000, 5001])
def test_official_mango_list_pages_proves_limit_boundary(total: int) -> None:
    call_ids = [f"call-{index:05d}" for index in range(total)]

    class Client:
        stats_result_poll_attempts = 1
        stats_result_poll_interval_sec = 0
        sleeper = staticmethod(lambda _seconds: None)

        def __init__(self) -> None:
            self.offset = 0
            self.limit = 0

        def post_command(
            self, path: str, payload: Mapping[str, object]
        ) -> Mapping[str, object]:
            if path.endswith("/request"):
                self.offset = int(payload["offset"])
                self.limit = int(payload["limit"])
                return {"key": f"page-{self.offset}"}
            page = call_ids[self.offset : self.offset + self.limit]
            return {
                "result": 1000,
                "status": "complete",
                "data": [{
                    "total_calls_count": total,
                    "list": [{"entry_id": value} for value in page],
                }],
            }

    proof = _REAL_POLL_MANGO_OFFICIAL_LIST_PAGES(
        Client(),
        since=datetime(2026, 8, 12, 10, tzinfo=timezone.utc),
        until=datetime(2026, 8, 12, 11, tzinfo=timezone.utc),
        expected_call_ids=call_ids,
    )
    assert proof["observed_count"] == total
    assert [page["rows"] for page in proof["pages"]] == (
        [5000, 0] if total == 5000 else [5000, 1, 0]
    )


@pytest.mark.parametrize(
    "mode", ["overlap", "missing_id", "malformed_total", "duplicate", "float_result"]
)
def test_official_mango_list_pages_fail_closed_on_bad_pages(mode: str) -> None:
    class Client:
        stats_result_poll_attempts = 1
        stats_result_poll_interval_sec = 0
        sleeper = staticmethod(lambda _seconds: None)

        def __init__(self) -> None:
            self.offset = 0

        def post_command(
            self, path: str, payload: Mapping[str, object]
        ) -> Mapping[str, object]:
            if path.endswith("/request"):
                self.offset = int(payload["offset"])
                return {"key": f"page-{self.offset}"}
            if mode == "float_result":
                return {"result": 1000.0, "status": "complete", "data": []}
            if mode == "malformed_total":
                ids, total = ["call-a"], False
            elif mode == "duplicate":
                ids, total = ["call-a", "call-a"], 3
            elif self.offset == 0:
                ids, total = ["call-a", "call-b"], 3
            else:
                ids = ["call-b"] if mode == "overlap" else ["call-c"]
                total = 3
            return {
                "result": 1000,
                "status": "complete",
                "data": [{"period": "2026-08-12", "total_calls_count": total, "list": [
                    {"entry_id": value} for value in ids
                ]}],
            }

    expected = ["call-a", "call-b", "call-d"] if mode == "missing_id" else [
        "call-a", "call-b", "call-c"
    ]
    with pytest.raises(RuntimeError):
        _REAL_POLL_MANGO_OFFICIAL_LIST_PAGES(
            Client(),
            since=datetime(2026, 8, 12, 10, tzinfo=timezone.utc),
            until=datetime(2026, 8, 12, 11, tzinfo=timezone.utc),
            expected_call_ids=expected,
        )


def test_official_mango_list_keeps_changed_daily_total_as_evidence() -> None:
    class Client:
        stats_result_poll_attempts = 1
        stats_result_poll_interval_sec = 0
        sleeper = staticmethod(lambda _seconds: None)

        def __init__(self) -> None:
            self.offset = 0

        def post_command(
            self, path: str, payload: Mapping[str, object]
        ) -> Mapping[str, object]:
            if path.endswith("/request"):
                self.offset = int(payload["offset"])
                return {"key": f"page-{self.offset}"}
            total = 582 if self.offset == 0 else 583
            ids = ["call-a"] if self.offset == 0 else []
            return {
                "result": 1000,
                "status": "complete",
                "data": [
                    {
                        "period": "2026-08-12",
                        "total_calls_count": total,
                        "list": [{"entry_id": value} for value in ids],
                    }
                ],
            }

    proof = _REAL_POLL_MANGO_OFFICIAL_LIST_PAGES(
        Client(),
        since=datetime(2026, 8, 12, 10, tzinfo=timezone.utc),
        until=datetime(2026, 8, 12, 11, tzinfo=timezone.utc),
        expected_call_ids=["call-a"],
    )

    assert proof["complete"] is True
    assert [
        page["buckets"][0]["declared_total_calls_count"]
        for page in proof["pages"]
    ] == [582, 583]


def test_basic_stats_silent_cap_is_blocked_by_official_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = replace(config_for(tmp_path), strict_ready_provenance=True)
    rows = [
        {
            "entry_id": call_id,
            "start": str(
                int(
                    datetime(
                        2026, 8, 12, 10, index, tzinfo=timezone.utc
                    ).timestamp()
                )
            ),
            "finish": str(
                int(
                    datetime(
                        2026, 8, 12, 10, index, 30, tzinfo=timezone.utc
                    ).timestamp()
                )
            ),
            "records": "[]",
        }
        for index, call_id in enumerate(("call-a", "call-b"))
    ]

    class CappedClient:
        def __init__(self, **_: object) -> None:
            pass

        def poll_call_history(
            self, *, since: datetime, until: datetime
        ) -> list[Mapping[str, object]]:
            return [
                row
                for row in rows
                if since
                <= calls_runtime.parse_datetime(str(row["start"]))
                <= until
            ]

    monkeypatch.setattr(calls_runtime, "MangoOfficeClient", CappedClient)
    monkeypatch.setattr(calls_runtime, "MangoRecordingDownloader", CappedClient)
    monkeypatch.setattr(
        calls_runtime,
        "poll_mango_official_list_pages",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("official list has call-c")
        ),
    )
    monkeypatch.setenv("MANGO_OFFICE_API_KEY", "present")
    monkeypatch.setenv("MANGO_OFFICE_API_SALT", "present")

    report = capture_mango_window(
        config,
        datetime(2026, 8, 12, 10, tzinfo=timezone.utc),
        datetime(2026, 8, 12, 11, tzinfo=timezone.utc),
    )

    assert report["status"] == "failed"
    assert report["reason"] == "official_mango_list_verification_failed"
    assert not config.capture_manifest.exists()
    assert not config.cursor_path.exists()


def with_dual_enumeration(
    evidence: Mapping[str, object],
) -> dict[str, object]:
    """Upgrade a compact synthetic strict fixture to the dual API contract."""

    result = json.loads(json.dumps(evidence))
    source = result["mango_enumeration_source"]
    assert isinstance(source, dict)
    original_intervals = list(source["covered_intervals"])
    rolling = [
        dict(interval)
        for interval in original_intervals
        if interval.get("scope") == "rolling_authority"
    ]
    auxiliary = [
        dict(interval)
        for interval in original_intervals
        if interval.get("scope") == "recovery_auxiliary"
    ]
    call_keys = list(result.get("call_keys") or [])
    calls_by_day = {
        key: list(value)
        for key, value in dict(result.get("calls_by_moscow_day") or {}).items()
    }
    raw_rows = sum(int(interval.get("rows") or 0) for interval in rolling)
    multiset = sorted(call_keys)
    while len(multiset) < raw_rows:
        multiset.append(f"__synthetic_unmapped_row_{len(multiset)}")
    multiset.sort()
    chunks = [
        {
            "since": interval["since"],
            "until": interval["until"],
            "result_complete": True,
            "rows": int(interval.get("rows") or 0),
        }
        for interval in rolling
    ]
    window_start = datetime.fromisoformat(str(source["rolling_since"]))
    window_end = datetime.fromisoformat(str(source["until"]))
    split = window_start + (window_end - window_start) / 3
    verification_chunks = [
        {
            "since": window_start.isoformat(),
            "until": split.isoformat(),
            "result_complete": True,
            "rows": raw_rows,
        },
        {
            "since": split.isoformat(),
            "until": window_end.isoformat(),
            "result_complete": True,
            "rows": 0,
        },
    ]

    def digest(value: object) -> str:
        return hashlib.sha256(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    pass_payload = {
        "rolling_since": source["rolling_since"],
        "until": source["until"],
        "requests": len(chunks),
        "raw_rows": raw_rows,
        "chunks": chunks,
        "partition_sha256": digest(
            [{"since": item["since"], "until": item["until"]} for item in chunks]
        ),
        "recordable_unique_rows": len(call_keys),
        "without_recording_rows": 0,
        "proven_duplicate_rows": raw_rows - len(call_keys),
        "quarantined_rows": 0,
        "error_rows": 0,
        "unexplained_rows": 0,
        "raw_balance_ok": True,
        "call_key_multiset": multiset,
        "call_key_multiset_sha256": digest(multiset),
        "raw_rows_sha256": digest({"synthetic_rows": multiset}),
        "call_keys": call_keys,
        "normalized_unique_count": len(call_keys),
        "call_keys_sha256": digest(call_keys),
        "calls_by_moscow_day": calls_by_day,
        "calls_by_moscow_day_sha256": digest(calls_by_day),
        "event_digest_sha256": digest(
            {"call_keys": call_keys, "calls_by_day": calls_by_day}
        ),
    }
    comparison = {
        "normalized_unique_count_equal": True,
        "call_keys_equal": True,
        "call_keys_sha256_equal": True,
        "calls_by_moscow_day_equal": True,
        "calls_by_moscow_day_sha256_equal": True,
        "event_digest_sha256_equal": True,
        "primary_raw_balance_ok": True,
        "verification_raw_balance_ok": True,
        "partition_sha256_different": True,
        "official_list_equal": True,
    }
    official_pages = [
        *(
            {
                "offset": 0,
                "rows": len(call_keys),
                "entry_ids": call_keys,
                "entry_ids_sha256": digest(sorted(call_keys)),
                "buckets": [
                    {
                        "period": window_start.astimezone(
                            ZoneInfo("Europe/Moscow")
                        ).date().isoformat(),
                        "declared_total_calls_count": len(call_keys),
                        "rows": len(call_keys),
                        "entry_ids": call_keys,
                        "entry_ids_sha256": digest(sorted(call_keys)),
                    }
                ],
                "status": "complete",
            }
            for _ in [None]
            if call_keys
        ),
        {
            "offset": len(call_keys),
            "rows": 0,
            "entry_ids": [],
            "entry_ids_sha256": digest([]),
            "buckets": [],
            "status": "complete",
        },
    ]
    official_list = calls_runtime.build_mango_official_list_proof(
        call_ids=call_keys,
        page_receipts=official_pages,
        since=window_start,
        until=window_end,
    )
    proof = {
        "schema_version": "mango_exact_dual_enumeration_v3",
        "normalization_version": "mango_rows_call_day_v2",
        "tenant_id": "foton",
        "base_url": calls_runtime.DEFAULT_MANGO_BASE_URL,
        "fields_sha256": digest(calls_runtime.DEFAULT_STATS_FIELDS),
        "rolling_since": source["rolling_since"],
        "until": source["until"],
        "proof_run_id": "synthetic-proof-run-v1",
        "observed_at": "2026-08-12T00:00:00+00:00",
        "passes_required": 2,
        "passes_completed": 2,
        "passes": [
            {"pass_id": "primary", **pass_payload},
            {
                "pass_id": "verification",
                **pass_payload,
                "requests": len(verification_chunks),
                "chunks": verification_chunks,
                "partition_sha256": digest(
                    [
                        {"since": item["since"], "until": item["until"]}
                        for item in verification_chunks
                    ]
                ),
            },
        ],
        "official_list": official_list,
        "comparison": comparison,
        "enumeration_consistency_ok": True,
        "mismatch_reason": "",
    }
    proof["proof_sha256"] = digest(proof)
    source["covered_intervals"] = [
        *(
            {**interval, "authority_pass": 1}
            for interval in rolling
        ),
        *(
            {**interval, "scope": "rolling_authority", "authority_pass": 2}
            for interval in verification_chunks
        ),
        *auxiliary,
    ]
    source["requests"] = len(source["covered_intervals"])
    source["enumeration_consistency_ok"] = True
    source["dual_enumeration"] = proof
    auxiliary_rows = sum(
        int(interval.get("rows") or 0) for interval in auxiliary
    )
    result["enumeration_consistency_ok"] = True
    result["api_requests"] = source["requests"]
    result["api_authoritative_rows_total"] = raw_rows * 2
    result["api_auxiliary_rows_total"] = auxiliary_rows
    result["api_rows_total"] = raw_rows * 2 + auxiliary_rows
    return result


def create_legacy_transfer_cursor(
    config: CallsTwoProcessesConfig,
    *,
    until: str,
    zero_proofs: Mapping[str, int],
) -> Mapping[str, object]:
    from mango_mvp.productization.capture_staging import (
        CaptureManifestStore,
        ManifestEntry,
    )

    CaptureManifestStore(config.capture_manifest).append(
        ManifestEntry(
            schema_version="capture_manifest_v1",
            created_at=until,
            tenant_id="foton",
            provider="mango",
            event_key="foton:mango:transferred-call",
            provider_call_id="transferred-call",
            recording_id=None,
            started_at=until,
            ended_at=None,
            direction="inbound",
            client_phone=None,
            manager_ref=None,
            status="recording_retry_expired",
        )
    )
    snapshot = calls_runtime.capture_manifest_snapshot(config.capture_manifest)
    cursor: Mapping[str, object] = {
        "schema_version": "mango_api_freshness_v1",
        "until": until,
        "updated_at": until,
        "capture_status": "ok",
        "host_id": "source-mac",
        "manifest_end_offset": snapshot["end_offset"],
        "manifest_snapshot_sha256": snapshot["sha256"],
        "mango_enumeration_complete": True,
        "mango_enumeration_source": {},
        "catch_up": True,
        "sla_mode": "catch_up",
        "call_keys": [],
        "calls_by_moscow_day": {
            day: [] for day in zero_proofs
        },
        "independent_zero_enumerations_by_day": dict(zero_proofs),
    }
    write_json(config.cursor_path, cursor)
    return cursor


def test_config_refuses_prod_and_stable_runtime_paths(tmp_path: Path) -> None:
    prod = config_for(tmp_path, timeline_name="customer_timeline_prod_20260709.sqlite")
    with pytest.raises(ValueError, match="prod"):
        prod.validate()

    stable = CallsTwoProcessesConfig(
        pipeline_root=tmp_path / "stable_runtime" / "calls",
        timeline_db=tmp_path / "staging" / "customer_timeline.sqlite",
        timeline_allowed_root=tmp_path / "staging",
        python_executable=Path(sys.executable),
        codex_binary=Path(sys.executable),
        codex_home_root=tmp_path / "codex_home",
    )
    with pytest.raises(ValueError, match="stable_runtime"):
        stable.validate()


def test_process_b_producer_explicitly_enables_strict_service_readiness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = config_for(tmp_path)
    increment = config.ingest_dir / "increment.jsonl"
    report = config.ingest_dir / "producer.json"

    def fake_run(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        assert "--strict-service-ready" in command
        assert command.count("--package-db") == 1
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            json.dumps({"rows_selected": 0, "events_written": 0}),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(calls_runtime.subprocess, "run", fake_run)

    result = run_increment_producer(config, increment, report, None)

    assert result["status"] == "ok"


def test_disk_preflight_creates_owner_only_pipeline_root(tmp_path: Path) -> None:
    config = replace(config_for(tmp_path), min_free_gib=0)
    previous = os.umask(0)
    try:
        report = calls_runtime.disk_preflight(config)
    finally:
        os.umask(previous)

    assert report["ok"] is True
    assert config.pipeline_root.stat().st_mode & 0o777 == 0o700


def test_process_a_lock_is_nonblocking_and_reports_holder(tmp_path: Path) -> None:
    lock = tmp_path / "process_a.lock"
    with process_lease(lock, stale_seconds=60) as first:
        assert first["pid"]
        with pytest.raises(LockBusy) as caught:
            with process_lease(lock, stale_seconds=60):
                pass
        assert caught.value.metadata["pid"] == first["pid"]


def test_killed_lock_owner_releases_kernel_flock_for_next_run(
    tmp_path: Path,
) -> None:
    lock = tmp_path / "killed-owner.lock"
    root = Path(__file__).resolve().parents[1]
    child_code = (
        "import sys,time\n"
        "from pathlib import Path\n"
        "from mango_mvp.customer_timeline.calls_two_processes import process_lease\n"
        "with process_lease(Path(sys.argv[1]), stale_seconds=60):\n"
        " print('locked', flush=True)\n"
        " time.sleep(60)\n"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", child_code, str(lock)],
        cwd=root,
        env={
            **os.environ,
            "PYTHONPATH": str(root / "src"),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "locked"
        with pytest.raises(LockBusy):
            with process_lease(lock, stale_seconds=60):
                pass
    finally:
        child.kill()
        child.wait(timeout=5)

    with process_lease(lock, stale_seconds=60) as acquired:
        assert acquired["pid"] == os.getpid()


def test_pipeline_and_direct_process_a_share_one_lock_while_capture_is_independent(
    tmp_path: Path,
) -> None:
    config = config_for(tmp_path)
    assert config.process_a_lock == config.pipeline_lock

    with process_lease(config.pipeline_lock, stale_seconds=60):
        pipeline = run_pipeline(
            config,
            command_runner=lambda *_args: pytest.fail("worker must not start"),
        )
        direct = run_process_a(
            config,
            command_runner=lambda *_args: pytest.fail("worker must not start"),
        )
        for _ in range(3):
            with process_lease(config.capture_lock, stale_seconds=60):
                pass

    assert pipeline["status"] == "locked"
    assert pipeline["stop_reason"] == "pipeline_locked"
    assert direct["status"] == "locked"
    assert direct["stop_reason"] == "process_a_locked"


def test_run_capture_writes_cursor_only_for_complete_enumeration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(calls_runtime, "cutover_authority_report", lambda *_a, **_k: {"ok": True})
    monkeypatch.setattr(calls_runtime, "disk_preflight", lambda *_a, **_k: {"ok": True})
    complete = config_for(tmp_path / "complete")
    report = run_capture(
        complete,
        since="2026-08-11T00:00:00+00:00",
        until="2026-08-11T01:00:00+00:00",
        capture_runner=lambda *_a: {
            "status": "ok",
            "mango_enumeration_complete": True,
            "manifest_end_offset": 0,
            "manifest_snapshot_sha256": "a" * 64,
        },
    )
    assert report["status"] == "ok"
    assert read_json(complete.cursor_path)["mango_enumeration_complete"] is True

    incomplete = config_for(tmp_path / "incomplete")
    failed = run_capture(
        incomplete,
        since="2026-08-11T00:00:00+00:00",
        until="2026-08-11T01:00:00+00:00",
        capture_runner=lambda *_a: {
            "status": "partial",
            "mango_enumeration_complete": False,
        },
    )
    assert failed["status"] == "failed"
    assert failed["stop_reason"] == "capture_or_enumeration_failed"
    assert not incomplete.cursor_path.exists()


def test_direct_strict_entrypoints_reject_dirty_worktree_before_callers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected_sha = "a" * 40
    host = tmp_path / "host_id"
    host.write_text("m1-host\n", encoding="utf-8")
    host.chmod(0o600)
    cutover = tmp_path / "cutover.json"
    cutover.write_text(
        json.dumps(
            {
                "schema_version": "mango_calls_cutover_v2",
                "active_host_id": "m1-host",
                "previous_host_id": "source-mac",
                "expected_code_sha": expected_sha,
                "source_cursor_sha256": "b" * 64,
                "previous_host_snapshot_sha256": "c" * 64,
                "previous_host_disabled_at": "2026-08-11T08:00:00+00:00",
                "previous_host_checked_at": "2026-08-11T08:05:00+00:00",
                "approved_at": "2026-08-11T08:10:00+00:00",
                "approved_by": "owner",
            }
        ),
        encoding="utf-8",
    )
    cutover.chmod(0o600)
    config = replace(
        config_for(tmp_path),
        expected_code_sha=expected_sha,
        host_id_path=host,
        cutover_manifest_path=cutover,
        expected_previous_host_id="source-mac",
        require_cutover_authority=True,
        strict_ready_provenance=True,
    )
    monkeypatch.setattr(calls_runtime, "current_git_sha", lambda _root: expected_sha)
    monkeypatch.setattr(
        "mango_mvp.productization.mango_calls_service_contract.current_git_sha",
        lambda _root: expected_sha,
    )
    monkeypatch.setattr(
        "mango_mvp.productization.mango_calls_service_contract.git_worktree_is_clean",
        lambda _root: False,
    )

    capture = run_capture(
        config,
        capture_runner=lambda *_args: pytest.fail("Mango caller must not run"),
    )
    pipeline = run_pipeline(
        config,
        command_runner=lambda *_args: pytest.fail("worker caller must not run"),
    )

    assert capture["status"] == pipeline["status"] == "failed"
    assert capture["stop_reason"] == pipeline["stop_reason"] == "cutover_authority_failed"
    assert "cutover_worktree_dirty_or_unverifiable" in capture["counters"]["authority"]["errors"]
    assert "cutover_worktree_dirty_or_unverifiable" in pipeline["authority"]["errors"]


def test_first_lineage_requires_fresh_old_host_proof_then_becomes_stable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected_sha = "a" * 40
    config = replace(
        config_for(tmp_path),
        expected_code_sha=expected_sha,
        expected_previous_host_id="source-mac",
        require_cutover_authority=True,
        strict_ready_provenance=True,
    )
    config.host_id_file.parent.mkdir(parents=True, exist_ok=True)
    config.host_id_file.write_text("m1-host\n", encoding="utf-8")
    config.host_id_file.chmod(0o600)
    config.cutover_manifest_file.write_text("{}", encoding="utf-8")
    config.cutover_manifest_file.chmod(0o600)
    config.cursor_path.write_bytes(b"transferred-cursor")
    cursor_sha = sha256_file(config.cursor_path)
    fresh_required: list[bool] = []

    def verify(**kwargs: object) -> dict[str, object]:
        require_fresh = bool(kwargs["require_fresh_previous_host_proof"])
        fresh_required.append(require_fresh)
        return {
            "ok": not require_fresh or len(fresh_required) > 2,
            "errors": [] if not require_fresh or len(fresh_required) > 2 else [
                "previous_host_proof_stale"
            ],
            "active_host_id": "m1-host",
            "source_cursor_sha256": cursor_sha,
        }

    monkeypatch.setattr(calls_runtime, "verify_cutover_authority", verify)

    stale = calls_runtime.cutover_authority_report(
        config, initialize_lineage=True
    )
    assert stale["ok"] is False
    assert stale["errors"] == ["previous_host_proof_stale"]
    assert not config.cutover_cursor_lineage_path.exists()

    initialized = calls_runtime.cutover_authority_report(
        config, initialize_lineage=True
    )
    assert initialized["ok"] is True
    assert initialized["source_cursor_lineage_ok"] is True
    assert fresh_required == [False, True, False, True]

    steady = calls_runtime.cutover_authority_report(config)
    assert steady["ok"] is True
    assert steady["source_cursor_lineage_ok"] is True
    assert fresh_required == [False, True, False, True, False]

    repeated = calls_runtime.cutover_authority_report(
        config, initialize_lineage=True
    )
    assert repeated["ok"] is True
    assert repeated["source_cursor_lineage_ok"] is True
    assert fresh_required == [False, True, False, True, False, False]


def test_lineage_initialization_rejects_transferred_cursor_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected_sha = "a" * 40
    config = replace(
        config_for(tmp_path),
        expected_code_sha=expected_sha,
        expected_previous_host_id="source-mac",
        require_cutover_authority=True,
        strict_ready_provenance=True,
    )
    config.host_id_file.parent.mkdir(parents=True, exist_ok=True)
    config.host_id_file.write_text("m1-host\n", encoding="utf-8")
    config.host_id_file.chmod(0o600)
    config.cutover_manifest_file.write_text("{}", encoding="utf-8")
    config.cutover_manifest_file.chmod(0o600)
    config.cursor_path.write_bytes(b"wrong-transferred-cursor")

    monkeypatch.setattr(
        calls_runtime,
        "verify_cutover_authority",
        lambda **_kwargs: {
            "ok": True,
            "errors": [],
            "active_host_id": "m1-host",
            "source_cursor_sha256": hashlib.sha256(
                b"expected-transferred-cursor"
            ).hexdigest(),
        },
    )

    report = calls_runtime.cutover_authority_report(
        config, initialize_lineage=True
    )

    assert report["ok"] is False
    assert report["source_cursor_lineage_ok"] is False
    assert "source_cursor_lineage_unproven" in report["errors"]
    assert not config.cutover_cursor_lineage_path.exists()


def test_controlled_read_only_lineage_does_not_unlock_service_cutover(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected_sha = "a" * 40
    config = replace(
        config_for(tmp_path),
        expected_code_sha=expected_sha,
        expected_previous_host_id="source-mac",
        require_cutover_authority=True,
        strict_ready_provenance=True,
    )
    config.host_id_file.parent.mkdir(parents=True, exist_ok=True)
    config.host_id_file.write_text("m1-host\n", encoding="utf-8")
    config.host_id_file.chmod(0o600)
    config.cutover_manifest_file.write_text("{}", encoding="utf-8")
    config.cutover_manifest_file.chmod(0o600)
    config.cursor_path.write_bytes(b"transferred-cursor")
    config.cursor_path.chmod(0o600)
    cursor_sha = sha256_file(config.cursor_path)
    fresh_required: list[bool] = []

    def verify(**kwargs: object) -> dict[str, object]:
        fresh_required.append(
            bool(kwargs["require_fresh_previous_host_proof"])
        )
        return {
            "ok": True,
            "errors": [],
            "active_host_id": "m1-host",
            "source_cursor_sha256": cursor_sha,
        }

    monkeypatch.setattr(calls_runtime, "verify_cutover_authority", verify)

    controlled = calls_runtime.controlled_read_only_cutover_authority_report(
        config
    )
    service = calls_runtime.cutover_authority_report(config)
    pipeline = run_pipeline(
        config,
        command_runner=lambda *_args: pytest.fail("worker must stay blocked"),
    )
    process_a = run_process_a(
        config,
        skip_capture=True,
        command_runner=lambda *_args: pytest.fail("worker must stay blocked"),
    )

    assert controlled["ok"] is True
    assert controlled["source_cursor_lineage_ok"] is True
    assert controlled["controlled_cursor_binding_ok"] is True
    assert controlled["lineage_mode"] == "controlled_read_only"
    assert controlled["shared_service_lineage_written"] is False
    assert service["ok"] is False
    assert service["source_cursor_lineage_ok"] is False
    assert pipeline["status"] == "failed"
    assert pipeline["stop_reason"] == "cutover_authority_failed"
    assert process_a["status"] == "failed"
    assert process_a["stop_reason"] == "cutover_authority_failed"
    assert fresh_required == [True, False, False, False]
    assert not config.cutover_cursor_lineage_path.exists()

    config.cursor_path.write_bytes(b"changed-cursor")
    config.cursor_path.chmod(0o600)
    mismatch = calls_runtime.controlled_read_only_cutover_authority_report(
        config
    )
    assert mismatch["ok"] is False
    assert "controlled_source_cursor_mismatch" in mismatch["errors"]
    assert not config.cutover_cursor_lineage_path.exists()


def test_controlled_read_only_lineage_rejects_manifest_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected_sha = "a" * 40
    config = replace(
        config_for(tmp_path),
        expected_code_sha=expected_sha,
        expected_previous_host_id="source-mac",
        require_cutover_authority=True,
        strict_ready_provenance=True,
    )
    config.host_id_file.parent.mkdir(parents=True, exist_ok=True)
    config.host_id_file.write_text("m1-host\n", encoding="utf-8")
    config.host_id_file.chmod(0o600)
    config.cutover_manifest_file.write_text("{}", encoding="utf-8")
    config.cutover_manifest_file.chmod(0o600)
    config.cursor_path.write_bytes(b"transferred-cursor")
    config.cursor_path.chmod(0o600)
    cursor_sha = sha256_file(config.cursor_path)

    def verify(**_kwargs: object) -> dict[str, object]:
        config.cutover_manifest_file.write_text(
            '{"changed":true}', encoding="utf-8"
        )
        config.cutover_manifest_file.chmod(0o600)
        return {
            "ok": True,
            "errors": [],
            "active_host_id": "m1-host",
            "source_cursor_sha256": cursor_sha,
        }

    monkeypatch.setattr(calls_runtime, "verify_cutover_authority", verify)

    report = calls_runtime.controlled_read_only_cutover_authority_report(
        config
    )

    assert report["ok"] is False
    assert "controlled_cutover_manifest_changed_during_check" in report[
        "errors"
    ]
    assert not config.cutover_cursor_lineage_path.exists()


def test_run_pipeline_reuses_unchanged_frozen_snapshot_without_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = config_for(tmp_path)
    config.capture_manifest.parent.mkdir(parents=True)
    config.capture_manifest.write_bytes(b"")
    snapshot = calls_runtime.capture_manifest_snapshot(
        config.capture_manifest, end_offset=0
    )
    write_json(
        config.cursor_path,
        {
            "mango_enumeration_complete": True,
            "manifest_end_offset": 0,
            "manifest_snapshot_sha256": snapshot["sha256"],
        },
    )
    monkeypatch.setattr(calls_runtime, "cutover_authority_report", lambda *_a, **_k: {"ok": True})
    monkeypatch.setattr(
        calls_runtime,
        "run_process_a",
        lambda *_a, **_k: {
            "status": "ok",
            "stop_reason": "",
            "downstream_ready": True,
            "counters": {
                "metadata": {
                    "audio_files": 0,
                    "db_open_work": False,
                    "skipped": {"already_ingested": 3},
                },
                "drop": {"reused": True},
            },
        },
    )
    report = run_pipeline(
        config,
        command_runner=lambda *_a: pytest.fail("worker must not start"),
        process_b_runner=lambda _config: {
            "status": "idle",
            "stop_reason": "drop_unchanged",
        },
    )

    assert report["status"] == "idle"
    assert report["stop_reason"] == "unchanged_snapshot"
    assert report["new"] == 0
    assert report["reused"] == 3


def test_run_pipeline_rejects_changed_capture_snapshot_before_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = config_for(tmp_path)
    config.capture_manifest.parent.mkdir(parents=True)
    config.capture_manifest.write_bytes(b"")
    write_json(
        config.cursor_path,
        {
            "mango_enumeration_complete": True,
            "manifest_end_offset": 0,
            "manifest_snapshot_sha256": "0" * 64,
        },
    )
    monkeypatch.setattr(calls_runtime, "cutover_authority_report", lambda *_a, **_k: {"ok": True})
    monkeypatch.setattr(
        calls_runtime,
        "run_process_a",
        lambda *_a, **_k: pytest.fail("Process A must not start"),
    )

    report = run_pipeline(config)

    assert report["status"] == "failed"
    assert report["stop_reason"] == "capture_snapshot_sha256_mismatch"


def test_process_lock_rejects_symlink_without_touching_target(tmp_path: Path) -> None:
    victim = tmp_path / "victim.txt"
    victim.write_text("must remain unchanged", encoding="utf-8")
    lock = tmp_path / "process_a.lock"
    lock.symlink_to(victim)

    with pytest.raises(RuntimeError, match="lock is unsafe"):
        with process_lease(lock, stale_seconds=60):
            pass

    assert lock.is_symlink()
    assert victim.read_text(encoding="utf-8") == "must remain unchanged"


def test_terminate_process_group_kills_children_after_parent_exits_on_term(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {"parent": True, "group": True}
    delivered: list[int] = []

    class FakeProcess:
        pid = 424242

        def poll(self) -> int | None:
            return None if state["parent"] else 0

        def wait(self, timeout: float) -> int:
            assert timeout > 0
            state["parent"] = False
            return 0

    def killpg(_pid: int, sent_signal: int) -> None:
        if sent_signal == 0:
            if not state["group"]:
                raise ProcessLookupError
            return
        delivered.append(sent_signal)
        if sent_signal == 15:
            state["parent"] = False
        elif sent_signal == 9:
            state["group"] = False

    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.os.killpg", killpg
    )
    terminate_process_group(FakeProcess(), grace_seconds=0.1)  # type: ignore[arg-type]

    assert delivered == [15, 9]
    assert state == {"parent": False, "group": False}


def test_hardlink_race_rejects_a_concurrent_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.mp3"
    source.write_bytes(b"synthetic audio")
    target = tmp_path / "target.mp3"
    real_link = os.link

    def race_link(src: object, dst: object, **_kwargs: object) -> None:
        assert Path(src) == source
        Path(dst).symlink_to(source)
        raise FileExistsError

    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.os.link", race_link
    )
    with pytest.raises(RuntimeError, match="concurrent audio target is unsafe"):
        hardlink_or_copy(source, target)

    assert target.is_symlink()
    assert source.read_bytes() == b"synthetic audio"
    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.os.link", real_link
    )


def test_run_cycle_imports_partial_ready_drop_and_keeps_partial_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.run_process_a",
        lambda *_args, **_kwargs: {
            "status": "partial",
            "stop_reason": "capture_audio_incomplete",
            "downstream_ready": True,
        },
    )

    def fake_b(*_args, **_kwargs):
        calls.append("b")
        return {"status": "ok"}

    monkeypatch.setattr("mango_mvp.customer_timeline.calls_two_processes.run_process_b", fake_b)
    report = run_cycle(config_for(tmp_path))
    assert calls == ["b"]
    assert report["status"] == "partial"
    assert report["stop_reason"] == "capture_audio_incomplete"


@pytest.mark.parametrize("first", [
    {"status": "partial", "downstream_ready": False},
    {"status": "failed", "downstream_ready": False},
])
def test_run_cycle_does_not_import_without_ready_drop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, first: dict[str, object]
) -> None:
    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.run_process_a",
        lambda *_args, **_kwargs: first,
    )
    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.run_process_b",
        lambda *_args, **_kwargs: pytest.fail("Process B must not start"),
    )
    assert run_cycle(config_for(tmp_path))["process_b"] is None


def test_capture_keeps_calls_without_recording_in_retry_queue(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = replace(config_for(tmp_path), api_window_hours=1)
    tenant = TenantRef("foton")
    no_recording = TelephonyCallEvent(
        tenant=tenant,
        provider="mango",
        provider_call_id="late",
        started_at=datetime(2026, 7, 9, tzinfo=timezone.utc),
        ended_at=None,
        direction=Direction.INBOUND,
        client_phone=None,
        manager_ref=None,
        recording_ref=None,
        raw_payload={},
    )
    ready = TelephonyCallEvent(
        tenant=tenant,
        provider="mango",
        provider_call_id="ready",
        started_at=datetime(2026, 7, 9, 1, tzinfo=timezone.utc),
        ended_at=None,
        direction=Direction.OUTBOUND,
        client_phone=None,
        manager_ref=None,
        recording_ref="recording-1",
        raw_payload={},
    )

    class FakeClient:
        def __init__(self, **_: object) -> None:
            self.calls = 0

        def poll_call_history(self, **_: object) -> list[dict[str, str]]:
            self.calls += 1
            return [{"id": "late"}, {"id": "ready"}] if self.calls == 1 else []

    class FakeMapper:
        def __init__(self) -> None:
            self.items = iter((no_recording, ready))

        def from_payload(self, **_: object) -> TelephonyCallEvent:
            return next(self.items)

    captured: list[TelephonyCallEvent] = []

    class Summary:
        failed = 0
        skipped_no_recording = 1

        def to_json_dict(self) -> dict[str, int]:
            return {"downloaded": 1, "failed": 0, "skipped_no_recording": 1}

    def fake_stage(*, events: list[TelephonyCallEvent], **_: object) -> Summary:
        captured.extend(events)
        return Summary()

    monkeypatch.setattr("mango_mvp.customer_timeline.calls_two_processes.MangoOfficeClient", FakeClient)
    monkeypatch.setattr("mango_mvp.customer_timeline.calls_two_processes.MangoOfficePayloadMapper", FakeMapper)
    monkeypatch.setattr("mango_mvp.customer_timeline.calls_two_processes.MangoRecordingDownloader", FakeClient)
    monkeypatch.setattr("mango_mvp.customer_timeline.calls_two_processes.stage_capture_events", fake_stage)
    monkeypatch.setenv("MANGO_OFFICE_API_KEY", "present")
    monkeypatch.setenv("MANGO_OFFICE_API_SALT", "present")

    report = capture_mango_window(
        config,
        datetime(2026, 7, 9, tzinfo=timezone.utc),
        datetime(2026, 7, 9, 2, tzinfo=timezone.utc),
    )

    assert [event.provider_call_id for event in captured] == ["late", "ready"]
    assert report["status"] == "ok"
    assert report["api_requests"] == 2
    assert report["api_events_without_recording"] == 1


def test_capture_reports_partial_when_one_download_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = replace(config_for(tmp_path), api_window_hours=1)

    class FakeClient:
        def __init__(self, **_: object) -> None:
            pass

        def poll_call_history(self, **_: object) -> list[dict[str, str]]:
            return []

    class Summary:
        failed = 1

        def to_json_dict(self) -> dict[str, int]:
            return {"downloaded": 0, "failed": 1}

    monkeypatch.setattr("mango_mvp.customer_timeline.calls_two_processes.MangoOfficeClient", FakeClient)
    monkeypatch.setattr("mango_mvp.customer_timeline.calls_two_processes.MangoRecordingDownloader", FakeClient)
    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.stage_capture_events",
        lambda **_: Summary(),
    )
    monkeypatch.setenv("MANGO_OFFICE_API_KEY", "present")
    monkeypatch.setenv("MANGO_OFFICE_API_SALT", "present")

    report = capture_mango_window(
        config,
        datetime(2026, 7, 9, tzinfo=timezone.utc),
        datetime(2026, 7, 9, 1, tzinfo=timezone.utc),
    )

    assert report["status"] == "partial"
    assert config.capture_manifest.exists()
    assert config.capture_manifest.stat().st_mode & 0o777 == 0o600


def test_capture_refuses_to_recreate_missing_manifest_for_prior_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = config_for(tmp_path)
    write_json(
        config.process_a_status_path,
        {
            "process": "process_a",
            "status": "ok",
            "checked_through": "2026-07-09T09:00:00+00:00",
            "data_through": "2026-07-09T09:00:00+00:00",
        },
    )

    class FakeClient:
        def __init__(self, **_: object) -> None:
            pass

        def poll_call_history(self, **_: object) -> list[dict[str, str]]:
            pytest.fail("API poll must not run when a prior manifest is missing")

    monkeypatch.setattr("mango_mvp.customer_timeline.calls_two_processes.MangoOfficeClient", FakeClient)
    monkeypatch.setattr("mango_mvp.customer_timeline.calls_two_processes.MangoRecordingDownloader", FakeClient)
    monkeypatch.setenv("MANGO_OFFICE_API_KEY", "present")
    monkeypatch.setenv("MANGO_OFFICE_API_SALT", "present")

    with pytest.raises(RuntimeError, match="missing for an existing runtime"):
        capture_mango_window(
            config,
            datetime(2026, 7, 9, 9, tzinfo=timezone.utc),
            datetime(2026, 7, 9, 10, tzinfo=timezone.utc),
        )

    assert not config.capture_manifest.exists()


@pytest.mark.parametrize("marker_kind", ["fifo", "dangling_symlink"])
def test_capture_prior_status_special_file_fails_closed_without_hanging(
    tmp_path: Path,
    marker_kind: str,
) -> None:
    config = config_for(tmp_path)
    marker = config.process_a_status_path
    marker.parent.mkdir(parents=True, exist_ok=True)
    if marker_kind == "fifo":
        os.mkfifo(marker)
    else:
        marker.symlink_to(tmp_path / "missing-status-target.json")
    child_code = """
import sys
from pathlib import Path
from mango_mvp.customer_timeline.calls_two_processes import (
    CallsTwoProcessesConfig, capture_runtime_has_prior_state,
)
root = Path(sys.argv[1])
config = CallsTwoProcessesConfig(
    pipeline_root=root / "pipeline",
    timeline_db=root / "staging" / "customer_timeline_staging.sqlite",
    timeline_allowed_root=root / "staging",
    python_executable=Path(sys.executable),
    codex_binary=Path(sys.executable),
    codex_home_root=root / "codex_home",
    min_free_gib=1,
)
print("blocked" if capture_runtime_has_prior_state(config) else "accepted")
"""

    completed = subprocess.run(
        [sys.executable, "-c", child_code, str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=2,
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": "src"},
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "blocked"


def test_capture_prior_status_swap_after_open_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from mango_mvp.customer_timeline import calls_two_processes as module

    config = config_for(tmp_path)
    marker = config.process_a_status_path
    replacement = tmp_path / "replacement-status.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("{}\n", encoding="utf-8")
    replacement.write_text(
        json.dumps({"checked_through": "2026-08-08T00:00:00+00:00"}) + "\n",
        encoding="utf-8",
    )
    real_open = module.os.open
    swapped = False

    def swap_after_open(path: object, flags: int, mode: int = 0o777) -> int:
        nonlocal swapped
        descriptor = real_open(path, flags, mode)
        if Path(path) == marker and not swapped:
            os.replace(replacement, marker)
            swapped = True
        return descriptor

    monkeypatch.setattr(module.os, "open", swap_after_open)

    assert module.read_regular_json_marker(marker) is None
    assert swapped is True
    assert module.capture_runtime_has_prior_state(config) is True


def test_capture_recovers_torn_manifest_when_api_window_is_empty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from mango_mvp.productization.capture_staging import (
        CaptureManifestStore,
        ManifestEntry,
        acknowledge_capture_recovery,
    )

    config = replace(config_for(tmp_path), api_window_hours=1)
    store = CaptureManifestStore(config.capture_manifest)
    store.append(
        ManifestEntry(
            schema_version="capture_manifest_v1",
            created_at="2026-07-09T08:00:00+00:00",
            tenant_id="foton",
            provider="mango",
            event_key="foton:mango:old-call",
            provider_call_id="old-call",
            recording_id=None,
            started_at="2026-07-09T08:00:00+00:00",
            ended_at=None,
            direction="inbound",
            client_phone=None,
                manager_ref=None,
                status="recording_retry_expired",
                error="recording_missing_after_retry_ttl",
            )
    )
    with config.capture_manifest.open("ab") as handle:
        handle.write(b'{"event_key":"unfinished"')

    class EmptyClient:
        def __init__(self, **_: object) -> None:
            pass

        def poll_call_history(self, **_: object) -> list[dict[str, str]]:
            return []

    monkeypatch.setattr("mango_mvp.customer_timeline.calls_two_processes.MangoOfficeClient", EmptyClient)
    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.MangoRecordingDownloader",
        EmptyClient,
    )
    monkeypatch.setenv("MANGO_OFFICE_API_KEY", "present")
    monkeypatch.setenv("MANGO_OFFICE_API_SALT", "present")
    since = datetime(2026, 7, 9, 9, tzinfo=timezone.utc)
    until = datetime(2026, 7, 9, 10, tzinfo=timezone.utc)

    first = capture_mango_window(config, since, until)

    assert first["status"] == "partial"
    assert first["incomplete_trailing_manifest_records"] == 0
    assert first["recovered_trailing_manifest_records"] == 1
    acknowledge_capture_recovery(
        config.capture_manifest,
        expected_count=1,
        expected_incident_sha256=str(first["recovery_incident_sha256"]),
    )

    second = capture_mango_window(config, since, until)

    assert second["status"] == "ok"
    assert second["incomplete_trailing_manifest_records"] == 0
    assert second["recovered_trailing_manifest_records"] == 0


def test_capture_reports_tail_recovered_during_new_append(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from mango_mvp.productization.capture_staging import CaptureManifestStore, ManifestEntry

    config = replace(config_for(tmp_path), api_window_hours=1)
    store = CaptureManifestStore(config.capture_manifest)
    store.append(
        ManifestEntry(
            schema_version="v1",
            created_at="2026-07-09T08:00:00+00:00",
            tenant_id="foton",
            provider="mango",
            event_key="foton:mango:expired",
            provider_call_id="expired",
            recording_id=None,
            started_at="2026-07-09T08:00:00+00:00",
            ended_at=None,
            direction="inbound",
            client_phone=None,
            manager_ref=None,
            status="recording_retry_expired",
            error="recording_missing_after_retry_ttl",
            remediation_code="manual_review_or_retry_if_recording_appears",
        )
    )
    with config.capture_manifest.open("ab") as handle:
        handle.write(b'{"event_key":"unfinished"')
    new_event = TelephonyCallEvent(
        tenant=TenantRef("foton"),
        provider="mango",
        provider_call_id="new-call",
        started_at=datetime(2026, 7, 9, 10, tzinfo=timezone.utc),
        ended_at=None,
        direction=Direction.INBOUND,
        client_phone=None,
        manager_ref=None,
        recording_ref="new-recording",
        raw_payload={},
    )

    class FakeClient:
        def __init__(self, **_: object) -> None:
            pass

        def poll_call_history(self, **_: object) -> list[dict[str, str]]:
            return [{"id": "new-call"}]

    class FakeMapper:
        def from_payload(self, **_: object) -> TelephonyCallEvent:
            return new_event

    class Summary:
        failed = 0
        skipped_no_recording = 0

        def to_json_dict(self) -> dict[str, int]:
            return {"downloaded": 1, "failed": 0, "skipped_no_recording": 0}

    def fake_stage(*, manifest_store: CaptureManifestStore, **_: object) -> Summary:
        manifest_store.append(
            ManifestEntry(
                schema_version="v1",
                created_at="2026-07-09T10:00:00+00:00",
                tenant_id="foton",
                provider="mango",
                event_key="foton:mango:new-call",
                provider_call_id="new-call",
                recording_id="new-recording",
                started_at="2026-07-09T10:00:00+00:00",
                ended_at=None,
                direction="inbound",
                client_phone=None,
                manager_ref=None,
                status="downloaded",
                local_audio_path="/synthetic/new-call.mp3",
            )
        )
        return Summary()

    monkeypatch.setattr("mango_mvp.customer_timeline.calls_two_processes.MangoOfficeClient", FakeClient)
    monkeypatch.setattr("mango_mvp.customer_timeline.calls_two_processes.MangoRecordingDownloader", FakeClient)
    monkeypatch.setattr("mango_mvp.customer_timeline.calls_two_processes.MangoOfficePayloadMapper", FakeMapper)
    monkeypatch.setattr("mango_mvp.customer_timeline.calls_two_processes.stage_capture_events", fake_stage)
    monkeypatch.setenv("MANGO_OFFICE_API_KEY", "present")
    monkeypatch.setenv("MANGO_OFFICE_API_SALT", "present")

    report = capture_mango_window(
        config,
        datetime(2026, 7, 9, 9, tzinfo=timezone.utc),
        datetime(2026, 7, 9, 10, tzinfo=timezone.utc),
    )

    assert report["status"] == "partial"
    assert report["incomplete_trailing_manifest_records"] == 0
    assert report["recovered_trailing_manifest_records"] == 1
    assert len(CaptureManifestStore(config.capture_manifest).read_entries()) == 2


def test_pending_recording_widens_poll_window_beyond_normal_overlap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = replace(config_for(tmp_path), api_window_hours=12, pending_recording_retry_hours=24)
    from mango_mvp.productization.capture_staging import CaptureManifestStore, ManifestEntry

    CaptureManifestStore(config.capture_manifest).append(
        ManifestEntry(
            schema_version="v1",
            created_at="2026-07-09T08:00:00+00:00",
            tenant_id="foton",
            provider="mango",
            event_key="foton:mango:pending-1",
            provider_call_id="pending-1",
            recording_id=None,
            started_at="2026-07-09T08:00:00+00:00",
            ended_at=None,
            direction="inbound",
            client_phone=None,
            manager_ref=None,
            status="skipped_no_recording",
        )
    )
    requested: list[datetime] = []

    class FakeClient:
        def __init__(self, **_: object) -> None:
            pass

        def poll_call_history(self, *, since: datetime, **_: object) -> list[dict[str, str]]:
            requested.append(since)
            return []

    class Summary:
        failed = 0
        skipped_no_recording = 0

        def to_json_dict(self) -> dict[str, int]:
            return {"downloaded": 0, "failed": 0, "skipped_no_recording": 0}

    monkeypatch.setattr("mango_mvp.customer_timeline.calls_two_processes.MangoOfficeClient", FakeClient)
    monkeypatch.setattr("mango_mvp.customer_timeline.calls_two_processes.MangoRecordingDownloader", FakeClient)
    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.stage_capture_events",
        lambda **_: Summary(),
    )
    monkeypatch.setenv("MANGO_OFFICE_API_KEY", "present")
    monkeypatch.setenv("MANGO_OFFICE_API_SALT", "present")

    report = capture_mango_window(
        config,
        datetime(2026, 7, 9, 10, tzinfo=timezone.utc),
        datetime(2026, 7, 9, 11, tzinfo=timezone.utc),
    )

    assert requested[0] == datetime(2026, 7, 9, 7, 30, tzinfo=timezone.utc)
    assert report["pending_recording_retries"] == 1
    assert report["status"] == "ok"


def test_recent_manifest_overlap_is_clamped_to_exact_enumeration_until(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = replace(
        config_for(tmp_path),
        api_window_hours=24,
        strict_ready_provenance=True,
    )
    from mango_mvp.productization.capture_staging import CaptureManifestStore, ManifestEntry

    audio = config.recordings_dir / "recent.wav"
    audio.parent.mkdir(parents=True, exist_ok=True)
    audio.write_bytes(b"synthetic-audio")
    CaptureManifestStore(config.capture_manifest).append(
        ManifestEntry(
            schema_version="v1",
            created_at="2026-08-11T11:56:00+00:00",
            tenant_id="foton",
            provider="mango",
            event_key="foton:mango:recent",
            provider_call_id="recent",
            recording_id="recording-recent",
            started_at="2026-08-11T11:55:00+00:00",
            ended_at="2026-08-11T11:56:00+00:00",
            direction="inbound",
            client_phone=None,
            manager_ref=None,
            status="downloaded",
            local_audio_path=str(audio),
        )
    )
    requested: list[tuple[datetime, datetime]] = []

    class EmptyClient:
        def __init__(self, **_: object) -> None:
            pass

        def poll_call_history(
            self, *, since: datetime, until: datetime
        ) -> list[dict[str, object]]:
            requested.append((since, until))
            return []

    class Summary:
        failed = 0

        def to_json_dict(self) -> dict[str, int]:
            return {"downloaded": 0, "failed": 0}

    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.MangoOfficeClient",
        EmptyClient,
    )
    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.MangoRecordingDownloader",
        EmptyClient,
    )
    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.stage_capture_events",
        lambda **_: Summary(),
    )
    monkeypatch.setenv("MANGO_OFFICE_API_KEY", "present")
    monkeypatch.setenv("MANGO_OFFICE_API_SALT", "present")
    exact_until = datetime(2026, 8, 11, 12, tzinfo=timezone.utc)

    report = capture_mango_window(
        config,
        datetime(2026, 8, 11, 11, tzinfo=timezone.utc),
        exact_until,
    )

    assert requested
    assert max(window_until for _window_since, window_until in requested) == exact_until
    assert all(window_until <= exact_until for _window_since, window_until in requested)
    assert report["mango_enumeration_source"]["until"] == exact_until.isoformat()
    calls_runtime.capture_enumeration_evidence_sha256(report)


def test_old_local_recovery_is_not_fabricated_as_current_api_enumeration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = replace(
        config_for(tmp_path),
        api_window_hours=24,
        strict_ready_provenance=True,
    )
    from mango_mvp.productization.capture_staging import CaptureManifestStore, ManifestEntry

    CaptureManifestStore(config.capture_manifest).append(
        ManifestEntry(
            schema_version="v1",
            created_at="2025-01-01T10:01:00+00:00",
            tenant_id="foton",
            provider="mango",
            event_key="foton:mango:old-failed",
            provider_call_id="old-failed",
            recording_id="recording-old",
            started_at="2025-01-01T10:00:00+00:00",
            ended_at="2025-01-01T10:01:00+00:00",
            direction="inbound",
            client_phone=None,
            manager_ref=None,
            status="failed",
            local_audio_path=str(config.recordings_dir / "missing.wav"),
            error="synthetic_missing_asset",
        )
    )
    staged: list[TelephonyCallEvent] = []
    old_event = TelephonyCallEvent(
        tenant=TenantRef("foton"),
        provider="mango",
        provider_call_id="old-failed",
        started_at=datetime(2025, 1, 1, 10, tzinfo=timezone.utc),
        ended_at=datetime(2025, 1, 1, 10, 1, tzinfo=timezone.utc),
        direction=Direction.INBOUND,
        client_phone=None,
        manager_ref=None,
        recording_ref="recording-old",
        raw_payload={},
    )

    class EmptyClient:
        def __init__(self, **_: object) -> None:
            pass

        def poll_call_history(
            self, *, since: datetime, until: datetime
        ) -> list[dict[str, object]]:
            return [
                {
                    "id": "old-failed",
                    "start": old_event.started_at.isoformat(),
                    "finish": old_event.ended_at.isoformat(),
                }
            ] if since <= old_event.started_at <= until else []

    class FakeMapper:
        def from_payload(self, **_: object) -> TelephonyCallEvent:
            return old_event

    class Summary:
        failed = 0

        def to_json_dict(self) -> dict[str, int]:
            return {"downloaded": 0, "failed": 0}

    def fake_stage(
        *, events: list[TelephonyCallEvent], **_: object
    ) -> Summary:
        staged.extend(events)
        return Summary()

    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.MangoOfficeClient",
        EmptyClient,
    )
    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.MangoRecordingDownloader",
        EmptyClient,
    )
    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.MangoOfficePayloadMapper",
        FakeMapper,
    )
    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.stage_capture_events",
        fake_stage,
    )
    monkeypatch.setenv("MANGO_OFFICE_API_KEY", "present")
    monkeypatch.setenv("MANGO_OFFICE_API_SALT", "present")

    report = capture_mango_window(
        config,
        datetime(2026, 8, 11, 11, tzinfo=timezone.utc),
        datetime(2026, 8, 11, 12, tzinfo=timezone.utc),
    )

    assert [event.provider_call_id for event in staged] == ["old-failed"]
    assert "2025-01-01" not in report["calls_by_moscow_day"]
    assert "old-failed" not in report["call_keys"]
    assert report["api_events_total"] == 0
    assert report["api_rows_total"] == 1
    assert report["api_authoritative_rows_total"] == 0
    assert report["api_auxiliary_rows_total"] == 1
    calls_runtime.capture_enumeration_evidence_sha256(report)


def test_recording_retry_ttl_uses_first_seen_and_expires_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = replace(config_for(tmp_path), pending_recording_retry_hours=24)
    from mango_mvp.productization.capture_staging import CaptureManifestStore, ManifestEntry

    store = CaptureManifestStore(config.capture_manifest)
    store.append(
        ManifestEntry(
            schema_version="v1",
            created_at="2026-07-10T10:00:00+00:00",
            tenant_id="foton",
            provider="mango",
            event_key="foton:mango:old-call-newly-seen",
            provider_call_id="old-call-newly-seen",
            recording_id=None,
            started_at="2025-01-01T00:00:00+00:00",
            ended_at=None,
            direction="inbound",
            client_phone=None,
            manager_ref=None,
            status="skipped_no_recording",
        )
    )
    requested: list[tuple[datetime, datetime]] = []

    class FakeClient:
        def __init__(self, **_: object) -> None:
            pass

        def poll_call_history(self, *, since: datetime, until: datetime) -> list[dict[str, str]]:
            requested.append((since, until))
            return []

    class Summary:
        failed = 0
        skipped_no_recording = 0

        def to_json_dict(self) -> dict[str, int]:
            return {"downloaded": 0, "failed": 0, "skipped_no_recording": 0}

    monkeypatch.setattr("mango_mvp.customer_timeline.calls_two_processes.MangoOfficeClient", FakeClient)
    monkeypatch.setattr("mango_mvp.customer_timeline.calls_two_processes.MangoRecordingDownloader", FakeClient)
    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.stage_capture_events",
        lambda **_: Summary(),
    )
    monkeypatch.setenv("MANGO_OFFICE_API_KEY", "present")
    monkeypatch.setenv("MANGO_OFFICE_API_SALT", "present")

    first = capture_mango_window(
        config,
        datetime(2026, 7, 12, 10, tzinfo=timezone.utc),
        datetime(2026, 7, 12, 11, tzinfo=timezone.utc),
    )
    lines_after_first = len(store.read_entries())
    second = capture_mango_window(
        config,
        datetime(2026, 7, 12, 11, tzinfo=timezone.utc),
        datetime(2026, 7, 12, 12, tzinfo=timezone.utc),
    )

    assert first["api_requests"] == 2
    assert first["status"] == "ok"
    assert first["pending_recording_expired"] == 1
    assert store.latest_by_event_key()["foton:mango:old-call-newly-seen"].status == "recording_retry_expired"
    assert second["status"] == "ok"
    assert len(store.read_entries()) == lines_after_first
    assert len(requested) == 3


def test_audio_integrity_quarantine_is_stable_and_never_retried_as_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from mango_mvp.productization.capture_staging import (
        CaptureManifestStore,
        ManifestEntry,
    )

    config = replace(config_for(tmp_path), pending_recording_retry_hours=24)
    store = CaptureManifestStore(config.capture_manifest)
    quarantined = ManifestEntry(
        schema_version="capture_manifest_v1",
        created_at="2026-07-09T08:00:00+00:00",
        tenant_id="foton",
        provider="mango",
        event_key="foton:mango:damaged-audio",
        provider_call_id="damaged-audio",
        recording_id="recording-damaged",
        started_at="2026-07-09T08:00:00+00:00",
        ended_at=None,
        direction="inbound",
        client_phone=None,
        manager_ref=None,
        status="audio_integrity_quarantined",
        error="capture_target_integrity_mismatch",
        remediation_code="manual_restore_or_quarantine_corrupted_audio",
        recovery_state="immutable_audio_violation",
    )
    store.append(quarantined)

    class EmptyClient:
        def __init__(self, **_: object) -> None:
            pass

        def poll_call_history(self, **_: object) -> list[dict[str, str]]:
            return []

    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.MangoOfficeClient",
        EmptyClient,
    )
    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.MangoRecordingDownloader",
        EmptyClient,
    )
    monkeypatch.setenv("MANGO_OFFICE_API_KEY", "present")
    monkeypatch.setenv("MANGO_OFFICE_API_SALT", "present")

    report = capture_mango_window(
        config,
        datetime(2026, 7, 20, 10, tzinfo=timezone.utc),
        datetime(2026, 7, 20, 11, tzinfo=timezone.utc),
    )

    latest = store.latest_by_event_key()[quarantined.event_key]
    assert latest == quarantined
    assert len(store.read_entries()) == 1
    assert report["status"] == "ok"
    assert report["pending_recording_retries"] == 0
    assert report["pending_recording_expired"] == 0
    assert report["open_audio_integrity_quarantined"] == 1
    assert report["capture_assets_complete"] is False


def test_expired_unknown_recording_is_reenumerated_daily_by_full_moscow_day(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = replace(
        config_for(tmp_path), api_window_hours=12, pending_recording_retry_hours=72
    )
    from mango_mvp.productization.capture_staging import CaptureManifestStore, ManifestEntry

    store = CaptureManifestStore(config.capture_manifest)
    expired = ManifestEntry(
        schema_version="v1",
        created_at="2026-07-11T09:00:00+00:00",
        tenant_id="foton",
        provider="mango",
        event_key="foton:mango:expired-unknown",
        provider_call_id="expired-unknown",
        recording_id=None,
        started_at="2025-01-01T10:00:00+00:00",
        ended_at="2025-01-01T10:20:00+00:00",
        direction="inbound",
        client_phone=None,
        manager_ref=None,
        status="recording_retry_expired",
        error="recording_missing_after_retry_ttl",
        remediation_code="manual_review_or_retry_if_recording_appears",
    )
    store.append(expired)
    requested: list[tuple[datetime, datetime]] = []

    class EmptyClient:
        def __init__(self, **_: object) -> None:
            pass

        def poll_call_history(
            self, *, since: datetime, until: datetime
        ) -> list[dict[str, str]]:
            requested.append((since, until))
            return []

    class Summary:
        failed = 0
        skipped_no_recording = 0

        def to_json_dict(self) -> dict[str, int]:
            return {"downloaded": 0, "failed": 0, "skipped_no_recording": 0}

    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.MangoOfficeClient",
        EmptyClient,
    )
    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.MangoRecordingDownloader",
        EmptyClient,
    )
    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.stage_capture_events",
        lambda **_: Summary(),
    )
    monkeypatch.setenv("MANGO_OFFICE_API_KEY", "present")
    monkeypatch.setenv("MANGO_OFFICE_API_SALT", "present")
    first_until = datetime(2026, 7, 12, 11, tzinfo=timezone.utc)

    first = capture_mango_window(
        config,
        first_until - timedelta(hours=1),
        first_until,
    )
    old_day_requests = [
        (start, end)
        for start, end in requested
        if start.year < 2026
    ]
    entries_after_first = store.read_entries()
    requested.clear()
    second = capture_mango_window(
        config,
        first_until + timedelta(hours=22),
        first_until + timedelta(hours=23),
    )

    assert len(old_day_requests) == 2
    assert all(end - start <= timedelta(hours=12) for start, end in old_day_requests)
    assert old_day_requests[0][0] == datetime(2024, 12, 31, 21, tzinfo=timezone.utc)
    assert old_day_requests[-1][1] == datetime(2025, 1, 1, 21, tzinfo=timezone.utc)
    assert first["pending_recording_expired"] == 0
    assert first["expired_recording_reenumerated_still_missing"] == 1
    assert first["capture_assets_complete"] is True
    assert "2025-01-02" not in first["independent_zero_enumerations_by_day"]
    assert len(entries_after_first) == 2
    assert entries_after_first[-1].status == "recording_retry_expired"
    assert entries_after_first[-1].created_at == first_until.isoformat()
    assert (
        entries_after_first[-1].recovery_state
        == "late_recording_reenumerated_still_missing"
    )
    assert len(requested) == 1
    assert second["expired_recording_reenumerated_still_missing"] == 0
    assert len(store.read_entries()) == len(entries_after_first)


def test_expired_unknown_recording_recovers_when_id_appears(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = replace(config_for(tmp_path), api_window_hours=12)
    from mango_mvp.productization.capture_staging import CaptureManifestStore, ManifestEntry

    store = CaptureManifestStore(config.capture_manifest)
    expired = ManifestEntry(
        schema_version="v1",
        created_at="2026-07-11T09:00:00+00:00",
        tenant_id="foton",
        provider="mango",
        event_key="foton:mango:expired-unknown",
        provider_call_id="expired-unknown",
        recording_id=None,
        started_at="2025-01-01T10:00:00+00:00",
        ended_at="2025-01-01T10:20:00+00:00",
        direction="inbound",
        client_phone=None,
        manager_ref=None,
        status="recording_retry_expired",
        error="recording_missing_after_retry_ttl",
        remediation_code="manual_review_or_retry_if_recording_appears",
    )
    store.append(expired)

    class LateClient:
        def __init__(self, **_: object) -> None:
            pass

        def poll_call_history(
            self, *, since: datetime, until: datetime
        ) -> list[dict[str, str]]:
            started = datetime(2025, 1, 1, 10, tzinfo=timezone.utc)
            if since <= started < until:
                return [
                    {
                        "id": "expired-unknown",
                        "started_at": started.isoformat(),
                        "ended_at": "2025-01-01T10:20:00+00:00",
                        "direction": "inbound",
                        "recording_ref": "recording-late",
                    }
                ]
            return []

    class Summary:
        failed = 0
        skipped_no_recording = 0

        def to_json_dict(self) -> dict[str, int]:
            return {"downloaded": 1, "failed": 0, "skipped_no_recording": 0}

    def fake_stage(
        *,
        events: list[TelephonyCallEvent],
        manifest_store: CaptureManifestStore,
        **_: object,
    ) -> Summary:
        assert [event.recording_ref for event in events] == ["recording-late"]
        manifest_store.append(
            replace(
                expired,
                created_at="2026-07-12T11:00:00+00:00",
                recording_id="recording-late",
                recording_ids=("recording-late",),
                status="downloaded",
                recovery_state="recovered_late_recording",
            )
        )
        return Summary()

    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.MangoOfficeClient",
        LateClient,
    )
    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.MangoRecordingDownloader",
        LateClient,
    )
    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.stage_capture_events",
        fake_stage,
    )
    monkeypatch.setenv("MANGO_OFFICE_API_KEY", "present")
    monkeypatch.setenv("MANGO_OFFICE_API_SALT", "present")

    report = capture_mango_window(
        config,
        datetime(2026, 7, 12, 10, tzinfo=timezone.utc),
        datetime(2026, 7, 12, 11, tzinfo=timezone.utc),
    )
    latest = store.latest_by_event_key()[expired.event_key]

    assert report["api_requests"] == 3
    assert report["expired_recording_reenumerated_still_missing"] == 0
    assert latest.status == "downloaded"
    assert latest.recording_id == "recording-late"
    assert latest.recovery_state == "recovered_late_recording"
    assert len(store.read_entries()) == 2


def test_expired_recording_is_recovered_on_last_bounded_attempt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = replace(config_for(tmp_path), api_window_hours=12, pending_recording_retry_hours=24)
    from mango_mvp.productization.capture_staging import CaptureManifestStore, ManifestEntry

    store = CaptureManifestStore(config.capture_manifest)
    pending = ManifestEntry(
        schema_version="v1",
        created_at="2026-07-10T10:00:00+00:00",
        tenant_id="foton",
        provider="mango",
        event_key="foton:mango:recovered",
        provider_call_id="recovered",
        recording_id=None,
        started_at="2025-01-01T00:00:00+00:00",
        ended_at="2025-01-01T00:20:00+00:00",
        direction="inbound",
        client_phone=None,
        manager_ref=None,
        status="skipped_no_recording",
    )
    store.append(pending)

    class FakeClient:
        def __init__(self, **_: object) -> None:
            pass

        def poll_call_history(self, *, since: datetime, **_: object) -> list[dict[str, str]]:
            if since.year < 2026:
                return [{
                    "id": "recovered",
                    "started_at": "2025-01-01T00:00:00+00:00",
                    "ended_at": "2025-01-01T00:20:00+00:00",
                    "direction": "inbound",
                    "recording_ref": "recording-late",
                }]
            return []

    class Summary:
        failed = 0
        skipped_no_recording = 0

        def to_json_dict(self) -> dict[str, int]:
            return {"downloaded": 1, "failed": 0, "skipped_no_recording": 0}

    def fake_stage(*, events: list[TelephonyCallEvent], **_: object) -> Summary:
        assert [event.recording_ref for event in events] == ["recording-late"]
        store.append(replace(pending, status="downloaded", recording_id="recording-late"))
        return Summary()

    monkeypatch.setattr("mango_mvp.customer_timeline.calls_two_processes.MangoOfficeClient", FakeClient)
    monkeypatch.setattr("mango_mvp.customer_timeline.calls_two_processes.MangoRecordingDownloader", FakeClient)
    monkeypatch.setattr("mango_mvp.customer_timeline.calls_two_processes.stage_capture_events", fake_stage)
    monkeypatch.setenv("MANGO_OFFICE_API_KEY", "present")
    monkeypatch.setenv("MANGO_OFFICE_API_SALT", "present")

    report = capture_mango_window(
        config,
        datetime(2026, 7, 12, 10, tzinfo=timezone.utc),
        datetime(2026, 7, 12, 11, tzinfo=timezone.utc),
    )

    assert report["api_requests"] == 2
    assert report["status"] == "ok"
    assert report["pending_recording_expired"] == 0
    assert store.latest_by_event_key()[pending.event_key].status == "downloaded"


def test_prepare_ingest_inputs_is_idempotent(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    source = config.recordings_dir / "call.mp3"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"audio")
    config.capture_manifest.parent.mkdir(parents=True, exist_ok=True)
    config.capture_manifest.write_text(
        json.dumps(
            {
                "schema_version": "capture_manifest_v1",
                "created_at": "2026-07-09T00:00:00+00:00",
                "tenant_id": "foton",
                "provider": "mango",
                "event_key": "event:1",
                "provider_call_id": "call-1",
                "recording_id": "recording-1",
                "recording_ids": ["recording-1"],
                "started_at": "2026-07-09T00:00:00+00:00",
                "direction": "inbound",
                "status": "downloaded",
                "local_audio_path": str(source),
                "size_bytes": source.stat().st_size,
                    "checksum_sha256": sha256_file(source),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    first = prepare_ingest_inputs(config)
    second = prepare_ingest_inputs(config)

    assert first["audio_files"] == second["audio_files"] == 1
    assert second["link_actions"] == {"exists_same_hash": 1}


def test_prepare_ingest_inputs_waits_for_recording_set_stabilization(tmp_path: Path) -> None:
    config = replace(
        config_for(tmp_path),
        pending_recording_retry_hours=24,
        recording_set_stabilization_minutes=120,
    )
    source = config.recordings_dir / "recent.mp3"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"audio")
    from mango_mvp.productization.capture_staging import CaptureManifestStore, ManifestEntry

    started = datetime.now(timezone.utc) - timedelta(hours=1)
    CaptureManifestStore(config.capture_manifest).append(
        ManifestEntry(
            schema_version="v1",
            created_at=started.isoformat(),
            tenant_id="foton",
            provider="mango",
            event_key="foton:mango:recent",
            provider_call_id="recent",
            recording_id="rec-1",
            recording_ids=("rec-1",),
            started_at=started.isoformat(),
            ended_at=None,
            direction="inbound",
            client_phone=None,
            manager_ref=None,
            status="downloaded",
            local_audio_path=str(source),
        )
    )

    result = prepare_ingest_inputs(config)

    assert result["audio_files"] == 0
    assert result["skipped"] == {"recording_set_stabilizing": 1}


def test_prepare_ingest_inputs_skips_call_already_present_in_working_db(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    source = config.recordings_dir / "known.mp3"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"audio")
    from mango_mvp.productization.capture_staging import CaptureManifestStore, ManifestEntry

    CaptureManifestStore(config.capture_manifest).append(
        ManifestEntry(
            schema_version="v1", created_at="2026-07-01T00:00:00+00:00", tenant_id="foton",
            provider="mango", event_key="foton:mango:known", provider_call_id="known",
            recording_id="rec-1", started_at="2026-07-01T00:00:00+00:00", ended_at=None,
            direction="inbound", client_phone=None, manager_ref=None, status="downloaded",
            local_audio_path=str(source),
        )
    )
    config.working_db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(config.working_db) as con:
        con.execute("CREATE TABLE call_records(source_call_id TEXT)")
        con.execute("INSERT INTO call_records VALUES ('known')")

    result = prepare_ingest_inputs(config)

    assert result["audio_files"] == 0
    assert result["skipped"] == {"already_in_working": 1}
    assert result["incomplete_total"] == 0


def test_prepare_ingest_inputs_keeps_missing_known_audio_visible(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    from mango_mvp.productization.capture_staging import CaptureManifestStore, ManifestEntry

    CaptureManifestStore(config.capture_manifest).append(
        ManifestEntry(
            schema_version="v1", created_at="2026-07-01T00:00:00+00:00", tenant_id="foton",
            provider="mango", event_key="foton:mango:known", provider_call_id="known",
            recording_id="rec-1", started_at="2026-07-01T00:00:00+00:00", ended_at=None,
            direction="inbound", client_phone=None, manager_ref=None, status="downloaded",
            local_audio_path=str(config.recordings_dir / "missing.mp3"),
        )
    )
    config.working_db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(config.working_db) as con:
        con.execute("CREATE TABLE call_records(source_call_id TEXT)")
        con.execute("INSERT INTO call_records VALUES ('known')")

    result = prepare_ingest_inputs(config)

    assert result["skipped"] == {"audio_file_missing": 1}
    assert result["incomplete_total"] == 1


def test_prepare_ingest_inputs_restores_missing_working_audio_for_known_call(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    source = config.recordings_dir / "known.mp3"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"audio")
    from mango_mvp.productization.capture_staging import CaptureManifestStore, ManifestEntry

    CaptureManifestStore(config.capture_manifest).append(
        ManifestEntry(
            schema_version="v1", created_at="2026-07-01T00:00:00+00:00", tenant_id="foton",
            provider="mango", event_key="foton:mango:known", provider_call_id="known",
            recording_id="rec-1", started_at="2026-07-01T00:00:00+00:00", ended_at=None,
            direction="inbound", client_phone=None, manager_ref=None, status="downloaded",
            local_audio_path=str(source),
        )
    )
    config.working_db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(config.working_db) as con:
        con.execute("CREATE TABLE call_records(source_call_id TEXT)")
        con.execute("INSERT INTO call_records VALUES ('known')")

    result = prepare_ingest_inputs(config)

    assert (config.working_audio_dir / source.name).read_bytes() == b"audio"
    assert result["incomplete_total"] == 0
    assert sum(result["link_actions"].values()) == 1


def test_existing_manifest_event_is_not_hidden_by_external_known_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = replace(config_for(tmp_path), api_window_hours=1)
    from mango_mvp.productization.capture_staging import CaptureManifestStore, ManifestEntry

    audio = config.recordings_dir / "existing.mp3"
    audio.parent.mkdir(parents=True, exist_ok=True)
    audio.write_bytes(b"existing")
    CaptureManifestStore(config.capture_manifest).append(
        ManifestEntry(
            schema_version="v1",
            created_at="2026-07-09T10:00:00+00:00",
            tenant_id="foton",
            provider="mango",
            event_key="foton:mango:call-multi",
            provider_call_id="call-multi",
            recording_id="rec-1",
            recording_ids=("rec-1",),
            started_at="2026-07-09T10:00:00+00:00",
            ended_at="2026-07-09T10:10:00+00:00",
            direction="inbound",
            client_phone=None,
            manager_ref=None,
            status="downloaded",
            local_audio_path=str(audio),
        )
    )
    captured: list[TelephonyCallEvent] = []

    class FakeClient:
        def __init__(self, **_: object) -> None:
            pass

        def poll_call_history(self, **_: object) -> list[dict[str, object]]:
            base = {"id": "call-multi", "started_at": "2026-07-09T10:00:00+00:00", "ended_at": "2026-07-09T10:10:00+00:00"}
            return [{**base, "records": ["rec-1", "rec-2"]}, {**base, "records": ["rec-1"]}]

    class Summary:
        failed = 0
        skipped_no_recording = 0
        needs_review_multiple_recordings = 1

        def to_json_dict(self) -> dict[str, int]:
            return {"downloaded": 0, "failed": 0, "skipped_no_recording": 0, "needs_review_multiple_recordings": 1}

    def fake_stage(*, events: list[TelephonyCallEvent], **_: object) -> Summary:
        captured.extend(events)
        store = CaptureManifestStore(config.capture_manifest)
        current = store.latest_by_event_key()["foton:mango:call-multi"]
        store.append(replace(current, status="multiple_recordings_needs_review", recording_ids=("rec-1", "rec-2"), recording_paths=(str(audio), str(audio))))
        return Summary()

    monkeypatch.setattr("mango_mvp.customer_timeline.calls_two_processes.MangoOfficeClient", FakeClient)
    monkeypatch.setattr("mango_mvp.customer_timeline.calls_two_processes.MangoRecordingDownloader", FakeClient)
    monkeypatch.setattr("mango_mvp.customer_timeline.calls_two_processes.stage_capture_events", fake_stage)
    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.read_known_processed_ids",
        lambda *_args: (set(), {"call-multi"}),
    )
    monkeypatch.setenv("MANGO_OFFICE_API_KEY", "present")
    monkeypatch.setenv("MANGO_OFFICE_API_SALT", "present")

    report = capture_mango_window(
        config,
        datetime(2026, 7, 9, 10, tzinfo=timezone.utc),
        datetime(2026, 7, 9, 11, tzinfo=timezone.utc),
    )

    assert [event.recording_refs for event in captured] == [("rec-1", "rec-2")]
    assert report["api_events_already_known_external"] == 0
    assert report["status"] == "ok"
    assert report["open_multiple_recordings_needs_review"] == 1


def test_known_processed_ids_only_accept_successful_downloads(tmp_path: Path) -> None:
    root = tmp_path / "product_data"
    package = root / "mango_update_after_test"
    package.mkdir(parents=True)
    rows = [
        {"action": "DOWNLOADED_RECORDING", "recording_id": "ready", "provider_call_id": "call-ready"},
        {"action": "FAILED_DOWNLOAD", "recording_id": "retry", "provider_call_id": "call-retry"},
    ]
    (package / "recording_download_manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    recordings, calls = read_known_processed_ids(root)

    assert recordings == {"ready"}
    assert calls == {"call-ready"}


def test_worker_command_is_drain_and_never_sync(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    command = worker_command(config, "resolve,analyze")
    assert "--poll-sec" in command
    assert "--max-idle-cycles" in command
    assert command[command.index("--stage-limit") + 1] == "20"
    assert "sync" not in command


def test_calls_runtime_requires_flex_codex_service_tier(tmp_path: Path) -> None:
    config = replace(config_for(tmp_path), codex_service_tier="fast")

    with pytest.raises(ValueError, match="codex_service_tier must be flex"):
        config.validate()


def test_pipeline_matches_ui_one_stage_at_a_time(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    calls: list[tuple[list[str], dict[str, str]]] = []
    ephemeral_paths: list[Path] = []

    def fake_runner(command, env, cwd):
        del cwd
        for key in (
            "CODEX_HOME",
            "MANGO_CODEX_PROCESS_HOME",
            "MANGO_CODEX_PROCESS_TMPDIR",
        ):
            path = Path(str(env[key]))
            assert path.is_dir()
            ephemeral_paths.append(path)
        calls.append((list(command), dict(env)))
        return {"rc": 0}

    result = run_sequential_pipeline_workers(config, {}, fake_runner)

    assert len(result) == len(SEQUENTIAL_PIPELINE_STAGES) == 4
    assert [command[command.index("--stages") + 1] for command, _ in calls] == list(
        SEQUENTIAL_PIPELINE_STAGES
    )
    assert calls[0][1]["DUAL_TRANSCRIBE_ENABLED"] == "0"
    assert calls[1][1]["DUAL_TRANSCRIBE_ENABLED"] == "1"
    assert calls[2][1]["DUAL_TRANSCRIBE_ENABLED"] == "1"
    assert calls[3][1]["DUAL_TRANSCRIBE_ENABLED"] == "1"
    assert all("sync" not in command for command, _ in calls)
    assert len({path.parent for path in ephemeral_paths}) == 4
    assert all(not path.exists() for path in ephemeral_paths)
    assert config.codex_home_root.is_dir()
    assert config.codex_home_root.stat().st_mode & 0o777 == 0o700
    assert not list(config.codex_home_root.iterdir())

    second = run_sequential_pipeline_workers(config, {}, fake_runner)
    assert len(second) == 4
    assert all(not path.exists() for path in ephemeral_paths)
    assert not list(config.codex_home_root.iterdir())


def test_codex_runtime_anchor_repairs_owned_mode_and_rejects_symlink(
    tmp_path: Path,
) -> None:
    config = config_for(tmp_path)
    config.codex_home_root.mkdir(mode=0o755)
    config.codex_home_root.chmod(0o755)

    resolved = ensure_codex_runtime_anchor(config)

    assert resolved == config.codex_home_root.resolve()
    assert resolved.stat().st_mode & 0o777 == 0o700

    config.codex_home_root.rmdir()
    victim = tmp_path / "victim"
    victim.mkdir(mode=0o700)
    config.codex_home_root.symlink_to(victim, target_is_directory=True)
    with pytest.raises((OSError, RuntimeError)):
        ensure_codex_runtime_anchor(config)


@pytest.mark.parametrize("terminate_raises", [False, True])
def test_real_stage_cleans_runtime_and_checks_group_after_parent_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    terminate_raises: bool,
) -> None:
    config = config_for(tmp_path)
    runtime_paths: list[Path] = []
    terminated: list[int] = []

    class ExitedParent:
        pid = 424242
        returncode = 0

        def __init__(self, *_args, **kwargs) -> None:
            runtime = Path(str(kwargs["env"]["CODEX_HOME"])).parent
            (runtime / ".sandbox_migration").write_text(
                "synthetic residue\n",
                encoding="utf-8",
            )
            runtime_paths.append(runtime)

        def poll(self) -> int:
            return 0

    def terminate(proc) -> None:
        terminated.append(proc.pid)
        if terminate_raises:
            raise RuntimeError("synthetic terminate failure")

    monkeypatch.setattr(calls_runtime, "pipeline_stages", lambda *_a, **_k: ("resolve",))
    monkeypatch.setattr(calls_runtime.subprocess, "Popen", ExitedParent)
    monkeypatch.setattr(calls_runtime, "terminate_process_group", terminate)

    if terminate_raises:
        with pytest.raises(RuntimeError, match="synthetic terminate failure"):
            run_sequential_pipeline_workers(
                config,
                {},
                calls_runtime.run_command,
                run_id="synthetic",
            )
    else:
        report = run_sequential_pipeline_workers(
            config,
            {},
            calls_runtime.run_command,
            run_id="synthetic",
        )
        assert report[0]["rc"] == 0

    assert terminated == [424242]
    assert runtime_paths
    assert all(not path.exists() for path in runtime_paths)
    assert not list(config.codex_home_root.iterdir())


def test_each_stage_timeout_is_capped_by_shared_four_hour_cycle() -> None:
    timeout = 4 * 60 * 60
    cycle_deadline = 100.0 + timeout

    assert stage_timeout_deadline(
        started_at=100.0,
        timeout_seconds=timeout,
        cycle_deadline=cycle_deadline,
    ) == (
        cycle_deadline
    )
    assert stage_timeout_deadline(
        started_at=14_000.0,
        timeout_seconds=timeout,
        cycle_deadline=cycle_deadline,
    ) == (
        cycle_deadline
    )


def test_prelude_command_obeys_shared_heavy_cycle_deadline(tmp_path: Path) -> None:
    timed_out = run_command(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        os.environ,
        tmp_path,
        deadline=calls_runtime.time.monotonic() + 0.05,
    )
    completed = run_command(
        [sys.executable, "-c", "print('ok')"],
        os.environ,
        tmp_path,
        deadline=calls_runtime.time.monotonic() + 2,
    )

    assert timed_out["rc"] == 124
    assert timed_out["timed_out"] is True
    assert timed_out["timeout_scope"] == "heavy_cycle"
    assert completed["rc"] == 0
    assert completed["timed_out"] is False


def test_single_asr_fallback_mode_is_rejected(tmp_path: Path) -> None:
    config = replace(config_for(tmp_path), asr_mode="gigaam_fallback")
    with pytest.raises(ValueError, match="single-ASR fallback is disabled"):
        config.validate()


def test_publish_ready_db_handles_space_path_wal_tmp(tmp_path: Path) -> None:
    config = replace(config_for(tmp_path), pipeline_root=tmp_path / "Mango analyse" / "pipeline")
    config.working_db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(config.working_db) as con:
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("CREATE TABLE call_records(id INTEGER PRIMARY KEY)")
        con.execute("INSERT INTO call_records(id) VALUES (1)")

    manifest = publish_ready_db(config, {"total": 1})

    assert manifest["quick_check"] == "ok"
    assert config.ready_db.exists()
    assert config.ready_db.stat().st_mode & 0o777 == 0o600
    assert config.ready_db.parent.stat().st_mode & 0o777 == 0o700
    assert config.ready_manifest.stat().st_mode & 0o777 == 0o600
    assert not config.ready_db.with_suffix(".sqlite.tmp-shm").exists()


def test_publish_ready_db_is_private_under_open_umask(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    config.working_db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(config.working_db) as con:
        con.execute("CREATE TABLE call_records(id INTEGER PRIMARY KEY)")
    previous = os.umask(0)
    try:
        publish_ready_db(config, {"total": 0})
    finally:
        os.umask(previous)

    assert config.ready_db.stat().st_mode & 0o777 == 0o600
    assert config.ready_db.parent.stat().st_mode & 0o777 == 0o700
    assert config.ready_manifest.stat().st_mode & 0o777 == 0o600


def test_ready_manifest_build_failure_keeps_previous_generation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = config_for(tmp_path)
    config.working_db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(config.working_db) as con:
        con.execute("CREATE TABLE call_records(id INTEGER PRIMARY KEY)")
        con.execute("INSERT INTO call_records VALUES (1)")
    publish_ready_db(config, {"total": 1})
    before_db = sha256_file(config.ready_db)
    before_manifest = sha256_file(config.ready_manifest)
    with sqlite3.connect(config.working_db) as con:
        con.execute("INSERT INTO call_records VALUES (2)")
    monkeypatch.setattr(
        calls_runtime,
        "_ready_verdicts",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("synthetic manifest failure")
        ),
    )

    with pytest.raises(RuntimeError, match="synthetic manifest failure"):
        publish_ready_db(config, {"total": 2})

    assert sha256_file(config.ready_db) == before_db
    assert sha256_file(config.ready_manifest) == before_manifest
    assert inspect_ready_publication(config.ready_db)["recovery_required"] is False


def test_ready_publication_recovers_crash_after_database_replace(
    tmp_path: Path,
) -> None:
    config = config_for(tmp_path)
    config.working_db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(config.working_db) as con:
        con.execute("CREATE TABLE call_records(id INTEGER PRIMARY KEY)")
        con.execute("INSERT INTO call_records VALUES (1)")
    publish_ready_db(config, {"total": 1})
    old_sha = sha256_file(config.ready_db)
    with sqlite3.connect(config.working_db) as con:
        con.execute("INSERT INTO call_records VALUES (2)")

    def crash_after_database(stage: str) -> None:
        if stage == "db_replaced":
            raise RuntimeError("synthetic crash after database")

    with pytest.raises(RuntimeError, match="synthetic crash after database"):
        publish_ready_db(
            config,
            {"total": 2},
            publication_checkpoint=crash_after_database,
        )

    assert sha256_file(config.ready_db) != old_sha
    assert inspect_ready_publication(config.ready_db)["recovery_required"] is True
    recovered = recover_ready_generation(config.ready_db)
    fingerprint = calls_runtime.ready_drop_fingerprint(config)

    assert recovered == {
        "status": "recovered",
        "recovered": True,
        "generation": "new",
    }
    assert fingerprint["manifest_valid"] is True
    with sqlite3.connect(config.ready_db) as con:
        assert con.execute("SELECT COUNT(*) FROM call_records").fetchone()[0] == 2
    assert recover_ready_generation(config.ready_db) == {
        "status": "clean",
        "recovered": False,
    }


def test_ready_publication_recovers_idempotently_after_manifest_replace(
    tmp_path: Path,
) -> None:
    config = config_for(tmp_path)
    config.working_db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(config.working_db) as con:
        con.execute("CREATE TABLE call_records(id INTEGER PRIMARY KEY)")
        con.execute("INSERT INTO call_records VALUES (1)")

    def crash_after_manifest(stage: str) -> None:
        if stage == "manifest_replaced":
            raise RuntimeError("synthetic crash after manifest")

    with pytest.raises(RuntimeError, match="synthetic crash after manifest"):
        publish_ready_db(
            config,
            {"total": 1},
            publication_checkpoint=crash_after_manifest,
        )

    first = recover_ready_generation(config.ready_db)
    pair = (sha256_file(config.ready_db), sha256_file(config.ready_manifest))
    second = recover_ready_generation(config.ready_db)

    assert first["generation"] == "new"
    assert second == {"status": "clean", "recovered": False}
    assert pair == (sha256_file(config.ready_db), sha256_file(config.ready_manifest))


def test_ready_publication_rolls_back_if_forward_manifest_is_missing(
    tmp_path: Path,
) -> None:
    config = config_for(tmp_path)
    config.working_db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(config.working_db) as con:
        con.execute("CREATE TABLE call_records(id INTEGER PRIMARY KEY)")
        con.execute("INSERT INTO call_records VALUES (1)")
    publish_ready_db(config, {"total": 1})
    old_pair = (sha256_file(config.ready_db), sha256_file(config.ready_manifest))
    with sqlite3.connect(config.working_db) as con:
        con.execute("INSERT INTO call_records VALUES (2)")

    def crash_after_database(stage: str) -> None:
        if stage == "db_replaced":
            raise RuntimeError("synthetic crash after database")

    with pytest.raises(RuntimeError):
        publish_ready_db(
            config,
            {"total": 2},
            publication_checkpoint=crash_after_database,
        )
    transaction = config.ready_db.parent / (
        f".{config.ready_db.name}.publication.txn"
    )
    (transaction / "new.manifest.json").unlink()

    recovered = recover_ready_generation(config.ready_db)

    assert recovered["generation"] == "old"
    assert old_pair == (sha256_file(config.ready_db), sha256_file(config.ready_manifest))
    restored_manifest = read_json(config.ready_manifest)
    assert config.ready_db.stat().st_mtime_ns == restored_manifest["ready_mtime_ns"]


def test_ready_publication_same_sha_rollback_restores_reusable_metadata(
    tmp_path: Path,
) -> None:
    config = config_for(tmp_path)
    config.working_db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(config.working_db) as connection:
        connection.execute(
            "CREATE TABLE call_records(source_call_id TEXT, started_at TEXT)"
        )
    create_empty_capture_manifest(config)
    snapshot = calls_runtime.capture_manifest_snapshot(
        config.capture_manifest,
        end_offset=0,
    )
    source = {
        "mode": "strict_service",
        "since": "2026-08-07T21:00:00+00:00",
        "rolling_since": "2026-08-07T21:00:00+00:00",
        "until": "2026-08-08T21:00:00+00:00",
        "cursor": "not_applicable_stats_request_result",
        "requests": 1,
        "pages": None,
        "pagination": "not_applicable_stats_request_result",
        "covered_intervals": [
            {
                "since": "2026-08-07T21:00:00+00:00",
                "until": "2026-08-08T21:00:00+00:00",
                "result_complete": True,
                "rows": 0,
                "scope": "rolling_authority",
            }
        ],
    }

    def evidence(zero_proofs: int) -> dict[str, object]:
        return with_dual_enumeration({
            "mango_enumeration_complete": True,
            "mango_enumeration_source": source,
            "call_keys": [],
            "calls_by_moscow_day": {"2026-08-08": []},
            "independent_zero_enumerations_by_day": {
                "2026-08-08": zero_proofs
            },
            "api_requests": 1,
            "api_rows_total": 0,
            "api_authoritative_rows_total": 0,
            "api_events_total": 0,
            "manifest_end_offset": 0,
            "manifest_snapshot_sha256": snapshot["sha256"],
        })

    first_evidence = evidence(2)
    publish_ready_db(
        config,
        {"total": 0},
        capture_evidence=first_evidence,
        manifest_end_offset=0,
    )
    old_db_sha = sha256_file(config.ready_db)
    old_mtime_ns = config.ready_db.stat().st_mtime_ns

    def crash_after_database(stage: str) -> None:
        if stage == "db_replaced":
            raise RuntimeError("synthetic same-sha crash after database")

    with pytest.raises(RuntimeError, match="synthetic same-sha crash"):
        publish_ready_db(
            config,
            {"total": 0},
            capture_evidence=evidence(2),
            manifest_end_offset=0,
            publication_checkpoint=crash_after_database,
        )

    assert sha256_file(config.ready_db) == old_db_sha
    replacement_inode = config.ready_db.stat().st_ino
    transaction = config.ready_db.parent / (
        f".{config.ready_db.name}.publication.txn"
    )
    staged_old_inode = (transaction / "old.sqlite").stat().st_ino
    (transaction / "new.manifest.json").unlink()

    recovered = recover_ready_generation(config.ready_db)
    recovered_stat = config.ready_db.stat()
    reused = publish_ready_db_if_changed(
        config,
        {"total": 0},
        changed=False,
        capture_evidence=first_evidence,
        manifest_end_offset=0,
    )

    assert recovered["generation"] == "old"
    assert recovered_stat.st_ino == staged_old_inode
    assert recovered_stat.st_ino != replacement_inode
    assert recovered_stat.st_mtime_ns == old_mtime_ns
    assert reused["reused"] is True


def test_ready_publication_inspection_does_not_alarm_during_active_commit(
    tmp_path: Path,
) -> None:
    config = config_for(tmp_path)
    config.working_db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(config.working_db) as con:
        con.execute("CREATE TABLE call_records(id INTEGER PRIMARY KEY)")
        con.execute("INSERT INTO call_records VALUES (1)")
    observations: list[Mapping[str, Any]] = []

    def inspect_while_locked(stage: str) -> None:
        if stage == "journal_written":
            observations.append(inspect_ready_publication(config.ready_db))

    publish_ready_db(
        config,
        {"total": 1},
        publication_checkpoint=inspect_while_locked,
    )

    assert observations == [
        {"status": "publication_active", "recovery_required": False}
    ]
    assert inspect_ready_publication(config.ready_db) == {
        "status": "clean",
        "recovery_required": False,
    }


def test_publish_ready_db_never_reports_new_after_safe_rollback(
    tmp_path: Path,
) -> None:
    config = config_for(tmp_path)
    config.working_db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(config.working_db) as con:
        con.execute("CREATE TABLE call_records(id INTEGER PRIMARY KEY)")
        con.execute("INSERT INTO call_records VALUES (1)")
    publish_ready_db(config, {"total": 1})
    old_pair = (
        sha256_file(config.ready_db),
        sha256_file(config.ready_manifest),
    )
    with sqlite3.connect(config.working_db) as con:
        con.execute("INSERT INTO call_records VALUES (2)")

    def lose_forward_manifest(stage: str) -> None:
        if stage == "journal_written":
            transaction = config.ready_db.parent / (
                f".{config.ready_db.name}.publication.txn"
            )
            (transaction / "new.manifest.json").unlink()

    with pytest.raises(RuntimeError, match="rolled back"):
        publish_ready_db(
            config,
            {"total": 2},
            publication_checkpoint=lose_forward_manifest,
        )

    assert old_pair == (
        sha256_file(config.ready_db),
        sha256_file(config.ready_manifest),
    )
    assert inspect_ready_publication(config.ready_db)["recovery_required"] is False


def test_ready_publication_recovery_refuses_unknown_canonical_hash(
    tmp_path: Path,
) -> None:
    config = config_for(tmp_path)
    config.working_db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(config.working_db) as con:
        con.execute("CREATE TABLE call_records(id INTEGER PRIMARY KEY)")
        con.execute("INSERT INTO call_records VALUES (1)")

    def crash_after_database(stage: str) -> None:
        if stage == "db_replaced":
            raise RuntimeError("synthetic crash after database")

    with pytest.raises(RuntimeError):
        publish_ready_db(
            config,
            {"total": 1},
            publication_checkpoint=crash_after_database,
        )
    config.ready_db.write_bytes(b"unknown generation")

    with pytest.raises(RuntimeError, match="unknown generation"):
        recover_ready_generation(config.ready_db)
    assert inspect_ready_publication(config.ready_db)["recovery_required"] is True


def test_process_b_recovers_interrupted_ready_publication_before_import(
    tmp_path: Path,
) -> None:
    config = config_for(tmp_path)
    create_ready_call_db(config.working_db)

    def crash_after_database(stage: str) -> None:
        if stage == "db_replaced":
            raise RuntimeError("synthetic crash after database")

    with pytest.raises(RuntimeError):
        publish_ready_db(
            config,
            {"total": 1},
            publication_checkpoint=crash_after_database,
        )
    with CustomerTimelineSQLiteStore(
        config.timeline_db, allowed_root=config.timeline_allowed_root
    ):
        pass

    report = run_process_b(config)

    assert report["status"] == "ok"
    assert inspect_ready_publication(config.ready_db)["recovery_required"] is False
    assert calls_runtime.ready_drop_fingerprint(config)["manifest_valid"] is True


def test_strict_cli_creates_owner_only_sqlite_under_open_umask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "strict-runtime" / "working.sqlite"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db}")
    monkeypatch.setenv("MANGO_STRICT_ASR_RUNTIME", "1")
    get_settings.cache_clear()
    previous = os.umask(0)
    try:
        assert mango_cli.main(["init-db"]) == 0
    finally:
        os.umask(previous)
        get_settings.cache_clear()

    assert db.stat().st_mode & 0o777 == 0o600
    assert db.parent.stat().st_mode & 0o777 == 0o700


def test_unchanged_working_db_rebuilds_modified_ready_copy(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    config.working_db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(config.working_db) as con:
        con.execute("CREATE TABLE call_records(id INTEGER PRIMARY KEY)")
        con.execute("INSERT INTO call_records VALUES (1)")
    publish_ready_db(config, {"total": 1})
    config.ready_db.write_bytes(b"damaged")

    result = publish_ready_db_if_changed(config, {"total": 1}, changed=False)

    assert result["reused"] is False
    assert sqlite_check(config.ready_db, "quick_check") == "ok"


def test_ready_manifest_rebuilds_when_only_enumeration_evidence_advances(
    tmp_path: Path,
) -> None:
    config = config_for(tmp_path)
    config.working_db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(config.working_db) as connection:
        connection.execute(
            "CREATE TABLE call_records(source_call_id TEXT, started_at TEXT)"
        )
    create_empty_capture_manifest(config)
    snapshot = calls_runtime.capture_manifest_snapshot(
        config.capture_manifest,
        end_offset=0,
    )
    day = "2026-08-08"
    source = {
        "mode": "strict_service",
        "since": "2026-08-07T21:00:00+00:00",
        "rolling_since": "2026-08-07T21:00:00+00:00",
        "until": "2026-08-08T21:00:00+00:00",
        "cursor": "not_applicable_stats_request_result",
        "requests": 1,
        "pages": None,
        "pagination": "not_applicable_stats_request_result",
        "covered_intervals": [
            {
                "since": "2026-08-07T21:00:00+00:00",
                "until": "2026-08-08T21:00:00+00:00",
                "result_complete": True,
                "rows": 0,
                "scope": "rolling_authority",
            }
        ],
    }

    def evidence(zero_proofs: int) -> dict[str, object]:
        local_source = json.loads(json.dumps(source))
        if zero_proofs == 0:
            local_source["until"] = "2026-08-08T12:00:00+00:00"
            local_source["covered_intervals"][0]["until"] = (
                "2026-08-08T12:00:00+00:00"
            )
        return with_dual_enumeration({
            "mango_enumeration_complete": True,
            "mango_enumeration_source": local_source,
            "call_keys": [],
            "calls_by_moscow_day": {day: []},
            "independent_zero_enumerations_by_day": {day: zero_proofs},
            "api_requests": 1,
            "api_rows_total": 0,
            "api_authoritative_rows_total": 0,
            "api_events_total": 0,
            "manifest_end_offset": 0,
            "manifest_snapshot_sha256": snapshot["sha256"],
        })

    first = publish_ready_db(
        config,
        {"total": 0},
        capture_evidence=evidence(0),
        manifest_end_offset=0,
    )
    second = publish_ready_db_if_changed(
        config,
        {"total": 0},
        changed=False,
        capture_evidence=evidence(2),
        manifest_end_offset=0,
    )
    repeated = publish_ready_db_if_changed(
        config,
        {"total": 0},
        changed=False,
        capture_evidence=evidence(2),
        manifest_end_offset=0,
    )

    assert first["daily_verdicts"][day]["mango_enumeration_complete"] is False
    assert first["consistency_ok"] is False
    assert first["closure_ok"] is False
    assert second["reused"] is False
    assert second["daily_verdicts"][day]["mango_enumeration_complete"] is True
    assert second["consistency_ok"] is True
    assert second["closure_ok"] is True
    assert first["enumeration_evidence_sha256"] != second[
        "enumeration_evidence_sha256"
    ]
    assert repeated["reused"] is True
    assert repeated["enumeration_evidence_sha256"] == second[
        "enumeration_evidence_sha256"
    ]


def test_green_ready_manifest_rebuilds_for_each_new_exact_capture_proof(
    tmp_path: Path,
) -> None:
    config = config_for(tmp_path)
    config.working_db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(config.working_db) as connection:
        connection.execute(
            "CREATE TABLE call_records(source_call_id TEXT, started_at TEXT)"
        )
    create_empty_capture_manifest(config)
    snapshot = calls_runtime.capture_manifest_snapshot(
        config.capture_manifest,
        end_offset=0,
    )
    def source(
        until: str, intervals: list[dict[str, object]]
    ) -> dict[str, object]:
        return {
            "mode": "strict_service",
            "since": "2026-08-07T21:00:00+00:00",
            "rolling_since": "2026-08-07T21:00:00+00:00",
            "until": until,
            "cursor": "not_applicable_stats_request_result",
            "requests": len(intervals),
            "pages": None,
            "pagination": "not_applicable_stats_request_result",
            "covered_intervals": intervals,
            "catch_up": False,
        }

    first_source = source(
        "2026-08-08T21:00:00+00:00",
        [
            {
                "since": "2026-08-07T21:00:00+00:00",
                "until": "2026-08-08T21:00:00+00:00",
                "result_complete": True,
                "rows": 0,
                "scope": "rolling_authority",
            }
        ],
    )
    two_day_source = source(
        "2026-08-09T21:00:00+00:00",
        [
            {
                "since": "2026-08-07T21:00:00+00:00",
                "until": "2026-08-09T21:00:00+00:00",
                "result_complete": True,
                "rows": 0,
                "scope": "rolling_authority",
            }
        ],
    )
    telemetry_churn_source = source(
        "2026-08-09T22:00:00+00:00",
        [
            {
                "since": "2026-08-07T21:00:00+00:00",
                "until": "2026-08-08T22:00:00+00:00",
                "result_complete": True,
                "rows": 0,
                "scope": "rolling_authority",
            },
            {
                "since": "2026-08-08T22:00:00+00:00",
                "until": "2026-08-09T22:00:00+00:00",
                "result_complete": True,
                "rows": 0,
                "scope": "rolling_authority",
            },
        ],
    )

    def evidence(
        enumeration_source: Mapping[str, object],
        zero_proofs: Mapping[str, int],
        **telemetry: object,
    ) -> dict[str, object]:
        return with_dual_enumeration({
            "mango_enumeration_complete": True,
            "mango_enumeration_source": enumeration_source,
            "call_keys": [],
            "calls_by_moscow_day": {day: [] for day in zero_proofs},
            "independent_zero_enumerations_by_day": dict(zero_proofs),
            "api_requests": len(enumeration_source["covered_intervals"]),
            "api_rows_total": 0,
            "api_authoritative_rows_total": 0,
            "api_events_total": 0,
            "manifest_end_offset": 0,
            "manifest_snapshot_sha256": snapshot["sha256"],
            **telemetry,
        })

    first = publish_ready_db(
        config,
        {"total": 0},
        capture_evidence=evidence(first_source, {"2026-08-08": 2}),
        manifest_end_offset=0,
    )
    semantic_change = publish_ready_db_if_changed(
        config,
        {"total": 0},
        changed=False,
        capture_evidence=evidence(
            two_day_source,
            {"2026-08-08": 2, "2026-08-09": 2},
        ),
        manifest_end_offset=0,
    )
    telemetry_only = publish_ready_db_if_changed(
        config,
        {"total": 0},
        changed=False,
        capture_evidence=evidence(
            telemetry_churn_source,
            {"2026-08-08": 2, "2026-08-09": 2},
            downloaded=123,
        ),
        manifest_end_offset=0,
    )
    fresh_same_window = evidence(
        telemetry_churn_source,
        {"2026-08-08": 2, "2026-08-09": 2},
        downloaded=123,
    )
    fresh_proof = fresh_same_window["mango_enumeration_source"][
        "dual_enumeration"
    ]
    fresh_proof["proof_run_id"] = "synthetic-proof-run-v2"
    fresh_proof["observed_at"] = "2026-08-12T00:15:00+00:00"
    fresh_proof["proof_sha256"] = calls_runtime._canonical_json_sha256(
        {
            key: value
            for key, value in fresh_proof.items()
            if key != "proof_sha256"
        }
    )
    fresh_result = publish_ready_db_if_changed(
        config,
        {"total": 0},
        changed=False,
        capture_evidence=fresh_same_window,
        manifest_end_offset=0,
    )
    exact_replay = publish_ready_db_if_changed(
        config,
        {"total": 0},
        changed=False,
        capture_evidence=fresh_same_window,
        manifest_end_offset=0,
    )

    assert first["consistency_ok"] is True
    assert semantic_change["consistency_ok"] is True
    assert semantic_change["reused"] is False
    assert first["enumeration_evidence_sha256"] != semantic_change[
        "enumeration_evidence_sha256"
    ]
    assert telemetry_only["reused"] is False
    assert telemetry_only["enumeration_evidence_sha256"] == semantic_change[
        "enumeration_evidence_sha256"
    ]
    assert telemetry_only["capture_proof_sha256"] != semantic_change[
        "capture_proof_sha256"
    ]
    assert fresh_result["reused"] is False
    assert exact_replay["reused"] is True
    assert fresh_result["enumeration_evidence_sha256"] == telemetry_only[
        "enumeration_evidence_sha256"
    ]
    assert fresh_result["capture_proof_sha256"] != telemetry_only[
        "capture_proof_sha256"
    ]


def test_real_empty_capture_caps_proofs_and_ignores_poll_telemetry_churn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(config_for(tmp_path), api_window_hours=12)

    class EmptyClient:
        def __init__(self, **_: object) -> None:
            pass

        def poll_call_history(self, **_: object) -> list[dict[str, object]]:
            return []

    monkeypatch.setattr(calls_runtime, "MangoOfficeClient", EmptyClient)
    monkeypatch.setattr(calls_runtime, "MangoRecordingDownloader", EmptyClient)
    monkeypatch.setenv("MANGO_OFFICE_API_KEY", "present")
    monkeypatch.setenv("MANGO_OFFICE_API_SALT", "present")
    since = datetime(2026, 8, 7, 21, tzinfo=timezone.utc)
    first_until = datetime(2026, 8, 8, 21, tzinfo=timezone.utc)
    second_until = first_until + timedelta(minutes=15)
    third_until = first_until + timedelta(minutes=30)

    first = capture_mango_window(config, since, first_until)
    calls_runtime.write_cursor(config.cursor_path, first_until, first)
    second = capture_mango_window(config, since, second_until)
    calls_runtime.write_cursor(config.cursor_path, second_until, second)
    third = capture_mango_window(config, since, third_until)

    day = "2026-08-08"
    assert first["independent_zero_enumerations_by_day"][day] == 1
    assert second["independent_zero_enumerations_by_day"][day] == 2
    assert third["independent_zero_enumerations_by_day"][day] == 2
    assert first["api_requests"] == 2
    assert second["api_requests"] == third["api_requests"] == 3
    assert first["mango_enumeration_source"]["until"] != second[
        "mango_enumeration_source"
    ]["until"]
    assert calls_runtime.capture_enumeration_evidence_sha256(first) != (
        calls_runtime.capture_enumeration_evidence_sha256(second)
    )
    assert calls_runtime.capture_enumeration_evidence_sha256(second) == (
        calls_runtime.capture_enumeration_evidence_sha256(third)
    )


def test_strict_capture_floors_midday_catch_up_to_moscow_midnight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(
        config_for(tmp_path),
        expected_code_sha="a" * 40,
        expected_previous_host_id="source-mac",
        require_cutover_authority=True,
        strict_ready_provenance=True,
        pending_recording_retry_hours=24,
        api_window_hours=12,
    )

    class EmptyClient:
        def __init__(self, **_: object) -> None:
            pass

        def poll_call_history(self, **_: object) -> list[dict[str, object]]:
            return []

    monkeypatch.setattr(calls_runtime, "MangoOfficeClient", EmptyClient)
    monkeypatch.setattr(
        calls_runtime,
        "MangoRecordingDownloader",
        EmptyClient,
    )
    monkeypatch.setattr(
        calls_runtime,
        "configured_host_id",
        lambda *_args, **_kwargs: "m1-host",
    )
    monkeypatch.setenv("MANGO_OFFICE_API_KEY", "present")
    monkeypatch.setenv("MANGO_OFFICE_API_SALT", "present")
    since = datetime(2026, 8, 5, 12, 34, tzinfo=timezone.utc)
    until = datetime(2026, 8, 9, 21, tzinfo=timezone.utc)

    capture = capture_mango_window(config, since, until)

    assert capture["mango_enumeration_source"]["rolling_since"] == (
        "2026-08-04T21:00:00+00:00"
    )
    assert capture["mango_enumeration_source"]["since"] == (
        "2026-08-04T21:00:00+00:00"
    )
    assert capture["independent_zero_enumerations_by_day"][
        "2026-08-05"
    ] == 2
    calls_runtime.capture_enumeration_exact_sha256(
        capture,
        expected_source_mode="strict_service",
        expected_until=until,
        expected_rolling_since=datetime(
            2026,
            8,
            4,
            21,
            tzinfo=timezone.utc,
        ),
    )


def test_automatic_strict_capture_window_uses_settled_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixed_now = datetime(2026, 8, 11, 12, 30, tzinfo=timezone.utc)

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz: object = None) -> datetime:
            return fixed_now if tz is not None else fixed_now.replace(tzinfo=None)

    monkeypatch.setattr(calls_runtime, "datetime", FixedDateTime)
    strict = replace(
        config_for(tmp_path),
        strict_ready_provenance=True,
        recording_set_stabilization_minutes=15,
    )

    start, end = calls_runtime.resolve_capture_window(
        strict,
        since=None,
        until=None,
    )

    assert end == fixed_now - timedelta(minutes=15)
    assert start == end - timedelta(hours=strict.first_lookback_hours)


def test_explicit_capture_until_is_not_shifted_by_stabilization(
    tmp_path: Path,
) -> None:
    strict = replace(
        config_for(tmp_path),
        strict_ready_provenance=True,
        recording_set_stabilization_minutes=15,
    )
    explicit = "2026-08-11T12:30:00+00:00"

    _start, end = calls_runtime.resolve_capture_window(
        strict,
        since="2026-08-11T11:30:00+00:00",
        until=explicit,
    )

    assert end == datetime(2026, 8, 11, 12, 30, tzinfo=timezone.utc)


def test_strict_capture_rejects_future_explicit_until_before_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixed_now = datetime(2026, 8, 11, 12, 30, tzinfo=timezone.utc)

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz: object = None) -> datetime:
            return fixed_now if tz is not None else fixed_now.replace(tzinfo=None)

    monkeypatch.setattr(calls_runtime, "datetime", FixedDateTime)
    strict = replace(config_for(tmp_path), strict_ready_provenance=True)

    with pytest.raises(ValueError, match="cannot be in the future"):
        calls_runtime.resolve_capture_window(
            strict,
            since="2026-08-11T11:30:00+00:00",
            until="2099-01-01T00:00:00+00:00",
        )


def test_process_a_rejects_future_explicit_until_before_capture_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strict = replace(
        config_for(tmp_path),
        expected_code_sha="a" * 40,
        expected_previous_host_id="source-mac",
        require_cutover_authority=True,
        strict_ready_provenance=True,
    )
    monkeypatch.setattr(
        calls_runtime,
        "cutover_authority_report",
        lambda *_args, **_kwargs: {"ok": True},
    )
    monkeypatch.setattr(
        calls_runtime,
        "disk_preflight",
        lambda *_args, **_kwargs: {"ok": True},
    )
    monkeypatch.setattr(
        calls_runtime,
        "environment_preflight",
        lambda *_args, **_kwargs: {"ok": True},
    )
    attempts: list[bool] = []

    report = run_process_a(
        strict,
        since="2026-08-11T11:30:00+00:00",
        until="2099-01-01T00:00:00+00:00",
        capture_runner=lambda *_args: (
            attempts.append(True) or {"status": "failed"}
        ),
        command_runner=lambda *_args: (
            attempts.append(True) or {"rc": 0}
        ),
    )

    assert report["status"] == "failed"
    assert report["stop_reason"] == "capture_enumeration_evidence_invalid"
    assert attempts == []


def _dual_capture_row(
    call_id: str,
    *,
    minute: int = 0,
    phone: str = "+70000000000",
) -> dict[str, object]:
    return {
        "call_id": call_id,
        "started_at": f"2026-08-11T10:{minute:02d}:00+00:00",
        "ended_at": f"2026-08-11T10:{minute + 1:02d}:00+00:00",
        "direction": "inbound",
        "client_phone": phone,
        "manager_ref": "101",
        "recording_ref": f"recording-{call_id}",
    }


def test_non_strict_duplicate_rows_keep_legacy_last_row_semantics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = replace(config_for(tmp_path), api_window_hours=24)
    first = {
        **_dual_capture_row("call-a", phone="+70000000000"),
        "recording_ref": "recording-z",
    }
    second = {
        **_dual_capture_row("call-a", phone="+71111111111"),
        "recording_ref": "recording-a",
    }

    class LegacyClient:
        def __init__(self, **_: object) -> None:
            pass

        def poll_call_history(self, **_: object) -> list[Mapping[str, object]]:
            return [first, second]

    class Downloader:
        def __init__(self, **_: object) -> None:
            pass

    class Summary:
        failed = 0

        def to_json_dict(self) -> dict[str, int]:
            return {"downloaded": 0, "failed": 0}

    staged: list[TelephonyCallEvent] = []

    def stage(*, events: Sequence[TelephonyCallEvent], **_: object) -> Summary:
        staged.extend(events)
        return Summary()

    monkeypatch.setattr(calls_runtime, "MangoOfficeClient", LegacyClient)
    monkeypatch.setattr(calls_runtime, "MangoRecordingDownloader", Downloader)
    monkeypatch.setattr(calls_runtime, "stage_capture_events", stage)
    monkeypatch.setenv("MANGO_OFFICE_API_KEY", "present")
    monkeypatch.setenv("MANGO_OFFICE_API_SALT", "present")

    report = capture_mango_window(
        config,
        datetime(2026, 8, 10, 21, tzinfo=timezone.utc),
        datetime(2026, 8, 11, 21, tzinfo=timezone.utc),
    )

    assert report["status"] == "ok"
    assert len(staged) == 1
    assert staged[0].client_phone == "+71111111111"
    assert staged[0].recording_refs == ("recording-z", "recording-a")


def _run_synthetic_dual_capture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    primary_rows: Sequence[Mapping[str, object]],
    verification_rows: Sequence[Mapping[str, object]],
    verification_error: bool = False,
    inclusive_end: bool = False,
) -> tuple[Mapping[str, object], list[TelephonyCallEvent], CallsTwoProcessesConfig]:
    config = replace(
        config_for(tmp_path),
        strict_ready_provenance=True,
        expected_code_sha="a" * 40,
        pending_recording_retry_hours=24,
        api_window_hours=24,
    )
    constructed = 0

    class DualClient:
        def __init__(self, **_: object) -> None:
            nonlocal constructed
            self.pass_index = constructed
            constructed += 1

        def poll_call_history(self, **window: object) -> list[Mapping[str, object]]:
            if self.pass_index == 1 and verification_error:
                raise TimeoutError("synthetic second pass timeout")
            rows = primary_rows if self.pass_index == 0 else verification_rows
            since = window["since"]
            until = window["until"]
            assert isinstance(since, datetime) and isinstance(until, datetime)
            return [
                dict(row)
                for row in rows
                if since <= datetime.fromisoformat(str(row["started_at"]))
                and (
                    datetime.fromisoformat(str(row["started_at"])) <= until
                    if inclusive_end
                    else datetime.fromisoformat(str(row["started_at"])) < until
                )
            ]

    class Downloader:
        def __init__(self, **_: object) -> None:
            pass

    class Summary:
        failed = 0

        def to_json_dict(self) -> dict[str, int]:
            return {"downloaded": 0, "failed": 0}

    staged: list[TelephonyCallEvent] = []

    def stage(*, events: Sequence[TelephonyCallEvent], **_: object) -> Summary:
        staged.extend(events)
        return Summary()

    monkeypatch.setattr(calls_runtime, "MangoOfficeClient", DualClient)
    monkeypatch.setattr(calls_runtime, "MangoRecordingDownloader", Downloader)
    monkeypatch.setattr(calls_runtime, "stage_capture_events", stage)
    monkeypatch.setenv("MANGO_OFFICE_API_KEY", "present")
    monkeypatch.setenv("MANGO_OFFICE_API_SALT", "present")
    report = capture_mango_window(
        config,
        datetime(2026, 8, 10, 21, tzinfo=timezone.utc),
        datetime(2026, 8, 11, 21, tzinfo=timezone.utc),
    )
    return report, staged, config


def test_controlled_capture_filters_before_downloader_and_rejects_extra_call(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pipeline = tmp_path / "pipeline"
    config = replace(
        config_for(tmp_path),
        pipeline_root=pipeline,
        strict_ready_provenance=True,
        runtime_authority_mode="isolated_controlled",
        processing_scope="controlled_1_prepare",
        stage_limit=1,
        expected_code_sha="a" * 40,
        expected_active_host_id="m1-host",
        production_cursor_guard_path=(
            tmp_path
            / ".mango_local"
            / "mango_calls_two_processes"
            / "state"
            / "mango_api_freshness.json"
        ),
        publication_root=pipeline / "publication",
        timeline_allowed_root=pipeline / "timeline",
        timeline_db=pipeline / "timeline" / "timeline.sqlite",
    )
    since = datetime(2026, 8, 11, 10, tzinfo=timezone.utc)
    until = datetime(2026, 8, 11, 10, 30, tzinfo=timezone.utc)
    request = ControlledCaptureRequest(
        source_call_id="call-a",
        expected_count=1,
        since=since,
        until=until,
        pipeline_root=pipeline.resolve(strict=False),
        tenant_id="foton",
        code_sha="a" * 40,
        host_id="m1-host",
        request_path=pipeline / "state" / "request.json",
        request_sha256="b" * 64,
    )
    monkeypatch.setattr(
        calls_runtime,
        "controlled_capture_request_for_config",
        lambda _config: request,
    )
    rows: list[Mapping[str, object]] = [_dual_capture_row("call-a")]

    class Client:
        def __init__(self, **_: object) -> None:
            pass

        def poll_call_history(self, **window: object) -> list[Mapping[str, object]]:
            start = window["since"]
            end = window["until"]
            return [
                row
                for row in rows
                if start <= datetime.fromisoformat(str(row["started_at"])) < end
            ]

    downloader_constructed = 0

    class Downloader:
        def __init__(self, **_: object) -> None:
            nonlocal downloader_constructed
            downloader_constructed += 1

    class Summary:
        failed = 0

        def to_json_dict(self) -> dict[str, object]:
            return {
                "total_events": 1,
                "downloaded": 1,
                "reused_existing_file": 0,
                "duplicate_recording": 0,
                "skipped_no_recording": 0,
                "already_manifested": 0,
                "dry_run_download": 0,
                "failed": 0,
                "needs_review_multiple_recordings": 0,
                "manifest_path": str(config.capture_manifest),
                "recordings_dir": str(config.recordings_dir),
                "integrity_quarantined": 0,
            }

    staged: list[str] = []

    def stage(*, events: Sequence[TelephonyCallEvent], **_: object) -> Summary:
        staged.extend(event.provider_call_id for event in events)
        config.capture_manifest.parent.mkdir(parents=True, exist_ok=True)
        config.capture_manifest.write_text("", encoding="utf-8")
        return Summary()

    monkeypatch.setattr(calls_runtime, "MangoOfficeClient", Client)
    monkeypatch.setattr(calls_runtime, "MangoRecordingDownloader", Downloader)
    monkeypatch.setattr(calls_runtime, "stage_capture_events", stage)
    monkeypatch.setenv("MANGO_OFFICE_API_KEY", "present")
    monkeypatch.setenv("MANGO_OFFICE_API_SALT", "present")

    ok = capture_mango_window(
        config, since, until, controlled_request=request
    )
    assert ok["status"] == "ok"
    assert ok["controlled_capture"]["attempted_other"] == 0
    assert staged == ["call-a"]
    assert downloader_constructed == 1

    rows.append(_dual_capture_row("call-b", minute=5))
    config.capture_manifest.unlink()
    staged.clear()
    blocked = capture_mango_window(
        config, since, until, controlled_request=request
    )
    assert blocked["reason"] == "controlled_capture_window_not_exactly_one"
    assert staged == []
    assert downloader_constructed == 1


def test_controlled_narrow_enumeration_requires_exact_authorized_binding(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pipeline = tmp_path / "pipeline"
    since = datetime(2026, 8, 11, 10, tzinfo=timezone.utc)
    until = datetime(2026, 8, 11, 10, 30, tzinfo=timezone.utc)
    request = ControlledCaptureRequest(
        source_call_id="call-a",
        expected_count=1,
        since=since,
        until=until,
        pipeline_root=pipeline.resolve(strict=False),
        tenant_id="foton",
        code_sha="a" * 40,
        host_id="m1-host",
        request_path=pipeline / "state" / "request.json",
        request_sha256="b" * 64,
    )
    config = replace(
        config_for(tmp_path),
        pipeline_root=pipeline,
        strict_ready_provenance=True,
        runtime_authority_mode="isolated_controlled",
        processing_scope="controlled_1_prepare",
        stage_limit=1,
        expected_code_sha="a" * 40,
        expected_active_host_id="m1-host",
    )
    monkeypatch.setattr(
        calls_runtime,
        "controlled_capture_request_for_config",
        lambda _config: request,
    )
    row = _dual_capture_row("call-a")

    class Client:
        def __init__(self, **_: object) -> None:
            pass

        def poll_call_history(self, **window: object) -> list[Mapping[str, object]]:
            start, end = window["since"], window["until"]
            observed = datetime.fromisoformat(str(row["started_at"]))
            return [row] if start <= observed < end else []

    class Summary:
        failed = 0

        def to_json_dict(self) -> dict[str, object]:
            return {"downloaded": 0, "failed": 0}

    monkeypatch.setattr(calls_runtime, "MangoOfficeClient", Client)
    monkeypatch.setattr(calls_runtime, "MangoRecordingDownloader", Client)
    monkeypatch.setattr(calls_runtime, "stage_capture_events", lambda **_: Summary())
    monkeypatch.setenv("MANGO_OFFICE_API_KEY", "present")
    monkeypatch.setenv("MANGO_OFFICE_API_SALT", "present")

    report = capture_mango_window(
        config,
        since,
        until,
        controlled_request=request,
    )
    binding = calls_runtime.controlled_enumeration_binding_for_request(request)
    day = since.astimezone(ZoneInfo("Europe/Moscow")).date()

    assert calls_runtime.validate_capture_enumeration_evidence(report) == [
        "strict_evidence_day_not_covered"
    ]
    assert calls_runtime.validate_capture_enumeration_evidence(
        report,
        controlled_binding=binding,
    ) == []
    assert calls_runtime.capture_enumeration_exact_sha256(
        report,
        expected_source_mode="strict_service",
        expected_until=until,
        expected_rolling_since=since,
        controlled_binding=binding,
    )
    verdict = calls_runtime.build_stage10_verdict(
        day=day,
        enumeration=report,
        capture_entries=[],
        ready_rows=[],
        controlled_binding=binding,
    )
    assert verdict["mango_enumeration_complete"] is True
    assert verdict["closure_ok"] is False

    missing_marker = json.loads(json.dumps(report))
    missing_marker.pop("controlled_capture")
    assert "strict_controlled_enumeration_binding_invalid" in (
        calls_runtime.validate_capture_enumeration_evidence(
            missing_marker,
            controlled_binding=binding,
        )
    )
    wrong_binding = replace(binding, request_sha256="c" * 64)
    assert "strict_controlled_enumeration_binding_invalid" in (
        calls_runtime.validate_capture_enumeration_evidence(
            report,
            controlled_binding=wrong_binding,
        )
    )
    wrong_count = json.loads(json.dumps(report))
    wrong_count["controlled_capture"]["enumerated_other_count"] = 1
    assert "strict_controlled_enumeration_binding_invalid" in (
        calls_runtime.validate_capture_enumeration_evidence(
            wrong_count,
            controlled_binding=binding,
        )
    )


def test_controlled_binding_rejects_full_day_service_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    report, _staged, _config = _run_synthetic_dual_capture(
        monkeypatch,
        tmp_path,
        primary_rows=[_dual_capture_row("call-a")],
        verification_rows=[_dual_capture_row("call-a")],
    )
    binding = calls_runtime.ControlledEnumerationBinding(
        request_sha256="b" * 64,
        source_call_id="call-a",
        since=datetime(2026, 8, 11, 10, tzinfo=timezone.utc),
        until=datetime(2026, 8, 11, 10, 30, tzinfo=timezone.utc),
    )

    assert calls_runtime.validate_capture_enumeration_evidence(report) == []
    assert "strict_controlled_enumeration_binding_invalid" in (
        calls_runtime.validate_capture_enumeration_evidence(
            report,
            controlled_binding=binding,
        )
    )


def test_controlled_ready_manifest_rejects_service_proof_without_marker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    report, _staged, config = _run_synthetic_dual_capture(
        monkeypatch,
        tmp_path,
        primary_rows=[_dual_capture_row("call-a")],
        verification_rows=[_dual_capture_row("call-a")],
    )
    create_ready_call_db(config.working_db)
    with sqlite3.connect(config.working_db) as connection:
        connection.execute(
            "UPDATE call_records SET source_call_id='call-a', "
            "started_at='2026-08-11T10:00:00+00:00'"
        )
    manifest = publish_ready_db(
        config,
        {"total": 1},
        capture_evidence=report,
        manifest_end_offset=report["manifest_end_offset"],
    )
    binding = calls_runtime.ControlledEnumerationBinding(
        request_sha256="b" * 64,
        source_call_id="call-a",
        since=datetime(2026, 8, 11, 10, tzinfo=timezone.utc),
        until=datetime(2026, 8, 11, 10, 30, tzinfo=timezone.utc),
    )

    errors = calls_runtime.validate_ready_manifest_payload(
        manifest,
        controlled_binding=binding,
    )
    assert "controlled_capture_proof_invalid" in errors
    assert "daily_verdict_enumeration_source_invalid" in errors


def test_controlled_full_orchestration_success_replay_and_late_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    owner_local = tmp_path / ".mango_local"
    pipeline = owner_local / "controlled-pilot"
    state_dir = pipeline / "state"
    state_dir.mkdir(parents=True, mode=0o700)
    host_path = state_dir / "host_id"
    host_path.write_text("m1-host\n", encoding="utf-8")
    host_path.chmod(0o600)
    request_path = state_dir / "request.json"
    request_path.write_text("{}", encoding="utf-8")
    request_path.chmod(0o600)
    production_cursor = (
        owner_local
        / "mango_calls_two_processes"
        / "state"
        / "mango_api_freshness.json"
    )
    production_cursor.parent.mkdir(parents=True, mode=0o700)
    production_cursor.write_text('{"sentinel":1}\n', encoding="utf-8")
    production_cursor.chmod(0o600)
    runtime_config_path = state_dir / "runtime.json"
    runtime_config_path.write_text("{}", encoding="utf-8")
    runtime_config_path.chmod(0o600)
    code_sha = calls_runtime.current_git_sha(Path(__file__).resolve().parents[1])
    config = CallsTwoProcessesConfig(
        pipeline_root=pipeline,
        timeline_db=pipeline / "timeline" / "timeline.sqlite",
        timeline_allowed_root=pipeline / "timeline",
        python_executable=Path(sys.executable),
        codex_binary=Path(sys.executable),
        codex_home_root=pipeline / "codex-home",
        tenant_id="foton",
        min_free_gib=1,
        stage_limit=1,
        processing_scope="controlled_1_prepare",
        runtime_authority_mode="isolated_controlled",
        controlled_capture_request_path=request_path,
        controlled_capture_request_sha256="b" * 64,
        production_cursor_guard_path=production_cursor,
        expected_code_sha=code_sha,
        expected_active_host_id="m1-host",
        host_id_path=host_path,
        require_cutover_authority=False,
        strict_ready_provenance=True,
        publication_root=pipeline / "publication",
    )
    request = ControlledCaptureRequest(
        source_call_id="TARGET",
        expected_count=1,
        since=datetime(2026, 8, 10, 10, tzinfo=timezone.utc),
        until=datetime(2026, 8, 10, 10, 30, tzinfo=timezone.utc),
        pipeline_root=pipeline,
        tenant_id="foton",
        code_sha=code_sha,
        host_id="m1-host",
        request_path=request_path,
        request_sha256="b" * 64,
    )
    mode = {"value": "first"}
    timeline_target_rows = {"value": 0}
    timeline_total_rows = {"value": 0}
    timeline_revision = {"value": 0}
    timeline_state = {"value": "present"}

    def capture(*_args: object, **_kwargs: object) -> Mapping[str, object]:
        return {
            "status": "ok",
            "mango_enumeration_complete": True,
            "enumeration_consistency_ok": True,
            "controlled_capture": {"attempted": 1, "attempted_other": 0},
            "downloaded": 0 if mode["value"] == "replay" else 1,
            "manifest_end_offset": 0,
            "manifest_snapshot_sha256": "c" * 64,
        }

    def write_controlled_cursor(
        path: Path, _until: datetime, _capture: Mapping[str, object]
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "manifest_end_offset": 0,
                    "manifest_snapshot_sha256": "c" * 64,
                }
            ),
            encoding="utf-8",
        )

    allowlist_path = state_dir / "controlled" / "allowlist.json"
    allowlist_path.parent.mkdir(parents=True, mode=0o700)
    allowlist_path.write_text("{}", encoding="utf-8")
    allowlist_path.chmod(0o600)
    scope = ControlledCallScope(
        source_call_id="TARGET",
        target_record_id=1,
        source_audio_sha256="d" * 64,
        source_audio_size_bytes=100,
        tenant_id="foton",
        code_sha=code_sha,
        host_id="m1-host",
        allowlist_path=allowlist_path,
        allowlist_sha256="e" * 64,
    )

    def command_runner(
        command: Sequence[str], _env: Mapping[str, str], _cwd: Path
    ) -> Mapping[str, object]:
        if "init-db" in command:
            config.working_db.parent.mkdir(parents=True, exist_ok=True)
            config.working_db.write_bytes(b"synthetic")
        return {"rc": 0, "command": str(command[-1]), "metrics": {}}

    def heavy(
        _config: CallsTwoProcessesConfig, **_kwargs: object
    ) -> Mapping[str, object]:
        with pytest.raises(LockBusy):
            with process_lease(
                _config.pipeline_lock,
                stale_seconds=60,
            ):
                pass
        transitioned = mode["value"] != "replay"
        return {
            "status": "ok",
            "execution_class": (
                "transitioned_to_ready" if transitioned else "idempotent_noop"
            ),
            "stages": [],
            "after": {
                "target": {"started_at": "2026-08-10T10:05:00+00:00"}
            },
        }

    def process_b(_config: CallsTwoProcessesConfig) -> Mapping[str, object]:
        if mode["value"] == "timeline-delete-failure":
            timeline_state["value"] = "absent"
            timeline_target_rows["value"] = 0
            timeline_total_rows["value"] = 0
            timeline_revision["value"] += 1
            raise RuntimeError("synthetic Timeline deletion failure")
        if mode["value"] == "timeline-foreign-commit-failure":
            timeline_total_rows["value"] += 1
            timeline_revision["value"] += 1
            raise RuntimeError("synthetic post-foreign-commit failure")
        writes = mode["value"] != "replay"
        if writes:
            timeline_state["value"] = "present"
            timeline_target_rows["value"] += 1
            timeline_total_rows["value"] += 1
            timeline_revision["value"] += 1
        if mode["value"] == "timeline-commit-failure":
            raise RuntimeError("synthetic post-commit failure")
        return {
            "status": "ok",
            "safety": {"writes_timeline_staging": writes},
            "counters": {"import": {"writes_applied": int(writes)}},
        }

    def preview(_path: Path, _day: object) -> Mapping[str, object]:
        if mode["value"] == "late-failure":
            raise RuntimeError("synthetic preview failure")
        if mode["value"] == "cursor-mutation":
            production_cursor.write_text('{"sentinel":2}\n', encoding="utf-8")
            production_cursor.chmod(0o600)
        return {"ok": True}

    monkeypatch.setattr(
        calls_runtime,
        "controlled_capture_request_for_config",
        lambda _config: request,
    )
    monkeypatch.setattr(calls_runtime, "disk_preflight", lambda _config: {"ok": True})
    monkeypatch.setattr(
        calls_runtime,
        "environment_preflight",
        lambda *_args, **_kwargs: {"ok": True, "runtime_fingerprint": {}},
    )
    monkeypatch.setattr(
        calls_runtime,
        "capture_enumeration_exact_sha256",
        lambda *_args, **_kwargs: "f" * 64,
    )
    monkeypatch.setattr(
        calls_runtime,
        "certify_capture_window",
        lambda _config, value, **_kwargs: value,
    )
    monkeypatch.setattr(calls_runtime, "write_cursor", write_controlled_cursor)
    monkeypatch.setattr(
        calls_runtime,
        "prepare_ingest_inputs",
        lambda *_args, **_kwargs: {
            "audio_files": 0 if mode["value"] == "replay" else 1
        },
    )
    monkeypatch.setattr(calls_runtime, "worker_environment", lambda _config: {})
    monkeypatch.setattr(
        calls_runtime,
        "create_isolated_controlled_allowlist",
        lambda *_args: scope,
    )
    monkeypatch.setattr(calls_runtime, "run_controlled_one", heavy)
    monkeypatch.setattr(calls_runtime, "call_db_counts", lambda _path: {"ready": 1})
    monkeypatch.setattr(
        calls_runtime,
        "publish_ready_db_if_changed",
        lambda *_args, **_kwargs: {"status": "ready", "consistency_ok": True},
    )
    monkeypatch.setattr(
        calls_runtime,
        "controlled_timeline_readback",
        lambda *_args: {"ok": True},
    )
    monkeypatch.setattr(
        calls_runtime,
        "controlled_timeline_effect_snapshot",
        lambda *_args: {
            "state": timeline_state["value"],
            "total_rows": timeline_total_rows["value"],
            "target_rows": timeline_target_rows["value"],
            "mango_rows": timeline_target_rows["value"],
            "quick_check": "ok",
            "logical_sha256": hashlib.sha256(
                str(timeline_revision["value"]).encode("ascii")
            ).hexdigest(),
        },
    )

    def run() -> Mapping[str, object]:
        return calls_runtime.run_controlled_one_from_request(
            config,
            command_runner=command_runner,
            capture_runner=capture,
            process_b_runner=process_b,
            preview_runner=preview,
            runtime_config_path=runtime_config_path,
        )

    first = run()
    assert first["status"] == "ok"
    assert first["attempted"] == 1 and first["processed"] == 1
    assert first["safety"]["production_cursor_unchanged"] is True
    assert first["safety"]["writes_timeline_staging"] is True

    mode["value"] = "replay"
    replay = run()
    assert replay["status"] == "ok"
    assert replay["downloaded"] == 0 and replay["processed"] == 0
    assert replay["safety"]["writes_timeline_staging"] is False

    mode["value"] = "late-failure"
    failed = run()
    assert failed["status"] == "failed"
    assert failed["attempted"] == 1 and failed["processed"] == 1
    assert failed["safety"]["writes_timeline_staging"] is True
    assert failed["safety"]["production_cursor_unchanged"] is True

    mode["value"] = "timeline-commit-failure"
    committed_failed = run()
    assert committed_failed["status"] == "failed"
    assert committed_failed["safety"]["writes_timeline_staging"] is True

    mode["value"] = "timeline-foreign-commit-failure"
    foreign_committed_failed = run()
    assert foreign_committed_failed["status"] == "failed"
    assert foreign_committed_failed["safety"]["writes_timeline_staging"] is True

    mode["value"] = "timeline-delete-failure"
    deleted_failed = run()
    assert deleted_failed["status"] == "failed"
    assert deleted_failed["safety"]["writes_timeline_staging"] is True

    mode["value"] = "cursor-mutation"
    mutated = run()
    assert mutated["status"] == "failed"
    assert mutated["safety"]["production_cursor_unchanged"] is False
    assert mutated["safety"]["production_cursor_written"] is None

    with process_lease(config.controlled_full_lock, stale_seconds=60):
        locked = run()
    assert locked["status"] == "locked"
    assert locked["attempted"] == 0


def test_controlled_full_invalid_config_does_not_write_report(
    tmp_path: Path,
) -> None:
    unsafe_root = tmp_path / "stable_runtime" / "controlled-pilot"
    timeline_root = tmp_path / "timeline"
    timeline_root.mkdir()
    config = CallsTwoProcessesConfig(
        pipeline_root=unsafe_root,
        timeline_allowed_root=timeline_root,
        timeline_db=timeline_root / "timeline.sqlite",
        python_executable=Path(sys.executable),
        codex_binary=Path(sys.executable),
        codex_home_root=tmp_path / "codex-home",
        processing_scope="controlled_1_prepare",
        runtime_authority_mode="isolated_controlled",
        strict_ready_provenance=True,
        require_cutover_authority=False,
        stage_limit=1,
    )

    report = calls_runtime.run_controlled_one_from_request(config)

    assert report["status"] == "failed"
    assert report["diagnostic"]["type"] == "ValueError"
    assert not unsafe_root.exists()


def test_controlled_timeline_snapshot_sees_durable_commit_after_exception(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    owner_local = tmp_path / ".mango_local"
    pipeline = owner_local / "controlled-pilot"
    timeline_root = pipeline / "timeline"
    timeline_root.mkdir(parents=True, mode=0o700)
    timeline_db = timeline_root / "timeline.sqlite"
    with CustomerTimelineSQLiteStore(
        timeline_db,
        allowed_root=timeline_root,
    ):
        pass
    config = replace(
        config_for(tmp_path),
        pipeline_root=pipeline,
        timeline_allowed_root=timeline_root,
        timeline_db=timeline_db,
        processing_scope="controlled_1_prepare",
        runtime_authority_mode="isolated_controlled",
        production_cursor_guard_path=owner_local / "production" / "cursor.json",
    )
    # Exercise the exact finally receipt boundary without touching Mango or ASR.
    def committed_then_failed(*_args: object, **_kwargs: object) -> Mapping[str, object]:
        with sqlite3.connect(timeline_db) as con:
            columns = [
                row[1]
                for row in con.execute("PRAGMA table_info(timeline_events)")
            ]
            values = {name: None for name in columns}
            values.update(
                {
                    "event_id": "evt-controlled",
                    "source_system": "mango_processed_summary",
                    "source_id": "provider:TARGET",
                    "dedupe_key": "dedupe-controlled",
                    "tenant_id": "foton",
                    "event_at": "2026-08-10T10:05:00+00:00",
                    "event_type": "mango_call",
                    "direction": "inbound",
                    "match_status": "unresolved",
                    "importance": 1,
                    "created_at": "2026-08-10T10:06:00+00:00",
                    "record_hash": "a" * 64,
                    "record_json": "{}",
                }
            )
            names = [name for name in columns if values[name] is not None]
            con.execute(
                f"INSERT INTO timeline_events ({','.join(names)}) VALUES ({','.join('?' for _ in names)})",
                tuple(values[name] for name in names),
            )
        raise RuntimeError("synthetic post-commit failure")

    before = calls_runtime.controlled_timeline_effect_snapshot(config, "TARGET")
    assert before["target_rows"] == 0
    with pytest.raises(RuntimeError, match="post-commit"):
        try:
            committed_then_failed()
        finally:
            after = calls_runtime.controlled_timeline_effect_snapshot(config, "TARGET")
            assert after["target_rows"] == 1


def test_controlled_timeline_readback_rejects_wrong_tenant_and_event_type(
    tmp_path: Path,
) -> None:
    config = config_for(tmp_path)
    with CustomerTimelineSQLiteStore(
        config.timeline_db,
        allowed_root=config.timeline_allowed_root,
    ):
        pass
    with sqlite3.connect(config.timeline_db) as con:
        columns = [row[1] for row in con.execute("PRAGMA table_info(timeline_events)")]
        required = {
            "event_id": "wrong",
            "dedupe_key": "wrong",
            "tenant_id": "other",
            "event_type": "email",
            "event_at": "2026-08-10T10:05:00+00:00",
            "source_system": "mango_processed_summary",
            "source_id": "provider:TARGET",
            "direction": "inbound",
            "match_status": "unresolved",
            "importance": 1,
            "created_at": "2026-08-10T10:06:00+00:00",
            "record_hash": "a" * 64,
            "record_json": "{}",
        }
        names = [name for name in columns if name in required]
        con.execute(
            f"INSERT INTO timeline_events ({','.join(names)}) VALUES ({','.join('?' for _ in names)})",
            tuple(required[name] for name in names),
        )
    assert calls_runtime.controlled_timeline_readback(config, "TARGET")["ok"] is False


def test_controlled_timeline_readback_rejects_target_plus_foreign_row(
    tmp_path: Path,
) -> None:
    config = config_for(tmp_path)
    with CustomerTimelineSQLiteStore(
        config.timeline_db,
        allowed_root=config.timeline_allowed_root,
    ):
        pass
    with sqlite3.connect(config.timeline_db) as con:
        columns = [row[1] for row in con.execute("PRAGMA table_info(timeline_events)")]
        for suffix, tenant_id, event_type, source_id in (
            ("target", config.tenant_id, "mango_call", "provider:TARGET"),
            ("foreign", "other", "email", "provider:OTHER"),
        ):
            required = {
                "event_id": suffix,
                "dedupe_key": suffix,
                "tenant_id": tenant_id,
                "event_type": event_type,
                "event_at": "2026-08-10T10:05:00+00:00",
                "source_system": "mango_processed_summary",
                "source_id": source_id,
                "direction": "inbound",
                "match_status": "unresolved",
                "importance": 1,
                "created_at": "2026-08-10T10:06:00+00:00",
                "record_hash": hashlib.sha256(suffix.encode()).hexdigest(),
                "record_json": "{}",
            }
            names = [name for name in columns if name in required]
            con.execute(
                f"INSERT INTO timeline_events ({','.join(names)}) "
                f"VALUES ({','.join('?' for _ in names)})",
                tuple(required[name] for name in names),
            )
    readback = calls_runtime.controlled_timeline_readback(config, "TARGET")
    assert readback["ok"] is False
    assert readback["total_rows"] == 2
    assert readback["target_rows"] == 1


def test_controlled_preview_imports_from_guarded_script_layout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        calls_runtime.sys,
        "path",
        [value for value in sys.path if value not in {"", str(root)}],
    )
    previous_scripts = sys.modules.pop("scripts", None)
    previous_coordinator = sys.modules.pop(
        "scripts.run_mango_calls_publication_coordinator", None
    )
    try:
        with pytest.raises(RuntimeError, match="publication coordinator config"):
            run_controlled_local_previews(
                tmp_path / "missing-runtime.json",
                date(2026, 8, 10),
                "TARGET",
            )
    finally:
        if previous_scripts is not None:
            sys.modules["scripts"] = previous_scripts
        if previous_coordinator is not None:
            sys.modules[
                "scripts.run_mango_calls_publication_coordinator"
            ] = previous_coordinator


def test_strict_dual_enumeration_accepts_reordered_identical_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    row_a = _dual_capture_row("call-a")
    row_b = _dual_capture_row("call-b", minute=2)

    report, staged, _config = _run_synthetic_dual_capture(
        monkeypatch,
        tmp_path,
        primary_rows=[row_a, row_b],
        verification_rows=[row_b, row_a],
    )
    assert report["status"] == "ok"
    assert report["enumeration_consistency_ok"] is True
    assert report["api_requests"] == 3
    assert report["api_authoritative_rows_total"] == 4
    assert report["mango_enumeration_source"]["dual_enumeration"][
        "comparison"
    ]["partition_sha256_different"] is True
    assert [event.provider_call_id for event in staged] == ["call-a", "call-b"]
    assert report["independent_zero_enumerations_by_day"] == {
        "2026-08-11": 0
    }
    calls_runtime.capture_enumeration_exact_sha256(
        report,
        expected_source_mode="strict_service",
        expected_until=datetime(2026, 8, 11, 21, tzinfo=timezone.utc),
        expected_rolling_since=datetime(
            2026, 8, 10, 21, tzinfo=timezone.utc
        ),
    )


def test_strict_dual_duplicate_recordings_are_unioned_deterministically(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first = {**_dual_capture_row("call-a"), "recording_ref": "recording-z"}
    second = {**_dual_capture_row("call-a"), "recording_ref": "recording-a"}

    report, staged, _config = _run_synthetic_dual_capture(
        monkeypatch,
        tmp_path,
        primary_rows=[first, second],
        verification_rows=[second, first],
    )

    assert report["status"] == "ok"
    assert len(staged) == 1
    assert staged[0].recording_refs == ("recording-a", "recording-z")


def test_strict_conflicting_duplicates_fail_before_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first = _dual_capture_row("call-a")
    conflicting = _dual_capture_row("call-a", phone="+71111111111")

    report, staged, config = _run_synthetic_dual_capture(
        monkeypatch,
        tmp_path,
        primary_rows=[first, conflicting],
        verification_rows=[first, conflicting],
    )

    assert report["reason"] == "primary_mango_enumeration_invalid"
    assert staged == []
    assert not config.capture_manifest.exists()


def test_strict_dual_accepts_explained_boundary_duplicates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    row_a = {
        **_dual_capture_row("call-a"),
        "started_at": "2026-08-11T05:00:00+00:00",
        "ended_at": "2026-08-11T05:01:00+00:00",
    }
    row_b = {
        **_dual_capture_row("call-b"),
        "started_at": "2026-08-11T05:00:00+00:00",
        "ended_at": "2026-08-11T05:02:00+00:00",
    }
    report, staged, _config = _run_synthetic_dual_capture(
        monkeypatch,
        tmp_path,
        primary_rows=[row_a, row_b],
        verification_rows=[row_a, row_b],
        inclusive_end=True,
    )

    proof = report["mango_enumeration_source"]["dual_enumeration"]
    assert report["status"] == "ok"
    assert [event.provider_call_id for event in staged] == ["call-a", "call-b"]
    assert [item["proven_duplicate_rows"] for item in proof["passes"]] == [0, 2]
    assert proof["comparison"]["partition_sha256_different"] is True


def test_strict_rolling_overlap_recovers_late_call_before_explicit_since(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = replace(
        config_for(tmp_path),
        strict_ready_provenance=True,
        pending_recording_retry_hours=24,
        api_window_hours=24,
    )
    late_row = {
        **_dual_capture_row("call-late"),
        "started_at": "2026-08-11T10:05:00+00:00",
        "ended_at": "2026-08-11T10:06:00+00:00",
    }
    upstream: list[Mapping[str, object]] = []

    class Client:
        def __init__(self, **_: object) -> None:
            pass

        def poll_call_history(
            self, *, since: datetime, until: datetime
        ) -> list[Mapping[str, object]]:
            return [
                row
                for row in upstream
                if since <= datetime.fromisoformat(str(row["started_at"])) < until
            ]

    class Downloader:
        def __init__(self, **_: object) -> None:
            pass

    class Summary:
        failed = 0

        def to_json_dict(self) -> dict[str, int]:
            return {"downloaded": 0, "failed": 0}

    staged: list[str] = []

    def stage(*, events: Sequence[TelephonyCallEvent], **_: object) -> Summary:
        staged.extend(event.provider_call_id for event in events)
        return Summary()

    monkeypatch.setattr(calls_runtime, "MangoOfficeClient", Client)
    monkeypatch.setattr(calls_runtime, "MangoRecordingDownloader", Downloader)
    monkeypatch.setattr(calls_runtime, "stage_capture_events", stage)
    monkeypatch.setenv("MANGO_OFFICE_API_KEY", "present")
    monkeypatch.setenv("MANGO_OFFICE_API_SALT", "present")

    first = capture_mango_window(
        config,
        datetime(2026, 8, 11, 9, 55, tzinfo=timezone.utc),
        datetime(2026, 8, 11, 10, tzinfo=timezone.utc),
    )
    upstream.append(late_row)
    explicit_since = datetime(2026, 8, 11, 10, 10, tzinfo=timezone.utc)
    second = capture_mango_window(
        config,
        explicit_since,
        datetime(2026, 8, 11, 10, 15, tzinfo=timezone.utc),
    )

    assert first["call_keys"] == []
    assert second["status"] == "ok"
    assert second["call_keys"] == ["call-late"]
    assert datetime.fromisoformat(
        second["mango_enumeration_source"]["rolling_since"]
    ) < datetime.fromisoformat(str(late_row["started_at"])) < explicit_since
    assert staged == ["call-late"]


@pytest.mark.parametrize(
    ("primary_rows", "verification_rows"),
    (
        (
            [_dual_capture_row("call-a")],
            [_dual_capture_row("call-a", phone="+71111111111")],
        ),
        (
            [],
            [_dual_capture_row("call-a")],
        ),
    ),
)
def test_strict_dual_enumeration_mismatch_is_before_any_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    primary_rows: Sequence[Mapping[str, object]],
    verification_rows: Sequence[Mapping[str, object]],
) -> None:
    report, staged, config = _run_synthetic_dual_capture(
        monkeypatch,
        tmp_path,
        primary_rows=primary_rows,
        verification_rows=verification_rows,
    )

    assert report["status"] == "failed"
    assert report["reason"] == "independent_mango_enumeration_mismatch"
    assert report["mango_enumeration_complete"] is False
    assert staged == []
    assert not config.capture_manifest.exists()
    assert not config.cursor_path.exists()


def test_strict_second_enumeration_error_is_before_any_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    row = _dual_capture_row("call-a")
    report, staged, config = _run_synthetic_dual_capture(
        monkeypatch,
        tmp_path,
        primary_rows=[row],
        verification_rows=[row],
        verification_error=True,
    )

    assert report == {
        "status": "failed",
        "reason": "verification_mango_enumeration_failed",
        "mango_enumeration_complete": False,
        "enumeration_consistency_ok": False,
        "api_requests": 1,
        "api_rows_total": 1,
    }
    assert staged == []
    assert not config.capture_manifest.exists()
    assert not config.cursor_path.exists()


def test_strict_mismatch_preserves_existing_cursor_and_manifest_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    row = _dual_capture_row("call-a")
    first, _staged, config = _run_synthetic_dual_capture(
        monkeypatch,
        tmp_path,
        primary_rows=[row],
        verification_rows=[row],
    )
    since = datetime(2026, 8, 10, 21, tzinfo=timezone.utc)
    until = datetime(2026, 8, 11, 21, tzinfo=timezone.utc)
    certified = calls_runtime.certify_capture_window(
        config,
        first,
        requested_since=since,
        requested_until=until,
        enumeration_evidence_sha256=(
            calls_runtime.capture_enumeration_exact_sha256(first)
        ),
    )
    calls_runtime.write_cursor(config.cursor_path, until, certified)
    cursor_before = config.cursor_path.read_bytes()
    manifest_before = config.capture_manifest.read_bytes()

    second, staged, _same_config = _run_synthetic_dual_capture(
        monkeypatch,
        tmp_path,
        primary_rows=[row],
        verification_rows=[],
    )

    assert second["status"] == "failed"
    assert staged == []
    assert config.cursor_path.read_bytes() == cursor_before
    assert config.capture_manifest.read_bytes() == manifest_before


def test_strict_empty_full_day_gets_two_same_run_zero_proofs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    report, staged, _config = _run_synthetic_dual_capture(
        monkeypatch,
        tmp_path,
        primary_rows=[],
        verification_rows=[],
    )

    assert report["status"] == "ok"
    assert report["api_requests"] == 3
    assert report["independent_zero_enumerations_by_day"] == {
        "2026-08-11": 2
    }
    assert staged == []


def test_strict_auxiliary_recovery_is_polled_once_and_excluded_from_proof(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from mango_mvp.productization.capture_staging import (
        CaptureManifestStore,
        ManifestEntry,
    )

    config = replace(
        config_for(tmp_path),
        strict_ready_provenance=True,
        pending_recording_retry_hours=24,
        api_window_hours=24,
    )
    CaptureManifestStore(config.capture_manifest).append(
        ManifestEntry(
            schema_version="capture_manifest_v1",
            created_at="2025-01-01T10:05:00+00:00",
            tenant_id="foton",
            provider="mango",
            event_key="foton:mango:old-call",
            provider_call_id="old-call",
            recording_id="recording-old-call",
            started_at="2025-01-01T10:00:00+00:00",
            ended_at="2025-01-01T10:05:00+00:00",
            direction="inbound",
            client_phone=None,
            manager_ref=None,
            status="failed",
            error="synthetic retry",
        )
    )
    current_row = _dual_capture_row("current-call")
    old_row = {
        **_dual_capture_row("old-call"),
        "started_at": "2025-01-01T10:00:00+00:00",
        "ended_at": "2025-01-01T10:05:00+00:00",
    }
    constructed = 0

    class Client:
        def __init__(self, **_: object) -> None:
            nonlocal constructed
            self.pass_index = constructed
            constructed += 1

        def poll_call_history(
            self, *, since: datetime, until: datetime
        ) -> list[Mapping[str, object]]:
            if since.year == 2025:
                assert self.pass_index == 0
                return [old_row]
            started = datetime.fromisoformat(str(current_row["started_at"]))
            return [current_row] if since <= started <= until else []

    class Downloader:
        def __init__(self, **_: object) -> None:
            pass

    class Summary:
        failed = 0

        def to_json_dict(self) -> dict[str, int]:
            return {"downloaded": 0, "failed": 0}

    staged: list[str] = []

    def stage(*, events: Sequence[TelephonyCallEvent], **_: object) -> Summary:
        staged.extend(event.provider_call_id for event in events)
        return Summary()

    monkeypatch.setattr(calls_runtime, "MangoOfficeClient", Client)
    monkeypatch.setattr(calls_runtime, "MangoRecordingDownloader", Downloader)
    monkeypatch.setattr(calls_runtime, "stage_capture_events", stage)
    monkeypatch.setenv("MANGO_OFFICE_API_KEY", "present")
    monkeypatch.setenv("MANGO_OFFICE_API_SALT", "present")

    report = capture_mango_window(
        config,
        datetime(2026, 8, 10, 21, tzinfo=timezone.utc),
        datetime(2026, 8, 11, 21, tzinfo=timezone.utc),
    )

    source = report["mango_enumeration_source"]
    proof = source["dual_enumeration"]
    assert report["status"] == "ok"
    assert report["api_requests"] == 4
    assert report["api_authoritative_rows_total"] == 2
    assert report["api_auxiliary_rows_total"] == 1
    assert report["call_keys"] == ["current-call"]
    assert [item["call_keys"] for item in proof["passes"]] == [
        ["current-call"],
        ["current-call"],
    ]
    assert set(staged) == {"current-call", "old-call"}
    auxiliary = [
        interval
        for interval in source["covered_intervals"]
        if interval["scope"] == "recovery_auxiliary"
    ]
    assert len(auxiliary) == 1
    assert "authority_pass" not in auxiliary[0]
    assert datetime.fromisoformat(auxiliary[0]["until"]) <= datetime.fromisoformat(
        source["rolling_since"]
    )
    calls_runtime.capture_enumeration_exact_sha256(report)


def test_strict_dual_proof_and_interval_tampering_fail_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    row = _dual_capture_row("call-a")
    report, _staged, _config = _run_synthetic_dual_capture(
        monkeypatch,
        tmp_path,
        primary_rows=[row],
        verification_rows=[row],
    )
    changed_digest = json.loads(json.dumps(report))
    changed_digest["mango_enumeration_source"]["dual_enumeration"][
        "passes"
    ][1]["raw_rows_sha256"] = "f" * 64
    digest_errors = calls_runtime.validate_capture_enumeration_evidence(
        changed_digest
    )
    assert "strict_dual_enumeration_proof_digest_invalid" in digest_errors

    changed_geometry = json.loads(json.dumps(report))
    second_interval = next(
        interval
        for interval in changed_geometry["mango_enumeration_source"][
            "covered_intervals"
        ]
        if interval.get("authority_pass") == 2
    )
    second_interval["since"] = "2026-08-10T22:00:00+00:00"
    geometry_errors = calls_runtime.validate_capture_enumeration_evidence(
        changed_geometry
    )
    assert "strict_enumeration_rolling_geometry_invalid" in geometry_errors
    assert "strict_enumeration_pass_chunks_mismatch" in geometry_errors


@pytest.mark.parametrize(
    ("case", "expected_error"),
    [
        ("passes_required_float", "strict_dual_enumeration_pass_count_invalid"),
        ("passes_completed_float", "strict_dual_enumeration_pass_count_invalid"),
        ("unique_count_bool", "strict_dual_enumeration_unique_count_mismatch"),
        ("official_limit_float", "strict_official_list_proof_invalid"),
        ("official_observed_bool", "strict_official_list_proof_invalid"),
        ("official_schema_v1", "strict_official_list_proof_invalid"),
        ("official_window_mismatch", "strict_official_list_proof_invalid"),
        ("official_extra_field", "strict_official_list_proof_invalid"),
    ],
)
def test_strict_dual_numeric_types_fail_closed(
    case: str,
    expected_error: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    report, _staged, _config = _run_synthetic_dual_capture(
        monkeypatch,
        tmp_path,
        primary_rows=[],
        verification_rows=[],
    )
    changed = json.loads(json.dumps(report))
    proof = changed["mango_enumeration_source"]["dual_enumeration"]
    if case == "passes_required_float":
        proof["passes_required"] = 2.0
    elif case == "passes_completed_float":
        proof["passes_completed"] = 2.0
    elif case == "unique_count_bool":
        for pass_payload in proof["passes"]:
            pass_payload["normalized_unique_count"] = False
    elif case == "official_limit_float":
        proof["official_list"]["page_limit"] = 5000.0
    elif case == "official_schema_v1":
        proof["official_list"]["schema_version"] = "mango_extended_total_pages_v1"
    elif case == "official_window_mismatch":
        request = proof["official_list"]["request"]
        changed_since = datetime.fromisoformat(request["since_utc"]) + timedelta(
            seconds=1
        )
        request["since_utc"] = changed_since.isoformat()
        request["start_date"] = changed_since.astimezone(
            ZoneInfo("Europe/Moscow")
        ).strftime("%d.%m.%Y %H:%M:%S")
    elif case == "official_extra_field":
        proof["official_list"]["unexpected"] = True
    else:
        proof["official_list"]["observed_count"] = False
    official = proof["official_list"]
    official_body = {
        key: value for key, value in official.items() if key != "proof_sha256"
    }
    official["proof_sha256"] = calls_runtime._canonical_json_sha256(
        official_body
    )
    proof_body = {
        key: value for key, value in proof.items() if key != "proof_sha256"
    }
    proof["proof_sha256"] = calls_runtime._canonical_json_sha256(proof_body)

    errors = calls_runtime.validate_capture_enumeration_evidence(changed)
    assert expected_error in errors
    from mango_mvp.productization.mango_calls_service_contract import (
        _dual_source_proof_is_green,
    )

    assert _dual_source_proof_is_green(
        changed["mango_enumeration_source"]
    ) is False


def test_v1_single_pass_certificate_is_anchor_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    row = _dual_capture_row("call-a")
    report, _staged, base_config = _run_synthetic_dual_capture(
        monkeypatch,
        tmp_path,
        primary_rows=[row],
        verification_rows=[row],
    )
    config = replace(base_config, expected_code_sha="a" * 40)
    requested_since = datetime(2026, 8, 10, 21, tzinfo=timezone.utc)
    requested_until = datetime(2026, 8, 11, 21, tzinfo=timezone.utc)
    exact_sha = calls_runtime.capture_enumeration_exact_sha256(report)
    certified = calls_runtime.certify_capture_window(
        config,
        report,
        requested_since=requested_since,
        requested_until=requested_until,
        enumeration_evidence_sha256=exact_sha,
    )
    legacy = json.loads(json.dumps(certified))
    legacy["until"] = requested_until.isoformat()
    legacy.pop("enumeration_consistency_ok", None)
    legacy.pop("api_auxiliary_rows_total", None)
    source = legacy["mango_enumeration_source"]
    source.pop("dual_enumeration", None)
    source.pop("enumeration_consistency_ok", None)
    source["covered_intervals"] = [
        {
            key: value
            for key, value in interval.items()
            if key != "authority_pass"
        }
        for interval in source["covered_intervals"]
        if interval.get("authority_pass") == 1
    ]
    source["requests"] = len(source["covered_intervals"])
    legacy["api_requests"] = source["requests"]
    legacy["api_rows_total"] = 1
    legacy["api_authoritative_rows_total"] = 1
    certificate = dict(legacy["capture_window_certificate"])
    certificate["schema_version"] = (
        calls_runtime.LEGACY_CAPTURE_WINDOW_CERTIFICATE_SCHEMA
    )
    certificate["enumeration_evidence_sha256"] = (
        calls_runtime.capture_enumeration_legacy_exact_sha256(legacy)
    )
    certificate_body = {
        key: value
        for key, value in certificate.items()
        if key != "certificate_sha256"
    }
    certificate["certificate_sha256"] = hashlib.sha256(
        json.dumps(
            certificate_body,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    legacy["capture_window_certificate"] = certificate

    with pytest.raises(RuntimeError, match="certificate is invalid"):
        calls_runtime.verified_capture_window(config, legacy)
    assert calls_runtime.verified_capture_window(
        config,
        legacy,
        allow_pre_dual_anchor=True,
    )[1] == requested_until
    with pytest.raises(RuntimeError, match="strict_dual_enumeration_missing"):
        calls_runtime.capture_enumeration_exact_sha256(legacy)


def test_enumeration_digest_rejects_noncanonical_colliding_day_key(
    tmp_path: Path,
) -> None:
    config = config_for(tmp_path)
    config.working_db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(config.working_db) as connection:
        connection.execute(
            "CREATE TABLE call_records(source_call_id TEXT, started_at TEXT)"
        )
    create_empty_capture_manifest(config)
    snapshot = calls_runtime.capture_manifest_snapshot(
        config.capture_manifest,
        end_offset=0,
    )
    source = {
        "mode": "strict_service",
        "since": "2026-08-07T21:00:00+00:00",
        "rolling_since": "2026-08-07T21:00:00+00:00",
        "until": "2026-08-08T21:00:00+00:00",
        "cursor": "not_applicable_stats_request_result",
        "requests": 1,
        "pages": None,
        "pagination": "not_applicable_stats_request_result",
        "covered_intervals": [
            {
                "since": "2026-08-07T21:00:00+00:00",
                "until": "2026-08-08T21:00:00+00:00",
                "result_complete": True,
                "rows": 0,
                "scope": "rolling_authority",
            }
        ],
    }
    clean = with_dual_enumeration({
        "mango_enumeration_complete": True,
        "mango_enumeration_source": source,
        "call_keys": [],
        "calls_by_moscow_day": {"2026-08-08": []},
        "independent_zero_enumerations_by_day": {"2026-08-08": 2},
        "api_requests": 1,
        "api_rows_total": 0,
        "api_authoritative_rows_total": 0,
        "api_events_total": 0,
        "manifest_end_offset": 0,
        "manifest_snapshot_sha256": snapshot["sha256"],
    })
    publish_ready_db(
        config,
        {"total": 0},
        capture_evidence=clean,
        manifest_end_offset=0,
    )
    malformed = {
        **clean,
        "calls_by_moscow_day": {
            "2026-08-08": [],
            " 2026-08-08 ": ["hidden-call"],
        },
    }

    with pytest.raises(RuntimeError, match="day_key_not_canonical"):
        publish_ready_db_if_changed(
            config,
            {"total": 0},
            changed=False,
            capture_evidence=malformed,
            manifest_end_offset=0,
        )


def test_positive_api_rows_can_never_close_an_empty_strict_day() -> None:
    source = {
        "mode": "strict_service",
        "since": "2026-08-07T21:00:00+00:00",
        "rolling_since": "2026-08-07T21:00:00+00:00",
        "until": "2026-08-08T21:00:00+00:00",
        "cursor": "not_applicable_stats_request_result",
        "requests": 1,
        "pages": None,
        "pagination": "not_applicable_stats_request_result",
        "covered_intervals": [
            {
                "since": "2026-08-07T21:00:00+00:00",
                "until": "2026-08-08T21:00:00+00:00",
                "result_complete": True,
                "rows": 1,
                "scope": "rolling_authority",
            }
        ],
    }
    malformed = {
        "mango_enumeration_complete": True,
        "mango_enumeration_source": source,
        "call_keys": [],
        "calls_by_moscow_day": {"2026-08-08": []},
        "independent_zero_enumerations_by_day": {"2026-08-08": 2},
        "api_requests": 1,
        "api_rows_total": 1,
        "api_authoritative_rows_total": 1,
        "api_events_total": 0,
    }

    with pytest.raises(RuntimeError, match="strict_api_rows_without_calls"):
        calls_runtime.capture_enumeration_evidence_sha256(malformed)
    verdict = calls_runtime.build_stage10_verdict(
        day=datetime(2026, 8, 8, tzinfo=timezone.utc).date(),
        enumeration=malformed,
        capture_entries=[],
        ready_rows=[],
    )
    assert verdict["consistency_ok"] is False
    assert verdict["closure_ok"] is False


@pytest.mark.parametrize("entrypoint", ("capture", "process_a"))
@pytest.mark.parametrize(
    "malformation",
    (
        "noncanonical_day",
        "string_call_collection",
        "invalid_source",
        "rolling_coverage_gap",
        "until_mismatch",
    ),
)
def test_runtime_rejects_malformed_enumeration_before_cursor_or_workers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entrypoint: str,
    malformation: str,
) -> None:
    config = replace(
        config_for(tmp_path),
        min_free_gib=1,
        expected_code_sha="a" * 40,
        expected_previous_host_id="source-mac",
        require_cutover_authority=True,
        strict_ready_provenance=True,
        pending_recording_retry_hours=24,
    )
    monkeypatch.setattr(
        calls_runtime,
        "cutover_authority_report",
        lambda *_args, **_kwargs: {"ok": True},
    )
    monkeypatch.setattr(
        calls_runtime,
        "disk_preflight",
        lambda *_args, **_kwargs: {"ok": True},
    )
    monkeypatch.setattr(
        calls_runtime,
        "environment_preflight",
        lambda *_args, **_kwargs: {
            "ok": True,
            "codex_network_ok": True,
        },
    )
    source = {
        "mode": "strict_service",
        "since": "2026-08-07T21:00:00+00:00",
        "rolling_since": "2026-08-07T21:00:00+00:00",
        "until": "2026-08-08T21:00:00+00:00",
        "cursor": "not_applicable_stats_request_result",
        "requests": 1,
        "pages": None,
        "pagination": "not_applicable_stats_request_result",
        "covered_intervals": [
            {
                "since": "2026-08-07T21:00:00+00:00",
                "until": "2026-08-08T21:00:00+00:00",
                "result_complete": True,
                "rows": 0,
                "scope": "rolling_authority",
            }
        ],
    }
    malformed: dict[str, object] = {
        "status": "ok",
        "mango_enumeration_complete": True,
        "mango_enumeration_source": source,
        "call_keys": [],
        "calls_by_moscow_day": {"2026-08-08": []},
        "independent_zero_enumerations_by_day": {"2026-08-08": 2},
        "api_requests": 1,
        "api_rows_total": 0,
        "api_authoritative_rows_total": 0,
        "api_events_total": 0,
    }
    if malformation == "noncanonical_day":
        malformed["calls_by_moscow_day"] = {
            "2026-08-08": [],
            " 2026-08-08 ": ["hidden-call"],
        }
    elif malformation == "string_call_collection":
        malformed.update(
            call_keys=["hidden-call"],
            calls_by_moscow_day={"2026-08-08": "hidden-call"},
            independent_zero_enumerations_by_day={"2026-08-08": 0},
            api_rows_total=1,
            api_events_total=1,
        )
        source["covered_intervals"][0]["rows"] = 1
    elif malformation == "invalid_source":
        malformed["mango_enumeration_source"] = {}
    elif malformation == "rolling_coverage_gap":
        malformed.update(
            call_keys=[],
            calls_by_moscow_day={},
            independent_zero_enumerations_by_day={},
        )
        source["covered_intervals"][0]["until"] = (
            "2026-08-07T22:00:00+00:00"
        )
    else:
        malformed.update(
            call_keys=[],
            calls_by_moscow_day={},
            independent_zero_enumerations_by_day={},
        )
        source["until"] = "2026-08-07T22:00:00+00:00"
        source["covered_intervals"][0]["until"] = source["until"]
    capture_runner = lambda *_args: malformed
    exact_window = {
        "since": "2026-08-07T21:00:00+00:00",
        "until": "2026-08-08T21:00:00+00:00",
    }

    if entrypoint == "capture":
        result = run_capture(
            config,
            capture_runner=capture_runner,
            **exact_window,
        )
    else:
        result = run_process_a(
            config,
            capture_runner=capture_runner,
            command_runner=lambda *_args: pytest.fail(
                "worker command must not run"
            ),
            **exact_window,
        )

    assert result["status"] == "failed"
    assert result["stop_reason"] == "capture_enumeration_evidence_invalid"
    assert read_json(config.cursor_path) == {}
    assert not config.working_db.exists()
    assert not config.ready_db.exists()


@pytest.mark.parametrize("entrypoint", ("capture", "process_a"))
def test_runtime_rejects_enumeration_that_omits_requested_rolling_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entrypoint: str,
) -> None:
    config = replace(
        config_for(tmp_path),
        min_free_gib=1,
        expected_code_sha="a" * 40,
        expected_previous_host_id="source-mac",
        require_cutover_authority=True,
        strict_ready_provenance=True,
        pending_recording_retry_hours=24,
    )
    monkeypatch.setattr(
        calls_runtime,
        "cutover_authority_report",
        lambda *_args, **_kwargs: {"ok": True},
    )
    monkeypatch.setattr(
        calls_runtime,
        "disk_preflight",
        lambda *_args, **_kwargs: {"ok": True},
    )
    monkeypatch.setattr(
        calls_runtime,
        "environment_preflight",
        lambda *_args, **_kwargs: {
            "ok": True,
            "codex_network_ok": True,
        },
    )
    omitted_start = "2026-08-08T21:00:00+00:00"
    requested_until = "2026-08-09T21:00:00+00:00"
    incomplete = {
        "status": "ok",
        "mango_enumeration_complete": True,
        "mango_enumeration_source": {
            "mode": "strict_service",
            "since": omitted_start,
            "rolling_since": omitted_start,
            "until": requested_until,
            "cursor": "not_applicable_stats_request_result",
            "requests": 1,
            "pages": None,
            "pagination": "not_applicable_stats_request_result",
            "covered_intervals": [
                {
                    "since": omitted_start,
                    "until": requested_until,
                    "result_complete": True,
                    "rows": 0,
                    "scope": "rolling_authority",
                }
            ],
        },
        "call_keys": [],
        "calls_by_moscow_day": {"2026-08-09": []},
        "independent_zero_enumerations_by_day": {"2026-08-09": 2},
        "api_requests": 1,
        "api_rows_total": 0,
        "api_authoritative_rows_total": 0,
        "api_events_total": 0,
    }
    capture_runner = lambda *_args: incomplete
    exact_window = {
        "since": "2026-08-07T21:00:00+00:00",
        "until": requested_until,
    }

    if entrypoint == "capture":
        result = run_capture(
            config,
            capture_runner=capture_runner,
            **exact_window,
        )
    else:
        result = run_process_a(
            config,
            capture_runner=capture_runner,
            command_runner=lambda *_args: pytest.fail(
                "worker command must not run"
            ),
            **exact_window,
        )

    assert result["status"] == "failed"
    assert result["stop_reason"] == "capture_enumeration_evidence_invalid"
    assert read_json(config.cursor_path) == {}
    assert not config.working_db.exists()
    assert not config.ready_db.exists()


def test_pipeline_rejects_shortened_cursor_after_exact_capture_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(
        config_for(tmp_path),
        min_free_gib=1,
        expected_code_sha="a" * 40,
        expected_previous_host_id="source-mac",
        require_cutover_authority=True,
        strict_ready_provenance=True,
        pending_recording_retry_hours=24,
    )
    monkeypatch.setattr(
        calls_runtime,
        "cutover_authority_report",
        lambda *_args, **_kwargs: {"ok": True},
    )
    monkeypatch.setattr(
        calls_runtime,
        "disk_preflight",
        lambda *_args, **_kwargs: {"ok": True},
    )
    monkeypatch.setattr(
        calls_runtime,
        "environment_preflight",
        lambda *_args, **_kwargs: {
            "ok": True,
            "codex_network_ok": True,
        },
    )
    create_empty_capture_manifest(config)
    snapshot = calls_runtime.capture_manifest_snapshot(
        config.capture_manifest,
        end_offset=0,
    )
    requested_since = "2026-08-07T21:00:00+00:00"
    requested_until = "2026-08-09T21:00:00+00:00"
    requested_until_input = "2026-08-09T21:00:00.654321+00:00"
    complete = with_dual_enumeration({
        "status": "ok",
        "mango_enumeration_complete": True,
        "mango_enumeration_source": {
            "mode": "strict_service",
            "since": requested_since,
            "rolling_since": requested_since,
            "until": requested_until,
            "cursor": "not_applicable_stats_request_result",
            "requests": 1,
            "pages": None,
            "pagination": "not_applicable_stats_request_result",
            "covered_intervals": [
                {
                    "since": requested_since,
                    "until": requested_until,
                    "result_complete": True,
                    "rows": 0,
                    "scope": "rolling_authority",
                }
            ],
        },
        "call_keys": [],
        "calls_by_moscow_day": {},
        "independent_zero_enumerations_by_day": {
            "2026-08-08": 2,
            "2026-08-09": 2,
        },
        "api_requests": 1,
        "api_rows_total": 0,
        "api_authoritative_rows_total": 0,
        "api_events_total": 0,
        "manifest_end_offset": 0,
        "manifest_snapshot_sha256": snapshot["sha256"],
    })
    captured = run_capture(
        config,
        since=requested_since,
        until=requested_until_input,
        capture_runner=lambda *_args: complete,
    )
    assert captured["status"] == "ok"
    cursor = dict(read_json(config.cursor_path))
    assert cursor["capture_window_certificate"]["requested_since"] == (
        requested_since
    )
    assert cursor["capture_window_certificate"]["requested_until"] == (
        requested_until
    )
    with pytest.raises(RuntimeError, match="certificate is invalid"):
        calls_runtime.verified_capture_window(
            replace(config, tenant_id="other-tenant"),
            cursor,
        )
    with pytest.raises(RuntimeError, match="certificate is invalid"):
        calls_runtime.verified_capture_window(
            replace(config, base_url="https://other.invalid"),
            cursor,
        )
    reached_after_validation: list[Mapping[str, object]] = []

    def stop_after_validation(*_args: object, **_kwargs: object) -> object:
        reached_after_validation.append(dict(_kwargs))
        raise LookupError("validation_reached")

    with monkeypatch.context() as local_patch:
        local_patch.setattr(
            calls_runtime,
            "prepare_ingest_inputs",
            stop_after_validation,
        )
        manual_resume = run_process_a(
            config,
            skip_capture=True,
            skip_workers=True,
        )
    assert reached_after_validation == [
        {
            "manifest_end_offset": 0,
            "expected_manifest_sha256": snapshot["sha256"],
        }
    ]
    assert manual_resume["stop_reason"] == "process_a_exception:LookupError"

    for mutation in ("until", "zero_proof", "missing_certificate"):
        tampered = json.loads(json.dumps(cursor))
        if mutation == "until":
            tampered["until"] = "2026-08-09T20:00:00+00:00"
        elif mutation == "zero_proof":
            tampered["independent_zero_enumerations_by_day"][
                "2026-08-08"
            ] = 1
        else:
            tampered.pop("capture_window_certificate")
        calls_runtime.write_json(config.cursor_path, tampered)
        capture_attempts: list[bool] = []
        rejected_capture = run_capture(
            config,
            until="2026-08-10T21:00:00+00:00",
            capture_runner=lambda *_args: (
                capture_attempts.append(True)
                or {"status": "failed", "reason": "must_not_run"}
            ),
        )
        assert rejected_capture["status"] == "failed"
        assert rejected_capture["stop_reason"] == (
            "capture_enumeration_evidence_invalid"
        )
        assert capture_attempts == []
    calls_runtime.write_json(config.cursor_path, cursor)

    shortened = json.loads(json.dumps(cursor))
    omitted_start = "2026-08-08T21:00:00+00:00"
    shortened["mango_enumeration_source"]["since"] = omitted_start
    shortened["mango_enumeration_source"]["rolling_since"] = omitted_start
    shortened["mango_enumeration_source"]["covered_intervals"] = [
        {
            "since": omitted_start,
            "until": requested_until,
            "result_complete": True,
            "rows": 0,
            "scope": "rolling_authority",
        }
    ]
    shortened["independent_zero_enumerations_by_day"] = {
        "2026-08-09": 2
    }
    calls_runtime.write_json(config.cursor_path, shortened)

    result = run_pipeline(
        config,
        command_runner=lambda *_args: pytest.fail("worker must not run"),
        process_b_runner=lambda *_args: pytest.fail(
            "process B must not run"
        ),
    )

    assert result["status"] == "failed"
    assert result["process_a"]["stop_reason"] == (
        "capture_enumeration_evidence_invalid"
    )
    assert not config.working_db.exists()
    assert not config.ready_db.exists()


def test_legacy_transfer_cursor_requires_continuous_explicit_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(
        config_for(tmp_path),
        expected_code_sha="a" * 40,
        expected_previous_host_id="source-mac",
        require_cutover_authority=True,
        strict_ready_provenance=True,
        pending_recording_retry_hours=24,
    )
    monkeypatch.setattr(
        calls_runtime,
        "cutover_authority_report",
        lambda *_args, **_kwargs: {"ok": True},
    )
    monkeypatch.setattr(
        calls_runtime,
        "disk_preflight",
        lambda *_args, **_kwargs: {"ok": True},
    )
    monkeypatch.setattr(
        calls_runtime,
        "environment_preflight",
        lambda *_args, **_kwargs: {
            "ok": True,
            "codex_network_ok": True,
        },
    )
    legacy_until = "2026-08-08T21:00:00+00:00"
    create_legacy_transfer_cursor(
        config,
        until=legacy_until,
        zero_proofs={"2026-08-08": 999},
    )
    legacy_bytes = config.cursor_path.read_bytes()

    no_explicit_attempts: list[bool] = []
    no_explicit = run_capture(
        config,
        until="2026-08-09T21:00:00+00:00",
        capture_runner=lambda *_args: (
            no_explicit_attempts.append(True)
            or {"status": "failed"}
        ),
    )
    assert no_explicit["status"] == "failed"
    assert no_explicit["stop_reason"] == (
        "capture_enumeration_evidence_invalid"
    )
    assert no_explicit_attempts == []

    for explicit_since, explicit_until in (
        (
            None,
            "2026-08-09T20:59:59+00:00",
        ),
        (
            "2026-08-07T21:00:00+00:00",
            "2026-08-08T20:59:59+00:00",
        ),
        (
            "2026-08-11T21:00:00+00:00",
            "2026-08-12T21:00:00+00:00",
        ),
    ):
        attempts: list[bool] = []
        rejected = run_capture(
            config,
            since=explicit_since,
            until=explicit_until,
            capture_runner=lambda *_args: (
                attempts.append(True) or {"status": "failed"}
            ),
        )
        assert rejected["status"] == "failed"
        assert rejected["stop_reason"] == (
            "capture_enumeration_evidence_invalid"
        )
        assert attempts == []

    failed = run_capture(
        config,
        since="2026-08-07T21:00:00+00:00",
        until="2026-08-09T21:00:00+00:00",
        capture_runner=lambda runtime_config, *_args: {
            "status": "failed",
            "migration_mode": runtime_config.legacy_cursor_migration_mode,
        },
    )
    assert failed["status"] == "failed"
    assert failed["counters"]["capture"]["migration_mode"] is True
    assert config.cursor_path.read_bytes() == legacy_bytes

    skipped = run_process_a(
        config,
        since="2026-08-07T21:00:00+00:00",
        until="2026-08-09T21:00:00+00:00",
        skip_capture=True,
        skip_workers=True,
        command_runner=lambda *_args: pytest.fail("worker must not run"),
    )
    assert skipped["status"] == "failed"
    assert skipped["stop_reason"] == "capture_enumeration_evidence_invalid"
    assert config.cursor_path.read_bytes() == legacy_bytes


def test_legacy_transfer_cursor_migrates_without_inheriting_zero_and_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(
        config_for(tmp_path),
        expected_code_sha="a" * 40,
        expected_previous_host_id="source-mac",
        require_cutover_authority=True,
        strict_ready_provenance=True,
        pending_recording_retry_hours=24,
        api_window_hours=12,
    )
    monkeypatch.setattr(
        calls_runtime,
        "cutover_authority_report",
        lambda *_args, **_kwargs: {"ok": True},
    )
    monkeypatch.setattr(
        calls_runtime,
        "disk_preflight",
        lambda *_args, **_kwargs: {"ok": True},
    )
    monkeypatch.setattr(
        calls_runtime,
        "environment_preflight",
        lambda *_args, **_kwargs: {
            "ok": True,
            "codex_network_ok": True,
        },
    )
    monkeypatch.setattr(
        calls_runtime,
        "configured_host_id",
        lambda *_args, **_kwargs: "m1-host",
    )

    class EmptyClient:
        def __init__(self, **_: object) -> None:
            pass

        def poll_call_history(self, **_: object) -> list[dict[str, object]]:
            return []

    monkeypatch.setattr(calls_runtime, "MangoOfficeClient", EmptyClient)
    monkeypatch.setattr(
        calls_runtime,
        "MangoRecordingDownloader",
        EmptyClient,
    )
    monkeypatch.setenv("MANGO_OFFICE_API_KEY", "present")
    monkeypatch.setenv("MANGO_OFFICE_API_SALT", "present")
    create_legacy_transfer_cursor(
        config,
        until="2026-08-08T21:00:00+00:00",
        zero_proofs={"2026-08-08": 999},
    )

    migrated = run_capture(
        config,
        since="2026-08-07T21:00:00+00:00",
        until="2026-08-09T21:00:00+00:00",
    )

    assert migrated["status"] == "ok"
    cursor = dict(read_json(config.cursor_path))
    assert cursor["independent_zero_enumerations_by_day"] == {
        "2026-08-08": 2,
        "2026-08-09": 2,
    }
    assert cursor["capture_window_certificate"]["schema_version"] == (
        "mango_capture_window_certificate_v2"
    )
    assert calls_runtime.verified_capture_window(config, cursor)[1] == (
        datetime(2026, 8, 9, 21, tzinfo=timezone.utc)
    )
    rotated_code_config = replace(config, expected_code_sha="b" * 40)
    assert calls_runtime.verified_capture_window(
        rotated_code_config,
        cursor,
    )[1] == datetime(2026, 8, 9, 21, tzinfo=timezone.utc)
    manifest_bytes = config.capture_manifest.read_bytes()
    changed_manifest = manifest_bytes.replace(
        b"transferred-call",
        b"transferred-fall",
        1,
    )
    assert changed_manifest != manifest_bytes
    config.capture_manifest.write_bytes(changed_manifest)
    with pytest.raises(RuntimeError, match="manifest prefix changed"):
        calls_runtime.verified_capture_window(config, cursor)
    manifest_attempts: list[bool] = []
    rejected_manifest = run_capture(
        config,
        since="2026-08-07T21:00:00+00:00",
        until="2026-08-10T21:00:00+00:00",
        capture_runner=lambda *_args: (
            manifest_attempts.append(True) or {"status": "failed"}
        ),
    )
    assert rejected_manifest["status"] == "failed"
    assert rejected_manifest["stop_reason"] == (
        "capture_enumeration_evidence_invalid"
    )
    assert manifest_attempts == []
    config.capture_manifest.write_bytes(manifest_bytes)
    split_intervals = json.loads(json.dumps(cursor))
    original_interval = split_intervals["mango_enumeration_source"][
        "covered_intervals"
    ][0]
    interval_start = datetime.fromisoformat(original_interval["since"])
    interval_end = datetime.fromisoformat(original_interval["until"])
    interval_middle = interval_start + (interval_end - interval_start) / 2
    split_intervals["mango_enumeration_source"]["covered_intervals"][0:1] = [
        {**original_interval, "until": interval_middle.isoformat()},
        {**original_interval, "since": interval_middle.isoformat()},
    ]
    split_intervals["mango_enumeration_source"]["requests"] += 1
    split_intervals["api_requests"] += 1
    with pytest.raises(
        RuntimeError,
        match="pass_chunks_mismatch|certificate evidence changed",
    ):
        calls_runtime.verified_capture_window(config, split_intervals)
    missing_cursor_until = json.loads(json.dumps(cursor))
    missing_cursor_until.pop("until")
    with pytest.raises(RuntimeError, match="cursor until differs"):
        calls_runtime.verified_capture_window(config, missing_cursor_until)

    certified_bytes = config.cursor_path.read_bytes()
    for explicit_since, explicit_until in (
        (
            None,
            "2026-08-09T20:59:59+00:00",
        ),
        (
            "2026-08-07T21:00:00+00:00",
            "2026-08-09T20:59:59+00:00",
        ),
        (
            "2026-08-12T21:00:00+00:00",
            "2026-08-13T21:00:00+00:00",
        ),
    ):
        attempts: list[bool] = []
        rejected = run_capture(
            config,
            since=explicit_since,
            until=explicit_until,
            capture_runner=lambda *_args: (
                attempts.append(True) or {"status": "failed"}
            ),
        )
        assert rejected["status"] == "failed"
        assert rejected["stop_reason"] == (
            "capture_enumeration_evidence_invalid"
        )
        assert attempts == []
        assert config.cursor_path.read_bytes() == certified_bytes

    for mutation in ("null_certificate", "removed_certificate"):
        tampered = json.loads(json.dumps(cursor))
        if mutation == "null_certificate":
            tampered["capture_window_certificate"] = None
        else:
            tampered.pop("capture_window_certificate")
        write_json(config.cursor_path, tampered)
        attempts: list[bool] = []
        rejected = run_capture(
            config,
            since="2026-08-07T21:00:00+00:00",
            until="2026-08-10T21:00:00+00:00",
            capture_runner=lambda *_args: (
                attempts.append(True) or {"status": "failed"}
            ),
        )
        assert rejected["status"] == "failed"
        assert rejected["stop_reason"] == (
            "capture_enumeration_evidence_invalid"
        )
        assert attempts == []
    write_json(config.cursor_path, cursor)

    reached_after_validation: list[Mapping[str, object]] = []

    def stop_after_validation(*_args: object, **kwargs: object) -> object:
        reached_after_validation.append(dict(kwargs))
        raise LookupError("validation_reached")

    monkeypatch.setattr(
        calls_runtime,
        "prepare_ingest_inputs",
        stop_after_validation,
    )
    resumed = run_process_a(
        config,
        skip_capture=True,
        skip_workers=True,
    )
    assert reached_after_validation == [
        {
            "manifest_end_offset": cursor["manifest_end_offset"],
            "expected_manifest_sha256": cursor[
                "manifest_snapshot_sha256"
            ],
        }
    ]
    assert resumed["stop_reason"] == "process_a_exception:LookupError"


def test_strict_capture_detects_cursor_change_before_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(
        config_for(tmp_path),
        expected_code_sha="a" * 40,
        expected_previous_host_id="source-mac",
        require_cutover_authority=True,
        strict_ready_provenance=True,
        pending_recording_retry_hours=24,
        legacy_cursor_migration_mode=True,
    )
    legacy = dict(
        create_legacy_transfer_cursor(
            config,
            until="2026-08-08T21:00:00+00:00",
            zero_proofs={"2026-08-08": 2},
        )
    )
    monkeypatch.setattr(
        calls_runtime,
        "configured_host_id",
        lambda *_args, **_kwargs: "m1-host",
    )
    changed = False

    class SwappingClient:
        def __init__(self, **_: object) -> None:
            pass

        def poll_call_history(self, **_: object) -> list[dict[str, object]]:
            nonlocal changed
            if not changed:
                changed = True
                write_json(
                    config.cursor_path,
                    {**legacy, "updated_at": "2026-08-08T21:00:01+00:00"},
                )
            return []

    monkeypatch.setattr(calls_runtime, "MangoOfficeClient", SwappingClient)
    monkeypatch.setattr(
        calls_runtime,
        "MangoRecordingDownloader",
        SwappingClient,
    )
    monkeypatch.setattr(
        calls_runtime,
        "stage_capture_events",
        lambda **_kwargs: pytest.fail(
            "cursor race must stop before staging or downloads"
        ),
    )
    monkeypatch.setenv("MANGO_OFFICE_API_KEY", "present")
    monkeypatch.setenv("MANGO_OFFICE_API_SALT", "present")

    with pytest.raises(RuntimeError, match="cursor changed during"):
        capture_mango_window(
            config,
            datetime(2026, 8, 7, 21, tzinfo=timezone.utc),
            datetime(2026, 8, 9, 21, tzinfo=timezone.utc),
        )
    assert changed is True


def test_process_a_capture_uses_capture_lock(
    tmp_path: Path,
) -> None:
    config = config_for(tmp_path)
    attempts: list[bool] = []

    with process_lease(
        config.capture_lock,
        stale_seconds=config.stale_lock_seconds,
    ):
        result = run_process_a(
            config,
            since="2026-08-08T00:00:00+00:00",
            until="2026-08-08T01:00:00+00:00",
            skip_workers=True,
            capture_runner=lambda *_args: (
                attempts.append(True) or {"status": "failed"}
            ),
        )

    assert result["status"] == "locked"
    assert result["stop_reason"] == "process_a_locked"
    assert attempts == []


@pytest.mark.parametrize(
    "schema_case",
    ("id_only", "missing_runtime_column", "missing_critical_constraints"),
)
def test_runtime_rejects_incomplete_working_schema_before_api_or_workers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    schema_case: str,
) -> None:
    config = replace(config_for(tmp_path), min_free_gib=1)
    if schema_case == "id_only":
        config.working_db.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(config.working_db) as connection:
            connection.execute("CREATE TABLE call_records(id INTEGER PRIMARY KEY)")
    elif schema_case == "missing_runtime_column":
        create_ready_call_db(config.working_db)
        with sqlite3.connect(config.working_db) as connection:
            connection.execute("ALTER TABLE call_records DROP COLUMN dead_letter_stage")
    else:
        config.working_db.parent.mkdir(parents=True, exist_ok=True)
        columns = ", ".join(
            f'"{column}" TEXT'
            for column in sorted(
                calls_runtime.REQUIRED_RUNTIME_CALL_RECORD_COLUMNS
            )
        )
        with sqlite3.connect(config.working_db) as connection:
            connection.execute(f"CREATE TABLE call_records ({columns})")
            connection.executemany(
                "INSERT INTO call_records(id, source_file) VALUES (?, ?)",
                (("duplicate", "same"), ("duplicate", "same")),
            )
    monkeypatch.setattr(
        calls_runtime,
        "cutover_authority_report",
        lambda *_args, **_kwargs: {"ok": True},
    )

    result = run_process_a(
        config,
        capture_runner=lambda *_args: pytest.fail("Mango API must not run"),
        command_runner=lambda *_args: pytest.fail("worker must not run"),
    )

    assert result["status"] == "failed"
    assert result["stop_reason"] == "working_db_invalid"
    assert read_json(config.cursor_path) == {}
    assert config.working_db.is_file()
    assert not config.ready_db.exists()


def test_capture_does_not_trust_or_wait_for_invalid_working_db(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(
        config_for(tmp_path),
        expected_code_sha="a" * 40,
        expected_previous_host_id="source-mac",
        require_cutover_authority=True,
        strict_ready_provenance=True,
        pending_recording_retry_hours=24,
    )
    config.working_db.parent.mkdir(parents=True, exist_ok=True)
    columns = ", ".join(
        f'"{column}" TEXT'
        for column in sorted(calls_runtime.REQUIRED_RUNTIME_CALL_RECORD_COLUMNS)
    )
    with sqlite3.connect(config.working_db) as connection:
        connection.execute(f"CREATE TABLE call_records ({columns})")
        connection.execute(
            "INSERT INTO call_records(id, source_file, source_call_id) "
            "VALUES ('not-a-primary-key', 'untrusted', 'must-not-dedupe')"
        )
    config.ready_db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(config.ready_db) as connection:
        connection.execute("CREATE TABLE call_records(source_call_id TEXT)")
        connection.execute(
            "INSERT INTO call_records VALUES ('must-not-dedupe')"
        )
    api_windows: list[tuple[datetime, datetime]] = []
    staged: list[TelephonyCallEvent] = []
    event = TelephonyCallEvent(
        tenant=TenantRef("foton"),
        provider="mango",
        provider_call_id="must-not-dedupe",
        started_at=datetime(2026, 8, 8, 10, tzinfo=timezone.utc),
        ended_at=None,
        direction=Direction.INBOUND,
        client_phone=None,
        manager_ref=None,
        recording_ref="recording-new",
        raw_payload={},
    )

    class EmptyClient:
        def __init__(self, **_: object) -> None:
            pass

        def poll_call_history(
            self, *, since: datetime, until: datetime
        ) -> list[dict[str, object]]:
            api_windows.append((since, until))
            return [
                {
                    "id": "must-not-dedupe",
                    "start": event.started_at.isoformat(),
                    "finish": event.started_at.isoformat(),
                }
            ] if since <= event.started_at <= until else []

    class FakeMapper:
        def from_payload(self, **_: object) -> TelephonyCallEvent:
            return event

    class Summary:
        failed = 0

        def to_json_dict(self) -> dict[str, int]:
            return {"downloaded": 1, "failed": 0}

    def fake_stage(
        *, events: Sequence[TelephonyCallEvent], **_: object
    ) -> Summary:
        staged.extend(events)
        return Summary()

    monkeypatch.setattr(calls_runtime, "MangoOfficeClient", EmptyClient)
    monkeypatch.setattr(calls_runtime, "MangoRecordingDownloader", EmptyClient)
    monkeypatch.setattr(calls_runtime, "MangoOfficePayloadMapper", FakeMapper)
    monkeypatch.setattr(
        calls_runtime,
        "read_ingested_call_ids",
        lambda *_args, **_kwargs: pytest.fail(
            "untrusted downstream DB must not be used for deduplication"
        ),
    )
    monkeypatch.setattr(calls_runtime, "stage_capture_events", fake_stage)
    monkeypatch.setattr(
        calls_runtime,
        "configured_host_id",
        lambda *_args, **_kwargs: "m1-host",
    )
    monkeypatch.setenv("MANGO_OFFICE_API_KEY", "present")
    monkeypatch.setenv("MANGO_OFFICE_API_SALT", "present")

    report = capture_mango_window(
        config,
        datetime(2026, 8, 7, 21, tzinfo=timezone.utc),
        datetime(2026, 8, 8, 21, tzinfo=timezone.utc),
    )

    assert report["status"] == "ok"
    assert api_windows
    assert report["api_events_total"] == 1
    assert report["api_events_already_known_external"] == 0
    assert [item.provider_call_id for item in staged] == ["must-not-dedupe"]
    assert config.working_db.is_file()


def test_missing_working_db_never_replaces_surviving_ready_generation(
    tmp_path: Path,
) -> None:
    config = config_for(tmp_path)
    config.working_db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(config.working_db) as connection:
        connection.execute("CREATE TABLE call_records(id INTEGER PRIMARY KEY)")
        connection.execute("INSERT INTO call_records VALUES (1)")
    publish_ready_db(config, {"total": 1})
    ready_pair = (
        sha256_file(config.ready_db),
        sha256_file(config.ready_manifest),
        config.ready_db.stat().st_ino,
        config.ready_db.stat().st_mtime_ns,
    )
    config.working_db.unlink()
    capture_attempts: list[bool] = []

    def failed_capture(*_args: object) -> Mapping[str, object]:
        capture_attempts.append(True)
        return {"status": "failed", "reason": "synthetic_api_failure"}

    capture = run_capture(
        config,
        capture_runner=failed_capture,
    )
    process_a = run_process_a(
        config,
        capture_runner=lambda *_args: pytest.fail("Mango API must not run"),
        command_runner=lambda *_args: pytest.fail("init-db must not run"),
    )

    assert capture_attempts == [True]
    assert capture["status"] == process_a["status"] == "failed"
    assert capture["stop_reason"] == "capture_or_enumeration_failed"
    assert process_a["stop_reason"] == (
        "working_db_missing_ready_generation_preserved"
    )
    assert not config.working_db.exists()
    assert ready_pair == (
        sha256_file(config.ready_db),
        sha256_file(config.ready_manifest),
        config.ready_db.stat().st_ino,
        config.ready_db.stat().st_mtime_ns,
    )
    assert read_json(config.cursor_path) == {}


@pytest.mark.parametrize("entrypoint", ("capture", "process_a"))
def test_invalid_working_db_never_replaces_surviving_ready_generation(
    tmp_path: Path,
    entrypoint: str,
) -> None:
    config = config_for(tmp_path)
    config.working_db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(config.working_db) as connection:
        connection.execute("CREATE TABLE call_records(id INTEGER PRIMARY KEY)")
        connection.execute("INSERT INTO call_records VALUES (1)")
    publish_ready_db(config, {"total": 1})
    ready_pair = (
        sha256_file(config.ready_db),
        sha256_file(config.ready_manifest),
        config.ready_db.stat().st_ino,
        config.ready_db.stat().st_mtime_ns,
    )
    config.working_db.write_bytes(b"")
    capture_attempts: list[bool] = []

    def invoke() -> Mapping[str, object]:
        if entrypoint == "capture":
            return run_capture(
                config,
                capture_runner=lambda *_args: (
                    capture_attempts.append(True)
                    or {"status": "failed", "reason": "synthetic_api_failure"}
                ),
            )
        return run_process_a(
            config,
            capture_runner=lambda *_args: pytest.fail(
                "Mango API must not run"
            ),
            command_runner=lambda *_args: pytest.fail(
                "worker command must not run"
            ),
        )

    first = invoke()
    second = invoke()

    assert first["status"] == second["status"] == "failed"
    expected_reason = (
        "capture_or_enumeration_failed"
        if entrypoint == "capture"
        else "working_db_invalid_ready_generation_preserved"
    )
    assert first["stop_reason"] == second["stop_reason"] == expected_reason
    assert capture_attempts == ([True, True] if entrypoint == "capture" else [])
    assert config.working_db.stat().st_size == 0
    assert ready_pair == (
        sha256_file(config.ready_db),
        sha256_file(config.ready_manifest),
        config.ready_db.stat().st_ino,
        config.ready_db.stat().st_mtime_ns,
    )
    assert read_json(config.cursor_path) == {}


def test_process_a_recovers_staged_ready_before_working_db_gate(
    tmp_path: Path,
) -> None:
    config = config_for(tmp_path)
    config.working_db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(config.working_db) as connection:
        connection.execute("CREATE TABLE call_records(id INTEGER PRIMARY KEY)")
        connection.execute("INSERT INTO call_records VALUES (1)")
    manifest = {
        "ready_db": str(config.ready_db),
        "sha256": sha256_file(config.working_db),
        "size_bytes": config.working_db.stat().st_size,
    }

    def crash_after_journal(stage: str) -> None:
        if stage == "journal_written":
            raise RuntimeError("synthetic crash after journal")

    with pytest.raises(RuntimeError, match="synthetic crash after journal"):
        commit_ready_generation(
            config.ready_db,
            config.working_db,
            manifest,
            checkpoint=crash_after_journal,
        )

    assert not config.working_db.exists()
    assert not config.ready_db.exists()
    assert inspect_ready_publication(config.ready_db)["recovery_required"] is True

    def invoke() -> Mapping[str, object]:
        return run_process_a(
            config,
            capture_runner=lambda *_args: pytest.fail(
                "Mango API must not run"
            ),
            command_runner=lambda *_args: pytest.fail(
                "init-db must not run"
            ),
        )

    first = invoke()
    first_pair = (
        sha256_file(config.ready_db),
        sha256_file(config.ready_manifest),
        config.ready_db.stat().st_ino,
        config.ready_db.stat().st_mtime_ns,
    )
    second = invoke()

    assert first["status"] == second["status"] == "failed"
    assert first["stop_reason"] == second["stop_reason"] == (
        "working_db_missing_ready_generation_preserved"
    )
    assert not config.working_db.exists()
    assert inspect_ready_publication(config.ready_db)["recovery_required"] is False
    assert first_pair == (
        sha256_file(config.ready_db),
        sha256_file(config.ready_manifest),
        config.ready_db.stat().st_ino,
        config.ready_db.stat().st_mtime_ns,
    )


def test_ready_manifest_with_pending_call_is_never_time_frozen_by_reuse(
    tmp_path: Path,
) -> None:
    from mango_mvp.productization.capture_staging import (
        CaptureManifestStore,
        ManifestEntry,
    )

    config = config_for(tmp_path)
    create_ready_call_db(config.working_db)
    with sqlite3.connect(config.working_db) as connection:
        connection.execute(
            "UPDATE call_records SET resolve_status='pending', "
            "analysis_status='pending', analysis_json=NULL"
        )
    store = CaptureManifestStore(config.capture_manifest)
    store.append(
        ManifestEntry(
            schema_version="capture_manifest_v1",
            created_at="2026-07-09T10:00:00+00:00",
            tenant_id="foton",
            provider="mango",
            event_key="foton:mango:provider-1",
            provider_call_id="provider-1",
            recording_id="recording-1",
            started_at="2026-07-09T10:00:00+00:00",
            ended_at=None,
            direction="inbound",
            client_phone=None,
            manager_ref=None,
            status="downloaded",
        )
    )
    manifest_end_offset = config.capture_manifest.stat().st_size
    snapshot = calls_runtime.capture_manifest_snapshot(
        config.capture_manifest,
        end_offset=manifest_end_offset,
    )
    evidence = with_dual_enumeration({
        "mango_enumeration_complete": True,
        "mango_enumeration_source": {
            "mode": "strict_service",
            "since": "2026-07-08T21:00:00+00:00",
            "rolling_since": "2026-07-08T21:00:00+00:00",
            "until": "2026-07-09T21:00:00+00:00",
            "cursor": "not_applicable_stats_request_result",
            "requests": 1,
            "pages": None,
            "pagination": "not_applicable_stats_request_result",
            "covered_intervals": [
                {
                    "since": "2026-07-08T21:00:00+00:00",
                    "until": "2026-07-09T21:00:00+00:00",
                    "result_complete": True,
                    "rows": 1,
                    "scope": "rolling_authority",
                }
            ],
        },
        "call_keys": ["provider-1"],
        "calls_by_moscow_day": {"2026-07-09": ["provider-1"]},
        "independent_zero_enumerations_by_day": {"2026-07-09": 0},
        "api_requests": 1,
        "api_rows_total": 1,
        "api_authoritative_rows_total": 1,
        "api_events_total": 1,
        "manifest_end_offset": manifest_end_offset,
        "manifest_snapshot_sha256": snapshot["sha256"],
    })
    first = publish_ready_db(
        config,
        {"total": 1},
        capture_evidence=evidence,
        manifest_end_offset=manifest_end_offset,
    )
    repeated = publish_ready_db_if_changed(
        config,
        {"total": 1},
        changed=False,
        capture_evidence=evidence,
        manifest_end_offset=manifest_end_offset,
    )

    assert first["daily_verdicts"]["2026-07-09"]["pending_unique"] == 1
    assert repeated["reused"] is False
    assert repeated["daily_verdicts"]["2026-07-09"]["pending_unique"] == 1


def test_network_outage_with_open_llm_work_remains_deferred(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(config_for(tmp_path), min_free_gib=1)
    create_ready_call_db(config.working_db)
    with sqlite3.connect(config.working_db) as connection:
        connection.execute(
            "UPDATE call_records SET resolve_status='pending', "
            "analysis_status='pending', analysis_json=NULL"
        )
    monkeypatch.setattr(
        calls_runtime,
        "cutover_authority_report",
        lambda *_args, **_kwargs: {"ok": True},
    )
    monkeypatch.setattr(
        calls_runtime,
        "disk_preflight",
        lambda *_args, **_kwargs: {"ok": True},
    )
    monkeypatch.setattr(
        calls_runtime,
        "environment_preflight",
        lambda *_args, **_kwargs: {
            "ok": True,
            "codex_network_ok": False,
        },
    )
    monkeypatch.setattr(
        calls_runtime,
        "prepare_ingest_inputs",
        lambda *_args, **_kwargs: {"audio_files": 0, "skipped_total": 0},
    )
    commands: list[list[str]] = []

    def command_runner(
        command: list[str], _environment: Mapping[str, str], _cwd: Path
    ) -> Mapping[str, object]:
        commands.append(command)
        return {"rc": 0, "command": command[-1]}

    result = run_process_a(
        config,
        skip_capture=True,
        command_runner=command_runner,
    )

    assert result["status"] == "deferred"
    assert result["stop_reason"] == "codex_network_unavailable"
    assert [
        command[command.index("--stages") + 1]
        for command in commands
        if "--stages" in command
    ] == ["transcribe", "backfill-second-asr"]
    assert not config.ready_db.exists()


def test_empty_runtime_persists_zero_proofs_and_reuses_closed_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(
        config_for(tmp_path),
        min_free_gib=1,
        pending_recording_retry_hours=24,
    )
    monkeypatch.setattr(
        calls_runtime,
        "cutover_authority_report",
        lambda *_args, **_kwargs: {"ok": True},
    )
    monkeypatch.setattr(
        calls_runtime,
        "disk_preflight",
        lambda *_args, **_kwargs: {"ok": True},
    )
    monkeypatch.setattr(
        calls_runtime,
        "environment_preflight",
        lambda *_args, **_kwargs: {
            "ok": True,
            "codex_network_ok": False,
        },
    )
    zero_proofs = iter((2, 2, 2))
    commands: list[list[str]] = []

    def command_runner(
        command: list[str], environment: Mapping[str, str], cwd: Path
    ) -> Mapping[str, object]:
        commands.append(command)
        return run_command(command, environment, cwd)

    def empty_capture(
        runtime_config: CallsTwoProcessesConfig,
        since: datetime,
        until: datetime,
    ) -> dict[str, object]:
        create_empty_capture_manifest(runtime_config)
        snapshot = calls_runtime.capture_manifest_snapshot(
            runtime_config.capture_manifest,
            end_offset=0,
        )
        return with_dual_enumeration(
            {
                "status": "ok",
                "downloaded": 0,
                "failed": 0,
                "mango_enumeration_complete": True,
                "mango_enumeration_source": {
                    "mode": "strict_service",
                    "since": since.isoformat(),
                    "rolling_since": since.isoformat(),
                    "until": until.isoformat(),
                    "cursor": "not_applicable_stats_request_result",
                    "requests": 1,
                    "pages": None,
                    "pagination": "not_applicable_stats_request_result",
                    "covered_intervals": [
                        {
                            "since": since.isoformat(),
                            "until": until.isoformat(),
                            "result_complete": True,
                            "rows": 0,
                            "scope": "rolling_authority",
                        }
                    ],
                },
                "call_keys": [],
                "calls_by_moscow_day": {"2026-08-08": []},
                "independent_zero_enumerations_by_day": {
                    "2026-08-08": next(zero_proofs)
                },
                "api_requests": 1,
                "api_rows_total": 0,
                "api_authoritative_rows_total": 0,
                "api_events_total": 0,
                "manifest_end_offset": 0,
                "manifest_snapshot_sha256": snapshot["sha256"],
            }
        )

    kwargs = {
        "since": "2026-08-07T21:00:00+00:00",
        "until": "2026-08-08T21:00:00+00:00",
        "capture_runner": empty_capture,
    }
    first = run_process_a(config, command_runner=command_runner, **kwargs)
    first_cursor = read_json(config.cursor_path)
    second = run_process_a(config, command_runner=command_runner, **kwargs)
    second_cursor = read_json(config.cursor_path)
    repeated = run_process_a(config, command_runner=command_runner, **kwargs)

    assert config.working_db.is_file()
    assert first["status"] == "ok"
    assert first["counters"]["drop"]["closure_ok"] is True
    assert first["counters"]["drop"]["reused"] is False
    assert first["counters"]["drop"]["enumeration_evidence_sha256"] == (
        calls_runtime.capture_enumeration_evidence_sha256(first_cursor)
    )
    assert second["status"] == "ok"
    assert second["counters"]["drop"]["reused"] is True
    assert second["counters"]["drop"]["closure_ok"] is True
    assert second["counters"]["drop"]["enumeration_evidence_sha256"] == (
        calls_runtime.capture_enumeration_evidence_sha256(second_cursor)
    )
    assert repeated["status"] == "ok"
    assert repeated["counters"]["drop"]["reused"] is True
    assert len(commands) == 1
    assert commands[0][-1] == "init-db"


def test_network_outage_runs_only_local_asr_stages(tmp_path: Path) -> None:
    normal = config_for(tmp_path)

    assert pipeline_stages(normal, include_llm=False) == ("transcribe", "backfill-second-asr")


def test_codex_network_probe_is_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args, **kwargs):
        del args, kwargs
        raise OSError("dns unavailable")

    monkeypatch.setattr("socket.getaddrinfo", fail)
    assert codex_network_available() is False


def test_environment_preflight_lists_failed_checks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("MANGO_OFFICE_API_KEY", raising=False)
    monkeypatch.delenv("MANGO_OFFICE_API_SALT", raising=False)
    config = replace(
        config_for(tmp_path),
        python_executable=tmp_path / "missing-python",
        codex_binary=tmp_path / "missing-codex",
    )
    report = environment_preflight(config, run_commands=True, require_mango_credentials=True)
    assert report["ok"] is False
    assert set(report["failed_checks"]) >= {
        "mango_credentials",
        "python_executable",
        "asr_modules",
        "codex_binary",
        "codex_auth",
    }


def test_module_preflight_checks_presence_without_loading_heavy_models(tmp_path: Path) -> None:
    command = module_probe_command(config_for(tmp_path))
    assert "find_spec" in command[-1]
    assert "import mlx_whisper" not in command[-1]
    assert "import gigaam" not in command[-1]


def test_command_path_includes_codex_binary_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = replace(config_for(tmp_path), codex_binary=tmp_path / "homebrew" / "bin" / "codex")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    assert command_path(config).split(os.pathsep)[0] == str(config.codex_binary.parent)


def test_pipeline_freshness_marks_old_data_stale(tmp_path: Path) -> None:
    config = replace(config_for(tmp_path), freshness_max_age_minutes=30)
    create_empty_capture_manifest(config)
    old = "2026-07-10T01:00:00+00:00"
    for path, process in (
        (config.process_a_status_path, "process_a"),
        (config.process_b_status_path, "process_b"),
    ):
        write_json(path, {"process": process, "status": "ok", "checked_through": old, "data_through": old})
    report = pipeline_freshness(config, now=datetime(2026, 7, 10, 2, 0, tzinfo=timezone.utc))
    assert report["status"] == "stale"
    assert report["stages"]["process_a"]["status"] == "stale"
    assert report["stages"]["process_b"]["status"] == "stale"


def test_pipeline_freshness_rejects_future_status_timestamps(
    tmp_path: Path,
) -> None:
    config = replace(config_for(tmp_path), freshness_max_age_minutes=30)
    create_empty_capture_manifest(config)
    future = "2099-01-01T00:00:00+00:00"
    for path, process in (
        (config.process_a_status_path, "process_a"),
        (config.process_b_status_path, "process_b"),
    ):
        write_json(
            path,
            {
                "process": process,
                "status": "ok",
                "checked_through": future,
                "data_through": future,
            },
        )

    report = pipeline_freshness(
        config, now=datetime(2026, 7, 10, 2, 0, tzinfo=timezone.utc)
    )

    assert report["status"] == "stale"
    assert report["stages"]["process_a"]["status"] == "future"
    assert report["stages"]["process_b"]["status"] == "future"


@pytest.mark.parametrize(
    ("last_status", "expected_status"),
    (
        ("deferred", "deferred"),
        ("blocked", "blocked"),
        ("locked", "locked"),
        ("unexpected", "invalid"),
    ),
)
def test_pipeline_freshness_never_turns_non_success_status_green(
    tmp_path: Path, last_status: str, expected_status: str
) -> None:
    config = replace(config_for(tmp_path), freshness_max_age_minutes=30)
    create_empty_capture_manifest(config)
    fresh = "2026-07-10T02:00:00+00:00"
    write_json(
        config.process_a_status_path,
        {
            "process": "process_a",
            "status": last_status,
            "stop_reason": "synthetic_stop",
            "checked_through": fresh,
            "data_through": fresh,
        },
    )
    write_json(
        config.process_b_status_path,
        {
            "process": "process_b",
            "status": "ok",
            "checked_through": fresh,
            "data_through": fresh,
        },
    )

    report = pipeline_freshness(
        config, now=datetime(2026, 7, 10, 2, 1, tzinfo=timezone.utc)
    )

    assert report["status"] == "stale"
    assert report["stages"]["process_a"]["status"] == expected_status
    assert report["stages"]["process_a"]["last_run_status"] == last_status


def test_future_live_pid_heartbeat_cannot_override_failed_process_a(
    tmp_path: Path,
) -> None:
    config = replace(config_for(tmp_path), freshness_max_age_minutes=30)
    create_empty_capture_manifest(config)
    now = datetime(2026, 7, 10, 2, 1, tzinfo=timezone.utc)
    fresh = "2026-07-10T02:00:00+00:00"
    write_json(
        config.process_a_status_path,
        {
            "process": "process_a",
            "status": "failed",
            "stop_reason": "runtime_fingerprint_mismatch",
            "checked_through": fresh,
            "data_through": fresh,
        },
    )
    write_json(
        config.process_b_status_path,
        {
            "process": "process_b",
            "status": "ok",
            "checked_through": fresh,
            "data_through": fresh,
        },
    )
    write_json(
        config.process_a_heartbeat_path,
        {
            "updated_at": "2099-01-01T00:00:00+00:00",
            "pid": os.getpid(),
            "stage": "transcribe",
        },
    )

    report = pipeline_freshness(config, now=now)

    assert report["status"] == "stale"
    assert report["heavy_heartbeat"]["status"] == "stale_or_dead"
    assert report["heavy_heartbeat"]["age_seconds"] < 0
    assert report["stages"]["process_a"]["status"] == "failed"
    assert (
        report["stages"]["process_a"]["stop_reason"]
        == "runtime_fingerprint_mismatch"
    )


def test_local_watchdog_raises_p0_for_foreign_manifest_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = config_for(tmp_path)
    monkeypatch.setattr(
        calls_runtime,
        "pipeline_freshness",
        lambda *_args, **_kwargs: {"status": "fresh", "stages": {}},
    )
    monkeypatch.setattr(
        calls_runtime,
        "cutover_authority_report",
        lambda *_args, **_kwargs: {
            "ok": True,
            "previous_host_disabled_at": "2026-08-11T08:00:00+00:00",
        },
    )
    monkeypatch.setattr(
        calls_runtime,
        "capture_manifest_snapshot",
        lambda *_args, **_kwargs: {"entries": []},
    )
    monkeypatch.setattr(calls_runtime, "configured_host_id", lambda *_a, **_k: "m1-host")
    monkeypatch.setattr(
        calls_runtime,
        "foreign_host_ids",
        lambda *_args, **_kwargs: ["old-mac-host"],
    )

    report = run_local_watchdog(config, now=datetime(2026, 8, 11, 9, tzinfo=timezone.utc))

    assert report["status"] == "p0"
    assert report["stop_reason"] == "foreign_host_or_cutover_authority_failed"
    assert report["foreign_host_ids"] == ["old-mac-host"]
    assert report["safe_alert"]["foreign_host_count"] == 1
    assert report["safety"]["read_only"] is True


def test_pipeline_freshness_does_not_call_missing_drop_fresh(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    write_json(
        config.process_b_status_path,
        {
            "process": "process_b",
            "status": "idle",
            "stop_reason": "drop_missing",
            "checked_at": "2026-07-10T02:00:00+00:00",
        },
    )
    report = pipeline_freshness(config, now=datetime(2026, 7, 10, 2, 1, tzinfo=timezone.utc))
    assert report["stages"]["process_b"]["status"] == "missing"


def test_pipeline_freshness_uses_recent_success_during_a_quiet_period(tmp_path: Path) -> None:
    config = replace(config_for(tmp_path), freshness_max_age_minutes=30)
    create_empty_capture_manifest(config)
    write_json(
        config.process_a_status_path,
        {
            "process": "process_a",
            "status": "ok",
            "checked_through": "2026-07-10T02:00:00+00:00",
            "data_through": "2026-07-10T01:00:00+00:00",
        },
    )
    report = pipeline_freshness(config, now=datetime(2026, 7, 10, 2, 5, tzinfo=timezone.utc))
    assert report["stages"]["process_a"]["status"] == "fresh"
    assert report["stages"]["process_a"]["age_seconds"] == 3900.0


def test_pipeline_freshness_missing_data_is_not_fresh(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    write_json(
        config.process_a_status_path,
        {"process": "process_a", "status": "ok", "checked_through": "2026-07-10T02:00:00+00:00"},
    )
    report = pipeline_freshness(config, now=datetime(2026, 7, 10, 2, 1, tzinfo=timezone.utc))
    assert report["stages"]["process_a"]["status"] == "missing"


def test_pipeline_freshness_missing_manifest_overrides_fresh_stage(tmp_path: Path) -> None:
    config = replace(config_for(tmp_path), freshness_max_age_minutes=30)
    fresh = "2026-07-10T02:00:00+00:00"
    for path, process in (
        (config.process_a_status_path, "process_a"),
        (config.process_b_status_path, "process_b"),
    ):
        write_json(
            path,
            {"process": process, "status": "ok", "checked_through": fresh, "data_through": fresh},
        )

    report = pipeline_freshness(config, now=datetime(2026, 7, 10, 2, 1, tzinfo=timezone.utc))

    assert report["status"] == "stale"
    assert report["stages"]["process_a"]["status"] == "missing"
    assert report["stages"]["process_a"]["stop_reason"] == "capture_manifest_missing"
    assert report["stages"]["process_b"]["status"] == "fresh"


def test_pipeline_freshness_keeps_partial_day_red_with_fresh_timestamps(tmp_path: Path) -> None:
    config = replace(config_for(tmp_path), freshness_max_age_minutes=30)
    create_empty_capture_manifest(config)
    fresh = "2026-07-10T02:00:00+00:00"
    write_json(
        config.process_a_status_path,
        {
            "process": "process_a",
            "status": "partial",
            "stop_reason": "capture_manifest_tail_incomplete",
            "checked_through": fresh,
            "data_through": fresh,
        },
    )
    write_json(
        config.process_b_status_path,
        {
            "process": "process_b",
            "status": "ok",
            "checked_through": fresh,
            "data_through": fresh,
        },
    )

    report = pipeline_freshness(config, now=datetime(2026, 7, 10, 2, 1, tzinfo=timezone.utc))

    assert report["status"] == "stale"
    assert report["stages"]["process_a"]["status"] == "partial"
    assert report["stages"]["process_b"]["status"] == "fresh"


def test_pipeline_freshness_fails_closed_on_unresolved_capture_recovery(tmp_path: Path) -> None:
    from mango_mvp.productization.capture_staging import CaptureManifestStore, ManifestEntry

    config = replace(config_for(tmp_path), freshness_max_age_minutes=30)
    store = CaptureManifestStore(config.capture_manifest)
    entry = ManifestEntry(
        schema_version="v1",
        created_at="2026-07-10T02:00:00+00:00",
        tenant_id="foton",
        provider="mango",
        event_key="foton:mango:before-crash",
        provider_call_id="before-crash",
        recording_id=None,
        started_at="2026-07-10T02:00:00+00:00",
        ended_at=None,
        direction="inbound",
        client_phone=None,
        manager_ref=None,
        status="recording_retry_expired",
    )
    store.append(entry)
    with config.capture_manifest.open("ab") as handle:
        handle.write(b'{"event_key":"unfinished"')
    store.append(replace(entry, event_key="foton:mango:after-crash", provider_call_id="after-crash"))
    fresh = "2026-07-10T02:00:00+00:00"
    write_json(
        config.process_a_status_path,
        {"process": "process_a", "status": "ok", "checked_through": fresh, "data_through": fresh},
    )
    write_json(
        config.process_b_status_path,
        {"process": "process_b", "status": "ok", "checked_through": fresh, "data_through": fresh},
    )

    report = pipeline_freshness(config, now=datetime(2026, 7, 10, 2, 1, tzinfo=timezone.utc))

    assert report["status"] == "stale"
    assert report["stages"]["process_a"]["status"] == "partial"
    assert report["stages"]["process_a"]["stop_reason"] == "capture_manifest_tail_incomplete"
    assert report["stages"]["process_a"]["capture_recovery_unresolved_count"] == 1
    assert report["stages"]["process_b"]["status"] == "fresh"


def test_pipeline_freshness_fails_closed_on_torn_tail_before_recovery(tmp_path: Path) -> None:
    from mango_mvp.productization.capture_staging import (
        CaptureManifestStore,
        ManifestEntry,
        capture_recovery_path,
    )

    config = replace(config_for(tmp_path), freshness_max_age_minutes=30)
    CaptureManifestStore(config.capture_manifest).append(
        ManifestEntry(
            schema_version="v1",
            created_at="2026-07-10T02:00:00+00:00",
            tenant_id="foton",
            provider="mango",
            event_key="foton:mango:before-crash",
            provider_call_id="before-crash",
            recording_id=None,
            started_at="2026-07-10T02:00:00+00:00",
            ended_at=None,
            direction="inbound",
            client_phone=None,
            manager_ref=None,
            status="recording_retry_expired",
        )
    )
    with config.capture_manifest.open("ab") as handle:
        handle.write(b'{"event_key":"unfinished"')
    assert not capture_recovery_path(config.capture_manifest).exists()
    fresh = "2026-07-10T02:00:00+00:00"
    for path, process in (
        (config.process_a_status_path, "process_a"),
        (config.process_b_status_path, "process_b"),
    ):
        write_json(
            path,
            {"process": process, "status": "ok", "checked_through": fresh, "data_through": fresh},
        )

    report = pipeline_freshness(config, now=datetime(2026, 7, 10, 2, 1, tzinfo=timezone.utc))

    assert report["status"] == "stale"
    assert report["stages"]["process_a"]["status"] == "partial"
    assert report["stages"]["process_a"]["capture_manifest_tail_status"] == "incomplete"
    assert report["stages"]["process_a"]["capture_recovery_status"] == "resolved"
    assert report["stages"]["process_b"]["status"] == "fresh"


def test_pipeline_freshness_fails_closed_on_invalid_capture_recovery(tmp_path: Path) -> None:
    from mango_mvp.productization.capture_staging import capture_recovery_path

    config = replace(config_for(tmp_path), freshness_max_age_minutes=30)
    recovery_path = capture_recovery_path(config.capture_manifest)
    recovery_path.parent.mkdir(parents=True, exist_ok=True)
    recovery_path.write_text("{invalid", encoding="utf-8")
    fresh = "2026-07-10T02:00:00+00:00"
    write_json(
        config.process_a_status_path,
        {"process": "process_a", "status": "ok", "checked_through": fresh, "data_through": fresh},
    )

    report = pipeline_freshness(config, now=datetime(2026, 7, 10, 2, 1, tzinfo=timezone.utc))

    assert report["status"] == "stale"
    assert report["stages"]["process_a"]["status"] == "failed"
    assert report["stages"]["process_a"]["stop_reason"] == "capture_recovery_ledger_invalid"


@pytest.mark.parametrize("manifest_kind", ["directory", "fifo"])
def test_pipeline_freshness_fails_closed_on_non_regular_manifest(
    tmp_path: Path,
    manifest_kind: str,
) -> None:
    config = replace(config_for(tmp_path), freshness_max_age_minutes=30)
    config.capture_manifest.parent.mkdir(parents=True, exist_ok=True)
    if manifest_kind == "directory":
        config.capture_manifest.mkdir()
    else:
        os.mkfifo(config.capture_manifest)
    fresh = "2026-07-10T02:00:00+00:00"
    write_json(
        config.process_a_status_path,
        {"process": "process_a", "status": "ok", "checked_through": fresh, "data_through": fresh},
    )

    report = pipeline_freshness(config, now=datetime(2026, 7, 10, 2, 1, tzinfo=timezone.utc))

    assert report["status"] == "stale"
    assert report["stages"]["process_a"]["status"] == "failed"
    assert report["stages"]["process_a"]["stop_reason"] == "capture_manifest_tail_invalid"


def test_ok_stage_is_not_published_before_durable_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from mango_mvp.productization.capture_staging import atomic_write_private_json as real_atomic_write

    config = replace(config_for(tmp_path), freshness_max_age_minutes=30)
    fresh = "2026-07-10T02:00:00+00:00"
    write_json(
        config.process_b_status_path,
        {"process": "process_b", "status": "ok", "checked_through": fresh, "data_through": fresh},
    )

    def crash_on_report(path: Path, payload: object, **kwargs: object) -> None:
        if path.parent == config.reports_dir:
            raise SystemExit(77)
        real_atomic_write(path, payload, **kwargs)

    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.atomic_write_private_json",
        crash_on_report,
    )

    with pytest.raises(SystemExit, match="77"):
        finalize_report(config, "synthetic-ok", "process_a", "ok", "", {})

    assert not config.process_a_status_path.exists()
    assert not list(config.reports_dir.glob("*_process_a.json"))
    freshness = pipeline_freshness(config, now=datetime(2026, 7, 10, 2, 1, tzinfo=timezone.utc))
    assert freshness["stages"]["process_a"]["status"] == "missing"
    assert freshness["status"] == "stale"


@pytest.mark.parametrize("red_status", ["failed", "partial"])
def test_red_stage_is_published_before_broken_report_directory(
    tmp_path: Path,
    red_status: str,
) -> None:
    config = config_for(tmp_path)
    config.reports_dir.parent.mkdir(parents=True, exist_ok=True)
    config.reports_dir.write_text("not-a-directory", encoding="utf-8")

    with pytest.raises(FileExistsError):
        finalize_report(config, "synthetic-red", "process_a", red_status, "synthetic", {})

    assert read_json(config.process_a_status_path)["status"] == red_status


def test_broken_reports_directory_cannot_leave_process_a_fresh(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = replace(config_for(tmp_path), min_free_gib=1, freshness_max_age_minutes=30)
    config.reports_dir.parent.mkdir(parents=True, exist_ok=True)
    config.reports_dir.write_text("not-a-directory", encoding="utf-8")
    fresh = "2026-07-10T02:00:00+00:00"
    for path, process in (
        (config.process_a_status_path, "process_a"),
        (config.process_b_status_path, "process_b"),
    ):
        write_json(
            path,
            {"process": process, "status": "ok", "checked_through": fresh, "data_through": fresh},
        )
    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.environment_preflight",
        lambda *_args, **_kwargs: {"ok": True, "codex_network_ok": True},
    )

    with pytest.raises(FileExistsError):
        run_process_a(config, skip_capture=True, skip_workers=True)

    assert read_json(config.process_a_status_path)["status"] == "failed"
    freshness = pipeline_freshness(config, now=datetime(2026, 7, 10, 2, 1, tzinfo=timezone.utc))
    assert freshness["stages"]["process_a"]["status"] == "failed"
    assert freshness["status"] == "stale"


def test_dead_letter_total_ignores_empty_stage_and_counts_failures() -> None:
    assert dead_letter_total({"dead_letter_stage": {"": 200, "transcribe": 2, "analyze": 1}}) == 3
    assert dead_letter_mass_failure({"total": 241, "dead_letter_stage": {"transcribe": 3}}) is False
    assert dead_letter_mass_failure({"total": 241, "dead_letter_stage": {"transcribe": 13}}) is True


def test_codex_runtime_does_not_copy_desktop_mcp_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    source = home / ".codex"
    source.mkdir(parents=True)
    (source / "auth.json").write_text('{"auth":"masked"}', encoding="utf-8")
    (source / "config.toml").write_text('[mcp_servers.live]\ncommand="unsafe"\n', encoding="utf-8")
    (source / "AGENTS.md").write_text("desktop personality", encoding="utf-8")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    target = prepare_codex_home(tmp_path / "runtime")

    assert "mcp_servers" not in (target / "config.toml").read_text(encoding="utf-8")
    assert "desktop personality" not in (target / "AGENTS.md").read_text(encoding="utf-8")
    assert (target / "auth.json").is_file()


@pytest.mark.skipif(sys.platform != "darwin", reason="Darwin extended ACL control")
def test_strict_codex_home_atomically_replaces_existing_auth_acl(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    source = home / ".codex"
    source.mkdir(parents=True)
    source_auth = source / "auth.json"
    source_auth.write_text('{"auth":"fresh"}', encoding="utf-8")
    source_auth.chmod(0o600)
    target = home / ".mango_local" / "codex-runtime"
    target.mkdir(parents=True, mode=0o700)
    target.chmod(0o700)
    stale_auth = target / "auth.json"
    stale_auth.write_text('{"auth":"stale"}', encoding="utf-8")
    stale_auth.chmod(0o600)
    subprocess.run(
        ["/bin/chmod", "+a", "everyone allow read", str(stale_auth)],
        check=True,
    )
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    prepared = prepare_codex_home(target, strict=True)

    assert prepared == target.resolve()
    assert read_stable_regular_bytes(
        stale_auth,
        label="test_isolated_auth",
        owner_only_mode=0o600,
    ) == b'{"auth":"fresh"}'


@pytest.mark.skipif(sys.platform != "darwin", reason="Darwin extended ACL control")
def test_strict_codex_home_rejects_extended_acl_on_runtime_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    (home / ".codex").mkdir(parents=True)
    target = home / ".mango_local" / "codex-runtime"
    target.mkdir(parents=True, mode=0o700)
    target.chmod(0o700)
    subprocess.run(
        ["/bin/chmod", "+a", "everyone allow read", str(target)],
        check=True,
    )
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    with pytest.raises(RuntimeError, match="extended_acl"):
        prepare_codex_home(target, strict=True)


def test_codex_home_revokes_stale_auth_when_source_auth_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    (home / ".codex").mkdir(parents=True)
    target = tmp_path / "runtime"
    target.mkdir(mode=0o700)
    stale_auth = target / "auth.json"
    stale_auth.write_text('{"auth":"revoked"}', encoding="utf-8")
    stale_auth.chmod(0o600)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    prepare_codex_home(target, strict=False)

    assert not stale_auth.exists()


def test_codex_home_rejects_dangling_optional_source_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    (home / ".codex").mkdir(parents=True)
    target = tmp_path / "runtime"
    target.mkdir(mode=0o700)
    stale_installation = target / "installation_id"
    stale_installation.symlink_to(target / "future-installation-id")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    with pytest.raises(RuntimeError, match="unsafe_or_missing"):
        prepare_codex_home(target, strict=False)

    assert os.path.lexists(stale_installation)


def test_codex_home_recovers_valid_atomic_auth_temp_after_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    (home / ".codex").mkdir(parents=True)
    target = home / ".mango_local" / "runtime"
    target.mkdir(parents=True, mode=0o700)
    target.chmod(0o700)
    residue = target / ".auth.json.crash123.tmp"
    residue.write_text('{"auth":"interrupted"}', encoding="utf-8")
    residue.chmod(0o600)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    prepare_codex_home(target, strict=True)

    assert not residue.exists()
    assert not (target / "auth.json").exists()


def test_codex_home_rejects_cloud_target_even_without_strict_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    source = home / ".codex"
    source.mkdir(parents=True)
    auth = source / "auth.json"
    auth.write_text('{"auth":"synthetic"}', encoding="utf-8")
    auth.chmod(0o600)
    target = home / "Yandex.Disk" / "runtime"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    with pytest.raises(RuntimeError, match="outside cloud"):
        prepare_codex_home(target, strict=False)

    assert not target.exists()


@pytest.mark.parametrize(
    "cloud_parts",
    [
        ("Yandex.Disk", "codex"),
        ("Library", "CloudStorage", "GoogleDrive-test"),
    ],
)
def test_strict_codex_home_rejects_auth_source_inside_cloud_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cloud_parts: tuple[str, ...],
) -> None:
    home = tmp_path / "home"
    cloud_source = home.joinpath(*cloud_parts)
    cloud_source.mkdir(parents=True)
    auth = cloud_source / "auth.json"
    auth.write_text('{"auth":"synthetic"}', encoding="utf-8")
    auth.chmod(0o600)
    (home / ".codex").symlink_to(cloud_source, target_is_directory=True)
    target = home / ".mango_local" / "runtime"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    with pytest.raises(RuntimeError, match="outside cloud"):
        prepare_codex_home(target, strict=True)

    assert not (target / "auth.json").exists()


def test_codex_wrapper_disables_desktop_tools(tmp_path: Path) -> None:
    captured = tmp_path / "args.txt"
    captured_env = tmp_path / "env.txt"
    fake = tmp_path / "fake-codex"
    fake.write_text(
        "#!/bin/zsh\n"
        f"/usr/bin/env > {shlex.quote(str(captured_env))}\n"
        f"printf '%s\\n' \"$@\" > {shlex.quote(str(captured))}\n",
        encoding="utf-8",
    )
    fake.chmod(0o700)
    wrapper = Path(__file__).resolve().parents[1] / "scripts" / "run_codex_cli_isolated.sh"
    codex_home = tmp_path / "codex-home"
    process_home = tmp_path / "process-home"
    process_tmp = tmp_path / "process-tmp"
    for directory in (codex_home, process_home, process_tmp):
        directory.mkdir(mode=0o700)
    env = {
        **os.environ,
        "CODEX_HOME": str(codex_home),
        "MANGO_CODEX_REAL_BIN": str(fake),
        "MANGO_CODEX_PROCESS_HOME": str(process_home),
        "MANGO_CODEX_PROCESS_TMPDIR": str(process_tmp),
        "MANGO_OFFICE_API_SALT": "must-not-reach-codex",
        "TALLANTO_API_KEY": "must-not-reach-codex",
        "GOOGLE_APPLICATION_CREDENTIALS": "must-not-reach-codex",
        "YANDEX_DISK_TOKEN": "must-not-reach-codex",
        "OPENAI_API_KEY": "must-not-reach-codex",
    }

    subprocess.run([str(wrapper), "exec", "--model", "test", "prompt"], env=env, check=True)

    args = captured.read_text(encoding="utf-8").splitlines()
    assert args[0] == "exec"
    assert args.count("--disable") == 5
    assert "apps" in args and "plugins" in args and "browser_use" in args
    assert args[-3:] == ["--model", "test", "prompt"]
    child_env = dict(
        line.split("=", 1)
        for line in captured_env.read_text(encoding="utf-8").splitlines()
        if "=" in line
    )
    assert child_env["HOME"] == str(process_home)
    assert child_env["CODEX_HOME"] == str(codex_home)
    assert child_env["TMPDIR"] == str(process_tmp)
    assert child_env["NO_COLOR"] == "1"
    assert not {
        "MANGO_CODEX_REAL_BIN",
        "MANGO_CODEX_PROCESS_HOME",
        "MANGO_CODEX_PROCESS_TMPDIR",
        "MANGO_OFFICE_API_SALT",
        "TALLANTO_API_KEY",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "YANDEX_DISK_TOKEN",
        "OPENAI_API_KEY",
    } & set(child_env)


def create_ready_call_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as con:
        con.execute(
            """
            CREATE TABLE call_records (
                id INTEGER PRIMARY KEY,
                source_call_id TEXT,
                source_filename TEXT NOT NULL,
                source_file TEXT NOT NULL UNIQUE,
                started_at TEXT,
                audio_codec TEXT,
                sample_rate INTEGER,
                channels INTEGER,
                phone TEXT,
                manager_name TEXT,
                direction TEXT,
                duration_sec REAL,
                transcription_status TEXT,
                sync_status TEXT,
                transcribe_attempts INTEGER DEFAULT 0,
                transcript_variants_json TEXT,
                transcript_manager TEXT,
                transcript_client TEXT,
                transcript_text TEXT,
                resolve_status TEXT,
                analysis_status TEXT,
                analysis_json TEXT,
                dead_letter_stage TEXT,
                pipeline_stage TEXT,
                pipeline_worker_id TEXT,
                pipeline_claimed_at TEXT,
                analysis_worker_id TEXT,
                analysis_claimed_at TEXT,
                resolve_attempts INTEGER DEFAULT 0,
                analyze_attempts INTEGER DEFAULT 0,
                sync_attempts INTEGER DEFAULT 0,
                next_retry_at TEXT,
                resolve_json TEXT,
                resolve_quality_score REAL,
                last_error TEXT,
                amocrm_contact_id TEXT,
                amocrm_lead_id TEXT,
                created_at TEXT,
                updated_at TEXT
            )
            """
        )
        con.execute(
            """
            INSERT INTO call_records (
                id, source_call_id, source_filename, source_file, started_at,
                phone, manager_name, direction, duration_sec,
                transcription_status, transcript_variants_json,
                resolve_status, analysis_status, analysis_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                1,
                "provider-1",
                "masked.mp3",
                "/ignored/masked.mp3",
                "2026-07-09T10:00:00+00:00",
                "",
                "manager",
                "inbound",
                60.0,
                "done",
                json.dumps(
                    {
                        "mode": "mono_or_fallback",
                        "primary_provider": "mlx",
                        "secondary_provider": "gigaam",
                        "full": {
                            "variant_a": "готовый Whisper",
                            "variant_b": "готовый GigaAM",
                        },
                    },
                    ensure_ascii=False,
                ),
                "done",
                "done",
                json.dumps({"call_type": "sales_call", "history_summary": "Обсуждался курс."}),
            ),
        )
    write_json(
        path.with_suffix(".manifest.json"),
        {
            "schema_version": "mango_calls_two_processes_v1",
            "status": "ready",
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "quick_check": "ok",
        },
    )


def test_legacy_asr_without_mode_is_not_skipped_as_fully_ready(
    tmp_path: Path,
) -> None:
    config = config_for(tmp_path)
    for database in (config.working_db, config.ready_db):
        create_ready_call_db(database)
        with sqlite3.connect(database) as connection:
            raw = connection.execute(
                "SELECT transcript_variants_json FROM call_records WHERE id=1"
            ).fetchone()[0]
            payload = json.loads(raw)
            payload.pop("mode")
            connection.execute(
                "UPDATE call_records SET transcript_variants_json=? WHERE id=1",
                (json.dumps(payload, ensure_ascii=False),),
            )

    assert read_fully_ready_call_ids(config) == set()


def test_legacy_asr_mode_is_normalized_once_and_republished_after_crash_gap(
    tmp_path: Path,
) -> None:
    config = config_for(tmp_path)
    create_ready_call_db(config.working_db)
    with sqlite3.connect(config.working_db) as connection:
        raw = connection.execute(
            "SELECT transcript_variants_json FROM call_records WHERE id=1"
        ).fetchone()[0]
        payload = json.loads(raw)
        payload.pop("mode")
        legacy_raw = json.dumps(payload, ensure_ascii=False)
        connection.execute(
            """
            UPDATE call_records
               SET transcript_variants_json=?, pipeline_stage='',
                   next_retry_at='2099-01-01T00:00:00+00:00',
                   resolve_json='{"old":true}', resolve_quality_score=99,
                   last_error='old synthetic error'
             WHERE id=1
            """,
            (legacy_raw,),
        )

    # Establish a sealed legacy generation, then model a crash after the
    # working-DB normalization but before ready publication.
    publish_ready_db(config, {"total": 1})
    first = normalize_unambiguous_legacy_asr_topologies(config.working_db)

    assert first == {
        "normalized": 1,
        "downstream_invalidated": 1,
        "state_normalized": 0,
        "dead_letter_state_normalized": 0,
        "resolve_state_normalized": 0,
        "blocked": 0,
        "blocked_reasons": {},
    }
    with sqlite3.connect(config.working_db) as connection:
        normalized_raw = connection.execute(
            "SELECT transcript_variants_json FROM call_records WHERE id=1"
        ).fetchone()[0]
    normalized = json.loads(normalized_raw)
    assert normalized["mode"] == "mono_or_fallback"
    assert normalized["legacy_topology_normalization"] == {
        "method": "complete_shape_xor_reset_downstream_v1",
        "source_json_sha256": hashlib.sha256(
            legacy_raw.encode("utf-8")
        ).hexdigest(),
    }
    with sqlite3.connect(config.working_db) as connection:
        statuses = connection.execute(
            """
            SELECT resolve_status, analysis_status,
                   resolve_attempts, analyze_attempts, pipeline_stage,
                   next_retry_at, resolve_json, resolve_quality_score,
                   analysis_json, last_error
              FROM call_records
             WHERE id=1
            """
        ).fetchone()
    assert statuses == (
        "pending",
        "pending",
        0,
        0,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    assert call_db_has_open_work(config.working_db) is True

    # Synthetic stand-in for the separately authorized future Resolve/Analyze
    # stages.  No model or network call occurs in this test.
    with sqlite3.connect(config.working_db) as connection:
        connection.execute(
            """
            UPDATE call_records
               SET resolve_status='done', analysis_status='done',
                   analysis_json=?
             WHERE id=1
            """,
            (
                json.dumps(
                    {
                        "call_type": "sales_call",
                        "history_summary": "Повторный синтетический анализ.",
                    },
                    ensure_ascii=False,
                ),
            ),
        )

    recovered = publish_ready_db_if_changed(
        config,
        {"total": 1},
        changed=False,
    )
    repeated = publish_ready_db_if_changed(
        config,
        {"total": 1},
        changed=False,
    )

    assert recovered["reused"] is False
    assert repeated["reused"] is True
    assert normalize_unambiguous_legacy_asr_topologies(config.working_db)[
        "normalized"
    ] == 0
    assert read_fully_ready_call_ids(config) == {"provider-1"}
    first_process_b = run_process_b(config)
    second_process_b = run_process_b(config)
    with sqlite3.connect(f"file:{config.timeline_db}?mode=ro", uri=True) as connection:
        timeline_call_count = int(
            connection.execute(
                """
                SELECT COUNT(*)
                  FROM timeline_events
                 WHERE event_type='mango_call'
                """
            ).fetchone()[0]
        )
    assert first_process_b["status"] == "ok"
    assert second_process_b["status"] == "ok"
    assert timeline_call_count == 1


@pytest.mark.parametrize("missing_status", [None, "", "\t", "\n", "\u00a0"])
def test_dual_asr_missing_resolve_state_is_normalized_before_analyze(
    tmp_path: Path,
    missing_status: str | None,
) -> None:
    config = config_for(tmp_path)
    create_ready_call_db(config.working_db)
    with sqlite3.connect(config.working_db) as connection:
        connection.execute(
            """
            UPDATE call_records
               SET resolve_status=?, analysis_status='pending',
                   resolve_attempts=1, analyze_attempts=2,
                   next_retry_at='2099-01-01T00:00:00+00:00',
                   resolve_json='{"stale":true}', analysis_json='{"stale":true}',
                   last_error='stale synthetic error'
             WHERE id=1
            """,
            (missing_status,),
        )

    result = normalize_unambiguous_legacy_asr_topologies(config.working_db)
    repeated = normalize_unambiguous_legacy_asr_topologies(config.working_db)

    assert result == {
        "normalized": 0,
        "downstream_invalidated": 1,
        "state_normalized": 1,
        "dead_letter_state_normalized": 0,
        "resolve_state_normalized": 1,
        "blocked": 0,
        "blocked_reasons": {},
    }
    assert repeated == {
        "normalized": 0,
        "downstream_invalidated": 0,
        "state_normalized": 0,
        "dead_letter_state_normalized": 0,
        "resolve_state_normalized": 0,
        "blocked": 0,
        "blocked_reasons": {},
    }
    with sqlite3.connect(config.working_db) as connection:
        state = connection.execute(
            """
            SELECT resolve_status, analysis_status, resolve_attempts,
                   analyze_attempts, next_retry_at, resolve_json,
                   analysis_json, last_error
              FROM call_records
             WHERE id=1
            """
        ).fetchone()
    assert state == ("pending", "pending", 0, 0, None, None, None, None)
    assert call_db_has_open_work(config.working_db) is True


def test_missing_resolve_state_counts_zero_quality_as_invalidated(
    tmp_path: Path,
) -> None:
    config = config_for(tmp_path)
    create_ready_call_db(config.working_db)
    with sqlite3.connect(config.working_db) as connection:
        connection.execute(
            """
            UPDATE call_records
               SET resolve_status=NULL, analysis_status='pending',
                   resolve_attempts=0, analyze_attempts=0,
                   resolve_json=NULL, resolve_quality_score=0,
                   analysis_json=NULL, last_error=NULL
             WHERE id=1
            """
        )

    result = normalize_unambiguous_legacy_asr_topologies(config.working_db)

    assert result["state_normalized"] == 1
    assert result["downstream_invalidated"] == 1
    with sqlite3.connect(config.working_db) as connection:
        assert connection.execute(
            "SELECT resolve_quality_score FROM call_records WHERE id=1"
        ).fetchone()[0] is None


def test_missing_resolve_state_without_downstream_payload_is_not_invalidated(
    tmp_path: Path,
) -> None:
    config = config_for(tmp_path)
    create_ready_call_db(config.working_db)
    with sqlite3.connect(config.working_db) as connection:
        connection.execute(
            """
            UPDATE call_records
               SET resolve_status=NULL, analysis_status='pending',
                   resolve_attempts=0, analyze_attempts=0,
                   resolve_json=NULL, resolve_quality_score=NULL,
                   analysis_json=NULL, last_error=NULL
             WHERE id=1
            """
        )

    result = normalize_unambiguous_legacy_asr_topologies(config.working_db)

    assert result["state_normalized"] == 1
    assert result["resolve_state_normalized"] == 1
    assert result["downstream_invalidated"] == 0


@pytest.mark.parametrize("lease_kind", ["pipeline", "analysis"])
def test_missing_resolve_state_recovers_expired_lease_once(
    tmp_path: Path,
    lease_kind: str,
) -> None:
    config = config_for(tmp_path)
    create_ready_call_db(config.working_db)
    assignments = {
        "pipeline": (
            "resolve_status=NULL, analysis_status='pending', "
            "pipeline_stage='resolve', pipeline_worker_id='old-worker', "
            "pipeline_claimed_at='2020-01-01T00:00:00+00:00'"
        ),
        "analysis": (
            "resolve_status=NULL, analysis_status='in_progress', "
            "analysis_worker_id='old-worker', "
            "analysis_claimed_at='2020-01-01T00:00:00+00:00'"
        ),
    }
    with sqlite3.connect(config.working_db) as connection:
        connection.execute(
            f"UPDATE call_records SET {assignments[lease_kind]} WHERE id=1"
        )

    first = normalize_unambiguous_legacy_asr_topologies(config.working_db)
    repeated = normalize_unambiguous_legacy_asr_topologies(config.working_db)

    assert first["state_normalized"] == 1
    assert first["blocked"] == 0
    assert repeated["state_normalized"] == 0
    assert repeated["blocked"] == 0
    with sqlite3.connect(config.working_db) as connection:
        state = connection.execute(
            """
            SELECT resolve_status, analysis_status, pipeline_stage,
                   pipeline_worker_id, pipeline_claimed_at,
                   analysis_worker_id, analysis_claimed_at
              FROM call_records
             WHERE id=1
            """
        ).fetchone()
    assert state == ("pending", "pending", None, None, None, None, None)


@pytest.mark.parametrize("lease_kind", ["pipeline", "analysis"])
def test_missing_resolve_state_preserves_live_lease_and_blocks(
    tmp_path: Path,
    lease_kind: str,
) -> None:
    config = config_for(tmp_path)
    create_ready_call_db(config.working_db)
    claimed_at = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(config.working_db) as connection:
        if lease_kind == "pipeline":
            connection.execute(
                """
                UPDATE call_records
                   SET resolve_status=NULL, analysis_status='pending',
                       pipeline_stage='resolve', pipeline_worker_id='live-worker',
                       pipeline_claimed_at=?
                 WHERE id=1
                """,
                (claimed_at,),
            )
        else:
            connection.execute(
                """
                UPDATE call_records
                   SET resolve_status=NULL, analysis_status='in_progress',
                       analysis_worker_id='live-worker', analysis_claimed_at=?
                 WHERE id=1
                """,
                (claimed_at,),
            )

    result = normalize_unambiguous_legacy_asr_topologies(config.working_db)

    assert result["state_normalized"] == 0
    assert result["blocked_reasons"] == {"resolve_state_missing_or_leased": 1}
    with sqlite3.connect(config.working_db) as connection:
        resolve_status = connection.execute(
            "SELECT resolve_status FROM call_records WHERE id=1"
        ).fetchone()[0]
    assert resolve_status is None


@pytest.mark.parametrize(
    "second_call_id",
    ["provider-1", "\tprovider-1", "provider-1\n", "provider-1\u00a0"],
)
def test_missing_resolve_state_rejects_normalized_duplicate_call_id(
    tmp_path: Path,
    second_call_id: str,
) -> None:
    config = config_for(tmp_path)
    create_ready_call_db(config.working_db)
    with sqlite3.connect(config.working_db) as connection:
        columns = [
            str(row[1])
            for row in connection.execute("PRAGMA table_info(call_records)")
        ]
        selected = [
            "2"
            if column == "id"
            else "'/ignored/masked-2.mp3'"
            if column == "source_file"
            else column
            for column in columns
        ]
        connection.execute(
            f"INSERT INTO call_records ({','.join(columns)}) "
            f"SELECT {','.join(selected)} FROM call_records WHERE id=1"
        )
        connection.execute(
            "UPDATE call_records SET source_call_id=? WHERE id=2",
            (second_call_id,),
        )
        connection.execute("UPDATE call_records SET resolve_status=NULL")

    result = normalize_unambiguous_legacy_asr_topologies(config.working_db)

    assert result["state_normalized"] == 0
    assert result["blocked_reasons"] == {"non_unique_source_call_id": 2}
    with sqlite3.connect(config.working_db) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM call_records WHERE resolve_status IS NULL"
        ).fetchone()[0] == 2


def test_ambiguous_legacy_asr_blocks_process_a_without_workers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = config_for(tmp_path)
    create_ready_call_db(config.working_db)
    with sqlite3.connect(config.working_db) as connection:
        raw = connection.execute(
            "SELECT transcript_variants_json FROM call_records WHERE id=1"
        ).fetchone()[0]
        payload = json.loads(raw)
        payload.pop("mode")
        payload["manager"] = {
            "variant_a": "manager Whisper",
            "variant_b": "manager GigaAM",
        }
        payload["client"] = {
            "variant_a": "client Whisper",
            "variant_b": "client GigaAM",
        }
        ambiguous_raw = json.dumps(payload, ensure_ascii=False)
        connection.execute(
            "UPDATE call_records SET transcript_variants_json=? WHERE id=1",
            (ambiguous_raw,),
        )
    create_empty_capture_manifest(config)
    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.environment_preflight",
        lambda *_args, **_kwargs: {"ok": True, "codex_network_ok": True},
    )
    commands: list[list[str]] = []

    def command_runner(
        command: list[str], _env: Mapping[str, str], _cwd: Path
    ) -> dict[str, object]:
        commands.append(command)
        return {"rc": 0, "command": "unexpected"}

    report = run_process_a(
        config,
        skip_capture=True,
        command_runner=command_runner,
    )

    assert report["status"] == "partial"
    assert report["stop_reason"] == "legacy_asr_topology_blocked"
    assert report["counters"]["metadata"]["legacy_topology_blocked"] == 1
    assert report["counters"]["metadata"]["legacy_topology_blocked_reasons"] == {
        "ambiguous_or_incomplete_topology": 1
    }
    assert commands == []
    with sqlite3.connect(config.working_db) as connection:
        assert connection.execute(
            "SELECT transcript_variants_json FROM call_records WHERE id=1"
        ).fetchone()[0] == ambiguous_raw


def test_empty_legacy_analysis_blocks_normalization(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    create_ready_call_db(config.working_db)
    with sqlite3.connect(config.working_db) as connection:
        raw = connection.execute(
            "SELECT transcript_variants_json FROM call_records WHERE id=1"
        ).fetchone()[0]
        payload = json.loads(raw)
        payload.pop("mode")
        legacy_raw = json.dumps(payload, ensure_ascii=False)
        connection.execute(
            """
            UPDATE call_records
               SET transcript_variants_json=?, analysis_json='{}'
             WHERE id=1
            """,
            (legacy_raw,),
        )

    result = normalize_unambiguous_legacy_asr_topologies(config.working_db)

    assert result == {
        "normalized": 0,
        "downstream_invalidated": 0,
        "state_normalized": 0,
        "dead_letter_state_normalized": 0,
        "resolve_state_normalized": 0,
        "blocked": 1,
        "blocked_reasons": {"terminal_payload_invalid": 1},
    }
    with sqlite3.connect(config.working_db) as connection:
        assert connection.execute(
            "SELECT transcript_variants_json FROM call_records WHERE id=1"
        ).fetchone()[0] == legacy_raw


@pytest.mark.parametrize("mode", ["mono_or_fallback", "stereo"])
def test_explicit_incomplete_asr_topology_is_blocked(
    tmp_path: Path,
    mode: str,
) -> None:
    config = config_for(tmp_path)
    create_ready_call_db(config.working_db)
    with sqlite3.connect(config.working_db) as connection:
        raw = connection.execute(
            "SELECT transcript_variants_json FROM call_records WHERE id=1"
        ).fetchone()[0]
        payload = json.loads(raw)
        payload["mode"] = mode
        payload.pop("full", None)
        if mode == "stereo":
            payload["manager"] = {"variant_a": "manager Whisper"}
        invalid_raw = json.dumps(payload, ensure_ascii=False)
        connection.execute(
            "UPDATE call_records SET transcript_variants_json=? WHERE id=1",
            (invalid_raw,),
        )

    result = normalize_unambiguous_legacy_asr_topologies(config.working_db)

    assert result["normalized"] == 0
    assert result["blocked_reasons"] == {"strict_asr_topology_invalid": 1}
    assert call_db_has_open_work(config.working_db) is False


def test_terminal_analysis_waiting_for_second_asr_is_blocked(
    tmp_path: Path,
) -> None:
    config = config_for(tmp_path)
    create_ready_call_db(config.working_db)
    with sqlite3.connect(config.working_db) as connection:
        raw = connection.execute(
            "SELECT transcript_variants_json FROM call_records WHERE id=1"
        ).fetchone()[0]
        payload = json.loads(raw)
        payload["full"]["variant_b"] = ""
        connection.execute(
            "UPDATE call_records SET transcript_variants_json=? WHERE id=1",
            (json.dumps(payload, ensure_ascii=False),),
        )

    result = normalize_unambiguous_legacy_asr_topologies(config.working_db)

    assert result["blocked"] == 1
    assert result["normalized"] == 0
    assert result["blocked_reasons"] == {
        "secondary_asr_after_downstream_terminal": 1
    }
    assert call_db_has_open_work(config.working_db) is False


def test_pending_downstream_waiting_for_second_asr_remains_open(
    tmp_path: Path,
) -> None:
    config = config_for(tmp_path)
    create_ready_call_db(config.working_db)
    with sqlite3.connect(config.working_db) as connection:
        raw = connection.execute(
            "SELECT transcript_variants_json FROM call_records WHERE id=1"
        ).fetchone()[0]
        payload = json.loads(raw)
        payload["full"]["variant_b"] = ""
        connection.execute(
            """
            UPDATE call_records
               SET transcript_variants_json=?, resolve_status='pending',
                   analysis_status='pending'
             WHERE id=1
            """,
            (json.dumps(payload, ensure_ascii=False),),
        )

    result = normalize_unambiguous_legacy_asr_topologies(config.working_db)

    assert result["blocked"] == 0
    assert result["normalized"] == 0
    assert call_db_has_open_work(config.working_db) is True


def test_valid_dual_asr_exception_is_not_blocked_by_provider(
    tmp_path: Path,
) -> None:
    config = config_for(tmp_path)
    create_ready_call_db(config.working_db)
    payload = {
        "mode": "mono_or_fallback",
        "primary_provider": "approved_external_provider",
        "dual_asr_exception": {
            "approved": True,
            "reason": "synthetic audited exception",
            "approved_by": "owner",
            "approved_at": "2026-07-01T00:00:00+00:00",
        },
    }
    with sqlite3.connect(config.working_db) as connection:
        connection.execute(
            "UPDATE call_records SET transcript_variants_json=? WHERE id=1",
            (json.dumps(payload, ensure_ascii=False),),
        )

    result = normalize_unambiguous_legacy_asr_topologies(config.working_db)

    assert result["blocked"] == 0
    assert result["normalized"] == 0


@pytest.mark.parametrize("use_exception", [False, True])
def test_explicit_ready_asr_with_empty_analysis_is_blocked(
    tmp_path: Path,
    use_exception: bool,
) -> None:
    config = config_for(tmp_path)
    create_ready_call_db(config.working_db)
    with sqlite3.connect(config.working_db) as connection:
        raw = connection.execute(
            "SELECT transcript_variants_json FROM call_records WHERE id=1"
        ).fetchone()[0]
        payload = json.loads(raw)
        if use_exception:
            payload = {
                "dual_asr_exception": {
                    "approved": True,
                    "reason": "synthetic audited exception",
                    "approved_by": "owner",
                    "approved_at": "2026-07-01T00:00:00+00:00",
                }
            }
        connection.execute(
            """
            UPDATE call_records
               SET transcript_variants_json=?, analysis_json='{}'
             WHERE id=1
            """,
            (json.dumps(payload, ensure_ascii=False),),
        )

    result = normalize_unambiguous_legacy_asr_topologies(config.working_db)

    assert result["normalized"] == 0
    assert result["blocked_reasons"] == {"terminal_payload_invalid": 1}


@pytest.mark.parametrize(
    ("base_mode", "opposite_key", "opposite_value"),
    [
        ("mono", "manager", "corrupt legacy shape"),
        ("stereo", "full", []),
    ],
)
def test_legacy_topology_rejects_malformed_opposite_shape(
    tmp_path: Path,
    base_mode: str,
    opposite_key: str,
    opposite_value: object,
) -> None:
    config = config_for(tmp_path)
    create_ready_call_db(config.working_db)
    with sqlite3.connect(config.working_db) as connection:
        raw = connection.execute(
            "SELECT transcript_variants_json FROM call_records WHERE id=1"
        ).fetchone()[0]
        payload = json.loads(raw)
        payload.pop("mode")
        if base_mode == "stereo":
            payload.pop("full")
            payload["manager"] = {
                "variant_a": "manager Whisper",
                "variant_b": "manager GigaAM",
            }
            payload["client"] = {
                "variant_a": "client Whisper",
                "variant_b": "client GigaAM",
            }
        payload[opposite_key] = opposite_value
        connection.execute(
            "UPDATE call_records SET transcript_variants_json=? WHERE id=1",
            (json.dumps(payload, ensure_ascii=False),),
        )

    result = normalize_unambiguous_legacy_asr_topologies(config.working_db)

    assert result["normalized"] == 0
    assert result["blocked_reasons"] == {
        "ambiguous_or_incomplete_topology": 1
    }


def test_legacy_topology_requires_exact_provider_names(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    create_ready_call_db(config.working_db)
    with sqlite3.connect(config.working_db) as connection:
        raw = connection.execute(
            "SELECT transcript_variants_json FROM call_records WHERE id=1"
        ).fetchone()[0]
        payload = json.loads(raw)
        payload.pop("mode")
        payload["primary_provider"] = " mlx "
        payload["secondary_provider"] = "gigaam "
        connection.execute(
            "UPDATE call_records SET transcript_variants_json=? WHERE id=1",
            (json.dumps(payload, ensure_ascii=False),),
        )

    result = normalize_unambiguous_legacy_asr_topologies(config.working_db)

    assert result["normalized"] == 0
    assert result["blocked_reasons"] == {"provider_mismatch": 1}


def test_legacy_topology_with_orphan_whitespace_lease_is_blocked(
    tmp_path: Path,
) -> None:
    config = config_for(tmp_path)
    create_ready_call_db(config.working_db)
    with sqlite3.connect(config.working_db) as connection:
        raw = connection.execute(
            "SELECT transcript_variants_json FROM call_records WHERE id=1"
        ).fetchone()[0]
        payload = json.loads(raw)
        payload.pop("mode")
        connection.execute(
            """
            UPDATE call_records
               SET transcript_variants_json=?, pipeline_worker_id=' '
             WHERE id=1
            """,
            (json.dumps(payload, ensure_ascii=False),),
        )

    result = normalize_unambiguous_legacy_asr_topologies(config.working_db)

    assert result["normalized"] == 0
    assert result["blocked_reasons"] == {"non_terminal_or_leased": 1}


def test_legacy_topology_rejects_partial_opposite_shape(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    create_ready_call_db(config.working_db)
    with sqlite3.connect(config.working_db) as connection:
        raw = connection.execute(
            "SELECT transcript_variants_json FROM call_records WHERE id=1"
        ).fetchone()[0]
        payload = json.loads(raw)
        payload.pop("mode")
        payload["manager"] = {"variant_a": "partial manager"}
        connection.execute(
            "UPDATE call_records SET transcript_variants_json=? WHERE id=1",
            (json.dumps(payload, ensure_ascii=False),),
        )

    result = normalize_unambiguous_legacy_asr_topologies(config.working_db)

    assert result["normalized"] == 0
    assert result["blocked_reasons"] == {
        "ambiguous_or_incomplete_topology": 1
    }


def test_empty_dead_letter_state_is_canonicalized_before_legacy_mode(
    tmp_path: Path,
) -> None:
    config = config_for(tmp_path)
    create_ready_call_db(config.working_db)
    with sqlite3.connect(config.working_db) as connection:
        raw = connection.execute(
            "SELECT transcript_variants_json FROM call_records WHERE id=1"
        ).fetchone()[0]
        payload = json.loads(raw)
        payload.pop("mode")
        connection.execute(
            """
            UPDATE call_records
               SET transcript_variants_json=?, dead_letter_stage=''
             WHERE id=1
            """,
            (json.dumps(payload, ensure_ascii=False),),
        )

    result = normalize_unambiguous_legacy_asr_topologies(config.working_db)

    assert result["state_normalized"] == 1
    assert result["dead_letter_state_normalized"] == 1
    assert result["resolve_state_normalized"] == 0
    assert result["normalized"] == 1
    with sqlite3.connect(config.working_db) as connection:
        dead_letter, normalized_raw = connection.execute(
            """
            SELECT dead_letter_stage, transcript_variants_json
              FROM call_records
             WHERE id=1
            """
        ).fetchone()
    assert dead_letter is None
    assert json.loads(normalized_raw)["mode"] == "mono_or_fallback"


def test_process_b_returns_locked_instead_of_traceback(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    create_ready_call_db(config.ready_db)
    holder = CustomerTimelineSQLiteStore(config.timeline_db, allowed_root=config.timeline_allowed_root)
    try:
        def fake_producer(_: CallsTwoProcessesConfig, out: Path, report: Path, since: str | None) -> dict[str, object]:
            del since
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("", encoding="utf-8")
            report.write_text("{}", encoding="utf-8")
            return {"status": "ok", "events_written": 0}

        result = run_process_b(config, producer_runner=fake_producer)
    finally:
        holder.close()

    assert result["status"] == "locked"
    assert result["stop_reason"] == "timeline_writer_locked"


def test_process_b_has_its_own_nonblocking_process_lock(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    create_ready_call_db(config.ready_db)

    with process_lease(config.process_b_lock, stale_seconds=60):
        result = run_process_b(config)

    assert result["status"] == "locked"
    assert result["stop_reason"] == "process_b_locked"


def test_process_b_is_idempotent_and_keeps_one_source_system(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    create_ready_call_db(config.ready_db)
    store = CustomerTimelineSQLiteStore(config.timeline_db, allowed_root=config.timeline_allowed_root)
    store.close()

    first = run_process_b(config)
    with sqlite3.connect(f"file:{config.timeline_db}?mode=ro", uri=True) as con:
        first_count = int(
            con.execute("SELECT COUNT(*) FROM timeline_events WHERE event_type='mango_call'").fetchone()[0]
        )
    second = run_process_b(config)
    with sqlite3.connect(f"file:{config.timeline_db}?mode=ro", uri=True) as con:
        second_count = int(
            con.execute("SELECT COUNT(*) FROM timeline_events WHERE event_type='mango_call'").fetchone()[0]
        )

    assert first["status"] == "ok"
    assert second["status"] == "ok"
    assert second["stop_reason"] == ""
    assert second["counters"]["import"]["status_counts"] == {"duplicate": 3}
    assert first_count == second_count == 1
    assert call_event_source_systems(config.timeline_db) == ["mango_processed_summary"]


def test_process_b_rebuilds_lost_timeline_from_unchanged_drop(
    tmp_path: Path,
) -> None:
    config = config_for(tmp_path)
    create_ready_call_db(config.ready_db)
    with CustomerTimelineSQLiteStore(
        config.timeline_db, allowed_root=config.timeline_allowed_root
    ):
        pass
    first = run_process_b(config)
    cursor_before = read_json(config.process_b_cursor_path)
    for candidate in (
        config.timeline_db,
        Path(str(config.timeline_db) + "-wal"),
        Path(str(config.timeline_db) + "-shm"),
    ):
        candidate.unlink(missing_ok=True)

    second = run_process_b(config)

    assert first["status"] == second["status"] == "ok"
    assert read_json(config.process_b_cursor_path)["sha256"] == cursor_before["sha256"]
    with sqlite3.connect(f"file:{config.timeline_db}?mode=ro", uri=True) as con:
        assert con.execute(
            "SELECT COUNT(*) FROM timeline_events WHERE event_type='mango_call'"
        ).fetchone()[0] == 1


def test_process_b_fails_loud_when_import_validation_fails(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    create_ready_call_db(config.ready_db)
    with CustomerTimelineSQLiteStore(config.timeline_db, allowed_root=config.timeline_allowed_root):
        pass

    def fake_producer(_: CallsTwoProcessesConfig, out: Path, report: Path, since: str | None) -> dict[str, object]:
        del since
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("", encoding="utf-8")
        report.write_text("{}", encoding="utf-8")
        return {"status": "ok", "events_written": 0}

    def invalid_import(_: object) -> dict[str, object]:
        return {
            "validation_ok": False,
            "summary": {"records_read": 1, "records_accepted": 1, "writes_applied": 1},
            "writes": {"status_counts": {"updated": 1}},
            "source_system": "mango_processed_summary",
        }

    report = run_process_b(config, producer_runner=fake_producer, import_runner=invalid_import)
    assert report["status"] == "failed"
    assert report["stop_reason"] == "import_validation_failed"


def test_process_b_rejects_producer_event_count_mismatch(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    create_ready_call_db(config.ready_db)
    with CustomerTimelineSQLiteStore(config.timeline_db, allowed_root=config.timeline_allowed_root):
        pass

    def incomplete_producer(
        _: CallsTwoProcessesConfig, out: Path, report: Path, since: str | None
    ) -> dict[str, object]:
        del since
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("", encoding="utf-8")
        report.write_text("{}", encoding="utf-8")
        return {"status": "ok", "rows_selected": 1, "events_written": 0}

    result = run_process_b(config, producer_runner=incomplete_producer)

    assert result["status"] == "failed"
    assert result["stop_reason"] == "producer_event_count_mismatch"
    assert read_json(config.process_b_cursor_path) == {}


def test_process_b_normalizes_unexpected_exception_and_keeps_cursor(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    create_ready_call_db(config.ready_db)
    with CustomerTimelineSQLiteStore(config.timeline_db, allowed_root=config.timeline_allowed_root):
        pass
    write_json(config.process_b_cursor_path, {"sha256": "previous"})

    def broken_producer(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise ValueError("unexpected local producer failure")

    report = run_process_b(config, producer_runner=broken_producer)

    assert report["status"] == "failed"
    assert report["stop_reason"] == "process_b_exception:ValueError"
    assert report["counters"]["diagnostic"]["type"] == "ValueError"
    assert read_json(config.process_b_cursor_path) == {"sha256": "previous"}


def test_process_b_invalid_config_returns_normalized_failure(tmp_path: Path) -> None:
    config = config_for(tmp_path, timeline_name="customer_timeline_prod_20260713.sqlite")

    report = run_process_b(config)

    assert report["status"] == "failed"
    assert report["stop_reason"].startswith("process_b_config_exception:")
    assert report["safety"]["writes_timeline_staging"] is False


def test_process_b_returns_in_memory_failure_when_report_path_is_broken(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    create_ready_call_db(config.ready_db)
    with CustomerTimelineSQLiteStore(config.timeline_db, allowed_root=config.timeline_allowed_root):
        pass
    config.reports_dir.parent.mkdir(parents=True, exist_ok=True)
    config.reports_dir.write_text("not a directory", encoding="utf-8")

    def broken_producer(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise ValueError("producer failed")

    report = run_process_b(config, producer_runner=broken_producer)

    assert report["status"] == "failed"
    assert report["stop_reason"] == "process_b_finalize_exception:FileExistsError"
    assert report["counters"]["original_stop_reason"] == "process_b_exception:ValueError"
    assert report["safety"]["writes_timeline_staging"] is False


def test_ingest_failure_counts_are_visible_in_compact_worker_report(tmp_path: Path) -> None:
    command = [
        sys.executable,
        "-c",
        "import json; print(json.dumps({'processed': 2, 'inserted': 1, 'failed': 1, 'failure_types': {'ValueError': 1}}))",
        "ingest",
    ]

    raw = run_command(command, os.environ, tmp_path)
    compact = compact_command_reports([raw])

    assert compact[0]["command"] == "ingest"
    assert compact[0]["metrics"]["failed"] == 1
    assert compact[0]["metrics"]["failure_types"] == {"ValueError": 1}


def test_process_b_does_not_skip_late_old_call_by_timestamp_cursor(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    create_ready_call_db(config.ready_db)
    store = CustomerTimelineSQLiteStore(config.timeline_db, allowed_root=config.timeline_allowed_root)
    store.close()
    first = run_process_b(config)
    assert first["status"] == "ok"
    old_sha = read_json(config.process_b_cursor_path)["sha256"]
    with sqlite3.connect(config.ready_db) as con:
        con.execute("UPDATE call_records SET duration_sec=61 WHERE id=1")
    # Re-seal the drop the way publish_ready_db does: the manifest describes the
    # republished sqlite, while the process B cursor still holds the old sha.
    write_json(
        config.ready_db.with_suffix(".manifest.json"),
        {
            "status": "ready",
            "quick_check": "ok",
            "sha256": sha256_file(config.ready_db),
            "size_bytes": config.ready_db.stat().st_size,
        },
    )
    assert read_json(config.process_b_cursor_path)["sha256"] == old_sha
    seen_since: list[str | None] = []

    def fake_producer(_: CallsTwoProcessesConfig, out: Path, report: Path, since: str | None) -> dict[str, object]:
        seen_since.append(since)
        out.write_text("", encoding="utf-8")
        report.write_text("{}", encoding="utf-8")
        return {"status": "ok", "events_written": 0}

    second = run_process_b(config, producer_runner=fake_producer)

    assert second["status"] == "ok"
    assert seen_since == [None]
    assert second["counters"]["producer_scan_mode"] == "full_drop_dedupe"


def test_controlled_process_b_reuses_exact_imported_drop_without_db_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(
        config_for(tmp_path),
        runtime_authority_mode="isolated_controlled",
    )
    request = ControlledCaptureRequest(
        source_call_id="TARGET",
        expected_count=1,
        since=datetime(2026, 8, 10, 10, tzinfo=timezone.utc),
        until=datetime(2026, 8, 10, 10, 30, tzinfo=timezone.utc),
        pipeline_root=config.pipeline_root,
        tenant_id=config.tenant_id,
        code_sha="a" * 40,
        host_id="m1-host",
        request_path=tmp_path / "request.json",
        request_sha256="b" * 64,
    )
    monkeypatch.setattr(
        calls_runtime,
        "controlled_capture_request_for_config",
        lambda _config: request,
    )
    create_ready_call_db(config.ready_db)
    with CustomerTimelineSQLiteStore(
        config.timeline_db,
        allowed_root=config.timeline_allowed_root,
    ):
        pass
    fingerprint = calls_runtime.ready_drop_fingerprint(config)
    write_json(
        config.process_b_cursor_path,
        {
            "schema_version": "mango_calls_process_b_cursor_v1",
            "sha256": fingerprint["sha256"],
            "size_bytes": fingerprint["size_bytes"],
        },
    )
    monkeypatch.setattr(
        calls_runtime,
        "controlled_timeline_effect_snapshot",
        lambda *_args: {
            "state": "present",
            "total_rows": 1,
            "target_rows": 1,
            "mango_rows": 1,
            "quick_check": "ok",
            "logical_sha256": "a" * 64,
        },
    )
    before = calls_runtime.controlled_timeline_effect_snapshot(config, "TARGET")
    producer_called = False

    def forbidden_producer(*_args: object, **_kwargs: object) -> Mapping[str, object]:
        nonlocal producer_called
        producer_called = True
        raise AssertionError("controlled replay must not produce or import")

    report = calls_runtime._run_process_b(
        config,
        producer_runner=forbidden_producer,
    )
    after = calls_runtime.controlled_timeline_effect_snapshot(config, "TARGET")

    assert report["status"] == "idle"
    assert report["stop_reason"] == "controlled_drop_already_imported"
    assert report["safety"]["writes_timeline_staging"] is False
    assert before == after
    assert producer_called is False


def test_controlled_process_b_does_not_skip_when_timeline_target_is_lost(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(
        config_for(tmp_path),
        runtime_authority_mode="isolated_controlled",
    )
    create_ready_call_db(config.ready_db)
    with CustomerTimelineSQLiteStore(
        config.timeline_db,
        allowed_root=config.timeline_allowed_root,
    ):
        pass
    fingerprint = calls_runtime.ready_drop_fingerprint(config)
    write_json(
        config.process_b_cursor_path,
        {
            "schema_version": "mango_calls_process_b_cursor_v1",
            "sha256": fingerprint["sha256"],
            "size_bytes": fingerprint["size_bytes"],
        },
    )
    request = ControlledCaptureRequest(
        source_call_id="TARGET",
        expected_count=1,
        since=datetime(2026, 8, 10, 10, tzinfo=timezone.utc),
        until=datetime(2026, 8, 10, 10, 30, tzinfo=timezone.utc),
        pipeline_root=config.pipeline_root,
        tenant_id=config.tenant_id,
        code_sha="a" * 40,
        host_id="m1-host",
        request_path=tmp_path / "request.json",
        request_sha256="b" * 64,
    )
    monkeypatch.setattr(
        calls_runtime,
        "controlled_capture_request_for_config",
        lambda _config: request,
    )
    producer_called = False

    def producer(
        _config: CallsTwoProcessesConfig,
        out: Path,
        report: Path,
        _since: str | None,
    ) -> Mapping[str, object]:
        nonlocal producer_called
        producer_called = True
        out.write_text("", encoding="utf-8")
        report.write_text("{}", encoding="utf-8")
        return {"status": "ok", "rows_selected": 0, "events_written": 0}

    def importer(_config: object) -> Mapping[str, object]:
        return {
            "validation_ok": True,
            "summary": {
                "records_read": 0,
                "records_accepted": 0,
                "records_rejected": 0,
                "writes_applied": 0,
            },
            "writes": {"status_counts": {}},
            "source_system": "mango_processed_summary",
        }

    report = calls_runtime._run_process_b(
        config,
        producer_runner=producer,
        import_runner=importer,
    )
    assert producer_called is True
    assert report["stop_reason"] != "controlled_drop_already_imported"


def test_parent_lifeline_wrapper_normal_exit_and_parent_sigkill(
    tmp_path: Path,
) -> None:
    normal = run_command(
        [sys.executable, "-c", "print('ok')"],
        os.environ,
        tmp_path / "normal",
        parent_lifeline=True,
    )
    assert normal["rc"] == 0

    child_pid_path = tmp_path / "child.pid"
    child_code = (
        "import os,sys,time;"
        "open(sys.argv[1],'w').write(str(os.getpid()));"
        "time.sleep(60)"
    )
    orchestrator_code = "\n".join(
        (
            "import os, subprocess, sys, time",
            "read_fd, write_fd = os.pipe()",
            "env = dict(os.environ)",
            "env['MANGO_CALLS_CONTROLLED_LIFELINE_FD'] = str(read_fd)",
            f"command = {parent_lifeline_subprocess_command([sys.executable, '-c', child_code, str(child_pid_path)])!r}",
            "proc = subprocess.Popen(command, env=env, pass_fds=(read_fd,), start_new_session=True)",
            "os.close(read_fd)",
            "print(proc.pid, flush=True)",
            "time.sleep(60)",
        )
    )
    orchestrator = subprocess.Popen(
        [sys.executable, "-c", orchestrator_code],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    assert orchestrator.stdout is not None
    helper_pid = int(orchestrator.stdout.readline().strip())
    deadline = time.monotonic() + 10
    while not child_pid_path.is_file() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert child_pid_path.is_file()
    child_pid = int(child_pid_path.read_text(encoding="utf-8"))
    os.kill(orchestrator.pid, signal.SIGKILL)
    orchestrator.wait(timeout=10)
    deadline = time.monotonic() + 10
    state = ""
    while time.monotonic() < deadline:
        state = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(child_pid)],
            check=False,
            text=True,
            capture_output=True,
        ).stdout.strip()
        if not state or state.startswith("Z"):
            break
        time.sleep(0.05)
    try:
        assert not state or state.startswith("Z")
    finally:
        try:
            os.killpg(helper_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_foton_pdn_sweep_blocks_phone() -> None:
    with pytest.raises(RuntimeError, match="pdn-sweep"):
        assert_no_pdn({"text": "Позвонить +7 999 123-45-67"})
    assert_no_pdn({"calls": 22, "status": "ok"})


def test_run_id_uuid_prefix_cannot_be_mistaken_for_phone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixed_uuid = type("FixedUuid", (), {"hex": "81234567890abcdef01234567890abcd"})()
    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.uuid.uuid4",
        lambda: fixed_uuid,
    )

    run_id = new_calls_run_id(datetime(2026, 8, 8, 10, 0, tzinfo=timezone.utc))

    assert run_id.endswith("-u81234567890a")
    assert_no_pdn({"run_id": run_id})


def test_locked_report_does_not_claim_work_or_publish_pid() -> None:
    payload = safe_daily_payload(
        {
            "schema_version": "v1",
            "run_id": "masked",
            "process": "process_a",
            "status": "locked",
            "stop_reason": "process_a_locked",
            "counters": {"lock": {"pid": 12345, "previous_pid": 111}},
            "safety": {"runs_asr": False, "runs_resolve_analyze": False},
        }
    )

    assert payload["counters"]["lock"] == {}
    assert payload["safety"]["runs_asr"] is False


def test_launchd_installer_defaults_to_near_realtime_900_seconds(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    config_path = tmp_path / "config.json"
    env_path = tmp_path / "mango.env"
    plist_path = tmp_path / "calls.plist"
    config_path.write_text(
        json.dumps(
            {
                "pipeline_root": str(config.pipeline_root),
                "timeline_db": str(config.timeline_db),
                "timeline_allowed_root": str(config.timeline_allowed_root),
                "python_executable": str(config.python_executable),
                "codex_binary": str(config.codex_binary),
                "codex_home_root": str(config.codex_home_root),
                "poll_overlap_minutes": 30,
                "require_cutover_authority": False,
                "strict_ready_provenance": False,
            }
        ),
        encoding="utf-8",
    )
    config_path.chmod(0o600)
    env_path.write_text("MANGO_OFFICE_API_KEY=x\nMANGO_OFFICE_API_SALT=y\n", encoding="utf-8")
    env_path.chmod(0o600)

    subprocess.run(
        [
            sys.executable,
            "scripts/install_mango_calls_two_processes_service.py",
            "--config",
            str(config_path),
            "--env-file",
            str(env_path),
            "--out",
            str(plist_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        stdout=subprocess.DEVNULL,
    )

    with plist_path.open("rb") as handle:
        payload = plistlib.load(handle)
    assert payload["StartInterval"] == 900


def test_process_b_fails_loud_on_stale_drop_manifest(tmp_path: Path) -> None:
    """A drop manifest that no longer matches the sealed sqlite must stop the
    import instead of passing as success: `ready_drop_fingerprint` already
    computes `manifest_mismatch`, and process B must honour it."""
    config = replace(config_for(tmp_path), manifest_recheck_sleep_sec=0.0)
    create_ready_call_db(config.ready_db)
    with CustomerTimelineSQLiteStore(config.timeline_db, allowed_root=config.timeline_allowed_root):
        pass
    write_json(
        config.ready_db.with_suffix(".manifest.json"),
        {
            "status": "ready",
            "sha256": "0" * 64,
            "size_bytes": config.ready_db.stat().st_size + 1,
            "quick_check": "ok",
        },
    )

    report = run_process_b(config)

    assert report["status"] == "failed"
    assert report["stop_reason"] == "drop_manifest_mismatch"
    assert report["counters"]["drop"]["manifest_mismatch"] is True
    with sqlite3.connect(f"file:{config.timeline_db}?mode=ro", uri=True) as con:
        written = int(
            con.execute("SELECT COUNT(*) FROM timeline_events WHERE event_type='mango_call'").fetchone()[0]
        )
    assert written == 0
    assert read_json(config.process_b_cursor_path) == {}


@pytest.mark.parametrize("manifest", [None, {"status": "ready", "quick_check": "ok"}])
def test_process_b_rejects_missing_or_incomplete_manifest(
    tmp_path: Path, manifest: dict[str, str] | None
) -> None:
    config = replace(config_for(tmp_path), manifest_recheck_sleep_sec=0.0)
    create_ready_call_db(config.ready_db)
    manifest_path = config.ready_db.with_suffix(".manifest.json")
    if manifest is None:
        manifest_path.unlink()
    else:
        write_json(manifest_path, manifest)
    with CustomerTimelineSQLiteStore(config.timeline_db, allowed_root=config.timeline_allowed_root):
        pass

    report = run_process_b(config)

    assert report["status"] == "failed"
    assert report["stop_reason"] == "drop_manifest_invalid"
    assert report["counters"]["drop"]["manifest_valid"] is False
    assert read_json(config.process_b_cursor_path) == {}


def test_prepare_ingest_inputs_counts_missing_capture_audio(tmp_path: Path) -> None:
    """A manifest row marked `downloaded` whose audio no longer exists must be
    counted in the report, not silently dropped."""
    from mango_mvp.productization.capture_staging import CaptureManifestStore, ManifestEntry

    config = config_for(tmp_path)
    config.recordings_dir.mkdir(parents=True, exist_ok=True)
    present = config.recordings_dir / "present.mp3"
    present.write_bytes(b"audio-bytes")
    empty = config.recordings_dir / "empty.mp3"
    empty.write_bytes(b"")

    store = CaptureManifestStore(config.capture_manifest)

    def entry(event_key: str, audio_path: str) -> ManifestEntry:
        return ManifestEntry(
            schema_version="v1",
            created_at="2026-07-09T10:00:00+00:00",
            tenant_id="foton",
            provider="mango",
            event_key=event_key,
            provider_call_id=event_key,
            recording_id=event_key,
            started_at="2026-07-09T10:00:00+00:00",
            ended_at=None,
            direction="inbound",
            client_phone=None,
            manager_ref=None,
            status="downloaded",
            local_audio_path=audio_path,
        )

    store.append(entry("ok-1", str(present)))
    store.append(entry("gone-1", str(config.recordings_dir / "vanished.mp3")))
    store.append(entry("empty-1", str(empty)))

    result = prepare_ingest_inputs(config)

    assert result["audio_files"] == 1
    assert result["skipped"] == {"audio_file_missing": 1, "audio_file_empty": 1}
    assert result["skipped_total"] == 2


def test_prepare_ingest_inputs_keeps_torn_manifest_tail_incomplete(tmp_path: Path) -> None:
    from mango_mvp.productization.capture_staging import CaptureManifestStore, ManifestEntry

    config = config_for(tmp_path)
    audio = config.recordings_dir / "present.mp3"
    audio.parent.mkdir(parents=True, exist_ok=True)
    audio.write_bytes(b"audio-bytes")
    CaptureManifestStore(config.capture_manifest).append(
        ManifestEntry(
            schema_version="v1",
            created_at="2026-07-09T10:00:00+00:00",
            tenant_id="foton",
            provider="mango",
            event_key="present-1",
            provider_call_id="present-1",
            recording_id="present-1",
            started_at="2026-07-09T10:00:00+00:00",
            ended_at=None,
            direction="inbound",
            client_phone=None,
            manager_ref=None,
            status="downloaded",
            local_audio_path=str(audio),
        )
    )
    with config.capture_manifest.open("ab") as handle:
        handle.write(b'{"event_key":"unfinished"')

    result = prepare_ingest_inputs(config)

    assert result["audio_files"] == 1
    assert result["incomplete_trailing_manifest_records"] == 1
    assert result["incomplete_total"] == 1


def test_process_a_processes_available_audio_then_marks_missing_partial(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from mango_mvp.productization.capture_staging import CaptureManifestStore, ManifestEntry

    config = replace(config_for(tmp_path), min_free_gib=1)
    store = CaptureManifestStore(config.capture_manifest)

    def entry(event_key: str, audio_path: str) -> ManifestEntry:
        return ManifestEntry(
            schema_version="v1",
            created_at="2026-07-09T10:00:00+00:00",
            tenant_id="foton",
            provider="mango",
            event_key=event_key,
            provider_call_id=event_key,
            recording_id=event_key,
            started_at="2026-07-09T10:00:00+00:00",
            ended_at=None,
            direction="inbound",
            client_phone=None,
            manager_ref=None,
            status="downloaded",
            local_audio_path=audio_path,
        )
    present = config.recordings_dir / "present.mp3"
    present.parent.mkdir(parents=True, exist_ok=True)
    present.write_bytes(b"audio")
    store.append(entry("present-1", str(present)))
    store.append(entry("gone-1", str(config.recordings_dir / "missing.mp3")))
    create_ready_call_db(config.working_db)
    commands: list[str] = []
    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.environment_preflight",
        lambda *_args, **_kwargs: {"ok": True, "codex_network_ok": True},
    )

    def fake_command(command: list[str], _env: dict[str, str], _cwd: Path) -> dict[str, object]:
        commands.append(" ".join(command))
        return {"rc": 0, "command": command[-1]}

    report = run_process_a(
        config,
        skip_capture=True,
        skip_workers=False,
        command_runner=fake_command,
    )

    assert report["status"] == "partial"
    assert report["stop_reason"] == "capture_asset_integrity_failed"
    assert report["counters"]["metadata"]["skipped_total"] == 1
    assert report["counters"]["drop"]["status"] == "blocked"
    assert not config.ready_db.exists()
    assert any(" ingest " in f" {command} " for command in commands)
    assert read_json(config.cursor_path) == {}


def test_process_a_partial_capture_publishes_available_work_and_advances_cursor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = replace(config_for(tmp_path), min_free_gib=1)
    create_ready_call_db(config.working_db)
    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.environment_preflight",
        lambda *_args, **_kwargs: {"ok": True, "codex_network_ok": True},
    )
    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.prepare_ingest_inputs",
        lambda _config: {"audio_files": 0, "skipped_total": 0},
    )

    report = run_process_a(
        config,
        since="2026-07-09T09:00:00+00:00",
        until="2026-07-09T10:00:00+00:00",
        skip_workers=True,
        capture_runner=lambda *_args: {"status": "partial", "downloaded": 1, "failed": 1},
    )

    assert report["status"] == "partial"
    assert report["counters"]["drop"]["status"] == "ready"
    assert report["downstream_ready"] is True
    assert config.ready_db.exists()
    assert read_json(config.cursor_path)["until"] == "2026-07-09T10:00:00+00:00"


def test_process_a_reports_recovered_manifest_tail_and_blocks_downstream(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from mango_mvp.productization.capture_staging import CaptureManifestStore, ManifestEntry

    config = replace(config_for(tmp_path), min_free_gib=1)
    create_ready_call_db(config.working_db)
    store = CaptureManifestStore(config.capture_manifest)

    def terminal_entry(call_id: str) -> ManifestEntry:
        return ManifestEntry(
            schema_version="v1",
            created_at="2026-07-09T08:00:00+00:00",
            tenant_id="foton",
            provider="mango",
            event_key=f"foton:mango:{call_id}",
            provider_call_id=call_id,
            recording_id=None,
            started_at="2026-07-09T08:00:00+00:00",
            ended_at=None,
            direction="inbound",
            client_phone=None,
            manager_ref=None,
            status="recording_retry_expired",
            error="recording_missing_after_retry_ttl",
            remediation_code="manual_review_or_retry_if_recording_appears",
        )

    store.append(terminal_entry("before-crash"))
    with config.capture_manifest.open("ab") as handle:
        handle.write(b'{"event_key":"unfinished"')

    def recover_during_capture(
        _config: CallsTwoProcessesConfig,
        _since: datetime,
        _until: datetime,
    ) -> dict[str, object]:
        CaptureManifestStore(config.capture_manifest).append(terminal_entry("after-restart"))
        return {"status": "ok"}

    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.environment_preflight",
        lambda *_args, **_kwargs: {"ok": True, "codex_network_ok": True},
    )

    report = run_process_a(
        config,
        since="2026-07-09T09:00:00+00:00",
        until="2026-07-09T10:00:00+00:00",
        skip_workers=True,
        capture_runner=recover_during_capture,
    )

    assert report["status"] == "partial"
    assert report["stop_reason"] == "capture_manifest_tail_incomplete"
    assert report["downstream_ready"] is False
    assert report["counters"]["drop"] == {
        "status": "blocked",
        "reason": "capture_manifest_tail_incomplete",
    }
    assert report["counters"]["metadata"]["recovered_trailing_manifest_records"] == 1
    assert not config.ready_db.exists()
    assert read_json(config.cursor_path) == {}
    assert CaptureManifestStore(config.capture_manifest).recovered_trailing_records == 0
    partial_report_path = Path(str(report["report_path"]))
    partial_report_bytes = partial_report_path.read_bytes()

    clean_retry = run_process_a(config, skip_capture=True, skip_workers=True)

    assert clean_retry["status"] == "ok"
    assert clean_retry["downstream_ready"] is True
    assert len(CaptureManifestStore(config.capture_manifest).read_entries()) == 2
    assert Path(str(clean_retry["report_path"])) != partial_report_path
    assert partial_report_path.read_bytes() == partial_report_bytes
    assert read_json(partial_report_path)["stop_reason"] == "capture_manifest_tail_incomplete"


def test_process_a_report_failure_leaves_recovery_unacknowledged(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from mango_mvp.productization.capture_staging import CaptureManifestStore, ManifestEntry

    config = replace(config_for(tmp_path), min_free_gib=1)
    create_ready_call_db(config.working_db)

    def terminal_entry(call_id: str) -> ManifestEntry:
        return ManifestEntry(
            schema_version="v1",
            created_at="2026-07-09T08:00:00+00:00",
            tenant_id="foton",
            provider="mango",
            event_key=f"foton:mango:{call_id}",
            provider_call_id=call_id,
            recording_id=None,
            started_at="2026-07-09T08:00:00+00:00",
            ended_at=None,
            direction="inbound",
            client_phone=None,
            manager_ref=None,
            status="recording_retry_expired",
        )

    store = CaptureManifestStore(config.capture_manifest)
    store.append(terminal_entry("before-crash"))
    with config.capture_manifest.open("ab") as handle:
        handle.write(b'{"event_key":"unfinished"')

    def recover_during_capture(*_args: object) -> dict[str, object]:
        CaptureManifestStore(config.capture_manifest).append(terminal_entry("after-restart"))
        return {"status": "ok"}

    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.environment_preflight",
        lambda *_args, **_kwargs: {"ok": True, "codex_network_ok": True},
    )

    from mango_mvp.productization.capture_staging import atomic_write_private_json as real_atomic_write

    def fail_report_write(path: Path, payload: object, **kwargs: object) -> None:
        if path.parent == config.reports_dir:
            raise OSError("synthetic report failure")
        real_atomic_write(path, payload, **kwargs)

    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.atomic_write_private_json",
        fail_report_write,
    )

    with pytest.raises(OSError, match="synthetic report failure"):
        run_process_a(
            config,
            since="2026-07-09T09:00:00+00:00",
            until="2026-07-09T10:00:00+00:00",
            skip_workers=True,
            capture_runner=recover_during_capture,
        )

    restarted = CaptureManifestStore(config.capture_manifest)
    assert restarted.recovered_trailing_records == 1
    assert restarted.recovery_incident_sha256
    assert read_json(config.process_a_status_path)["status"] == "failed"
    assert pipeline_freshness(config)["stages"]["process_a"]["status"] == "failed"
    assert read_json(config.cursor_path) == {}
    assert not config.ready_db.exists()


@pytest.mark.parametrize("raise_after_ack", [False, True])
def test_process_a_ack_failure_preserves_partial_incident_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    raise_after_ack: bool,
) -> None:
    from mango_mvp.productization.capture_staging import (
        CaptureManifestStore,
        ManifestEntry,
        acknowledge_capture_recovery as real_acknowledge,
    )

    config = replace(config_for(tmp_path), min_free_gib=1, foton_daily_dir=tmp_path / "daily")
    create_ready_call_db(config.working_db)

    def terminal_entry(call_id: str) -> ManifestEntry:
        return ManifestEntry(
            schema_version="v1",
            created_at="2026-07-09T08:00:00+00:00",
            tenant_id="foton",
            provider="mango",
            event_key=f"foton:mango:{call_id}",
            provider_call_id=call_id,
            recording_id=None,
            started_at="2026-07-09T08:00:00+00:00",
            ended_at=None,
            direction="inbound",
            client_phone=None,
            manager_ref=None,
            status="recording_retry_expired",
        )

    store = CaptureManifestStore(config.capture_manifest)
    store.append(terminal_entry("before-crash"))
    with config.capture_manifest.open("ab") as handle:
        handle.write(b'{"event_key":"unfinished"')

    def recover_during_capture(*_args: object) -> dict[str, object]:
        CaptureManifestStore(config.capture_manifest).append(terminal_entry("after-restart"))
        return {"status": "ok"}

    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.environment_preflight",
        lambda *_args, **_kwargs: {"ok": True, "codex_network_ok": True},
    )

    def fail_acknowledge(*args: object, **kwargs: object) -> int:
        if raise_after_ack:
            real_acknowledge(*args, **kwargs)
        raise RuntimeError("synthetic acknowledgement failure")

    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.acknowledge_capture_recovery",
        fail_acknowledge,
    )

    failed = run_process_a(
        config,
        since="2026-07-09T09:00:00+00:00",
        until="2026-07-09T10:00:00+00:00",
        skip_workers=True,
        capture_runner=recover_during_capture,
    )

    reports = [read_json(path) for path in sorted(config.reports_dir.glob("*_process_a.json"))]
    partial = [report for report in reports if report.get("status") == "partial"]
    failures = [report for report in reports if report.get("status") == "failed"]
    restarted = CaptureManifestStore(config.capture_manifest)

    assert failed["status"] == "failed"
    assert len(reports) == 2
    assert len(partial) == 1
    assert len(failures) == 1
    assert failed["run_id"] != partial[0]["run_id"]
    assert partial[0]["stop_reason"] == "capture_manifest_tail_incomplete"
    assert failures[0]["counters"]["diagnostic"]["preserved_report_run_id"] == partial[0]["run_id"]
    daily_failure = read_json(Path(str(failed["daily_report_path"])))
    assert daily_failure["counters"]["preserved_report"]["run_id"] == partial[0]["run_id"]
    assert "diagnostic" not in daily_failure["counters"]
    assert restarted.recovered_trailing_records == (0 if raise_after_ack else 1)
    assert read_json(config.cursor_path) == {}
    assert not config.ready_db.exists()


def test_process_a_runs_workers_for_existing_open_db_work_without_reingest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = replace(config_for(tmp_path), min_free_gib=1)
    create_ready_call_db(config.working_db)
    with sqlite3.connect(config.working_db) as con:
        con.execute("UPDATE call_records SET transcription_status='pending', analysis_status='pending'")
    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.environment_preflight",
        lambda *_args, **_kwargs: {"ok": True, "codex_network_ok": True},
    )
    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.prepare_ingest_inputs",
        lambda _config: {"audio_files": 0, "skipped_total": 0},
    )
    commands: list[str] = []

    def command_runner(command: list[str], *_args: object) -> dict[str, object]:
        commands.append(" ".join(command))
        return {"rc": 0, "command": command[-1]}

    report = run_process_a(config, skip_capture=True, command_runner=command_runner)

    assert report["status"] == "ok"
    assert report["counters"]["metadata"]["db_open_work"] is True
    assert any("worker" in command for command in commands)
    assert not any(" init-db" in command or " ingest" in command for command in commands)


def test_process_a_stops_heavy_prelude_after_init_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = replace(config_for(tmp_path), min_free_gib=1)
    create_ready_call_db(config.working_db)
    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.environment_preflight",
        lambda *_args, **_kwargs: {"ok": True, "codex_network_ok": True},
    )
    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.prepare_ingest_inputs",
        lambda _config: {"audio_files": 1, "skipped_total": 0},
    )
    commands: list[str] = []

    def fail_init(command: list[str], *_args: object) -> dict[str, object]:
        commands.append(" ".join(command))
        return {"rc": 9, "command": command[-1]}

    report = run_process_a(
        config,
        skip_capture=True,
        command_runner=fail_init,
    )

    assert report["status"] == "failed"
    assert report["stop_reason"] == "worker_command_failed"
    assert len(commands) == 1
    assert "init-db" in commands[0]
    assert " ingest " not in f" {commands[0]} "


def test_process_a_complete_existing_db_runs_no_commands(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = replace(config_for(tmp_path), min_free_gib=1)
    create_ready_call_db(config.working_db)
    source = config.recordings_dir / "known.mp3"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"audio")
    from mango_mvp.productization.capture_staging import CaptureManifestStore, ManifestEntry

    CaptureManifestStore(config.capture_manifest).append(
        ManifestEntry(
            schema_version="v1", created_at="2026-07-09T10:00:00+00:00", tenant_id="foton",
            provider="mango", event_key="foton:mango:provider-1", provider_call_id="provider-1",
            recording_id="rec-1", started_at="2026-07-09T10:00:00+00:00", ended_at=None,
            direction="inbound", client_phone=None, manager_ref=None, status="downloaded",
            local_audio_path=str(source),
        )
    )
    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.environment_preflight",
        lambda *_args, **_kwargs: {"ok": True, "codex_network_ok": True},
    )
    first = run_process_a(
        config,
        skip_capture=True,
        command_runner=lambda *_args: pytest.fail("complete second run must stay empty"),
    )
    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.publish_ready_db",
        lambda *_args, **_kwargs: pytest.fail("unchanged second run must reuse ready DB"),
    )
    report = run_process_a(
        config,
        skip_capture=True,
        command_runner=lambda *_args: pytest.fail("complete second run must stay empty"),
    )

    assert first["status"] == "ok"
    assert report["status"] == "ok"
    assert report["counters"]["metadata"]["db_open_work"] is False
    assert report["counters"]["drop"]["reused"] is True


def test_call_db_open_work_includes_interrupted_secondary_backfill(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    create_ready_call_db(config.working_db)
    with sqlite3.connect(config.working_db) as con:
        con.execute(
            "UPDATE call_records SET transcription_status='done', resolve_status='done', "
            "analysis_status='done', pipeline_stage='backfill-second-asr'"
        )

    assert call_db_has_open_work(config.working_db) is True


@pytest.mark.parametrize(
    ("stage", "status_update"),
    (
        (
            "transcribe",
            "transcription_status='in_progress', resolve_status='pending', "
            "analysis_status='pending'",
        ),
        (
            "resolve",
            "transcription_status='done', resolve_status='in_progress', "
            "analysis_status='pending'",
        ),
    ),
)
def test_interrupted_primary_or_resolve_lease_is_recoverable_open_work(
    tmp_path: Path, stage: str, status_update: str
) -> None:
    config = config_for(tmp_path)
    create_ready_call_db(config.working_db)
    with sqlite3.connect(config.working_db) as con:
        con.execute(
            f"UPDATE call_records SET {status_update}, pipeline_stage=?, "
            "pipeline_worker_id='killed-worker', "
            "pipeline_claimed_at='2020-01-01T00:00:00+00:00'",
            (stage,),
        )

    assert call_db_has_open_work(config.working_db) is True


def test_call_db_open_work_excludes_future_retry_and_exhausted_attempts(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    create_ready_call_db(config.working_db)
    with sqlite3.connect(config.working_db) as con:
        con.execute(
            "UPDATE call_records SET transcription_status='failed', transcribe_attempts=3, "
            "next_retry_at='2099-01-01T00:00:00+00:00'"
        )

    assert call_db_has_open_work(config.working_db) is False


def test_call_db_open_work_includes_stale_analyze_claim(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    create_ready_call_db(config.working_db)
    with sqlite3.connect(config.working_db) as con:
        con.execute(
            "UPDATE call_records SET transcription_status='done', resolve_status='done', "
            "analysis_status='in_progress', analysis_claimed_at='2020-01-01T00:00:00+00:00'"
        )

    assert call_db_has_open_work(config.working_db) is True


def test_call_db_open_work_excludes_terminal_secondary_asr_retry(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    create_ready_call_db(config.working_db)
    payload = {
        "mode": "mono_or_fallback",
        "secondary_provider": "gigaam",
        "full": {"variant_a": "текст", "variant_b": ""},
    }
    with sqlite3.connect(config.working_db) as con:
        con.execute(
            "UPDATE call_records SET transcription_status='done', resolve_status='done', "
            "analysis_status='done', transcript_variants_json=?",
            (json.dumps(payload, ensure_ascii=False),),
        )

    assert call_db_has_open_work(config.working_db) is False


def test_missing_downloaded_capture_is_returned_for_recovery(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    from mango_mvp.productization.capture_staging import CaptureManifestStore, ManifestEntry

    CaptureManifestStore(config.capture_manifest).append(
        ManifestEntry(
            schema_version="v1",
            created_at="2026-07-09T10:00:00+00:00",
            tenant_id="foton",
            provider="mango",
            event_key="foton:mango:gone-1",
            provider_call_id="gone-1",
            recording_id="recording-1",
            started_at="2026-07-09T10:00:00+00:00",
            ended_at=None,
            direction="inbound",
            client_phone=None,
            manager_ref=None,
            status="downloaded",
            local_audio_path=str(config.recordings_dir / "missing.mp3"),
        )
    )

    recovered = missing_capture_recovery_events(config)

    assert len(recovered) == 1
    assert recovered[0].provider_call_id == "gone-1"
    assert recovered[0].recording_ref == "recording-1"


def test_failed_capture_stays_in_recovery_queue(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    from mango_mvp.productization.capture_staging import CaptureManifestStore, ManifestEntry

    CaptureManifestStore(config.capture_manifest).append(
        ManifestEntry(
            schema_version="v1",
            created_at="2026-07-09T10:00:00+00:00",
            tenant_id="foton",
            provider="mango",
            event_key="foton:mango:failed-1",
            provider_call_id="failed-1",
            recording_id="recording-1",
            started_at="2026-07-09T10:00:00+00:00",
            ended_at=None,
            direction="inbound",
            client_phone=None,
            manager_ref=None,
            status="failed",
            local_audio_path=str(config.recordings_dir / "failed.mp3"),
        )
    )

    recovered = missing_capture_recovery_events(config)

    assert [event.provider_call_id for event in recovered] == ["failed-1"]


def test_expired_capture_with_known_recording_is_rechecked_at_most_daily(
    tmp_path: Path,
) -> None:
    config = config_for(tmp_path)
    from mango_mvp.productization.capture_staging import CaptureManifestStore, ManifestEntry

    attempted_at = datetime.now(timezone.utc) - timedelta(hours=25)
    CaptureManifestStore(config.capture_manifest).append(
        ManifestEntry(
            schema_version="v1",
            created_at=attempted_at.isoformat(),
            tenant_id="foton",
            provider="mango",
            event_key="foton:mango:expired-known",
            provider_call_id="expired-known",
            recording_id="recording-known",
            recording_ids=("recording-known",),
            started_at="2026-07-09T10:00:00+00:00",
            ended_at=None,
            direction="inbound",
            client_phone=None,
            manager_ref=None,
            status="recording_retry_expired",
            error="recording_missing_after_retry_ttl",
        )
    )

    assert missing_capture_recovery_events(
        config, now=attempted_at + timedelta(hours=23)
    ) == ()
    due = missing_capture_recovery_events(
        config, now=attempted_at + timedelta(hours=24)
    )
    assert [event.provider_call_id for event in due] == ["expired-known"]


def test_recovery_is_not_filtered_by_known_processed_ids(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = replace(config_for(tmp_path), api_window_hours=1)
    from mango_mvp.productization.capture_staging import CaptureManifestStore, ManifestEntry

    CaptureManifestStore(config.capture_manifest).append(
        ManifestEntry(
            schema_version="v1",
            created_at="2026-07-09T10:00:00+00:00",
            tenant_id="foton",
            provider="mango",
            event_key="foton:mango:failed-1",
            provider_call_id="failed-1",
            recording_id="recording-1",
            started_at="2026-07-09T10:00:00+00:00",
            ended_at=None,
            direction="inbound",
            client_phone=None,
            manager_ref=None,
            status="failed",
            local_audio_path=str(config.recordings_dir / "failed.mp3"),
        )
    )

    class FakeClient:
        def __init__(self, **_: object) -> None:
            pass

        def poll_call_history(self, **_: object) -> list[dict[str, str]]:
            return []

    class Summary:
        failed = 0
        skipped_no_recording = 0

        def to_json_dict(self) -> dict[str, int]:
            return {"downloaded": 1, "failed": 0, "skipped_no_recording": 0}

    captured: list[TelephonyCallEvent] = []

    def fake_stage(*, events: list[TelephonyCallEvent], **_: object) -> Summary:
        captured.extend(events)
        return Summary()

    monkeypatch.setattr("mango_mvp.customer_timeline.calls_two_processes.MangoOfficeClient", FakeClient)
    monkeypatch.setattr("mango_mvp.customer_timeline.calls_two_processes.MangoRecordingDownloader", FakeClient)
    monkeypatch.setattr("mango_mvp.customer_timeline.calls_two_processes.stage_capture_events", fake_stage)
    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.read_known_processed_ids",
        lambda _root: ({"recording-1"}, {"failed-1"}),
    )
    monkeypatch.setenv("MANGO_OFFICE_API_KEY", "present")
    monkeypatch.setenv("MANGO_OFFICE_API_SALT", "present")

    capture_mango_window(
        config,
        datetime(2026, 7, 9, tzinfo=timezone.utc),
        datetime(2026, 7, 9, 1, tzinfo=timezone.utc),
    )

    assert [event.provider_call_id for event in captured] == ["failed-1"]


def test_process_b_registers_call_audio_artifact(tmp_path: Path) -> None:
    """The recording path is the only pointer a manager has back to the call;
    it must land in `event_artifacts`, not only inside record_json."""
    config = config_for(tmp_path)
    create_ready_call_db(config.ready_db)
    with CustomerTimelineSQLiteStore(config.timeline_db, allowed_root=config.timeline_allowed_root):
        pass

    report = run_process_b(config)
    assert report["status"] == "ok"

    with sqlite3.connect(f"file:{config.timeline_db}?mode=ro", uri=True) as con:
        rows = con.execute(
            "SELECT artifact_type, path FROM event_artifacts WHERE source_system='mango_processed_summary'"
        ).fetchall()
    assert [row[0] for row in rows] == ["call_audio"]
    assert rows[0][1] == "/ignored/masked.mp3"


def test_call_audio_artifact_path_never_reaches_a_projection(tmp_path: Path) -> None:
    """Capture filenames embed the client phone, so the artifact path is PDn:
    the read projection must expose only `has_path`, never the path itself."""
    from mango_mvp.customer_timeline.read_api import project_artifact

    config = config_for(tmp_path)
    create_ready_call_db(config.ready_db)
    with CustomerTimelineSQLiteStore(config.timeline_db, allowed_root=config.timeline_allowed_root):
        pass
    assert run_process_b(config)["status"] == "ok"

    with sqlite3.connect(f"file:{config.timeline_db}?mode=ro", uri=True) as con:
        con.row_factory = sqlite3.Row
        stored = [dict(row) for row in con.execute("SELECT * FROM event_artifacts")]
    assert len(stored) == 1
    assert stored[0]["path"]

    projected = project_artifact(stored[0])
    assert "path" not in projected
    assert projected["has_path"] is True
    assert stored[0]["path"] not in json.dumps(projected, ensure_ascii=False)


def test_mlx_cache_is_released_once_after_success_and_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mango_mvp.services import transcribe

    releases: list[str] = []
    monkeypatch.setattr(
        transcribe,
        "release_mlx_free_cache",
        lambda: releases.append("released") or True,
    )

    assert transcribe.run_with_mlx_cache_release(lambda: "ok", mlx_executed=True) == "ok"
    assert releases == ["released"]

    def fail() -> str:
        raise RuntimeError("synthetic failure")

    with pytest.raises(RuntimeError, match="synthetic failure"):
        transcribe.run_with_mlx_cache_release(fail, mlx_executed=True)
    assert releases == ["released", "released"]


def test_transcribe_does_not_apply_a_global_mlx_cache_limit() -> None:
    import inspect

    from mango_mvp.services import transcribe

    assert "set_cache_limit" not in inspect.getsource(transcribe)
