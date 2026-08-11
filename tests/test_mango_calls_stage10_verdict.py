from __future__ import annotations

import json
import os
import subprocess
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from mango_mvp.productization.mango_calls_service_contract import (
    CUTOVER_MANIFEST_SCHEMA,
    READY_MANIFEST_SCHEMA,
    approved_runtime_fingerprint,
    build_stage10_verdict,
    moscow_day_bounds_utc,
    safe_alert_payload,
    stage_capacity_report,
    validate_ready_manifest_payload,
    verify_cutover_authority,
)


DAY = date(2026, 8, 10)
NOW = datetime(2026, 8, 11, 9, tzinfo=timezone.utc)


def _capture(call_key: str, status: str = "downloaded", **extra: object) -> dict[str, object]:
    return {
        "provider_call_id": call_key,
        "event_key": f"event:{call_key}",
        "started_at": "2026-08-10T12:00:00+00:00",
        "created_at": "2026-08-10T12:01:00+00:00",
        "status": status,
        **extra,
    }


def _ready(call_key: str, **extra: object) -> dict[str, object]:
    return {
        "source_call_id": call_key,
        "started_at": "2026-08-10T12:00:00+00:00",
        "transcription_status": "done",
        "transcript_variants_json": json.dumps(
            {
                "primary_provider": "mlx",
                "secondary_provider": "gigaam",
                "full": {
                    "variant_a": "синтетический Whisper",
                    "variant_b": "синтетический GigaAM",
                },
            }
        ),
        "resolve_status": "done",
        "analysis_status": "done",
        "analysis_json": "{\"history_summary\":\"ok\"}",
        **extra,
    }


def _enumeration(*keys: str, complete: bool = True, zero_proofs: int = 0) -> dict[str, object]:
    return {
        "mango_enumeration_complete": complete,
        "call_keys": list(keys),
        "independent_zero_enumerations": zero_proofs,
        "mango_enumeration_source": {
            "mode": "strict_service",
            "since": "2026-08-09T21:00:00+00:00",
            "until": "2026-08-10T21:00:00+00:00",
            "cursor": "not_applicable_stats_request_result",
            "requests": 1,
            "pages": None,
            "pagination": "not_applicable_stats_request_result",
            "covered_intervals": [
                {
                    "since": "2026-08-09T21:00:00+00:00",
                    "until": "2026-08-10T21:00:00+00:00",
                    "result_complete": True,
                }
            ],
        },
    }


def test_pending_day_is_consistent_but_not_closed() -> None:
    result = build_stage10_verdict(
        day=DAY,
        enumeration=_enumeration("call-ready", "call-pending"),
        capture_entries=[_capture("call-ready"), _capture("call-pending", "skipped_no_recording")],
        ready_rows=[_ready("call-ready")],
        now=NOW,
    )

    assert result["mango_unique"] == 2
    assert result["ready_unique"] == 1
    assert result["pending_unique"] == 1
    assert result["unexplained_missing"] == 0
    assert result["consistency_ok"] is True
    assert result["closure_ok"] is False


@pytest.mark.parametrize(
    "open_lease",
    (
        {
            "pipeline_stage": "analyze",
            "pipeline_worker_id": "synthetic-worker",
            "pipeline_claimed_at": "2026-08-11T08:59:00+00:00",
        },
        {
            "analysis_worker_id": "synthetic-analyzer",
            "analysis_claimed_at": "2026-08-11T08:59:00+00:00",
        },
    ),
)
def test_open_worker_lease_stays_pending_and_cannot_close(
    open_lease: dict[str, str],
) -> None:
    result = build_stage10_verdict(
        day=DAY,
        enumeration=_enumeration("leased"),
        capture_entries=[_capture("leased")],
        ready_rows=[_ready("leased", **open_lease)],
        now=NOW,
    )

    assert result["ready_unique"] == 0
    assert result["ready_incomplete_unique"] == 1
    assert result["pending_unique"] == 1
    assert result["consistency_ok"] is True
    assert result["closure_ok"] is False


def test_expired_missing_recording_stays_in_reasoned_quarantine() -> None:
    result = build_stage10_verdict(
        day=DAY,
        enumeration=_enumeration("missing-audio"),
        capture_entries=[
            _capture(
                "missing-audio",
                "recording_retry_expired",
                error="recording_missing_after_retry_ttl",
                remediation_code="manual_review_or_retry_if_recording_appears",
            )
        ],
        ready_rows=[],
        now=NOW + timedelta(days=3),
    )

    assert result["quarantine_unique"] == 1
    assert result["quarantine_without_reason"] == 0
    assert result["pending_unique"] == 0
    assert result["consistency_ok"] is True
    assert result["closure_ok"] is True


def test_two_recordings_quarantine_only_that_call() -> None:
    result = build_stage10_verdict(
        day=DAY,
        enumeration=_enumeration("multi", "normal"),
        capture_entries=[
            _capture(
                "multi",
                "multiple_recordings_needs_review",
                remediation_code="manual_recording_selection",
            ),
            _capture("normal"),
        ],
        ready_rows=[_ready("normal")],
        now=NOW,
    )

    assert result["quarantine_unique"] == 1
    assert result["ready_unique"] == 1
    assert result["consistency_ok"] is True


@pytest.mark.parametrize(
    ("enumeration", "entries", "rows", "field"),
    [
        (_enumeration("lost", complete=False), [], [], "mango_enumeration_complete"),
        (_enumeration("lost"), [], [], "unexplained_missing"),
        (_enumeration("dup", "dup"), [_capture("dup")], [], "duplicate_call_keys"),
        (
            _enumeration("overlap"),
            [
                _capture(
                    "overlap",
                    "recording_retry_expired",
                    error="expired",
                    remediation_code="manual_review_or_retry_if_recording_appears",
                )
            ],
            [_ready("overlap")],
            "state_overlap_count",
        ),
    ],
)
def test_incomplete_enumeration_missing_duplicate_and_overlap_are_red(
    enumeration: dict[str, object],
    entries: list[dict[str, object]],
    rows: list[dict[str, object]],
    field: str,
) -> None:
    result = build_stage10_verdict(
        day=DAY,
        enumeration=enumeration,
        capture_entries=entries,
        ready_rows=rows,
        now=NOW,
    )

    assert not result["consistency_ok"]
    assert not result["closure_ok"]
    assert result[field] is False or int(result[field]) > 0


def test_empty_day_requires_two_independent_complete_zero_enumerations() -> None:
    first = build_stage10_verdict(
        day=DAY,
        enumeration=_enumeration(complete=True, zero_proofs=1),
        capture_entries=[],
        ready_rows=[],
        now=NOW,
    )
    second = build_stage10_verdict(
        day=DAY,
        enumeration=_enumeration(complete=True, zero_proofs=2),
        capture_entries=[],
        ready_rows=[],
        now=NOW,
    )

    assert first["consistency_ok"] is False
    assert second["consistency_ok"] is True
    assert second["closure_ok"] is True


def test_ready_without_dual_asr_resolve_or_analyze_cannot_close() -> None:
    row = _ready(
        "bad-ready",
        transcript_variants_json="{}",
        resolve_status="pending",
        analysis_status="pending",
        analysis_json="{}",
    )
    result = build_stage10_verdict(
        day=DAY,
        enumeration=_enumeration("bad-ready"),
        capture_entries=[_capture("bad-ready")],
        ready_rows=[row],
        now=NOW,
    )

    assert result["consistency_ok"] is True
    assert result["ready_without_dual_asr_or_explicit_exception"] == 1
    assert result["ready_without_resolve"] == 1
    assert result["ready_without_analyze"] == 1
    assert result["closure_ok"] is False


def test_dual_asr_must_be_paired_inside_the_same_transcript_block() -> None:
    row = _ready(
        "cross-block",
        transcript_variants_json=json.dumps(
            {
                "primary_provider": "mlx",
                "secondary_provider": "gigaam",
                "manager": {"variant_a": "только Whisper"},
                "client": {"variant_b": "только GigaAM"},
            },
            ensure_ascii=False,
        ),
    )
    result = build_stage10_verdict(
        day=DAY,
        enumeration=_enumeration("cross-block"),
        capture_entries=[_capture("cross-block")],
        ready_rows=[row],
        now=NOW,
    )

    assert result["ready_without_dual_asr_or_explicit_exception"] == 1
    assert result["closure_ok"] is False


@pytest.mark.parametrize(
    "exception",
    [
        True,
        {"approved": True, "reason": "", "approved_by": "owner"},
        {"approved": True, "reason": "synthetic", "approved_by": "owner"},
    ],
)
def test_dual_asr_exception_requires_audit_fields(exception: object) -> None:
    row = _ready(
        "bad-exception",
        transcript_variants_json=json.dumps({"dual_asr_exception": exception}),
    )
    result = build_stage10_verdict(
        day=DAY,
        enumeration=_enumeration("bad-exception"),
        capture_entries=[_capture("bad-exception")],
        ready_rows=[row],
        now=NOW,
    )
    assert result["ready_without_dual_asr_or_explicit_exception"] == 1
    assert result["closure_ok"] is False


def test_dual_asr_exception_with_complete_audit_record_is_accepted() -> None:
    row = _ready(
        "approved-exception",
        transcript_variants_json=json.dumps(
            {
                "dual_asr_exception": {
                    "approved": True,
                    "reason": "synthetic provider outage",
                    "approved_by": "owner",
                    "approved_at": "2026-08-11T08:00:00+00:00",
                }
            }
        ),
    )
    result = build_stage10_verdict(
        day=DAY,
        enumeration=_enumeration("approved-exception"),
        capture_entries=[_capture("approved-exception")],
        ready_rows=[row],
        now=NOW,
    )
    assert result["ready_without_dual_asr_or_explicit_exception"] == 0
    assert result["closure_ok"] is True


def test_dual_asr_exception_from_the_future_is_rejected() -> None:
    row = _ready(
        "future-exception",
        transcript_variants_json=json.dumps(
            {
                "dual_asr_exception": {
                    "approved": True,
                    "reason": "synthetic",
                    "approved_by": "owner",
                    "approved_at": "2026-08-11T10:00:00+00:00",
                }
            }
        ),
    )
    result = build_stage10_verdict(
        day=DAY,
        enumeration=_enumeration("future-exception"),
        capture_entries=[_capture("future-exception")],
        ready_rows=[row],
        now=NOW,
    )
    assert result["ready_without_dual_asr_or_explicit_exception"] == 1
    assert result["closure_ok"] is False


def test_partial_current_day_can_be_consistent_but_never_closed() -> None:
    enumeration = _enumeration("partial-day")
    source = dict(enumeration["mango_enumeration_source"])
    source["until"] = "2026-08-10T15:00:00+00:00"
    source["covered_intervals"] = [
        {
            "since": "2026-08-09T21:00:00+00:00",
            "until": "2026-08-10T15:00:00+00:00",
            "result_complete": True,
        }
    ]
    enumeration["mango_enumeration_source"] = source
    result = build_stage10_verdict(
        day=DAY,
        enumeration=enumeration,
        capture_entries=[_capture("partial-day")],
        ready_rows=[_ready("partial-day")],
        now=datetime(2026, 8, 10, 15, 1, tzinfo=timezone.utc),
    )
    assert result["consistency_ok"] is True
    assert result["closure_ok"] is False


@pytest.mark.parametrize(
    "entry",
    [
        _capture("quarantine", "multiple_recordings_needs_review"),
        _capture(
            "quarantine",
            "recording_retry_expired",
            error="recording_missing_after_retry_ttl",
        ),
    ],
)
def test_quarantine_requires_explicit_next_handling(entry: dict[str, object]) -> None:
    result = build_stage10_verdict(
        day=DAY,
        enumeration=_enumeration("quarantine"),
        capture_entries=[entry],
        ready_rows=[],
        now=NOW + timedelta(days=3),
    )
    assert result["quarantine_without_reason"] == 1
    assert result["consistency_ok"] is False


def test_moscow_midnight_uses_utc_storage_without_day_leak() -> None:
    start, end = moscow_day_bounds_utc(DAY)
    assert start.isoformat() == "2026-08-09T21:00:00+00:00"
    assert end.isoformat() == "2026-08-10T21:00:00+00:00"
    result = build_stage10_verdict(
        day=DAY,
        enumeration=_enumeration("before", "after"),
        capture_entries=[
            _capture("before", started_at="2026-08-10T20:59:59+00:00"),
            _capture("after", started_at="2026-08-10T21:00:00+00:00"),
        ],
        ready_rows=[],
        now=NOW,
    )
    assert result["pending_unique"] == 1
    assert result["unexplained_missing"] == 1


def _ready_manifest(**extra: object) -> dict[str, object]:
    return {
        "schema_version": READY_MANIFEST_SCHEMA,
        "status": "ready",
        "consistency_ok": True,
        "closure_ok": True,
        "producer_git_sha": "a" * 40,
        "host_id": "m1-host",
        "run_id": "run-1",
        "mango_window": {"since": "a", "until": "b"},
        "mango_enumeration_complete": True,
        "quick_check": "ok",
        "integrity_check": "ok",
        "runtime_fingerprint": approved_runtime_fingerprint(),
        **extra,
    }


def _strict_ready_manifest_for(verdict: dict[str, object]) -> dict[str, object]:
    source = verdict["mango_enumeration_source"]
    return {
        "schema_version": READY_MANIFEST_SCHEMA,
        "created_at_utc": "2026-08-11T09:00:00+00:00",
        "published_at": "2026-08-11T09:00:01+00:00",
        "status": "ready",
        "consistency_ok": verdict["consistency_ok"],
        "closure_ok": verdict["closure_ok"],
        "producer_git_sha": "a" * 40,
        "host_id": "m1-host",
        "run_id": "run-1",
        "mango_window": {
            "since": "2026-08-09T21:00:00+00:00",
            "until": "2026-08-10T21:00:00+00:00",
        },
        "mango_enumeration_complete": True,
        "mango_enumeration_source": source,
        "moscow_dates": [DAY.isoformat()],
        "daily_verdicts": {DAY.isoformat(): verdict},
        "manifest_snapshot": {"end_offset": 1, "sha256": "b" * 64},
        "provenance_mode": "strict_service",
        "quick_check": "ok",
        "integrity_check": "ok",
        "runtime_fingerprint": approved_runtime_fingerprint(),
    }


@pytest.mark.parametrize(
    ("changes", "expected_error"),
    [
        ({"pending_awaiting_recording": 1}, "daily_verdict_pending_or_zero_proof_mismatch"),
        ({"oldest_pending_age_minutes": 1}, "daily_verdict_pending_or_zero_proof_mismatch"),
        ({"mango_unique": 0, "ready_unique": 0}, "daily_verdict_pending_or_zero_proof_mismatch"),
    ],
)
def test_ready_manifest_recomputes_pending_and_zero_day_invariants(
    changes: dict[str, object], expected_error: str
) -> None:
    verdict = dict(
        build_stage10_verdict(
            day=DAY,
            enumeration=_enumeration("complete"),
            capture_entries=[_capture("complete")],
            ready_rows=[_ready("complete")],
            now=NOW,
        )
    )
    verdict.update(changes)
    errors = validate_ready_manifest_payload(
        _strict_ready_manifest_for(verdict), require_closure=True
    )
    assert expected_error in errors


def test_valid_strict_ready_manifest_is_accepted() -> None:
    verdict = dict(
        build_stage10_verdict(
            day=DAY,
            enumeration=_enumeration("complete"),
            capture_entries=[_capture("complete")],
            ready_rows=[_ready("complete")],
            now=NOW,
        )
    )
    assert validate_ready_manifest_payload(
        _strict_ready_manifest_for(verdict), require_closure=True
    ) == []


def test_ready_manifest_rejects_unknown_model_code_and_unclosed_day() -> None:
    manifest = _ready_manifest(closure_ok=False)
    manifest["runtime_fingerprint"]["resolve"]["model"] = "unknown"  # type: ignore[index]
    errors = validate_ready_manifest_payload(
        manifest,
        require_closure=True,
        expected_code_sha="b" * 40,
        expected_host_id="other-host",
    )

    assert "closure_not_proven" in errors
    assert "producer_git_sha_mismatch" in errors
    assert "host_id_mismatch" in errors
    assert "resolve_model_mismatch" in errors


def test_cutover_authority_requires_fresh_old_host_proof_and_exact_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    host = tmp_path / "host_id"
    host.write_text("m1-host\n", encoding="utf-8")
    host.chmod(0o600)
    cutover = tmp_path / "cutover_manifest.json"
    payload = {
        "schema_version": CUTOVER_MANIFEST_SCHEMA,
        "active_host_id": "m1-host",
        "expected_code_sha": "a" * 40,
        "source_cursor_sha256": "b" * 64,
        "previous_host_snapshot_sha256": "c" * 64,
        "previous_host_disabled_at": (NOW - timedelta(minutes=20)).isoformat(),
        "previous_host_checked_at": (NOW - timedelta(minutes=10)).isoformat(),
        "approved_at": (NOW - timedelta(minutes=5)).isoformat(),
        "approved_by": "owner",
    }
    cutover.write_text(json.dumps(payload), encoding="utf-8")
    cutover.chmod(0o600)
    monkeypatch.setattr(
        "mango_mvp.productization.mango_calls_service_contract.current_git_sha",
        lambda _: "a" * 40,
    )

    report = verify_cutover_authority(
        cutover_manifest_path=cutover,
        host_id_path=host,
        expected_code_sha="a" * 40,
        project_root=tmp_path,
        now=NOW,
    )
    assert report["ok"] is True

    payload["previous_host_checked_at"] = (NOW - timedelta(hours=2)).isoformat()
    cutover.write_text(json.dumps(payload), encoding="utf-8")
    assert verify_cutover_authority(
        cutover_manifest_path=cutover,
        host_id_path=host,
        expected_code_sha="a" * 40,
        project_root=tmp_path,
        now=NOW,
    )["ok"] is False


def test_controlled_ten_capacity_uses_all_four_stages_and_peak_snapshot() -> None:
    benchmark = {
        "audio_hours": 1,
        **{
            stage: {"wall_seconds": 300, "peak_memory_bytes": 8_000_000_000, "swap_bytes": 0}
            for stage in ("whisper", "gigaam", "resolve", "analyze")
        },
    }
    report = stage_capacity_report(
        benchmark=benchmark,
        peak_snapshot={"peak_audio_hours_per_hour": 1.2},
        physical_memory_bytes=32_000_000_000,
    )

    assert report["headroom_ratio"] == 2.5
    assert report["capacity_ok"] is True
    assert stage_capacity_report(
        benchmark={key: value for key, value in benchmark.items() if key != "analyze"},
        peak_snapshot={"peak_audio_hours_per_hour": 1.2},
        physical_memory_bytes=32_000_000_000,
    )["capacity_ok"] is False


def test_safe_alert_projection_drops_pii_paths_and_diagnostics() -> None:
    result = safe_alert_payload(
        {
            "status": "failed",
            "pending_unique": 2,
            "phone": "+79991234567",
            "db_path": "/Users/person/private.sqlite",
            "diagnostic": {"prompt": "secret"},
        }
    )
    assert result == {"status": "failed", "pending_unique": 2}


def test_cutover_files_must_be_owner_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    host = tmp_path / "host_id"
    host.write_text("m1-host", encoding="utf-8")
    host.chmod(0o644)
    cutover = tmp_path / "cutover.json"
    cutover.write_text("{}", encoding="utf-8")
    cutover.chmod(0o600)
    monkeypatch.setattr(
        "mango_mvp.productization.mango_calls_service_contract.current_git_sha",
        lambda _: "a" * 40,
    )
    with pytest.raises(RuntimeError, match="host_id_must_be_owner_only"):
        verify_cutover_authority(
            cutover_manifest_path=cutover,
            host_id_path=host,
            expected_code_sha="a" * 40,
            project_root=tmp_path,
            now=NOW,
        )
