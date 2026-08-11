from __future__ import annotations

import json
import os
import subprocess
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from tests.conftest import dual_strict_source, dualize_strict_enumeration

from mango_mvp.productization.mango_calls_service_contract import (
    CALLS_PROCESS_MATCHER_VERSION,
    CUTOVER_MANIFEST_SCHEMA,
    PREVIOUS_HOST_SHUTDOWN_SNAPSHOT_SCHEMA,
    READY_MANIFEST_SCHEMA,
    REQUIRED_CALLS_LAUNCHD_LABELS,
    REQUIRED_CALLS_LOCK_NAMES,
    approved_runtime_fingerprint,
    build_stage10_verdict,
    current_git_sha,
    git_worktree_is_clean,
    moscow_day_bounds_utc,
    safe_alert_payload,
    sha256_file,
    stage_capacity_report,
    validate_ready_manifest_payload,
    verify_cutover_authority,
)


DAY = date(2026, 8, 10)
NOW = datetime(2026, 8, 11, 9, tzinfo=timezone.utc)


def _write_previous_host_snapshot(
    path: Path,
    *,
    captured_at: datetime,
    disabled_at: datetime,
    cursor_sha: str = "b" * 64,
    **overrides: object,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": PREVIOUS_HOST_SHUTDOWN_SNAPSHOT_SCHEMA,
        "previous_host_id": "source-mac",
        "source_cursor_sha256": cursor_sha,
        "previous_host_disabled_at": disabled_at.isoformat(),
        "captured_at_utc": captured_at.isoformat(),
        "probe_ok": True,
        "launchd_scan_complete": True,
        "checked_launchd_labels": list(REQUIRED_CALLS_LAUNCHD_LABELS),
        "active_calls_labels": [],
        "plist_scan_complete": True,
        "active_calls_plists": [],
        "process_scan_complete": True,
        "process_matcher_version": CALLS_PROCESS_MATCHER_VERSION,
        "active_calls_pids": [],
        "active_calls_commands": [],
        "cron_scan_complete": True,
        "active_calls_cron_entries": [],
        "lock_scan_complete": True,
        "checked_lock_names": list(REQUIRED_CALLS_LOCK_NAMES),
        "held_lock_names": [],
        **overrides,
    }
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    path.chmod(0o600)
    return payload


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
                "mode": "mono_or_fallback",
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
    unique_keys = sorted(set(keys))
    return dualize_strict_enumeration({
        "mango_enumeration_complete": complete,
        "call_keys": unique_keys,
        "calls_by_moscow_day": {DAY.isoformat(): sorted(keys)} if keys else {},
        "independent_zero_enumerations_by_day": {
            DAY.isoformat(): 0 if keys else zero_proofs
        },
        "api_requests": 1,
        "api_rows_total": len(keys),
        "api_authoritative_rows_total": len(keys),
        "api_events_total": len(unique_keys),
        "mango_enumeration_source": {
            "mode": "strict_service",
            "since": "2026-08-09T21:00:00+00:00",
            "rolling_since": "2026-08-09T21:00:00+00:00",
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
                    "rows": len(keys),
                    "scope": "rolling_authority",
                }
            ],
        },
    })


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


def test_state_outside_enumeration_does_not_corrupt_pending_counters() -> None:
    result = dict(
        build_stage10_verdict(
            day=DAY,
            enumeration=_enumeration("enumerated-ready"),
            capture_entries=[
                _capture("enumerated-ready"),
                _capture("foreign-pending", "skipped_no_recording"),
            ],
            ready_rows=[_ready("enumerated-ready")],
            now=NOW,
        )
    )

    assert result["state_not_in_mango_enumeration"] == 1
    assert result["pending_unique"] == 0
    assert result["pending_awaiting_recording"] == 0
    assert result["pending_over_sla"] == 0
    assert result["oldest_pending_age_minutes"] == 0
    assert result["consistency_ok"] is False
    assert validate_ready_manifest_payload(
        _strict_ready_manifest_for(result), require_consistency=False
    ) == []


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
    assert result["quarantine_items"] == [
        {
            "call_key": "missing-audio",
            "started_at": "2026-08-10T12:00:00+00:00",
            "code": "recording_retry_expired",
            "reason": "Аудиозапись не появилась в Mango в течение 72 часов.",
            "action": (
                "Проверить запись в Mango и повторить загрузку вручную, "
                "если файл появился."
            ),
        }
    ]


def test_audio_integrity_incident_is_reasoned_quarantine_not_pending() -> None:
    result = build_stage10_verdict(
        day=DAY,
        enumeration=_enumeration("damaged-audio", "normal"),
        capture_entries=[
            _capture(
                "damaged-audio",
                "audio_integrity_quarantined",
                error="capture_target_integrity_mismatch",
                remediation_code="manual_restore_or_quarantine_corrupted_audio",
                recovery_state="immutable_audio_violation",
            ),
            _capture("normal"),
        ],
        ready_rows=[_ready("normal")],
        now=NOW,
    )

    assert result["quarantine_unique"] == 1
    assert result["quarantine_without_reason"] == 0
    assert result["pending_unique"] == 0
    assert result["ready_unique"] == 1
    assert result["consistency_ok"] is True
    assert result["closure_ok"] is True
    serialized = json.dumps(result["quarantine_items"], ensure_ascii=False)
    assert "capture_target_integrity_mismatch" not in serialized
    assert "manual_restore_or_quarantine_corrupted_audio" not in serialized


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
    ("code", "capture_extra", "dead_letter_stage"),
    [
        (
            "multiple_recordings_needs_review",
            {"remediation_code": "manual_recording_selection"},
            "",
        ),
        (
            "recording_retry_expired",
            {
                "error": "SECRET +79990001122 /Users/private/file.mp3",
                "remediation_code": "manual_review_or_retry_if_recording_appears",
            },
            "",
        ),
        (
            "audio_integrity_quarantined",
            {
                "error": "capture_target_integrity_mismatch",
                "remediation_code": "manual_restore_or_quarantine_corrupted_audio",
                "recovery_state": "immutable_audio_violation",
            },
            "",
        ),
        ("dead_letter_transcribe", {}, "transcribe"),
        ("dead_letter_resolve", {}, "resolve"),
        ("dead_letter_analyze", {}, "analyze"),
    ],
)
def test_all_quarantine_codes_have_static_manager_safe_guidance(
    code: str,
    capture_extra: dict[str, object],
    dead_letter_stage: str,
) -> None:
    capture_status = code if not dead_letter_stage else "downloaded"
    ready_rows = (
        [
            _ready(
                "quarantine",
                dead_letter_stage=dead_letter_stage,
                last_error="SECRET +79990001122 /Users/private/file.mp3",
            )
        ]
        if dead_letter_stage
        else []
    )
    result = build_stage10_verdict(
        day=DAY,
        enumeration=_enumeration("quarantine"),
        capture_entries=[_capture("quarantine", capture_status, **capture_extra)],
        ready_rows=ready_rows,
        now=NOW + timedelta(days=3),
    )

    assert result["quarantine_items"][0]["code"] == code
    serialized = json.dumps(result["quarantine_items"], ensure_ascii=False)
    for forbidden in ("SECRET", "+79990001122", "/Users/", "last_error"):
        assert forbidden not in serialized


def test_duplicate_recording_has_safe_guidance() -> None:
    result = build_stage10_verdict(
        day=DAY,
        enumeration=_enumeration("duplicate", "canonical"),
        capture_entries=[
            _capture(
                "duplicate",
                "duplicate_recording",
                canonical_event_key="event:canonical",
            ),
            _capture("canonical"),
        ],
        ready_rows=[_ready("canonical")],
        now=NOW,
    )

    assert result["quarantine_items"][0]["code"] == "duplicate_recording"
    assert result["consistency_ok"] is True


def test_recovered_late_recording_leaves_no_current_quarantine_item() -> None:
    recovered = build_stage10_verdict(
        day=DAY,
        enumeration=_enumeration("late"),
        capture_entries=[
            _capture(
                "late",
                "recording_retry_expired",
                error="recording_missing_after_retry_ttl",
                remediation_code="manual_review_or_retry_if_recording_appears",
                created_at="2026-08-07T12:00:00+00:00",
            ),
            _capture("late", "downloaded", created_at="2026-08-11T08:00:00+00:00"),
        ],
        ready_rows=[_ready("late")],
        now=NOW,
    )

    assert recovered["quarantine_unique"] == 0
    assert recovered["quarantine_items"] == []


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


def test_dual_asr_stereo_requires_both_complete_physical_channels() -> None:
    row = _ready(
        "one-stereo-channel",
        transcript_variants_json=json.dumps(
            {
                "mode": "stereo",
                "primary_provider": "mlx",
                "secondary_provider": "gigaam",
                "manager": {
                    "variant_a": "Whisper manager",
                    "variant_b": "GigaAM manager",
                },
            },
            ensure_ascii=False,
        ),
    )
    result = build_stage10_verdict(
        day=DAY,
        enumeration=_enumeration("one-stereo-channel"),
        capture_entries=[_capture("one-stereo-channel")],
        ready_rows=[row],
        now=NOW,
    )

    assert result["ready_without_dual_asr_or_explicit_exception"] == 1
    assert result["closure_ok"] is False


def test_strict_dual_asr_rejects_full_block_without_mode() -> None:
    row = _ready(
        "legacy-full",
        transcript_variants_json=json.dumps(
            {
                "primary_provider": "mlx",
                "secondary_provider": "gigaam",
                "full": {
                    "variant_a": "Whisper full",
                    "variant_b": "GigaAM full",
                },
            },
            ensure_ascii=False,
        ),
    )
    result = build_stage10_verdict(
        day=DAY,
        enumeration=_enumeration("legacy-full"),
        capture_entries=[_capture("legacy-full")],
        ready_rows=[row],
        now=NOW,
    )

    assert result["ready_without_dual_asr_or_explicit_exception"] == 1
    assert result["closure_ok"] is False


@pytest.mark.parametrize("invalid_variant", [{"bad": 1}, ["bad"], True])
def test_strict_dual_asr_rejects_non_string_variants(
    invalid_variant: object,
) -> None:
    row = _ready(
        "invalid-variant",
        transcript_variants_json=json.dumps(
            {
                "mode": "mono_or_fallback",
                "primary_provider": "mlx",
                "secondary_provider": "gigaam",
                "full": {
                    "variant_a": invalid_variant,
                    "variant_b": "GigaAM full",
                },
            },
            ensure_ascii=False,
        ),
    )
    result = build_stage10_verdict(
        day=DAY,
        enumeration=_enumeration("invalid-variant"),
        capture_entries=[_capture("invalid-variant")],
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
        {
            "approved": True,
            "reason": {"bad": 1},
            "approved_by": "owner",
            "approved_at": "2026-08-11T08:00:00+00:00",
        },
        {
            "approved": True,
            "reason": "synthetic",
            "approved_by": ["bad"],
            "approved_at": "2026-08-11T08:00:00+00:00",
        },
        {
            "approved": True,
            "reason": "synthetic",
            "approved_by": "owner",
            "approved_at": {"bad": 1},
        },
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
                "rows": 1,
                "scope": "rolling_authority",
            }
        ]
    enumeration["mango_enumeration_source"] = dual_strict_source(
        source,
        call_keys=["partial-day"],
        calls_by_day={DAY.isoformat(): ["partial-day"]},
    )
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


def test_first_empty_day_proof_is_valid_red_manifest_evidence() -> None:
    verdict = dict(
        build_stage10_verdict(
            day=DAY,
            enumeration=_enumeration(zero_proofs=1),
            capture_entries=[],
            ready_rows=[],
            now=NOW,
        )
    )
    manifest = _strict_ready_manifest_for(verdict)

    assert verdict["mango_enumeration_complete"] is False
    assert verdict["consistency_ok"] is False
    assert verdict["closure_ok"] is False
    assert validate_ready_manifest_payload(
        manifest,
        require_consistency=False,
    ) == []


def test_strict_manifest_rejects_compatibility_daily_source() -> None:
    verdict = dict(
        build_stage10_verdict(
            day=DAY,
            enumeration=_enumeration("complete"),
            capture_entries=[_capture("complete")],
            ready_rows=[_ready("complete")],
            now=NOW,
        )
    )
    verdict["mango_enumeration_source"] = {
        "mode": "compatibility_not_for_service",
        "since": "not_proven",
        "until": "not_proven",
    }

    errors = validate_ready_manifest_payload(_strict_ready_manifest_for(verdict))

    assert "daily_verdict_enumeration_source_not_strict" in errors


def test_strict_manifest_rejects_compatibility_top_level_source() -> None:
    verdict = dict(
        build_stage10_verdict(
            day=DAY,
            enumeration=_enumeration("complete"),
            capture_entries=[_capture("complete")],
            ready_rows=[_ready("complete")],
            now=NOW,
        )
    )
    manifest = _strict_ready_manifest_for(verdict)
    manifest["mango_enumeration_source"] = {
        **manifest["mango_enumeration_source"],
        "mode": "compatibility_not_for_service",
    }

    errors = validate_ready_manifest_payload(manifest)

    assert "mango_enumeration_source_not_strict" in errors


def test_ready_manifest_rejects_tampered_quarantine_guidance() -> None:
    verdict = dict(
        build_stage10_verdict(
            day=DAY,
            enumeration=_enumeration("missing-audio"),
            capture_entries=[
                _capture(
                    "missing-audio",
                    "recording_retry_expired",
                    error="recording_missing_after_retry_ttl",
                    remediation_code=(
                        "manual_review_or_retry_if_recording_appears"
                    ),
                )
            ],
            ready_rows=[],
            now=NOW + timedelta(days=3),
        )
    )
    verdict["quarantine_items"] = [
        {
            **dict(verdict["quarantine_items"][0]),
            "reason": "raw internal error: /Users/private/call.mp3",
        }
    ]

    errors = validate_ready_manifest_payload(
        _strict_ready_manifest_for(verdict), require_closure=True
    )

    assert "daily_verdict_quarantine_items_invalid" in errors


@pytest.mark.parametrize(
    "mutation",
    ("missing", "duplicate", "extra_field", "unknown_code", "wrong_day"),
)
def test_ready_manifest_rejects_malformed_quarantine_items(mutation: str) -> None:
    verdict = dict(
        build_stage10_verdict(
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
    )
    items = json.loads(json.dumps(verdict["quarantine_items"]))
    if mutation == "missing":
        items = []
    elif mutation == "duplicate":
        items = [items[0], dict(items[0])]
    elif mutation == "extra_field":
        items[0]["error"] = "raw"
    elif mutation == "unknown_code":
        items[0]["code"] = "unknown"
    elif mutation == "wrong_day":
        items[0]["started_at"] = "2026-08-09T12:00:00+00:00"
    verdict["quarantine_items"] = items

    assert "daily_verdict_quarantine_items_invalid" in validate_ready_manifest_payload(
        _strict_ready_manifest_for(verdict), require_closure=True
    )


def test_required_day_can_be_green_while_another_day_keeps_generation_red() -> None:
    green = dict(
        build_stage10_verdict(
            day=DAY,
            enumeration=_enumeration("complete"),
            capture_entries=[_capture("complete")],
            ready_rows=[_ready("complete")],
            now=NOW,
        )
    )
    old_day = DAY - timedelta(days=1)
    red = json.loads(json.dumps(green))
    red.update(
        day=old_day.isoformat(),
        mango_unique=2,
        unexplained_missing=1,
        consistency_ok=False,
        closure_ok=False,
    )
    red_source = red["mango_enumeration_source"]
    red_source.update(
        since="2026-08-08T21:00:00+00:00",
        rolling_since="2026-08-08T21:00:00+00:00",
        until="2026-08-09T21:00:00+00:00",
    )
    red_source["covered_intervals"] = [
        {
            "since": "2026-08-08T21:00:00+00:00",
            "until": "2026-08-09T21:00:00+00:00",
            "result_complete": True,
            "scope": "rolling_authority",
        }
    ]
    red["mango_enumeration_source"] = dual_strict_source(
        red_source,
        call_keys=["complete", "missing"],
        calls_by_day={old_day.isoformat(): ["complete", "missing"]},
    )
    manifest = _strict_ready_manifest_for(green)
    manifest.update(consistency_ok=False, closure_ok=False)
    manifest["moscow_dates"] = [old_day.isoformat(), DAY.isoformat()]
    manifest["daily_verdicts"] = {
        old_day.isoformat(): red,
        DAY.isoformat(): green,
    }

    assert "consistency_not_proven" in validate_ready_manifest_payload(manifest)
    assert validate_ready_manifest_payload(manifest, required_day=DAY) == []
    assert "required_day_consistency_not_proven" in validate_ready_manifest_payload(
        manifest,
        required_day=old_day,
    )


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
    snapshot = tmp_path / "previous_host_shutdown_snapshot.json"
    disabled_at = NOW - timedelta(minutes=20)
    checked_at = NOW - timedelta(minutes=10)
    _write_previous_host_snapshot(
        snapshot, captured_at=checked_at, disabled_at=disabled_at
    )
    payload = {
        "schema_version": CUTOVER_MANIFEST_SCHEMA,
        "active_host_id": "m1-host",
        "previous_host_id": "source-mac",
        "expected_code_sha": "a" * 40,
        "source_cursor_sha256": "b" * 64,
        "previous_host_snapshot_sha256": sha256_file(snapshot),
        "previous_host_disabled_at": disabled_at.isoformat(),
        "previous_host_checked_at": checked_at.isoformat(),
        "approved_at": (NOW - timedelta(minutes=5)).isoformat(),
        "approved_by": "owner",
    }
    cutover.write_text(json.dumps(payload), encoding="utf-8")
    cutover.chmod(0o600)
    monkeypatch.setattr(
        "mango_mvp.productization.mango_calls_service_contract.current_git_sha",
        lambda _: "a" * 40,
    )
    monkeypatch.setattr(
        "mango_mvp.productization.mango_calls_service_contract.git_worktree_is_clean",
        lambda _: True,
    )

    report = verify_cutover_authority(
        cutover_manifest_path=cutover,
        host_id_path=host,
        previous_host_snapshot_path=snapshot,
        expected_previous_host_id="source-mac",
        expected_code_sha="a" * 40,
        project_root=tmp_path,
        now=NOW,
    )
    assert report["ok"] is True

    monkeypatch.setattr(
        "mango_mvp.productization.mango_calls_service_contract.git_worktree_is_clean",
        lambda _: False,
    )
    dirty = verify_cutover_authority(
        cutover_manifest_path=cutover,
        host_id_path=host,
        previous_host_snapshot_path=snapshot,
        expected_previous_host_id="source-mac",
        expected_code_sha="a" * 40,
        project_root=tmp_path,
        now=NOW,
    )
    assert dirty["ok"] is False
    assert "cutover_worktree_dirty_or_unverifiable" in dirty["errors"]
    monkeypatch.setattr(
        "mango_mvp.productization.mango_calls_service_contract.git_worktree_is_clean",
        lambda _: True,
    )

    stale_checked_at = NOW - timedelta(hours=2)
    stale_disabled_at = stale_checked_at - timedelta(minutes=10)
    _write_previous_host_snapshot(
        snapshot, captured_at=stale_checked_at, disabled_at=stale_disabled_at
    )
    payload["previous_host_snapshot_sha256"] = sha256_file(snapshot)
    payload["previous_host_disabled_at"] = stale_disabled_at.isoformat()
    payload["previous_host_checked_at"] = stale_checked_at.isoformat()
    cutover.write_text(json.dumps(payload), encoding="utf-8")
    assert verify_cutover_authority(
        cutover_manifest_path=cutover,
        host_id_path=host,
        previous_host_snapshot_path=snapshot,
        expected_previous_host_id="source-mac",
        expected_code_sha="a" * 40,
        project_root=tmp_path,
        now=NOW,
        require_fresh_previous_host_proof=True,
    )["ok"] is False


@pytest.mark.parametrize(
    "snapshot_override,expected_error",
    [
        ({"previous_host_id": "another-mac"}, "previous_host_snapshot_host_id_mismatch"),
        ({"source_cursor_sha256": "f" * 64}, "previous_host_snapshot_cursor_sha256_mismatch"),
        ({"checked_launchd_labels": []}, "previous_host_required_labels_unchecked"),
        ({"active_calls_pids": [4242]}, "previous_host_calls_process_active"),
        ({"active_calls_plists": ["com.mango.calls-pipeline.plist"]}, "previous_host_calls_process_active"),
        ({"active_calls_cron_entries": ["calls pipeline"]}, "previous_host_calls_process_active"),
        ({"held_lock_names": ["pipeline"]}, "previous_host_calls_process_active"),
    ],
)
def test_cutover_authority_rejects_unbound_or_active_previous_host_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    snapshot_override: dict[str, object],
    expected_error: str,
) -> None:
    host = tmp_path / "host_id"
    host.write_text("m1-host\n", encoding="utf-8")
    host.chmod(0o600)
    checked_at = NOW - timedelta(minutes=10)
    disabled_at = NOW - timedelta(minutes=20)
    snapshot = tmp_path / "snapshot.json"
    _write_previous_host_snapshot(
        snapshot,
        captured_at=checked_at,
        disabled_at=disabled_at,
        **snapshot_override,
    )
    manifest = tmp_path / "cutover.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": CUTOVER_MANIFEST_SCHEMA,
                "active_host_id": "m1-host",
                "previous_host_id": "source-mac",
                "expected_code_sha": "a" * 40,
                "source_cursor_sha256": "b" * 64,
                "previous_host_snapshot_sha256": sha256_file(snapshot),
                "previous_host_disabled_at": disabled_at.isoformat(),
                "previous_host_checked_at": checked_at.isoformat(),
                "approved_at": (NOW - timedelta(minutes=5)).isoformat(),
                "approved_by": "owner",
            }
        ),
        encoding="utf-8",
    )
    manifest.chmod(0o600)
    monkeypatch.setattr(
        "mango_mvp.productization.mango_calls_service_contract.current_git_sha",
        lambda _: "a" * 40,
    )
    monkeypatch.setattr(
        "mango_mvp.productization.mango_calls_service_contract.git_worktree_is_clean",
        lambda _: True,
    )

    report = verify_cutover_authority(
        cutover_manifest_path=manifest,
        host_id_path=host,
        previous_host_snapshot_path=snapshot,
        expected_previous_host_id="source-mac",
        expected_code_sha="a" * 40,
        project_root=tmp_path,
        now=NOW,
    )

    assert report["ok"] is False
    assert expected_error in report["errors"]


def test_cutover_authority_rejects_missing_empty_tampered_or_readable_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    host = tmp_path / "host_id"
    host.write_text("m1-host\n", encoding="utf-8")
    host.chmod(0o600)
    checked_at = NOW - timedelta(minutes=10)
    disabled_at = NOW - timedelta(minutes=20)
    snapshot = tmp_path / "snapshot.json"
    _write_previous_host_snapshot(
        snapshot, captured_at=checked_at, disabled_at=disabled_at
    )
    manifest = tmp_path / "cutover.json"
    payload = {
        "schema_version": CUTOVER_MANIFEST_SCHEMA,
        "active_host_id": "m1-host",
        "previous_host_id": "source-mac",
        "expected_code_sha": "a" * 40,
        "source_cursor_sha256": "b" * 64,
        "previous_host_snapshot_sha256": sha256_file(snapshot),
        "previous_host_disabled_at": disabled_at.isoformat(),
        "previous_host_checked_at": checked_at.isoformat(),
        "approved_at": (NOW - timedelta(minutes=5)).isoformat(),
        "approved_by": "owner",
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    manifest.chmod(0o600)
    monkeypatch.setattr(
        "mango_mvp.productization.mango_calls_service_contract.current_git_sha",
        lambda _: "a" * 40,
    )
    monkeypatch.setattr(
        "mango_mvp.productization.mango_calls_service_contract.git_worktree_is_clean",
        lambda _: True,
    )

    def report(path: Path | None) -> dict[str, object]:
        return dict(
            verify_cutover_authority(
                cutover_manifest_path=manifest,
                host_id_path=host,
                previous_host_snapshot_path=path,
                expected_previous_host_id="source-mac",
                expected_code_sha="a" * 40,
                project_root=tmp_path,
                now=NOW,
            )
        )

    assert "previous_host_snapshot_missing_or_invalid" in report(None)["errors"]
    snapshot.write_text("{}", encoding="utf-8")
    payload["previous_host_snapshot_sha256"] = sha256_file(snapshot)
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    empty = report(snapshot)
    assert empty["ok"] is False
    assert "previous_host_snapshot_schema_mismatch" in empty["errors"]
    snapshot.write_text("{\"tampered\":true}", encoding="utf-8")
    assert "previous_host_snapshot_sha256_mismatch" in report(snapshot)["errors"]
    snapshot.chmod(0o644)
    assert "previous_host_snapshot_missing_or_invalid" in report(snapshot)["errors"]


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
            "quarantine_items": [
                {"call_key": "secret-call", "reason": "private"}
            ],
            "call_key": "secret-call",
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


def _commit_test_repo(path: Path, content: str) -> str:
    path.mkdir()
    subprocess.run(["/usr/bin/git", "init", "-q"], cwd=path, check=True)
    (path / "tracked.py").write_text(content, encoding="utf-8")
    subprocess.run(["/usr/bin/git", "add", "tracked.py"], cwd=path, check=True)
    subprocess.run(
        [
            "/usr/bin/git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=path,
        check=True,
    )
    return subprocess.check_output(
        ["/usr/bin/git", "rev-parse", "HEAD"], cwd=path, text=True
    ).strip()


def test_git_authority_ignores_inherited_repository_redirection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    actual = tmp_path / "actual"
    foreign = tmp_path / "foreign"
    actual_sha = _commit_test_repo(actual, "actual = True\n")
    _commit_test_repo(foreign, "foreign = True\n")
    monkeypatch.setenv("GIT_DIR", str(foreign / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(foreign))

    assert current_git_sha(actual) == actual_sha
    (actual / "tracked.py").write_text("actual = False\n", encoding="utf-8")
    assert git_worktree_is_clean(actual) is False


@pytest.mark.parametrize("flag", ["--assume-unchanged", "--skip-worktree"])
def test_git_authority_rejects_hidden_index_flags(
    tmp_path: Path, flag: str
) -> None:
    repo = tmp_path / "repo"
    _commit_test_repo(repo, "clean = True\n")
    subprocess.run(
        ["/usr/bin/git", "update-index", flag, "tracked.py"],
        cwd=repo,
        check=True,
    )
    (repo / "tracked.py").write_text("clean = False\n", encoding="utf-8")

    assert subprocess.check_output(
        ["/usr/bin/git", "status", "--porcelain"], cwd=repo, text=True
    ) == ""
    assert git_worktree_is_clean(repo) is False
