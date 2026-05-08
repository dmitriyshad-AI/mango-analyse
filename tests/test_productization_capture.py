from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from mango_mvp.productization.capture import CaptureAction, CapturePlanner, InMemorySeenCallStore
from mango_mvp.productization.contracts import Direction, RecordingAsset, TelephonyCallEvent, TenantRef


def _event(call_id: str, recording_url: Optional[str] = "https://records.example/call.mp3") -> TelephonyCallEvent:
    return TelephonyCallEvent(
        tenant=TenantRef("foton"),
        provider="mango",
        provider_call_id=call_id,
        started_at=datetime(2026, 5, 7, 9, 0, tzinfo=timezone.utc),
        ended_at=datetime(2026, 5, 7, 9, 5, tzinfo=timezone.utc),
        direction=Direction.INBOUND,
        client_phone="+79990000000",
        manager_ref="101",
        recording_url=recording_url,
        raw_payload={"call_id": call_id},
    )


def test_capture_planner_creates_shadow_ingest_candidate() -> None:
    planner = CapturePlanner(seen_store=InMemorySeenCallStore())
    event = _event("CALL-1")

    decision = planner.plan_event(event)

    assert decision.action == CaptureAction.ENQUEUE_SHADOW_CAPTURE
    assert decision.candidate is not None
    assert decision.candidate.event_key == "foton:mango:CALL-1"
    assert decision.candidate.audio_ref == "https://records.example/call.mp3"
    assert decision.candidate.client_phone == "+79990000000"


def test_capture_planner_prefers_resolved_recording_asset() -> None:
    planner = CapturePlanner(seen_store=InMemorySeenCallStore())
    event = _event("CALL-2")
    recording = RecordingAsset(event_key=event.event_key, uri="s3://tenant/call-2.mp3")

    decision = planner.plan_event(event, recording=recording)

    assert decision.action == CaptureAction.ENQUEUE_SHADOW_CAPTURE
    assert decision.candidate is not None
    assert decision.candidate.audio_ref == "s3://tenant/call-2.mp3"


def test_capture_planner_skips_duplicate_events() -> None:
    planner = CapturePlanner(seen_store=InMemorySeenCallStore())
    event = _event("CALL-3")

    first = planner.plan_event(event)
    second = planner.plan_event(event)

    assert first.action == CaptureAction.ENQUEUE_SHADOW_CAPTURE
    assert second.action == CaptureAction.SKIP_DUPLICATE
    assert second.candidate is None


def test_capture_planner_skips_events_without_recording_when_required() -> None:
    planner = CapturePlanner(seen_store=InMemorySeenCallStore(), require_recording=True)
    event = _event("CALL-4", recording_url=None)

    decision = planner.plan_event(event)

    assert decision.action == CaptureAction.SKIP_NO_RECORDING
    assert decision.reason == "recording_reference_missing"
    assert decision.candidate is None


def test_capture_planner_can_allow_metadata_only_shadow_capture() -> None:
    planner = CapturePlanner(seen_store=InMemorySeenCallStore(), require_recording=False)
    event = _event("CALL-5", recording_url=None)

    decision = planner.plan_event(event)

    assert decision.action == CaptureAction.ENQUEUE_SHADOW_CAPTURE
    assert decision.candidate is not None
    assert decision.candidate.audio_ref is None
