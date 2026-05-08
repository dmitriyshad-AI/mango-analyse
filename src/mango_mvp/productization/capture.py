from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, Mapping, Optional, Protocol, Sequence

from mango_mvp.productization.contracts import (
    CaptureIngestCandidate,
    RecordingAsset,
    TelephonyCallEvent,
)


class CaptureAction(str, Enum):
    ENQUEUE_SHADOW_CAPTURE = "enqueue_shadow_capture"
    SKIP_DUPLICATE = "skip_duplicate"
    SKIP_NO_RECORDING = "skip_no_recording"


class SeenCallStore(Protocol):
    def contains(self, event_key: str) -> bool:
        """Return True when the normalized provider call event is already known."""

    def remember(self, event_key: str) -> None:
        """Mark a normalized provider call event as planned or captured."""


class InMemorySeenCallStore:
    def __init__(self, initial_keys: Optional[Iterable[str]] = None) -> None:
        self._keys = set(initial_keys or [])

    def contains(self, event_key: str) -> bool:
        return event_key in self._keys

    def remember(self, event_key: str) -> None:
        self._keys.add(event_key)


@dataclass(frozen=True)
class CaptureDecision:
    action: CaptureAction
    event: TelephonyCallEvent
    reason: str
    candidate: Optional[CaptureIngestCandidate] = None


class CapturePlanner:
    def __init__(self, seen_store: SeenCallStore, require_recording: bool = True) -> None:
        self.seen_store = seen_store
        self.require_recording = require_recording

    def plan_batch(
        self,
        events: Iterable[TelephonyCallEvent],
        recordings_by_event_key: Optional[Mapping[str, RecordingAsset]] = None,
    ) -> Sequence[CaptureDecision]:
        recordings = recordings_by_event_key or {}
        decisions = []
        for event in events:
            decisions.append(self.plan_event(event, recordings.get(event.event_key)))
        return tuple(decisions)

    def plan_event(
        self,
        event: TelephonyCallEvent,
        recording: Optional[RecordingAsset] = None,
    ) -> CaptureDecision:
        if self.seen_store.contains(event.event_key):
            return CaptureDecision(
                action=CaptureAction.SKIP_DUPLICATE,
                event=event,
                reason="event_key_already_seen",
            )

        audio_ref = _resolve_audio_ref(event=event, recording=recording)
        if self.require_recording and not audio_ref:
            return CaptureDecision(
                action=CaptureAction.SKIP_NO_RECORDING,
                event=event,
                reason="recording_reference_missing",
            )

        self.seen_store.remember(event.event_key)
        return CaptureDecision(
            action=CaptureAction.ENQUEUE_SHADOW_CAPTURE,
            event=event,
            reason="ready_for_shadow_capture",
            candidate=CaptureIngestCandidate.from_event(event=event, audio_ref=audio_ref),
        )


def _resolve_audio_ref(
    event: TelephonyCallEvent,
    recording: Optional[RecordingAsset],
) -> Optional[str]:
    if recording is not None:
        return recording.uri
    if event.recording_url:
        return event.recording_url
    if event.recording_ref:
        return event.recording_ref
    return None
