from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Optional

from mango_mvp.channels.feedback import RISK_FEEDBACK_EVENTS, feedback_loop_safety_contract
from mango_mvp.channels.signals import signal_engine_safety_contract
from mango_mvp.channels.storage import (
    ACTION_STATUS_APPROVED,
    ACTION_STATUS_PROPOSED,
    DRAFT_STATUS_NEEDS_REVIEW,
    channel_store_safety_contract,
)


CHANNEL_WORKSPACE_SCHEMA_VERSION = "channel_workspace_v1"


@dataclass(frozen=True)
class ChannelWorkspaceInboxItem:
    session_key: str
    channel: str
    channel_thread_id: str
    latest_message_at: Optional[str]
    latest_text_excerpt: str
    messages: int
    drafts_needing_review: int
    actions_needing_review: int
    high_signals: int
    risk_feedback_events: int
    hot_lead: bool = False
    requires_manager_review: bool = False
    status: str = "quiet"
    reasons: tuple[str, ...] = field(default_factory=tuple)

    def to_json_dict(self) -> Mapping[str, Any]:
        payload = asdict(self)
        payload["reasons"] = list(self.reasons)
        return payload


@dataclass(frozen=True)
class ChannelWorkspaceSummary:
    created_at: datetime
    inbox: tuple[ChannelWorkspaceInboxItem, ...]
    metrics: Mapping[str, Any]
    safety: Mapping[str, bool]
    source_summary: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        object.__setattr__(self, "inbox", tuple(self.inbox))
        object.__setattr__(self, "metrics", dict(self.metrics))
        object.__setattr__(self, "safety", dict(self.safety))
        object.__setattr__(self, "source_summary", dict(self.source_summary))

    def to_json_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": CHANNEL_WORKSPACE_SCHEMA_VERSION,
            "created_at": self.created_at.isoformat(),
            "metrics": dict(self.metrics),
            "inbox": [item.to_json_dict() for item in self.inbox],
            "safety": dict(self.safety),
            "source_summary": dict(self.source_summary),
        }


def build_channel_workspace_summary(
    store: Any,
    *,
    session_key: Optional[str] = None,
    limit: int = 50,
    created_at: Optional[datetime] = None,
) -> ChannelWorkspaceSummary:
    if limit <= 0 or limit > 500:
        raise ValueError("limit must be between 1 and 500")
    snapshot = store.snapshot(include_raw_payload=False)
    session_filter = str(session_key).strip() if session_key else None

    sessions = list(snapshot.get("sessions") or ())
    messages = list(snapshot.get("messages") or ())
    drafts = list(snapshot.get("drafts") or ())
    actions = list(snapshot.get("actions") or ())
    decisions = list(snapshot.get("signal_decisions") or ())
    feedback_events = list(((snapshot.get("feedback") or {}).get("events")) or ())
    if session_filter:
        sessions = [item for item in sessions if item.get("session_key") == session_filter]
        messages = [item for item in messages if item.get("session_key") == session_filter]
        drafts = [item for item in drafts if item.get("session_key") == session_filter]
        actions = [item for item in actions if item.get("session_key") == session_filter]
        decisions = [item for item in decisions if item.get("session_key") == session_filter]
        feedback_events = [item for item in feedback_events if item.get("session_key") == session_filter]

    inbox = build_inbox_items(
        sessions=sessions,
        messages=messages,
        drafts=drafts,
        actions=actions,
        decisions=decisions,
        feedback_events=feedback_events,
        limit=limit,
    )
    metrics = build_workspace_metrics(inbox, decisions=decisions, feedback_events=feedback_events)
    return ChannelWorkspaceSummary(
        created_at=created_at or datetime.now(timezone.utc),
        inbox=inbox,
        metrics=metrics,
        safety=channel_workspace_safety_contract(),
        source_summary=snapshot.get("summary") or {},
    )


def build_inbox_items(
    *,
    sessions: list[Mapping[str, Any]],
    messages: list[Mapping[str, Any]],
    drafts: list[Mapping[str, Any]],
    actions: list[Mapping[str, Any]],
    decisions: list[Mapping[str, Any]],
    feedback_events: list[Mapping[str, Any]],
    limit: int,
) -> tuple[ChannelWorkspaceInboxItem, ...]:
    by_session: dict[str, Mapping[str, Any]] = {}
    for session in sessions:
        key = str(session.get("session_key") or "")
        if key:
            by_session[key] = session
    for message in messages:
        key = str(message.get("session_key") or "")
        if key and key not in by_session:
            msg_payload = message.get("message") or {}
            by_session[key] = {
                "session_key": key,
                "channel": msg_payload.get("channel"),
                "channel_thread_id": msg_payload.get("channel_thread_id"),
            }

    items: list[ChannelWorkspaceInboxItem] = []
    for key, session in by_session.items():
        session_messages = [item for item in messages if item.get("session_key") == key]
        latest_message = latest_message_for_session(session_messages)
        latest_payload = latest_message.get("message") if latest_message else {}
        session_drafts = [item for item in drafts if item.get("session_key") == key]
        session_actions = [item for item in actions if item.get("session_key") == key]
        session_decisions = [item for item in decisions if item.get("session_key") == key]
        session_feedback = [item for item in feedback_events if item.get("session_key") == key]

        drafts_needing_review = sum(1 for item in session_drafts if item.get("status") == DRAFT_STATUS_NEEDS_REVIEW)
        actions_needing_review = sum(
            1 for item in session_actions if item.get("status") in {ACTION_STATUS_PROPOSED, ACTION_STATUS_APPROVED}
        )
        high_signals = count_high_signals(session_decisions)
        risk_feedback = sum(1 for item in session_feedback if item.get("event_type") in RISK_FEEDBACK_EVENTS)
        hot_lead = any(decision_has_signal(item, "hot_lead") for item in session_decisions)
        requires_manager_review = any(decision_requires_manager_review(item) for item in session_decisions)

        reasons = build_item_reasons(
            drafts_needing_review=drafts_needing_review,
            actions_needing_review=actions_needing_review,
            high_signals=high_signals,
            risk_feedback=risk_feedback,
            hot_lead=hot_lead,
            requires_manager_review=requires_manager_review,
        )
        items.append(
            ChannelWorkspaceInboxItem(
                session_key=key,
                channel=str(session.get("channel") or latest_payload.get("channel") or ""),
                channel_thread_id=str(session.get("channel_thread_id") or latest_payload.get("channel_thread_id") or ""),
                latest_message_at=str(latest_payload.get("received_at") or "") or None,
                latest_text_excerpt=trim_text(latest_payload.get("text") or ""),
                messages=len(session_messages),
                drafts_needing_review=drafts_needing_review,
                actions_needing_review=actions_needing_review,
                high_signals=high_signals,
                risk_feedback_events=risk_feedback,
                hot_lead=hot_lead,
                requires_manager_review=requires_manager_review,
                status=resolve_item_status(
                    drafts_needing_review=drafts_needing_review,
                    actions_needing_review=actions_needing_review,
                    high_signals=high_signals,
                    risk_feedback=risk_feedback,
                    hot_lead=hot_lead,
                ),
                reasons=reasons,
            )
        )
    items.sort(key=lambda item: (item.status != "needs_review", item.latest_message_at or "", item.session_key))
    return tuple(items[:limit])


def build_workspace_metrics(
    inbox: tuple[ChannelWorkspaceInboxItem, ...],
    *,
    decisions: list[Mapping[str, Any]],
    feedback_events: list[Mapping[str, Any]],
) -> Mapping[str, Any]:
    return {
        "sessions": len(inbox),
        "needs_review_sessions": sum(1 for item in inbox if item.status == "needs_review"),
        "attention_sessions": sum(1 for item in inbox if item.status == "attention"),
        "quiet_sessions": sum(1 for item in inbox if item.status == "quiet"),
        "messages": sum(item.messages for item in inbox),
        "drafts_needing_review": sum(item.drafts_needing_review for item in inbox),
        "actions_needing_review": sum(item.actions_needing_review for item in inbox),
        "high_signals": sum(item.high_signals for item in inbox),
        "hot_leads": sum(1 for item in inbox if item.hot_lead),
        "risk_feedback_events": sum(item.risk_feedback_events for item in inbox),
        "signal_decisions": len(decisions),
        "feedback_events": len(feedback_events),
        "read_only": True,
    }


def latest_message_for_session(messages: list[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not messages:
        return {}
    return sorted(
        messages,
        key=lambda item: str((item.get("message") or {}).get("received_at") or item.get("inserted_at") or ""),
    )[-1]


def count_high_signals(decisions: list[Mapping[str, Any]]) -> int:
    count = 0
    for decision in decisions:
        for signal in decision.get("signals") or ():
            if signal.get("severity") == "high":
                count += 1
    return count


def decision_has_signal(decision: Mapping[str, Any], signal_type: str) -> bool:
    return any(signal.get("signal_type") == signal_type for signal in decision.get("signals") or ())


def decision_requires_manager_review(decision: Mapping[str, Any]) -> bool:
    flags = decision.get("policy_flags") or {}
    if flags.get("requires_manager_review"):
        return True
    answer = decision.get("safe_answer") or {}
    return bool(answer.get("requires_approval", True))


def resolve_item_status(
    *,
    drafts_needing_review: int,
    actions_needing_review: int,
    high_signals: int,
    risk_feedback: int,
    hot_lead: bool,
) -> str:
    if drafts_needing_review or actions_needing_review:
        return "needs_review"
    if high_signals or risk_feedback or hot_lead:
        return "attention"
    return "quiet"


def build_item_reasons(
    *,
    drafts_needing_review: int,
    actions_needing_review: int,
    high_signals: int,
    risk_feedback: int,
    hot_lead: bool,
    requires_manager_review: bool,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if drafts_needing_review:
        reasons.append("drafts_need_manager_review")
    if actions_needing_review:
        reasons.append("actions_need_manager_review")
    if high_signals:
        reasons.append("high_signal_detected")
    if hot_lead:
        reasons.append("hot_lead")
    if risk_feedback:
        reasons.append("risk_feedback")
    if requires_manager_review:
        reasons.append("manager_review_required")
    return tuple(dict.fromkeys(reasons))


def trim_text(value: Any, *, limit: int = 140) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "..."


def channel_workspace_safety_contract() -> Mapping[str, bool]:
    return {
        **channel_store_safety_contract(),
        **signal_engine_safety_contract(),
        **feedback_loop_safety_contract(),
        "read_only_workspace": True,
        "network_calls": False,
        "llm_calls": False,
        "rag_used": False,
        "live_send": False,
        "write_crm": False,
        "write_tallanto": False,
        "write_runtime_db": False,
        "run_asr": False,
        "run_ra": False,
    }


__all__ = [
    "CHANNEL_WORKSPACE_SCHEMA_VERSION",
    "ChannelWorkspaceInboxItem",
    "ChannelWorkspaceSummary",
    "build_channel_workspace_summary",
    "channel_workspace_safety_contract",
]
