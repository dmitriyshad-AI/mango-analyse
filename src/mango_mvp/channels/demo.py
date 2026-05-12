from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from mango_mvp.channels.feedback import build_action_feedback_event
from mango_mvp.channels.persistence import ChannelSQLiteStore, channel_sqlite_safety_contract
from mango_mvp.channels.preview_service import build_channel_draft_preview
from mango_mvp.channels.signals import build_channel_signal_decision
from mango_mvp.channels.storage import ChannelActionRecord
from mango_mvp.channels.contracts import ChannelMessage
from mango_mvp.channels.workspace import build_channel_workspace_summary, channel_workspace_safety_contract


CHANNEL_DEMO_SCHEMA_VERSION = "channel_demo_workspace_v1"
DEMO_START = datetime(2026, 5, 11, 9, 0, tzinfo=timezone.utc)


class DemoClock:
    def __init__(self, start: datetime = DEMO_START) -> None:
        self.value = start

    def __call__(self) -> datetime:
        current = self.value
        self.value = self.value + timedelta(seconds=1)
        return current


def build_demo_messages(*, start: datetime = DEMO_START) -> tuple[ChannelMessage, ...]:
    return (
        ChannelMessage(
            channel="telegram_business",
            channel_message_id="demo-tg-1",
            channel_thread_id="demo-thread-hot-lead",
            channel_user_id="tg-user-1",
            direction="inbound",
            text="Здравствуйте, хочу оплатить курс сегодня. Можно быстро связаться?",
            received_at=start,
            raw_payload={"telegram_update": "redacted"},
        ),
        ChannelMessage(
            channel="site_chat",
            channel_message_id="demo-site-1",
            channel_thread_id="demo-thread-price-question",
            channel_user_id="site-visitor-1",
            direction="inbound",
            text="Подскажите стоимость и есть ли рассрочка?",
            received_at=start + timedelta(minutes=2),
            raw_payload={"webhook_payload": "redacted"},
        ),
        ChannelMessage(
            channel="crm_chat",
            channel_message_id="demo-crm-1",
            channel_thread_id="demo-thread-handoff",
            channel_user_id="crm-contact-1",
            direction="inbound",
            text="Мне нужен менеджер, перезвоните завтра утром",
            received_at=start + timedelta(minutes=4),
            raw_payload={"crm_dialog_payload": "redacted"},
        ),
    )


def build_channel_demo_workspace(
    db_path: Path | str,
    *,
    created_at: Optional[datetime] = None,
) -> Mapping[str, Any]:
    clock = DemoClock(created_at or DEMO_START)
    store = ChannelSQLiteStore(db_path, clock=clock)
    for message in build_demo_messages(start=created_at or DEMO_START):
        context = demo_context_for_message(message)
        preview = build_channel_draft_preview(message, context=context)
        store.upsert_preview(preview, actor="channel_demo")
        decision = build_channel_signal_decision(message=message, preview=preview, context=context)
        store.record_signal_decision(decision, actor="channel_demo_signal_engine")
        maybe_record_demo_feedback(store, preview.draft_id, decision.decision_id)

    workspace = build_channel_workspace_summary(store, created_at=created_at or DEMO_START)
    result = {
        "schema_version": CHANNEL_DEMO_SCHEMA_VERSION,
        "db_path": str(Path(db_path)),
        "workspace": workspace.to_json_dict(),
        "safety": channel_demo_safety_contract(),
    }
    store.close()
    return result


def demo_context_for_message(message: ChannelMessage) -> Mapping[str, Any]:
    if "hot-lead" in message.channel_thread_id:
        return {"priority": "hot", "safe_draft_text": "Здравствуйте! Передадим менеджеру, он свяжется с вами."}
    if "price-question" in message.channel_thread_id:
        return {"requires_commercial_review": True}
    if "handoff" in message.channel_thread_id:
        return {"force_manager_handoff": True, "follow_up_required": True}
    return {}


def maybe_record_demo_feedback(store: ChannelSQLiteStore, draft_id: str, decision_id: str) -> None:
    actions = store.list_actions(draft_id=draft_id)
    action = first_manager_review_action(actions)
    if action is None:
        return
    event = build_action_feedback_event(
        action,
        "action_dismissed",
        actor="demo_manager",
        occurred_at=DEMO_START + timedelta(minutes=10),
        decision_id=decision_id,
        reason="Demo risk marker: manager must approve before any live action.",
    )
    store.record_feedback_event(event)


def first_manager_review_action(actions: tuple[ChannelActionRecord, ...]) -> Optional[ChannelActionRecord]:
    for action in actions:
        if action.action.requires_approval:
            return action
    return None


def channel_demo_safety_contract() -> Mapping[str, bool]:
    return {
        **channel_sqlite_safety_contract(),
        **channel_workspace_safety_contract(),
        "demo_only": True,
        "uses_real_credentials": False,
        "live_send": False,
        "write_crm": False,
        "write_tallanto": False,
        "write_runtime_db": False,
        "network_calls": False,
    }


__all__ = [
    "CHANNEL_DEMO_SCHEMA_VERSION",
    "DEMO_START",
    "DemoClock",
    "build_channel_demo_workspace",
    "build_demo_messages",
    "channel_demo_safety_contract",
]
