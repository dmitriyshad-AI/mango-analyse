from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Optional

from mango_mvp.channels.telegram_pilot_store import (
    PILOT_FEEDBACK_TOPIC_WRONG,
    PILOT_FEEDBACK_UNSAFE_FACT_ATTEMPT,
    TELEGRAM_PILOT_STORE_SCHEMA_VERSION,
    TelegramPilotSQLiteStore,
    guard_telegram_pilot_path,
)


TELEGRAM_PILOT_METRICS_SCHEMA_VERSION = "telegram_pilot_metrics_v1"


@dataclass(frozen=True)
class TelegramPilotMetrics:
    messages: int
    drafts: int
    manager_marked_useful: int = 0
    manager_marked_needs_edit: int = 0
    unsafe_attempts: int = 0

    @property
    def useful_ratio(self) -> float:
        return 0.0 if self.drafts <= 0 else self.manager_marked_useful / self.drafts

    def to_json_dict(self) -> Mapping[str, float | int | str]:
        return {
            "schema_version": TELEGRAM_PILOT_METRICS_SCHEMA_VERSION,
            "messages": self.messages,
            "drafts": self.drafts,
            "manager_marked_useful": self.manager_marked_useful,
            "manager_marked_needs_edit": self.manager_marked_needs_edit,
            "unsafe_attempts": self.unsafe_attempts,
            "useful_ratio": self.useful_ratio,
        }


def build_telegram_pilot_metrics(summary: Mapping[str, int]) -> TelegramPilotMetrics:
    return TelegramPilotMetrics(
        messages=int(summary.get("messages") or 0),
        drafts=int(summary.get("drafts") or 0),
        manager_marked_useful=int(summary.get("manager_marked_useful") or 0),
        manager_marked_needs_edit=int(summary.get("manager_marked_needs_edit") or 0),
        unsafe_attempts=int(summary.get("unsafe_attempts") or 0),
    )


UNSAFE_FLAG_TOKENS = {
    "unsafe_fact_attempt",
    "forbidden_fact_attempt",
    "forbidden_promise",
    "unauthorized_fact",
    "unsafe_precise_fact",
}


@dataclass(frozen=True)
class TelegramPilotDailyMetrics:
    day: str
    incoming_messages: int
    drafts_created: int
    useful_drafts: int
    needs_edit_drafts: int
    manager_only_drafts: int
    blocked_drafts: int
    failed_drafts: int
    topic_errors: int
    unsafe_fact_attempts: int
    avg_seconds_to_draft: Optional[float]
    useful_share: Optional[float]
    unsafe_draft_ids: tuple[str, ...] = ()
    topic_error_draft_ids: tuple[str, ...] = ()

    def to_json_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": TELEGRAM_PILOT_METRICS_SCHEMA_VERSION,
            "day": self.day,
            "incoming_messages": self.incoming_messages,
            "drafts_created": self.drafts_created,
            "useful_drafts": self.useful_drafts,
            "needs_edit_drafts": self.needs_edit_drafts,
            "manager_only_drafts": self.manager_only_drafts,
            "blocked_drafts": self.blocked_drafts,
            "failed_drafts": self.failed_drafts,
            "topic_errors": self.topic_errors,
            "unsafe_fact_attempts": self.unsafe_fact_attempts,
            "avg_seconds_to_draft": self.avg_seconds_to_draft,
            "useful_share": self.useful_share,
            "unsafe_draft_ids": list(self.unsafe_draft_ids),
            "topic_error_draft_ids": list(self.topic_error_draft_ids),
            "store_schema_version": TELEGRAM_PILOT_STORE_SCHEMA_VERSION,
            "safety": telegram_pilot_metrics_safety_contract(),
        }


def build_daily_metrics(
    store_or_db_path: TelegramPilotSQLiteStore | Path | str,
    day: date | str,
) -> TelegramPilotDailyMetrics:
    close_store = False
    if isinstance(store_or_db_path, TelegramPilotSQLiteStore):
        store = store_or_db_path
    else:
        store = TelegramPilotSQLiteStore.open_read_only(store_or_db_path)
        close_store = True
    try:
        summary = store.daily_summary(day)
        drafts = store.list_drafts(day=day)
        feedback_events = store.list_feedback_events(day=day)
    finally:
        if close_store:
            store.close()

    unsafe_draft_ids: set[str] = set()
    for draft in drafts:
        flags = {str(item).strip().lower() for item in draft.get("safety_flags", [])}
        if flags & UNSAFE_FLAG_TOKENS:
            unsafe_draft_ids.add(str(draft.get("draft_id")))

    topic_error_draft_ids: set[str] = set()
    for event in feedback_events:
        event_type = str(event.get("event_type") or "").strip().lower()
        metadata = event.get("metadata") if isinstance(event.get("metadata"), Mapping) else {}
        draft_id = str(event.get("draft_id") or "")
        if event_type == PILOT_FEEDBACK_UNSAFE_FACT_ATTEMPT or bool(metadata.get("unsafe_fact_attempted")):
            unsafe_draft_ids.add(draft_id)
        if event_type == PILOT_FEEDBACK_TOPIC_WRONG or bool(metadata.get("llm_topic_error")):
            topic_error_draft_ids.add(draft_id)

    reviewed = summary.useful_drafts + summary.needs_edit_drafts + summary.manager_only_drafts
    useful_share = round(summary.useful_drafts / reviewed, 4) if reviewed else None
    return TelegramPilotDailyMetrics(
        day=summary.day,
        incoming_messages=summary.incoming_messages,
        drafts_created=summary.drafts_created,
        useful_drafts=summary.useful_drafts,
        needs_edit_drafts=summary.needs_edit_drafts,
        manager_only_drafts=summary.manager_only_drafts,
        blocked_drafts=summary.blocked_drafts,
        failed_drafts=summary.failed_drafts,
        topic_errors=len({item for item in topic_error_draft_ids if item}),
        unsafe_fact_attempts=len({item for item in unsafe_draft_ids if item}),
        avg_seconds_to_draft=summary.avg_seconds_to_draft,
        useful_share=useful_share,
        unsafe_draft_ids=tuple(sorted(item for item in unsafe_draft_ids if item)),
        topic_error_draft_ids=tuple(sorted(item for item in topic_error_draft_ids if item)),
    )


def write_daily_metrics_report(metrics: TelegramPilotDailyMetrics, out_dir: Path | str) -> Mapping[str, str]:
    target = Path(out_dir)
    guard_telegram_pilot_path(target)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / f"telegram_pilot_daily_report_{metrics.day}.json"
    md_path = target / f"telegram_pilot_daily_report_{metrics.day}.md"
    json_path.write_text(json.dumps(metrics.to_json_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_daily_metrics_markdown(metrics), encoding="utf-8")
    return {"json_path": str(json_path), "markdown_path": str(md_path)}


def render_daily_metrics_markdown(metrics: TelegramPilotDailyMetrics) -> str:
    useful_share = "" if metrics.useful_share is None else f"{metrics.useful_share:.2%}"
    avg_seconds = "" if metrics.avg_seconds_to_draft is None else f"{metrics.avg_seconds_to_draft:.1f}"
    return "\n".join(
        [
            f"# Telegram pilot daily report: {metrics.day}",
            "",
            f"- incoming_messages: {metrics.incoming_messages}",
            f"- drafts_created: {metrics.drafts_created}",
            f"- useful_drafts: {metrics.useful_drafts}",
            f"- needs_edit_drafts: {metrics.needs_edit_drafts}",
            f"- manager_only_drafts: {metrics.manager_only_drafts}",
            f"- blocked_drafts: {metrics.blocked_drafts}",
            f"- failed_drafts: {metrics.failed_drafts}",
            f"- topic_errors: {metrics.topic_errors}",
            f"- unsafe_fact_attempts: {metrics.unsafe_fact_attempts}",
            f"- useful_share: {useful_share}",
            f"- avg_seconds_to_draft: {avg_seconds}",
            "",
            "No client send, CRM write, Tallanto write, or stable_runtime write is performed by this report.",
            "",
        ]
    )


def telegram_pilot_metrics_safety_contract() -> Mapping[str, bool]:
    return {
        "network_calls": False,
        "live_send": False,
        "client_send": False,
        "write_crm": False,
        "write_tallanto": False,
        "write_stable_runtime": False,
        "reads_local_pilot_store": True,
    }
