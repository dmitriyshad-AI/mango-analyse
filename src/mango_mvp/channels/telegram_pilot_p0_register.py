from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from mango_mvp.channels.conversation_orchestrator import P0_THEME_ROUTING_RULES
from mango_mvp.channels.telegram_pilot_store import guard_telegram_pilot_path


TELEGRAM_PILOT_P0_REGISTER_SCHEMA_VERSION = "telegram_pilot_p0_register_v1"
AUTONOMOUS_ROUTES = {"bot_answer_self", "bot_answer_self_for_pilot"}
P0_TEXT_RE = re.compile(
    r"возврат|верн\w+\s+деньг|отказ\w*\s+от\s+обуч|жалоб|недовол|претензи|суд|иск|прокуратур|роспотребнадзор",
    re.I,
)
P0_FLAG_MARKERS = ("high_risk", "refund", "legal", "complaint", "negative_feedback", "combined_high_risk")
P0_REGISTER_FIELDS = (
    "schema_version",
    "created_at",
    "brand",
    "chat_id",
    "topic_id",
    "route",
    "severity",
    "risk_code",
    "trigger",
    "client_send_executed",
    "input_text",
    "answer_text",
    "safety_flags_json",
    "metadata_json",
)


@dataclass(frozen=True)
class TelegramPilotP0Record:
    brand: str
    chat_id: str
    topic_id: str
    route: str
    severity: str
    risk_code: str
    trigger: str
    client_send_executed: bool
    input_text: str
    answer_text: str
    safety_flags: tuple[str, ...]
    metadata: Mapping[str, Any]
    created_at: str

    def to_row(self) -> Mapping[str, str]:
        return {
            "schema_version": TELEGRAM_PILOT_P0_REGISTER_SCHEMA_VERSION,
            "created_at": self.created_at,
            "brand": self.brand,
            "chat_id": self.chat_id,
            "topic_id": self.topic_id,
            "route": self.route,
            "severity": self.severity,
            "risk_code": self.risk_code,
            "trigger": self.trigger,
            "client_send_executed": "true" if self.client_send_executed else "false",
            "input_text": self.input_text,
            "answer_text": self.answer_text,
            "safety_flags_json": json.dumps(list(self.safety_flags), ensure_ascii=False),
            "metadata_json": json.dumps(dict(self.metadata), ensure_ascii=False, sort_keys=True),
        }


def build_p0_register_record(
    *,
    brand: str,
    chat_id: str | int,
    input_text: str,
    answer_text: str,
    topic_id: str,
    route: str,
    safety_flags: Sequence[str] = (),
    client_send_executed: bool,
    metadata: Optional[Mapping[str, Any]] = None,
    created_at: Optional[datetime] = None,
) -> Optional[TelegramPilotP0Record]:
    trigger = p0_trigger(topic_id=topic_id, input_text=input_text, safety_flags=safety_flags)
    if not trigger:
        return None
    risk_code = str((P0_THEME_ROUTING_RULES.get(topic_id) or {}).get("risk_code") or trigger)
    severity = "P0_BLOCKED"
    if str(route or "") in AUTONOMOUS_ROUTES:
        severity = "P0_AUTONOMOUS_ROUTE_ATTEMPT"
    now = created_at or datetime.now(timezone.utc)
    if now.tzinfo is None or now.utcoffset() is None:
        now = now.replace(tzinfo=timezone.utc)
    return TelegramPilotP0Record(
        brand=str(brand or ""),
        chat_id=str(chat_id),
        topic_id=str(topic_id or ""),
        route=str(route or ""),
        severity=severity,
        risk_code=risk_code,
        trigger=trigger,
        client_send_executed=bool(client_send_executed),
        input_text=str(input_text or "")[:1200],
        answer_text=str(answer_text or "")[:1200],
        safety_flags=tuple(str(item) for item in safety_flags if str(item).strip()),
        metadata=dict(metadata or {}),
        created_at=now.astimezone(timezone.utc).isoformat(),
    )


def p0_trigger(*, topic_id: str, input_text: str, safety_flags: Sequence[str] = ()) -> str:
    if topic_id in P0_THEME_ROUTING_RULES:
        return f"topic:{topic_id}"
    flags = " ".join(str(item or "").casefold() for item in safety_flags)
    if any(marker in flags for marker in P0_FLAG_MARKERS):
        return "safety_flags"
    if P0_TEXT_RE.search(str(input_text or "")):
        return "input_text"
    return ""


def append_p0_register_record(path: str | Path, record: TelegramPilotP0Record) -> Path:
    target = Path(path)
    guard_telegram_pilot_path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    exists = target.exists() and target.stat().st_size > 0
    with target.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=P0_REGISTER_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: record.to_row().get(field, "") for field in P0_REGISTER_FIELDS})
    return target
