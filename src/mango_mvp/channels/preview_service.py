from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Optional

from mango_mvp.channels.actions import build_channel_recommended_actions
from mango_mvp.channels.contracts import (
    BotReply,
    ChannelDirection,
    ChannelMessage,
    ChannelSession,
    stable_digest,
)


CHANNEL_PREVIEW_SCHEMA_VERSION = "channel_preview_v1"
DEFAULT_DRAFT_STATUS = "needs_review"
DEFAULT_DRAFT_TEXT = "Здравствуйте! Спасибо за сообщение. Уточним детали и вернемся с ответом."


@dataclass(frozen=True)
class ChannelDraftPreview:
    draft_id: str
    idempotency_key: str
    source_message: ChannelMessage
    session: ChannelSession
    reply: BotReply
    status: str = DEFAULT_DRAFT_STATUS
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    blocked_reasons: tuple[str, ...] = field(default_factory=tuple)
    safety: Mapping[str, bool] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.draft_id.strip():
            raise ValueError("draft_id must not be empty")
        if not self.idempotency_key.strip():
            raise ValueError("idempotency_key must not be empty")
        if self.session.channel != self.source_message.channel:
            raise ValueError("session channel must match source message channel")
        if self.session.channel_thread_id != self.source_message.channel_thread_id:
            raise ValueError("session thread must match source message thread")
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        object.__setattr__(self, "status", str(self.status or "").strip() or DEFAULT_DRAFT_STATUS)
        object.__setattr__(self, "blocked_reasons", tuple(str(item).strip() for item in self.blocked_reasons if str(item).strip()))
        object.__setattr__(self, "safety", dict(self.safety or default_preview_safety()))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_json_dict(self, *, include_raw_payload: bool = False) -> Mapping[str, Any]:
        source = self.source_message.to_json_dict()
        if not include_raw_payload:
            source = dict(source)
            source.pop("raw_payload", None)
        return {
            "schema_version": CHANNEL_PREVIEW_SCHEMA_VERSION,
            "draft_id": self.draft_id,
            "idempotency_key": self.idempotency_key,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "source_message": source,
            "session": self.session.to_json_dict(),
            "reply": self.reply.to_json_dict(),
            "blocked_reasons": list(self.blocked_reasons),
            "safety": dict(self.safety),
            "metadata": dict(self.metadata),
        }


class ChannelPreviewService:
    def __init__(self, *, default_draft_text: str = DEFAULT_DRAFT_TEXT) -> None:
        self.default_draft_text = str(default_draft_text or DEFAULT_DRAFT_TEXT).strip() or DEFAULT_DRAFT_TEXT

    def build_preview(
        self,
        message: ChannelMessage,
        *,
        session: Optional[ChannelSession] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> ChannelDraftPreview:
        if message.direction != ChannelDirection.INBOUND:
            raise ValueError("channel draft preview supports inbound messages only")
        resolved_session = session or ChannelSession.from_message(message)
        if resolved_session.channel != message.channel:
            raise ValueError("session channel must match source message channel")
        if resolved_session.channel_thread_id != message.channel_thread_id:
            raise ValueError("session thread must match source message thread")
        context_payload = dict(context or {})
        draft_id = stable_draft_id(message)
        draft_text = build_safe_draft_text(message, context=context_payload, default_text=self.default_draft_text)
        catalog_context_used = bool(context_payload.get("question_catalog_answer") or context_payload.get("approved_question_answer"))
        actions = build_channel_recommended_actions(
            message=message,
            session=resolved_session,
            draft_id=draft_id,
            draft_text=draft_text,
            context=context_payload,
        )
        reply = BotReply(
            text=draft_text,
            recommended_actions=actions,
            confidence=0.35,
            requires_approval=True,
            safety_flags=tuple(
                flag
                for flag in (
                "draft_only",
                "requires_manager_approval",
                "live_send_disabled",
                "no_llm_used",
                "no_rag_used",
                "question_catalog_context_used" if catalog_context_used else "",
                )
                if flag
            ),
            metadata={
                "draft_id": draft_id,
                "source_message_idempotency_key": message.idempotency_key,
                "preview_mode": "deterministic_safe_draft",
                "question_catalog_context_used": catalog_context_used,
            },
        )
        return ChannelDraftPreview(
            draft_id=draft_id,
            idempotency_key=stable_preview_idempotency_key(message),
            source_message=message,
            session=resolved_session,
            reply=reply,
            blocked_reasons=("live_send_disabled", "manager_approval_required"),
            safety=default_preview_safety(),
            metadata={
                "context_keys": tuple(sorted(str(key) for key in context_payload.keys())),
                "preview_mode": "deterministic_safe_draft",
                "question_catalog_context_used": catalog_context_used,
            },
        )


def build_channel_draft_preview(
    message: ChannelMessage,
    *,
    session: Optional[ChannelSession] = None,
    context: Optional[Mapping[str, Any]] = None,
) -> ChannelDraftPreview:
    return ChannelPreviewService().build_preview(message, session=session, context=context)


def build_safe_draft_text(
    message: ChannelMessage,
    *,
    context: Mapping[str, Any],
    default_text: str,
) -> str:
    override = str(context.get("safe_draft_text") or "").strip()
    if override:
        return override
    catalog_answer = context.get("question_catalog_answer") or context.get("approved_question_answer")
    if isinstance(catalog_answer, Mapping):
        final_answer = str(catalog_answer.get("final_approved_answer") or "").strip()
        approved_for_bot = catalog_answer.get("approved_for_bot") is True or str(catalog_answer.get("approved_for_bot")).lower() == "yes"
        required_fact_keys = tuple(str(item).strip() for item in catalog_answer.get("required_fact_keys") or () if str(item).strip())
        facts_fresh = context.get("question_catalog_facts_fresh") is True
        if approved_for_bot and final_answer and (not required_fact_keys or facts_fresh):
            return final_answer
        return "Здравствуйте! Спасибо за вопрос. Передадим его менеджеру и вернемся с проверенным ответом."
    if message.attachments and not message.text:
        return "Здравствуйте! Получили вложение. Проверим детали и вернемся с ответом."
    if "?" in message.text or any(marker in message.text.lower() for marker in ("подскаж", "сколько", "можно", "как ")):
        return "Здравствуйте! Спасибо за вопрос. Уточним детали и вернемся с ответом."
    return default_text


def stable_draft_id(message: ChannelMessage) -> str:
    digest = stable_digest(
        {
            "source_message_idempotency_key": message.idempotency_key,
            "channel": message.channel,
            "thread": message.channel_thread_id,
        }
    )
    return f"channel_draft:{message.channel}:{digest[:32]}"


def stable_preview_idempotency_key(message: ChannelMessage) -> str:
    return f"channel_preview:{message.channel}:{stable_digest({'message_key': message.idempotency_key})[:32]}"


def default_preview_safety() -> Mapping[str, bool]:
    return {
        "network_calls": False,
        "llm_calls": False,
        "rag_used": False,
        "live_send": False,
        "write_crm": False,
        "write_tallanto": False,
        "write_runtime_db": False,
        "run_asr": False,
        "run_ra": False,
        "requires_manager_approval": True,
    }
