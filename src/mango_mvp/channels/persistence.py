from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from mango_mvp.channels.contracts import (
    BotReply,
    ChannelAttachment,
    ChannelDirection,
    ChannelMessage,
    ChannelSession,
    RecommendedAction,
    ReplyButton,
    require_text,
    require_timezone,
)
from mango_mvp.channels.feedback import (
    CHANNEL_FEEDBACK_SCHEMA_VERSION,
    FeedbackEvent,
    FeedbackStoreResult,
    feedback_loop_safety_contract,
    summarize_feedback_events,
)
from mango_mvp.channels.preview_service import ChannelDraftPreview
from mango_mvp.channels.signals import (
    CHANNEL_SIGNALS_SCHEMA_VERSION,
    CustomerSignal,
    SafeAnswer,
    SignalDecision,
    SignalEvidence,
    SignalPolicy,
    signal_engine_safety_contract,
)
from mango_mvp.channels.storage import (
    ACTION_STATUS_PROPOSED,
    CHANNEL_STORAGE_SCHEMA_VERSION,
    DRAFT_STATUS_NEEDS_REVIEW,
    ChannelActionRecord,
    ChannelDraftRecord,
    ChannelHistoryEvent,
    ChannelMessageRecord,
    ChannelPreviewStoreResult,
    ChannelStoreWriteResult,
    build_manager_visible_context,
    channel_store_safety_contract,
    ensure_message_session_match,
    now_utc,
    stable_history_event_id,
    validate_action_status,
    validate_action_transition,
    validate_draft_status,
    validate_draft_transition,
)


CHANNEL_SQLITE_SCHEMA_VERSION = "channel_sqlite_storage_v1"
CHANNEL_SQLITE_MIGRATION_ID = "20260511_001_channel_sqlite_storage"
RUNTIME_DB_FILENAMES = {
    "runtime.db",
    "stable_runtime.db",
    "mango_runtime.db",
    "amo_runtime.db",
    "calls.db",
    "transcripts.db",
}
FORBIDDEN_PERSISTED_PAYLOAD_KEYS = {
    "raw_payload",
    "provider_raw_payload",
    "webhook_payload",
    "telegram_update",
    "telegram_raw_update",
    "crm_dialog_payload",
}


@dataclass(frozen=True)
class ChannelSQLiteOpenResult:
    db_path: str
    schema_version: str
    read_only: bool

    def to_json_dict(self) -> Mapping[str, Any]:
        return {
            "db_path": self.db_path,
            "schema_version": self.schema_version,
            "read_only": self.read_only,
        }


class ChannelSQLiteStore:
    """Persistent channel store for client-hosted product roots.

    This store is intentionally limited to local product/channel SQLite files.
    It rejects `stable_runtime` paths and keeps all side effects local: no CRM,
    Tallanto, ASR/R+A, network calls, or live channel sends.
    """

    def __init__(
        self,
        db_path: Path | str,
        *,
        read_only: bool = False,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self.db_path = Path(db_path)
        guard_channel_sqlite_path(self.db_path)
        self.read_only = bool(read_only)
        self._clock = clock or now_utc
        self._con = self._connect()
        if not self.read_only:
            self.bootstrap()

    @classmethod
    def open_read_only(cls, db_path: Path | str) -> "ChannelSQLiteStore":
        return cls(db_path, read_only=True)

    def close(self) -> None:
        self._con.close()

    def __enter__(self) -> "ChannelSQLiteStore":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    @property
    def open_result(self) -> ChannelSQLiteOpenResult:
        return ChannelSQLiteOpenResult(
            db_path=str(self.db_path),
            schema_version=CHANNEL_SQLITE_SCHEMA_VERSION,
            read_only=self.read_only,
        )

    def bootstrap(self) -> None:
        self._ensure_writable()
        self._con.executescript(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
              migration_id TEXT PRIMARY KEY,
              schema_version TEXT NOT NULL,
              applied_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS channel_sessions (
              session_key TEXT PRIMARY KEY,
              channel TEXT NOT NULL,
              channel_thread_id TEXT NOT NULL,
              normalized_customer_id TEXT,
              crm_contact_id TEXT,
              updated_at TEXT NOT NULL,
              inserted_at TEXT NOT NULL,
              record_json TEXT NOT NULL,
              UNIQUE(channel, channel_thread_id)
            );

            CREATE TABLE IF NOT EXISTS channel_messages (
              idempotency_key TEXT PRIMARY KEY,
              session_key TEXT NOT NULL,
              channel TEXT NOT NULL,
              channel_message_id TEXT NOT NULL,
              channel_thread_id TEXT NOT NULL,
              channel_user_id TEXT NOT NULL,
              direction TEXT NOT NULL,
              text TEXT NOT NULL,
              received_at TEXT NOT NULL,
              inserted_at TEXT NOT NULL,
              record_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS channel_drafts (
              draft_id TEXT PRIMARY KEY,
              preview_idempotency_key TEXT NOT NULL UNIQUE,
              session_key TEXT NOT NULL,
              source_message_idempotency_key TEXT NOT NULL,
              status TEXT NOT NULL,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              record_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS channel_actions (
              idempotency_key TEXT PRIMARY KEY,
              draft_id TEXT NOT NULL,
              session_key TEXT NOT NULL,
              source_message_idempotency_key TEXT NOT NULL,
              action_type TEXT NOT NULL,
              status TEXT NOT NULL,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              record_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS channel_history (
              seq INTEGER PRIMARY KEY AUTOINCREMENT,
              event_id TEXT NOT NULL UNIQUE,
              event_type TEXT NOT NULL,
              entity_type TEXT NOT NULL,
              entity_id TEXT NOT NULL,
              session_key TEXT,
              created_at TEXT NOT NULL,
              actor TEXT NOT NULL,
              record_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS channel_signal_decisions (
              decision_id TEXT PRIMARY KEY,
              session_key TEXT NOT NULL,
              message_idempotency_key TEXT NOT NULL,
              created_at TEXT NOT NULL,
              record_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS channel_customer_signals (
              idempotency_key TEXT PRIMARY KEY,
              decision_id TEXT NOT NULL,
              session_key TEXT NOT NULL,
              signal_type TEXT NOT NULL,
              severity TEXT NOT NULL,
              requires_manager_review INTEGER NOT NULL,
              requires_notification INTEGER NOT NULL,
              record_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS channel_feedback_events (
              idempotency_key TEXT PRIMARY KEY,
              event_type TEXT NOT NULL,
              session_key TEXT NOT NULL,
              entity_type TEXT NOT NULL,
              entity_id TEXT NOT NULL,
              decision_id TEXT,
              draft_id TEXT,
              action_idempotency_key TEXT,
              source_system TEXT,
              imported_read_only INTEGER NOT NULL,
              occurred_at TEXT NOT NULL,
              sentiment_bucket TEXT NOT NULL,
              record_json TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS ix_channel_messages_session_time
              ON channel_messages(session_key, received_at);
            CREATE INDEX IF NOT EXISTS ix_channel_drafts_status_time
              ON channel_drafts(status, updated_at);
            CREATE INDEX IF NOT EXISTS ix_channel_actions_status_time
              ON channel_actions(status, updated_at);
            CREATE INDEX IF NOT EXISTS ix_channel_history_entity
              ON channel_history(entity_type, entity_id, created_at);
            CREATE INDEX IF NOT EXISTS ix_channel_signals_session_type
              ON channel_customer_signals(session_key, signal_type);
            CREATE INDEX IF NOT EXISTS ix_channel_feedback_session_time
              ON channel_feedback_events(session_key, occurred_at);
            """
        )
        self._con.execute(
            """
            INSERT OR IGNORE INTO schema_migrations (migration_id, schema_version, applied_at)
            VALUES (?, ?, ?)
            """,
            (CHANNEL_SQLITE_MIGRATION_ID, CHANNEL_SQLITE_SCHEMA_VERSION, self._now().isoformat()),
        )
        self._con.commit()

    def upsert_session(self, session: ChannelSession, *, actor: str = "system") -> ChannelStoreWriteResult:
        self._ensure_writable()
        if not isinstance(session, ChannelSession):
            raise TypeError("session must be ChannelSession")
        key = session.session_key
        payload = session.to_json_dict()
        existing = self._fetch_one("SELECT record_json FROM channel_sessions WHERE session_key = ?", (key,))
        now = self._now()
        if existing is None:
            self._write_session(session, inserted_at=now)
            event = self._append_history(
                event_type="session_created",
                entity_type="channel_session",
                entity_id=key,
                session_key=key,
                actor=actor,
                payload={"channel": session.channel, "channel_thread_id": session.channel_thread_id},
                now=now,
            )
            self._con.commit()
            return ChannelStoreWriteResult("channel_session", key, True, "created", event.event_id)
        if json_loads(existing["record_json"]) == payload:
            return ChannelStoreWriteResult("channel_session", key, False, "duplicate", None)
        self._write_session(session, inserted_at=now)
        event = self._append_history(
            event_type="session_updated",
            entity_type="channel_session",
            entity_id=key,
            session_key=key,
            actor=actor,
            payload={"channel": session.channel, "channel_thread_id": session.channel_thread_id},
            now=now,
        )
        self._con.commit()
        return ChannelStoreWriteResult("channel_session", key, False, "updated", event.event_id)

    def upsert_message(
        self,
        message: ChannelMessage,
        *,
        session: Optional[ChannelSession] = None,
        actor: str = "system",
    ) -> ChannelStoreWriteResult:
        self._ensure_writable()
        resolved_session = session or ChannelSession.from_message(message)
        ensure_message_session_match(message, resolved_session)
        key = message.idempotency_key
        if self._fetch_one("SELECT idempotency_key FROM channel_messages WHERE idempotency_key = ?", (key,)):
            return ChannelStoreWriteResult("channel_message", key, False, "duplicate", None)
        now = self._now()
        self._write_session(resolved_session, inserted_at=now)
        record = ChannelMessageRecord(
            message=message,
            session_key=resolved_session.session_key,
            inserted_at=now,
            metadata={"source": "channel_sqlite_store"},
        )
        self._con.execute(
            """
            INSERT INTO channel_messages (
              idempotency_key, session_key, channel, channel_message_id,
              channel_thread_id, channel_user_id, direction, text,
              received_at, inserted_at, record_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                key,
                resolved_session.session_key,
                message.channel,
                message.channel_message_id,
                message.channel_thread_id,
                message.channel_user_id,
                message.direction.value,
                message.text,
                message.received_at.isoformat(),
                now.isoformat(),
                json_dumps(record.to_json_dict(include_raw_payload=False)),
            ),
        )
        event = self._append_history(
            event_type="message_received" if message.direction.value == "inbound" else "message_recorded",
            entity_type="channel_message",
            entity_id=key,
            session_key=resolved_session.session_key,
            actor=actor,
            payload={
                "channel": message.channel,
                "channel_thread_id": message.channel_thread_id,
                "direction": message.direction.value,
            },
            now=now,
        )
        self._con.commit()
        return ChannelStoreWriteResult("channel_message", key, True, "created", event.event_id)

    def upsert_preview(self, preview: ChannelDraftPreview, *, actor: str = "system") -> ChannelPreviewStoreResult:
        self._ensure_writable()
        if not isinstance(preview, ChannelDraftPreview):
            raise TypeError("preview must be ChannelDraftPreview")
        history_before = self._history_count()
        message_result = self.upsert_message(preview.source_message, session=preview.session, actor=actor)
        existing = self._fetch_one(
            "SELECT record_json FROM channel_drafts WHERE preview_idempotency_key = ?",
            (preview.idempotency_key,),
        )
        if existing is not None:
            draft = draft_record_from_json(json_loads(existing["record_json"]))
            return ChannelPreviewStoreResult(
                draft_id=draft.draft_id,
                preview_idempotency_key=preview.idempotency_key,
                created=False,
                message_created=message_result.created,
                actions_total=len(draft.preview.reply.recommended_actions),
                actions_created=0,
                history_events_created=self._history_count() - history_before,
                status=draft.status,
            )

        now = self._now()
        draft = ChannelDraftRecord(
            preview=preview,
            status=DRAFT_STATUS_NEEDS_REVIEW,
            created_at=now,
            updated_at=now,
            manager_context=build_manager_visible_context(preview),
            metadata={"source": "channel_preview_service"},
        )
        self._write_draft(draft)
        self._append_history(
            event_type="draft_created",
            entity_type="channel_draft",
            entity_id=draft.draft_id,
            session_key=draft.session_key,
            actor=actor,
            payload={
                "preview_idempotency_key": preview.idempotency_key,
                "status": draft.status,
                "recommended_actions": len(preview.reply.recommended_actions),
            },
            now=now,
        )
        actions_created = 0
        for action in preview.reply.recommended_actions:
            if self._upsert_action(action, draft=draft, actor=actor):
                actions_created += 1
        self._con.commit()
        return ChannelPreviewStoreResult(
            draft_id=draft.draft_id,
            preview_idempotency_key=preview.idempotency_key,
            created=True,
            message_created=message_result.created,
            actions_total=len(preview.reply.recommended_actions),
            actions_created=actions_created,
            history_events_created=self._history_count() - history_before,
            status=draft.status,
        )

    def transition_draft(
        self,
        draft_id: str,
        status: str,
        *,
        actor: str,
        reason: Optional[str] = None,
        manager_context: Optional[Mapping[str, Any]] = None,
    ) -> ChannelDraftRecord:
        self._ensure_writable()
        normalized_status = validate_draft_status(status)
        key = require_text(draft_id, "draft_id")
        current = self.get_draft(key)
        if current is None:
            raise KeyError(f"unknown draft_id: {key}")
        reason_text = optional_clean_text(reason)
        if (
            current.status == normalized_status
            and current.status_reason == reason_text
            and (manager_context is None or dict(manager_context) == dict(current.manager_context))
        ):
            return current
        validate_draft_transition(current.status, normalized_status)
        now = self._now()
        updated = current.with_status(
            normalized_status,
            actor=actor,
            reason=reason_text,
            manager_context=manager_context,
            now=now,
        )
        self._write_draft(updated)
        self._append_history(
            event_type="draft_status_changed",
            entity_type="channel_draft",
            entity_id=key,
            session_key=updated.session_key,
            actor=actor,
            payload={
                "from_status": current.status,
                "to_status": updated.status,
                "reason": reason_text,
                "live_send_executed": False,
            },
            now=now,
        )
        self._con.commit()
        return updated

    def transition_action(
        self,
        action_idempotency_key: str,
        status: str,
        *,
        actor: str,
        reason: Optional[str] = None,
    ) -> ChannelActionRecord:
        self._ensure_writable()
        normalized_status = validate_action_status(status)
        key = require_text(action_idempotency_key, "action_idempotency_key")
        current = self.get_action(key)
        if current is None:
            raise KeyError(f"unknown action idempotency key: {key}")
        reason_text = optional_clean_text(reason)
        if current.status == normalized_status and current.status_reason == reason_text and current.actor == actor:
            return current
        validate_action_transition(current.status, normalized_status)
        now = self._now()
        updated = current.with_status(normalized_status, actor=actor, reason=reason_text, now=now)
        self._write_action(updated)
        self._append_history(
            event_type="action_status_changed",
            entity_type="recommended_action",
            entity_id=key,
            session_key=updated.session_key,
            actor=actor,
            payload={
                "from_status": current.status,
                "to_status": updated.status,
                "reason": reason_text,
                "live_execution_performed": False,
            },
            now=now,
        )
        self._con.commit()
        return updated

    def record_signal_decision(self, decision: SignalDecision, *, actor: str = "system") -> ChannelStoreWriteResult:
        self._ensure_writable()
        if not isinstance(decision, SignalDecision):
            raise TypeError("decision must be SignalDecision")
        existing = self._fetch_one(
            "SELECT decision_id FROM channel_signal_decisions WHERE decision_id = ?",
            (decision.decision_id,),
        )
        if existing is not None:
            return ChannelStoreWriteResult("signal_decision", decision.decision_id, False, "duplicate", None)
        payload = scrub_channel_persisted_json(decision.to_json_dict())
        self._con.execute(
            """
            INSERT INTO channel_signal_decisions (
              decision_id, session_key, message_idempotency_key, created_at, record_json
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                decision.decision_id,
                decision.session_key,
                decision.message_idempotency_key,
                decision.created_at.isoformat(),
                json_dumps(payload),
            ),
        )
        for signal in decision.signals:
            policy = signal.policy
            self._con.execute(
                """
                INSERT OR IGNORE INTO channel_customer_signals (
                  idempotency_key, decision_id, session_key, signal_type, severity,
                  requires_manager_review, requires_notification, record_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    signal.idempotency_key,
                    decision.decision_id,
                    decision.session_key,
                    signal.signal_type,
                    signal.severity,
                    1 if signal.requires_manager_review else 0,
                    1 if policy and policy.requires_notification else 0,
                    json_dumps(scrub_channel_persisted_json(signal.to_json_dict())),
                ),
            )
        event = self._append_history(
            event_type="signal_decision_recorded",
            entity_type="signal_decision",
            entity_id=decision.decision_id,
            session_key=decision.session_key,
            actor=actor,
            payload={
                "message_idempotency_key": decision.message_idempotency_key,
                "signal_count": len(decision.signals),
                "requires_manager_review": decision.requires_manager_review,
                "allow_live_execution": False,
            },
            now=self._now(),
        )
        self._con.commit()
        return ChannelStoreWriteResult("signal_decision", decision.decision_id, True, "created", event.event_id)

    def record_feedback_event(self, event: FeedbackEvent) -> FeedbackStoreResult:
        self._ensure_writable()
        if not isinstance(event, FeedbackEvent):
            raise TypeError("event must be FeedbackEvent")
        key = require_text(event.idempotency_key, "event.idempotency_key")
        if self._fetch_one("SELECT idempotency_key FROM channel_feedback_events WHERE idempotency_key = ?", (key,)):
            return FeedbackStoreResult(event_id=key, created=False, status="duplicate")
        self._con.execute(
            """
            INSERT INTO channel_feedback_events (
              idempotency_key, event_type, session_key, entity_type, entity_id,
              decision_id, draft_id, action_idempotency_key, source_system,
              imported_read_only, occurred_at, sentiment_bucket, record_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                key,
                event.event_type,
                event.session_key,
                event.entity_type,
                event.entity_id,
                event.decision_id,
                event.draft_id,
                event.action_idempotency_key,
                event.source_system,
                1 if event.imported_read_only else 0,
                event.occurred_at.isoformat(),
                event.sentiment_bucket,
                json_dumps(event.to_json_dict()),
            ),
        )
        self._con.commit()
        return FeedbackStoreResult(event_id=key, created=True, status="created")

    def record_feedback_many(self, events: Sequence[FeedbackEvent]) -> tuple[FeedbackStoreResult, ...]:
        return tuple(self.record_feedback_event(event) for event in events)

    def get_session(self, session_key: str) -> Optional[ChannelSession]:
        row = self._fetch_one("SELECT record_json FROM channel_sessions WHERE session_key = ?", (session_key,))
        if row is None:
            return None
        return session_from_json(json_loads(row["record_json"]))

    def get_message(self, idempotency_key: str) -> Optional[ChannelMessageRecord]:
        row = self._fetch_one("SELECT record_json FROM channel_messages WHERE idempotency_key = ?", (idempotency_key,))
        if row is None:
            return None
        return message_record_from_json(json_loads(row["record_json"]))

    def get_draft(self, draft_id: str) -> Optional[ChannelDraftRecord]:
        row = self._fetch_one("SELECT record_json FROM channel_drafts WHERE draft_id = ?", (draft_id,))
        if row is None:
            return None
        return draft_record_from_json(json_loads(row["record_json"]))

    def get_action(self, idempotency_key: str) -> Optional[ChannelActionRecord]:
        row = self._fetch_one("SELECT record_json FROM channel_actions WHERE idempotency_key = ?", (idempotency_key,))
        if row is None:
            return None
        return action_record_from_json(json_loads(row["record_json"]))

    def get_feedback_event(self, idempotency_key: str) -> Optional[FeedbackEvent]:
        row = self._fetch_one(
            "SELECT record_json FROM channel_feedback_events WHERE idempotency_key = ?",
            (idempotency_key,),
        )
        if row is None:
            return None
        return feedback_event_from_json(json_loads(row["record_json"]))

    def list_drafts(
        self,
        *,
        status: Optional[str] = None,
        session_key: Optional[str] = None,
    ) -> tuple[ChannelDraftRecord, ...]:
        clauses: list[str] = []
        params: list[Any] = []
        if status is not None:
            clauses.append("status = ?")
            params.append(validate_draft_status(status))
        if session_key:
            clauses.append("session_key = ?")
            params.append(require_text(session_key, "session_key"))
        query = "SELECT record_json FROM channel_drafts"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at, draft_id"
        rows = self._con.execute(query, params).fetchall()
        return tuple(draft_record_from_json(json_loads(row["record_json"])) for row in rows)

    def list_actions(
        self,
        *,
        status: Optional[str] = None,
        session_key: Optional[str] = None,
        draft_id: Optional[str] = None,
    ) -> tuple[ChannelActionRecord, ...]:
        clauses: list[str] = []
        params: list[Any] = []
        if status is not None:
            clauses.append("status = ?")
            params.append(validate_action_status(status))
        if session_key:
            clauses.append("session_key = ?")
            params.append(require_text(session_key, "session_key"))
        if draft_id:
            clauses.append("draft_id = ?")
            params.append(require_text(draft_id, "draft_id"))
        query = "SELECT record_json FROM channel_actions"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at, idempotency_key"
        rows = self._con.execute(query, params).fetchall()
        return tuple(action_record_from_json(json_loads(row["record_json"])) for row in rows)

    def list_history(
        self,
        *,
        session_key: Optional[str] = None,
        entity_type: Optional[str] = None,
        entity_id: Optional[str] = None,
    ) -> tuple[ChannelHistoryEvent, ...]:
        clauses: list[str] = []
        params: list[Any] = []
        if session_key:
            clauses.append("session_key = ?")
            params.append(require_text(session_key, "session_key"))
        if entity_type:
            clauses.append("entity_type = ?")
            params.append(require_text(entity_type, "entity_type"))
        if entity_id:
            clauses.append("entity_id = ?")
            params.append(require_text(entity_id, "entity_id"))
        query = "SELECT record_json FROM channel_history"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY seq"
        rows = self._con.execute(query, params).fetchall()
        return tuple(history_event_from_json(json_loads(row["record_json"])) for row in rows)

    def list_signal_decisions(self, *, session_key: Optional[str] = None) -> tuple[Mapping[str, Any], ...]:
        if session_key:
            rows = self._con.execute(
                "SELECT record_json FROM channel_signal_decisions WHERE session_key = ? ORDER BY created_at, decision_id",
                (require_text(session_key, "session_key"),),
            ).fetchall()
        else:
            rows = self._con.execute(
                "SELECT record_json FROM channel_signal_decisions ORDER BY created_at, decision_id"
            ).fetchall()
        return tuple(json_loads(row["record_json"]) for row in rows)

    def list_feedback_events(
        self,
        *,
        session_key: Optional[str] = None,
        event_type: Optional[str] = None,
        decision_id: Optional[str] = None,
        entity_type: Optional[str] = None,
    ) -> tuple[FeedbackEvent, ...]:
        clauses: list[str] = []
        params: list[Any] = []
        if session_key:
            clauses.append("session_key = ?")
            params.append(require_text(session_key, "session_key"))
        if event_type:
            clauses.append("event_type = ?")
            params.append(require_text(event_type, "event_type"))
        if decision_id:
            clauses.append("decision_id = ?")
            params.append(require_text(decision_id, "decision_id"))
        if entity_type:
            clauses.append("entity_type = ?")
            params.append(require_text(entity_type, "entity_type"))
        query = "SELECT record_json FROM channel_feedback_events"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY occurred_at, idempotency_key"
        rows = self._con.execute(query, params).fetchall()
        return tuple(feedback_event_from_json(json_loads(row["record_json"])) for row in rows)

    def summary(self) -> Mapping[str, Any]:
        draft_status_counts = self._counts_by("channel_drafts", "status")
        action_status_counts = self._counts_by("channel_actions", "status")
        return {
            "schema_version": CHANNEL_SQLITE_SCHEMA_VERSION,
            "storage_schema_version": CHANNEL_STORAGE_SCHEMA_VERSION,
            "backend": "sqlite",
            "read_only": self.read_only,
            "sessions": self._table_count("channel_sessions"),
            "messages": self._table_count("channel_messages"),
            "drafts": self._table_count("channel_drafts"),
            "actions": self._table_count("channel_actions"),
            "history_events": self._table_count("channel_history"),
            "signal_decisions": self._table_count("channel_signal_decisions"),
            "customer_signals": self._table_count("channel_customer_signals"),
            "feedback_events": self._table_count("channel_feedback_events"),
            "draft_status_counts": draft_status_counts,
            "action_status_counts": action_status_counts,
            "validation_ok": True,
            "blocked": 0,
            "warnings": 0,
        }

    def feedback_summary(self, *, session_key: Optional[str] = None) -> Mapping[str, Any]:
        return summarize_feedback_events(self.list_feedback_events(session_key=session_key))

    def snapshot(self, *, include_raw_payload: bool = False) -> Mapping[str, Any]:
        if include_raw_payload:
            # The store never persists raw payloads, so opt-in cannot reveal them.
            include_raw_payload = False
        sessions = self._con.execute("SELECT record_json FROM channel_sessions ORDER BY updated_at, session_key").fetchall()
        messages = self._con.execute("SELECT record_json FROM channel_messages ORDER BY inserted_at, idempotency_key").fetchall()
        return {
            "schema_version": CHANNEL_SQLITE_SCHEMA_VERSION,
            "summary": self.summary(),
            "open_result": self.open_result.to_json_dict(),
            "sessions": [json_loads(row["record_json"]) for row in sessions],
            "messages": [json_loads(row["record_json"]) for row in messages],
            "drafts": [draft.to_json_dict(include_raw_payload=False) for draft in self.list_drafts()],
            "actions": [action.to_json_dict() for action in self.list_actions()],
            "history": [event.to_json_dict() for event in self.list_history()],
            "signal_decisions": list(self.list_signal_decisions()),
            "feedback": {
                "schema_version": CHANNEL_FEEDBACK_SCHEMA_VERSION,
                "summary": self.feedback_summary(),
                "events": [event.to_json_dict() for event in self.list_feedback_events()],
            },
            "safety": channel_sqlite_safety_contract(),
        }

    def _connect(self) -> sqlite3.Connection:
        if self.read_only:
            uri = f"file:{self.db_path}?mode=ro"
            con = sqlite3.connect(uri, uri=True)
            con.execute("PRAGMA query_only = ON")
        else:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")
        return con

    def _ensure_writable(self) -> None:
        if self.read_only:
            raise PermissionError("ChannelSQLiteStore is opened in read-only mode")

    def _now(self) -> datetime:
        value = self._clock()
        require_timezone(value, "clock value")
        return value

    def _fetch_one(self, query: str, params: Sequence[Any]) -> Optional[sqlite3.Row]:
        return self._con.execute(query, tuple(params)).fetchone()

    def _write_session(self, session: ChannelSession, *, inserted_at: datetime) -> None:
        self._con.execute(
            """
            INSERT INTO channel_sessions (
              session_key, channel, channel_thread_id, normalized_customer_id,
              crm_contact_id, updated_at, inserted_at, record_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_key) DO UPDATE SET
              normalized_customer_id = excluded.normalized_customer_id,
              crm_contact_id = excluded.crm_contact_id,
              updated_at = excluded.updated_at,
              record_json = excluded.record_json
            """,
            (
                session.session_key,
                session.channel,
                session.channel_thread_id,
                session.normalized_customer_id,
                session.crm_contact_id,
                session.updated_at.isoformat(),
                inserted_at.isoformat(),
                json_dumps(session.to_json_dict()),
            ),
        )

    def _write_draft(self, draft: ChannelDraftRecord) -> None:
        self._con.execute(
            """
            INSERT INTO channel_drafts (
              draft_id, preview_idempotency_key, session_key, source_message_idempotency_key,
              status, created_at, updated_at, record_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(draft_id) DO UPDATE SET
              status = excluded.status,
              updated_at = excluded.updated_at,
              record_json = excluded.record_json
            """,
            (
                draft.draft_id,
                draft.idempotency_key,
                draft.session_key,
                draft.preview.source_message.idempotency_key,
                draft.status,
                draft.created_at.isoformat(),
                draft.updated_at.isoformat(),
                json_dumps(draft.to_json_dict(include_raw_payload=False)),
            ),
        )

    def _write_action(self, action: ChannelActionRecord) -> None:
        self._con.execute(
            """
            INSERT INTO channel_actions (
              idempotency_key, draft_id, session_key, source_message_idempotency_key,
              action_type, status, created_at, updated_at, record_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(idempotency_key) DO UPDATE SET
              status = excluded.status,
              updated_at = excluded.updated_at,
              record_json = excluded.record_json
            """,
            (
                action.idempotency_key,
                action.draft_id,
                action.session_key,
                action.source_message_idempotency_key,
                action.action.action_type,
                action.status,
                action.created_at.isoformat(),
                action.updated_at.isoformat(),
                json_dumps(action.to_json_dict()),
            ),
        )

    def _upsert_action(self, action: RecommendedAction, *, draft: ChannelDraftRecord, actor: str) -> bool:
        key = require_text(action.idempotency_key, "action.idempotency_key")
        if self._fetch_one("SELECT idempotency_key FROM channel_actions WHERE idempotency_key = ?", (key,)):
            return False
        now = self._now()
        record = ChannelActionRecord(
            action=action,
            draft_id=draft.draft_id,
            session_key=draft.session_key,
            source_message_idempotency_key=draft.preview.source_message.idempotency_key,
            status=ACTION_STATUS_PROPOSED,
            created_at=now,
            updated_at=now,
            metadata={"source": "channel_preview_recommended_action"},
        )
        self._write_action(record)
        self._append_history(
            event_type="action_proposed",
            entity_type="recommended_action",
            entity_id=key,
            session_key=draft.session_key,
            actor=actor,
            payload={
                "draft_id": draft.draft_id,
                "action_type": action.action_type,
                "requires_approval": action.requires_approval,
            },
            now=now,
        )
        return True

    def _append_history(
        self,
        *,
        event_type: str,
        entity_type: str,
        entity_id: str,
        session_key: Optional[str],
        actor: str,
        payload: Mapping[str, Any],
        now: datetime,
    ) -> ChannelHistoryEvent:
        seq = self._next_history_seq()
        event = ChannelHistoryEvent(
            event_id=stable_history_event_id(
                seq=seq,
                event_type=event_type,
                entity_type=entity_type,
                entity_id=entity_id,
                created_at=now,
            ),
            event_type=event_type,
            entity_type=entity_type,
            entity_id=entity_id,
            session_key=session_key,
            created_at=now,
            actor=actor,
            payload=payload,
        )
        self._con.execute(
            """
            INSERT INTO channel_history (
              seq, event_id, event_type, entity_type, entity_id, session_key,
              created_at, actor, record_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                seq,
                event.event_id,
                event.event_type,
                event.entity_type,
                event.entity_id,
                event.session_key,
                event.created_at.isoformat(),
                event.actor,
                json_dumps(event.to_json_dict()),
            ),
        )
        return event

    def _next_history_seq(self) -> int:
        row = self._con.execute("SELECT COALESCE(MAX(seq), 0) + 1 AS next_seq FROM channel_history").fetchone()
        return int(row["next_seq"])

    def _history_count(self) -> int:
        return self._table_count("channel_history")

    def _table_count(self, table_name: str) -> int:
        row = self._con.execute(f"SELECT COUNT(*) AS total FROM {table_name}").fetchone()
        return int(row["total"])

    def _counts_by(self, table_name: str, column_name: str) -> Mapping[str, int]:
        rows = self._con.execute(
            f"SELECT {column_name} AS key, COUNT(*) AS total FROM {table_name} GROUP BY {column_name} ORDER BY {column_name}"
        ).fetchall()
        return {str(row["key"]): int(row["total"]) for row in rows}


def guard_channel_sqlite_path(db_path: Path) -> None:
    candidate = db_path.expanduser()
    if not candidate.name:
        raise ValueError("channel SQLite db path must include a filename")
    try:
        resolved = candidate.resolve(strict=False)
    except RuntimeError as exc:
        raise ValueError(f"refusing unsafe channel SQLite path: {db_path}") from exc
    if candidate.name in RUNTIME_DB_FILENAMES or resolved.name in RUNTIME_DB_FILENAMES:
        raise ValueError(f"refusing runtime-looking DB filename: {candidate.name}")
    if "stable_runtime" in candidate.parts or "stable_runtime" in resolved.parts:
        raise ValueError("refusing channel SQLite DB under stable_runtime")


def channel_sqlite_safety_contract() -> Mapping[str, bool]:
    return {
        **channel_store_safety_contract(),
        **signal_engine_safety_contract(),
        **feedback_loop_safety_contract(),
        "write_runtime_db": False,
        "write_crm": False,
        "write_tallanto": False,
        "live_send": False,
        "network_calls": False,
        "stores_raw_payload_by_default": False,
        "writes_local_channel_product_db": True,
    }


def json_dumps(value: Mapping[str, Any]) -> str:
    return json.dumps(scrub_channel_persisted_json(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def json_loads(value: str) -> Mapping[str, Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("stored channel JSON record must be an object")
    return parsed


def scrub_channel_persisted_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            text_key = str(key)
            normalized_key = text_key.strip().lower()
            if normalized_key in FORBIDDEN_PERSISTED_PAYLOAD_KEYS or "raw_payload" in normalized_key:
                continue
            result[text_key] = scrub_channel_persisted_json(item)
        return result
    if isinstance(value, (list, tuple)):
        return [scrub_channel_persisted_json(item) for item in value]
    return value


def parse_datetime(value: str, field_name: str) -> datetime:
    text = require_text(value, field_name)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def optional_clean_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def attachment_from_json(payload: Mapping[str, Any]) -> ChannelAttachment:
    return ChannelAttachment(
        kind=payload["kind"],
        uri=payload["uri"],
        content_type=payload.get("content_type"),
        size_bytes=payload.get("size_bytes"),
        metadata=payload.get("metadata") or {},
    )


def button_from_json(payload: Mapping[str, Any]) -> ReplyButton:
    return ReplyButton(label=payload["label"], action=payload["action"], payload=payload.get("payload") or {})


def recommended_action_from_json(payload: Mapping[str, Any]) -> RecommendedAction:
    return RecommendedAction(
        action_type=payload["action_type"],
        target_system=payload["target_system"],
        entity_type=payload["entity_type"],
        entity_id=payload.get("entity_id"),
        title=payload.get("title") or "",
        summary=payload.get("summary") or "",
        payload=payload.get("payload") or {},
        confidence=payload.get("confidence"),
        requires_approval=bool(payload.get("requires_approval", True)),
        idempotency_key=payload.get("idempotency_key"),
    )


def bot_reply_from_json(payload: Mapping[str, Any]) -> BotReply:
    return BotReply(
        text=payload.get("text") or "",
        buttons=tuple(button_from_json(item) for item in payload.get("buttons") or ()),
        attachments=tuple(attachment_from_json(item) for item in payload.get("attachments") or ()),
        recommended_actions=tuple(recommended_action_from_json(item) for item in payload.get("recommended_actions") or ()),
        confidence=payload.get("confidence"),
        requires_approval=bool(payload.get("requires_approval", True)),
        safety_flags=tuple(payload.get("safety_flags") or ()),
        metadata=payload.get("metadata") or {},
    )


def message_from_json(payload: Mapping[str, Any]) -> ChannelMessage:
    return ChannelMessage(
        channel=payload["channel"],
        channel_message_id=payload["channel_message_id"],
        channel_thread_id=payload["channel_thread_id"],
        channel_user_id=payload["channel_user_id"],
        direction=ChannelDirection(payload["direction"]),
        text=payload.get("text") or "",
        received_at=parse_datetime(payload["received_at"], "received_at"),
        attachments=tuple(attachment_from_json(item) for item in payload.get("attachments") or ()),
        raw_payload=payload.get("raw_payload") or {},
        metadata=payload.get("metadata") or {},
    )


def session_from_json(payload: Mapping[str, Any]) -> ChannelSession:
    return ChannelSession(
        channel=payload["channel"],
        channel_thread_id=payload["channel_thread_id"],
        normalized_customer_id=payload.get("normalized_customer_id"),
        crm_contact_id=payload.get("crm_contact_id"),
        state=payload.get("state") or {},
        context_summary=payload.get("context_summary"),
        updated_at=parse_datetime(payload["updated_at"], "updated_at"),
    )


def preview_from_json(payload: Mapping[str, Any]) -> ChannelDraftPreview:
    return ChannelDraftPreview(
        draft_id=payload["draft_id"],
        idempotency_key=payload["idempotency_key"],
        source_message=message_from_json(payload["source_message"]),
        session=session_from_json(payload["session"]),
        reply=bot_reply_from_json(payload["reply"]),
        status=payload.get("status") or DRAFT_STATUS_NEEDS_REVIEW,
        created_at=parse_datetime(payload["created_at"], "created_at"),
        blocked_reasons=tuple(payload.get("blocked_reasons") or ()),
        safety=payload.get("safety") or {},
        metadata=payload.get("metadata") or {},
    )


def message_record_from_json(payload: Mapping[str, Any]) -> ChannelMessageRecord:
    return ChannelMessageRecord(
        message=message_from_json(payload["message"]),
        session_key=payload["session_key"],
        inserted_at=parse_datetime(payload["inserted_at"], "inserted_at"),
        metadata=payload.get("metadata") or {},
    )


def draft_record_from_json(payload: Mapping[str, Any]) -> ChannelDraftRecord:
    return ChannelDraftRecord(
        preview=preview_from_json(payload["preview"]),
        status=payload.get("status") or DRAFT_STATUS_NEEDS_REVIEW,
        created_at=parse_datetime(payload["created_at"], "created_at"),
        updated_at=parse_datetime(payload["updated_at"], "updated_at"),
        approved_by=payload.get("approved_by"),
        rejected_by=payload.get("rejected_by"),
        status_reason=payload.get("status_reason"),
        manager_context=payload.get("manager_context") or {},
        metadata=payload.get("metadata") or {},
    )


def action_record_from_json(payload: Mapping[str, Any]) -> ChannelActionRecord:
    return ChannelActionRecord(
        action=recommended_action_from_json(payload["action"]),
        draft_id=payload["draft_id"],
        session_key=payload["session_key"],
        source_message_idempotency_key=payload["source_message_idempotency_key"],
        status=payload.get("status") or ACTION_STATUS_PROPOSED,
        created_at=parse_datetime(payload["created_at"], "created_at"),
        updated_at=parse_datetime(payload["updated_at"], "updated_at"),
        status_reason=payload.get("status_reason"),
        actor=payload.get("actor"),
        metadata=payload.get("metadata") or {},
    )


def history_event_from_json(payload: Mapping[str, Any]) -> ChannelHistoryEvent:
    return ChannelHistoryEvent(
        event_id=payload["event_id"],
        event_type=payload["event_type"],
        entity_type=payload["entity_type"],
        entity_id=payload["entity_id"],
        session_key=payload.get("session_key"),
        created_at=parse_datetime(payload["created_at"], "created_at"),
        actor=payload["actor"],
        payload=payload.get("payload") or {},
    )


def signal_evidence_from_json(payload: Mapping[str, Any]) -> SignalEvidence:
    return SignalEvidence(
        source_type=payload["source_type"],
        source_id=payload["source_id"],
        excerpt=payload.get("excerpt") or "",
        markers=tuple(payload.get("markers") or ()),
        weight=payload.get("weight", 1.0),
        metadata=payload.get("metadata") or {},
    )


def signal_policy_from_json(payload: Mapping[str, Any]) -> SignalPolicy:
    return SignalPolicy(
        signal_type=payload["signal_type"],
        autonomy_level=payload["autonomy_level"],
        requires_manager_review=bool(payload.get("requires_manager_review", True)),
        requires_notification=bool(payload.get("requires_notification", False)),
        allow_autonomous_reply=bool(payload.get("allow_autonomous_reply", False)),
        allow_live_execution=bool(payload.get("allow_live_execution", False)),
        description=payload.get("description") or "",
    )


def customer_signal_from_json(payload: Mapping[str, Any]) -> CustomerSignal:
    return CustomerSignal(
        signal_type=payload["signal_type"],
        title=payload["title"],
        summary=payload["summary"],
        evidence=tuple(signal_evidence_from_json(item) for item in payload.get("evidence") or ()),
        confidence=payload.get("confidence", 0.5),
        severity=payload.get("severity") or "info",
        policy=signal_policy_from_json(payload["policy"]) if payload.get("policy") else None,
        source_action_types=tuple(payload.get("source_action_types") or ()),
        metadata=payload.get("metadata") or {},
        idempotency_key=payload.get("idempotency_key"),
    )


def safe_answer_from_json(payload: Mapping[str, Any]) -> SafeAnswer:
    return SafeAnswer(
        text=payload["text"],
        answer_type=payload.get("answer_type") or "draft",
        requires_approval=bool(payload.get("requires_approval", True)),
        blocked_reasons=tuple(payload.get("blocked_reasons") or ()),
        source_signal_types=tuple(payload.get("source_signal_types") or ()),
        safety_flags=tuple(payload.get("safety_flags") or ()),
        metadata=payload.get("metadata") or {},
    )


def signal_decision_from_json(payload: Mapping[str, Any]) -> SignalDecision:
    return SignalDecision(
        decision_id=payload["decision_id"],
        message_idempotency_key=payload["message_idempotency_key"],
        session_key=payload["session_key"],
        signals=tuple(customer_signal_from_json(item) for item in payload.get("signals") or ()),
        safe_answer=safe_answer_from_json(payload["safe_answer"]),
        recommended_action_types=tuple(payload.get("recommended_action_types") or ()),
        policy_flags=payload.get("policy_flags") or {},
        blocked_reasons=tuple(payload.get("blocked_reasons") or ()),
        created_at=parse_datetime(payload["created_at"], "created_at"),
        metadata=payload.get("metadata") or {},
    )


def feedback_event_from_json(payload: Mapping[str, Any]) -> FeedbackEvent:
    return FeedbackEvent(
        event_type=payload["event_type"],
        session_key=payload["session_key"],
        actor=payload["actor"],
        occurred_at=parse_datetime(payload["occurred_at"], "occurred_at"),
        entity_type=payload.get("entity_type"),
        entity_id=payload.get("entity_id"),
        message_idempotency_key=payload.get("message_idempotency_key"),
        decision_id=payload.get("decision_id"),
        draft_id=payload.get("draft_id"),
        action_idempotency_key=payload.get("action_idempotency_key"),
        source_system=payload.get("source_system"),
        imported_read_only=bool(payload.get("imported_read_only", False)),
        value=payload.get("value"),
        score=payload.get("score"),
        metadata=payload.get("metadata") or {},
        idempotency_key=payload.get("idempotency_key"),
    )


__all__ = [
    "CHANNEL_SQLITE_MIGRATION_ID",
    "CHANNEL_SQLITE_SCHEMA_VERSION",
    "ChannelSQLiteOpenResult",
    "ChannelSQLiteStore",
    "channel_sqlite_safety_contract",
    "guard_channel_sqlite_path",
    "signal_decision_from_json",
]
