from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from mango_mvp.channels.contracts import ChannelMessage, require_text, stable_digest


TELEGRAM_PILOT_STORE_SCHEMA_VERSION = "telegram_pilot_store_v1"
PILOT_DRAFT_STATUS_NEEDS_REVIEW = "needs_review"
PILOT_DRAFT_STATUS_MANAGER_MARKED_USEFUL = "manager_marked_useful"
PILOT_DRAFT_STATUS_MANAGER_MARKED_NEEDS_EDIT = "manager_marked_needs_edit"
PILOT_DRAFT_STATUS_MANAGER_ONLY = "manager_only"
PILOT_DRAFT_STATUS_BLOCKED = "blocked"
PILOT_DRAFT_STATUS_FAILED = "failed"
PILOT_FEEDBACK_USEFUL = PILOT_DRAFT_STATUS_MANAGER_MARKED_USEFUL
PILOT_FEEDBACK_NEEDS_EDIT = PILOT_DRAFT_STATUS_MANAGER_MARKED_NEEDS_EDIT
PILOT_FEEDBACK_MANAGER_ONLY = PILOT_DRAFT_STATUS_MANAGER_ONLY
PILOT_FEEDBACK_TOPIC_WRONG = "topic_wrong"
PILOT_FEEDBACK_UNSAFE_FACT_ATTEMPT = "unsafe_fact_attempt"
ALLOWED_PILOT_DRAFT_STATUSES = {
    PILOT_DRAFT_STATUS_NEEDS_REVIEW,
    PILOT_DRAFT_STATUS_MANAGER_MARKED_USEFUL,
    PILOT_DRAFT_STATUS_MANAGER_MARKED_NEEDS_EDIT,
    PILOT_DRAFT_STATUS_MANAGER_ONLY,
    PILOT_DRAFT_STATUS_BLOCKED,
    PILOT_DRAFT_STATUS_FAILED,
}
FEEDBACK_STATUS_TRANSITIONS = {
    PILOT_FEEDBACK_USEFUL: PILOT_DRAFT_STATUS_MANAGER_MARKED_USEFUL,
    PILOT_FEEDBACK_NEEDS_EDIT: PILOT_DRAFT_STATUS_MANAGER_MARKED_NEEDS_EDIT,
    PILOT_FEEDBACK_MANAGER_ONLY: PILOT_DRAFT_STATUS_MANAGER_ONLY,
}
RUNTIME_DB_FILENAMES = {
    "runtime.db",
    "stable_runtime.db",
    "mango_runtime.db",
    "amo_runtime.db",
    "calls.db",
    "transcripts.db",
}

Clock = Callable[[], datetime]


@dataclass(frozen=True)
class StoreResult:
    record_type: str
    record_id: str
    created: bool


def guard_telegram_pilot_store_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    guard_telegram_pilot_path(resolved)
    return resolved


class TelegramPilotStore:
    def __init__(self, path: str | Path) -> None:
        self.path = guard_telegram_pilot_store_path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def upsert_message(self, message: ChannelMessage) -> StoreResult:
        if not isinstance(message, ChannelMessage):
            raise TypeError("message must be ChannelMessage")
        with self._connect() as con:
            created = con.execute(
                """
                INSERT OR IGNORE INTO pilot_messages
                  (idempotency_key, channel_thread_id, channel_message_id, received_at, text_hash, record_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    message.idempotency_key,
                    message.channel_thread_id,
                    message.channel_message_id,
                    message.received_at.isoformat(),
                    stable_digest({"text": message.text}),
                    json.dumps(message.to_json_dict(), ensure_ascii=False, sort_keys=True),
                ),
            ).rowcount == 1
        return StoreResult("pilot_message", message.idempotency_key, created)

    def upsert_draft(
        self,
        *,
        draft_id: str,
        message_idempotency_key: str,
        draft_text: str,
        context: Mapping[str, Any],
        status: str = "needs_review",
        created_at: Optional[datetime] = None,
    ) -> StoreResult:
        draft = require_text(draft_id, "draft_id")
        now = created_at or datetime.now(timezone.utc)
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        payload = {
            "schema_version": TELEGRAM_PILOT_STORE_SCHEMA_VERSION,
            "draft_id": draft,
            "message_idempotency_key": message_idempotency_key,
            "draft_text": draft_text,
            "context": dict(context),
            "status": status,
            "created_at": now.isoformat(),
        }
        with self._connect() as con:
            created = con.execute(
                """
                INSERT OR IGNORE INTO pilot_drafts
                  (draft_id, message_idempotency_key, status, created_at, draft_text_hash, record_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    draft,
                    require_text(message_idempotency_key, "message_idempotency_key"),
                    require_text(status, "status"),
                    now.isoformat(),
                    stable_digest({"draft_text": draft_text}),
                    json.dumps(payload, ensure_ascii=False, sort_keys=True),
                ),
            ).rowcount == 1
        return StoreResult("pilot_draft", draft, created)

    def record_feedback(
        self,
        *,
        draft_id: str,
        feedback: str,
        manager_chat_id: str,
        occurred_at: Optional[datetime] = None,
    ) -> StoreResult:
        now = occurred_at or datetime.now(timezone.utc)
        key = f"pilot_feedback:{stable_digest({'draft_id': draft_id, 'feedback': feedback, 'manager_chat_id': manager_chat_id, 'at': now.isoformat()})[:32]}"
        payload = {
            "schema_version": TELEGRAM_PILOT_STORE_SCHEMA_VERSION,
            "feedback_id": key,
            "draft_id": draft_id,
            "feedback": feedback,
            "manager_chat_id": manager_chat_id,
            "occurred_at": now.isoformat(),
        }
        with self._connect() as con:
            created = con.execute(
                """
                INSERT OR IGNORE INTO pilot_feedback
                  (feedback_id, draft_id, feedback, manager_chat_id, occurred_at, record_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    key,
                    require_text(draft_id, "draft_id"),
                    require_text(feedback, "feedback"),
                    require_text(manager_chat_id, "manager_chat_id"),
                    now.isoformat(),
                    json.dumps(payload, ensure_ascii=False, sort_keys=True),
                ),
            ).rowcount == 1
        return StoreResult("pilot_feedback", key, created)

    def daily_summary(self, day: date) -> Mapping[str, Any]:
        start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
        end = datetime.fromtimestamp(start.timestamp() + 86400, tz=timezone.utc)
        with self._connect() as con:
            messages = scalar_count(con, "pilot_messages", "received_at", start, end)
            drafts = scalar_count(con, "pilot_drafts", "created_at", start, end)
            useful = con.execute(
                """
                SELECT COUNT(*) FROM pilot_feedback
                WHERE occurred_at >= ? AND occurred_at < ? AND feedback = 'manager_marked_useful'
                """,
                (start.isoformat(), end.isoformat()),
            ).fetchone()[0]
            needs_edit = con.execute(
                """
                SELECT COUNT(*) FROM pilot_feedback
                WHERE occurred_at >= ? AND occurred_at < ? AND feedback = 'manager_marked_needs_edit'
                """,
                (start.isoformat(), end.isoformat()),
            ).fetchone()[0]
        return {
            "schema_version": TELEGRAM_PILOT_STORE_SCHEMA_VERSION,
            "date": day.isoformat(),
            "messages": messages,
            "drafts": drafts,
            "manager_marked_useful": useful,
            "manager_marked_needs_edit": needs_edit,
        }

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path)
        con.row_factory = sqlite3.Row
        return con

    def _init_schema(self) -> None:
        with self._connect() as con:
            con.executescript(
                """
                CREATE TABLE IF NOT EXISTS pilot_messages (
                  idempotency_key TEXT PRIMARY KEY,
                  channel_thread_id TEXT NOT NULL,
                  channel_message_id TEXT NOT NULL,
                  received_at TEXT NOT NULL,
                  text_hash TEXT NOT NULL,
                  record_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS pilot_drafts (
                  draft_id TEXT PRIMARY KEY,
                  message_idempotency_key TEXT NOT NULL,
                  status TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  draft_text_hash TEXT NOT NULL,
                  record_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS pilot_feedback (
                  feedback_id TEXT PRIMARY KEY,
                  draft_id TEXT NOT NULL,
                  feedback TEXT NOT NULL,
                  manager_chat_id TEXT NOT NULL,
                  occurred_at TEXT NOT NULL,
                  record_json TEXT NOT NULL
                );
                """
            )


def scalar_count(con: sqlite3.Connection, table: str, column: str, start: datetime, end: datetime) -> int:
    return int(
        con.execute(
            f"SELECT COUNT(*) FROM {table} WHERE {column} >= ? AND {column} < ?",
            (start.isoformat(), end.isoformat()),
        ).fetchone()[0]
    )


@dataclass(frozen=True)
class TelegramPilotDraftWriteResult:
    message_key: str
    draft_id: str
    message_created: bool
    context_created: bool
    draft_created: bool
    status: str

    @property
    def created(self) -> bool:
        return self.draft_created

    def to_json_dict(self) -> Mapping[str, Any]:
        return {
            "message_key": self.message_key,
            "draft_id": self.draft_id,
            "message_created": self.message_created,
            "context_created": self.context_created,
            "draft_created": self.draft_created,
            "status": self.status,
        }


@dataclass(frozen=True)
class TelegramPilotFeedbackResult:
    feedback_id: str
    draft_id: str
    event_type: str
    created: bool
    status: str

    def to_json_dict(self) -> Mapping[str, Any]:
        return {
            "feedback_id": self.feedback_id,
            "draft_id": self.draft_id,
            "event_type": self.event_type,
            "created": self.created,
            "status": self.status,
        }


@dataclass(frozen=True)
class TelegramPilotDailySummary:
    day: str
    incoming_messages: int
    drafts_created: int
    useful_drafts: int
    needs_edit_drafts: int
    manager_only_drafts: int
    blocked_drafts: int
    failed_drafts: int
    feedback_events: int
    draft_status_counts: Mapping[str, int]
    feedback_type_counts: Mapping[str, int]
    avg_seconds_to_draft: Optional[float] = None

    def to_json_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": TELEGRAM_PILOT_STORE_SCHEMA_VERSION,
            "day": self.day,
            "incoming_messages": self.incoming_messages,
            "drafts_created": self.drafts_created,
            "useful_drafts": self.useful_drafts,
            "needs_edit_drafts": self.needs_edit_drafts,
            "manager_only_drafts": self.manager_only_drafts,
            "blocked_drafts": self.blocked_drafts,
            "failed_drafts": self.failed_drafts,
            "feedback_events": self.feedback_events,
            "draft_status_counts": dict(self.draft_status_counts),
            "feedback_type_counts": dict(self.feedback_type_counts),
            "avg_seconds_to_draft": self.avg_seconds_to_draft,
            "safety": telegram_pilot_store_safety_contract(),
        }


class TelegramPilotSQLiteStore:
    """Private local SQLite store for the Telegram manager draft pilot."""

    def __init__(
        self,
        path: str | Path,
        *,
        read_only: bool = False,
        clock: Optional[Clock] = None,
    ) -> None:
        self.path = guard_telegram_pilot_store_path(path)
        self.read_only = bool(read_only)
        self._clock = clock or now_utc
        self._con = self._connect()
        if not self.read_only:
            self._init_schema()

    @classmethod
    def open_read_only(cls, path: str | Path) -> "TelegramPilotSQLiteStore":
        return cls(path, read_only=True)

    def close(self) -> None:
        self._con.close()

    def __enter__(self) -> "TelegramPilotSQLiteStore":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def upsert_message_context_draft(
        self,
        message: ChannelMessage | Mapping[str, Any],
        *,
        context: Mapping[str, Any],
        draft_text: str,
        prompt_version: str,
        knowledge_base_version: str,
        status: str = PILOT_DRAFT_STATUS_NEEDS_REVIEW,
        topic_id: Optional[str] = None,
        route: Optional[str] = None,
        safety_flags: Sequence[str] = (),
        blocked_reason: Optional[str] = None,
        draft_metadata: Optional[Mapping[str, Any]] = None,
        actor: str = "system",
    ) -> TelegramPilotDraftWriteResult:
        self._ensure_writable()
        msg = normalize_pilot_message(message)
        message_key = msg["message_key"]
        now = self._now()
        normalized_status = validate_pilot_draft_status(status)
        message_created = self._insert_message(msg, inserted_at=now)
        context_created = self._insert_context(message_key, context, inserted_at=now)

        existing = self._fetch_one("SELECT draft_id, status FROM tgm_pilot_drafts WHERE message_key = ?", (message_key,))
        if existing is not None:
            self._con.commit()
            return TelegramPilotDraftWriteResult(
                message_key=message_key,
                draft_id=str(existing["draft_id"]),
                message_created=message_created,
                context_created=context_created,
                draft_created=False,
                status=str(existing["status"]),
            )

        draft_id = stable_pilot_draft_id(message_key)
        draft_payload = {
            "schema_version": TELEGRAM_PILOT_STORE_SCHEMA_VERSION,
            "draft_id": draft_id,
            "message_key": message_key,
            "status": normalized_status,
            "draft_text": require_text(draft_text, "draft_text"),
            "topic_id": optional_text(topic_id),
            "route": optional_text(route),
            "prompt_version": require_text(prompt_version, "prompt_version"),
            "knowledge_base_version": require_text(knowledge_base_version, "knowledge_base_version"),
            "blocked_reason": optional_text(blocked_reason),
            "safety_flags": [str(item).strip() for item in safety_flags if str(item).strip()],
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "actor": require_text(actor, "actor"),
            "metadata": dict(draft_metadata or {}),
            "safety": telegram_pilot_store_safety_contract(),
        }
        self._con.execute(
            """
            INSERT INTO tgm_pilot_drafts (
              draft_id, message_key, status, draft_text, topic_id, route,
              prompt_version, knowledge_base_version, blocked_reason,
              created_at, updated_at, draft_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                draft_id,
                message_key,
                normalized_status,
                draft_payload["draft_text"],
                draft_payload["topic_id"],
                draft_payload["route"],
                draft_payload["prompt_version"],
                draft_payload["knowledge_base_version"],
                draft_payload["blocked_reason"],
                now.isoformat(),
                now.isoformat(),
                json.dumps(draft_payload, ensure_ascii=False, sort_keys=True),
            ),
        )
        self._con.commit()
        return TelegramPilotDraftWriteResult(message_key, draft_id, message_created, context_created, True, normalized_status)

    def record_feedback(
        self,
        draft_id: str,
        event_type: str,
        *,
        actor: str,
        occurred_at: Optional[datetime] = None,
        reason: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        feedback_id: Optional[str] = None,
    ) -> TelegramPilotFeedbackResult:
        self._ensure_writable()
        draft = require_text(draft_id, "draft_id")
        event = normalize_pilot_key(event_type, "event_type")
        actor_text = require_text(actor, "actor")
        occurred = occurred_at or self._now()
        require_timezone(occurred, "occurred_at")
        if self._fetch_one("SELECT draft_id FROM tgm_pilot_drafts WHERE draft_id = ?", (draft,)) is None:
            raise KeyError(f"unknown draft_id: {draft}")
        key = optional_text(feedback_id) or stable_feedback_id(draft_id=draft, event_type=event, actor=actor_text)
        if self._fetch_one("SELECT feedback_id FROM tgm_pilot_feedback WHERE feedback_id = ?", (key,)) is not None:
            return TelegramPilotFeedbackResult(key, draft, event, False, "duplicate")

        feedback_payload = {
            "schema_version": TELEGRAM_PILOT_STORE_SCHEMA_VERSION,
            "feedback_id": key,
            "draft_id": draft,
            "event_type": event,
            "actor": actor_text,
            "occurred_at": occurred.isoformat(),
            "reason": optional_text(reason),
            "metadata": dict(metadata or {}),
            "safety": telegram_pilot_store_safety_contract(),
        }
        self._con.execute(
            """
            INSERT INTO tgm_pilot_feedback (
              feedback_id, draft_id, event_type, actor, occurred_at,
              reason, metadata_json, feedback_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                key,
                draft,
                event,
                actor_text,
                occurred.isoformat(),
                optional_text(reason),
                json.dumps(dict(metadata or {}), ensure_ascii=False, sort_keys=True),
                json.dumps(feedback_payload, ensure_ascii=False, sort_keys=True),
            ),
        )
        if event in FEEDBACK_STATUS_TRANSITIONS:
            self._update_draft_status(draft, FEEDBACK_STATUS_TRANSITIONS[event], updated_at=occurred)
        self._con.commit()
        return TelegramPilotFeedbackResult(key, draft, event, True, "created")

    def get_draft(self, draft_id: str) -> Optional[Mapping[str, Any]]:
        row = self._fetch_one("SELECT draft_json FROM tgm_pilot_drafts WHERE draft_id = ?", (draft_id,))
        return None if row is None else json.loads(row["draft_json"])

    def list_messages(self, *, day: Optional[date | str] = None) -> tuple[Mapping[str, Any], ...]:
        return self._list_records("tgm_pilot_messages", "message_json", "received_at", day=day)

    def list_drafts(self, *, day: Optional[date | str] = None) -> tuple[Mapping[str, Any], ...]:
        return self._list_records("tgm_pilot_drafts", "draft_json", "created_at", day=day)

    def list_feedback_events(self, *, day: Optional[date | str] = None) -> tuple[Mapping[str, Any], ...]:
        return self._list_records("tgm_pilot_feedback", "feedback_json", "occurred_at", day=day)

    def daily_summary(self, day: date | str) -> TelegramPilotDailySummary:
        day_text, start, end = day_bounds(day)
        params = (start.isoformat(), end.isoformat())
        incoming = self._count(
            "SELECT COUNT(*) FROM tgm_pilot_messages WHERE direction = 'inbound' AND received_at >= ? AND received_at < ?",
            params,
        )
        drafts = self._count("SELECT COUNT(*) FROM tgm_pilot_drafts WHERE created_at >= ? AND created_at < ?", params)
        feedback = self._count("SELECT COUNT(*) FROM tgm_pilot_feedback WHERE occurred_at >= ? AND occurred_at < ?", params)
        status_counts = self._counts_by("tgm_pilot_drafts", "status", "created_at >= ? AND created_at < ?", params)
        feedback_counts = self._counts_by("tgm_pilot_feedback", "event_type", "occurred_at >= ? AND occurred_at < ?", params)
        return TelegramPilotDailySummary(
            day=day_text,
            incoming_messages=incoming,
            drafts_created=drafts,
            useful_drafts=int(feedback_counts.get(PILOT_FEEDBACK_USEFUL, 0)),
            needs_edit_drafts=int(feedback_counts.get(PILOT_FEEDBACK_NEEDS_EDIT, 0)),
            manager_only_drafts=int(feedback_counts.get(PILOT_FEEDBACK_MANAGER_ONLY, 0)),
            blocked_drafts=int(status_counts.get(PILOT_DRAFT_STATUS_BLOCKED, 0)),
            failed_drafts=int(status_counts.get(PILOT_DRAFT_STATUS_FAILED, 0)),
            feedback_events=feedback,
            draft_status_counts=status_counts,
            feedback_type_counts=feedback_counts,
            avg_seconds_to_draft=self._avg_seconds_to_draft(start, end),
        )

    def summary(self) -> Mapping[str, Any]:
        return {
            "schema_version": TELEGRAM_PILOT_STORE_SCHEMA_VERSION,
            "messages": self._table_count("tgm_pilot_messages"),
            "contexts": self._table_count("tgm_pilot_contexts"),
            "drafts": self._table_count("tgm_pilot_drafts"),
            "feedback_events": self._table_count("tgm_pilot_feedback"),
            "draft_status_counts": self._counts_by("tgm_pilot_drafts", "status"),
            "feedback_type_counts": self._counts_by("tgm_pilot_feedback", "event_type"),
            "read_only": self.read_only,
            "safety": telegram_pilot_store_safety_contract(),
        }

    def _connect(self) -> sqlite3.Connection:
        if self.read_only:
            con = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
            con.execute("PRAGMA query_only = ON")
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            con = sqlite3.connect(self.path)
        con.row_factory = sqlite3.Row
        return con

    def _init_schema(self) -> None:
        self._ensure_writable()
        self._con.executescript(
            """
            CREATE TABLE IF NOT EXISTS tgm_pilot_messages (
              message_key TEXT PRIMARY KEY,
              channel TEXT NOT NULL,
              channel_message_id TEXT NOT NULL,
              channel_thread_id TEXT NOT NULL,
              channel_user_id TEXT NOT NULL,
              direction TEXT NOT NULL,
              text TEXT NOT NULL,
              received_at TEXT NOT NULL,
              inserted_at TEXT NOT NULL,
              message_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS tgm_pilot_contexts (
              context_id TEXT PRIMARY KEY,
              message_key TEXT NOT NULL UNIQUE,
              inserted_at TEXT NOT NULL,
              context_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS tgm_pilot_drafts (
              draft_id TEXT PRIMARY KEY,
              message_key TEXT NOT NULL UNIQUE,
              status TEXT NOT NULL,
              draft_text TEXT NOT NULL,
              topic_id TEXT,
              route TEXT,
              prompt_version TEXT NOT NULL,
              knowledge_base_version TEXT NOT NULL,
              blocked_reason TEXT,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              draft_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS tgm_pilot_feedback (
              feedback_id TEXT PRIMARY KEY,
              draft_id TEXT NOT NULL,
              event_type TEXT NOT NULL,
              actor TEXT NOT NULL,
              occurred_at TEXT NOT NULL,
              reason TEXT,
              metadata_json TEXT NOT NULL,
              feedback_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS ix_tgm_pilot_messages_received
              ON tgm_pilot_messages(received_at, channel_thread_id);
            CREATE INDEX IF NOT EXISTS ix_tgm_pilot_drafts_created
              ON tgm_pilot_drafts(created_at, status);
            CREATE INDEX IF NOT EXISTS ix_tgm_pilot_feedback_occurred
              ON tgm_pilot_feedback(occurred_at, event_type);
            """
        )
        self._con.commit()

    def _ensure_writable(self) -> None:
        if self.read_only:
            raise PermissionError("TelegramPilotSQLiteStore is opened in read-only mode")

    def _now(self) -> datetime:
        value = self._clock()
        require_timezone(value, "clock value")
        return value

    def _fetch_one(self, query: str, params: Sequence[Any]) -> Optional[sqlite3.Row]:
        return self._con.execute(query, tuple(params)).fetchone()

    def _insert_message(self, msg: Mapping[str, Any], *, inserted_at: datetime) -> bool:
        if self._fetch_one("SELECT message_key FROM tgm_pilot_messages WHERE message_key = ?", (msg["message_key"],)):
            return False
        payload = {"schema_version": TELEGRAM_PILOT_STORE_SCHEMA_VERSION, **dict(msg), "inserted_at": inserted_at.isoformat()}
        self._con.execute(
            """
            INSERT INTO tgm_pilot_messages (
              message_key, channel, channel_message_id, channel_thread_id,
              channel_user_id, direction, text, received_at, inserted_at, message_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                msg["message_key"],
                msg["channel"],
                msg["channel_message_id"],
                msg["channel_thread_id"],
                msg["channel_user_id"],
                msg["direction"],
                msg["text"],
                msg["received_at"],
                inserted_at.isoformat(),
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
            ),
        )
        return True

    def _insert_context(self, message_key: str, context: Mapping[str, Any], *, inserted_at: datetime) -> bool:
        if self._fetch_one("SELECT context_id FROM tgm_pilot_contexts WHERE message_key = ?", (message_key,)):
            return False
        context_id = "telegram_pilot_context:" + stable_digest({"message_key": message_key})[:32]
        payload = {
            "schema_version": TELEGRAM_PILOT_STORE_SCHEMA_VERSION,
            "context_id": context_id,
            "message_key": message_key,
            "inserted_at": inserted_at.isoformat(),
            "context": dict(context),
            "safety": telegram_pilot_store_safety_contract(),
        }
        self._con.execute(
            "INSERT INTO tgm_pilot_contexts (context_id, message_key, inserted_at, context_json) VALUES (?, ?, ?, ?)",
            (context_id, message_key, inserted_at.isoformat(), json.dumps(payload, ensure_ascii=False, sort_keys=True)),
        )
        return True

    def _update_draft_status(self, draft_id: str, status: str, *, updated_at: datetime) -> None:
        row = self._fetch_one("SELECT draft_json FROM tgm_pilot_drafts WHERE draft_id = ?", (draft_id,))
        if row is None:
            raise KeyError(f"unknown draft_id: {draft_id}")
        payload = json.loads(row["draft_json"])
        payload["status"] = status
        payload["updated_at"] = updated_at.isoformat()
        self._con.execute(
            "UPDATE tgm_pilot_drafts SET status = ?, updated_at = ?, draft_json = ? WHERE draft_id = ?",
            (status, updated_at.isoformat(), json.dumps(payload, ensure_ascii=False, sort_keys=True), draft_id),
        )

    def _list_records(
        self,
        table: str,
        json_column: str,
        time_column: str,
        *,
        day: Optional[date | str],
    ) -> tuple[Mapping[str, Any], ...]:
        if day is None:
            rows = self._con.execute(f"SELECT {json_column} AS payload FROM {table} ORDER BY {time_column}").fetchall()
        else:
            _, start, end = day_bounds(day)
            rows = self._con.execute(
                f"SELECT {json_column} AS payload FROM {table} WHERE {time_column} >= ? AND {time_column} < ? ORDER BY {time_column}",
                (start.isoformat(), end.isoformat()),
            ).fetchall()
        return tuple(json.loads(row["payload"]) for row in rows)

    def _count(self, query: str, params: Sequence[Any] = ()) -> int:
        return int(self._con.execute(query, tuple(params)).fetchone()[0])

    def _table_count(self, table_name: str) -> int:
        return self._count(f"SELECT COUNT(*) FROM {table_name}")

    def _counts_by(
        self,
        table_name: str,
        column_name: str,
        where_clause: Optional[str] = None,
        params: Sequence[Any] = (),
    ) -> Mapping[str, int]:
        query = f"SELECT {column_name}, COUNT(*) FROM {table_name}"
        if where_clause:
            query += f" WHERE {where_clause}"
        query += f" GROUP BY {column_name} ORDER BY {column_name}"
        rows = self._con.execute(query, tuple(params)).fetchall()
        return {str(row[0]): int(row[1]) for row in rows}

    def _avg_seconds_to_draft(self, start: datetime, end: datetime) -> Optional[float]:
        rows = self._con.execute(
            """
            SELECT m.received_at, d.created_at
            FROM tgm_pilot_drafts d
            JOIN tgm_pilot_messages m ON m.message_key = d.message_key
            WHERE d.created_at >= ? AND d.created_at < ?
            """,
            (start.isoformat(), end.isoformat()),
        ).fetchall()
        values = [
            max(0.0, (parse_datetime(row["created_at"]) - parse_datetime(row["received_at"])).total_seconds())
            for row in rows
        ]
        return round(sum(values) / len(values), 3) if values else None


def normalize_pilot_message(message: ChannelMessage | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(message, ChannelMessage):
        return {
            "message_key": message.idempotency_key,
            "channel": message.channel,
            "channel_message_id": message.channel_message_id,
            "channel_thread_id": message.channel_thread_id,
            "channel_user_id": message.channel_user_id,
            "direction": message.direction.value,
            "text": message.text,
            "received_at": message.received_at.isoformat(),
            "metadata": dict(message.metadata),
        }
    payload = dict(message)
    channel = normalize_pilot_key(payload.get("channel") or "telegram_bot", "channel")
    channel_message_id = require_text(payload.get("channel_message_id") or payload.get("message_id"), "channel_message_id")
    channel_thread_id = require_text(payload.get("channel_thread_id") or payload.get("chat_id"), "channel_thread_id")
    direction = str(payload.get("direction") or "inbound").strip().lower()
    text = require_text(payload.get("text"), "text")
    received_at = parse_datetime(payload.get("received_at") or payload.get("date") or now_utc())
    message_key = optional_text(payload.get("message_key") or payload.get("idempotency_key"))
    if not message_key:
        message_key = "telegram_pilot_message:" + stable_digest(
            {"channel": channel, "thread": channel_thread_id, "message": channel_message_id, "direction": direction}
        )[:32]
    return {
        "message_key": message_key,
        "channel": channel,
        "channel_message_id": channel_message_id,
        "channel_thread_id": channel_thread_id,
        "channel_user_id": require_text(payload.get("channel_user_id") or payload.get("user_id") or channel_thread_id, "channel_user_id"),
        "direction": direction,
        "text": text,
        "received_at": received_at.isoformat(),
        "metadata": dict(payload.get("metadata") or {}),
    }


def validate_pilot_draft_status(status: str) -> str:
    normalized = normalize_pilot_key(status, "draft status")
    if normalized not in ALLOWED_PILOT_DRAFT_STATUSES:
        raise ValueError(f"unsupported Telegram pilot draft status: {status!r}")
    return normalized


def stable_pilot_draft_id(message_key: str) -> str:
    return "telegram_pilot_draft:" + stable_digest({"message_key": require_text(message_key, "message_key")})[:32]


def stable_feedback_id(*, draft_id: str, event_type: str, actor: str) -> str:
    return "telegram_pilot_feedback:" + stable_digest(
        {"draft_id": require_text(draft_id, "draft_id"), "event_type": event_type, "actor": require_text(actor, "actor")}
    )[:32]


def day_bounds(value: date | str) -> tuple[str, datetime, datetime]:
    if isinstance(value, str):
        day = date.fromisoformat(value)
    elif isinstance(value, datetime):
        day = value.date()
    else:
        day = value
    start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    return day.isoformat(), start, start + timedelta(days=1)


def guard_telegram_pilot_path(path: str | Path) -> None:
    candidate = Path(path).expanduser()
    try:
        resolved = candidate.resolve(strict=False)
    except RuntimeError as exc:
        raise ValueError(f"refusing unsafe Telegram pilot path: {path}") from exc
    if "stable_runtime" in candidate.parts or "stable_runtime" in resolved.parts:
        raise ValueError("refusing Telegram pilot path under stable_runtime")
    if candidate.name in RUNTIME_DB_FILENAMES or resolved.name in RUNTIME_DB_FILENAMES:
        raise ValueError(f"refusing runtime-looking DB filename: {candidate.name}")


def telegram_pilot_store_safety_contract() -> Mapping[str, bool]:
    return {
        "network_calls": False,
        "live_send": False,
        "client_send": False,
        "write_crm": False,
        "write_tallanto": False,
        "write_stable_runtime": False,
        "write_runtime_db": False,
        "stores_private_full_text_locally": True,
    }


def parse_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        text = require_text(value, "datetime")
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def require_timezone(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")


def optional_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def normalize_pilot_key(value: Any, field_name: str) -> str:
    text = require_text(value, field_name).lower()
    allowed = set("abcdefghijklmnopqrstuvwxyz0123456789_.:-")
    if any(ch not in allowed for ch in text):
        raise ValueError(f"{field_name} contains unsupported characters: {value!r}")
    return text


def now_utc() -> datetime:
    return datetime.now(timezone.utc)
