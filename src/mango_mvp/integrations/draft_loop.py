from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Protocol, Sequence
from zoneinfo import ZoneInfo

from mango_mvp.channels.brand_boundary import foreign_brand_rule
from mango_mvp.channels.dialogue_memory import (
    MEMORY_PROVENANCE_ENV,
    _dialog_summary_candidate,
    _dialog_summary_rolling_enabled,
    update_dialogue_memory_after_answer,
)
from mango_mvp.channels.subscription_llm import SubscriptionDraftResult
from mango_mvp.channels.subscription_llm_parts.semantic_reading import SemanticReading
from mango_mvp.integrations.amo_wappi_phase1 import (
    AmoWappiHttpError,
    AmoWappiPhase1Config,
    ManagerEditLogRecord,
    WappiPhase1Client,
    append_manager_edit_log,
)


DEFAULT_DRAFT_LOOP_DIR = Path.home() / ".mango_local" / "draft_loop"
DEFAULT_STOP_PATH = Path.home() / ".mango_secrets" / "STOP_DRAFT_LOOP"
DEFAULT_DEBOUNCE_SECONDS = 60
DEFAULT_HISTORY_LIMIT = 15
DEFAULT_WAPPI_MESSAGE_FETCH_LIMIT = 50
MIN_RAW_HISTORY_LIMIT = 12
OLDER_DIALOGUE_SUMMARY_PREFIX = "Ранее в диалоге:"
CONFIG_FINGERPRINT_SCHEMA_VERSION = "draft_loop_config_fingerprint_v1_2026_06_10"
DEFAULT_AUTH_ERROR_LIMIT = 3
MOSCOW_TZ = ZoneInfo("Europe/Moscow")
_SERVICE_HISTORY_MARKER_RE = re.compile(
    r"\b(?:message_id|chat_id|profile_id|lead_id|source_system|dedupe_key|event_id)\b|[{}\[\]]",
    re.IGNORECASE,
)
class DraftLoopError(RuntimeError):
    pass


class DraftLoopConfigError(DraftLoopError):
    pass


class DraftLoopPaginationChanged(DraftLoopError):
    pass


def _is_deferred_fetch_exception(exc: BaseException) -> bool:
    message = str(exc).casefold()
    return isinstance(exc, DraftLoopPaginationChanged) or (
        isinstance(exc, AmoWappiHttpError)
        and "http 400" in message
        and "сохранена для повторной отправки" in message
    )


def _memory_provenance_enabled() -> bool:
    explicit = os.getenv(MEMORY_PROVENANCE_ENV)
    if explicit is not None:
        return str(explicit).strip().lower() in {"1", "true", "yes", "on"}
    return str(os.getenv("TELEGRAM_DIRECT_PATH_PILOT_CONFIG") or "").strip() == "pilot_gold_v1"


@dataclass(frozen=True, order=True)
class DraftLoopKey:
    profile_id: str
    chat_id: str

    def __post_init__(self) -> None:
        if not str(self.profile_id or "").strip() or not str(self.chat_id or "").strip():
            raise DraftLoopConfigError("Draft loop key requires both profile_id and chat_id.")
        object.__setattr__(self, "profile_id", str(self.profile_id).strip())
        object.__setattr__(self, "chat_id", str(self.chat_id).strip())

    @property
    def value(self) -> str:
        return f"{self.profile_id}:{self.chat_id}"


@dataclass(frozen=True)
class DraftLoopProfile:
    profile_id: str
    brand: str
    channel: str = "telegram"


@dataclass(frozen=True)
class DraftLoopPair:
    key: DraftLoopKey
    lead_id: str
    expected_brand: str
    not_before_ts: int = 0
    source: str = "manual"
    match_key: str = ""
    contact_id: str = ""
    auto_note: str = ""


@dataclass(frozen=True)
class DraftLoopConfig:
    profiles: Mapping[str, DraftLoopProfile]
    pairs: Mapping[DraftLoopKey, DraftLoopPair] = field(default_factory=dict)
    auto_pairs_path: Path | None = None
    allowed_test_lead_ids: frozenset[str] = field(default_factory=frozenset)
    state_path: Path = DEFAULT_DRAFT_LOOP_DIR / "state.json"
    journal_path: Path = DEFAULT_DRAFT_LOOP_DIR / "journal.jsonl"
    manager_edit_log_path: Path = DEFAULT_DRAFT_LOOP_DIR / "manager_edits.jsonl"
    heartbeat_path: Path = DEFAULT_DRAFT_LOOP_DIR / "heartbeat.json"
    stop_path: Path = DEFAULT_STOP_PATH
    debounce_seconds: int = DEFAULT_DEBOUNCE_SECONDS
    history_limit: int = DEFAULT_HISTORY_LIMIT
    chat_limit: int = 0
    all_personal_mode: bool = False
    auth_error_limit: int = DEFAULT_AUTH_ERROR_LIMIT
    manager_outgoing_visible: bool | None = None
    config_fingerprint: Mapping[str, str] = field(default_factory=dict)

    def brand_for_profile(self, profile_id: str) -> str:
        profile = self.profiles.get(str(profile_id).strip())
        if profile is None:
            raise DraftLoopConfigError(f"Unknown Wappi profile_id: {profile_id!r}")
        return profile.brand

    def pair_for(self, key: DraftLoopKey) -> DraftLoopPair | None:
        return self.pairs_snapshot().get(key)

    def pairs_snapshot(self) -> dict[DraftLoopKey, DraftLoopPair]:
        result = dict(self.pairs)
        if self.auto_pairs_path is not None:
            path = self.auto_pairs_path.expanduser()
            if path.exists():
                result.update(load_pairs_file(path, default_source="auto"))
        return result

    def phase1_config(self) -> AmoWappiPhase1Config:
        allowed = {str(item) for item in self.allowed_test_lead_ids}
        allowed.update(str(pair.lead_id) for pair in self.pairs_snapshot().values() if str(pair.lead_id))
        return AmoWappiPhase1Config(
            profile_brand_map={profile_id: profile.brand for profile_id, profile in self.profiles.items()},
            allowed_test_lead_ids=frozenset(allowed),
            manager_edit_log_path=self.manager_edit_log_path,
        )


def build_draft_loop_config_fingerprint(
    snapshot_path: Path | str | None,
    *,
    gold_pack_version: str,
    repo_root: Path | str | None = None,
) -> Mapping[str, str]:
    root = Path(repo_root).expanduser() if repo_root is not None else Path(__file__).resolve().parents[3]
    tree_hash = "unknown"
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short=8", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        tree_hash = completed.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        tree_hash = "unknown"
    kb_release_dir = ""
    if snapshot_path is not None:
        kb_release_dir = Path(snapshot_path).expanduser().parent.name
    return {
        "schema_version": CONFIG_FINGERPRINT_SCHEMA_VERSION,
        "tree_hash": tree_hash,
        "kb_release_dir": kb_release_dir,
        "gold_pack_version": str(gold_pack_version or ""),
    }


def build_draft_loop_code_identity(repo_root: Path | str | None = None) -> Mapping[str, str]:
    root = Path(repo_root).expanduser() if repo_root is not None else Path(__file__).resolve().parents[3]
    code_root = str(root.resolve())
    git_sha = "unknown"
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        if len(lines) >= 2:
            code_root, git_sha = lines[0], lines[1]
    except (OSError, subprocess.SubprocessError):
        pass
    return {"code_root": code_root, "git_sha": git_sha}


@dataclass(frozen=True)
class WappiHistoryMessage:
    profile_id: str
    chat_id: str
    message_id: str
    text: str
    message_type: str
    timestamp: int
    from_me: bool
    contact_name: str = ""
    from_where: str = ""
    raw: Mapping[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.profile_id, self.chat_id, self.message_id)

    @property
    def is_inbound_text(self) -> bool:
        return (
            not self.from_me
            and self.message_type in {"text", "image", "photo", "video", "document"}
            and bool(self.text.strip())
        )

    @property
    def is_inbound_actionable(self) -> bool:
        return not self.from_me


class DraftBotProvider(Protocol):
    def build_draft(self, client_message: str, *, context: Mapping[str, Any] | None = None) -> SubscriptionDraftResult:
        ...


class AmoDraftNoteClient(Protocol):
    def add_draft_note_to_test_lead(self, lead_id: int | str, **kwargs: Any) -> Mapping[str, Any]:
        ...

    def add_draft_note_to_contact(self, contact_id: int | str, **kwargs: Any) -> Mapping[str, Any]:
        ...


ContextBuilder = Callable[..., Mapping[str, Any]]
AutoResolver = Callable[..., Optional[Mapping[str, Any]]]


class DraftLoopJournal:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path).expanduser()

    def append(self, event: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(event)
        payload.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

    def rows(self) -> list[Mapping[str, Any]]:
        if not self.path.exists():
            return []
        result: list[Mapping[str, Any]] = []
        for raw in self.path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            try:
                decoded = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(decoded, Mapping):
                result.append(decoded)
        return result

    def processed_message_keys(self) -> set[tuple[str, str, str]]:
        processed: set[tuple[str, str, str]] = set()
        for row in self.rows():
            if str(row.get("status") or "") != "note_written":
                continue
            profile_id = str(row.get("profile_id") or "")
            chat_id = str(row.get("chat_id") or "")
            covered = row.get("covered_message_ids")
            message_ids = (
                [str(item or "") for item in covered]
                if isinstance(covered, Sequence) and not isinstance(covered, (str, bytes, bytearray))
                else [str(row.get("message_id") or "")]
            )
            if profile_id and chat_id:
                processed.update((profile_id, chat_id, message_id) for message_id in message_ids if message_id)
        return processed


class DraftLoopState:
    def __init__(self, path: Path | str, *, persist: bool = True) -> None:
        self.path = Path(path).expanduser()
        self.persist = bool(persist)
        self.payload = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {
                "processed": [],
                "pending_notes": {},
                "dialogue_memory": {},
                "manager_edit_matches": [],
                "auth_error_count": 0,
                "quarantined_pairs": {},
                "deferred_pair_missing": {},
                "inbound_not_before_ts": 0,
            }
        try:
            decoded = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise DraftLoopConfigError(f"Invalid draft loop state JSON: {self.path}") from exc
        if not isinstance(decoded, dict):
            raise DraftLoopConfigError("Draft loop state must be a JSON object.")
        decoded.setdefault("processed", [])
        decoded.setdefault("pending_notes", {})
        decoded.setdefault("dialogue_memory", {})
        decoded.setdefault("manager_edit_matches", [])
        decoded.setdefault("auth_error_count", 0)
        decoded.setdefault("quarantined_pairs", {})
        decoded.setdefault("deferred_pair_missing", {})
        decoded.setdefault("inbound_not_before_ts", 0)
        return decoded

    def ensure_inbound_not_before_ts(self, now_ts: int) -> int:
        try:
            current = int(self.payload.get("inbound_not_before_ts") or 0)
        except (TypeError, ValueError):
            current = 0
        if current > 0:
            return current
        current = max(1, int(now_ts))
        self.payload["inbound_not_before_ts"] = current
        self.save()
        return current

    def processed_keys(self) -> set[tuple[str, str, str]]:
        result: set[tuple[str, str, str]] = set()
        for item in self.payload.get("processed") or []:
            if not isinstance(item, Mapping):
                continue
            profile_id = str(item.get("profile_id") or "")
            chat_id = str(item.get("chat_id") or "")
            message_id = str(item.get("message_id") or "")
            if profile_id and chat_id and message_id:
                result.add((profile_id, chat_id, message_id))
        return result

    def mark_processed(self, message: WappiHistoryMessage) -> None:
        self.mark_processed_key(message.profile_id, message.chat_id, message.message_id)

    def mark_processed_key(self, profile_id: str, chat_id: str, message_id: str) -> None:
        items = list(self.payload.get("processed") or [])
        marker = {"profile_id": profile_id, "chat_id": chat_id, "message_id": message_id}
        if marker not in items:
            items.append(marker)
        self.payload["processed"] = items[-5000:]
        deferred = dict(self.payload.get("deferred_pair_missing") or {})
        deferred.pop(f"{profile_id}\t{chat_id}\t{message_id}", None)
        self.payload["deferred_pair_missing"] = deferred

    def pair_missing_is_deferred(self, message: WappiHistoryMessage) -> bool:
        deferred = self.payload.get("deferred_pair_missing")
        return isinstance(deferred, Mapping) and _message_state_key(message) in deferred

    def deferred_pair_missing_messages(self, profile_id: str, chat_id: str) -> list[WappiHistoryMessage]:
        deferred = self.payload.get("deferred_pair_missing")
        if not isinstance(deferred, Mapping):
            return []
        result: list[WappiHistoryMessage] = []
        for item in deferred.values():
            if not isinstance(item, Mapping):
                continue
            if str(item.get("profile_id") or "") != profile_id or str(item.get("chat_id") or "") != chat_id:
                continue
            result.append(
                WappiHistoryMessage(
                    profile_id=profile_id,
                    chat_id=chat_id,
                    message_id=str(item.get("message_id") or ""),
                    text=str(item.get("text") or ""),
                    message_type=str(item.get("message_type") or "text"),
                    timestamp=int(item.get("timestamp") or 0),
                    from_me=bool(item.get("from_me")),
                    contact_name=str(item.get("contact_name") or ""),
                    from_where=str(item.get("from_where") or ""),
                )
            )
        return [item for item in result if item.message_id and item.text.strip()]

    def deferred_pair_missing_chat_ids(self, profile_id: str) -> tuple[str, ...]:
        deferred = self.payload.get("deferred_pair_missing")
        if not isinstance(deferred, Mapping):
            return ()
        return tuple(
            sorted(
                {
                    str(item.get("chat_id") or "")
                    for item in deferred.values()
                    if isinstance(item, Mapping)
                    and str(item.get("profile_id") or "") == profile_id
                    and str(item.get("chat_id") or "")
                }
            )
        )

    def defer_pair_missing(self, message: WappiHistoryMessage) -> None:
        deferred = dict(self.payload.get("deferred_pair_missing") or {})
        deferred[_message_state_key(message)] = {
            "profile_id": message.profile_id,
            "chat_id": message.chat_id,
            "message_id": message.message_id,
            "text": message.text,
            "message_type": message.message_type,
            "timestamp": message.timestamp,
            "from_me": message.from_me,
            "contact_name": message.contact_name,
            "from_where": message.from_where,
        }
        self.payload["deferred_pair_missing"] = deferred

    def pending_notes(self) -> dict[str, Mapping[str, Any]]:
        raw = self.payload.get("pending_notes")
        return dict(raw) if isinstance(raw, Mapping) else {}

    def pending_message_keys(self) -> set[tuple[str, str, str]]:
        result: set[tuple[str, str, str]] = set()
        for payload in self.pending_notes().values():
            if not isinstance(payload, Mapping):
                continue
            profile_id = str(payload.get("profile_id") or "")
            chat_id = str(payload.get("chat_id") or "")
            message_ids = payload.get("covered_message_ids") or (payload.get("message_id"),)
            if not profile_id or not chat_id:
                continue
            for message_id in message_ids:
                normalized = str(message_id or "").strip()
                if normalized:
                    result.add((profile_id, chat_id, normalized))
        return result

    def set_pending(self, message: WappiHistoryMessage, payload: Mapping[str, Any]) -> None:
        pending = dict(self.payload.get("pending_notes") or {})
        pending[_message_state_key(message)] = dict(payload)
        self.payload["pending_notes"] = pending

    def clear_pending(self, state_key: str) -> None:
        pending = dict(self.payload.get("pending_notes") or {})
        pending.pop(state_key, None)
        self.payload["pending_notes"] = pending

    def dialogue_memory_for(self, key: DraftLoopKey) -> Mapping[str, Any]:
        raw = self.payload.get("dialogue_memory")
        if not isinstance(raw, Mapping):
            return {}
        item = raw.get(key.value)
        return dict(item) if isinstance(item, Mapping) else {}

    def set_dialogue_memory(self, key: DraftLoopKey, memory: Mapping[str, Any]) -> None:
        data = dict(self.payload.get("dialogue_memory") or {})
        data[key.value] = dict(memory)
        self.payload["dialogue_memory"] = data

    def manager_edit_match_keys(self) -> set[str]:
        result: set[str] = set()
        for item in self.payload.get("manager_edit_matches") or []:
            if isinstance(item, str) and item:
                result.add(item)
        return result

    def mark_manager_edit_match(self, key: str) -> None:
        if not key:
            return
        items = [str(item) for item in (self.payload.get("manager_edit_matches") or []) if str(item)]
        if key not in items:
            items.append(key)
        self.payload["manager_edit_matches"] = items[-5000:]

    def auth_error_count(self) -> int:
        try:
            return int(self.payload.get("auth_error_count") or 0)
        except (TypeError, ValueError):
            return 0

    def set_auth_error_count(self, value: int) -> None:
        self.payload["auth_error_count"] = max(0, int(value))

    def quarantined_pairs(self) -> dict[str, Mapping[str, Any]]:
        raw = self.payload.get("quarantined_pairs")
        return dict(raw) if isinstance(raw, Mapping) else {}

    def is_pair_quarantined(self, key: DraftLoopKey) -> bool:
        return key.value in self.quarantined_pairs()

    def quarantine_pair(self, key: DraftLoopKey, *, reason: str, lead_id: str = "", detail: str = "") -> None:
        rows = dict(self.payload.get("quarantined_pairs") or {})
        rows[key.value] = {
            "profile_id": key.profile_id,
            "chat_id": key.chat_id,
            "lead_id": str(lead_id or ""),
            "reason": str(reason or ""),
            "detail": str(detail or "")[:300],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self.payload["quarantined_pairs"] = rows

    def save(self) -> None:
        if not self.persist:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(self.payload, ensure_ascii=False, indent=2, sort_keys=True)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{self.path.name}.", suffix=".tmp", dir=str(self.path.parent))
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(data)
            handle.write("\n")
        os.replace(tmp_name, self.path)


def _message_state_key(message: WappiHistoryMessage) -> str:
    return f"{message.profile_id}\t{message.chat_id}\t{message.message_id}"


def _is_auth_error_exception(exc: BaseException) -> bool:
    text = str(exc).casefold()
    return any(marker in text for marker in ("401", "403", "unauthorized", "forbidden", "auth"))


def _is_allowlist_desync_exception(exc: BaseException) -> bool:
    text = str(exc).casefold()
    return (
        "not in allowlist" in text
        or "write blocked" in text
        or "draft-note write blocked" in text
        or ("lead_id" in text and ("403" in text or "forbidden" in text))
    )


def _draft_log_priority(row: Mapping[str, Any]) -> int:
    event = str(row.get("event") or "")
    status = str(row.get("status") or "")
    if event in {"note_written", "note_retried"} or status == "note_written":
        return 30
    if status == "note_pending":
        return 20
    if status == "dry_run":
        return 10
    return 0


def _draft_row_ts(row: Mapping[str, Any]) -> int:
    timestamp = row.get("timestamp")
    try:
        value = int(float(timestamp))
        if value > 0:
            return value
    except (TypeError, ValueError):
        pass
    return _parse_iso_epoch(str(row.get("created_at") or ""))


def _parse_iso_epoch(value: str) -> int:
    raw = str(value or "").strip()
    if not raw:
        return 0
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return 0
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.astimezone(timezone.utc).timestamp())


def _iso_from_epoch(value: int) -> str:
    if not value:
        return ""
    return datetime.fromtimestamp(int(value), tz=timezone.utc).isoformat()


def _is_private_dialog(dialog: Mapping[str, Any], *, channel: str) -> bool:
    dialog_type = str(dialog.get("type") or "").casefold()
    if str(channel or "").casefold() == "max":
        if "isGroup" in dialog:
            is_group_raw = dialog.get("isGroup")
        elif "is_group" in dialog:
            is_group_raw = dialog.get("is_group")
        else:
            return False
        is_group = str(is_group_raw).strip().casefold()
        if dialog_type != "dialog" or is_group not in {"0", "false", "no"}:
            return False
        participants = tuple(item for item in (dialog.get("participants") or ()) if isinstance(item, Mapping))
        peers = tuple(item for item in participants if not _dialog_truthy_flag(item, "is_me", "IsMe"))
        return (not participants or len(peers) == 1) and not any(
            _dialog_truthy_flag(item, "is_bot", "IsBot", "bot") for item in (dialog, *peers)
        )
    if str(channel or "").casefold() == "telegram":
        if dialog_type not in {"user", "private", "personal"}:
            return False
        user = dialog.get("user") if isinstance(dialog.get("user"), Mapping) else {}
        return not any(
            _dialog_truthy_flag(user, *keys)
            for keys in (
                ("IsBot", "is_bot"),
                ("IsDeleted", "is_deleted"),
                ("IsFake", "is_fake"),
                ("IsSelf", "is_self"),
                ("IsSupport", "is_support"),
            )
        )
    return False


def _dialog_truthy_flag(payload: Mapping[str, Any], *keys: str) -> bool:
    return any(
        payload.get(key) is True or str(payload.get(key) or "").strip().casefold() in {"1", "true", "yes", "on"}
        for key in keys
    )


def wappi_message_from_raw(profile_id: str, raw: Mapping[str, Any]) -> WappiHistoryMessage | None:
    message_id = str(raw.get("id") or raw.get("message_id") or "").strip()
    chat_id = str(raw.get("chatId") or raw.get("chat_id") or "").strip()
    if not message_id or not chat_id:
        return None
    raw_text = str(raw.get("body") or raw.get("text") or "").strip()
    message_type = str(raw.get("type") or raw.get("message_type") or ("text" if raw_text else "")).strip().casefold()
    caption = str(raw.get("caption") or "").strip()
    text = raw_text if message_type == "text" else caption
    timestamp_raw = raw.get("time") or raw.get("timestamp") or 0
    try:
        timestamp = int(float(timestamp_raw))
    except (TypeError, ValueError, OverflowError):
        timestamp = 0
    while timestamp > 4_102_444_800:
        timestamp = int(timestamp / 1000)
    return WappiHistoryMessage(
        profile_id=str(profile_id),
        chat_id=chat_id,
        message_id=message_id,
        text=text,
        message_type=message_type,
        timestamp=timestamp,
        from_me=bool(raw.get("fromMe") or raw.get("is_me")),
        contact_name=str(raw.get("contact_name") or raw.get("senderName") or "").strip(),
        from_where=str(raw.get("from_where") or "").strip(),
        raw=dict(raw),
    )


class AmoWappiDraftLoop:
    def __init__(
        self,
        *,
        config: DraftLoopConfig,
        wappi_client: WappiPhase1Client,
        amo_client: AmoDraftNoteClient,
        bot_provider: DraftBotProvider,
        context_builder: ContextBuilder,
        journal: DraftLoopJournal | None = None,
        state: DraftLoopState | None = None,
        auto_resolver: AutoResolver | None = None,
        trusted_auto_customer_context_builder: ContextBuilder | None = None,
        now_fn: Callable[[], datetime] | None = None,
        code_identity: Mapping[str, str] | None = None,
    ) -> None:
        self.config = config
        self.wappi_client = wappi_client
        self.amo_client = amo_client
        self.bot_provider = bot_provider
        self.context_builder = context_builder
        self.journal = journal or DraftLoopJournal(config.journal_path)
        self.state = state or DraftLoopState(config.state_path)
        self.auto_resolver = auto_resolver
        self.trusted_auto_customer_context_builder = trusted_auto_customer_context_builder
        self.now_fn = now_fn or (lambda: datetime.now(timezone.utc))
        self.code_identity = dict(code_identity or build_draft_loop_code_identity())

    def run_once(self, *, dry_run: bool = True) -> Mapping[str, Any]:
        if not dry_run:
            return self._run_once(dry_run=False)
        original_state = self.state
        dry_state = DraftLoopState(original_state.path, persist=False)
        dry_state.payload = deepcopy(original_state.payload)
        self.state = dry_state
        try:
            result = self._run_once(dry_run=True)
            operational_state_changed = False
            for key in ("manager_edit_matches", "auth_error_count"):
                if original_state.payload.get(key) != dry_state.payload.get(key):
                    original_state.payload[key] = deepcopy(dry_state.payload.get(key))
                    operational_state_changed = True
            if operational_state_changed:
                original_state.save()
            return result
        finally:
            self.state = original_state

    def _run_once(self, *, dry_run: bool) -> Mapping[str, Any]:
        stop_active = self.config.stop_path.expanduser().exists()
        if self.state.auth_error_count() >= max(1, int(self.config.auth_error_limit)):
            summary = {
                "processed": 0,
                "deferred": 0,
                "deferred_fetch": 0,
                "skipped": 0,
                "bot_calls": 0,
                "retried_pending": 0,
                "stop_active": stop_active,
                "dry_run": dry_run,
                "auth_error": True,
                "auth_error_count": self.state.auth_error_count(),
                "stopped": True,
                "message_outcomes": {},
            }
            self._write_heartbeat("auth_error", summary)
            return summary
        retried = 0 if dry_run or stop_active else self.retry_pending_notes()
        processed_count = 0
        deferred_count = 0
        deferred_fetch_count = 0
        skipped_count = 0
        bot_calls = 0
        manager_edit_count = 0
        auto_resolver_counts: Counter[str] = Counter()
        # Cycle-wide reconciliation of the per-message terminal outcome bookkeeping that
        # `_process_chat_messages` returns per call (see the "message_outcomes" contract note
        # on that method). Keyed the same way as pending/deferred state entries
        # (`_message_state_key`: "profile_id\tchat_id\tmessage_id") so outcomes from different
        # chats/profiles never collide even though message_id alone is not globally unique.
        # This is what lets a caller (e.g. an all-personal replay) prove "every inbound message
        # got exactly one terminal outcome" from the real run_once() entry point instead of
        # re-deriving it from journal rows.
        message_outcomes: dict[str, str] = {}
        now_epoch = int(self.now_fn().timestamp())
        inbound_not_before_ts = (
            self.state.ensure_inbound_not_before_ts(now_epoch)
            if self.config.all_personal_mode
            else 0
        )
        seen_processed = (
            self.state.processed_keys()
            | self.state.pending_message_keys()
            | self.journal.processed_message_keys()
        )
        try:
            for profile in self.config.profiles.values():
                try:
                    dialogs = list(self._iter_dialogs(profile))
                except DraftLoopPaginationChanged as exc:
                    deferred_fetch_count += 1
                    self.journal.append(
                        {
                            "event": "dialogs_deferred",
                            "profile_id": profile.profile_id,
                            "channel": profile.channel,
                            "reason": "wappi_dialog_pagination_changed",
                            "error": str(exc)[:500],
                            "created_at": self.now_fn().astimezone(timezone.utc).isoformat(),
                        }
                    )
                    continue
                listed_chat_ids = {str(item.get("id") or "").strip() for item in dialogs}
                for chat_id in self.state.deferred_pair_missing_chat_ids(profile.profile_id):
                    if chat_id not in listed_chat_ids and self.config.pair_for(DraftLoopKey(profile.profile_id, chat_id)):
                        deferred_dialog = {"id": chat_id, "_deferred_only": True}
                        if profile.channel == "telegram":
                            deferred_dialog["type"] = "user"
                        elif profile.channel == "max":
                            deferred_dialog.update({"type": "DIALOG", "isGroup": False})
                        dialogs.append(deferred_dialog)
                for dialog in dialogs:
                    if not isinstance(dialog, Mapping):
                        continue
                    chat_id = str(dialog.get("id") or "").strip()
                    if not chat_id:
                        continue
                    if not _is_private_dialog(dialog, channel=profile.channel):
                        self.journal.append({"event": "chat_skipped", "profile_id": profile.profile_id, "chat_id": chat_id, "reason": "non_private"})
                        skipped_count += 1
                        continue
                    if bool(dialog.get("_deferred_only")):
                        messages = []
                    else:
                        try:
                            messages = self._fetch_messages(profile, chat_id)
                        except Exception as exc:  # noqa: BLE001
                            if not _is_deferred_fetch_exception(exc):
                                raise
                            deferred_fetch_count += 1
                            self.journal.append(
                                {
                                    "event": "deferred_fetch",
                                    "profile_id": profile.profile_id,
                                    "chat_id": chat_id,
                                    "channel": profile.channel,
                                    "reason": "wappi_fetch_messages_deferred",
                                    "error": str(exc)[:500],
                                    "created_at": self.now_fn().astimezone(timezone.utc).isoformat(),
                                }
                            )
                            continue
                    deferred_messages = self.state.deferred_pair_missing_messages(profile.profile_id, chat_id)
                    if deferred_messages:
                        messages = sorted(
                            {item.key: item for item in [*deferred_messages, *messages]}.values(),
                            key=lambda item: (item.timestamp, item.message_id),
                        )
                    manager_edit_count += self._classify_manager_edits(profile, chat_id, messages)
                    pending_inbound = [
                        item
                        for item in messages
                        if item.is_inbound_actionable and item.key not in seen_processed
                    ]
                    if (
                        inbound_not_before_ts
                        and self.config.pair_for(DraftLoopKey(profile.profile_id, chat_id)) is None
                    ):
                        pending_inbound = [
                            item for item in pending_inbound if item.timestamp >= inbound_not_before_ts
                        ]
                    if not pending_inbound:
                        continue
                    if any(item.timestamp > now_epoch - self.config.debounce_seconds for item in pending_inbound):
                        deferred_count += 1
                        continue
                    inbound_new = pending_inbound
                    if stop_active:
                        for item in inbound_new:
                            self.journal.append(_message_event("stop_raw_inbound", item, status="stop_not_processed"))
                        continue
                    result = self._process_chat_messages(profile, dialog, messages, inbound_new, dry_run=dry_run)
                    processed_count += int(result.get("processed", 0))
                    skipped_count += int(result.get("skipped", 0))
                    bot_calls += int(result.get("bot_calls", 0))
                    auto_reason = str(result.get("auto_resolver_reason") or "")
                    if auto_reason:
                        auto_resolver_counts[auto_reason] += 1
                    call_outcomes = result.get("message_outcomes") or {}
                    for item in inbound_new:
                        outcome = call_outcomes.get(item.message_id)
                        if outcome:
                            message_outcomes[_message_state_key(item)] = outcome
        except Exception as exc:  # noqa: BLE001
            if not _is_auth_error_exception(exc):
                raise
            auth_error_count = self.state.auth_error_count() + 1
            self.state.set_auth_error_count(auth_error_count)
            self.state.save()
            summary = {
                "processed": processed_count,
                "deferred": deferred_count,
                "deferred_fetch": deferred_fetch_count,
                "skipped": skipped_count,
                "bot_calls": bot_calls,
                "retried_pending": retried,
                "stop_active": stop_active,
                "dry_run": dry_run,
                "manager_edits_classified": manager_edit_count,
                "auto_resolver_counts": dict(auto_resolver_counts),
                "auth_error": True,
                "auth_error_count": auth_error_count,
                "stopped": auth_error_count >= max(1, int(self.config.auth_error_limit)),
                "error": str(exc)[:300],
                "message_outcomes": dict(message_outcomes),
            }
            self._write_heartbeat("auth_error", summary)
            return summary
        previous_auth_error_count = self.state.auth_error_count()
        self.state.set_auth_error_count(0)
        summary = {
            "processed": processed_count,
            "deferred": deferred_count,
            "deferred_fetch": deferred_fetch_count,
            "skipped": skipped_count,
            "bot_calls": bot_calls,
            "retried_pending": retried,
            "stop_active": stop_active,
            "dry_run": dry_run,
            "manager_edits_classified": manager_edit_count,
            "auto_resolver_counts": dict(auto_resolver_counts),
            "message_outcomes": dict(message_outcomes),
            "auth_error": False,
            "auth_error_count": 0,
        }
        if not dry_run and not stop_active:
            self.state.save()
        elif manager_edit_count or previous_auth_error_count:
            self.state.save()
        self._write_heartbeat("stop" if stop_active else "ok", summary)
        return summary

    def _iter_dialogs(self, profile: DraftLoopProfile) -> Iterable[Mapping[str, Any]]:
        configured_limit = int(self.config.chat_limit)
        if configured_limit > 0:
            dialogs_payload = self.wappi_client.list_chats(
                channel=profile.channel,
                profile_id=profile.profile_id,
                limit=max(1, min(configured_limit, 100)),
            )
            dialogs = dialogs_payload.get("dialogs") if isinstance(dialogs_payload, Mapping) else []
            if isinstance(dialogs, Sequence) and not isinstance(dialogs, (str, bytes, bytearray)):
                yield from (dialog for dialog in dialogs if isinstance(dialog, Mapping))
            return
        page_limit = 100
        offset = 0
        first_signature: tuple[str, ...] = ()
        previous_anchor = ""
        dialogs_by_id: dict[str, Mapping[str, Any]] = {}
        while True:
            dialogs_payload = self.wappi_client.list_chats(
                channel=profile.channel,
                profile_id=profile.profile_id,
                limit=page_limit,
                offset=offset,
            )
            dialogs = dialogs_payload.get("dialogs") if isinstance(dialogs_payload, Mapping) else []
            if not isinstance(dialogs, Sequence) or isinstance(dialogs, (str, bytes, bytearray)):
                return
            page = [dialog for dialog in dialogs if isinstance(dialog, Mapping)]
            if not page:
                break
            page_ids = tuple(str(dialog.get("id") or "").strip() for dialog in page)
            if not all(page_ids):
                raise DraftLoopPaginationChanged("Wappi dialog page contains a row without id")
            if not first_signature:
                first_signature = page_ids
            if previous_anchor and previous_anchor not in page_ids:
                raise DraftLoopPaginationChanged("Wappi dialog pagination boundary changed; retry the profile next cycle")
            dialogs_by_id.update((dialog_id, dialog) for dialog_id, dialog in zip(page_ids, page))
            if len(page) < page_limit:
                break
            previous_anchor = page_ids[-1]
            offset += page_limit - 1
        if offset:
            head_payload = self.wappi_client.list_chats(
                channel=profile.channel,
                profile_id=profile.profile_id,
                limit=page_limit,
                offset=0,
            )
            head = head_payload.get("dialogs") if isinstance(head_payload, Mapping) else []
            head_signature = tuple(
                str(item.get("id") or "").strip()
                for item in head
                if isinstance(item, Mapping)
            )
            if head_signature != first_signature:
                raise DraftLoopPaginationChanged("Wappi dialog list changed during pagination; retry next cycle")
        yield from dialogs_by_id.values()

    def _fetch_messages(self, profile: DraftLoopProfile, chat_id: str) -> list[WappiHistoryMessage]:
        page_limit = 100
        offset = 0
        raw_messages: list[Mapping[str, Any]] = []
        first_signature: tuple[str, ...] = ()
        previous_anchor = ""
        seen_page_signatures: set[tuple[str, ...]] = set()
        key = DraftLoopKey(profile.profile_id, chat_id)
        pair = self.config.pair_for(key)
        processed = self.state.processed_keys()
        stop_before_ts = int(pair.not_before_ts or 0) if pair is not None else 0
        if pair is None and self.config.all_personal_mode:
            stop_before_ts = self.state.ensure_inbound_not_before_ts(int(self.now_fn().timestamp()))
        while True:
            payload = self.wappi_client.get_chat_messages(
                channel=profile.channel,
                profile_id=profile.profile_id,
                chat_id=chat_id,
                limit=page_limit,
                offset=offset,
                order="desc",
                mark_all=False,
            )
            page = payload.get("messages") if isinstance(payload, Mapping) else []
            if not isinstance(page, Sequence) or isinstance(page, (str, bytes, bytearray)):
                break
            page_rows = [item for item in page if isinstance(item, Mapping)]
            page_ids = tuple(str(item.get("id") or item.get("message_id") or "").strip() for item in page_rows)
            if page_rows and (not all(page_ids) or page_ids in seen_page_signatures):
                raise DraftLoopPaginationChanged("Wappi message pagination is not stable; retry the chat next cycle")
            if previous_anchor and previous_anchor not in page_ids:
                raise DraftLoopPaginationChanged("Wappi message pagination boundary changed; retry the chat next cycle")
            seen_page_signatures.add(page_ids)
            if not first_signature:
                first_signature = page_ids
            raw_messages.extend(page_rows)
            parsed_page = [wappi_message_from_raw(profile.profile_id, item) for item in page_rows]
            parsed_page = [item for item in parsed_page if item is not None]
            reached_known = any(item.key in processed for item in parsed_page)
            reached_start = bool(
                stop_before_ts
                and parsed_page
                and min(item.timestamp for item in parsed_page) <= stop_before_ts
            )
            if len(page_rows) < page_limit or reached_known or reached_start:
                break
            previous_anchor = page_ids[-1]
            offset += page_limit - 1
        if offset:
            head_payload = self.wappi_client.get_chat_messages(
                channel=profile.channel,
                profile_id=profile.profile_id,
                chat_id=chat_id,
                limit=page_limit,
                offset=0,
                order="desc",
                mark_all=False,
            )
            head = head_payload.get("messages") if isinstance(head_payload, Mapping) else []
            head_signature = tuple(
                str(item.get("id") or item.get("message_id") or "").strip()
                for item in head
                if isinstance(item, Mapping)
            )
            if head_signature != first_signature:
                raise DraftLoopPaginationChanged("Wappi message list changed during pagination; retry the chat next cycle")
        messages: list[WappiHistoryMessage] = []
        deduplicated_raw = {
            str(raw.get("id") or raw.get("message_id") or "").strip(): raw for raw in raw_messages
        }
        for raw in deduplicated_raw.values():
            item = wappi_message_from_raw(profile.profile_id, raw)
            if item is not None:
                messages.append(item)
        messages.sort(key=lambda item: (item.timestamp, item.message_id))
        return messages

    def _classify_manager_edits(self, profile: DraftLoopProfile, chat_id: str, messages: Sequence[WappiHistoryMessage]) -> int:
        key = DraftLoopKey(profile.profile_id, chat_id)
        pair = self.config.pair_for(key)
        if pair is None:
            return 0
        draft_by_message: dict[str, Mapping[str, Any]] = {}
        for row in self.journal.rows():
            if str(row.get("profile_id") or "") != profile.profile_id or str(row.get("chat_id") or "") != chat_id:
                continue
            if str(row.get("event") or "") not in {"note_written", "note_retried"} and str(row.get("status") or "") != "note_written":
                continue
            message_id = str(row.get("message_id") or "")
            draft_text = str(row.get("bot_draft_text") or "")
            if not message_id or not draft_text:
                continue
            current = draft_by_message.get(message_id)
            if current is None or _draft_log_priority(row) >= _draft_log_priority(current):
                draft_by_message[message_id] = row
        if not draft_by_message:
            return 0
        drafts = [
            DraftWindow(
                profile_id=profile.profile_id,
                chat_id=chat_id,
                message_id=message_id,
                bot_draft_text=str(row.get("bot_draft_text") or ""),
                draft_ts=_draft_row_ts(row),
                superseded=False,
            )
            for message_id, row in sorted(draft_by_message.items(), key=lambda item: _draft_row_ts(item[1]))
        ]
        outgoing = [
            OutgoingWindowMessage(message_id=item.message_id, text=item.text, sent_ts=item.timestamp)
            for item in messages
            if item.from_me and item.message_type == "text" and item.text.strip()
        ]
        now_ts = int(self.now_fn().timestamp())
        classified = classify_manager_edit_windows(drafts, outgoing, now_ts=now_ts)
        if not classified:
            return 0
        outgoing_by_id = {item.message_id: item for item in outgoing}
        seen = self.state.manager_edit_match_keys()
        written = 0
        for row in classified:
            message_id = str(row.get("message_id") or "")
            match_class = str(row.get("match_class") or "")
            matched_message_id = str(row.get("matched_message_id") or "")
            match_key = f"{profile.profile_id}\t{chat_id}\t{message_id}\t{matched_message_id}\t{match_class}"
            if match_key in seen:
                continue
            draft_row = draft_by_message.get(message_id) or {}
            sent = outgoing_by_id.get(matched_message_id)
            append_manager_edit_log(
                self.config.manager_edit_log_path,
                ManagerEditLogRecord(
                    lead_id=pair.lead_id,
                    brand=pair.expected_brand,
                    profile_id=profile.profile_id,
                    chat_id=chat_id,
                    message_id=message_id,
                    matched_message_id=matched_message_id,
                    draft_route=str(draft_row.get("route") or ""),
                    match_class=match_class,
                    ratio=float(row.get("ratio") or 0.0),
                    draft_ts=_iso_from_epoch(_draft_row_ts(draft_row)),
                    sent_ts=_iso_from_epoch(sent.sent_ts) if sent is not None else "",
                    window_closed=bool(row.get("window_closed")),
                    bot_draft_text=str(draft_row.get("bot_draft_text") or ""),
                    manager_sent_text=sent.text if sent is not None else "",
                    reason_codes=tuple(str(item) for item in (draft_row.get("safety_flags") or ())),
                ),
            )
            self.state.mark_manager_edit_match(match_key)
            seen.add(match_key)
            written += 1
        return written

    def _write_heartbeat(self, status: str, summary: Mapping[str, Any]) -> None:
        target = self.config.heartbeat_path.expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "draft_loop_heartbeat_v1_2026_06_11",
            "last_cycle_at": self.now_fn().astimezone(timezone.utc).isoformat(),
            "status": str(status or ""),
            "auth_error_count": self.state.auth_error_count(),
            "code_root": str(self.code_identity.get("code_root") or "unknown"),
            "git_sha": str(self.code_identity.get("git_sha") or "unknown"),
            "summary": dict(summary),
        }
        data = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(data)
            handle.write("\n")
        os.replace(tmp_name, target)

    def _process_chat_messages(
        self,
        profile: DraftLoopProfile,
        dialog: Mapping[str, Any],
        messages: Sequence[WappiHistoryMessage],
        inbound_new: Sequence[WappiHistoryMessage],
        *,
        dry_run: bool,
    ) -> Mapping[str, Any]:
        # Per-message terminal outcome bookkeeping (audit/metric only; does not change
        # AMO write behaviour). Conversational batching is unchanged: several inbound
        # messages can still resolve to a single draft/note. Every message_id that
        # enters this call via `inbound_new` must end up with exactly one entry here
        # before any `return` -- never zero (lost), never more than one (duplicated).
        # Vocabulary: covered_by_draft, note_written, pair_missing, brand_mismatch,
        # manual_review, write_unknown, unsupported_inbound, plus not_before_skipped
        # for the pre-existing pairing-watermark skip (no content/write reason fits
        # the seven names above without misrepresenting what happened), plus
        # identity_conflict for a contact_id change on an already-paired chat (a
        # widget candidate whose contact disagrees with the persisted pair) -- kept
        # distinct from pair_missing because a conflict is not an absent pair.
        message_outcomes: dict[str, str] = {}
        key = DraftLoopKey(profile.profile_id, inbound_new[-1].chat_id)
        pair = self.config.pair_for(key)
        previous_pair = pair
        auto_pair_skipped = 0
        candidate = (
            self._resolve_auto_candidate(key, profile, dialog, messages, inbound_new[-1])
            if self.auto_resolver is not None
            else None
        )
        if candidate and str(candidate.get("status") or "") == "matched":
            candidate_contact_id = str(candidate.get("contact_id") or "").strip()
            if (
                previous_pair is not None
                and previous_pair.contact_id
                and candidate_contact_id
                and previous_pair.contact_id != candidate_contact_id
            ):
                self.journal.append(
                    {
                        "event": "auto_pair_identity_conflict",
                        "status": "manual_review",
                        "profile_id": key.profile_id,
                        "chat_id": key.chat_id,
                        "previous_contact_id": previous_pair.contact_id,
                        "candidate_contact_id": candidate_contact_id,
                    }
                )
                candidate = {
                    "status": "rejected",
                    "reason": "auto_pair_contact_changed",
                    "contact_id": candidate_contact_id,
                }
                pair = None
            else:
                now_ts = int(self.now_fn().timestamp())
                pair = DraftLoopPair(
                    key=key,
                    lead_id=str(candidate.get("lead_id") or ""),
                    expected_brand=profile.brand,
                    not_before_ts=previous_pair.not_before_ts if previous_pair is not None else now_ts,
                    source=str(candidate.get("source") or "auto").strip() or "auto",
                    match_key=str(candidate.get("match_key") or ""),
                    contact_id=candidate_contact_id,
                    auto_note=_auto_pair_note(profile=profile, candidate=candidate),
                )
            if pair is not None and not dry_run and self.config.auto_pairs_path is not None:
                changed = persist_auto_pair(self.config.auto_pairs_path, pair, replace=True)
                if changed:
                    self.journal.append(
                        {
                            "event": "auto_pair_created" if previous_pair is None else "auto_pair_updated",
                            "status": "created" if previous_pair is None else "updated",
                            "profile_id": key.profile_id,
                            "chat_id": key.chat_id,
                            "lead_id": pair.lead_id,
                            "contact_id": pair.contact_id,
                            "match_key": pair.match_key,
                            "not_before_ts": pair.not_before_ts,
                            "lead_snapshot": dict(candidate.get("lead_snapshot") or {}),
                        }
                    )
                for item in inbound_new:
                    if not self.state.pair_missing_is_deferred(item):
                        self.state.defer_pair_missing(item)
                self.state.save()
        elif self.auto_resolver is not None:
            pair = None
        if pair is None:
            for item in inbound_new:
                if self.state.pair_missing_is_deferred(item):
                    continue
                self.journal.append(
                    {
                        **_message_event("pair_missing", item, status="skipped"),
                        "auto_candidate": dict(candidate or {}),
                    }
                )
                if not dry_run:
                    self.state.defer_pair_missing(item)
            if inbound_new and not dry_run:
                self.state.save()
            reason = str((candidate or {}).get("reason") or (candidate or {}).get("status") or "not_enabled")
            # A contact_id change (auto_pair_contact_changed, set above when the widget
            # candidate's contact disagrees with the persisted pair) is an identity
            # conflict, not an ordinary "no pair configured yet" -- collapsing both into
            # "pair_missing" hides the real cause from anything reading message_outcomes.
            outcome = "identity_conflict" if reason == "auto_pair_contact_changed" else "pair_missing"
            for item in inbound_new:
                message_outcomes[item.message_id] = outcome
            if dry_run:
                return {
                    "processed": len(inbound_new),
                    "skipped": len(inbound_new),
                    "bot_calls": 0,
                    "auto_resolver_reason": reason,
                    "message_outcomes": message_outcomes,
                }
            return {
                "processed": 0,
                "skipped": len(inbound_new),
                "bot_calls": 0,
                "auto_resolver_reason": reason,
                "message_outcomes": message_outcomes,
            }
        if self.state.is_pair_quarantined(key):
            for item in inbound_new:
                if self.state.pair_missing_is_deferred(item):
                    continue
                self.journal.append(
                    {
                        **_message_event("pair_quarantined", item, status="skipped"),
                        "lead_id": pair.lead_id,
                        "quarantine": dict(self.state.quarantined_pairs().get(key.value) or {}),
                    }
                )
                if not dry_run:
                    self.state.defer_pair_missing(item)
            if not dry_run:
                self.state.save()
            for item in inbound_new:
                message_outcomes[item.message_id] = "manual_review"
            return {"processed": 0, "skipped": len(inbound_new), "bot_calls": 0, "message_outcomes": message_outcomes}
        too_old = [
            item
            for item in inbound_new
            if (0 < item.timestamp <= pair.not_before_ts)
            and not self.state.pair_missing_is_deferred(item)
        ]
        skipped_before = auto_pair_skipped + len(too_old)
        if too_old:
            for item in too_old:
                self.journal.append(
                    {
                        **_message_event("not_before_skipped", item, status="skipped"),
                        "lead_id": pair.lead_id,
                        "not_before_ts": pair.not_before_ts,
                    }
                )
                if not dry_run:
                    self.state.mark_processed(item)
                message_outcomes[item.message_id] = "not_before_skipped"
            inbound_new = [item for item in inbound_new if item not in too_old]
            if not inbound_new:
                if not dry_run:
                    self.state.save()
                return {"processed": 0, "skipped": len(too_old), "bot_calls": 0, "message_outcomes": message_outcomes}
        brand = self.config.brand_for_profile(profile.profile_id)
        if brand != pair.expected_brand:
            self.journal.append(
                {
                    **_message_event("brand_pair_mismatch", inbound_new[-1], status="skipped"),
                    "expected_brand": pair.expected_brand,
                    "actual_brand": brand,
                    "lead_id": pair.lead_id,
                }
            )
            for item in inbound_new:
                message_outcomes[item.message_id] = "brand_mismatch"
            if dry_run:
                return {
                    "processed": len(inbound_new),
                    "skipped": len(inbound_new),
                    "bot_calls": 0,
                    "message_outcomes": message_outcomes,
                }
            return {
                "processed": 0,
                "skipped": len(inbound_new),
                "bot_calls": 0,
                "message_outcomes": message_outcomes,
            }
        manual_review_messages = [
            item for item in inbound_new if not item.is_inbound_text or item.timestamp <= 0
        ]
        manual_review_required = bool(manual_review_messages)
        manual_review_message_ids = {item.message_id for item in manual_review_messages}
        client_message = (
            "Входящее сообщение без доступного текста требует ручной проверки."
            if manual_review_required
            else inbound_new[-1].text
        )
        auto_unconfirmed = str(pair.source or "").strip().casefold() == "auto"
        auto_context_trusted = self.context_builder is self.trusted_auto_customer_context_builder
        previous_memory = (
            {}
            if auto_unconfirmed and not auto_context_trusted
            else self.state.dialogue_memory_for(key)
        )
        history = _prompt_history_lines(
            messages,
            recent_limit=self.config.history_limit,
            brand=brand,
            dialogue_memory=previous_memory,
        )
        context = self._build_context(
            key,
            history,
            client_message,
            brand,
            channel=profile.channel,
            dialogue_memory=previous_memory,
            current_message_id=inbound_new[-1].message_id,
        )
        if auto_unconfirmed and not (
            auto_context_trusted and _confirmed_auto_customer_context(context)
        ):
            context = _without_unconfirmed_auto_customer_memory(context)
            memory_status = "unavailable_auto_unconfirmed"
        elif isinstance(context.get("read_only_customer_context"), Mapping):
            memory_status = "connected"
        else:
            memory_status = "unavailable"
        result = (
            SubscriptionDraftResult(
                message_type="manager_only",
                route="manager_only",
                draft_text=_unsupported_inbound_manager_note(manual_review_messages),
                safety_flags=("manager_approval_required", "no_auto_send", "unsupported_inbound_requires_review"),
                provider="deterministic_manual_control",
            )
            if manual_review_required
            else self.bot_provider.build_draft(client_message, context=context)
        )
        bot_call_count = 0 if manual_review_required else 1
        route = str(getattr(result, "route", "") or "")
        safety_flags = tuple(str(item) for item in (getattr(result, "safety_flags", ()) or ()))
        draft_text = str(getattr(result, "draft_text", "") or "")
        if not dry_run and not manual_review_required and _memory_provenance_enabled():
            memory_source = (
                context.get("dialogue_memory_state")
                if isinstance(context.get("dialogue_memory_state"), Mapping)
                else context.get("dialogue_memory_view")
            )
            semantic_reading = SemanticReading.from_result(result)
            updated_memory = update_dialogue_memory_after_answer(
                memory_source if isinstance(memory_source, Mapping) else {},
                answer_text=draft_text,
                route=route,
                fact_refs=tuple(getattr(result, "context_used", ()) or ()),
                safety_flags=safety_flags,
                semantic_reading=semantic_reading.to_memory_dict() if semantic_reading is not None else None,
                semantic_frame=_semantic_frame_from_result(result),
                dialog_summary=_dialog_summary_from_result(result),
                memory_llm_fn=None,
                context=context,
            )
            self.state.set_dialogue_memory(key, updated_memory.to_json_dict())
        last_message = inbound_new[-1]
        pending_payload = {
            "profile_id": last_message.profile_id,
            "chat_id": last_message.chat_id,
            "message_id": last_message.message_id,
            "covered_message_ids": [item.message_id for item in inbound_new],
            "lead_id": pair.lead_id,
            "contact_id": pair.contact_id,
            "note_entity_type": "lead" if pair.lead_id else "contact",
            "note_entity_id": pair.lead_id or pair.contact_id,
            "brand": brand,
            "route": route,
            "safety_flags": list(safety_flags),
            "bot_draft_text": draft_text,
            "auto_note": pair.auto_note,
            "memory_status": memory_status,
            "config_fingerprint": dict(self.config.config_fingerprint or {}),
            "status": "note_pending",
        }
        self.journal.append({**pending_payload, "event": "draft_created", "status": "dry_run" if dry_run else "note_pending"})
        if dry_run:
            for item in inbound_new:
                message_outcomes[item.message_id] = (
                    "unsupported_inbound" if item.message_id in manual_review_message_ids else "covered_by_draft"
                )
            return {
                "processed": len(inbound_new),
                "skipped": skipped_before,
                "bot_calls": bot_call_count,
                "message_outcomes": message_outcomes,
            }
        pending_payload = {
            **pending_payload,
            "write_started_at": self.now_fn().astimezone(timezone.utc).isoformat(),
        }
        self.state.set_pending(last_message, pending_payload)
        self.state.save()
        try:
            note_response = self._write_note(pending_payload, retry=False)
        except Exception as exc:  # noqa: BLE001
            if _is_allowlist_desync_exception(exc):
                self.state.quarantine_pair(key, reason="allowlist_desync", lead_id=pair.lead_id, detail=str(exc))
                failed_status = "manual_review"
                event = "allowlist_desync"
            else:
                failed_status = "write_outcome_unknown"
                event = "note_write_outcome_unknown"
            self.state.payload["pending_notes"][_message_state_key(last_message)] = {
                **pending_payload,
                "status": failed_status,
                "error": str(exc)[:300],
            }
            self.state.save()
            self.journal.append(
                {
                    **pending_payload,
                    "event": event,
                    "status": "quarantined" if failed_status == "manual_review" else failed_status,
                    "error": str(exc)[:300],
                }
            )
            if failed_status != "manual_review" and _is_auth_error_exception(exc):
                raise
            write_outcome = "manual_review" if failed_status == "manual_review" else "write_unknown"
            for item in inbound_new:
                message_outcomes[item.message_id] = write_outcome
            return {
                "processed": 0,
                "skipped": len(inbound_new) + skipped_before,
                "bot_calls": bot_call_count,
                "message_outcomes": message_outcomes,
            }
        self.state.clear_pending(_message_state_key(last_message))
        for item in inbound_new:
            self.state.mark_processed(item)
        self.journal.append(
            {
                **pending_payload,
                "event": "note_written",
                "status": "note_written",
                "note_id": note_response.get("note_id"),
                "deduplicated": bool(note_response.get("deduplicated")),
            }
        )
        for item in inbound_new:
            message_outcomes[item.message_id] = (
                "unsupported_inbound" if item.message_id in manual_review_message_ids else "note_written"
            )
        return {
            "processed": len(inbound_new),
            "skipped": skipped_before,
            "bot_calls": bot_call_count,
            "message_outcomes": message_outcomes,
        }

    def _build_context(
        self,
        key: DraftLoopKey,
        history: Sequence[str],
        client_message: str,
        brand: str,
        *,
        channel: str = "telegram",
        dialogue_memory: Mapping[str, Any],
        current_message_id: str,
    ) -> Mapping[str, Any]:
        try:
            return self.context_builder(
                key,
                history,
                client_message,
                brand,
                channel=channel,
                dialogue_memory=dialogue_memory,
                current_message_id=current_message_id,
            )
        except TypeError:
            try:
                return self.context_builder(
                    key,
                    history,
                    client_message,
                    brand,
                    dialogue_memory=dialogue_memory,
                    current_message_id=current_message_id,
                )
            except TypeError:
                return self.context_builder(key, history, client_message, brand)

    def _resolve_auto_candidate(
        self,
        key: DraftLoopKey,
        profile: DraftLoopProfile,
        dialog: Mapping[str, Any],
        messages: Sequence[WappiHistoryMessage],
        message: WappiHistoryMessage,
    ) -> Mapping[str, Any] | None:
        if self.auto_resolver is None:
            return None
        try:
            candidate = self.auto_resolver(key=key, profile=profile, dialog=dialog, messages=messages, message=message)
        except TypeError:
            try:
                candidate = self.auto_resolver(key, message)
            except Exception as exc:  # noqa: BLE001
                if _is_auth_error_exception(exc):
                    raise
                return {"status": "rejected", "reason": "auto_resolver_unavailable", "error": str(exc)[:300]}
        except Exception as exc:  # noqa: BLE001
            if _is_auth_error_exception(exc):
                raise
            return {"status": "rejected", "reason": "auto_resolver_unavailable", "error": str(exc)[:300]}
        return dict(candidate) if isinstance(candidate, Mapping) else None

    def retry_pending_notes(self) -> int:
        retries = 0
        state_changed = False
        for state_key, payload in list(self.state.pending_notes().items()):
            if not isinstance(payload, Mapping):
                self.state.clear_pending(state_key)
                state_changed = True
                continue
            if str(payload.get("status") or "") == "manual_review":
                continue
            try:
                note_response = self._readback_pending_note(payload)
            except Exception as exc:  # noqa: BLE001
                if _is_allowlist_desync_exception(exc):
                    key = DraftLoopKey(str(payload.get("profile_id") or ""), str(payload.get("chat_id") or ""))
                    self.state.quarantine_pair(
                        key,
                        reason="allowlist_desync",
                        lead_id=str(payload.get("lead_id") or ""),
                        detail=str(exc),
                    )
                    self.state.payload["pending_notes"][state_key] = {
                        **dict(payload),
                        "status": "manual_review",
                        "error": str(exc)[:300],
                    }
                    self.journal.append(
                        {
                            **dict(payload),
                            "event": "allowlist_desync",
                            "status": "quarantined",
                            "error": str(exc)[:300],
                        }
                    )
                    state_changed = True
                    continue
                note_response = {"status": "readback_error", "note_id": None, "error": str(exc)[:300]}

            note_id = note_response.get("note_id")
            retry_event = "note_confirmed_by_readback"
            if not note_id:
                started_at = _parse_iso_epoch(str(payload.get("write_started_at") or ""))
                age_seconds = max(0, int(self.now_fn().timestamp()) - started_at) if started_at else 900
                if not note_id:
                    status = "manual_review" if age_seconds >= 900 else "write_outcome_unknown"
                    self.state.payload["pending_notes"][state_key] = {
                        **dict(payload),
                        "status": status,
                        "readback_attempts": int(payload.get("readback_attempts") or 0) + 1,
                        "error": str(note_response.get("error") or "")[:300],
                    }
                    self.journal.append(
                        {
                            **dict(payload),
                            "event": "note_readback_missing",
                            "status": status,
                            "error": str(note_response.get("error") or "")[:300],
                        }
                    )
                    state_changed = True
                    continue

            self.state.clear_pending(state_key)
            for message_id in payload.get("covered_message_ids") or (payload.get("message_id"),):
                if str(message_id or "").strip():
                    self.state.mark_processed_key(
                        str(payload.get("profile_id") or ""),
                        str(payload.get("chat_id") or ""),
                        str(message_id),
                    )
            self.journal.append(
                {
                    **dict(payload),
                    "event": retry_event,
                    "status": "note_written",
                    "note_id": note_id,
                    "deduplicated": bool(note_response.get("deduplicated", retry_event == "note_confirmed_by_readback")),
                }
            )
            retries += 1
            state_changed = True
        if state_changed:
            self.state.save()
        return retries

    def build_retro_report(self, *, lookback_hours: int = 48, limit: int = 30) -> Mapping[str, Any]:
        rows: list[dict[str, Any]] = []
        bot_calls = 0
        now_ts = int(self.now_fn().timestamp())
        max_rows = max(1, int(limit))
        lookback_seconds = max(1, int(lookback_hours)) * 3600
        for pair in self.config.pairs_snapshot().values():
            profile = self.config.profiles.get(pair.key.profile_id)
            if profile is None:
                continue
            watermark = int(pair.not_before_ts or now_ts)
            start_ts = max(0, watermark - lookback_seconds)
            messages = self._fetch_messages(profile, pair.key.chat_id)
            inbound = [
                item
                for item in messages
                if item.is_inbound_text and start_ts <= item.timestamp < watermark
            ]
            for item in inbound:
                history_before = [candidate for candidate in messages if candidate.timestamp <= item.timestamp]
                next_outgoing = next(
                    (
                        candidate
                        for candidate in messages
                        if candidate.from_me
                        and candidate.message_type == "text"
                        and candidate.text.strip()
                        and candidate.timestamp >= item.timestamp
                    ),
                    None,
                )
                brand = self.config.brand_for_profile(pair.key.profile_id)
                previous_memory = self.state.dialogue_memory_for(pair.key)
                context = self._build_context(
                    pair.key,
                    _prompt_history_lines(history_before, recent_limit=self.config.history_limit, brand=brand),
                    item.text,
                    brand,
                    channel=profile.channel,
                    dialogue_memory=previous_memory,
                    current_message_id=item.message_id,
                )
                result = self.bot_provider.build_draft(item.text, context=context)
                bot_calls += 1
                rows.append(
                    {
                        "profile_id": pair.key.profile_id,
                        "chat_id": pair.key.chat_id,
                        "lead_id": pair.lead_id,
                        "brand": brand,
                        "message_id": item.message_id,
                        "timestamp": item.timestamp,
                        "client_text": item.text,
                        "bot_route": str(getattr(result, "route", "") or ""),
                        "bot_draft_text": str(getattr(result, "draft_text", "") or ""),
                        "employee_message_id": next_outgoing.message_id if next_outgoing else "",
                        "employee_text": next_outgoing.text if next_outgoing else "",
                        "employee_timestamp": next_outgoing.timestamp if next_outgoing else 0,
                    }
                )
                if len(rows) >= max_rows:
                    return {
                        "schema_version": "draft_loop_retro_compare_v1_2026_06_12",
                        "lookback_hours": int(lookback_hours),
                        "limit": max_rows,
                        "rows": rows,
                        "summary": {"rows": len(rows), "bot_calls": bot_calls},
                    }
        return {
            "schema_version": "draft_loop_retro_compare_v1_2026_06_12",
            "lookback_hours": int(lookback_hours),
            "limit": max_rows,
            "rows": rows,
            "summary": {"rows": len(rows), "bot_calls": bot_calls},
        }

    def _write_note(self, payload: Mapping[str, Any], *, retry: bool) -> Mapping[str, Any]:
        del retry
        outgoing_note = ""
        if self.config.manager_outgoing_visible is False:
            outgoing_note = "Важно: бот не видит ответы менеджера в Wappi-истории."
        auto_note = str(payload.get("auto_note") or "").strip()
        if auto_note:
            outgoing_note = "\n".join(item for item in (auto_note, outgoing_note) if item)
        memory_note = _manager_memory_status_note(str(payload.get("memory_status") or ""))
        if memory_note:
            outgoing_note = "\n".join(item for item in (outgoing_note, memory_note) if item)
        entity_type = str(payload.get("note_entity_type") or ("lead" if payload.get("lead_id") else "contact"))
        entity_id = str(payload.get("note_entity_id") or payload.get("lead_id") or payload.get("contact_id") or "")
        writer = (
            self.amo_client.add_draft_note_to_test_lead
            if entity_type == "lead"
            else self.amo_client.add_draft_note_to_contact
        )
        return writer(
            entity_id,
            config=self.config.phase1_config(),
            draft_text=str(payload.get("bot_draft_text") or ""),
            brand=str(payload.get("brand") or ""),
            profile_id=str(payload.get("profile_id") or ""),
            chat_id=str(payload.get("chat_id") or ""),
            message_id=str(payload.get("message_id") or ""),
            route=str(payload.get("route") or ""),
            safety_flags=tuple(payload.get("safety_flags") or ()),
            outgoing_visibility_note=outgoing_note,
        )

    def _readback_pending_note(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        reader = getattr(self.amo_client, "find_existing_draft_note", None)
        if not callable(reader):
            return {"status": "unavailable", "note_id": None}
        entity_type = str(payload.get("note_entity_type") or ("lead" if payload.get("lead_id") else "contact"))
        entity_id = str(payload.get("note_entity_id") or payload.get("lead_id") or payload.get("contact_id") or "")
        return reader(
            entity_type=entity_type,
            entity_id=entity_id,
            profile_id=str(payload.get("profile_id") or ""),
            chat_id=str(payload.get("chat_id") or ""),
            message_id=str(payload.get("message_id") or ""),
        )


def _manager_memory_status_note(status: str) -> str:
    normalized = str(status or "").strip().casefold()
    if normalized == "connected":
        return "Память клиента: подключена."
    if normalized == "unavailable_auto_unconfirmed":
        return "Память клиента: недоступна — автоматическая привязка не подтверждена."
    if normalized == "unavailable":
        return "Память клиента: недоступна."
    return ""


def _unsupported_inbound_manager_note(messages: Sequence[WappiHistoryMessage]) -> str:
    kinds = ", ".join(dict.fromkeys(str(item.message_type or "unknown") for item in messages)) or "unknown"
    return (
        "Ручная проверка: в новом входящем пакете Wappi есть сообщение без доступного текста "
        f"(тип: {kinds}). Откройте диалог, прослушайте или просмотрите вложение и ответьте вручную."
    )


def _without_unconfirmed_auto_customer_memory(context: Mapping[str, Any]) -> Mapping[str, Any]:
    cleaned = {
        str(key): value
        for key, value in context.items()
        if str(key) not in _UNCONFIRMED_AUTO_CUSTOMER_CONTEXT_KEYS
    }
    for memory_key in ("dialogue_memory_view", "dialogue_memory_state"):
        memory = cleaned.get(memory_key)
        if not isinstance(memory, Mapping):
            continue
        cleaned[memory_key] = {
            str(key): value
            for key, value in memory.items()
            if str(key) not in _UNCONFIRMED_AUTO_MEMORY_KEYS
        }
    return cleaned


def _confirmed_auto_customer_context(context: Mapping[str, Any]) -> bool:
    customer_context = context.get("read_only_customer_context")
    if not isinstance(customer_context, Mapping):
        return False
    timeline_context = customer_context.get("timeline_context")
    if not isinstance(timeline_context, Mapping):
        return False
    safety = timeline_context.get("safety")
    return bool(
        customer_context.get("schema_version") == "bot_safe_crm_context_v1_2026_06_21"
        and customer_context.get("source") == "customer_timeline_bot_context"
        and customer_context.get("found") is True
        and customer_context.get("allowed_only") is True
        and timeline_context.get("schema_version") == customer_context.get("schema_version")
        and timeline_context.get("source") == customer_context.get("source")
        and timeline_context.get("found") is True
        and timeline_context.get("allowed_only") is True
        and isinstance(timeline_context.get("bot_context"), Mapping)
        and timeline_context["bot_context"].get("allowed_only") is True
        and isinstance(safety, Mapping)
        and safety.get("source_api") == "bot_context"
        and safety.get("customer_profile_included") is False
        and safety.get("raw_timeline_events_included") is False
        and safety.get("raw_ids_included") is False
        and safety.get("pii_scan_passed") is True
    )


def _message_event(event: str, message: WappiHistoryMessage, *, status: str) -> dict[str, Any]:
    return {
        "event": event,
        "status": status,
        "profile_id": message.profile_id,
        "chat_id": message.chat_id,
        "message_id": message.message_id,
        "message_type": message.message_type,
        "timestamp": message.timestamp,
        "from_me": message.from_me,
    }


def _auto_pair_note(*, profile: DraftLoopProfile, candidate: Mapping[str, Any]) -> str:
    match_key = str(candidate.get("match_key") or "auto").strip()
    brand = "Фотон" if profile.brand == "foton" else "УНПК"
    channel = "Telegram" if profile.channel == "telegram" else "Max"
    if str(candidate.get("source") or "").strip() == "wappi_amo_widget":
        return f"Привязка подтверждена виджетом Wappi→AMO ({match_key}). Канал: {brand}, {channel}."
    return (
        f"Автоматическая привязка не подтверждена ({match_key}). Канал: {brand}, {channel}. "
        "Проверьте карточку и учебный центр перед использованием черновика."
    )


def _raw_history_limit(value: int | None = None) -> int:
    try:
        candidate = int(value or DEFAULT_HISTORY_LIMIT)
    except (TypeError, ValueError):
        candidate = DEFAULT_HISTORY_LIMIT
    return max(MIN_RAW_HISTORY_LIMIT, min(DEFAULT_HISTORY_LIMIT, candidate))


def _clean_prompt_history_text(text: str, *, max_chars: int = 240) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    if not value or _SERVICE_HISTORY_MARKER_RE.search(value):
        return ""
    if len(value) <= max_chars:
        return value
    return value[: max(0, max_chars - 1)].rstrip() + "…"


def _history_line(item: WappiHistoryMessage) -> str:
    if item.message_type not in {"text", "image", "photo", "video", "document"} or not item.text.strip():
        return ""
    text = _clean_prompt_history_text(item.text)
    if not text:
        return ""
    prefix = "Ответ" if item.from_me else "Клиент"
    return f"{prefix}: {text}"


def _mentions_other_brand(text: str, brand: str) -> bool:
    return foreign_brand_rule(text, active_brand=brand, facts={}) is not None


def _older_dialogue_summary(messages: Sequence[WappiHistoryMessage], *, brand: str = "") -> str:
    parts: list[str] = []
    for item in messages:
        line = _history_line(item)
        if not line or _mentions_other_brand(line, brand):
            continue
        parts.append(line)
        if len(parts) >= 6:
            break
    if not parts:
        return ""
    summary = f"{OLDER_DIALOGUE_SUMMARY_PREFIX} " + "; ".join(parts)
    return summary[:700].rstrip()


def _history_lines(messages: Sequence[WappiHistoryMessage], *, limit: int | None = None) -> tuple[str, ...]:
    lines = [line for item in messages if (line := _history_line(item))]
    return tuple(lines[-_raw_history_limit(limit):])


def _dialog_summary_from_result(result: object) -> str:
    metadata = getattr(result, "metadata", None)
    if not isinstance(metadata, Mapping):
        return ""
    candidate = str(metadata.get("dialog_summary_candidate") or "").strip()
    if candidate:
        return candidate
    direct = metadata.get("direct_path")
    if isinstance(direct, Mapping):
        return str(direct.get("dialog_summary_candidate") or "").strip()
    return ""


def _semantic_frame_from_result(result: object) -> Mapping[str, Any] | None:
    metadata = getattr(result, "metadata", None)
    if not isinstance(metadata, Mapping):
        return None
    frame = metadata.get("semantic_frame")
    if not isinstance(frame, Mapping):
        frame = metadata.get("semantic_frame_shadow")
    if not isinstance(frame, Mapping):
        direct = metadata.get("direct_path")
        if isinstance(direct, Mapping):
            frame = direct.get("semantic_frame") if isinstance(direct.get("semantic_frame"), Mapping) else direct.get("semantic_frame_shadow")
    return dict(frame) if isinstance(frame, Mapping) and frame else None


def _prompt_history_lines(
    messages: Sequence[WappiHistoryMessage],
    *,
    recent_limit: int | None = None,
    brand: str = "",
    dialogue_memory: Mapping[str, object] | None = None,
) -> tuple[str, ...]:
    text_messages = [
        item
        for item in messages
        if item.message_type in {"text", "image", "photo", "video", "document"} and item.text.strip()
    ]
    raw_limit = _raw_history_limit(recent_limit)
    recent = text_messages[-raw_limit:]
    older = text_messages[:-raw_limit]
    lines: list[str] = []
    summary = ""
    if _dialog_summary_rolling_enabled(None) and isinstance(dialogue_memory, Mapping):
        summary = _dialog_summary_candidate(dialogue_memory.get("conversation_summary_short"), active_brand=brand)
        if summary:
            if _mentions_other_brand(summary, brand):
                summary = ""
            else:
                summary = f"{OLDER_DIALOGUE_SUMMARY_PREFIX} {summary}"[:700].rstrip()
    if not summary:
        summary = _older_dialogue_summary(older, brand=brand)
    if summary:
        lines.append(summary)
    lines.extend(_history_lines(recent, limit=raw_limit))
    return tuple(lines)


_EMOJI_RE = re.compile(r"[\U00010000-\U0010ffff]", re.UNICODE)
_PUNCT_RE = re.compile(r"[\s\W_]+", re.UNICODE)


def normalize_manager_text(text: str) -> str:
    cleaned = _EMOJI_RE.sub("", str(text or "").casefold().replace("ё", "е"))
    return _PUNCT_RE.sub(" ", cleaned).strip()


def manager_text_ratio(left: str, right: str) -> float:
    a = normalize_manager_text(left)
    b = normalize_manager_text(right)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


@dataclass(frozen=True)
class DraftWindow:
    profile_id: str
    chat_id: str
    message_id: str
    bot_draft_text: str
    draft_ts: int
    superseded: bool = False


@dataclass(frozen=True)
class OutgoingWindowMessage:
    message_id: str
    text: str
    sent_ts: int


def classify_manager_edit_windows(
    drafts: Sequence[DraftWindow],
    outgoing: Sequence[OutgoingWindowMessage],
    *,
    now_ts: int,
    window_seconds: int | None = None,
) -> list[dict[str, Any]]:
    assigned_outgoing: set[str] = set()
    best_by_draft: dict[str, tuple[float, OutgoingWindowMessage]] = {}
    for sent in outgoing:
        candidates: list[tuple[float, DraftWindow]] = []
        for draft in drafts:
            deadline_ts = _manager_edit_deadline_ts(draft.draft_ts, window_seconds=window_seconds)
            if sent.sent_ts < draft.draft_ts or sent.sent_ts > deadline_ts:
                continue
            ratio = manager_text_ratio(draft.bot_draft_text, sent.text)
            candidates.append((ratio, draft))
        if not candidates:
            continue
        candidates.sort(key=lambda item: (item[0], item[1].draft_ts), reverse=True)
        ratio, draft = candidates[0]
        if sent.message_id in assigned_outgoing:
            continue
        assigned_outgoing.add(sent.message_id)
        key = draft.message_id
        current = best_by_draft.get(key)
        if current is None or ratio > current[0]:
            best_by_draft[key] = (ratio, sent)
    result: list[dict[str, Any]] = []
    for draft in drafts:
        match = best_by_draft.get(draft.message_id)
        if match is not None:
            ratio, sent = match
            match_class = "unedited" if ratio >= 0.95 else "edited" if ratio >= 0.5 else "replaced"
            result.append(
                {
                    "message_id": draft.message_id,
                    "matched_message_id": sent.message_id,
                    "ratio": ratio,
                    "match_class": match_class,
                    "window_closed": sent.sent_ts >= draft.draft_ts,
                }
            )
            continue
        deadline_ts = _manager_edit_deadline_ts(draft.draft_ts, window_seconds=window_seconds)
        closed = now_ts >= deadline_ts
        if not closed:
            continue
        had_outgoing = any(draft.draft_ts <= sent.sent_ts <= deadline_ts for sent in outgoing)
        if draft.superseded:
            match_class = "superseded"
        elif had_outgoing:
            match_class = "replaced"
        else:
            match_class = "no_reply"
        result.append(
            {
                "message_id": draft.message_id,
                "matched_message_id": "",
                "ratio": 0.0,
                "match_class": match_class,
                "window_closed": True,
            }
        )
    return result


def _manager_edit_deadline_ts(draft_ts: int, *, window_seconds: int | None) -> int:
    if window_seconds is not None:
        return int(draft_ts) + int(window_seconds)
    draft_dt = datetime.fromtimestamp(int(draft_ts), tz=timezone.utc).astimezone(MOSCOW_TZ)
    deadline_date = draft_dt.date() + timedelta(days=1)
    while deadline_date.weekday() >= 5:
        deadline_date += timedelta(days=1)
    deadline = datetime.combine(deadline_date, time(23, 59, 59), tzinfo=MOSCOW_TZ)
    return int(deadline.astimezone(timezone.utc).timestamp())


def load_profiles_file(path: Path | str) -> dict[str, DraftLoopProfile]:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes, bytearray)):
        raise DraftLoopConfigError("amo_wappi_profiles.json must be a list.")
    result: dict[str, DraftLoopProfile] = {}
    for index, row in enumerate(payload):
        if not isinstance(row, Mapping):
            raise DraftLoopConfigError(f"amo_wappi_profiles.json row {index} must be an object.")
        profile_id = str(row.get("profile_id") or "").strip()
        brand = str(row.get("brand") or "").strip().casefold()
        channel = str(row.get("channel") or "").strip().casefold()
        if not profile_id or brand not in {"foton", "unpk"} or channel not in {"telegram", "max"}:
            raise DraftLoopConfigError(f"amo_wappi_profiles.json row {index} is invalid.")
        if profile_id in result:
            raise DraftLoopConfigError(f"Duplicate Wappi profile_id: {profile_id!r}.")
        result[profile_id] = DraftLoopProfile(profile_id=profile_id, brand=brand, channel=channel)
    if not result:
        raise DraftLoopConfigError("amo_wappi_profiles.json must contain at least one valid profile.")
    return result


def load_pairs_file(path: Path | str, *, default_source: str = "manual") -> dict[DraftLoopKey, DraftLoopPair]:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    rows: Iterable[Any]
    if isinstance(payload, Mapping) and isinstance(payload.get("pairs"), Sequence):
        rows = payload["pairs"]
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        rows = payload
    else:
        raise DraftLoopConfigError("draft_loop_pairs.json must be a list or {'pairs': [...]} object.")
    result: dict[DraftLoopKey, DraftLoopPair] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        if not row.get("profile_id"):
            raise DraftLoopConfigError("draft_loop_pairs entries must include profile_id; bare chat_id is forbidden.")
        key = DraftLoopKey(str(row.get("profile_id") or ""), str(row.get("chat_id") or ""))
        pair = DraftLoopPair(
            key=key,
            lead_id=str(row.get("lead_id") or "").strip(),
            expected_brand=str(row.get("expected_brand") or "").strip().casefold(),
            not_before_ts=_safe_int(row.get("not_before_ts")),
            source=str(row.get("source") or default_source or "manual").strip() or "manual",
            match_key=str(row.get("match_key") or "").strip(),
            contact_id=str(row.get("contact_id") or "").strip(),
            auto_note=str(row.get("auto_note") or "").strip(),
        )
        if not (pair.lead_id or pair.contact_id) or pair.expected_brand not in {"foton", "unpk"}:
            raise DraftLoopConfigError("draft_loop_pairs entries require lead_id or contact_id and expected_brand.")
        result[key] = pair
    return result


def _safe_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _pair_to_json(pair: DraftLoopPair) -> dict[str, Any]:
    row: dict[str, Any] = {
        "profile_id": pair.key.profile_id,
        "chat_id": pair.key.chat_id,
        "lead_id": str(pair.lead_id),
        "expected_brand": str(pair.expected_brand),
    }
    if pair.not_before_ts:
        row["not_before_ts"] = int(pair.not_before_ts)
    if pair.source:
        row["source"] = pair.source
    if pair.match_key:
        row["match_key"] = pair.match_key
    if pair.contact_id:
        row["contact_id"] = pair.contact_id
    if pair.auto_note:
        row["auto_note"] = pair.auto_note
    return row


def persist_auto_pair(path: Path | str, pair: DraftLoopPair, *, replace: bool = False) -> bool:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    current = load_pairs_file(target, default_source="auto") if target.exists() else {}
    existing = current.get(pair.key)
    if existing == pair or (existing is not None and not replace):
        return False
    current[pair.key] = pair
    rows = [_pair_to_json(item) for item in sorted(current.values(), key=lambda value: value.key)]
    data = json.dumps({"pairs": rows}, ensure_ascii=False, indent=2, sort_keys=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(data)
        handle.write("\n")
    os.replace(tmp_name, target)
    return True


_UNCONFIRMED_AUTO_CUSTOMER_CONTEXT_KEYS = frozenset(
    {
        "read_only_customer_context",
        "amo_context",
        "tallanto_context",
        "timeline_context",
        "customer_context_summary",
        "crm_context_summary",
        "tallanto_context_summary",
        "timeline_context_summary",
        "known_client_fields",
        "lead_stage",
        "client_segment",
        "payment_context",
    }
)
_UNCONFIRMED_AUTO_MEMORY_KEYS = frozenset(
    {
        "crm_known_slots",
        "customer_memory",
        "read_only_customer_context",
        "timeline_context",
    }
)
