from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Protocol, Sequence

from mango_mvp.channels.subscription_llm import SubscriptionDraftResult
from mango_mvp.integrations.amo_wappi_phase1 import AmoWappiPhase1Config, WappiPhase1Client


DEFAULT_DRAFT_LOOP_DIR = Path.home() / ".mango_local" / "draft_loop"
DEFAULT_STOP_PATH = Path.home() / ".mango_secrets" / "STOP_DRAFT_LOOP"
DEFAULT_DEBOUNCE_SECONDS = 60
DEFAULT_HISTORY_LIMIT = 10
CONFIG_FINGERPRINT_SCHEMA_VERSION = "draft_loop_config_fingerprint_v1_2026_06_10"


class DraftLoopError(RuntimeError):
    pass


class DraftLoopConfigError(DraftLoopError):
    pass


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


@dataclass(frozen=True)
class DraftLoopConfig:
    profiles: Mapping[str, DraftLoopProfile]
    pairs: Mapping[DraftLoopKey, DraftLoopPair] = field(default_factory=dict)
    allowed_test_lead_ids: frozenset[str] = field(default_factory=frozenset)
    state_path: Path = DEFAULT_DRAFT_LOOP_DIR / "state.json"
    journal_path: Path = DEFAULT_DRAFT_LOOP_DIR / "journal.jsonl"
    manager_edit_log_path: Path = DEFAULT_DRAFT_LOOP_DIR / "manager_edits.jsonl"
    stop_path: Path = DEFAULT_STOP_PATH
    debounce_seconds: int = DEFAULT_DEBOUNCE_SECONDS
    history_limit: int = DEFAULT_HISTORY_LIMIT
    manager_outgoing_visible: bool | None = None
    config_fingerprint: Mapping[str, str] = field(default_factory=dict)

    def brand_for_profile(self, profile_id: str) -> str:
        profile = self.profiles.get(str(profile_id).strip())
        if profile is None:
            raise DraftLoopConfigError(f"Unknown Wappi profile_id: {profile_id!r}")
        return profile.brand

    def pair_for(self, key: DraftLoopKey) -> DraftLoopPair | None:
        return self.pairs.get(key)

    def phase1_config(self) -> AmoWappiPhase1Config:
        return AmoWappiPhase1Config(
            profile_brand_map={profile_id: profile.brand for profile_id, profile in self.profiles.items()},
            allowed_test_lead_ids=frozenset(str(item) for item in self.allowed_test_lead_ids),
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
        return not self.from_me and self.message_type == "text" and bool(self.text.strip())


class DraftBotProvider(Protocol):
    def build_draft(self, client_message: str, *, context: Mapping[str, Any] | None = None) -> SubscriptionDraftResult:
        ...


class AmoDraftNoteClient(Protocol):
    def add_draft_note_to_test_lead(self, lead_id: int | str, **kwargs: Any) -> Mapping[str, Any]:
        ...


ContextBuilder = Callable[[DraftLoopKey, Sequence[str], str, str], Mapping[str, Any]]
AutoResolver = Callable[[DraftLoopKey, WappiHistoryMessage], Optional[Mapping[str, Any]]]


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
            message_id = str(row.get("message_id") or "")
            if profile_id and chat_id and message_id:
                processed.add((profile_id, chat_id, message_id))
        return processed


class DraftLoopState:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path).expanduser()
        self.payload = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"processed": [], "pending_notes": {}}
        try:
            decoded = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise DraftLoopConfigError(f"Invalid draft loop state JSON: {self.path}") from exc
        if not isinstance(decoded, dict):
            raise DraftLoopConfigError("Draft loop state must be a JSON object.")
        decoded.setdefault("processed", [])
        decoded.setdefault("pending_notes", {})
        return decoded

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
        items = list(self.payload.get("processed") or [])
        marker = {"profile_id": message.profile_id, "chat_id": message.chat_id, "message_id": message.message_id}
        if marker not in items:
            items.append(marker)
        self.payload["processed"] = items[-5000:]

    def pending_notes(self) -> dict[str, Mapping[str, Any]]:
        raw = self.payload.get("pending_notes")
        return dict(raw) if isinstance(raw, Mapping) else {}

    def set_pending(self, message: WappiHistoryMessage, payload: Mapping[str, Any]) -> None:
        pending = dict(self.payload.get("pending_notes") or {})
        pending[_message_state_key(message)] = dict(payload)
        self.payload["pending_notes"] = pending

    def clear_pending(self, state_key: str) -> None:
        pending = dict(self.payload.get("pending_notes") or {})
        pending.pop(state_key, None)
        self.payload["pending_notes"] = pending

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(self.payload, ensure_ascii=False, indent=2, sort_keys=True)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{self.path.name}.", suffix=".tmp", dir=str(self.path.parent))
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(data)
            handle.write("\n")
        os.replace(tmp_name, self.path)


def _message_state_key(message: WappiHistoryMessage) -> str:
    return f"{message.profile_id}\t{message.chat_id}\t{message.message_id}"


def _is_private_dialog(dialog: Mapping[str, Any]) -> bool:
    dialog_type = str(dialog.get("type") or "").casefold()
    return not dialog_type or dialog_type == "user"


def wappi_message_from_raw(profile_id: str, raw: Mapping[str, Any]) -> WappiHistoryMessage | None:
    message_id = str(raw.get("id") or raw.get("message_id") or "").strip()
    chat_id = str(raw.get("chatId") or raw.get("chat_id") or "").strip()
    if not message_id or not chat_id:
        return None
    message_type = str(raw.get("type") or raw.get("message_type") or "").strip().casefold()
    text = str(raw.get("body") or raw.get("text") or raw.get("caption") or "").strip()
    timestamp_raw = raw.get("time") or raw.get("timestamp") or 0
    try:
        timestamp = int(float(timestamp_raw))
    except (TypeError, ValueError):
        timestamp = 0
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
        now_fn: Callable[[], datetime] | None = None,
    ) -> None:
        self.config = config
        self.wappi_client = wappi_client
        self.amo_client = amo_client
        self.bot_provider = bot_provider
        self.context_builder = context_builder
        self.journal = journal or DraftLoopJournal(config.journal_path)
        self.state = state or DraftLoopState(config.state_path)
        self.auto_resolver = auto_resolver
        self.now_fn = now_fn or (lambda: datetime.now(timezone.utc))

    def run_once(self, *, dry_run: bool = True) -> Mapping[str, Any]:
        stop_active = self.config.stop_path.expanduser().exists()
        retried = 0 if dry_run or stop_active else self.retry_pending_notes()
        processed_count = 0
        deferred_count = 0
        skipped_count = 0
        bot_calls = 0
        now_epoch = int(self.now_fn().timestamp())
        seen_processed = self.state.processed_keys() | self.journal.processed_message_keys()
        for profile in self.config.profiles.values():
            if profile.channel != "telegram":
                continue
            dialogs_payload = self.wappi_client.list_telegram_chats(profile_id=profile.profile_id, limit=50)
            dialogs = dialogs_payload.get("dialogs") if isinstance(dialogs_payload, Mapping) else []
            for dialog in dialogs if isinstance(dialogs, Sequence) else []:
                if not isinstance(dialog, Mapping):
                    continue
                chat_id = str(dialog.get("id") or "").strip()
                if not chat_id:
                    continue
                if not _is_private_dialog(dialog):
                    self.journal.append({"event": "chat_skipped", "profile_id": profile.profile_id, "chat_id": chat_id, "reason": "non_private"})
                    skipped_count += 1
                    continue
                messages = self._fetch_messages(profile.profile_id, chat_id)
                inbound_new = [
                    item
                    for item in messages
                    if item.is_inbound_text and item.key not in seen_processed and item.timestamp <= now_epoch - self.config.debounce_seconds
                ]
                if not inbound_new:
                    recent_inbound = [item for item in messages if item.is_inbound_text and item.key not in seen_processed]
                    if recent_inbound:
                        deferred_count += 1
                    continue
                if stop_active:
                    for item in inbound_new:
                        self.journal.append(_message_event("stop_raw_inbound", item, status="stop_not_processed"))
                    continue
                result = self._process_chat_messages(profile, messages, inbound_new, dry_run=dry_run)
                processed_count += int(result.get("processed", 0))
                skipped_count += int(result.get("skipped", 0))
                bot_calls += int(result.get("bot_calls", 0))
        if not dry_run and not stop_active:
            self.state.save()
        return {
            "processed": processed_count,
            "deferred": deferred_count,
            "skipped": skipped_count,
            "bot_calls": bot_calls,
            "retried_pending": retried,
            "stop_active": stop_active,
            "dry_run": dry_run,
        }

    def _fetch_messages(self, profile_id: str, chat_id: str) -> list[WappiHistoryMessage]:
        payload = self.wappi_client.get_telegram_chat_messages(
            profile_id=profile_id,
            chat_id=chat_id,
            limit=max(20, self.config.history_limit * 2),
            order="desc",
            mark_all=False,
        )
        raw_messages = payload.get("messages") if isinstance(payload, Mapping) else []
        messages: list[WappiHistoryMessage] = []
        for raw in raw_messages if isinstance(raw_messages, Sequence) else []:
            if not isinstance(raw, Mapping):
                continue
            item = wappi_message_from_raw(profile_id, raw)
            if item is not None:
                messages.append(item)
        messages.sort(key=lambda item: (item.timestamp, item.message_id))
        return messages

    def _process_chat_messages(
        self,
        profile: DraftLoopProfile,
        messages: Sequence[WappiHistoryMessage],
        inbound_new: Sequence[WappiHistoryMessage],
        *,
        dry_run: bool,
    ) -> Mapping[str, int]:
        key = DraftLoopKey(profile.profile_id, inbound_new[-1].chat_id)
        pair = self.config.pair_for(key)
        if pair is None:
            candidate = self.auto_resolver(key, inbound_new[-1]) if self.auto_resolver else None
            self.journal.append(
                {
                    **_message_event("pair_missing", inbound_new[-1], status="skipped"),
                    "auto_candidate": dict(candidate or {}),
                }
            )
            return {"processed": 0, "skipped": len(inbound_new), "bot_calls": 0}
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
            return {"processed": 0, "skipped": len(inbound_new), "bot_calls": 0}
        history = _history_lines(messages[-self.config.history_limit :])
        client_message = inbound_new[-1].text
        context = self.context_builder(key, history, client_message, brand)
        result = self.bot_provider.build_draft(client_message, context=context)
        route = str(getattr(result, "route", "") or "")
        safety_flags = tuple(str(item) for item in (getattr(result, "safety_flags", ()) or ()))
        draft_text = str(getattr(result, "draft_text", "") or "")
        last_message = inbound_new[-1]
        pending_payload = {
            "profile_id": last_message.profile_id,
            "chat_id": last_message.chat_id,
            "message_id": last_message.message_id,
            "lead_id": pair.lead_id,
            "brand": brand,
            "route": route,
            "safety_flags": list(safety_flags),
            "bot_draft_text": draft_text,
            "config_fingerprint": dict(self.config.config_fingerprint or {}),
            "status": "note_pending",
        }
        self.journal.append({**pending_payload, "event": "draft_created", "status": "dry_run" if dry_run else "note_pending"})
        if dry_run:
            return {"processed": 0, "skipped": 0, "bot_calls": 1}
        self.state.set_pending(last_message, pending_payload)
        self.state.save()
        self._write_note(pending_payload, retry=False)
        self.state.clear_pending(_message_state_key(last_message))
        for item in inbound_new:
            self.state.mark_processed(item)
        self.journal.append({**pending_payload, "event": "note_written", "status": "note_written"})
        return {"processed": len(inbound_new), "skipped": 0, "bot_calls": 1}

    def retry_pending_notes(self) -> int:
        retries = 0
        for state_key, payload in list(self.state.pending_notes().items()):
            if not isinstance(payload, Mapping):
                self.state.clear_pending(state_key)
                continue
            if str(payload.get("status") or "") == "manual_review":
                continue
            if bool(payload.get("retry_attempted")):
                self.state.payload["pending_notes"][state_key] = {**dict(payload), "status": "manual_review"}
                self.journal.append({**dict(payload), "event": "note_retry_failed", "status": "manual_review"})
                continue
            try:
                self._write_note(payload, retry=True)
            except Exception as exc:  # noqa: BLE001
                self.state.payload["pending_notes"][state_key] = {
                    **dict(payload),
                    "retry_attempted": True,
                    "status": "manual_review",
                    "error": str(exc)[:300],
                }
                self.journal.append({**dict(payload), "event": "note_retry_failed", "status": "manual_review", "error": str(exc)[:300]})
            else:
                self.state.clear_pending(state_key)
                self.journal.append({**dict(payload), "event": "note_retried", "status": "note_written"})
                retries += 1
        if retries:
            self.state.save()
        return retries

    def _write_note(self, payload: Mapping[str, Any], *, retry: bool) -> None:
        del retry
        outgoing_note = ""
        if self.config.manager_outgoing_visible is False:
            outgoing_note = "Важно: бот не видит ответы менеджера в Wappi-истории."
        self.amo_client.add_draft_note_to_test_lead(
            str(payload.get("lead_id") or ""),
            config=self.config.phase1_config(),
            draft_text=str(payload.get("bot_draft_text") or ""),
            brand=str(payload.get("brand") or ""),
            profile_id=str(payload.get("profile_id") or ""),
            route=str(payload.get("route") or ""),
            safety_flags=tuple(payload.get("safety_flags") or ()),
            outgoing_visibility_note=outgoing_note,
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


def _history_lines(messages: Sequence[WappiHistoryMessage]) -> tuple[str, ...]:
    lines: list[str] = []
    for item in messages:
        if item.message_type != "text" or not item.text.strip():
            continue
        prefix = "Ответ" if item.from_me else "Клиент"
        lines.append(f"{prefix}: {item.text.strip()}")
    return tuple(lines[-DEFAULT_HISTORY_LIMIT:])


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
    window_seconds: int = 4 * 60 * 60,
) -> list[dict[str, Any]]:
    assigned_outgoing: set[str] = set()
    best_by_draft: dict[str, tuple[float, OutgoingWindowMessage]] = {}
    for sent in outgoing:
        candidates: list[tuple[float, DraftWindow]] = []
        for draft in drafts:
            if sent.sent_ts < draft.draft_ts or sent.sent_ts > draft.draft_ts + window_seconds:
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
        closed = now_ts >= draft.draft_ts + window_seconds
        if not closed:
            continue
        had_outgoing = any(draft.draft_ts <= sent.sent_ts <= draft.draft_ts + window_seconds for sent in outgoing)
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


def load_profiles_file(path: Path | str) -> dict[str, DraftLoopProfile]:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes, bytearray)):
        raise DraftLoopConfigError("amo_wappi_profiles.json must be a list.")
    result: dict[str, DraftLoopProfile] = {}
    for row in payload:
        if not isinstance(row, Mapping):
            continue
        profile_id = str(row.get("profile_id") or "").strip()
        brand = str(row.get("brand") or "").strip().casefold()
        channel = str(row.get("channel") or "").strip().casefold()
        if not profile_id or brand not in {"foton", "unpk"} or channel != "telegram":
            continue
        result[profile_id] = DraftLoopProfile(profile_id=profile_id, brand=brand, channel=channel)
    return result


def load_pairs_file(path: Path | str) -> dict[DraftLoopKey, DraftLoopPair]:
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
        )
        if not pair.lead_id or pair.expected_brand not in {"foton", "unpk"}:
            raise DraftLoopConfigError("draft_loop_pairs entries require lead_id and expected_brand.")
        result[key] = pair
    return result
