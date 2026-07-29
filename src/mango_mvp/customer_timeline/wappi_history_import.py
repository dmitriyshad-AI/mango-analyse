from __future__ import annotations

import json
import hashlib
import math
import os
import re
import sqlite3
import subprocess
import time
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from functools import partial
from pathlib import Path
from threading import Lock
from typing import Any, Mapping, Optional, Protocol, Sequence
from urllib import parse as url_parse

from mango_mvp.customer_timeline.contracts import (
    BotContextChunk,
    CustomerIdentity,
    IdentityLink,
    IdentityMatchClass,
    IdentityStatus,
    TimelineDirection,
    TimelineEvent,
    TimelineEventType,
    TimelineParticipant,
)
from mango_mvp.customer_timeline.ids import (
    normalize_email,
    normalize_key,
    optional_text,
    require_text,
    stable_customer_id,
    stable_digest,
)
from mango_mvp.customer_timeline.import_cli import safety_ok, timeline_import_cli_safety_contract
from mango_mvp.customer_timeline.ingestion import (
    PHONE_IDENTITY_LINK_TYPES,
    TimelineImportReport,
    TimelineImportService,
    TimelineNormalizedBatch,
    TimelineSourceRecord,
    compact_text,
    parse_source_datetime,
    scrub_timeline_persisted_json,
)
from mango_mvp.customer_timeline.safety import (
    blocked_live_actions,
    guard_customer_timeline_output_path,
    is_customer_timeline_prod_path,
)
from mango_mvp.customer_timeline.store import (
    CustomerTimelineSQLiteStore,
    customer_timeline_readonly_uri,
    guard_customer_timeline_sqlite_path,
    json_dumps,
)
from mango_mvp.integrations.amo_wappi_phase1 import (
    AMO_WAPPI_ENV_FILE,
    DEFAULT_AMO_WAPPI_CONFIG_PATH,
    AmoWappiConfigError,
    AmoWappiHttpError,
    AmoWappiPhase1Config,
    WappiClientConfig,
    WappiPhase1Client,
    _json_http_request,
    load_env_file,
)
from mango_mvp.integrations.amo_wappi_transport import DefaultDenyTransport, SafeTransportPolicy
from mango_mvp.integrations.amo_wappi_auto_resolver import (
    DEFAULT_AMO_MCP_ENV_PATH,
    DEFAULT_STOPLIST_PATH,
    AmoAutoResolver,
    build_amo_auto_resolver,
    load_phone_stoplist,
    max_dialog_phone,
)
from mango_mvp.integrations.draft_loop import (
    DraftLoopProfile,
    DraftLoopKey,
    DraftLoopPair,
    WappiHistoryMessage,
    _is_deferred_fetch_exception,
    build_draft_loop_code_identity,
    load_pairs_file,
    wappi_message_from_raw,
)
from mango_mvp.utils.phone import normalize_phone


WAPPI_HISTORY_IMPORT_SCHEMA_VERSION = "wappi_history_timeline_import_v2"
SOURCE_SYSTEM_BY_CHANNEL = {"telegram": "wappi_telegram", "max": "wappi_max"}
EVENT_TYPE_BY_CHANNEL = {
    "telegram": TimelineEventType.TELEGRAM_MESSAGE,
    "max": TimelineEventType.MAX_MESSAGE,
}
RESOLVED_MATCH_CLASS_BY_IDENTITY_AUTHORITY = {
    "draft_loop_pair": IdentityMatchClass.MANUAL,
    "timeline_identity": IdentityMatchClass.STRONG_UNIQUE,
    "amo_auto_resolver": IdentityMatchClass.STRONG_UNIQUE,
    "wappi_amo_widget": IdentityMatchClass.STRONG_UNIQUE,
    "amo_talk_authoritative": IdentityMatchClass.STRONG_UNIQUE,
    "wappi_provisional": IdentityMatchClass.INFERRED,
}
WAPPI_MESSAGE_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b", re.I)
WAPPI_MESSAGE_PHONE_RE = re.compile(r"(?<!\d)(?:\+?7|8)(?:[\s()\-]*\d){10}(?!\d)")


def _is_exact_authority_override(
    existing_customer: str,
    existing_authority: str,
    proposed_customer: str,
    proposed_authority: str,
) -> bool:
    return bool(
        existing_customer
        and proposed_customer
        and proposed_customer != existing_customer
        and proposed_authority in WAPPI_EXACT_AMO_AUTHORITIES
        and existing_authority not in WAPPI_EXACT_AMO_AUTHORITIES
    )


class WappiPhysicalRequestBudgetExceeded(RuntimeError):
    pass


@dataclass
class WappiPhysicalRequestBudget:
    limit: int
    used: int = 0
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def take(self) -> None:
        with self._lock:
            if self.used >= self.limit:
                raise WappiPhysicalRequestBudgetExceeded("Wappi physical request budget exhausted")
            self.used += 1


class WappiHistoryClient(Protocol):
    transport: object

    def list_chats(
        self,
        *,
        channel: str,
        profile_id: str,
        limit: int = 50,
        offset: int = 0,
        order: str = "desc",
        show_all: bool = False,
    ) -> Mapping[str, Any]:
        ...

    def get_chat_messages(
        self,
        *,
        channel: str,
        profile_id: str,
        chat_id: str,
        limit: int = 50,
        offset: int = 0,
        order: str = "desc",
        mark_all: bool = False,
    ) -> Mapping[str, Any]:
        ...


@dataclass(frozen=True)
class WappiProfileSpec:
    profile_id: str
    brand: str
    channel: str
    label: str = ""

    def __post_init__(self) -> None:
        profile_id = require_text(self.profile_id, "profile_id")
        brand = normalize_brand(self.brand)
        channel = str(self.channel or "").strip().casefold()
        if channel not in SOURCE_SYSTEM_BY_CHANNEL:
            raise AmoWappiConfigError(f"Wappi profile channel must be telegram or max: {self.channel!r}")
        object.__setattr__(self, "profile_id", profile_id)
        object.__setattr__(self, "brand", brand)
        object.__setattr__(self, "channel", channel)
        object.__setattr__(self, "label", str(self.label or "").strip())

    @property
    def source_system(self) -> str:
        return SOURCE_SYSTEM_BY_CHANNEL[self.channel]


@dataclass(frozen=True)
class WappiFetchLimits:
    chat_limit_per_profile: int = 50
    messages_per_chat: int = 100
    message_limit_total: int = 2000
    request_limit_total: int = 500
    page_size: int = 100
    sleep_seconds: float = 0.2
    show_all_chats: bool = False
    complete_message_history: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "chat_limit_per_profile", max(0, int(self.chat_limit_per_profile)))
        object.__setattr__(self, "messages_per_chat", max(0, int(self.messages_per_chat)))
        object.__setattr__(self, "message_limit_total", max(0, int(self.message_limit_total)))
        object.__setattr__(self, "request_limit_total", max(1, int(self.request_limit_total)))
        object.__setattr__(self, "page_size", max(1, min(int(self.page_size), 100)))
        object.__setattr__(self, "sleep_seconds", max(0.0, float(self.sleep_seconds)))
        object.__setattr__(self, "complete_message_history", bool(self.complete_message_history))


@dataclass(frozen=True)
class WappiHistoryImportConfig:
    timeline_db: Path
    allowed_root: Path
    tenant_id: str = "foton"
    env_file: Path = AMO_WAPPI_ENV_FILE
    phase1_config: Path = DEFAULT_AMO_WAPPI_CONFIG_PATH
    pairs_file: Optional[Path] = Path.home() / ".mango_secrets" / "draft_loop_pairs.json"
    auto_pairs_file: Optional[Path] = Path.home() / ".mango_secrets" / "draft_loop_auto_pairs.json"
    amo_auto_resolver_enabled: bool = False
    amo_mcp_env_file: Optional[Path] = DEFAULT_AMO_MCP_ENV_PATH
    shared_phone_stoplist: Optional[Path] = DEFAULT_STOPLIST_PATH
    apply: bool = False
    require_nonempty_profiles: bool = False
    require_widget_linkage: bool = False
    widget_link_db: Optional[Path] = None
    refresh_widget_links: bool = True
    widget_coverage_only: bool = False
    actor: str = "wappi_history_timeline_import"
    idempotency_key: Optional[str] = None
    out_path: Optional[Path] = None
    checkpoint_dir: Optional[Path] = None
    limits: WappiFetchLimits = field(default_factory=WappiFetchLimits)

    def __post_init__(self) -> None:
        root = Path(self.allowed_root).expanduser().resolve(strict=False)
        timeline_db = guard_customer_timeline_output_path(guard_customer_timeline_sqlite_path(self.timeline_db), root)
        if self.apply and is_customer_timeline_prod_path(timeline_db):
            raise ValueError("Wappi history apply must not target a production Customer Timeline")
        out_path = guard_customer_timeline_output_path(self.out_path, root) if self.out_path else None
        widget_link_db = (
            guard_customer_timeline_output_path(self.widget_link_db, root)
            if self.widget_link_db
            else None
        )
        object.__setattr__(self, "allowed_root", root)
        object.__setattr__(self, "timeline_db", timeline_db)
        object.__setattr__(self, "env_file", Path(self.env_file).expanduser())
        object.__setattr__(self, "phase1_config", Path(self.phase1_config).expanduser())
        object.__setattr__(self, "pairs_file", Path(self.pairs_file).expanduser() if self.pairs_file else None)
        object.__setattr__(self, "auto_pairs_file", Path(self.auto_pairs_file).expanduser() if self.auto_pairs_file else None)
        object.__setattr__(self, "amo_mcp_env_file", Path(self.amo_mcp_env_file).expanduser() if self.amo_mcp_env_file else None)
        object.__setattr__(self, "shared_phone_stoplist", Path(self.shared_phone_stoplist).expanduser() if self.shared_phone_stoplist else None)
        object.__setattr__(self, "tenant_id", normalize_key(self.tenant_id, "tenant_id"))
        object.__setattr__(self, "actor", require_text(self.actor, "actor"))
        object.__setattr__(self, "out_path", out_path)
        object.__setattr__(self, "widget_link_db", widget_link_db)
        object.__setattr__(self, "require_widget_linkage", bool(self.require_widget_linkage))
        object.__setattr__(self, "widget_coverage_only", bool(self.widget_coverage_only))
        if self.widget_coverage_only and self.apply:
            raise ValueError("Widget coverage-only mode cannot apply Timeline writes")
        if self.widget_coverage_only and self.widget_link_db is None:
            raise ValueError("Widget coverage-only mode requires widget_link_db")
        if self.widget_coverage_only and not self.refresh_widget_links:
            raise ValueError("Widget coverage-only mode must refresh the chat catalogue")
        checkpoint_dir = guard_customer_timeline_output_path(self.checkpoint_dir, root) if self.checkpoint_dir else None
        object.__setattr__(self, "checkpoint_dir", checkpoint_dir)
        if checkpoint_dir is not None and not self.limits.complete_message_history:
            raise ValueError("Wappi fetch checkpoint requires complete_message_history=True")


# --- Wappi fetch checkpoint -------------------------------------------------
# Resumes a large history load from the last CONFIRMED chat instead of page 1.
# Reuses the checkpoint pattern already in this package (amo_incremental.py,
# tallanto_cards_sync.py): universe fingerprint, honest reset reason, atomic
# write, and "a partial pass never confirms freshness". Deliberately identity
# based, not positional: the catalogue is re-listed every run (cheap) and only
# already-CONFIRMED chats are skipped, so a reordered catalogue cannot make the
# importer silently skip an unread chat. Stores digests and offsets only --
# never message text, names, phones, emails or tokens.
WAPPI_HISTORY_CHECKPOINT_SCHEMA_VERSION = "customer_timeline_wappi_history_checkpoint_v1"
WAPPI_INCREMENTAL_FULL_AUDIT_DAYS = 7


def wappi_history_checkpoint_path(checkpoint_dir: Path) -> Path:
    return Path(checkpoint_dir) / "wappi_history_checkpoint.json"


def wappi_checkpoint_token(value: str) -> str:
    # A Telegram chat_id is the peer's user id: keep only a one-way digest.
    return stable_digest({"wappi_chat": str(value or "")})[:32]


def wappi_checkpoint_anchor(tokens: Sequence[str]) -> str:
    # Anchors are computed over id sequences only, never over raw payloads:
    # unread counters and message previews would otherwise force a reset every run.
    return stable_digest({"tokens": list(tokens)})


def _wappi_full_audit_state(entry: Mapping[str, Any]) -> tuple[bool, bool]:
    value = str(entry.get("full_audit_at") or "").strip()
    if not value:
        return True, False
    try:
        audited_at = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return True, True
    if audited_at.tzinfo is None:
        audited_at = audited_at.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    clock_anomaly = audited_at > now + timedelta(hours=1)
    return clock_anomaly or now - audited_at >= timedelta(days=WAPPI_INCREMENTAL_FULL_AUDIT_DAYS), clock_anomaly


def wappi_fetch_universe_fingerprint(
    profile: WappiProfileSpec, limits: WappiFetchLimits, *, tenant_id: str = ""
) -> str:
    return stable_digest(
        {
            "tenant_id": tenant_id,
            "profile_id": profile.profile_id,
            "brand": profile.brand,
            "channel": profile.channel,
            "show_all_chats": bool(limits.show_all_chats),
            "page_size": int(limits.page_size),
            "chat_limit_per_profile": int(limits.chat_limit_per_profile),
            "messages_per_chat": int(limits.messages_per_chat),
            "complete_message_history": bool(limits.complete_message_history),
        }
    )


def load_wappi_history_checkpoint(checkpoint_dir: Optional[Path]) -> Mapping[str, Any]:
    if checkpoint_dir is None:
        return {}
    try:
        payload = json.loads(wappi_history_checkpoint_path(checkpoint_dir).read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, OSError):
        return {}
    if not isinstance(payload, Mapping):
        return {}
    if payload.get("schema_version") != WAPPI_HISTORY_CHECKPOINT_SCHEMA_VERSION:
        return {}
    profiles_state = payload.get("profiles")
    if not isinstance(profiles_state, Mapping):
        return {}
    # A corrupted entry must degrade like a corrupted file: be ignored, not crash the
    # nightly step into the same failure every single night.
    return {"schema_version": payload.get("schema_version"), "profiles": {
        str(key): entry
        for key, entry in profiles_state.items()
        if _wappi_checkpoint_entry_is_valid(entry)
    }}


def _wappi_safe_int(value: Any) -> Optional[int]:
    if value is None:
        return 0
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str) and value.strip().lstrip("+-").isdigit():
        try:
            parsed = int(value.strip())
        except ValueError:
            return None
    else:
        return None
    return parsed if -(2**63) <= parsed <= 2**63 - 1 else None


def _wappi_checkpoint_entry_is_valid(value: Any) -> bool:
    if not isinstance(value, Mapping) or not isinstance(value.get("chats_done", []), list):
        return False
    for key in ("chat_markers", "full_audit_markers"):
        markers = value.get(key, {})
        if not isinstance(markers, Mapping) or any(
            not isinstance(marker, str)
            or (parsed := _wappi_safe_int(stamp)) is None
            or parsed < 0
            for marker, stamp in markers.items()
        ):
            return False
    if any(key in value and not isinstance(value[key], bool) for key in ("complete", "incremental_cycle")):
        return False
    if any(key in value and not isinstance(value[key], str) for key in ("full_audit_at", "full_audit_started_at")):
        return False
    return bool(
        _wappi_active_chat_is_valid(value.get("active_chat"))
        and _wappi_safe_int(value.get("timeline_rows")) is not None
        and _wappi_safe_int(value.get("catalog_next_offset")) is not None
    )


def _wappi_active_chat_is_valid(value: Any) -> bool:
    if value in (None, {}):
        return True
    if not isinstance(value, Mapping):
        return False
    offset = _wappi_safe_int(value.get("message_offset"))
    page_offset = _wappi_safe_int(value.get("page_offset"))
    return bool(
        isinstance(value.get("chat"), str)
        and isinstance(value.get("page_anchor") or "", str)
        and offset is not None
        and offset >= 0
        and page_offset is not None
        and page_offset >= 0
    )


def save_wappi_history_checkpoint(checkpoint_dir: Optional[Path], profiles: Mapping[str, Any]) -> None:
    if checkpoint_dir is None:
        return
    from mango_mvp.customer_timeline.amo_incremental import _atomic_write_text

    path = wappi_history_checkpoint_path(checkpoint_dir)
    if not profiles:
        if path.exists():
            path.unlink()
        return
    _atomic_write_text(
        path,
        json.dumps(
            {"schema_version": WAPPI_HISTORY_CHECKPOINT_SCHEMA_VERSION, "profiles": dict(profiles)},
            ensure_ascii=False,
            sort_keys=True,
        ),
    )


def wappi_timeline_state(
    db_path: Path, *, tenant_id: str, profiles: Sequence[WappiProfileSpec] = ()
) -> dict[str, Mapping[str, Any]]:
    state = {
        f"{profile.source_system}:{profile.profile_id}": {"rows": 0, "source_digest": stable_digest([])}
        for profile in profiles
    }
    if not state or not Path(db_path).exists():
        return state
    con = open_readonly_sqlite(db_path)
    try:
        if not sqlite_table_exists(con, "timeline_events"):
            return state
        for profile in profiles:
            prefix = f"{profile.profile_id}:"
            source_ids = [
                str(row[0])
                for row in con.execute(
                    "SELECT source_id FROM timeline_events "
                    "WHERE tenant_id = ? AND source_system = ? AND substr(source_id, 1, ?) = ? "
                    "ORDER BY source_id",
                    (tenant_id, profile.source_system, len(prefix), prefix),
                )
            ]
            state[f"{profile.source_system}:{profile.profile_id}"] = {
                "rows": len(source_ids),
                "source_digest": stable_digest(source_ids),
            }
    finally:
        con.close()
    return state


def usable_wappi_checkpoint_profiles(
    checkpoint: Mapping[str, Any], *, db_row_counts: Mapping[str, int],
    db_source_digests: Optional[Mapping[str, str]] = None,
) -> dict[str, Any]:
    """Drop entries whose staging DB lost the rows the checkpoint calls confirmed."""
    profiles_state = checkpoint.get("profiles") if isinstance(checkpoint, Mapping) else None
    usable: dict[str, Any] = {}
    if not isinstance(profiles_state, Mapping):
        return usable
    for key, entry in profiles_state.items():
        if not isinstance(entry, Mapping):
            continue
        if int(entry.get("timeline_rows") or 0) > int(db_row_counts.get(str(key), 0)):
            continue
        if db_source_digests is not None and str(entry.get("timeline_source_digest") or "") != str(
            db_source_digests.get(str(key)) or ""
        ):
            continue
        usable[str(key)] = dict(entry)
    return usable


WAPPI_WIDGET_LINK_SCHEMA = "wappi_amo_link_map_v3"
WAPPI_AMO_EVENT_ORIGIN = {
    "wappi_telegram": "pro.wappi.tg",
    "wappi_max": "pro.wappi.3",
}
WAPPI_AMO_EVENT_MATCH_WINDOW_SEC = 15
WAPPI_AMO_EVENT_MIN_MATCHES = 2
WAPPI_CATALOG_MAX_PASSES = 5
WAPPI_RESOLVED_LINK_STATUSES = {
    "resolved_contact_only",
    "resolved_one_lead",
    "resolved_multiple_leads",
}
WAPPI_TECHNICAL_LINK_STATUSES = {
    "auth_error",
    "rate_limit",
    "timeout",
    "http_5xx",
    "invalid_response",
    "lookup_error",
    "request_limit",
}
WAPPI_EXACT_AMO_AUTHORITIES = {"wappi_amo_widget", "amo_talk_authoritative"}


def _ensure_wappi_widget_link_schema(con: sqlite3.Connection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS wappi_amo_links (
          channel TEXT NOT NULL,
          profile_id TEXT NOT NULL,
          chat_id TEXT NOT NULL,
          contact_id TEXT,
          lead_ids_json TEXT NOT NULL,
          status TEXT NOT NULL,
          checked_at TEXT NOT NULL,
          response_sha256 TEXT NOT NULL,
          resolution_source TEXT NOT NULL DEFAULT 'wappi_widget',
          last_timestamp INTEGER NOT NULL DEFAULT 0,
          matched_points INTEGER NOT NULL DEFAULT 0,
          PRIMARY KEY (channel, profile_id, chat_id)
        )
        """
    )
    columns = {str(row[1]) for row in con.execute("PRAGMA table_info(wappi_amo_links)")}
    migrations = {
        "resolution_source": "TEXT NOT NULL DEFAULT 'wappi_widget'",
        "last_timestamp": "INTEGER NOT NULL DEFAULT 0",
        "matched_points": "INTEGER NOT NULL DEFAULT 0",
        "amo_talk_id": "TEXT NOT NULL DEFAULT ''",
        "amo_chat_id": "TEXT NOT NULL DEFAULT ''",
    }
    for name, declaration in migrations.items():
        if name not in columns:
            con.execute(f"ALTER TABLE wappi_amo_links ADD COLUMN {name} {declaration}")
    con.execute(
        """
        UPDATE wappi_amo_links
        SET status = 'candidate', resolution_source = 'amo_event_sequence_candidate'
        WHERE resolution_source = 'amo_event_sequence'
        """
    )


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _wappi_link_check_is_stale(value: Any, *, now: datetime | None = None) -> bool:
    try:
        checked_at = datetime.fromisoformat(str(value or ""))
        if checked_at.tzinfo is None:
            checked_at = checked_at.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return True
    current = now or datetime.now(timezone.utc)
    return current.timestamp() - checked_at.timestamp() >= 24 * 60 * 60


def _extract_wappi_total_count(payload: Mapping[str, Any]) -> tuple[bool, int | None]:
    pending: list[Mapping[str, Any]] = [payload]
    visited: set[int] = set()
    while pending:
        item = pending.pop()
        if id(item) in visited:
            continue
        visited.add(id(item))
        for key in ("total_count", "totalCount"):
            if key in item:
                raw_total = item.get(key)
                if isinstance(raw_total, bool) or not re.fullmatch(r"[0-9]+", str(raw_total).strip()):
                    return True, None
                return True, int(raw_total)
        pending.extend(
            value
            for key in ("data", "meta", "pagination")
            if isinstance((value := item.get(key)), Mapping)
        )
    return False, None


def _find_wappi_widget_contact(
    client: WappiPhase1Client,
    *,
    profile: WappiProfileSpec,
    runtime_profile: Mapping[str, Any],
    dialog: Mapping[str, Any],
    crm_id: str,
) -> Mapping[str, Any]:
    user = dialog.get("user") if isinstance(dialog.get("user"), Mapping) else {}
    chat_id = extract_chat_id(dialog)
    if profile.channel == "max" and str(dialog.get("type") or "").strip().upper() == "DIALOG":
        peers = tuple(
            item for item in (dialog.get("participants") or ())
            if isinstance(item, Mapping) and not item.get("is_me")
        )
        chat_id = str(dialog.get("max_user_id") or "").strip() or (
            str(peers[0].get("user_id") or "").strip() if len(peers) == 1 else chat_id
        )
    if not chat_id:
        raise AmoWappiConfigError("Wappi dialog does not contain one authoritative peer id")
    manager = dialog.get("manager") if isinstance(dialog.get("manager"), Mapping) else {}
    return client.find_amocrm_contact(
        channel=profile.channel,
        chat_id=chat_id,
        phone=str(dialog.get("phone") or user.get("Phone") or user.get("phone") or "").strip(),
        username=str(user.get("Username") or user.get("username") or dialog.get("username") or "").strip(),
        platform=str(runtime_profile.get("platform") or profile.channel).strip(),
        crm_id=crm_id,
        profile_uuid=str(runtime_profile.get("uuid") or runtime_profile.get("profile_id") or "").strip(),
        manager=str(manager.get("id") or dialog.get("manager_id") or "").strip(),
    )


def _widget_lookup_error_status(exc: Exception) -> str:
    message = str(exc).casefold()
    if "401" in message or "403" in message:
        return "auth_error"
    if "429" in message or "rate limit" in message:
        return "rate_limit"
    if re.search(r"http\s+5\d\d", message):
        return "http_5xx"
    if "timeout" in message or "timed out" in message:
        return "timeout"
    if "invalid json" in message or "invalid response" in message:
        return "invalid_response"
    return "lookup_error"


def _widget_linkage_status(status: str, lead_ids: Sequence[str]) -> str:
    if status == "resolved":
        if not lead_ids:
            return "resolved_contact_only"
        if len(lead_ids) == 1:
            return "resolved_one_lead"
        return "resolved_multiple_leads"
    if status == "missing":
        return "widget_no_contact"
    if status == "conflict":
        return "relation_conflict"
    return status


def collect_wappi_widget_links(
    *,
    client: WappiPhase1Client,
    profiles: Sequence[WappiProfileSpec],
    runtime_profiles: Mapping[tuple[str, str], Mapping[str, Any]],
    crm_id: str,
    db_path: Path,
    limits: WappiFetchLimits,
    workers: int = 4,
    force_recheck: bool = False,
) -> Mapping[str, Any]:
    """Persist the widget's authoritative IDs without storing chat text or credentials."""
    target = Path(db_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        raise ValueError("Wappi AMO link DB must not be a symlink")
    if not target.exists():
        descriptor = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(descriptor)
    os.chmod(target, 0o600)
    counts: Counter[str] = Counter()
    operations: Counter[str] = Counter()
    profile_reports: dict[str, Mapping[str, Any]] = {}
    requests = 0
    limit_hit = False

    def lookup(
        profile: WappiProfileSpec,
        runtime_profile: Mapping[str, Any],
        chat_id: str,
        dialog: Mapping[str, Any],
        previous: Mapping[str, Any] | None,
    ) -> tuple[str, str, tuple[str, ...], str, int, Mapping[str, Any] | None]:
        try:
            result = _find_wappi_widget_contact(
                client,
                profile=profile,
                runtime_profile=runtime_profile,
                dialog=dialog,
                crm_id=crm_id,
            )
        except WappiPhysicalRequestBudgetExceeded:
            return chat_id, "", (), "request_limit", _safe_int(dialog.get("last_timestamp")), previous
        except Exception as exc:  # noqa: BLE001 - only a safe error class is persisted.
            return chat_id, "", (), _widget_lookup_error_status(exc), _safe_int(dialog.get("last_timestamp")), previous
        if not isinstance(result, Mapping):
            return chat_id, "", (), "invalid_response", _safe_int(dialog.get("last_timestamp")), previous
        contact = result.get("contact") if isinstance(result.get("contact"), Mapping) else {}
        contact_id = str(contact.get("id") or "").strip()
        if contact_id and (not contact_id.isdigit() or int(contact_id) <= 0):
            return chat_id, "", (), "invalid_response", _safe_int(dialog.get("last_timestamp")), previous
        embedded = contact.get("_embedded")
        if embedded is not None and not isinstance(embedded, Mapping):
            return chat_id, "", (), "invalid_response", _safe_int(dialog.get("last_timestamp")), previous
        raw_leads: list[Mapping[str, Any]] = []
        for candidate in (
            result.get("leads"),
            embedded.get("leads") if isinstance(embedded, Mapping) else None,
        ):
            if candidate is None:
                continue
            if not isinstance(candidate, Sequence) or isinstance(candidate, (str, bytes, bytearray)):
                return chat_id, "", (), "invalid_response", _safe_int(dialog.get("last_timestamp")), previous
            if any(not isinstance(item, Mapping) for item in candidate):
                return chat_id, "", (), "invalid_response", _safe_int(dialog.get("last_timestamp")), previous
            raw_leads.extend(candidate)
        lead_ids = tuple(sorted({str(item.get("id") or "").strip() for item in raw_leads}))
        if any(not lead_id.isdigit() or int(lead_id) <= 0 for lead_id in lead_ids):
            return chat_id, "", (), "invalid_response", _safe_int(dialog.get("last_timestamp")), previous
        status = "resolved" if contact_id else "missing"
        return chat_id, contact_id, lead_ids, status, _safe_int(dialog.get("last_timestamp")), previous

    with sqlite3.connect(target, timeout=60.0) as con:
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA busy_timeout=60000")
        _ensure_wappi_widget_link_schema(con)
        con.commit()
        for profile in profiles:
            profile_key = f"{profile.channel}:{profile.profile_id}"
            runtime_profile = runtime_profiles.get((profile.channel, profile.profile_id))
            if runtime_profile is None:
                counts["profile_missing"] += 1
                profile_reports[profile_key] = {
                    "accounting_complete": False,
                    "linkage_complete": False,
                    "error": "profile_missing",
                }
                continue
            catalog: dict[str, Mapping[str, Any]] = {}
            missing_chat_ids: set[str] = set()
            total_count: int | None = None
            passes = 0
            catalog_error = ""
            for pass_index in range(WAPPI_CATALOG_MAX_PASSES):
                if requests >= limits.request_limit_total:
                    catalog_error = "request_limit"
                    limit_hit = True
                    break
                passes = pass_index + 1
                offset = 0
                while (
                    (limits.complete_message_history or offset < limits.chat_limit_per_profile)
                    and requests < limits.request_limit_total
                ):
                    page_limit = (
                        limits.page_size
                        if limits.complete_message_history
                        else min(limits.page_size, limits.chat_limit_per_profile - offset)
                    )
                    try:
                        payload = client.list_chats(
                            channel=profile.channel,
                            profile_id=profile.profile_id,
                            limit=page_limit,
                            offset=offset,
                            order="asc",
                            show_all=limits.show_all_chats,
                        )
                    except WappiPhysicalRequestBudgetExceeded:
                        catalog_error = "request_limit"
                        limit_hit = True
                        break
                    except Exception as exc:  # noqa: BLE001 - only a safe error class is reported.
                        catalog_error = _widget_lookup_error_status(exc)
                        break
                    requests += 1
                    if not isinstance(payload, Mapping):
                        catalog_error = "invalid_response"
                        break
                    total_present, observed_total = _extract_wappi_total_count(payload)
                    if total_present and observed_total is None:
                        catalog_error = "invalid_response"
                        break
                    if observed_total is not None:
                        total_count = max(total_count or 0, observed_total)
                    dialogs = extract_wappi_items(payload, "dialogs", "chats", "items", "data")
                    if not dialogs:
                        break
                    for dialog in dialogs:
                        chat_id = extract_chat_id(dialog)
                        if chat_id:
                            catalog[chat_id] = dialog
                        else:
                            missing_chat_ids.add(stable_digest(dict(dialog)))
                    if total_count is not None and len(catalog) == total_count:
                        break
                    if len(dialogs) < page_limit:
                        break
                    offset += page_limit
                    if total_count is not None and offset >= total_count:
                        break
                if total_count is not None and len(catalog) == total_count:
                    break
                if total_count is None or catalog_error:
                    break
            if catalog_error:
                counts[catalog_error] += 1
            accounting_complete = bool(
                total_count is not None
                and (limits.complete_message_history or total_count <= limits.chat_limit_per_profile)
                and len(catalog) == total_count
                and not missing_chat_ids
            )
            if total_count is None:
                counts["total_count_missing"] += 1
            if (
                not limits.complete_message_history
                and total_count is not None
                and total_count > limits.chat_limit_per_profile
            ):
                counts["chat_limit_hit"] += 1
            if not accounting_complete and passes >= WAPPI_CATALOG_MAX_PASSES:
                counts["catalog_unstable"] += 1
            personal_dialogs = {
                chat_id: dialog
                for chat_id, dialog in catalog.items()
                if is_personal_wappi_dialog(profile, dialog)
            }
            non_personal = len(catalog) - len(personal_dialogs)
            if non_personal:
                counts["non_personal"] += non_personal
            if missing_chat_ids:
                counts["chat_id_missing"] += len(missing_chat_ids)
            pending: list[tuple[str, Mapping[str, Any], Mapping[str, Any] | None]] = []
            status_by_chat: dict[str, str] = {}
            for chat_id, dialog in personal_dialogs.items():
                    cached = con.execute(
                        """
                        SELECT status, checked_at, last_timestamp, contact_id, lead_ids_json
                        FROM wappi_amo_links
                        WHERE channel = ? AND profile_id = ? AND chat_id = ?
                        """,
                        (profile.channel, profile.profile_id, chat_id),
                    ).fetchone()
                    if cached is not None:
                        dialog_last_timestamp = _safe_int(dialog.get("last_timestamp"))
                        cached_last_timestamp = _safe_int(cached[2])
                        should_recheck = (
                            force_recheck
                            or str(cached[0]) in WAPPI_TECHNICAL_LINK_STATUSES
                            or (str(cached[0]) == "missing" and cached_last_timestamp == 0)
                            or (
                                str(cached[0]) in {"missing", "candidate"}
                                and _wappi_link_check_is_stale(cached[1])
                            )
                            or (
                                str(cached[0]) in {"missing", "resolved"}
                                and cached_last_timestamp > 0
                                and dialog_last_timestamp > cached_last_timestamp
                            )
                        )
                        if should_recheck:
                            pending.append(
                                (
                                    chat_id,
                                    dialog,
                                    {
                                        "status": str(cached[0]),
                                        "last_timestamp": cached_last_timestamp,
                                        "contact_id": str(cached[3] or ""),
                                        "lead_ids": tuple(json.loads(str(cached[4] or "[]"))),
                                    },
                                )
                            )
                            operations[f"recheck_{str(cached[0])}"] += 1
                            continue
                        con.execute(
                            """
                            UPDATE wappi_amo_links
                            SET last_timestamp = MAX(last_timestamp, ?)
                            WHERE channel = ? AND profile_id = ? AND chat_id = ?
                            """,
                            (
                                _safe_int(dialog.get("last_timestamp")),
                                profile.channel,
                                profile.profile_id,
                                chat_id,
                            ),
                        )
                        cached_status = str(cached[0])
                        cached_leads = tuple(json.loads(str(cached[4] or "[]")))
                        status_by_chat[chat_id] = _widget_linkage_status(cached_status, cached_leads)
                        operations[f"cached_{cached_status}"] += 1
                        continue
                    pending.append((chat_id, dialog, None))
            remaining = max(0, limits.request_limit_total - requests)
            skipped_pending = pending[remaining:]
            if skipped_pending:
                limit_hit = True
                for chat_id, _dialog, _previous in skipped_pending:
                    status_by_chat[chat_id] = "request_limit"
            pending = pending[:remaining]
            requests += len(pending)
            with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
                resolved_rows = pool.map(
                    lambda item: lookup(profile, runtime_profile, item[0], item[1], item[2]),
                    pending,
                )
                for chat_id, contact_id, lead_ids, status, last_timestamp, previous in resolved_rows:
                        if previous is not None:
                            previous_contact_id = str(previous.get("contact_id") or "")
                            relation_changed = bool(
                                str(previous.get("status") or "") == "resolved"
                                and status not in WAPPI_TECHNICAL_LINK_STATUSES
                                and (status != "resolved" or contact_id != previous_contact_id)
                            )
                            if relation_changed:
                                contact_id = previous_contact_id
                                lead_ids = tuple(previous.get("lead_ids") or ())
                                status = "conflict"
                            elif status in WAPPI_TECHNICAL_LINK_STATUSES:
                                contact_id = previous_contact_id
                                lead_ids = tuple(previous.get("lead_ids") or ())
                        status_by_chat[chat_id] = _widget_linkage_status(status, lead_ids)
                        safe_result = {"status": status, "contact_id": contact_id, "lead_ids": lead_ids}
                        con.execute(
                            """
                            INSERT INTO wappi_amo_links
                              (channel, profile_id, chat_id, contact_id, lead_ids_json, status, checked_at,
                               response_sha256, resolution_source, last_timestamp, matched_points)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                            ON CONFLICT(channel, profile_id, chat_id) DO UPDATE SET
                              contact_id = excluded.contact_id,
                              lead_ids_json = excluded.lead_ids_json,
                              status = excluded.status,
                              checked_at = excluded.checked_at,
                              response_sha256 = excluded.response_sha256,
                              resolution_source = excluded.resolution_source,
                              last_timestamp = MAX(wappi_amo_links.last_timestamp, excluded.last_timestamp),
                              matched_points = excluded.matched_points
                            """,
                            (
                                profile.channel,
                                profile.profile_id,
                                chat_id,
                                contact_id or None,
                                json.dumps(lead_ids, separators=(",", ":")),
                                status,
                                datetime.now(timezone.utc).isoformat(),
                                stable_digest(safe_result),
                                "wappi_widget_conflict" if status == "conflict" else "wappi_widget",
                                last_timestamp,
                            ),
                        )
                        con.commit()
                        operations[status] += 1
            profile_status_counts = Counter(status_by_chat.values())
            counts.update(profile_status_counts)
            resolved_count = sum(profile_status_counts[name] for name in WAPPI_RESOLVED_LINK_STATUSES)
            linkage_complete = bool(accounting_complete and resolved_count == len(personal_dialogs))
            if profile_status_counts["request_limit"]:
                limit_hit = True
            profile_reports[profile_key] = {
                "total_count": total_count,
                "unique_catalogued": len(catalog),
                "personal_catalogued": len(personal_dialogs),
                "non_personal": len(catalog) - len(personal_dialogs),
                "chat_id_missing": len(missing_chat_ids),
                "catalog_passes": passes,
                "catalog_error": catalog_error or None,
                "accounting_complete": accounting_complete,
                "linkage_complete": linkage_complete,
                "linkage_status_counts": dict(sorted(profile_status_counts.items())),
            }
    accounting_complete = len(profile_reports) == len(profiles) and bool(profile_reports) and all(
        bool(report.get("accounting_complete")) for report in profile_reports.values()
    )
    linkage_complete = len(profile_reports) == len(profiles) and bool(profile_reports) and all(
        bool(report.get("linkage_complete")) for report in profile_reports.values()
    )
    technical_failures = sum(counts[name] for name in WAPPI_TECHNICAL_LINK_STATUSES)
    physical_requests = readonly_wappi_physical_request_count(client)
    personal_chats_seen = sum(
        int(report.get("personal_catalogued") or 0) for report in profile_reports.values()
    )
    return {
        "schema_version": WAPPI_WIDGET_LINK_SCHEMA,
        "requests": requests,
        "physical_requests": physical_requests,
        "personal_chats_seen": personal_chats_seen,
        "personal_chats_total": personal_chats_seen if accounting_complete else None,
        "counts": dict(sorted(counts.items())),
        "operations": dict(sorted(operations.items())),
        "profiles": profile_reports,
        "request_limit_hit": limit_hit,
        "accounting_complete": accounting_complete,
        "linkage_complete": linkage_complete,
        "complete": (
            accounting_complete
            and linkage_complete
            and not limit_hit
            and technical_failures == 0
        ),
    }


def load_wappi_widget_links(db_path: Path | None) -> Mapping[tuple[str, str, str], Mapping[str, Any]]:
    """Strictly read-only snapshot of the locally persisted `wappi_amo_links` cache
    (BLOK A2). Opens the file `mode=ro` with `PRAGMA query_only = ON` via the same
    `open_readonly_sqlite` helper this module already uses for the Timeline DB: no
    CREATE/ALTER/INSERT/commit is ever issued, so a caller such as the offline
    unmatched-link report can never mutate the cache file (mtime/size/hash stay
    fixed -- см. test_load_wappi_widget_links_is_strictly_read_only_and_leaves_file_untouched).

    `db_path is None` or a cache file that has not been created yet both mean
    "nothing collected so far" and yield an empty mapping -- the same state
    `collect_wappi_widget_links` starts from before its first run. A file that does
    exist but has no readable `wappi_amo_links` table (wrong path, empty/corrupt
    file, or a pre-migration schema) is a genuine diagnostic problem and raises
    instead of silently creating or migrating schema.
    """
    if db_path is None or not Path(db_path).exists():
        return {}
    resolved = Path(db_path)
    con: sqlite3.Connection | None = None
    try:
        con = open_readonly_sqlite(resolved)
        rows = con.execute(
            """
            SELECT channel, profile_id, chat_id, contact_id, lead_ids_json, status,
                   resolution_source, last_timestamp, matched_points
            FROM wappi_amo_links
            """
        ).fetchall()
    except sqlite3.DatabaseError as exc:
        raise ValueError(
            f"Wappi widget link cache at {resolved} has no readable wappi_amo_links "
            "table (missing table/column or not a SQLite file); read-only load "
            "never creates or migrates schema"
        ) from exc
    finally:
        if con is not None:
            con.close()
    result: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in rows:
        lead_ids = json.loads(str(row["lead_ids_json"] or "[]"))
        result[(str(row["channel"]), str(row["profile_id"]), str(row["chat_id"]))] = {
            "status": str(row["status"]),
            "contact_id": str(row["contact_id"] or ""),
            "lead_ids": tuple(str(item) for item in lead_ids),
            "resolution_source": str(row["resolution_source"] or ""),
            "last_timestamp": int(row["last_timestamp"] or 0),
            "matched_points": int(row["matched_points"] or 0),
        }
    return result


def summarize_wappi_widget_link_cache(
    widget_links: Mapping[tuple[str, str, str], Mapping[str, Any]],
    *,
    personal_chats_total: int | None,
) -> Mapping[str, Any]:
    """Summarize the persisted cache after all exact AMO bridges have run."""
    status_counts: Counter[str] = Counter()
    authority_counts: Counter[str] = Counter()
    for link in widget_links.values():
        status = str(link.get("status") or "")
        status_counts[_widget_linkage_status(status, tuple(link.get("lead_ids") or ()))] += 1
        if status == "resolved":
            authority_counts[str(link.get("resolution_source") or "unknown_source")] += 1
    link_rows_total = len(widget_links)
    measured = personal_chats_total is not None
    cache_row_count_delta = (
        link_rows_total - int(personal_chats_total)
        if measured
        else None
    )
    return {
        "scope": "amo_entity_link_not_timeline_customer_attribution",
        "link_rows_total": link_rows_total,
        "personal_chats_total": int(personal_chats_total) if measured else None,
        "cache_row_count_delta": cache_row_count_delta,
        "counts": dict(sorted(status_counts.items())),
        "amo_entity_links_by_authority": dict(sorted(authority_counts.items())),
    }


def _wappi_talk_bridge_warnings(
    *,
    post_bridge_cache: Mapping[str, Any],
    talk_report: Mapping[str, Any],
) -> tuple[str, ...]:
    warnings: list[str] = []
    candidates = int(post_bridge_cache.get("counts", {}).get("candidate") or 0)
    if (
        talk_report.get("setup_error")
        or talk_report.get("setup_unavailable")
        or (candidates and not talk_report)
    ):
        warnings.append("wappi_amo_talk:bridge_unavailable")
    if int(talk_report.get("lookup_error") or 0) or int(
        talk_report.get("invalid_response") or 0
    ):
        warnings.append("wappi_amo_talk:bridge_degraded")
    return tuple(warnings)


def enrich_wappi_widget_links_from_timeline_amo_events(
    *,
    timeline_db: Path,
    widget_link_db: Path,
    tenant_id: str = "foton",
) -> Mapping[str, Any]:
    """Record timing-based AMO candidates without promoting them to identity truth."""
    tenant = normalize_key(tenant_id, "tenant_id")
    with sqlite3.connect(widget_link_db) as link_con:
        _ensure_wappi_widget_link_schema(link_con)
        missing = {
            (str(channel), str(profile_id), str(chat_id))
            for channel, profile_id, chat_id in link_con.execute(
                "SELECT channel, profile_id, chat_id FROM wappi_amo_links WHERE status = 'missing'"
            )
        }
        if not missing:
            return {
                "missing_before": 0,
                "candidates": 0,
                "ambiguous": 0,
                "cross_chat_ambiguous": 0,
                "insufficient": 0,
            }

        points_by_chat: dict[tuple[str, str, str], list[tuple[int, str, str]]] = defaultdict(list)
        amo_by_origin_direction: dict[tuple[str, str], list[tuple[int, str, str, str, str]]] = defaultdict(list)
        with open_readonly_sqlite(timeline_db) as timeline_con:
            for row in timeline_con.execute(
                """
                SELECT source_system,
                       json_extract(record_json, '$.metadata.profile_id') AS profile_id,
                       json_extract(record_json, '$.metadata.chat_id') AS chat_id,
                       direction,
                       CAST(strftime('%s', event_at) AS INTEGER) AS event_ts,
                       COALESCE(json_extract(record_json, '$.metadata.message_id'), source_id) AS message_key
                FROM timeline_events
                WHERE tenant_id = ?
                  AND source_system IN ('wappi_telegram', 'wappi_max')
                  AND superseded_by IS NULL
                  AND json_valid(record_json)
                """,
                (tenant,),
            ):
                channel = "telegram" if str(row[0]) == "wappi_telegram" else "max"
                key = (channel, str(row[1] or ""), str(row[2] or ""))
                event_ts = _safe_int(row[4])
                message_key = str(row[5] or "").strip()
                if key in missing and event_ts > 0 and message_key and str(row[3]) in {"inbound", "outbound"}:
                    points_by_chat[key].append((event_ts, str(row[3]), message_key))

            for row in timeline_con.execute(
                """
                SELECT json_extract(record_json, '$.record.payload.value_after[0].message.origin') AS origin,
                       json_extract(record_json, '$.record.payload.type') AS event_type,
                       json_extract(record_json, '$.record.payload.created_at') AS event_ts,
                       json_extract(record_json, '$.record.payload.value_after[0].message.talk_id') AS talk_id,
                       json_extract(record_json, '$.record.payload._embedded.entity.linked_talk_contact_id') AS contact_id,
                       json_extract(record_json, '$.record.payload.entity_id') AS lead_id,
                       json_extract(record_json, '$.record.payload.id') AS event_id,
                       json_extract(record_json, '$.record.payload.entity_type') AS entity_type
                FROM timeline_events
                WHERE tenant_id = ?
                  AND source_system = 'amocrm_event'
                  AND superseded_by IS NULL
                  AND json_valid(record_json)
                  AND json_extract(record_json, '$.record.payload.type')
                      IN ('incoming_chat_message', 'outgoing_chat_message')
                  AND json_extract(record_json, '$.record.payload.entity_type') = 'lead'
                """,
                (tenant,),
            ):
                direction = "inbound" if str(row[1]) == "incoming_chat_message" else "outbound"
                event_ts = _safe_int(row[2])
                talk_id, contact_id, lead_id, event_id = (str(row[index] or "") for index in range(3, 7))
                if str(row[0]) and event_ts > 0 and talk_id and contact_id and lead_id and event_id:
                    amo_by_origin_direction[(str(row[0]), direction)].append(
                        (event_ts, talk_id, contact_id, lead_id, event_id)
                    )

        event_timestamps: dict[tuple[str, str], list[int]] = {}
        for key, rows in amo_by_origin_direction.items():
            rows.sort(key=lambda item: item[0])
            event_timestamps[key] = [item[0] for item in rows]

        counts: Counter[str] = Counter()
        now = datetime.now(timezone.utc).isoformat()
        confirmed_by_chat: dict[tuple[str, str, str], tuple[str, str, tuple[str, ...], int]] = {}
        for key in sorted(missing):
            channel, _profile_id, _chat_id = key
            source_system = SOURCE_SYSTEM_BY_CHANNEL[channel]
            origin = WAPPI_AMO_EVENT_ORIGIN[source_system]
            candidate_edges: dict[str, list[tuple[int, int, str, str, str]]] = defaultdict(list)
            unique_points: dict[str, tuple[int, str]] = {}
            conflicting_message_keys: set[str] = set()
            for point_ts, direction, message_key in points_by_chat.get(key, ()):
                previous_point = unique_points.get(message_key)
                if previous_point is not None and previous_point != (point_ts, direction):
                    conflicting_message_keys.add(message_key)
                    continue
                unique_points[message_key] = (point_ts, direction)
            points = sorted(
                point
                for message_key, point in unique_points.items()
                if message_key not in conflicting_message_keys
            )
            for point_index, (point_ts, direction) in enumerate(points):
                event_key = (origin, direction)
                rows = amo_by_origin_direction.get(event_key, ())
                timestamps = event_timestamps.get(event_key, ())
                left = bisect_left(timestamps, point_ts - WAPPI_AMO_EVENT_MATCH_WINDOW_SEC)
                right = bisect_right(timestamps, point_ts + WAPPI_AMO_EVENT_MATCH_WINDOW_SEC)
                for event_ts, talk_id, contact_id, lead_id, event_id in rows[left:right]:
                    candidate_edges[talk_id].append(
                        (abs(event_ts - point_ts), point_index, event_id, contact_id, lead_id)
                    )

            confirmed: list[tuple[str, str, tuple[str, ...], int]] = []
            for talk_id, edges in candidate_edges.items():
                used_points: set[int] = set()
                used_events: set[str] = set()
                selected: list[tuple[int, int, str, str, str]] = []
                for edge in sorted(edges):
                    if edge[1] in used_points or edge[2] in used_events:
                        continue
                    selected.append(edge)
                    used_points.add(edge[1])
                    used_events.add(edge[2])
                contacts = {edge[3] for edge in selected}
                lead_ids = tuple(sorted({edge[4] for edge in selected}))
                if len(selected) >= WAPPI_AMO_EVENT_MIN_MATCHES and len(contacts) == 1 and lead_ids:
                    confirmed.append((talk_id, next(iter(contacts)), lead_ids, len(selected)))

            if len(confirmed) != 1:
                counts["ambiguous" if len(confirmed) > 1 else "insufficient"] += 1
                continue
            confirmed_by_chat[key] = confirmed[0]

        chats_by_talk: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
        for key, (talk_id, _contact_id, _lead_ids, _matched_points) in confirmed_by_chat.items():
            chats_by_talk[talk_id].add(key)

        for key, (talk_id, contact_id, lead_ids, matched_points) in sorted(confirmed_by_chat.items()):
            if len(chats_by_talk[talk_id]) != 1:
                counts["cross_chat_ambiguous"] += 1
                continue
            channel, profile_id, chat_id = key
            safe_result = {
                "status": "candidate",
                "contact_id": contact_id,
                "lead_ids": lead_ids,
                "resolution_source": "amo_event_sequence_candidate",
                "matched_points": matched_points,
            }
            link_con.execute(
                """
                UPDATE wappi_amo_links
                SET contact_id = ?, lead_ids_json = ?, status = 'candidate', checked_at = ?,
                    response_sha256 = ?, resolution_source = 'amo_event_sequence_candidate', matched_points = ?,
                    amo_talk_id = ?, amo_chat_id = ''
                WHERE channel = ? AND profile_id = ? AND chat_id = ? AND status = 'missing'
                """,
                (
                    contact_id,
                    json.dumps(lead_ids, separators=(",", ":")),
                    now,
                    stable_digest(safe_result),
                    matched_points,
                    talk_id,
                    channel,
                    profile_id,
                    chat_id,
                ),
            )
            counts["candidates"] += 1
        link_con.commit()
    return {
        "missing_before": len(missing),
        "candidates": counts["candidates"],
        "ambiguous": counts["ambiguous"],
        "cross_chat_ambiguous": counts["cross_chat_ambiguous"],
        "insufficient": counts["insufficient"],
    }


def _amo_talk_identity(payload: Mapping[str, Any], *, expected_talk_id: str) -> tuple[str, str, str, str] | None:
    talk_id, contact_id, chat_id, entity_id = (
        str(payload.get(name) or "").strip()
        for name in ("talk_id", "contact_id", "chat_id", "entity_id")
    )
    if (
        talk_id != expected_talk_id
        or not all(value.isdigit() and int(value) > 0 for value in (talk_id, contact_id, entity_id))
        or not chat_id
        or str(payload.get("entity_type") or "") != "lead"
    ):
        return None
    embedded = payload.get("_embedded") if isinstance(payload.get("_embedded"), Mapping) else {}
    embedded_ids = lambda name: {  # noqa: E731 - local one-use validator keeps the gate compact.
        str(item.get("id") or "").strip() for item in embedded.get(name) or () if isinstance(item, Mapping)
    }
    if embedded_ids("contacts") != {contact_id} or embedded_ids("leads") != {entity_id}:
        return None
    return talk_id, chat_id, contact_id, entity_id


def confirm_wappi_widget_candidates_from_amo_talks(*, widget_link_db: Path, amo_client: Any) -> Mapping[str, int]:
    """Promote timing candidates only after exact read-only AMO Talk confirmation."""
    counts: Counter[str] = Counter()
    with sqlite3.connect(widget_link_db) as con:
        _ensure_wappi_widget_link_schema(con)
        candidates = con.execute(
            "SELECT channel, profile_id, chat_id, contact_id, lead_ids_json, amo_talk_id "
            "FROM wappi_amo_links WHERE status='candidate' "
            "AND resolution_source='amo_event_sequence_candidate' AND amo_talk_id!=''"
        ).fetchall()
        talk_owners: dict[str, int] = Counter(str(row[5]) for row in candidates)

        def mark_conflict(channel: str, profile_id: str, chat_id: str, talk_id: str) -> None:
            con.execute(
                "UPDATE wappi_amo_links SET status='conflict', checked_at=?, "
                "resolution_source='amo_talk_conflict', response_sha256=? "
                "WHERE channel=? AND profile_id=? AND chat_id=? AND status='candidate' AND amo_talk_id=?",
                (
                    datetime.now(timezone.utc).isoformat(),
                    stable_digest({"status": "conflict", "talk_id": talk_id}),
                    channel,
                    profile_id,
                    chat_id,
                    talk_id,
                ),
            )

        for channel, profile_id, wappi_chat_id, candidate_contact, candidate_leads_json, talk_id in candidates:
            talk_id = str(talk_id)
            if talk_owners[talk_id] != 1:
                mark_conflict(str(channel), str(profile_id), str(wappi_chat_id), talk_id)
                counts["cross_chat_conflict"] += 1
                continue
            try:
                payload = amo_client.amo_api_get(path=f"/api/v4/talks/{talk_id}", params={}, limit=1)
            except Exception:  # noqa: BLE001 - fail closed; no secret-bearing exception is persisted.
                counts["lookup_error"] += 1
                continue
            if not isinstance(payload, Mapping):
                counts["invalid_response"] += 1
                continue
            identity = _amo_talk_identity(payload, expected_talk_id=talk_id)
            if identity is None:
                counts["invalid_response"] += 1
                continue
            _confirmed_talk_id, amo_chat_id, contact_id, lead_id = identity
            try:
                raw_candidate_leads = json.loads(str(candidate_leads_json or "[]"))
                candidate_leads = {str(item) for item in raw_candidate_leads}
            except (TypeError, json.JSONDecodeError):
                counts["invalid_response"] += 1
                continue
            if str(candidate_contact or "") != contact_id or candidate_leads != {lead_id}:
                mark_conflict(str(channel), str(profile_id), str(wappi_chat_id), talk_id)
                counts["identity_conflict"] += 1
                continue
            safe_result = {"status": "resolved", "contact_id": contact_id, "lead_ids": (lead_id,),
                           "talk_id": talk_id, "chat_id": amo_chat_id}
            con.execute(
                "UPDATE wappi_amo_links SET contact_id=?, lead_ids_json=?, status='resolved', checked_at=?, "
                "response_sha256=?, resolution_source='amo_talk_authoritative', amo_chat_id=? "
                "WHERE channel=? AND profile_id=? AND chat_id=? AND status='candidate' AND amo_talk_id=?",
                (
                    contact_id,
                    json.dumps((lead_id,), separators=(",", ":")),
                    datetime.now(timezone.utc).isoformat(),
                    stable_digest(safe_result),
                    amo_chat_id,
                    channel,
                    profile_id,
                    wappi_chat_id,
                    talk_id,
                ),
            )
            counts["resolved"] += 1
        con.commit()
    return {name: len(candidates) if name == "candidates" else counts[name] for name in (
        "candidates", "resolved", "identity_conflict", "cross_chat_conflict", "invalid_response", "lookup_error"
    )}


def _build_safe_amo_talk_client(env_file: Path) -> Any:
    from mango_mvp.existing_clients.amo_step1_snapshot import AmoMcpClient, read_mcp_env

    config = read_mcp_env(env_file)
    parsed = url_parse.urlparse(config.connector_url)
    if parsed.scheme != "https" or (parsed.hostname or "").casefold() != "api.fotonai.online":
        raise ValueError("AMO Talk reads require the HTTPS api.fotonai.online proxy")
    return AmoMcpClient(config)


def hydrate_wappi_widget_contacts(
    *,
    timeline_db: Path,
    allowed_root: Path,
    widget_links: Mapping[tuple[str, str, str], Mapping[str, Any]],
    amo_mcp_env_file: Path | None,
    tenant_id: str = "foton",
    workers: int = 4,
    amo_client: Any = None,
) -> Mapping[str, Any]:
    """Fetch only widget-proven AMO contacts that are absent from Timeline."""
    from types import SimpleNamespace

    from mango_mvp.customer_timeline.amo_incremental import load_amo_link_index, normalize_cards_source
    from mango_mvp.customer_timeline.ingestion import AmoSnapshotNormalizer
    from mango_mvp.existing_clients.amo_step1_snapshot import AmoMcpClient, embedded_items, read_mcp_env

    db_path = guard_customer_timeline_output_path(timeline_db, Path(allowed_root))
    wanted = {
        str(item.get("contact_id") or "").strip()
        for item in widget_links.values()
        if str(item.get("status") or "") == "resolved" and str(item.get("contact_id") or "").strip()
    }
    existing: set[str] = set()
    with open_readonly_sqlite(db_path) as con:
        if sqlite_table_exists(con, "identity_links"):
            existing.update(
                str(row[0])
                for row in con.execute(
                    "SELECT DISTINCT link_value FROM identity_links WHERE tenant_id = ? AND link_type = 'amo_contact_id'",
                    (tenant_id,),
                )
            )
    missing = tuple(sorted(wanted - existing))
    if not missing:
        return {"requested": 0, "fetched": 0, "normalized": 0, "fetch_errors": 0, "write_status_counts": {}}
    if amo_client is None:
        if amo_mcp_env_file is None:
            raise ValueError("AMO MCP env file is required to hydrate Wappi contacts")
        amo_client = AmoMcpClient(read_mcp_env(amo_mcp_env_file))

    batches = chunks(missing, 50)

    def fetch_contacts(batch: Sequence[str]) -> tuple[Mapping[str, Any], ...]:
        try:
            payload = amo_client.amo_api_get(
                path="contacts",
                params={"filter[id][]": list(batch), "with": "leads"},
                limit=len(batch),
            )
        except Exception:  # noqa: BLE001 - aggregate only; never expose raw AMO payloads.
            return ()
        requested = set(batch)
        return tuple(
            contact
            for contact in embedded_items(payload, "contacts")
            if str(contact.get("id") or "") in requested
        )

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        fetched_batches = tuple(pool.map(fetch_contacts, batches))
    fetched_by_id = {
        str(contact.get("id")): contact
        for batch in fetched_batches
        for contact in batch
    }
    fallback_ids = tuple(contact_id for contact_id in missing if contact_id not in fetched_by_id)

    def fetch_contact(contact_id: str) -> Mapping[str, Any] | None:
        if not contact_id.isdigit():
            return None
        try:
            contact = amo_client.amo_api_get(
                path=f"contacts/{int(contact_id)}",
                params={"with": "leads"},
                limit=1,
            )
        except Exception:  # noqa: BLE001 - aggregate only; never expose raw AMO payloads.
            return None
        return contact if str(contact.get("id") or "") == contact_id else None

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        fallback_contacts = tuple(pool.map(fetch_contact, fallback_ids))
    fetched_by_id.update(
        (str(contact.get("id")), contact)
        for contact in fallback_contacts
        if contact is not None
    )
    fetched = tuple(fetched_by_id[contact_id] for contact_id in missing if contact_id in fetched_by_id)
    link_index = load_amo_link_index(db_path, tenant_id=tenant_id)
    rows, normalization = normalize_cards_source(
        fetched,
        pages=1,
        page_cap_hit=False,
        path="contacts",
        entity_type="contact",
        cursor_name="amo_contacts_widget_hydrate",
        link_index=link_index,
        config=SimpleNamespace(max_pages=1),
    )
    observed_at = datetime.now(timezone.utc)
    records = tuple(
        TimelineSourceRecord(
            source_system="amo_contacts_widget_hydrate",
            source_ref=str(row["source_ref"]),
            payload=row,
            observed_at=observed_at,
        )
        for row in rows
    )
    with CustomerTimelineSQLiteStore(db_path, allowed_root=allowed_root) as store:
        report = TimelineImportService(store).import_records(
            records,
            normalizer=AmoSnapshotNormalizer(tenant_id=tenant_id),
            tenant_id=tenant_id,
            source_ref="amocrm:contacts:wappi_widget_hydrate",
            idempotency_key=stable_digest(sorted(missing)),
            dry_run=False,
            actor="wappi_widget_contact_hydrate",
        )
    return {
        "requested": len(missing),
        "batches": len(batches),
        "fallback_requested": len(fallback_ids),
        "fallback_fetched": sum(contact is not None for contact in fallback_contacts),
        "fetched": len(fetched),
        "normalized": len(rows),
        "fetch_errors": len(missing) - len(fetched),
        "normalization": {key: value for key, value in normalization.items() if not key.startswith("_")},
        "write_status_counts": dict(report.write_status_counts),
        "errors": len(report.errors),
    }


@dataclass(frozen=True)
class WappiChatResolution:
    status: str
    customer_id: Optional[str] = None
    opportunity_id: Optional[str] = None
    lead_id: str = ""
    lead_ids: Sequence[str] = field(default_factory=tuple)
    contact_id: str = ""
    expected_brand: str = ""
    reason: str = ""
    candidate_customer_ids: Sequence[str] = field(default_factory=tuple)
    pair_source: str = ""
    resolution_source: str = "draft_loop_pair"
    match_key: str = ""
    evidence: Mapping[str, Any] = field(default_factory=dict)

    @property
    def resolved(self) -> bool:
        return self.status == "resolved" and bool(self.customer_id)


@dataclass
class WappiFetchStats:
    chats_seen: int = 0
    chats_loaded: int = 0
    messages_seen: int = 0
    records_built: int = 0
    linked_by_pair: int = 0
    linked_by_timeline: int = 0
    linked_by_amo_auto: int = 0
    linked_by_amo_widget: int = 0
    linked_by_amo_talk: int = 0
    linked_by_provisional: int = 0
    pending_attribution: int = 0
    skipped_empty: int = 0
    skipped_bad_message: int = 0
    skipped_chat_id_missing: int = 0
    duplicate_chat_ids: int = 0
    duplicate_source_ids: int = 0
    requests: int = 0
    request_limit_hit: bool = False
    message_limit_hit: bool = False
    chat_limit_hit: bool = False
    pagination_drift_detected: bool = False
    chat_snapshot_drift_detected: bool = False
    message_page_drift_detected: bool = False
    checkpoint_network_error: bool = False
    checkpoint_no_progress: bool = False
    checkpoint_chats_skipped: int = 0
    checkpoint_chats_confirmed: int = 0
    incremental_chats_skipped: int = 0
    incremental_chats_new: int = 0
    incremental_chats_changed: int = 0
    incremental_chats_without_marker: int = 0
    incremental_chats_marker_regressed: int = 0
    full_history_audit: bool = False
    full_audit_clock_anomaly: bool = False
    resolution_status_counts: Counter[str] = field(default_factory=Counter)
    coverage_counts: Counter[str] = field(default_factory=Counter)
    amo_auto_status_counts: Counter[str] = field(default_factory=Counter)
    amo_auto_calls: int = 0
    personal_chats: int = 0
    widget_calls: int = 0
    widget_resolved_chats: int = 0
    widget_pending_chats: int = 0

    def to_json_dict(self) -> Mapping[str, Any]:
        return {
            "chats_seen": self.chats_seen,
            "chats_loaded": self.chats_loaded,
            "messages_seen": self.messages_seen,
            "records_built": self.records_built,
            "linked_by_pair": self.linked_by_pair,
            "linked_by_timeline": self.linked_by_timeline,
            "linked_by_amo_auto": self.linked_by_amo_auto,
            "linked_by_amo_widget": self.linked_by_amo_widget,
            "linked_by_amo_talk": self.linked_by_amo_talk,
            "linked_by_amo_event_sequence": 0,
            "linked_by_provisional": self.linked_by_provisional,
            "pending_attribution": self.pending_attribution,
            "skipped_empty": self.skipped_empty,
            "skipped_bad_message": self.skipped_bad_message,
            "skipped_chat_id_missing": self.skipped_chat_id_missing,
            "duplicate_chat_ids": self.duplicate_chat_ids,
            "duplicate_source_ids": self.duplicate_source_ids,
            "requests": self.requests,
            "request_limit_hit": self.request_limit_hit,
            "message_limit_hit": self.message_limit_hit,
            "chat_limit_hit": self.chat_limit_hit,
            "pagination_drift_detected": self.pagination_drift_detected,
            "chat_snapshot_drift_detected": self.chat_snapshot_drift_detected,
            "message_page_drift_detected": self.message_page_drift_detected,
            "checkpoint_network_error": self.checkpoint_network_error,
            "checkpoint_no_progress": self.checkpoint_no_progress,
            "checkpoint_chats_skipped": self.checkpoint_chats_skipped,
            "checkpoint_chats_confirmed": self.checkpoint_chats_confirmed,
            "incremental_chats_skipped": self.incremental_chats_skipped,
            "incremental_chats_new": self.incremental_chats_new,
            "incremental_chats_changed": self.incremental_chats_changed,
            "incremental_chats_without_marker": self.incremental_chats_without_marker,
            "incremental_chats_marker_regressed": self.incremental_chats_marker_regressed,
            "full_history_audit": self.full_history_audit,
            "full_audit_clock_anomaly": self.full_audit_clock_anomaly,
            "resolution_status_counts": dict(self.resolution_status_counts),
            "coverage_counts": dict(self.coverage_counts),
            "amo_auto_status_counts": dict(self.amo_auto_status_counts),
            "amo_auto_calls": self.amo_auto_calls,
            "personal_chats": self.personal_chats,
            "widget_calls": self.widget_calls,
            "widget_resolved_chats": self.widget_resolved_chats,
            "widget_pending_chats": self.widget_pending_chats,
        }


class WappiHistoryTimelineNormalizer:
    def __init__(self, *, tenant_id: str, source_system: str) -> None:
        self.tenant_id = normalize_key(tenant_id, "tenant_id")
        self.source_system = normalize_key(source_system, "source_system")
        if self.source_system not in SOURCE_SYSTEM_BY_CHANNEL.values():
            raise ValueError(f"unsupported Wappi source_system: {source_system!r}")

    def normalize(self, record: TimelineSourceRecord) -> TimelineNormalizedBatch:
        payload = record.payload
        if record.source_system != self.source_system:
            raise ValueError(f"record source_system does not match normalizer: {record.source_system}")
        if truthy(payload.get("allowed_for_bot")):
            raise ValueError("Wappi history must be loaded with allowed_for_bot=False")
        channel = normalize_wappi_channel(payload.get("channel"))
        brand = normalize_brand(payload.get("brand"))
        source_ref = require_text(payload.get("source_ref") or record.source_ref, "source_ref")
        message_id = require_text(payload.get("message_id") or payload.get("message_sha256"), "message_id")
        chat_id = require_text(payload.get("chat_id"), "chat_id")
        text = str(payload.get("text") or "").strip()
        event_at = parse_source_datetime(payload.get("event_at") or payload.get("timestamp_iso"), record.observed_at)
        event_time_status = str(payload.get("event_time_status") or "source_valid")
        resolution_status = str(payload.get("resolution_status") or "pending_attribution")
        resolved_customer_id = optional_text(payload.get("resolved_customer_id"))
        identity_authority = str(payload.get("identity_authority") or "")
        resolution_evidence = (
            dict(payload.get("resolution_evidence") or {})
            if isinstance(payload.get("resolution_evidence"), Mapping)
            else {}
        )
        if resolved_customer_id and resolution_status != "resolved":
            raise ValueError("resolved Wappi customer requires resolution_status=resolved")
        if resolved_customer_id and not identity_authority:
            raise ValueError("resolved Wappi customer requires identity_authority")
        if resolved_customer_id and identity_authority not in RESOLVED_MATCH_CLASS_BY_IDENTITY_AUTHORITY:
            raise ValueError(f"unsupported resolved Wappi identity_authority: {identity_authority!r}")
        resolved_match_class = RESOLVED_MATCH_CLASS_BY_IDENTITY_AUTHORITY.get(
            identity_authority,
            IdentityMatchClass.UNMATCHED,
        )
        is_provisional = identity_authority == "wappi_provisional"
        identity_confidence = 0.35 if is_provisional else 0.9
        direction = TimelineDirection.OUTBOUND if truthy(payload.get("from_me")) else TimelineDirection.INBOUND
        participant_role = "manager" if direction == TimelineDirection.OUTBOUND else "client"
        source_id = require_text(payload.get("timeline_source_id") or f"{payload.get('profile_id')}:{chat_id}:{message_id}", "source_id")
        event = TimelineEvent(
            tenant_id=self.tenant_id,
            customer_id=resolved_customer_id,
            opportunity_id=optional_text(payload.get("resolved_opportunity_id")),
            event_type=EVENT_TYPE_BY_CHANNEL[channel],
            event_at=event_at,
            source_system=self.source_system,
            source_id=source_id,
            source_ref=source_ref,
            direction=direction,
            participants=(TimelineParticipant(role=participant_role, ref=chat_id, channel=f"wappi_{channel}"),),
            actor_name=optional_text(payload.get("contact_name")),
            actor_ref=chat_id,
            subject=f"Wappi {channel} message",
            text_preview=compact_text(text, limit=240),
            summary=compact_text(text, limit=240),
            match_status=resolved_match_class if resolved_customer_id else IdentityMatchClass.UNMATCHED,
            confidence=identity_confidence if resolved_customer_id else 0.0,
            record={
                "message": scrub_timeline_persisted_json(
                    {
                        "channel": channel,
                        "brand": brand,
                        "profile_id": payload.get("profile_id"),
                        "chat_id": chat_id,
                        "message_id": message_id,
                        "direction": direction.value,
                        "text": text,
                        "allowed_for_bot": False,
                        "resolution_status": resolution_status,
                        "event_time_status": event_time_status,
                    }
                )
            },
            metadata={
                "source_system": self.source_system,
                "brand": brand,
                "profile_id": payload.get("profile_id"),
                "chat_id": chat_id,
                "message_id": message_id,
                "identity_authority": identity_authority,
                "lead_id": str(payload.get("lead_id") or ""),
                "lead_ids": tuple(str(item) for item in (payload.get("lead_ids") or ()) if str(item)),
                "contact_id": str(payload.get("contact_id") or ""),
                "match_key": str(payload.get("match_key") or ""),
                "allowed_for_bot_reason": "wappi_history_manager_only",
                "allowed_for_bot": False,
                "requires_manager_review": True,
                "pending_attribution": not bool(resolved_customer_id) or is_provisional,
                "resolution_reason": str(payload.get("resolution_reason") or ""),
                "resolution_evidence": scrub_timeline_persisted_json(resolution_evidence),
                "brand_context_authorized": payload.get("brand_context_authorized"),
                "event_time_status": event_time_status,
            },
            created_at=event_at,
        )
        if not resolved_customer_id:
            if truthy(payload.get("preserve_existing_event")):
                return TimelineNormalizedBatch(
                    source_record=record,
                    conflicts=(
                        pending_wappi_attribution_conflict(
                            self.tenant_id,
                            payload,
                            source_ref,
                            message_id=message_id,
                            resolution_status=resolution_status,
                        ),
                    ),
                )
            return TimelineNormalizedBatch(
                source_record=record,
                events=(event,),
                conflicts=(
                    pending_wappi_attribution_conflict(
                        self.tenant_id,
                        payload,
                        source_ref,
                        message_id=message_id,
                        resolution_status=resolution_status,
                    ),
                ),
            )
        link_value = f"wappi_{channel}:{payload.get('profile_id')}:{chat_id}"
        customers: tuple[CustomerIdentity, ...] = ()
        if is_provisional:
            provisional_source_ref = f"wappi_provisional:{self.source_system}:{payload.get('profile_id')}:{chat_id}"
            customers = (
                CustomerIdentity(
                    tenant_id=self.tenant_id,
                    customer_id=resolved_customer_id,
                    identity_status=IdentityStatus.PARTIAL,
                    display_name=optional_text(payload.get("contact_name")),
                    source_ref=provisional_source_ref,
                    first_seen_at=event_at if event_time_status == "source_valid" else None,
                    last_seen_at=event_at if event_time_status == "source_valid" else None,
                    touch_count=1,
                    summary={
                        "source_system": self.source_system,
                        "provisional_wappi_family": True,
                    },
                    metadata={
                        "provisional_wappi_family": True,
                        "brand": brand,
                        "profile_id": payload.get("profile_id"),
                    },
                    created_at=event_at,
                    updated_at=event_at,
                ),
            )
        link = IdentityLink(
            tenant_id=self.tenant_id,
            customer_id=resolved_customer_id,
            link_type="channel_session_id",
            link_value=link_value,
            source_system=self.source_system,
            source_ref=f"{self.source_system}:chat:{payload.get('profile_id')}:{chat_id}",
            match_class=resolved_match_class,
            confidence=identity_confidence,
            evidence={
                "identity_authority": identity_authority,
                "lead_id": str(payload.get("lead_id") or ""),
                "lead_ids": tuple(str(item) for item in (payload.get("lead_ids") or ()) if str(item)),
                "contact_id": str(payload.get("contact_id") or ""),
                "match_key": str(payload.get("match_key") or ""),
                "brand_context_authorized": payload.get("brand_context_authorized"),
                "resolution_evidence": scrub_timeline_persisted_json(resolution_evidence),
            },
            first_seen_at=event_at if event_time_status == "source_valid" else None,
            last_seen_at=event_at if event_time_status == "source_valid" else None,
        )
        chunks: tuple[BotContextChunk, ...] = ()
        if text and not is_provisional:
            chunks = (
                BotContextChunk(
                    tenant_id=self.tenant_id,
                    customer_id=resolved_customer_id,
                    opportunity_id=optional_text(payload.get("resolved_opportunity_id")),
                    event_id=event.event_id,
                    source_ref=source_ref,
                    source_system=self.source_system,
                    chunk_type="channel_message",
                    text=text,
                    summary=compact_text(text, limit=160),
                    event_at=event_at,
                    freshness_score=0.7,
                    relevance_tags=(f"wappi_{channel}", f"brand:{brand}", "manager_only"),
                    allowed_for_bot=False,
                    requires_manager_review=True,
                    metadata={
                        "brand": brand,
                        "channel": f"wappi_{channel}",
                        "allowed_for_bot_reason": "wappi_history_manager_only",
                        "brand_context_authorized": payload.get("brand_context_authorized"),
                        "event_time_status": event_time_status,
                    },
                    created_at=event_at,
                ),
            )
        return TimelineNormalizedBatch(
            source_record=record,
            customers=customers,
            identity_links=(link,),
            events=(event,),
            bot_context_chunks=chunks,
            conflicts=(
                pending_wappi_attribution_conflict(
                    self.tenant_id,
                    payload,
                    source_ref,
                    message_id=message_id,
                    resolution_status="provisional_wappi_family",
                ),
            )
            if is_provisional
            else (),
        )


def quarantine_conflicting_wappi_events(
    store: CustomerTimelineSQLiteStore,
    *,
    tenant_id: str,
    conflicts: Sequence[tuple[str, str, str, str]],
    actor: str,
) -> Mapping[str, int]:
    counts: Counter[str] = Counter()
    conflicted_customers: set[str] = set()
    for source_system, source_id, reason, customer_id in sorted(set(conflicts)):
        conflicted_customers.add(customer_id)
        row = store._con.execute(
            """
            SELECT event_id, record_json, record_hash
            FROM timeline_events
            WHERE tenant_id = ? AND source_system = ? AND source_id = ?
              AND superseded_by IS NULL
            """,
            (tenant_id, source_system, source_id),
        ).fetchone()
        if row is None:
            continue
        payload = json.loads(str(row["record_json"] or "{}"))
        metadata = dict(payload.get("metadata") or {})
        metadata.update(
            {
                "allowed_for_bot": False,
                "requires_manager_review": True,
                "pending_attribution": True,
                "brand_context_authorized": False,
                "resolution_reason": reason,
            }
        )
        payload["metadata"] = metadata
        safe_payload = scrub_timeline_persisted_json(payload)
        after_hash = stable_digest(safe_payload)
        if after_hash != str(row["record_hash"]):
            store._con.execute(
                "UPDATE timeline_events SET record_json = ?, record_hash = ? WHERE event_id = ?",
                (json_dumps(safe_payload), after_hash, row["event_id"]),
            )
            store._append_audit_log(
                tenant_id=tenant_id,
                action="wappi_identity_conflict_quarantined",
                entity_type="timeline_event",
                entity_id=str(row["event_id"]),
                actor=actor,
                ingestion_run_id=None,
                before_hash=str(row["record_hash"]),
                after_hash=after_hash,
                metadata={"reason": reason, "source_system": source_system},
                now=store._now(),
            )
            counts["existing_event_quarantined"] += 1
        counts["bot_context_chunks_revoked"] += store.revoke_bot_context_chunks_for_event(
            tenant_id,
            event_id=str(row["event_id"]),
            source_system=source_system,
            reason="wappi_identity_conflict",
            actor=actor,
        )
    for customer_id in sorted(conflicted_customers):
        rows = store._con.execute(
            """
            SELECT chunk_id, record_json, record_hash
            FROM bot_context_chunks
            WHERE tenant_id = ? AND customer_id = ? AND chunk_type = 'bot_safe_summary'
              AND superseded_by IS NULL
            """,
            (tenant_id, customer_id),
        ).fetchall()
        retired_ids: list[str] = []
        for row in rows:
            payload = json.loads(str(row["record_json"] or "{}"))
            metadata = dict(payload.get("metadata") or {})
            metadata["retired_reason"] = "wappi_identity_conflict"
            payload.update(
                {
                    "allowed_for_bot": False,
                    "requires_manager_review": True,
                    "metadata": metadata,
                }
            )
            safe_payload = scrub_timeline_persisted_json(payload)
            after_hash = stable_digest(safe_payload)
            store._con.execute(
                """
                UPDATE bot_context_chunks
                SET allowed_for_bot = 0, requires_manager_review = 1,
                    superseded_by = ?, record_json = ?, record_hash = ?
                WHERE chunk_id = ?
                """,
                (str(row["chunk_id"]), json_dumps(safe_payload), after_hash, row["chunk_id"]),
            )
            store._append_audit_log(
                tenant_id=tenant_id,
                action="bot_safe_summary_revoked_for_wappi_conflict",
                entity_type="bot_context_chunk",
                entity_id=str(row["chunk_id"]),
                actor=actor,
                ingestion_run_id=None,
                before_hash=str(row["record_hash"]),
                after_hash=after_hash,
                metadata={"customer_id": customer_id},
                now=store._now(),
            )
            retired_ids.append(str(row["chunk_id"]))
        if retired_ids:
            store._delete_bot_context_fts_for_chunk_ids(retired_ids)
            counts["bot_safe_summaries_revoked"] += len(retired_ids)
    return dict(counts)


def run_wappi_history_import(
    config: WappiHistoryImportConfig,
    *,
    client: WappiHistoryClient | None = None,
    amo_auto_resolver: AmoAutoResolver | None = None,
    amo_talk_client: Any = None,
) -> Mapping[str, Any]:
    code_identity_start = dict(build_draft_loop_code_identity())
    code_root = Path(str(code_identity_start.get("code_root") or Path(__file__).resolve().parents[3]))
    input_hashes_start = {
        "importer": file_sha256(Path(__file__)),
        "phase1_config": file_sha256(config.phase1_config),
        "pairs_file": file_sha256(config.pairs_file),
        "auto_pairs_file": file_sha256(config.auto_pairs_file),
        "shared_phone_stoplist": file_sha256(config.shared_phone_stoplist),
    }
    worktree_start = git_worktree_provenance(code_root)
    db_identity_start = timeline_db_identity(config.timeline_db)
    db_identity_validation_base = db_identity_start
    phase1 = AmoWappiPhase1Config.from_file(config.phase1_config)
    profiles = profiles_from_phase1_config(phase1)
    client_was_provided = client is not None
    if client is None:
        client = build_readonly_wappi_client(
            config.env_file,
            request_limit_total=config.limits.request_limit_total,
        )
    assert_readonly_wappi_client(client)
    widget_crm_id = str(os.getenv("AMO_WAPPI_CRM_ID") or "").strip()
    widget_profiles: dict[tuple[str, str], Mapping[str, Any]] = {}
    if widget_crm_id and isinstance(client, WappiPhase1Client):
        for item in client.list_all_profiles():
            profile_id = str(item.get("profile_id") or "").strip()
            platform = str(item.get("channel") or item.get("platform") or "").strip().casefold()
            channel = "telegram" if platform in {"tg", "telegram"} else "max" if platform == "max" else ""
            if profile_id and channel:
                widget_profiles[(channel, profile_id)] = item
    widget_setup_errors: list[str] = []
    if config.require_widget_linkage:
        if not widget_crm_id:
            widget_setup_errors.append("wappi_amo_widget:crm_id_missing")
        if not isinstance(client, WappiPhase1Client):
            widget_setup_errors.append("wappi_amo_widget:unsupported_client")
        missing_profiles = sorted(
            (profile.channel, profile.profile_id)
            for profile in profiles
            if (profile.channel, profile.profile_id) not in widget_profiles
        )
        widget_setup_errors.extend(
            f"wappi_amo_widget:profile_missing:{channel}:{profile_id}"
            for channel, profile_id in missing_profiles
        )
    widget_link_report: Mapping[str, Any] = {}
    widget_event_link_report: Mapping[str, Any] = {}
    widget_talk_link_report: Mapping[str, Any] = {}
    widget_contact_hydrate_report: Mapping[str, Any] = {}
    if config.widget_link_db is not None:
        if not config.refresh_widget_links and not config.widget_link_db.exists():
            widget_setup_errors.append("wappi_amo_widget:reuse_link_db_missing")
        elif not config.refresh_widget_links:
            widget_link_report = {
                "schema_version": WAPPI_WIDGET_LINK_SCHEMA,
                "complete": True,
                "reused": True,
            }
        elif not widget_crm_id or not isinstance(client, WappiPhase1Client):
            widget_setup_errors.append("wappi_amo_widget:link_db_setup_unavailable")
        else:
            widget_link_report = collect_wappi_widget_links(
                client=client,
                profiles=profiles,
                runtime_profiles=widget_profiles,
                crm_id=widget_crm_id,
                db_path=config.widget_link_db,
                limits=config.limits,
                force_recheck=config.widget_coverage_only,
            )
            if config.require_widget_linkage and not widget_link_report.get("complete"):
                widget_setup_errors.append("wappi_amo_widget:link_db_collection_incomplete")
        if config.widget_coverage_only:
            accounting_complete = bool(widget_link_report.get("accounting_complete"))
            linkage_complete = bool(widget_link_report.get("linkage_complete"))
            if not accounting_complete:
                widget_setup_errors.append("wappi_amo_widget:accounting_incomplete")
            if not linkage_complete:
                widget_setup_errors.append("wappi_amo_widget:linkage_incomplete")
            safety = {
                "network_calls": True,
                "wappi_transport": "DefaultDenyTransport",
                "wappi_read_only_methods": ["GET", "POST Wappi AMO contact lookup"],
                "read_messages": False,
                "write_customer_timeline": False,
                "send_messenger": False,
                "write_crm": False,
                "write_tallanto": False,
                "blocked_live_actions": blocked_live_actions(),
            }
            validation_ok = not widget_setup_errors and accounting_complete and linkage_complete
            return {
                "schema_version": WAPPI_HISTORY_IMPORT_SCHEMA_VERSION,
                "mode": "widget_coverage_only",
                "dry_run": True,
                "validation_ok": validation_ok,
                "limit_hits": sorted(set(widget_setup_errors)),
                "provenance": {
                    "code_root": code_identity_start.get("code_root"),
                    "git_sha": code_identity_start.get("git_sha"),
                    "worktree": git_worktree_provenance(code_root),
                    "input_hashes": input_hashes_start,
                    "timeline_db": db_identity_start,
                },
                "summary": {
                    "tenant_id": config.tenant_id,
                    "profiles": len(profiles),
                    "records_built": 0,
                    "messages_read": 0,
                    "writes_applied": 0,
                    "amo_widget_link_map": widget_link_report,
                },
                "safety": safety,
            }
        if config.widget_link_db.exists():
            widget_event_link_report = enrich_wappi_widget_links_from_timeline_amo_events(
                timeline_db=config.timeline_db,
                widget_link_db=config.widget_link_db,
                tenant_id=config.tenant_id,
            )
            if amo_talk_client is not None or config.amo_mcp_env_file is not None:
                try:
                    talk_client = amo_talk_client or (
                        _build_safe_amo_talk_client(config.amo_mcp_env_file)
                        if config.amo_mcp_env_file is not None
                        else None
                    )
                    widget_talk_link_report = (
                        confirm_wappi_widget_candidates_from_amo_talks(
                            widget_link_db=config.widget_link_db,
                            amo_client=talk_client,
                        )
                        if talk_client is not None
                        else {"setup_unavailable": 1}
                    )
                except Exception:  # noqa: BLE001 - report only a safe class, never connector secrets.
                    widget_talk_link_report = {"setup_error": 1}
    widget_links = load_wappi_widget_links(config.widget_link_db)
    widget_link_cache_after_bridges = summarize_wappi_widget_link_cache(
        widget_links,
        personal_chats_total=(
            int(widget_link_report["personal_chats_total"])
            if widget_link_report.get("accounting_complete")
            and "personal_chats_total" in widget_link_report
            else None
        ),
    )
    if config.apply and widget_links and not widget_setup_errors:
        widget_contact_hydrate_report = hydrate_wappi_widget_contacts(
            timeline_db=config.timeline_db,
            allowed_root=config.allowed_root,
            widget_links=widget_links,
            amo_mcp_env_file=config.amo_mcp_env_file,
            tenant_id=config.tenant_id,
        )
        # The hydrate step is our own audited staging write; detect drift only after it.
        db_identity_validation_base = timeline_db_identity(config.timeline_db)
    if not client_was_provided and config.widget_link_db is not None:
        client = build_readonly_wappi_client(
            config.env_file,
            request_limit_total=config.limits.request_limit_total,
        )
        assert_readonly_wappi_client(client)
    pairs = load_wappi_pairs(config.pairs_file, config.auto_pairs_file)
    local_phone_stoplist, local_phone_stoplist_error = (
        load_phone_stoplist(config.shared_phone_stoplist)
        if config.shared_phone_stoplist is not None
        else (set(), "shared_phone_stoplist_unavailable")
    )
    if amo_auto_resolver is None and config.amo_auto_resolver_enabled:
        if config.amo_mcp_env_file is None or config.shared_phone_stoplist is None:
            raise ValueError("AMO auto resolver requires amo_mcp_env_file and shared_phone_stoplist.")
        amo_auto_resolver = build_amo_auto_resolver(
            amo_mcp_env_file=config.amo_mcp_env_file,
            shared_phone_stoplist=config.shared_phone_stoplist,
            user_agent="mango-wappi-history-auto-resolver/1.0",
            require_known_brand=True,
        )
    resolver = WappiPairCustomerResolver.from_store(
        config.timeline_db,
        tenant_id=config.tenant_id,
        pairs=pairs,
        amo_auto_resolver=amo_auto_resolver,
        widget_client=client if widget_profiles else None,
        widget_crm_id=widget_crm_id,
        widget_profiles=widget_profiles,
        widget_links=widget_links,
        widget_required=config.require_widget_linkage,
        shared_phone_stoplist=local_phone_stoplist,
        shared_phone_stoplist_error=local_phone_stoplist_error,
    )
    widget_prime_report = resolver.prime_widget_chat_resolutions(profiles)
    message_identity_prime_report = resolver.prime_existing_message_identity_resolutions(profiles)
    provisional_prime_report = resolver.prime_provisional_chat_resolutions(profiles)
    checkpoint_state: Mapping[str, Any] = {}
    next_checkpoint: Optional[dict[str, Any]] = None
    if config.checkpoint_dir is not None:
        next_checkpoint = {}
        current_timeline_state = wappi_timeline_state(
            config.timeline_db, tenant_id=config.tenant_id, profiles=profiles
        )
        checkpoint_state = {
            "profiles": usable_wappi_checkpoint_profiles(
                load_wappi_history_checkpoint(config.checkpoint_dir),
                db_row_counts={key: int(value["rows"]) for key, value in current_timeline_state.items()},
                db_source_digests={
                    key: str(value["source_digest"]) for key, value in current_timeline_state.items()
                },
            )
        }
    records, fetch_stats_by_profile = fetch_wappi_history_records(
        client=client,
        profiles=profiles,
        resolver=resolver,
        limits=config.limits,
        tenant_id=config.tenant_id,
        checkpoint=checkpoint_state,
        next_checkpoint=next_checkpoint,
    )
    network_source_ids = {
        str(record.payload.get("timeline_source_id") or "")
        for record in records
        if str(record.payload.get("timeline_source_id") or "")
    }
    local_relink_records = load_existing_unmatched_wappi_records(
        config.timeline_db,
        tenant_id=config.tenant_id,
        chat_resolutions=resolver.chat_resolutions,
        exclude_source_ids=network_source_ids,
    )
    records = (*records, *local_relink_records)
    existing_source_ids = load_existing_wappi_source_ids(
        config.timeline_db,
        tenant_id=config.tenant_id,
        source_systems=set(SOURCE_SYSTEM_BY_CHANNEL.values()),
        source_ids=[str(record.payload.get("timeline_source_id") or "") for record in records],
    )
    existing_event_assignments = load_existing_wappi_event_customers(
        config.timeline_db,
        tenant_id=config.tenant_id,
        source_systems=set(SOURCE_SYSTEM_BY_CHANNEL.values()),
        source_ids=[str(record.payload.get("timeline_source_id") or "") for record in records],
    )
    provisional_customer_ids = load_provisional_customer_ids(
        config.timeline_db,
        tenant_id=config.tenant_id,
    )
    provisional_upgrades: dict[str, str] = {}
    exact_authority_overrides = 0
    duplicate_count = 0
    blocked_customer_relink_conflicts = 0
    unresolved_kept_existing = 0
    conflicting_existing_events: list[tuple[str, str, str, str]] = []
    guarded_records: list[TimelineSourceRecord] = []
    for record in records:
        source_id = str(record.payload.get("timeline_source_id") or "")
        if source_id in existing_source_ids:
            duplicate_count += 1
            profile_id = str(record.payload.get("profile_id") or "")
            stats_key = (record.source_system, profile_id)
            if stats_key in fetch_stats_by_profile:
                fetch_stats_by_profile[stats_key].duplicate_source_ids += 1
        existing_customer, existing_authority = existing_event_assignments.get(
            (record.source_system, source_id),
            ("", ""),
        )
        proposed_customer = str(record.payload.get("resolved_customer_id") or "").strip()
        proposed_authority = str(record.payload.get("identity_authority") or "")
        candidate_customers = {
            str(item)
            for item in (record.payload.get("candidate_customer_ids") or ())
            if str(item)
        }
        rival_candidates = candidate_customers - {existing_customer, proposed_customer}
        exact_override = _is_exact_authority_override(
            existing_customer,
            existing_authority,
            proposed_customer,
            proposed_authority,
        )
        provisional_upgrade = bool(
            existing_customer
            and proposed_customer
            and proposed_customer != existing_customer
            and existing_customer in provisional_customer_ids
            and existing_authority not in WAPPI_EXACT_AMO_AUTHORITIES
            and str(record.payload.get("identity_authority") or "") != "wappi_provisional"
        )
        if provisional_upgrade:
            provisional_upgrades[existing_customer] = proposed_customer
        elif exact_override:
            exact_authority_overrides += 1
        elif existing_customer and (
            (proposed_authority == "wappi_provisional" and not rival_candidates)
            or (not proposed_customer and not (candidate_customers - {existing_customer}))
        ):
            # A missing/provisional answer is not a rival identity. Keep the
            # previously proven event byte-for-byte and do not spam conflicts.
            unresolved_kept_existing += 1
            continue
        elif existing_customer and proposed_customer != existing_customer:
            blocked_customer_relink_conflicts += 1
            existing_reason = str(record.payload.get("resolution_reason") or "")
            conflict_reason = (
                existing_reason
                if existing_reason == "existing_wappi_chat_customer_conflict"
                else "existing_wappi_source_customer_conflict"
            )
            conflicting_existing_events.append(
                (record.source_system, source_id, conflict_reason, existing_customer)
            )
            guarded_records.append(
                replace_wappi_record_resolution(
                    record,
                    reason=conflict_reason,
                    status="pending_attribution",
                )
            )
            profile_id = str(record.payload.get("profile_id") or "")
            stats_key = (record.source_system, profile_id)
            if stats_key in fetch_stats_by_profile and proposed_customer:
                stats = fetch_stats_by_profile[stats_key]
                stats.pending_attribution += 1
                if stats.linked_by_amo_auto > 0 and record.payload.get("identity_authority") == "amo_auto_resolver":
                    stats.linked_by_amo_auto -= 1
                elif stats.linked_by_amo_widget > 0 and record.payload.get("identity_authority") == "wappi_amo_widget":
                    stats.linked_by_amo_widget -= 1
                elif stats.linked_by_amo_talk > 0 and record.payload.get("identity_authority") == "amo_talk_authoritative":
                    stats.linked_by_amo_talk -= 1
                elif stats.linked_by_timeline > 0 and record.payload.get("identity_authority") == "timeline_identity":
                    stats.linked_by_timeline -= 1
                elif stats.linked_by_provisional > 0 and record.payload.get("identity_authority") == "wappi_provisional":
                    stats.linked_by_provisional -= 1
                elif stats.linked_by_pair > 0:
                    stats.linked_by_pair -= 1
                stats.resolution_status_counts["existing_wappi_source_customer_conflict"] += 1
            continue
        guarded_records.append(record)
    records = tuple(guarded_records)
    profile_id_counts = Counter(profile.profile_id for profile in profiles)
    profile_report_keys = {
        (profile.source_system, profile.profile_id): (
            profile.profile_id
            if profile_id_counts[profile.profile_id] == 1
            else f"{profile.channel}:{profile.profile_id}"
        )
        for profile in profiles
    }
    profile_reports = {
        profile_report_keys[(profile.source_system, profile.profile_id)]: {
            "profile_id": profile.profile_id,
            "brand": profile.brand,
            "channel": profile.channel,
            "source_system": profile.source_system,
            **fetch_stats_by_profile.get((profile.source_system, profile.profile_id), WappiFetchStats()).to_json_dict(),
        }
        for profile in profiles
    }
    limit_hits = [
        f"{profile_id}:{field}"
        for profile_id, report in sorted(profile_reports.items())
        for field in (
            "chat_limit_hit",
            "message_limit_hit",
            "request_limit_hit",
            "pagination_drift_detected",
            "checkpoint_no_progress",
        )
        if report.get(field)
    ]
    limit_hits.extend(widget_setup_errors)
    attribution_warnings: list[str] = []
    pending_attribution_count = sum(
        stats.pending_attribution for stats in fetch_stats_by_profile.values()
    )
    if pending_attribution_count:
        attribution_warnings.append("wappi_identity:pending_attribution")
    if not config.require_widget_linkage and widget_link_report and not widget_link_report.get("complete"):
        attribution_warnings.append("wappi_amo_widget:link_db_collection_incomplete")
    attribution_warnings.extend(
        _wappi_talk_bridge_warnings(
            post_bridge_cache=widget_link_cache_after_bridges,
            talk_report=widget_talk_link_report,
        )
    )
    if config.require_widget_linkage and resolver.widget_missing_personal_chats:
        limit_hits.append("wappi_amo_widget:personal_chat_without_contact")
    elif resolver.widget_missing_personal_chats:
        attribution_warnings.append("wappi_amo_widget:personal_chat_without_contact")
    # Conflicting historical ownership is quarantined per record above: the old
    # customer stays untouched and a pending-attribution conflict is recorded.
    # It must not discard every unrelated, safely attributed message in the batch.
    if blocked_customer_relink_conflicts:
        attribution_warnings.append("wappi_amo_widget:existing_customer_conflict")
        if config.require_widget_linkage:
            limit_hits.append("wappi_amo_widget:existing_customer_conflict")
    if config.require_widget_linkage:
        for (source_system, profile_id), stats in sorted(fetch_stats_by_profile.items()):
            if (
                stats.personal_chats != stats.widget_calls
                or stats.personal_chats != stats.widget_resolved_chats
                or stats.widget_pending_chats
            ):
                profile_key = profile_report_keys[(source_system, profile_id)]
                limit_hits.append(f"{profile_key}:widget_personal_chat_coverage_incomplete")
    empty_profiles = sorted(
        profile_id
        for profile_id, report in profile_reports.items()
        if int(report.get("records_built") or 0) == 0
        and not (
            int(report.get("chats_loaded") or 0) > 0
            and int(report.get("incremental_chats_skipped") or 0)
            + int(report.get("checkpoint_chats_skipped") or 0)
            == int(report.get("chats_loaded") or 0)
        )
    )
    if config.require_nonempty_profiles:
        limit_hits.extend(f"{profile_id}:empty_profile" for profile_id in empty_profiles)
    input_hashes_pre_apply = {
        "importer": file_sha256(Path(__file__)),
        "phase1_config": file_sha256(config.phase1_config),
        "pairs_file": file_sha256(config.pairs_file),
        "auto_pairs_file": file_sha256(config.auto_pairs_file),
        "shared_phone_stoplist": file_sha256(config.shared_phone_stoplist),
    }
    worktree_pre_apply = git_worktree_provenance(code_root)
    db_identity_pre_apply = timeline_db_identity(config.timeline_db)
    if (
        input_hashes_pre_apply != input_hashes_start
        or worktree_pre_apply != worktree_start
        or db_identity_pre_apply.get("identity_digest") != db_identity_validation_base.get("identity_digest")
    ):
        limit_hits.append("provenance_drift")

    # A checkpointed run is allowed to stop early and still WRITE what it confirmed
    # (that is the whole point: otherwise nothing ever accumulates). It is never
    # allowed to claim freshness -- validation_ok stays False until the checkpoint
    # reports terminal complete, so the nightly service will not publish "latest".
    checkpoint_complete = next_checkpoint is None or all(
        bool(entry.get("complete")) for entry in next_checkpoint.values()
    )
    checkpoint_deferred: list[str] = []
    if next_checkpoint is not None and not checkpoint_complete:
        checkpoint_deferred = [
            marker
            for marker in limit_hits
            if marker.rsplit(":", 1)[-1]
            in {"request_limit_hit", "empty_profile", "widget_personal_chat_coverage_incomplete"}
        ]
    blocking_limit_hits = [marker for marker in limit_hits if marker not in checkpoint_deferred]

    import_reports: dict[str, Mapping[str, Any]] = {}
    write_status_counts: Counter[str] = Counter()
    normalized_counts: Counter[str] = Counter()
    errors: list[Mapping[str, Any]] = []
    stale_conflict_cleanup: dict[str, int] = {}
    provisional_cleanup: Mapping[str, int] = {}
    store_summary_before: Optional[Mapping[str, Any]] = None
    store_summary_after: Optional[Mapping[str, Any]] = None
    grouped = group_records_by_source_system(records)
    records_by_resolution_reason = Counter(str(record.payload.get("resolution_reason") or record.payload.get("resolution_status") or "unknown") for record in records)
    records_by_identity_authority = Counter(str(record.payload.get("identity_authority") or "unknown") for record in records)
    # Validate the complete fetched set before opening the staging DB writable.
    for source_system, group in grouped.items():
        preview = TimelineImportService(_DryRunStore()).import_records(
            group,
            normalizer=WappiHistoryTimelineNormalizer(tenant_id=config.tenant_id, source_system=source_system),
            tenant_id=config.tenant_id,
            source_ref=f"wappi_history:{source_system}",
            idempotency_key=config.idempotency_key or stable_digest([record.to_json_dict() for record in group]),
            dry_run=True,
            actor=config.actor,
        )
        import_reports[source_system] = sanitize_wappi_import_report(preview.to_json_dict())
        normalized_counts.update({key: int(value) for key, value in preview.normalized_counts.items()})
        errors.extend(sanitize_wappi_import_error(item.to_json_dict()) for item in preview.errors)

    apply_effective = config.apply and not errors and not blocking_limit_hits
    if apply_effective:
        import_reports = {}
        write_status_counts.clear()
        normalized_counts.clear()
        errors.clear()
        store = CustomerTimelineSQLiteStore(config.timeline_db, allowed_root=config.allowed_root)
        try:
            store_summary_before = store.summary()
            with store.bulk_write():
                for source_system, group in grouped.items():
                    report = TimelineImportService(store).import_records(
                        group,
                        normalizer=WappiHistoryTimelineNormalizer(
                            tenant_id=config.tenant_id,
                            source_system=source_system,
                        ),
                        tenant_id=config.tenant_id,
                        source_ref=f"wappi_history:{source_system}",
                        idempotency_key=config.idempotency_key
                        or stable_digest([record.to_json_dict() for record in group]),
                        dry_run=False,
                        actor=config.actor,
                    )
                    if report.errors:
                        raise RuntimeError(f"Wappi {source_system} import failed after validation")
                    import_reports[source_system] = sanitize_wappi_import_report(report.to_json_dict())
                    write_status_counts.update(report.write_status_counts)
                    normalized_counts.update({key: int(value) for key, value in report.normalized_counts.items()})
                for old_customer_id, new_customer_id in sorted(provisional_upgrades.items()):
                    store.record_customer_id_mapping(
                        config.tenant_id,
                        old_customer_id=old_customer_id,
                        new_customer_id=new_customer_id,
                        reason="wappi_provisional_exact_identity_upgrade",
                        mapping_kind="alias",
                        source_refs=("wappi_history_import",),
                        actor=config.actor,
                    )
                    write_status_counts["customer_id_mapping_upserted"] += 1
                quarantined = quarantine_conflicting_wappi_events(
                    store,
                    tenant_id=config.tenant_id,
                    conflicts=conflicting_existing_events,
                    actor=config.actor,
                )
                write_status_counts.update(quarantined)
        finally:
            store.close()
        provisional_cleanup = remove_orphaned_provisional_customers(
            config.timeline_db,
            tenant_id=config.tenant_id,
            customer_ids=tuple(provisional_upgrades),
        )
        stale_conflict_cleanup = close_resolved_wappi_pending_conflicts(
            config.timeline_db,
            tenant_id=config.tenant_id,
            records=records,
        )
        with CustomerTimelineSQLiteStore(
            config.timeline_db,
            allowed_root=config.allowed_root,
            read_only=True,
        ) as store_ro:
            store_summary_after = store_ro.summary()
    checkpoint_committed = False
    if next_checkpoint is not None and apply_effective:
        # Commit the checkpoint only AFTER the rows it vouches for are in the DB.
        # A crash before this point simply replays the window; source_id keeps the
        # replay idempotent, so the checkpoint can lag but can never run ahead.
        current_timeline_state = wappi_timeline_state(
            config.timeline_db, tenant_id=config.tenant_id, profiles=profiles
        )
        stamp = datetime.now(timezone.utc).isoformat()
        # Merge, do not overwrite: a profile that was not touched this run (dropped from
        # the phase1 config, or written by a concurrent run) must keep its progress.
        merged = dict(checkpoint_state.get("profiles") or {})
        merged.update(
            {
                key: {
                    **entry,
                    "timeline_rows": int(current_timeline_state.get(str(key), {}).get("rows", 0)),
                    "timeline_source_digest": str(
                        current_timeline_state.get(str(key), {}).get("source_digest") or ""
                    ),
                    "updated_at": stamp,
                }
                for key, entry in next_checkpoint.items()
            }
        )
        save_wappi_history_checkpoint(
            config.checkpoint_dir,
            merged,
        )
        checkpoint_committed = True
    amo_read_active = bool(
        amo_auto_resolver is not None or widget_contact_hydrate_report.get("requested")
    )
    safety = {
        **timeline_import_cli_safety_contract(write_product_timeline_db=apply_effective),
        "read_local_files_only": False,
        "network_calls": True,
        "wappi_transport": "DefaultDenyTransport",
        "wappi_read_only_methods": ["GET", "POST Wappi AMO contact lookup"],
        "wappi_mark_all": False,
        "amo_auto_resolver_enabled": amo_auto_resolver is not None,
        "amo_transport": "AmoMcpClient" if amo_read_active else "disabled",
        "amo_read_only_methods": ["GET"] if amo_read_active else [],
        "send_messenger": False,
        "write_crm": False,
        "write_tallanto": False,
        "blocked_live_actions": blocked_live_actions(),
    }
    attribution_complete = not attribution_warnings
    validation_ok = (
        not errors
        and not limit_hits
        and safety_ok(safety)
        and checkpoint_complete
    )
    full_audit_profiles = sum(stats.full_history_audit for stats in fetch_stats_by_profile.values())
    full_audit_fresh_for_all_profiles = bool(next_checkpoint) and all(
        not _wappi_full_audit_state(entry)[0] for entry in next_checkpoint.values()
    )
    history_validation_mode = (
        "full_audit"
        if full_audit_profiles == len(fetch_stats_by_profile)
        else "incremental_catalog"
        if full_audit_profiles == 0
        else "mixed"
    )
    publish_ready = validation_ok and attribution_complete
    records_reliably_linked = sum(
        stats.linked_by_pair
        + stats.linked_by_timeline
        + stats.linked_by_amo_auto
        + stats.linked_by_amo_widget
        + stats.linked_by_amo_talk
        for stats in fetch_stats_by_profile.values()
    )
    messages_newly_saved = 0
    if store_summary_before is not None and store_summary_after is not None:
        messages_newly_saved = max(
            0,
            int(store_summary_after["counts"]["timeline_events"])
            - int(store_summary_before["counts"]["timeline_events"]),
        )
    messages_present_in_timeline = sum(
        len(
            load_existing_wappi_source_ids(
                config.timeline_db,
                tenant_id=config.tenant_id,
                source_systems={source_system},
                source_ids=[str(record.payload.get("timeline_source_id") or "") for record in group],
            )
        )
        for source_system, group in grouped.items()
    )
    code_identity_end = dict(build_draft_loop_code_identity())
    worktree_end = git_worktree_provenance(code_root)
    db_identity_end = timeline_db_identity(config.timeline_db)
    return {
        "schema_version": WAPPI_HISTORY_IMPORT_SCHEMA_VERSION,
        "provenance": {
            "code_root": code_identity_start.get("code_root"),
            "git_sha": code_identity_start.get("git_sha"),
            "git_sha_end": code_identity_end.get("git_sha"),
            "worktree": worktree_end,
            "worktree_start": worktree_start,
            "worktree_pre_apply": worktree_pre_apply,
            "input_hashes": input_hashes_pre_apply,
            "input_hashes_start": input_hashes_start,
            "timeline_db": db_identity_end,
            "timeline_db_start": db_identity_start,
            "timeline_db_after_hydrate": db_identity_validation_base,
            "timeline_db_pre_apply": db_identity_pre_apply,
            "input_source_id_set_hash": stable_digest(
                sorted(
                    (record.source_system, str(record.payload.get("timeline_source_id") or ""))
                    for record in records
                )
            ),
            "limits": {
                "chat_limit_per_profile": config.limits.chat_limit_per_profile,
                "messages_per_chat": config.limits.messages_per_chat,
                "message_limit_total": config.limits.message_limit_total,
                "complete_message_history": config.limits.complete_message_history,
                "request_limit_total": config.limits.request_limit_total,
                "page_size": config.limits.page_size,
                "sleep_seconds": config.limits.sleep_seconds,
                "show_all_chats": config.limits.show_all_chats,
                "require_nonempty_profiles": config.require_nonempty_profiles,
                "require_widget_linkage": config.require_widget_linkage,
            },
        },
        "mode": "apply" if apply_effective else ("apply_blocked" if config.apply else "dry_run_preview"),
        "dry_run": not apply_effective,
        "validation_ok": validation_ok,
        "history_validation": {
            "mode": history_validation_mode,
            "catalog_incremental_passed": validation_ok,
            "full_audit_completed_this_run": validation_ok and history_validation_mode == "full_audit",
            "full_audit_fresh_for_all_profiles": full_audit_fresh_for_all_profiles,
            "full_audit_passed": validation_ok and full_audit_fresh_for_all_profiles,
            "full_audit_interval_days": WAPPI_INCREMENTAL_FULL_AUDIT_DAYS,
        },
        "fetch_complete": validation_ok,
        "attribution_complete": attribution_complete,
        "publish_ready": publish_ready,
        "limit_hits": limit_hits,
        "attribution_warnings": sorted(set(attribution_warnings)),
        "checkpoint": {
            "enabled": next_checkpoint is not None,
            "schema_version": WAPPI_HISTORY_CHECKPOINT_SCHEMA_VERSION,
            "path": str(wappi_history_checkpoint_path(config.checkpoint_dir)) if config.checkpoint_dir else None,
            "complete": checkpoint_complete,
            "committed": checkpoint_committed,
            "deferred_limit_hits": checkpoint_deferred,
            "profiles": {
                key: {
                    "complete": bool(entry.get("complete")),
                    "stop_reason": entry.get("stop_reason"),
                    "reset_reason": entry.get("reset_reason"),
                    "catalog_next_offset": entry.get("catalog_next_offset"),
                    "catalog_chats_seen": entry.get("catalog_chats_seen"),
                    "chats_done": len(entry.get("chats_done") or ()),
                    "resumed_from": entry.get("resumed_from"),
                    "active_chat_message_offset": (entry.get("active_chat") or {}).get("message_offset"),
                }
                for key, entry in sorted((next_checkpoint or {}).items())
            },
        },
        "summary": {
            "tenant_id": config.tenant_id,
            "profiles": len(profiles),
            "records_built": len(records),
            "messages_newly_saved": messages_newly_saved,
            "messages_present_in_timeline": messages_present_in_timeline,
            "records_reliably_linked": records_reliably_linked,
            "attribution_complete": attribution_complete,
            "linked_by_pair": sum(stats.linked_by_pair for stats in fetch_stats_by_profile.values()),
            "linked_by_timeline": sum(stats.linked_by_timeline for stats in fetch_stats_by_profile.values()),
            "linked_by_amo_auto": sum(stats.linked_by_amo_auto for stats in fetch_stats_by_profile.values()),
            "linked_by_amo_widget": sum(stats.linked_by_amo_widget for stats in fetch_stats_by_profile.values()),
            "linked_by_amo_talk": sum(stats.linked_by_amo_talk for stats in fetch_stats_by_profile.values()),
            "linked_by_amo_event_sequence": 0,
            "linked_by_provisional": sum(
                stats.linked_by_provisional for stats in fetch_stats_by_profile.values()
            ),
            "pending_attribution": pending_attribution_count,
            "incremental_chats_skipped": sum(
                stats.incremental_chats_skipped for stats in fetch_stats_by_profile.values()
            ),
            "incremental_chats_new": sum(stats.incremental_chats_new for stats in fetch_stats_by_profile.values()),
            "incremental_chats_changed": sum(
                stats.incremental_chats_changed for stats in fetch_stats_by_profile.values()
            ),
            "incremental_chats_without_marker": sum(
                stats.incremental_chats_without_marker for stats in fetch_stats_by_profile.values()
            ),
            "incremental_chats_marker_regressed": sum(
                stats.incremental_chats_marker_regressed for stats in fetch_stats_by_profile.values()
            ),
            "requests": sum(stats.requests for stats in fetch_stats_by_profile.values()),
            "physical_requests": readonly_wappi_physical_request_count(client),
            "amo_auto_enabled": amo_auto_resolver is not None,
            "amo_auto_calls": sum(stats.amo_auto_calls for stats in fetch_stats_by_profile.values()),
            "amo_widget_enabled": bool(widget_profiles),
            "amo_widget_calls": resolver.widget_calls,
            "amo_widget_missing_personal_chats": resolver.widget_missing_personal_chats,
            "amo_widget_link_map": widget_link_report,
            "amo_widget_link_map_after_bridges": widget_link_cache_after_bridges,
            "amo_widget_contact_hydrate": widget_contact_hydrate_report,
            "amo_event_sequence_link_map": widget_event_link_report,
            "amo_talk_authoritative_link_map": widget_talk_link_report,
            "amo_widget_map_prime": widget_prime_report,
            "message_identity_prime": message_identity_prime_report,
            "provisional_wappi_prime": provisional_prime_report,
            "write_applied": apply_effective,
            "writes_applied": (
                sum(value for key, value in write_status_counts.items() if key != "duplicate")
                if apply_effective
                else 0
            ),
            "writes_attempted": sum(write_status_counts.values()) if apply_effective else 0,
            "duplicate_source_ids_before_import": duplicate_count,
            "blocked_customer_relink_conflicts": blocked_customer_relink_conflicts,
            "unresolved_kept_existing": unresolved_kept_existing,
            "provisional_customer_upgrades": len(provisional_upgrades),
            "exact_authority_overrides": exact_authority_overrides,
            "local_unmatched_relink_records": len(local_relink_records),
            "blocked_chat_relink_conflicts": int(
                records_by_resolution_reason.get("existing_wappi_chat_customer_conflict", 0)
            ),
            "pending_reason_counts": {
                key: value
                for key, value in sorted(records_by_resolution_reason.items())
                if key not in {"resolved", "pending_attribution"}
            },
            "transport": "DefaultDenyTransport",
            "send_messenger": False,
            "limit_hits": limit_hits,
            "attribution_warnings": sorted(set(attribution_warnings)),
            "empty_profiles": empty_profiles,
            "checkpoint_enabled": next_checkpoint is not None,
            "checkpoint_complete": checkpoint_complete,
            "checkpoint_paused_profiles": sorted(
                key for key, entry in (next_checkpoint or {}).items() if not entry.get("complete")
            ),
        },
        "profiles": profile_reports,
        "records": {
            "by_source_system": {key: len(value) for key, value in grouped.items()},
            "by_resolution_reason": dict(records_by_resolution_reason),
            "by_identity_authority": dict(records_by_identity_authority),
        },
        "normalization": {"counts": dict(normalized_counts)},
        "writes": {
            "target": {"db_path": str(config.timeline_db), "allowed_root": str(config.allowed_root)},
            "applied": apply_effective,
            "status_counts": dict(write_status_counts),
            "import_groups_single_transaction": True if apply_effective else None,
            "post_import_cleanup_same_transaction": False if apply_effective else None,
            "all_db_mutations_single_transaction": False if apply_effective else None,
        },
        "stale_conflict_cleanup": stale_conflict_cleanup,
        "provisional_cleanup": provisional_cleanup,
        "import_reports": import_reports,
        "errors": errors,
        "store_summary_before": store_summary_before,
        "store_summary_after": store_summary_after,
        "examples": anonymized_examples(records, limit=5),
        "safety": {**safety, "ok": safety_ok(safety)},
    }


class _DryRunStore:
    pass


class WappiPairCustomerResolver:
    def __init__(
        self,
        resolutions: Mapping[DraftLoopKey, WappiChatResolution],
        *,
        db_path: Path,
        tenant_id: str,
        amo_auto_resolver: AmoAutoResolver | None = None,
        widget_client: WappiPhase1Client | None = None,
        widget_crm_id: str = "",
        widget_profiles: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
        widget_links: Mapping[tuple[str, str, str], Mapping[str, Any]] | None = None,
        widget_required: bool = False,
        local_identity_customers: Mapping[tuple[str, str], Sequence[str]] | None = None,
        ambiguous_identity_values: Sequence[tuple[str, str]] = (),
        customer_brands: Mapping[str, str] | None = None,
        supported_customer_ids: Sequence[str] = (),
        chat_customer_ids: Mapping[tuple[str, str, str], Sequence[str]] | None = None,
        exact_chat_customer_ids: Mapping[tuple[str, str, str], Sequence[str]] | None = None,
        provisional_customer_ids: Sequence[str] = (),
        shared_phone_stoplist: Sequence[str] = (),
        shared_phone_stoplist_error: str = "",
    ) -> None:
        self._resolutions = dict(resolutions)
        self._db_path = Path(db_path)
        self._tenant_id = normalize_key(tenant_id, "tenant_id")
        self._amo_auto_resolver = amo_auto_resolver
        self._widget_client = widget_client
        self._widget_crm_id = str(widget_crm_id or "").strip()
        self._widget_profiles: dict[tuple[str, str], Mapping[str, Any]] = {}
        for key, item in (widget_profiles or {}).items():
            if isinstance(key, tuple) and len(key) == 2:
                self._widget_profiles[(str(key[0]), str(key[1]))] = item
                continue
            platform = str(item.get("channel") or item.get("platform") or "").strip().casefold()
            channel = "telegram" if platform in {"tg", "telegram"} else "max" if platform == "max" else ""
            if channel:
                self._widget_profiles[(channel, str(key))] = item
        self._widget_links = dict(widget_links or {})
        self._widget_required = bool(widget_required)
        self._widget_calls = 0
        self._widget_missing_personal_chats = 0
        self._widget_chat_resolutions: dict[tuple[str, str, str], WappiChatResolution] = {}
        self._chat_resolutions: dict[tuple[str, str, str], WappiChatResolution] = {}
        self._local_identity_customers = {
            key: tuple(sorted(set(values))) for key, values in (local_identity_customers or {}).items()
        }
        self._ambiguous_identity_values = frozenset(ambiguous_identity_values)
        self._customer_brands = dict(customer_brands or {})
        self._supported_customer_ids = frozenset(supported_customer_ids)
        self._chat_customer_ids = {
            key: tuple(sorted(set(values))) for key, values in (chat_customer_ids or {}).items()
        }
        self._exact_chat_customer_ids = {
            key: tuple(sorted(set(values))) for key, values in (exact_chat_customer_ids or {}).items()
        }
        self._provisional_customer_ids = frozenset(provisional_customer_ids)
        self._shared_phone_stoplist = frozenset(shared_phone_stoplist)
        self._shared_phone_stoplist_error = str(shared_phone_stoplist_error or "")

    @property
    def amo_auto_calls(self) -> int:
        return int(getattr(self._amo_auto_resolver, "calls", 0)) if self._amo_auto_resolver is not None else 0

    @property
    def widget_calls(self) -> int:
        return self._widget_calls

    @property
    def widget_missing_personal_chats(self) -> int:
        return self._widget_missing_personal_chats

    @property
    def widget_chat_resolutions(self) -> Mapping[tuple[str, str, str], WappiChatResolution]:
        return dict(self._widget_chat_resolutions)

    @property
    def chat_resolutions(self) -> Mapping[tuple[str, str, str], WappiChatResolution]:
        return dict(self._chat_resolutions)

    def prime_widget_chat_resolutions(
        self,
        profiles: Sequence[WappiProfileSpec],
    ) -> Mapping[str, int]:
        """Resolve every proven row in the persistent map, including old chats."""
        profile_index = {(item.channel, item.profile_id): item for item in profiles}
        counts: Counter[str] = Counter()
        for (channel, profile_id, chat_id), cached in sorted(self._widget_links.items()):
            if str(cached.get("status") or "") != "resolved":
                counts["skipped_unresolved"] += 1
                continue
            profile = profile_index.get((channel, profile_id))
            if profile is None:
                counts["profile_missing"] += 1
                continue
            contact_id = str(cached.get("contact_id") or "").strip()
            lead_ids = tuple(
                sorted({str(item).strip() for item in cached.get("lead_ids") or () if str(item).strip()})
            )
            resolution = self._resolve_widget_candidate_to_customer(
                profile=profile,
                contact_id=contact_id,
                lead_ids=lead_ids,
                resolution_source=str(cached.get("resolution_source") or "wappi_amo_widget"),
            )
            guarded = self._guard_chat_customer(profile, chat_id, resolution)
            key = (profile.source_system, profile_id, chat_id)
            self._widget_chat_resolutions[key] = guarded
            self._chat_resolutions[key] = guarded
            counts["resolved" if guarded.resolved else "pending_attribution"] += 1
        return dict(sorted(counts.items()))

    def prime_existing_message_identity_resolutions(
        self,
        profiles: Sequence[WappiProfileSpec],
    ) -> Mapping[str, int]:
        """Count message-body identifiers as candidates; never treat them as sender identity."""
        profile_index = {(item.source_system, item.profile_id): item for item in profiles}
        values_by_chat: dict[tuple[str, str, str], set[tuple[str, str]]] = defaultdict(set)
        with open_readonly_sqlite(self._db_path) as con:
            for row in con.execute(
                """
                SELECT source_system,
                       json_extract(record_json, '$.metadata.profile_id') AS profile_id,
                       json_extract(record_json, '$.metadata.chat_id') AS chat_id,
                       COALESCE(json_extract(record_json, '$.record.message.text'), text_preview, summary, '') AS text
                FROM timeline_events
                WHERE tenant_id = ?
                  AND source_system IN ('wappi_telegram', 'wappi_max')
                  AND customer_id IS NULL
                  AND direction = 'inbound'
                  AND superseded_by IS NULL
                  AND json_valid(record_json)
                """,
                (self._tenant_id,),
            ):
                source_system = str(row[0])
                profile_id = str(row[1] or "").strip()
                chat_id = str(row[2] or "").strip()
                channel = "telegram" if source_system == "wappi_telegram" else "max"
                if (channel, profile_id, chat_id) not in self._widget_links:
                    continue
                text = str(row[3] or "")
                for candidate in WAPPI_MESSAGE_EMAIL_RE.findall(text):
                    email = normalize_email(candidate)
                    if email:
                        values_by_chat[(source_system, profile_id, chat_id)].add(("email", email))
                for candidate in WAPPI_MESSAGE_PHONE_RE.findall(text):
                    phone = normalize_phone(candidate)
                    if phone and not self._shared_phone_stoplist_error and phone not in self._shared_phone_stoplist:
                        values_by_chat[(source_system, profile_id, chat_id)].add(("phone", phone))

        counts: Counter[str] = Counter()
        for key, identity_values in sorted(values_by_chat.items()):
            current = self._chat_resolutions.get(key)
            if current is not None and current.resolved:
                counts["already_resolved"] += 1
                continue
            source_system, profile_id, _chat_id = key
            if profile_index.get((source_system, profile_id)) is None:
                counts["profile_missing"] += 1
                continue
            owner_sets: list[set[str]] = []
            ambiguous = False
            for identity_key in sorted(identity_values):
                if identity_key in self._ambiguous_identity_values:
                    ambiguous = True
                    continue
                owners = set(self._local_identity_customers.get(identity_key, ()))
                if len(owners) == 1:
                    owner_sets.append(owners)
                elif len(owners) > 1:
                    ambiguous = True
            candidate_customers = set().union(*owner_sets) if owner_sets else set()
            if ambiguous or len(candidate_customers) > 1:
                counts["ambiguous"] += 1
                continue
            if len(candidate_customers) != 1:
                counts["unmatched"] += 1
                continue
            counts["candidate_unique"] += 1
        return dict(sorted(counts.items()))

    def prime_provisional_chat_resolutions(
        self,
        profiles: Sequence[WappiProfileSpec],
    ) -> Mapping[str, int]:
        """Keep genuine personal Wappi history without pretending it has a CRM identity."""
        profile_index = {(item.channel, item.profile_id): item for item in profiles}
        counts: Counter[str] = Counter()
        for (channel, profile_id, chat_id), cached in sorted(self._widget_links.items()):
            if str(cached.get("status") or "") not in {"missing", "candidate"}:
                continue
            profile = profile_index.get((channel, profile_id))
            if profile is None:
                counts["profile_missing"] += 1
                continue
            key = (profile.source_system, profile_id, chat_id)
            current = self._chat_resolutions.get(key)
            if current is not None and current.resolved:
                counts["already_resolved"] += 1
                continue
            source_ref = f"wappi_provisional:{profile.source_system}:{profile_id}:{chat_id}"
            resolution = WappiChatResolution(
                status="resolved",
                customer_id=stable_customer_id(tenant_id=self._tenant_id, source_ref=source_ref),
                expected_brand=profile.brand,
                reason="provisional_wappi_family",
                pair_source="wappi_provisional",
                resolution_source="wappi_provisional",
                match_key="channel_session_id",
                evidence={
                    "provisional_wappi_family": True,
                    "brand_context_authorized": False,
                },
            )
            self._chat_resolutions[key] = resolution
            counts["created"] += 1
        return dict(sorted(counts.items()))

    def record_coverage(self, *, profile: WappiProfileSpec, dialog: Mapping[str, Any], stats: WappiFetchStats) -> None:
        chat_id = extract_chat_id(dialog)
        if profile.channel == "telegram":
            stats.coverage_counts["tg_chats"] += 1
            if chat_id.isdigit():
                stats.coverage_counts["tg_chat_id_digit"] += 1
            else:
                stats.coverage_counts["tg_username_only"] += 1
            return
        if profile.channel == "max":
            stats.coverage_counts["max_chats"] += 1
            phone, source = max_dialog_phone(dialog)
            if not phone:
                stats.coverage_counts[source] += 1
                return
            stats.coverage_counts["max_phone_present"] += 1
            stoplist_error = self._shared_phone_stoplist_error or str(
                getattr(self._amo_auto_resolver, "stoplist_error", "") or ""
            )
            if stoplist_error:
                stats.coverage_counts[stoplist_error] += 1
            elif phone in self._shared_phone_stoplist:
                stats.coverage_counts["max_phone_in_stoplist"] += 1
            else:
                stats.coverage_counts["max_phone_outside_stoplist"] += 1
            if self._amo_auto_resolver is None:
                stats.coverage_counts["max_phone_auto_resolver_disabled"] += 1
            else:
                stats.coverage_counts["max_phone_auto_resolver_enabled"] += 1
            return
        stats.coverage_counts["unsupported_channel"] += 1

    @classmethod
    def from_store(
        cls,
        db_path: Path,
        *,
        tenant_id: str,
        pairs: Mapping[DraftLoopKey, DraftLoopPair],
        amo_auto_resolver: AmoAutoResolver | None = None,
        widget_client: WappiPhase1Client | None = None,
        widget_crm_id: str = "",
        widget_profiles: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
        widget_links: Mapping[tuple[str, str, str], Mapping[str, Any]] | None = None,
        widget_required: bool = False,
        shared_phone_stoplist: Sequence[str] = (),
        shared_phone_stoplist_error: str = "",
    ) -> "WappiPairCustomerResolver":
        tenant = normalize_key(tenant_id, "tenant_id")
        if not db_path.exists():
            return cls(
                {},
                db_path=db_path,
                tenant_id=tenant,
                amo_auto_resolver=amo_auto_resolver,
                widget_client=widget_client,
                widget_crm_id=widget_crm_id,
                widget_profiles=widget_profiles,
                widget_links=widget_links,
                widget_required=widget_required,
                shared_phone_stoplist=shared_phone_stoplist,
                shared_phone_stoplist_error=shared_phone_stoplist_error,
            )
        resolutions: dict[DraftLoopKey, WappiChatResolution] = {}
        local_identity_customers: dict[tuple[str, str], set[str]] = {}
        ambiguous_identity_values: set[tuple[str, str]] = set()
        customer_brands: dict[str, str] = {}
        supported_customer_ids: set[str] = set()
        chat_customer_ids: dict[tuple[str, str, str], set[str]] = {}
        exact_chat_customer_ids: dict[tuple[str, str, str], set[str]] = {}
        provisional_customer_ids: set[str] = set()
        with open_readonly_sqlite(db_path) as con:
            provisional_customer_ids.update(
                str(row["customer_id"])
                for row in con.execute(
                    """
                    SELECT customer_id
                    FROM customer_identities
                    WHERE tenant_id = ?
                      AND json_extract(record_json, '$.metadata.provisional_wappi_family') = 1
                    """,
                    (tenant,),
                )
            )
            safe_customer_ids = {
                str(row["customer_id"])
                for row in con.execute(
                    """
                    SELECT customer_id
                    FROM customer_identities
                    WHERE tenant_id = ?
                      AND identity_status IN ('strong', 'partial')
                      AND COALESCE(json_extract(record_json, '$.metadata.provisional_wappi_family'), 0) != 1
                    """,
                    (tenant,),
                )
            }
            identity_rows = con.execute(
                """
                SELECT link_type, link_value, customer_id, match_class
                FROM identity_links
                WHERE tenant_id = ?
                  AND link_type IN (
                    'telegram_user_id', 'telegram_username', 'max_user_id',
                    'phone', 'mango_client_phone', 'whatsapp_phone', 'email',
                    'amo_contact_id', 'amo_lead_id'
                  )
                """,
                (tenant,),
            ).fetchall()
            identity_owners: dict[tuple[str, str], set[str]] = {}
            identity_classes: dict[tuple[str, str], set[str]] = {}
            for row in identity_rows:
                raw_link_type = str(row["link_type"])
                link_type = "phone" if raw_link_type in PHONE_IDENTITY_LINK_TYPES else raw_link_type
                key = (link_type, str(row["link_value"]))
                customer_id = str(row["customer_id"] or "")
                if customer_id:
                    identity_owners.setdefault(key, set()).add(customer_id)
                identity_classes.setdefault(key, set()).add(str(row["match_class"] or ""))
            ambiguous_identity_values.update(
                key
                for key in identity_classes
                if len(identity_owners.get(key, ())) != 1
                or not identity_classes[key].issubset({"strong_unique", "manual"})
                or not identity_owners.get(key, set()).issubset(safe_customer_ids)
            )
            for row in identity_rows:
                raw_link_type = str(row["link_type"])
                link_type = "phone" if raw_link_type in PHONE_IDENTITY_LINK_TYPES else raw_link_type
                key = (link_type, str(row["link_value"]))
                customer_id = str(row["customer_id"])
                if (
                    str(row["match_class"] or "") not in {"strong_unique", "manual"}
                    or key in ambiguous_identity_values
                    or customer_id not in safe_customer_ids
                    or not customer_id
                ):
                    continue
                local_identity_customers.setdefault(key, set()).add(customer_id)
            supported_customer_ids.update(
                str(row["customer_id"])
                for row in con.execute(
                    """
                    SELECT MIN(customer_id) AS customer_id
                    FROM identity_links
                    WHERE tenant_id = ?
                      AND link_type IN ('amo_contact_id', 'tallanto_student_id')
                      AND customer_id IS NOT NULL
                      AND customer_id != ''
                    GROUP BY link_type, link_value
                    HAVING COUNT(DISTINCT customer_id) = 1
                       AND SUM(CASE WHEN match_class NOT IN ('strong_unique', 'manual') THEN 1 ELSE 0 END) = 0
                    """,
                    (tenant,),
                )
            )
            brand_sets: dict[str, set[str]] = {}
            for row in con.execute(
                """
                SELECT customer_id, json_extract(record_json, '$.product_context.brand') AS brand
                FROM customer_opportunities
                WHERE tenant_id = ?
                """,
                (tenant,),
            ):
                raw_brand = str(row["brand"] or "").strip().casefold()
                if raw_brand not in {"foton", "unpk"}:
                    continue
                brand = raw_brand
                if brand in {"foton", "unpk"}:
                    brand_sets.setdefault(str(row["customer_id"]), set()).add(brand)
            for row in con.execute(
                """
                SELECT customer_id,
                       COALESCE(
                         json_extract(record_json, '$.metadata.brand'),
                         json_extract(record_json, '$.record.brand')
                       ) AS brand
                FROM timeline_events
                WHERE tenant_id = ?
                  AND customer_id IS NOT NULL
                  AND source_system NOT IN ('wappi_telegram', 'wappi_max')
                """,
                (tenant,),
            ):
                raw_brand = str(row["brand"] or "").strip().casefold()
                if raw_brand not in {"foton", "unpk"}:
                    continue
                brand = raw_brand
                if brand in {"foton", "unpk"}:
                    brand_sets.setdefault(str(row["customer_id"]), set()).add(brand)
            customer_brands.update(
                {
                    customer_id: next(iter(brands))
                    for customer_id, brands in brand_sets.items()
                    if len(brands) == 1
                }
            )
            for row in con.execute(
                """
                SELECT link_value, customer_id
                FROM identity_links
                WHERE tenant_id = ? AND link_type = 'channel_session_id'
                """,
                (tenant,),
            ):
                parts = str(row["link_value"]).split(":", 2)
                if len(parts) == 3 and parts[0] in SOURCE_SYSTEM_BY_CHANNEL.values():
                    chat_key = (parts[0], parts[1], parts[2])
                    chat_customer_ids.setdefault(chat_key, set()).add(str(row["customer_id"]))
            if sqlite_table_exists(con, "timeline_events"):
                for row in con.execute(
                    """
                    SELECT source_system,
                           json_extract(record_json, '$.metadata.profile_id') AS profile_id,
                           json_extract(record_json, '$.metadata.chat_id') AS chat_id,
                           customer_id,
                           COALESCE(json_extract(record_json, '$.metadata.identity_authority'), '') AS identity_authority
                    FROM timeline_events
                    WHERE tenant_id = ?
                      AND source_system IN ('wappi_telegram', 'wappi_max')
                      AND customer_id IS NOT NULL
                      AND superseded_by IS NULL
                    """,
                    (tenant,),
                ):
                    if row["profile_id"] and row["chat_id"]:
                        chat_key = (str(row["source_system"]), str(row["profile_id"]), str(row["chat_id"]))
                        chat_customer_ids.setdefault(chat_key, set()).add(str(row["customer_id"]))
                        if str(row["identity_authority"] or "") in WAPPI_EXACT_AMO_AUTHORITIES:
                            exact_chat_customer_ids.setdefault(chat_key, set()).add(str(row["customer_id"]))
            for key, pair in pairs.items():
                lead_ids = lookup_amo_link_customers(
                    con,
                    tenant_id=tenant,
                    link_type="amo_lead_id",
                    link_value=str(pair.lead_id or ""),
                )
                contact_ids = lookup_amo_link_customers(
                    con,
                    tenant_id=tenant,
                    link_type="amo_contact_id",
                    link_value=str(pair.contact_id or ""),
                )
                opportunity_ids, opportunity_id = lookup_amo_opportunity_customers(
                    con,
                    tenant_id=tenant,
                    lead_id=str(pair.lead_id or ""),
                )
                candidate_sets = [items for items in (lead_ids, contact_ids, opportunity_ids) if items]
                candidate_union = set().union(*candidate_sets) if candidate_sets else set()
                if candidate_sets and all(items == candidate_sets[0] for items in candidate_sets) and len(candidate_union) == 1:
                    resolutions[key] = WappiChatResolution(
                        status="resolved",
                        customer_id=next(iter(candidate_union)),
                        opportunity_id=opportunity_id or None,
                        lead_id=str(pair.lead_id),
                        contact_id=str(pair.contact_id or ""),
                        expected_brand=pair.expected_brand,
                        pair_source=pair.source,
                        resolution_source="draft_loop_pair",
                    )
                elif len(candidate_union) > 1:
                    resolutions[key] = WappiChatResolution(
                        status="pending_attribution",
                        lead_id=str(pair.lead_id),
                        contact_id=str(pair.contact_id or ""),
                        expected_brand=pair.expected_brand,
                        reason="pair_matches_multiple_or_conflicting_customers",
                        candidate_customer_ids=tuple(sorted(candidate_union)),
                        pair_source=pair.source,
                        resolution_source="draft_loop_pair",
                    )
                else:
                    resolutions[key] = WappiChatResolution(
                        status="pending_attribution",
                        lead_id=str(pair.lead_id),
                        contact_id=str(pair.contact_id or ""),
                        expected_brand=pair.expected_brand,
                        reason="pair_has_no_customer_in_timeline",
                        pair_source=pair.source,
                        resolution_source="draft_loop_pair",
                    )
        return cls(
            resolutions,
            db_path=db_path,
            tenant_id=tenant,
            amo_auto_resolver=amo_auto_resolver,
            widget_client=widget_client,
            widget_crm_id=widget_crm_id,
            widget_profiles=widget_profiles,
            widget_links=widget_links,
            widget_required=widget_required,
            local_identity_customers=local_identity_customers,
            ambiguous_identity_values=tuple(ambiguous_identity_values),
            customer_brands=customer_brands,
            supported_customer_ids=tuple(supported_customer_ids),
            chat_customer_ids=chat_customer_ids,
            exact_chat_customer_ids=exact_chat_customer_ids,
            provisional_customer_ids=tuple(provisional_customer_ids),
            shared_phone_stoplist=shared_phone_stoplist,
            shared_phone_stoplist_error=shared_phone_stoplist_error,
        )

    def resolve(self, *, profile: WappiProfileSpec, chat_id: str) -> WappiChatResolution:
        key = DraftLoopKey(profile.profile_id, chat_id)
        resolution = self._resolutions.get(key)
        if resolution is None:
            return WappiChatResolution(status="pending_attribution", expected_brand=profile.brand, reason="draft_loop_pair_missing")
        if resolution.expected_brand and resolution.expected_brand != profile.brand:
            return WappiChatResolution(
                status="pending_attribution",
                lead_id=resolution.lead_id,
                contact_id=resolution.contact_id,
                expected_brand=resolution.expected_brand,
                reason="draft_loop_pair_brand_mismatch",
                pair_source=resolution.pair_source,
                resolution_source="draft_loop_pair",
            )
        return resolution

    def resolve_chat(
        self,
        *,
        profile: WappiProfileSpec,
        dialog: Mapping[str, Any],
        messages: Sequence[WappiHistoryMessage],
    ) -> WappiChatResolution:
        chat_id = extract_chat_id(dialog) or (messages[0].chat_id if messages else "")
        widget_resolution = self._resolve_with_amo_widget(profile=profile, dialog=dialog)
        if widget_resolution is not None:
            if (
                not self._widget_required
                and not widget_resolution.resolved
                and not widget_resolution.candidate_customer_ids
                and widget_resolution.reason == "wappi_widget_timeline_identity_missing_or_conflicting"
            ):
                widget_resolution = None
        if widget_resolution is not None:
            guarded = self._guard_chat_customer(profile, chat_id, widget_resolution)
            if is_personal_wappi_dialog(profile, dialog) and chat_id:
                key = (profile.source_system, profile.profile_id, chat_id)
                self._widget_chat_resolutions[key] = guarded
                self._chat_resolutions[key] = guarded
            return guarded
        primed = self._chat_resolutions.get((profile.source_system, profile.profile_id, chat_id))
        if (
            primed is not None
            and (primed.resolved or primed.candidate_customer_ids)
            and primed.resolution_source != "wappi_provisional"
            and is_personal_wappi_dialog(profile, dialog)
        ):
            return primed
        pair_resolution = self.resolve(profile=profile, chat_id=chat_id)
        if pair_resolution.reason != "draft_loop_pair_missing":
            guarded = self._guard_chat_customer(
                profile,
                chat_id,
                self._guard_pair_context(profile, chat_id, dialog, pair_resolution),
            )
            self._remember_chat_resolution(profile, chat_id, dialog, guarded)
            return guarded
        timeline_resolution = self._resolve_with_timeline_identity(
            profile=profile,
            chat_id=chat_id,
            dialog=dialog,
        )
        if timeline_resolution is not None:
            guarded = self._guard_chat_customer(profile, chat_id, timeline_resolution)
            self._remember_chat_resolution(profile, chat_id, dialog, guarded)
            return guarded
        if self._amo_auto_resolver is None:
            return primed if primed is not None else pair_resolution
        guarded = self._guard_chat_customer(
            profile,
            chat_id,
            self._resolve_with_amo_auto(profile=profile, chat_id=chat_id, dialog=dialog, messages=messages),
        )
        if (
            not guarded.resolved
            and not guarded.candidate_customer_ids
            and primed is not None
            and primed.resolution_source == "wappi_provisional"
        ):
            guarded = primed
        self._remember_chat_resolution(profile, chat_id, dialog, guarded)
        return guarded

    def _remember_chat_resolution(
        self,
        profile: WappiProfileSpec,
        chat_id: str,
        dialog: Mapping[str, Any],
        resolution: WappiChatResolution,
    ) -> None:
        if chat_id and is_personal_wappi_dialog(profile, dialog):
            self._chat_resolutions[(profile.source_system, profile.profile_id, chat_id)] = resolution

    def _resolve_with_amo_widget(
        self,
        *,
        profile: WappiProfileSpec,
        dialog: Mapping[str, Any],
    ) -> WappiChatResolution | None:
        chat_id = extract_chat_id(dialog)
        cached = self._widget_links.get((profile.channel, profile.profile_id, chat_id))
        if cached is not None:
            self._widget_calls += 1
            cached_status = str(cached.get("status") or "").strip()
            if cached_status != "resolved":
                self._widget_missing_personal_chats += 1
                fail_closed = cached_status in WAPPI_TECHNICAL_LINK_STATUSES or cached_status not in {
                    "candidate",
                    "missing",
                    "resolved",
                }
                if self._widget_required or fail_closed:
                    return WappiChatResolution(
                        status="pending_attribution",
                        expected_brand=profile.brand,
                        reason=(
                            "wappi_widget_contact_unconfirmed"
                            if cached_status == "candidate"
                            else f"wappi_widget_{cached_status}"
                            if fail_closed
                            else "wappi_widget_contact_missing"
                        ),
                        resolution_source="wappi_amo_widget",
                    )
                return None
            contact_id = str(cached.get("contact_id") or "").strip()
            lead_ids = tuple(sorted({str(item).strip() for item in cached.get("lead_ids") or () if str(item).strip()}))
            if contact_id:
                return self._resolve_widget_candidate_to_customer(
                    profile=profile,
                    contact_id=contact_id,
                    lead_ids=lead_ids,
                    resolution_source=str(cached.get("resolution_source") or "wappi_amo_widget"),
                )
            self._widget_missing_personal_chats += 1
            if self._widget_required:
                return WappiChatResolution(
                    status="pending_attribution",
                    expected_brand=profile.brand,
                    reason="wappi_widget_contact_missing",
                    resolution_source="wappi_amo_widget",
                )
            return None
        runtime_profile = self._widget_profiles.get((profile.channel, profile.profile_id))
        if self._widget_client is None or not self._widget_crm_id or runtime_profile is None:
            if self._widget_required:
                self._widget_missing_personal_chats += 1
                return WappiChatResolution(
                    status="pending_attribution",
                    expected_brand=profile.brand,
                    reason="wappi_widget_unavailable",
                    resolution_source="wappi_amo_widget",
                )
            return None
        if not is_personal_wappi_dialog(profile, dialog):
            return None
        self._widget_calls += 1
        try:
            payload = _find_wappi_widget_contact(
                self._widget_client,
                profile=profile,
                runtime_profile=runtime_profile,
                dialog=dialog,
                crm_id=self._widget_crm_id,
            )
        except AmoWappiConfigError:
            self._widget_missing_personal_chats += 1
            return WappiChatResolution(
                status="pending_attribution",
                expected_brand=profile.brand,
                reason="wappi_widget_peer_id_missing_or_ambiguous",
                resolution_source="wappi_amo_widget",
            )
        contact = payload.get("contact") if isinstance(payload.get("contact"), Mapping) else {}
        contact_id = str(contact.get("id") or "").strip()
        raw_leads: list[Mapping[str, Any]] = []
        for candidate in (
            payload.get("leads"),
            contact.get("_embedded", {}).get("leads") if isinstance(contact.get("_embedded"), Mapping) else None,
        ):
            if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes, bytearray)):
                raw_leads.extend(item for item in candidate if isinstance(item, Mapping))
        lead_ids = tuple(sorted({str(item.get("id") or "").strip() for item in raw_leads if str(item.get("id") or "").strip()}))
        if not contact_id:
            self._widget_missing_personal_chats += 1
            if self._widget_required:
                return WappiChatResolution(
                    status="pending_attribution",
                    expected_brand=profile.brand,
                    reason="wappi_widget_contact_missing",
                    resolution_source="wappi_amo_widget",
                )
            return None
        return self._resolve_widget_candidate_to_customer(
            profile=profile,
            contact_id=contact_id,
            lead_ids=lead_ids,
        )

    def _resolve_widget_candidate_to_customer(
        self,
        *,
        profile: WappiProfileSpec,
        contact_id: str,
        lead_ids: Sequence[str],
        resolution_source: str = "wappi_amo_widget",
    ) -> WappiChatResolution:
        source = resolution_source if resolution_source in WAPPI_EXACT_AMO_AUTHORITIES else "wappi_amo_widget"
        match_key = "wappi_widget_contact"
        contact_owners = set(self._local_identity_customers.get(("amo_contact_id", contact_id), ()))
        lead_owner_sets = tuple(
            set(self._local_identity_customers.get(("amo_lead_id", lead_id), ()))
            for lead_id in lead_ids
        )
        all_owners = set(contact_owners)
        for owners in lead_owner_sets:
            all_owners.update(owners)
        nonempty_lead_owners = tuple(owners for owners in lead_owner_sets if owners)
        if not contact_owners and nonempty_lead_owners:
            first_lead_owners = nonempty_lead_owners[0]
            if len(first_lead_owners) == 1 and all(
                owners == first_lead_owners for owners in nonempty_lead_owners
            ):
                contact_owners = set(first_lead_owners)
                match_key = "wappi_widget_lead"
        if len(contact_owners) != 1 or any(owners and owners != contact_owners for owners in lead_owner_sets):
            self._widget_missing_personal_chats += 1
            return WappiChatResolution(
                status="pending_attribution",
                lead_id=lead_ids[0] if lead_ids else "",
                lead_ids=tuple(lead_ids),
                contact_id=contact_id,
                expected_brand=profile.brand,
                reason="wappi_widget_timeline_identity_missing_or_conflicting",
                candidate_customer_ids=tuple(sorted(all_owners)),
                resolution_source=source,
                match_key=match_key,
            )
        customer_id = next(iter(contact_owners))
        customer_brand = self._customer_brands.get(customer_id, "unknown")
        brand_authorized = customer_brand == profile.brand
        return WappiChatResolution(
            status="resolved",
            customer_id=customer_id,
            lead_id=lead_ids[0] if lead_ids else "",
            lead_ids=tuple(lead_ids),
            contact_id=contact_id,
            expected_brand=profile.brand,
            pair_source=source,
            resolution_source=source,
            match_key=match_key,
            reason=(
                "wappi_widget_unique_cross_brand_person_match"
                if customer_brand != "unknown" and not brand_authorized
                else ""
            ),
            evidence={
                "lead_count": len(lead_ids),
                "customer_brand": customer_brand,
                "profile_brand": profile.brand,
                "brand_context_authorized": brand_authorized,
            },
        )

    def _guard_pair_context(
        self,
        profile: WappiProfileSpec,
        chat_id: str,
        dialog: Mapping[str, Any],
        resolution: WappiChatResolution,
    ) -> WappiChatResolution:
        if not resolution.resolved:
            return resolution
        if profile.channel == "telegram" and (
            not chat_id.isdigit()
            or str(dialog.get("type") or "").casefold() not in {"user", "private", "personal"}
        ):
            reason = "draft_loop_pair_non_personal_chat"
            identity_kind, identity_value = "", ""
        elif profile.channel == "max" and not consistent_max_dialog_phone(dialog)[0]:
            reason = "draft_loop_pair_max_phone_missing_or_ambiguous"
            identity_kind, identity_value = "", ""
        else:
            identity_kind, identity_value = (
                ("telegram_user_id", chat_id)
                if profile.channel == "telegram"
                else ("phone", consistent_max_dialog_phone(dialog)[0])
            )
            identity_key = (identity_kind, identity_value)
            identity_owners = self._local_identity_customers.get(identity_key, ())
            if identity_key in self._ambiguous_identity_values:
                reason = "draft_loop_pair_identity_ambiguous"
            elif identity_kind == "phone" and self._shared_phone_stoplist_error:
                reason = self._shared_phone_stoplist_error
            elif identity_kind == "phone" and identity_value in self._shared_phone_stoplist:
                reason = "shared_phone"
            elif identity_owners and identity_owners != (str(resolution.customer_id),):
                reason = "draft_loop_pair_identity_customer_conflict"
            elif str(resolution.customer_id) not in self._supported_customer_ids:
                reason = "draft_loop_pair_support_missing"
            else:
                customer_brand = self._customer_brands.get(str(resolution.customer_id), "unknown")
                if customer_brand == profile.brand:
                    return resolution
                reason = "draft_loop_pair_brand_unknown" if customer_brand == "unknown" else "draft_loop_pair_brand_mismatch"
        return WappiChatResolution(
            status="pending_attribution",
            lead_id=resolution.lead_id,
            contact_id=resolution.contact_id,
            expected_brand=profile.brand,
            reason=reason,
            candidate_customer_ids=(str(resolution.customer_id),),
            pair_source=resolution.pair_source,
            resolution_source="draft_loop_pair",
        )

    def _guard_chat_customer(
        self,
        profile: WappiProfileSpec,
        chat_id: str,
        resolution: WappiChatResolution,
    ) -> WappiChatResolution:
        if not resolution.resolved:
            return resolution
        owners = self._chat_customer_ids.get((profile.source_system, profile.profile_id, chat_id), ())
        if not owners or owners == (resolution.customer_id,):
            return resolution
        exact_owners = self._exact_chat_customer_ids.get(
            (profile.source_system, profile.profile_id, chat_id),
            (),
        )
        exact_owner_conflict = bool(exact_owners and exact_owners != (resolution.customer_id,))
        if resolution.resolution_source != "wappi_provisional" and set(owners).issubset(
            self._provisional_customer_ids
        ) and not exact_owner_conflict:
            return resolution
        if resolution.resolution_source in WAPPI_EXACT_AMO_AUTHORITIES and (
            not exact_owners or exact_owners == (resolution.customer_id,)
        ):
            return resolution
        return WappiChatResolution(
            status="pending_attribution",
            expected_brand=profile.brand,
            reason="existing_wappi_chat_customer_conflict",
            candidate_customer_ids=tuple(sorted(set(owners) | {str(resolution.customer_id)})),
            resolution_source=resolution.resolution_source,
            match_key=resolution.match_key,
        )

    def _resolve_with_timeline_identity(
        self,
        *,
        profile: WappiProfileSpec,
        chat_id: str,
        dialog: Mapping[str, Any],
    ) -> WappiChatResolution | None:
        strong_keys, weak_keys, key_error = wappi_dialog_identity_keys(profile, chat_id, dialog)
        if key_error:
            return WappiChatResolution(
                status="pending_attribution",
                expected_brand=profile.brand,
                reason=key_error,
                resolution_source="timeline_identity",
            )
        matched: list[tuple[str, str, tuple[str, ...]]] = []
        for identity_kind, identity_value in strong_keys:
            if (identity_kind, identity_value) in self._ambiguous_identity_values:
                return WappiChatResolution(
                    status="pending_attribution",
                    expected_brand=profile.brand,
                    reason="timeline_identity_ambiguous_value",
                    resolution_source="timeline_identity",
                    match_key=identity_kind,
                )
            if identity_kind == "phone" and self._shared_phone_stoplist_error:
                return WappiChatResolution(
                    status="pending_attribution",
                    expected_brand=profile.brand,
                    reason=self._shared_phone_stoplist_error,
                    resolution_source="timeline_identity",
                    match_key=identity_kind,
                )
            if identity_kind == "phone" and identity_value in self._shared_phone_stoplist:
                return WappiChatResolution(
                    status="pending_attribution",
                    expected_brand=profile.brand,
                    reason="shared_phone",
                    resolution_source="timeline_identity",
                    match_key=identity_kind,
                )
            owners = self._local_identity_customers.get((identity_kind, identity_value), ())
            if owners:
                matched.append((identity_kind, identity_value, owners))
        if not matched:
            return None
        customer_ids = tuple(sorted({customer_id for _, _, owners in matched for customer_id in owners}))
        if len(customer_ids) != 1:
            return WappiChatResolution(
                status="pending_attribution",
                expected_brand=profile.brand,
                reason="timeline_identity_signal_conflict",
                candidate_customer_ids=customer_ids,
                resolution_source="timeline_identity",
                match_key="+".join(sorted({kind for kind, _, _ in matched})),
            )
        customer_id = customer_ids[0]
        for identity_kind, identity_value in weak_keys:
            weak_owners = self._local_identity_customers.get((identity_kind, identity_value), ())
            if weak_owners and weak_owners != (customer_id,):
                return WappiChatResolution(
                    status="pending_attribution",
                    expected_brand=profile.brand,
                    reason="timeline_identity_weak_signal_conflict",
                    candidate_customer_ids=tuple(sorted({customer_id, *weak_owners})),
                    resolution_source="timeline_identity",
                    match_key=identity_kind,
                )
        match_key = "+".join(sorted({kind for kind, _, _ in matched}))
        if customer_id not in self._supported_customer_ids:
            return WappiChatResolution(
                status="pending_attribution",
                expected_brand=profile.brand,
                reason="timeline_identity_support_missing",
                candidate_customer_ids=(customer_id,),
                resolution_source="timeline_identity",
                match_key=match_key,
            )
        customer_brand = self._customer_brands.get(customer_id, "unknown")
        reason = "timeline_identity_unique_brand_match"
        if customer_brand == "unknown":
            reason = "timeline_identity_unique_brand_unverified"
        elif customer_brand != profile.brand:
            reason = "timeline_identity_unique_cross_brand_person_match"
        return WappiChatResolution(
            status="resolved",
            customer_id=customer_id,
            expected_brand=profile.brand,
            reason=reason,
            resolution_source="timeline_identity",
            match_key=match_key,
            evidence={
                "customer_brand": customer_brand,
                "profile_brand": profile.brand,
                "brand_context_authorized": customer_brand == profile.brand,
            },
        )

    def _resolve_with_amo_auto(
        self,
        *,
        profile: WappiProfileSpec,
        chat_id: str,
        dialog: Mapping[str, Any],
        messages: Sequence[WappiHistoryMessage],
    ) -> WappiChatResolution:
        if not chat_id:
            return WappiChatResolution(status="pending_attribution", expected_brand=profile.brand, reason="chat_id_missing", resolution_source="amo_auto_resolver")
        key = DraftLoopKey(profile.profile_id, chat_id)
        draft_profile = DraftLoopProfile(profile_id=profile.profile_id, brand=profile.brand, channel=profile.channel)
        try:
            auto_result = self._amo_auto_resolver(
                key=key,
                profile=draft_profile,
                dialog=dialog,
                messages=messages,
                message=messages[-1] if messages else None,
                identity_only=True,
            )
        except Exception:  # noqa: BLE001 - keep the batch moving without exposing AMO payloads.
            return WappiChatResolution(
                status="pending_attribution",
                expected_brand=profile.brand,
                reason="amo_auto_lookup_error",
                resolution_source="amo_auto_resolver",
            )
        status = str(auto_result.get("status") or "").strip()
        reason = str(auto_result.get("reason") or status or "amo_auto_unresolved").strip()
        lead_id = str(auto_result.get("lead_id") or "").strip()
        contact_id = str(auto_result.get("contact_id") or "").strip()
        match_key = str(auto_result.get("match_key") or "").strip()
        if status != "matched":
            if contact_id and match_key:
                identity_resolution = self._resolve_amo_candidate_to_customer(
                    profile=profile,
                    lead_id=lead_id,
                    contact_id=contact_id,
                    match_key=match_key,
                    auto_result=auto_result,
                    identity_only=True,
                )
                if identity_resolution.resolved or identity_resolution.candidate_customer_ids:
                    return identity_resolution
            lead_snapshot = auto_result.get("lead_snapshot") if isinstance(auto_result.get("lead_snapshot"), Mapping) else {}
            organization_values = auto_result.get("organization_values") or lead_snapshot.get("organization_values") or ()
            return WappiChatResolution(
                status="pending_attribution",
                lead_id=lead_id,
                contact_id=contact_id,
                expected_brand=profile.brand,
                reason=reason,
                resolution_source="amo_auto_resolver",
                match_key=match_key,
                evidence={
                    "organization_brand": str(auto_result.get("organization_brand") or lead_snapshot.get("organization_brand") or ""),
                    "organization_value_count": len(organization_values) if isinstance(organization_values, Sequence) and not isinstance(organization_values, (str, bytes, bytearray)) else 0,
                },
            )
        return self._resolve_amo_candidate_to_customer(
            profile=profile,
            lead_id=lead_id,
            contact_id=contact_id,
            match_key=match_key,
            auto_result=auto_result,
        )

    def _resolve_amo_candidate_to_customer(
        self,
        *,
        profile: WappiProfileSpec,
        lead_id: str,
        contact_id: str,
        match_key: str,
        auto_result: Mapping[str, Any],
        identity_only: bool = False,
    ) -> WappiChatResolution:
        if not self._db_path.exists():
            return WappiChatResolution(
                status="pending_attribution",
                lead_id=lead_id,
                contact_id=contact_id,
                expected_brand=profile.brand,
                reason="amo_auto_no_timeline_db",
                resolution_source="amo_auto_resolver",
                match_key=match_key,
            )
        contact_ids = set(self._local_identity_customers.get(("amo_contact_id", contact_id), ()))
        lead_ids = set(self._local_identity_customers.get(("amo_lead_id", lead_id), ()))
        opportunity_ids: set[str] = set()
        opportunity_id = ""
        if lead_id and not identity_only:
            with open_readonly_sqlite(self._db_path) as con:
                opportunity_ids, opportunity_id = lookup_amo_opportunity_customers(
                    con,
                    tenant_id=self._tenant_id,
                    lead_id=lead_id,
                )
        if identity_only:
            lead_ids = set()
        candidate_sets = [items for items in (lead_ids, contact_ids, opportunity_ids) if items]
        candidate_union = set().union(*candidate_sets) if candidate_sets else set()
        lead_snapshot = auto_result.get("lead_snapshot") if isinstance(auto_result.get("lead_snapshot"), Mapping) else {}
        organization_values = lead_snapshot.get("organization_values") or ()
        evidence = {
            "exact_match_kind": match_key,
            "single_active_lead": not identity_only,
            "identity_only": identity_only,
            "organization_brand": str(lead_snapshot.get("organization_brand") or ""),
            "organization_value_count": len(organization_values) if isinstance(organization_values, Sequence) and not isinstance(organization_values, (str, bytes, bytearray)) else 0,
            "timeline_identity_sources": tuple(
                source
                for source, values in (
                    ("amo_lead_id", lead_ids),
                    ("amo_contact_id", contact_ids),
                    ("amo_opportunity", opportunity_ids),
                )
                if values
            ),
        }
        if candidate_sets and all(items == candidate_sets[0] for items in candidate_sets) and len(candidate_union) == 1:
            customer_id = next(iter(candidate_union))
            customer_brand = self._customer_brands.get(customer_id, "unknown")
            evidence.update(
                {
                    "customer_brand": customer_brand,
                    "profile_brand": profile.brand,
                    "brand_context_authorized": customer_brand == profile.brand,
                }
            )
            return WappiChatResolution(
                status="resolved",
                customer_id=customer_id,
                opportunity_id=opportunity_id or None,
                lead_id=lead_id,
                contact_id=contact_id,
                expected_brand=profile.brand,
                reason="amo_auto_exact_identity_without_opportunity" if identity_only else "",
                pair_source="amo_auto_resolver",
                resolution_source="amo_auto_resolver",
                match_key=match_key,
                evidence=evidence,
            )
        if len(candidate_union) > 1:
            return WappiChatResolution(
                status="pending_attribution",
                lead_id=lead_id,
                contact_id=contact_id,
                expected_brand=profile.brand,
                reason="amo_auto_matches_multiple_or_conflicting_customers",
                candidate_customer_ids=tuple(sorted(candidate_union)),
                pair_source="amo_auto_resolver",
                resolution_source="amo_auto_resolver",
                match_key=match_key,
                evidence=evidence,
            )
        return WappiChatResolution(
            status="pending_attribution",
            lead_id=lead_id,
            contact_id=contact_id,
            expected_brand=profile.brand,
            reason="amo_auto_has_no_customer_in_timeline",
            pair_source="amo_auto_resolver",
            resolution_source="amo_auto_resolver",
            match_key=match_key,
            evidence=evidence,
        )


def fetch_wappi_history_records(
    *,
    client: WappiHistoryClient,
    profiles: Sequence[WappiProfileSpec],
    resolver: WappiPairCustomerResolver,
    limits: WappiFetchLimits,
    tenant_id: str,
    checkpoint: Optional[Mapping[str, Any]] = None,
    next_checkpoint: Optional[dict[str, Any]] = None,
) -> tuple[tuple[TimelineSourceRecord, ...], dict[tuple[str, str], WappiFetchStats]]:
    # next_checkpoint is an out-parameter (same shape as amo_incremental's
    # fetch_endpoint_checkpointed) so the existing 2-tuple contract is unchanged.
    checkpoint_enabled = next_checkpoint is not None
    checkpoint_profiles = (checkpoint or {}).get("profiles") if isinstance(checkpoint, Mapping) else None
    records: list[TimelineSourceRecord] = []
    stats_by_profile: dict[tuple[str, str], WappiFetchStats] = {
        (profile.source_system, profile.profile_id): WappiFetchStats()
        for profile in profiles
    }
    seen_source_ids: set[str] = set()
    total_messages = 0
    total_requests = 0
    per_profile_message_limit = (
        0
        if limits.complete_message_history
        else max(1, limits.message_limit_total // max(1, len(profiles)))
        if limits.message_limit_total
        else 0
    )
    for profile_index, profile in enumerate(profiles):
        stats = stats_by_profile[(profile.source_system, profile.profile_id)]
        profile_amo_calls_start = resolver.amo_auto_calls
        # Without a checkpoint the budget stays global (unchanged behaviour). With one,
        # the REMAINING budget is split across the REMAINING profiles: otherwise the
        # alphabetically first profile eats everything every night and the second
        # channel never loads at all. Unused share rolls over to the next profile.
        profile_budget_limit = limits.request_limit_total
        if checkpoint_enabled:
            share = max(
                1,
                (limits.request_limit_total - total_requests) // max(1, len(profiles) - profile_index),
            )
            profile_budget_limit = min(limits.request_limit_total, total_requests + share)
        offset = 0
        profile_messages = 0
        chat_ids_seen: set[str] = set()
        dialogs_snapshot: list[Mapping[str, Any]] = []
        chat_page_specs: list[tuple[int, int]] = []
        checkpoint_key = f"{profile.source_system}:{profile.profile_id}"
        fingerprint = wappi_fetch_universe_fingerprint(profile, limits, tenant_id=tenant_id)
        entry, reset_reason = _wappi_resume_entry(checkpoint_profiles, checkpoint_key, fingerprint)
        full_audit_due, clock_anomaly = _wappi_full_audit_state(entry)
        incremental_base_complete = bool(
            (entry.get("complete") or entry.get("incremental_cycle")) and not full_audit_due
        )
        stats.full_history_audit = not incremental_base_complete
        stats.full_audit_clock_anomaly = clock_anomaly
        full_audit_at = str(entry.get("full_audit_at") or "")
        full_audit_started_at = str(entry.get("full_audit_started_at") or "")
        full_audit_markers = {
            str(key): _wappi_safe_int(value) or 0
            for key, value in (entry.get("full_audit_markers") or {}).items()
            if isinstance(key, str) and _wappi_safe_int(value) is not None
        }
        if not incremental_base_complete and entry.get("complete"):
            full_audit_markers = {}
            full_audit_started_at = datetime.now(timezone.utc).isoformat()
        elif not incremental_base_complete and not full_audit_started_at:
            full_audit_started_at = datetime.now(timezone.utc).isoformat()
        confirmed_tokens: set[str] = {str(item) for item in (entry.get("chats_done") or ())}
        confirmed_at_start = set(confirmed_tokens)
        chat_markers = {
            str(key): _wappi_safe_int(value) or 0
            for key, value in (entry.get("chat_markers") or {}).items()
            if isinstance(key, str) and _wappi_safe_int(value) is not None
        }
        tail_checked: set[str] = set()
        resumed_from = len(confirmed_tokens)
        active_chat = entry.get("active_chat") if isinstance(entry.get("active_chat"), Mapping) else {}
        catalog_page_tokens: list[tuple[int, tuple[str, ...]]] = []
        catalog_total: Optional[int] = None
        catalog_complete = False
        stop_reason = "source_exhausted"
        while (
            (limits.complete_message_history or len(dialogs_snapshot) < limits.chat_limit_per_profile)
            and total_requests < profile_budget_limit
        ):
            page_limit = (
                limits.page_size
                if limits.complete_message_history
                else min(limits.page_size, limits.chat_limit_per_profile - len(dialogs_snapshot))
            )
            if page_limit <= 0:
                break
            try:
                payload = client.list_chats(
                    channel=profile.channel,
                    profile_id=profile.profile_id,
                    limit=page_limit,
                    offset=offset,
                    order="asc",
                    show_all=limits.show_all_chats,
                )
            except WappiPhysicalRequestBudgetExceeded:
                stats.request_limit_hit = True
                stop_reason = "request_budget"
                break
            except AmoWappiHttpError:
                if not checkpoint_enabled:
                    raise
                stats.checkpoint_network_error = True
                stop_reason = "network_error"
                break
            total_requests += 1
            stats.requests += 1
            sleep_if_needed(limits.sleep_seconds)
            dialogs = extract_wappi_items(payload, "dialogs", "chats", "items", "data")
            total_present, observed_total = _extract_wappi_total_count(payload)
            if total_present and observed_total is None:
                stats.pagination_drift_detected = True
                stop_reason = "pagination_drift"
                break
            if observed_total is not None:
                catalog_total = max(catalog_total or 0, observed_total)
            chat_page_specs.append((offset, page_limit))
            catalog_page_tokens.append(
                (offset, tuple(wappi_checkpoint_token(extract_chat_id(item)) for item in dialogs))
            )
            if not dialogs:
                catalog_complete = catalog_total is None or offset >= catalog_total
                if not catalog_complete:
                    stats.pagination_drift_detected = True
                    stop_reason = "pagination_drift"
                break
            stats.chats_seen += len(dialogs)
            for dialog in dialogs:
                if not limits.complete_message_history and len(dialogs_snapshot) >= limits.chat_limit_per_profile:
                    break
                chat_id = extract_chat_id(dialog)
                if not chat_id:
                    stats.skipped_chat_id_missing += 1
                    continue
                if chat_id in chat_ids_seen:
                    stats.duplicate_chat_ids += 1
                    continue
                chat_ids_seen.add(chat_id)
                dialogs_snapshot.append(dialog)
                stats.chats_loaded += 1
            next_offset = offset + len(dialogs)
            if catalog_total is not None and next_offset >= catalog_total:
                catalog_complete = True
                break
            if len(dialogs) < page_limit and catalog_total is None:
                catalog_complete = True
                break
            offset = next_offset
        verification_chat_ids: set[str] = set()
        for verification_offset, verification_limit in chat_page_specs:
            if total_requests >= profile_budget_limit:
                stats.request_limit_hit = True
                stop_reason = "request_budget"
                break
            try:
                verification_payload = client.list_chats(
                    channel=profile.channel,
                    profile_id=profile.profile_id,
                    limit=verification_limit,
                    offset=verification_offset,
                    order="asc",
                    show_all=limits.show_all_chats,
                )
            except WappiPhysicalRequestBudgetExceeded:
                stats.request_limit_hit = True
                stop_reason = "request_budget"
                break
            except AmoWappiHttpError:
                if not checkpoint_enabled:
                    raise
                stats.checkpoint_network_error = True
                stop_reason = "network_error"
                break
            else:
                total_requests += 1
                stats.requests += 1
                sleep_if_needed(limits.sleep_seconds)
                verification_dialogs = extract_wappi_items(
                    verification_payload,
                    "dialogs",
                    "chats",
                    "items",
                    "data",
                )
                for item in verification_dialogs:
                    chat_id = extract_chat_id(item)
                    if chat_id and chat_id in verification_chat_ids:
                        stats.duplicate_chat_ids += 1
                    elif chat_id:
                        verification_chat_ids.add(chat_id)
                        if chat_id not in chat_ids_seen:
                            chat_ids_seen.add(chat_id)
                            dialogs_snapshot.append(item)
                            stats.chats_loaded += 1
        stats.chat_snapshot_drift_detected = chat_ids_seen != verification_chat_ids
        if checkpoint_enabled and limits.complete_message_history and catalog_total is None:
            stats.pagination_drift_detected = True
            stop_reason = "pagination_drift"
        if not stats.chat_snapshot_drift_detected and catalog_complete and catalog_total is not None and (
            len(chat_ids_seen) != catalog_total or len(verification_chat_ids) != catalog_total
        ):
            stats.pagination_drift_detected = True
            stop_reason = "pagination_drift"
        if entry and not incremental_base_complete and entry.get("catalog_page_anchor") and not reset_reason and catalog_complete:
            current_catalog_anchor = wappi_checkpoint_anchor(
                tuple(token for _offset, page_tokens in catalog_page_tokens for token in page_tokens)
            )
            if current_catalog_anchor != entry.get("catalog_page_anchor"):
                reset_reason = "catalog_page_drift"
                tail_checked.clear()
        if not limits.complete_message_history and len(dialogs_snapshot) >= limits.chat_limit_per_profile:
            stats.chat_limit_hit = True
        if total_requests >= profile_budget_limit:
            stats.request_limit_hit = True
            stats.checkpoint_no_progress = checkpoint_enabled and bool(dialogs_snapshot)
            stop_reason = "request_budget"
            _stash_wappi_checkpoint(
                next_checkpoint, checkpoint_key, fingerprint=fingerprint, confirmed=confirmed_tokens,
                active_chat=active_chat, catalog_pages=catalog_page_tokens,
                chat_markers=chat_markers,
                incremental_cycle=incremental_base_complete,
                full_audit_at=full_audit_at,
                full_audit_started_at=full_audit_started_at,
                full_audit_markers=full_audit_markers,
                chats_total=len(dialogs_snapshot),
                resumed_from=resumed_from, complete=False, stop_reason=stop_reason, reset_reason=reset_reason,
            )
            if checkpoint_enabled:
                continue
            break
        if not limits.complete_message_history and total_messages >= limits.message_limit_total:
            break
        profile_widget_calls_start = resolver.widget_calls
        full_audit_verified = {
            wappi_checkpoint_token(extract_chat_id(item))
            for item in dialogs_snapshot
            if extract_chat_id(item)
            and wappi_checkpoint_token(extract_chat_id(item)) in full_audit_markers
            and full_audit_markers[wappi_checkpoint_token(extract_chat_id(item))]
            == _safe_int(item.get("last_timestamp"))
        }
        dialogs_snapshot.sort(key=lambda item: (
            wappi_checkpoint_token(extract_chat_id(item)) in
            (confirmed_at_start if incremental_base_complete else full_audit_verified)
        ))
        tail_required: set[str] = set()
        for dialog in dialogs_snapshot:
            if not limits.complete_message_history and (
                total_messages >= limits.message_limit_total or profile_messages >= per_profile_message_limit
            ):
                stats.message_limit_hit = True
                break
            chat_id = extract_chat_id(dialog)
            chat_token = wappi_checkpoint_token(chat_id) if chat_id else ""
            dialog_marker = _safe_int(dialog.get("last_timestamp"))
            saved_marker = chat_markers.get(chat_token)
            if not incremental_base_complete and chat_token in full_audit_verified:
                stats.checkpoint_chats_skipped += 1
                continue
            if incremental_base_complete and chat_token and dialog_marker > 0 and saved_marker == dialog_marker:
                stats.incremental_chats_skipped += 1
                continue
            if incremental_base_complete:
                if chat_token not in confirmed_at_start:
                    stats.incremental_chats_new += 1
                elif saved_marker is None or dialog_marker <= 0:
                    stats.incremental_chats_without_marker += 1
                elif dialog_marker < saved_marker:
                    stats.incremental_chats_marker_regressed += 1
                else:
                    stats.incremental_chats_changed += 1
            is_tail_check = bool(incremental_base_complete and chat_token and chat_token in confirmed_at_start)
            if is_tail_check and chat_token in tail_checked:
                stats.checkpoint_chats_skipped += 1
                continue
            resume_offset = (
                int(active_chat.get("message_offset") or 0)
                if checkpoint_enabled and active_chat and str(active_chat.get("chat") or "") == chat_token
                else 0
            )
            resolver.record_coverage(profile=profile, dialog=dialog, stats=stats)
            try:
                messages = fetch_chat_messages(
                    client,
                    profile=profile,
                    chat_id=chat_id,
                    limits=limits,
                    request_counter=stats,
                    request_budget=max(0, profile_budget_limit - total_requests),
                    start_offset=resume_offset,
                    resume_anchor=str(active_chat.get("page_anchor") or "") if resume_offset else "",
                    resume_anchor_offset=(
                        _wappi_safe_int(active_chat.get("page_offset"))
                        if resume_offset and "page_offset" in active_chat
                        else None
                    ),
                )
            except AmoWappiHttpError:
                if not checkpoint_enabled:
                    raise
                stats.checkpoint_network_error = True
                stop_reason = "network_error"
                break
            total_requests += int(getattr(fetch_chat_messages, "last_request_count", 0))
            stop_after_chat = False
            if bool(getattr(fetch_chat_messages, "last_request_limit_hit", False)):
                stats.request_limit_hit = True
                stop_reason = "request_budget"
                if not checkpoint_enabled:
                    break
                # With a checkpoint the partially read chat is still written and its
                # resume offset is remembered: source_id keeps the write idempotent.
                stop_after_chat = True
            if bool(getattr(fetch_chat_messages, "last_limit_hit", False)):
                stats.message_limit_hit = True
            if bool(getattr(fetch_chat_messages, "last_pagination_drift_detected", False)):
                stats.message_page_drift_detected = True
                stats.pagination_drift_detected = True
                break
            resolution = resolver.resolve_chat(profile=profile, dialog=dialog, messages=messages)
            stats.amo_auto_calls = resolver.amo_auto_calls - profile_amo_calls_start
            stats.widget_calls = resolver.widget_calls - profile_widget_calls_start
            if is_personal_wappi_dialog(profile, dialog):
                stats.personal_chats += 1
                if resolution.resolved and resolution.resolution_source in {
                    "wappi_amo_widget",
                    "amo_talk_authoritative",
                }:
                    stats.widget_resolved_chats += 1
                else:
                    stats.widget_pending_chats += 1
            stats.amo_auto_status_counts[f"{resolution.resolution_source}:{resolution.reason or resolution.status}"] += 1
            for message in messages:
                if not limits.complete_message_history and (
                    total_messages >= limits.message_limit_total or profile_messages >= per_profile_message_limit
                ):
                    stats.message_limit_hit = True
                    break
                stats.messages_seen += 1
                if not message.text.strip():
                    stats.skipped_empty += 1
                    continue
                source_id = wappi_source_id(profile, message)
                if source_id in seen_source_ids:
                    stats.duplicate_source_ids += 1
                    continue
                seen_source_ids.add(source_id)
                records.append(wappi_message_to_record(profile=profile, message=message, resolution=resolution))
                total_messages += 1
                profile_messages += 1
                stats.records_built += 1
                stats.resolution_status_counts[resolution.reason or resolution.status] += 1
                if resolution.resolved:
                    if resolution.resolution_source == "amo_auto_resolver":
                        stats.linked_by_amo_auto += 1
                    elif resolution.resolution_source == "wappi_amo_widget":
                        stats.linked_by_amo_widget += 1
                    elif resolution.resolution_source == "amo_talk_authoritative":
                        stats.linked_by_amo_talk += 1
                    elif resolution.resolution_source == "timeline_identity":
                        stats.linked_by_timeline += 1
                    elif resolution.resolution_source == "wappi_provisional":
                        stats.linked_by_provisional += 1
                        stats.pending_attribution += 1
                    else:
                        stats.linked_by_pair += 1
                else:
                    stats.pending_attribution += 1
            if checkpoint_enabled and chat_token:
                if stop_after_chat:
                    active_chat = {
                        "chat": chat_token,
                        "message_offset": int(getattr(fetch_chat_messages, "last_next_offset", 0)),
                        "page_anchor": str(getattr(fetch_chat_messages, "last_page_anchor", "")),
                        "page_offset": int(getattr(fetch_chat_messages, "last_page_offset", 0)),
                    }
                elif is_tail_check:
                    tail_checked.add(chat_token)
                    active_chat = {}
                else:
                    confirmed_tokens.add(chat_token)
                    stats.checkpoint_chats_confirmed += 1
                    active_chat = {}
                if not stop_after_chat and dialog_marker > 0:
                    chat_markers[chat_token] = dialog_marker
                if not stop_after_chat and not incremental_base_complete:
                    full_audit_markers[chat_token] = max(0, dialog_marker)
                    full_audit_verified.add(chat_token)
            if stats.request_limit_hit:
                break
        if checkpoint_enabled:
            current_tokens = {
                wappi_checkpoint_token(extract_chat_id(item))
                for item in dialogs_snapshot
                if extract_chat_id(item)
            }
            profile_complete = not (
                stats.request_limit_hit
                or stats.checkpoint_network_error
                or stats.message_limit_hit
                or stats.chat_limit_hit
                or stats.pagination_drift_detected
                or stats.chat_snapshot_drift_detected
                or not catalog_complete
                or not tail_required.issubset(tail_checked)
                or (not incremental_base_complete and not current_tokens.issubset(full_audit_verified))
            )
            if stats.pagination_drift_detected:
                stop_reason = "pagination_drift"
            if stats.chat_snapshot_drift_detected:
                stop_reason = "catalog_drift"
            if profile_complete:
                confirmed_tokens.intersection_update(current_tokens)
                chat_markers = {key: value for key, value in chat_markers.items() if key in current_tokens}
                if not incremental_base_complete:
                    full_audit_at = datetime.now(timezone.utc).isoformat()
                    full_audit_started_at = ""
                    full_audit_markers = {}
            _stash_wappi_checkpoint(
                next_checkpoint,
                checkpoint_key,
                fingerprint=fingerprint,
                confirmed=confirmed_tokens,
                active_chat=active_chat,
                catalog_pages=catalog_page_tokens,
                chat_markers=chat_markers,
                incremental_cycle=incremental_base_complete,
                full_audit_at=full_audit_at,
                full_audit_started_at=full_audit_started_at,
                full_audit_markers=full_audit_markers,
                chats_total=len(dialogs_snapshot),
                resumed_from=resumed_from,
                complete=profile_complete,
                stop_reason="source_exhausted" if profile_complete else stop_reason,
                reset_reason=reset_reason,
            )
    return tuple(records), stats_by_profile


def _wappi_resume_entry(
    checkpoint_profiles: Any, key: str, fingerprint: str
) -> tuple[dict[str, Any], Optional[str]]:
    entry = checkpoint_profiles.get(key) if isinstance(checkpoint_profiles, Mapping) else None
    if not isinstance(entry, Mapping):
        return {}, None
    if entry.get("fingerprint") != fingerprint:
        return {}, "fingerprint_changed"
    return dict(entry), None


def _stash_wappi_checkpoint(
    next_checkpoint: Optional[dict[str, Any]],
    key: str,
    *,
    fingerprint: str,
    confirmed: set[str],
    active_chat: Mapping[str, Any],
    catalog_pages: Sequence[tuple[int, tuple[str, ...]]],
    chat_markers: Optional[Mapping[str, int]] = None,
    incremental_cycle: bool = False,
    full_audit_at: str = "",
    full_audit_started_at: str = "",
    full_audit_markers: Optional[Mapping[str, int]] = None,
    chats_total: int,
    resumed_from: int,
    complete: bool,
    stop_reason: str,
    reset_reason: Optional[str],
) -> None:
    if next_checkpoint is None:
        return
    last_offset, last_tokens = catalog_pages[-1] if catalog_pages else (0, ())
    catalog_tokens = tuple(token for _offset, page_tokens in catalog_pages for token in page_tokens)
    next_checkpoint[key] = {
        "fingerprint": fingerprint,
        "complete": bool(complete),
        "incremental_cycle": bool(incremental_cycle),
        "full_audit_at": full_audit_at,
        "full_audit_started_at": full_audit_started_at,
        "full_audit_markers": dict(sorted((full_audit_markers or {}).items())),
        "stop_reason": stop_reason,
        "reset_reason": reset_reason,
        "catalog_next_offset": int(last_offset) + len(last_tokens),
        "catalog_page_anchor": wappi_checkpoint_anchor(catalog_tokens),
        "catalog_chats_seen": int(chats_total),
        "resumed_from": int(resumed_from),
        "chats_done": sorted(confirmed),
        "chat_markers": dict(sorted((chat_markers or {}).items())),
        "active_chat": dict(active_chat) if active_chat else None,
    }


def is_personal_wappi_dialog(profile: WappiProfileSpec, dialog: Mapping[str, Any]) -> bool:
    dialog_type = str(dialog.get("type") or "").strip()
    if profile.channel == "telegram":
        if dialog_type.casefold() not in {"user", "private", "personal"}:
            return False
        user = dialog.get("user") if isinstance(dialog.get("user"), Mapping) else {}
        return not any(
            _wappi_truthy_flag(user, *keys)
            for keys in (
                ("IsBot", "is_bot"),
                ("IsDeleted", "is_deleted"),
                ("IsFake", "is_fake"),
                ("IsSelf", "is_self"),
                ("IsSupport", "is_support"),
            )
        )
    if profile.channel != "max" or dialog_type.upper() != "DIALOG":
        return False
    participants = tuple(item for item in (dialog.get("participants") or ()) if isinstance(item, Mapping))
    peers = tuple(item for item in participants if not _wappi_truthy_flag(item, "is_me", "IsMe"))
    if participants and len(peers) != 1:
        return False
    return not any(
        _wappi_truthy_flag(item, "is_bot", "IsBot", "bot")
        for item in (dialog, *peers)
    )


def _wappi_truthy_flag(payload: Mapping[str, Any], *keys: str) -> bool:
    for key in keys:
        value = payload.get(key)
        if value is True or str(value or "").strip().casefold() in {"1", "true", "yes", "on"}:
            return True
    return False


def fetch_chat_messages(
    client: WappiHistoryClient,
    *,
    profile: WappiProfileSpec,
    chat_id: str,
    limits: WappiFetchLimits,
    request_counter: WappiFetchStats,
    request_budget: int,
    start_offset: int = 0,
    resume_anchor: str = "",
    resume_anchor_offset: Optional[int] = None,
) -> tuple[WappiHistoryMessage, ...]:
    messages: list[WappiHistoryMessage] = []
    offset = max(0, int(start_offset))
    request_count = 0
    limit_hit = False
    request_limit_hit = request_budget <= 0
    pagination_drift_detected = False
    page_anchor_value = resume_anchor
    page_anchor_offset = max(0, int(resume_anchor_offset or 0))
    next_offset = offset
    page_signatures: list[tuple[int, int, tuple[str, ...]]] = []
    if offset > 0 and resume_anchor and not request_limit_hit:
        # Re-read the last CONFIRMED message page: if it drifted, the saved
        # offset points at the wrong place, so restart this chat from zero.
        anchor_offset = (
            max(0, int(resume_anchor_offset))
            if resume_anchor_offset is not None
            else max(0, offset - limits.page_size)
        )
        anchor_payload = client.get_chat_messages(
            channel=profile.channel,
            profile_id=profile.profile_id,
            chat_id=chat_id,
            limit=limits.page_size,
            offset=anchor_offset,
            order="asc",
            mark_all=False,
        )
        request_count += 1
        request_counter.requests += 1
        anchor_items = extract_wappi_items(anchor_payload, "messages", "items", "data")
        current_anchor = wappi_checkpoint_anchor(
            tuple(str(item.get("id") or item.get("message_id") or "") for item in anchor_items)
        )
        if current_anchor != resume_anchor:
            offset = 0
            page_anchor_value = ""
        if request_count >= request_budget:
            # The anchor probe ate the last request: the chat is NOT finished,
            # otherwise the caller would confirm it with zero messages read.
            request_limit_hit = True
    while (
        (limits.complete_message_history or len(messages) < limits.messages_per_chat)
        and request_count < request_budget
    ):
        page_limit = (
            limits.page_size
            if limits.complete_message_history
            else min(limits.page_size, limits.messages_per_chat - len(messages))
        )
        if page_limit <= 0:
            break
        try:
            payload = client.get_chat_messages(
                channel=profile.channel,
                profile_id=profile.profile_id,
                chat_id=chat_id,
                limit=page_limit,
                offset=offset,
                order="asc",
                mark_all=False,
            )
        except WappiPhysicalRequestBudgetExceeded:
            request_limit_hit = True
            break
        request_count += 1
        request_counter.requests += 1
        sleep_if_needed(limits.sleep_seconds)
        raw_messages = extract_wappi_items(payload, "messages", "items", "data")
        if not raw_messages:
            break
        page_ids = tuple(str(item.get("id") or item.get("message_id") or "") for item in raw_messages)
        page_signatures.append((offset, page_limit, page_ids))
        page_anchor_value = wappi_checkpoint_anchor(page_ids)
        page_anchor_offset = offset
        next_offset = offset + len(raw_messages)
        for raw in raw_messages:
            item = wappi_message_from_raw(profile.profile_id, {**dict(raw), "chat_id": chat_id})
            if item is None:
                request_counter.skipped_bad_message += 1
                continue
            messages.append(item)
        if len(raw_messages) < page_limit:
            break
        offset += page_limit
        if not limits.complete_message_history and len(messages) >= limits.messages_per_chat:
            limit_hit = True
        elif request_count >= request_budget:
            request_limit_hit = True
    for verification_offset, verification_limit, expected_ids in page_signatures:
        if request_count >= request_budget:
            request_limit_hit = True
            break
        try:
            verification_payload = client.get_chat_messages(
                channel=profile.channel,
                profile_id=profile.profile_id,
                chat_id=chat_id,
                limit=verification_limit,
                offset=verification_offset,
                order="asc",
                mark_all=False,
            )
        except WappiPhysicalRequestBudgetExceeded:
            request_limit_hit = True
            break
        request_count += 1
        request_counter.requests += 1
        sleep_if_needed(limits.sleep_seconds)
        verification_items = extract_wappi_items(verification_payload, "messages", "items", "data")
        verification_ids = tuple(str(item.get("id") or item.get("message_id") or "") for item in verification_items)
        if verification_ids[: len(expected_ids)] != expected_ids:
            pagination_drift_detected = True
    setattr(fetch_chat_messages, "last_request_count", request_count)
    setattr(fetch_chat_messages, "last_limit_hit", limit_hit)
    setattr(fetch_chat_messages, "last_request_limit_hit", request_limit_hit)
    setattr(fetch_chat_messages, "last_pagination_drift_detected", pagination_drift_detected)
    setattr(fetch_chat_messages, "last_next_offset", next_offset)
    setattr(fetch_chat_messages, "last_page_anchor", page_anchor_value)
    setattr(fetch_chat_messages, "last_page_offset", page_anchor_offset)
    return tuple(sorted(messages, key=lambda item: (item.timestamp, item.message_id)))


def wappi_message_to_record(
    *,
    profile: WappiProfileSpec,
    message: WappiHistoryMessage,
    resolution: WappiChatResolution,
) -> TimelineSourceRecord:
    source_system = profile.source_system
    message_sha256 = stable_digest(
        {
            "profile_id": profile.profile_id,
            "chat_id": message.chat_id,
            "message_id": message.message_id,
            "timestamp": message.timestamp,
            "from_me": message.from_me,
            "text": message.text,
        }
    )
    source_id = wappi_source_id(profile, message)
    event_at, event_time_status = safe_wappi_event_time(message.timestamp)
    payload = {
        "source_system": source_system,
        "source_ref": f"{source_system}:{profile.profile_id}:{message.chat_id}:{message.message_id}",
        "channel": profile.channel,
        "brand": profile.brand,
        "profile_id": profile.profile_id,
        "chat_id": message.chat_id,
        "message_id": message.message_id,
        "message_sha256": message_sha256,
        "timeline_source_id": source_id,
        "event_at": event_at.isoformat(),
        "event_time_status": event_time_status,
        "timestamp": message.timestamp,
        "from_me": message.from_me,
        "direction": "outbound" if message.from_me else "inbound",
        "message_type": message.message_type,
        "text": message.text,
        "contact_name": message.contact_name,
        "from_where": message.from_where,
        "allowed_for_bot": False,
        "resolution_status": resolution.status if resolution.resolved else "pending_attribution",
        "resolution_reason": resolution.reason,
        "resolved_customer_id": resolution.customer_id,
        "resolved_opportunity_id": resolution.opportunity_id,
        "lead_id": resolution.lead_id,
        "lead_ids": tuple(resolution.lead_ids),
        "contact_id": resolution.contact_id,
        "pair_source": resolution.pair_source,
        "identity_authority": resolution.resolution_source,
        "match_key": resolution.match_key,
        "candidate_customer_ids": tuple(resolution.candidate_customer_ids),
        "resolution_evidence": dict(resolution.evidence),
        "brand_context_authorized": resolution.evidence.get("brand_context_authorized"),
    }
    return TimelineSourceRecord(
        source_system=source_system,
        source_ref=str(payload["source_ref"]),
        payload=payload,
        observed_at=event_at,
    )


def safe_wappi_event_time(value: Any) -> tuple[datetime, str]:
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    try:
        timestamp = float(value)
        if not math.isfinite(timestamp) or timestamp <= 0:
            return epoch, "invalid_epoch"
        return datetime.fromtimestamp(timestamp, tz=timezone.utc), "source_valid"
    except (TypeError, ValueError, OverflowError, OSError):
        return epoch, "invalid_epoch"


def build_readonly_wappi_client(
    env_file: Path = AMO_WAPPI_ENV_FILE,
    *,
    request_limit_total: int = 500,
) -> WappiPhase1Client:
    load_env_file(env_file)
    config = WappiClientConfig.from_env()
    return WappiPhase1Client(
        config,
        transport=build_wappi_readonly_transport(config, request_limit_total=request_limit_total),
    )


def consistent_max_dialog_phone(dialog: Mapping[str, Any]) -> tuple[str, str]:
    participants = tuple(item for item in (dialog.get("participants") or ()) if isinstance(item, Mapping))
    peers = tuple(item for item in participants if item.get("is_me") is False)
    phone_participants = peers if len(peers) == 1 else participants
    phones = {
        phone
        for phone in (
            normalize_phone(dialog.get("phone") or dialog.get("number") or ""),
            *(
                normalize_phone(item.get("phone") or item.get("number") or "")
                for item in phone_participants
            ),
        )
        if phone
    }
    if len(phones) == 1:
        return next(iter(phones)), "max_consistent_phone"
    return ("", "max_multi_phone" if phones else "max_phone_missing")


def wappi_dialog_identity_keys(
    profile: WappiProfileSpec,
    chat_id: str,
    dialog: Mapping[str, Any],
) -> tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str], ...], str]:
    strong: list[tuple[str, str]] = []
    weak: list[tuple[str, str]] = []
    if profile.channel == "telegram":
        if str(dialog.get("type") or "").casefold() not in {"user", "private", "personal"}:
            return (), (), "timeline_identity_non_personal_chat"
        user = dialog.get("user") if isinstance(dialog.get("user"), Mapping) else {}
        telegram_ids = {
            cleaned
            for value in (chat_id, user.get("ID"), user.get("id"))
            if (cleaned := _canonical_telegram_id(value))
        }
        if len(telegram_ids) > 1:
            return (), (), "timeline_identity_telegram_id_conflict"
        strong.extend(("telegram_user_id", value) for value in telegram_ids)
        phone = normalize_phone(user.get("Phone") or user.get("phone") or "")
        if phone:
            strong.append(("phone", phone))
        username = str(user.get("Username") or user.get("username") or "").strip().lstrip("@").casefold()
        if username:
            weak.append(("telegram_username", username))
    elif profile.channel == "max":
        phone, phone_status = consistent_max_dialog_phone(dialog)
        if phone_status == "max_multi_phone":
            return (), (), "timeline_identity_max_phone_conflict"
        if phone:
            strong.append(("phone", phone))
        participants = tuple(item for item in (dialog.get("participants") or ()) if isinstance(item, Mapping))
        peers = tuple(item for item in participants if item.get("is_me") is False)
        client_participants = peers if len(peers) == 1 else tuple(
            item for item in participants
            if phone and normalize_phone(item.get("phone") or item.get("number") or "") == phone
        )
        max_ids = {
            str(item.get("user_id") or "").strip()
            for item in client_participants
            if str(item.get("user_id") or "").strip()
        }
        if len(max_ids) == 1:
            strong.append(("max_user_id", next(iter(max_ids))))
        # MAX username has its own identity namespace, which the Timeline contract
        # does not model yet. Phone and max_user_id remain the trusted keys.
    return tuple(dict.fromkeys(strong)), tuple(dict.fromkeys(weak)), ""


def _canonical_telegram_id(value: Any) -> str:
    text = str(value or "").strip()
    return text if re.fullmatch(r"[0-9]+", text) else ""


def build_wappi_readonly_transport(
    config: WappiClientConfig,
    *,
    request_limit_total: int = 500,
) -> DefaultDenyTransport:
    wappi_host = url_parse.urlparse(config.base_url).netloc.casefold()
    budget = WappiPhysicalRequestBudget(max(1, int(request_limit_total)))
    return DefaultDenyTransport(
        partial(_readonly_wappi_request_with_backoff, request_budget=budget),
        policy=SafeTransportPolicy(
            wappi_hosts=frozenset(host for host in (wappi_host,) if host),
            amo_read_hosts=frozenset(),
            ai_office_hosts=frozenset(),
        ),
    )


def _readonly_wappi_request_with_backoff(
    *,
    request_budget: WappiPhysicalRequestBudget | None = None,
    **kwargs: Any,
) -> Mapping[str, Any]:
    retry_delays = (0.0, 2.0, 10.0, 30.0, 60.0, 120.0)
    for attempt, delay in enumerate(retry_delays, start=1):
        if delay:
            time.sleep(delay)
        if request_budget is not None:
            request_budget.take()
        try:
            return _json_http_request(**kwargs)
        except AmoWappiHttpError as exc:
            message = str(exc)
            folded = message.casefold()
            temporary_http_400 = "http 400" in folded and any(
                marker in folded for marker in ("driver not ready", "try_again_later", "повторите запрос чуть позже")
            )
            transient = _is_deferred_fetch_exception(exc) or temporary_http_400 or "Request failed:" in message or any(
                f"HTTP {code}" in message for code in (429, 500, 502, 503, 504)
            )
            if not transient or attempt == len(retry_delays):
                raise
    raise AssertionError("unreachable")


def readonly_wappi_physical_request_count(client: WappiHistoryClient) -> int | None:
    inner = getattr(getattr(client, "transport", None), "inner", None)
    budget = getattr(inner, "keywords", {}).get("request_budget") if isinstance(inner, partial) else None
    return int(budget.used) if isinstance(budget, WappiPhysicalRequestBudget) else None


def timeline_db_identity(path: Path) -> Mapping[str, Any]:
    stat = path.stat()
    with open_readonly_sqlite(path) as con:
        logical = {
            "user_version": int(con.execute("PRAGMA user_version").fetchone()[0]),
            "audit_seq": _sqlite_scalar(con, "SELECT COALESCE(MAX(seq), 0) FROM audit_log")
            if sqlite_table_exists(con, "audit_log")
            else 0,
            "cursor_updated_at": str(
                _sqlite_scalar(con, "SELECT COALESCE(MAX(updated_at), '') FROM ingestion_cursors")
                if sqlite_table_exists(con, "ingestion_cursors")
                else ""
            ),
            "counts": {
                table: int(_sqlite_scalar(con, f"SELECT COUNT(*) FROM {table}"))
                for table in (
                    "customer_identities",
                    "identity_links",
                    "customer_opportunities",
                    "timeline_events",
                    "timeline_conflicts",
                    "bot_context_chunks",
                )
                if sqlite_table_exists(con, table)
            },
        }
        physical = {
            "page_count": int(con.execute("PRAGMA page_count").fetchone()[0]),
            "freelist_count": int(con.execute("PRAGMA freelist_count").fetchone()[0]),
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
        for suffix in ("-wal",):
            sidecar = Path(f"{path}{suffix}")
            sidecar_stat = sidecar.stat() if sidecar.exists() else None
            physical[f"{suffix[1:]}_size_bytes"] = int(sidecar_stat.st_size) if sidecar_stat else 0
            physical[f"{suffix[1:]}_mtime_ns"] = int(sidecar_stat.st_mtime_ns) if sidecar_stat else 0
    return {**logical, **physical, "identity_digest": stable_digest(logical)}


def _sqlite_scalar(con: sqlite3.Connection, sql: str) -> Any:
    row = con.execute(sql).fetchone()
    return row[0] if row is not None else None


def file_sha256(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_worktree_provenance(root: Path) -> Mapping[str, Any]:
    git_env = dict(os.environ)
    for key in (
        "GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR",
        "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    ):
        git_env.pop(key, None)
    try:
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            check=True,
            capture_output=True,
            timeout=10,
            env=git_env,
        ).stdout
        diff = subprocess.run(
            ["git", "-C", str(root), "diff", "--binary", "HEAD"],
            check=True,
            capture_output=True,
            timeout=30,
            env=git_env,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {"dirty": None, "tracked_diff_sha256": None}
    return {
        "dirty": bool(status.strip()),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def assert_readonly_wappi_client(client: WappiHistoryClient) -> None:
    transport = getattr(client, "transport", None)
    if not isinstance(transport, DefaultDenyTransport):
        raise RuntimeError("Wappi history import requires WappiPhase1Client with DefaultDenyTransport.")
    if transport.policy.amo_read_hosts or transport.policy.ai_office_hosts:
        raise RuntimeError("Wappi history import requires a Wappi-only read-only transport policy.")


def profiles_from_phase1_config(config: AmoWappiPhase1Config) -> tuple[WappiProfileSpec, ...]:
    profiles: list[WappiProfileSpec] = []
    for profile_id, metadata in sorted(config.profile_metadata.items()):
        if not isinstance(metadata, Mapping):
            metadata = {"brand": config.brand_for_profile(profile_id)}
        profiles.append(
            WappiProfileSpec(
                profile_id=profile_id,
                brand=config.brand_for_profile(profile_id),
                channel=require_text(metadata.get("channel"), "channel"),
                label=str(metadata.get("label") or ""),
            )
        )
    if not profiles:
        raise AmoWappiConfigError("Wappi phase1 config has no profiles.")
    return tuple(profiles)


def load_wappi_pairs(
    pairs_file: Optional[Path],
    auto_pairs_file: Optional[Path],
) -> dict[DraftLoopKey, DraftLoopPair]:
    pairs: dict[DraftLoopKey, DraftLoopPair] = {}
    for path, source in ((pairs_file, "manual"), (auto_pairs_file, "auto")):
        if path is None:
            continue
        expanded = path.expanduser()
        if expanded.exists():
            pairs.update(load_pairs_file(expanded, default_source=source))
    return pairs


def load_existing_unmatched_wappi_records(
    db_path: Path,
    *,
    tenant_id: str,
    chat_resolutions: Mapping[tuple[str, str, str], WappiChatResolution],
    exclude_source_ids: Sequence[str] = (),
) -> tuple[TimelineSourceRecord, ...]:
    if not db_path.exists() or not chat_resolutions:
        return ()
    tenant = normalize_key(tenant_id, "tenant_id")
    excluded = set(exclude_source_ids)
    result: list[TimelineSourceRecord] = []
    with open_readonly_sqlite(db_path) as con:
        if not sqlite_table_exists(con, "timeline_events"):
            return ()
        rows = con.execute(
            """
            SELECT event.source_system, event.source_id, event.source_ref, event.record_json,
                   event.customer_id,
                   COALESCE(json_extract(event.record_json, '$.metadata.identity_authority'), '') AS identity_authority,
                   COALESCE(json_extract(customer.record_json, '$.metadata.provisional_wappi_family'), 0) AS provisional
            FROM timeline_events AS event
            LEFT JOIN customer_identities AS customer
              ON customer.tenant_id = event.tenant_id AND customer.customer_id = event.customer_id
            WHERE event.tenant_id = ?
              AND event.source_system IN ('wappi_telegram', 'wappi_max')
              AND event.superseded_by IS NULL
            ORDER BY event.source_system, event.source_id
            """,
            (tenant,),
        )
        for row in rows:
            source_id = str(row["source_id"] or "").strip()
            if not source_id or source_id in excluded:
                continue
            event = json.loads(str(row["record_json"] or "{}"))
            metadata = event.get("metadata") if isinstance(event.get("metadata"), Mapping) else {}
            profile_id = str(metadata.get("profile_id") or "").strip()
            chat_id = str(metadata.get("chat_id") or "").strip()
            resolution = chat_resolutions.get((str(row["source_system"]), profile_id, chat_id))
            if resolution is None or not resolution.resolved:
                continue
            existing_customer = str(row["customer_id"] or "").strip()
            existing_authority = str(row["identity_authority"] or "").strip()
            if (
                existing_customer
                and existing_customer != resolution.customer_id
                and existing_authority in WAPPI_EXACT_AMO_AUTHORITIES
            ):
                continue
            exact_override = (
                resolution.resolution_source in WAPPI_EXACT_AMO_AUTHORITIES
                and existing_authority not in WAPPI_EXACT_AMO_AUTHORITIES
            )
            if existing_customer and not bool(row["provisional"]) and not exact_override:
                continue
            message = (
                event.get("record", {}).get("message", {})
                if isinstance(event.get("record"), Mapping)
                else {}
            )
            if not isinstance(message, Mapping):
                message = {}
            message_id = str(metadata.get("message_id") or message.get("message_id") or "").strip()
            text = str(message.get("text") or event.get("text_preview") or event.get("summary") or "").strip()
            if not message_id or not text:
                continue
            event_at = parse_source_datetime(
                event.get("event_at"),
                datetime(1970, 1, 1, tzinfo=timezone.utc),
            )
            channel = str(message.get("channel") or "").strip().casefold()
            if channel not in SOURCE_SYSTEM_BY_CHANNEL:
                channel = "telegram" if str(row["source_system"]) == "wappi_telegram" else "max"
            payload = {
                "source_system": str(row["source_system"]),
                "source_ref": str(row["source_ref"] or source_id),
                "channel": channel,
                "brand": str(message.get("brand") or metadata.get("brand") or "unknown"),
                "profile_id": profile_id,
                "chat_id": chat_id,
                "message_id": message_id,
                "message_sha256": stable_digest(
                    {
                        "source_system": str(row["source_system"]),
                        "profile_id": profile_id,
                        "chat_id": chat_id,
                        "message_id": message_id,
                        "text": text,
                    }
                ),
                "timeline_source_id": source_id,
                "event_at": event_at.isoformat(),
                "event_time_status": "source_valid",
                "timestamp": event_at.timestamp(),
                "from_me": str(event.get("direction") or "") == TimelineDirection.OUTBOUND.value,
                "direction": str(event.get("direction") or ""),
                "message_type": str(message.get("message_type") or "text"),
                "text": text,
                "contact_name": str(event.get("actor_name") or ""),
                "from_where": "",
                "allowed_for_bot": False,
                "resolution_status": "resolved",
                "resolution_reason": resolution.reason,
                "resolved_customer_id": resolution.customer_id,
                "resolved_opportunity_id": resolution.opportunity_id,
                "lead_id": resolution.lead_id,
                "lead_ids": tuple(resolution.lead_ids),
                "contact_id": resolution.contact_id,
                "pair_source": resolution.pair_source,
                "identity_authority": resolution.resolution_source,
                "match_key": resolution.match_key,
                "candidate_customer_ids": tuple(resolution.candidate_customer_ids),
                "resolution_evidence": dict(resolution.evidence),
                "brand_context_authorized": resolution.evidence.get("brand_context_authorized"),
                "local_unmatched_relink": True,
            }
            result.append(
                TimelineSourceRecord(
                    source_system=str(row["source_system"]),
                    source_ref=str(payload["source_ref"]),
                    payload=payload,
                    observed_at=event_at,
                )
            )
    return tuple(result)


def load_existing_wappi_source_ids(
    db_path: Path,
    *,
    tenant_id: str,
    source_systems: set[str],
    source_ids: Sequence[str],
) -> set[str]:
    if not source_ids or not db_path.exists():
        return set()
    tenant = normalize_key(tenant_id, "tenant_id")
    found: set[str] = set()
    with open_readonly_sqlite(db_path) as con:
        if not sqlite_table_exists(con, "timeline_events"):
            return set()
        ids = tuple(dict.fromkeys(item for item in source_ids if item))
        for source_system in sorted(source_systems):
            for chunk in chunks(ids, 800):
                placeholders = ",".join("?" for _ in chunk)
                found.update(
                    str(row["source_id"])
                    for row in con.execute(
                        f"""
                        SELECT source_id
                        FROM timeline_events
                        WHERE tenant_id = ?
                          AND source_system = ?
                          AND source_id IN ({placeholders})
                        """,
                        (tenant, source_system, *chunk),
                    )
                )
    return found


def load_existing_wappi_event_customers(
    db_path: Path,
    *,
    tenant_id: str,
    source_systems: set[str],
    source_ids: Sequence[str],
) -> dict[tuple[str, str], tuple[str, str]]:
    if not source_ids or not db_path.exists():
        return {}
    tenant = normalize_key(tenant_id, "tenant_id")
    found: dict[tuple[str, str], tuple[str, str]] = {}
    with open_readonly_sqlite(db_path) as con:
        if not sqlite_table_exists(con, "timeline_events"):
            return {}
        ids = tuple(dict.fromkeys(item for item in source_ids if item))
        for source_system in sorted(source_systems):
            for chunk in chunks(ids, 800):
                placeholders = ",".join("?" for _ in chunk)
                for row in con.execute(
                    f"""
                    SELECT source_system, source_id, customer_id,
                           COALESCE(json_extract(record_json, '$.metadata.identity_authority'), '') AS identity_authority
                    FROM timeline_events
                    WHERE tenant_id = ?
                      AND source_system = ?
                      AND source_id IN ({placeholders})
                      AND superseded_by IS NULL
                    """,
                    (tenant, source_system, *chunk),
                ):
                    source_id = str(row["source_id"] or "").strip()
                    customer_id = str(row["customer_id"] or "").strip()
                    if source_id and customer_id:
                        found[(str(row["source_system"]), source_id)] = (
                            customer_id,
                            str(row["identity_authority"] or ""),
                        )
    return found


def load_provisional_customer_ids(db_path: Path, *, tenant_id: str) -> set[str]:
    if not db_path.exists():
        return set()
    tenant = normalize_key(tenant_id, "tenant_id")
    with open_readonly_sqlite(db_path) as con:
        if not sqlite_table_exists(con, "customer_identities"):
            return set()
        return {
            str(row["customer_id"])
            for row in con.execute(
                """
                SELECT customer_id
                FROM customer_identities
                WHERE tenant_id = ?
                  AND json_extract(record_json, '$.metadata.provisional_wappi_family') = 1
                """,
                (tenant,),
            )
        }


def remove_orphaned_provisional_customers(
    db_path: Path,
    *,
    tenant_id: str,
    customer_ids: Sequence[str],
) -> Mapping[str, int]:
    """Remove only provisional shells after their events and links moved to an exact family."""
    candidates = tuple(sorted({str(item) for item in customer_ids if str(item)}))
    if not candidates:
        return {"candidates": 0, "removed": 0, "retained_with_references": 0}
    tenant = normalize_key(tenant_id, "tenant_id")
    removed = 0
    retained = 0
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        con.execute("BEGIN IMMEDIATE")
        tables_with_customer_id: list[str] = []
        for table_row in con.execute("PRAGMA table_list"):
            table_name = str(table_row[1])
            if table_name.startswith("sqlite_") or table_name in {
                "customer_identities",
                "customer_id_mappings",
            }:
                continue
            quoted = table_name.replace('"', '""')
            if any(str(column[1]) == "customer_id" for column in con.execute(f'PRAGMA table_info("{quoted}")')):
                tables_with_customer_id.append(table_name)
        for customer_id in candidates:
            row = con.execute(
                """
                SELECT record_json
                FROM customer_identities
                WHERE tenant_id = ? AND customer_id = ?
                """,
                (tenant, customer_id),
            ).fetchone()
            if row is None:
                continue
            payload = json.loads(str(row["record_json"] or "{}"))
            if not bool((payload.get("metadata") or {}).get("provisional_wappi_family")):
                retained += 1
                continue
            has_reference = False
            for table_name in tables_with_customer_id:
                quoted = table_name.replace('"', '""')
                if con.execute(
                    f'SELECT 1 FROM "{quoted}" WHERE customer_id = ? LIMIT 1',
                    (customer_id,),
                ).fetchone():
                    has_reference = True
                    break
            if has_reference:
                retained += 1
                continue
            con.execute(
                "DELETE FROM customer_identities WHERE tenant_id = ? AND customer_id = ?",
                (tenant, customer_id),
            )
            removed += 1
        con.commit()
    return {
        "candidates": len(candidates),
        "removed": removed,
        "retained_with_references": retained,
    }


def close_resolved_wappi_pending_conflicts(
    db_path: Path,
    *,
    tenant_id: str,
    records: Sequence[TimelineSourceRecord],
) -> dict[str, int]:
    resolved_source_ids = {
        (
            str(record.payload.get("source_system") or ""),
            str(record.payload.get("profile_id") or ""),
            str(record.payload.get("chat_id") or ""),
            str(record.payload.get("message_id") or ""),
        )
        for record in records
        if str(record.payload.get("resolved_customer_id") or "").strip()
        and str(record.payload.get("identity_authority") or "") != "wappi_provisional"
    }
    resolved_source_ids.discard(("", "", "", ""))
    if not resolved_source_ids or not db_path.exists():
        return {"resolved_pending_conflicts_closed": 0}

    now = datetime.now(timezone.utc).isoformat()
    closed = 0
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")
        rows = con.execute(
            """
            SELECT conflict_id, record_json
            FROM timeline_conflicts
            WHERE tenant_id = ?
              AND conflict_type = 'pending_attribution'
              AND status = 'open'
            """,
            (tenant_id,),
        ).fetchall()
        for row in rows:
            payload = json.loads(str(row["record_json"] or "{}"))
            metadata = dict(payload.get("metadata") or {})
            key = tuple(
                str(metadata.get(name) or "")
                for name in ("source_system", "profile_id", "chat_id", "message_id")
            )
            if key not in resolved_source_ids:
                continue
            metadata["superseded_by"] = "resolved_wappi_timeline_event"
            metadata["resolved_by"] = "wappi_history_auto_resolver"
            payload["metadata"] = metadata
            payload["status"] = "resolved"
            payload["resolved_at"] = now
            safe_payload = scrub_timeline_persisted_json(payload)
            con.execute(
                """
                UPDATE timeline_conflicts
                SET status = 'resolved',
                    resolved_at = ?,
                    record_json = ?,
                    record_hash = ?
                WHERE conflict_id = ?
                """,
                (
                    now,
                    json.dumps(safe_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                    stable_digest(safe_payload),
                    row["conflict_id"],
                ),
            )
            closed += 1
        con.commit()
    return {"resolved_pending_conflicts_closed": closed}


def replace_wappi_record_resolution(
    record: TimelineSourceRecord,
    *,
    reason: str,
    status: str = "pending_attribution",
) -> TimelineSourceRecord:
    payload = dict(record.payload)
    payload.update(
        {
            "resolution_status": status,
            "resolution_reason": reason,
            "resolved_customer_id": None,
            "resolved_opportunity_id": None,
            "preserve_existing_event": True,
            "identity_authority": str(payload.get("identity_authority") or "wappi_relink_guard"),
        }
    )
    return TimelineSourceRecord(
        source_system=record.source_system,
        source_ref=record.source_ref,
        payload=payload,
        source_path=record.source_path,
        observed_at=record.observed_at,
    )


def lookup_amo_link_customers(
    con: sqlite3.Connection,
    *,
    tenant_id: str,
    link_type: str,
    link_value: str,
) -> set[str]:
    normalized_value = str(link_value or "").strip()
    if not normalized_value:
        return set()
    return {
        str(row["customer_id"])
        for row in con.execute(
            """
            SELECT customer_id FROM identity_links
            WHERE tenant_id = ?
              AND link_type = ?
              AND link_value = ?
              AND match_class = 'strong_unique'
            """,
            (tenant_id, normalize_key(link_type, "link_type"), normalized_value),
        )
        if row["customer_id"]
    }


def lookup_amo_opportunity_customers(
    con: sqlite3.Connection,
    *,
    tenant_id: str,
    lead_id: str,
) -> tuple[set[str], str]:
    normalized_lead = str(lead_id or "").strip()
    if not normalized_lead:
        return set(), ""
    customer_ids: set[str] = set()
    opportunity_id = ""
    for row in con.execute(
        """
        SELECT customer_id, opportunity_id FROM customer_opportunities
        WHERE tenant_id = ?
          AND source_system = 'amocrm_snapshot'
          AND opportunity_type = 'amo_deal'
          AND source_id = ?
        """,
        (tenant_id, normalized_lead),
    ):
        if row["customer_id"]:
            customer_ids.add(str(row["customer_id"]))
            opportunity_id = str(row["opportunity_id"] or opportunity_id)
    return customer_ids, opportunity_id


def pending_wappi_attribution_conflict(
    tenant_id: str,
    payload: Mapping[str, Any],
    source_ref: str,
    *,
    message_id: str,
    resolution_status: str,
) -> Mapping[str, Any]:
    return {
        "tenant_id": tenant_id,
        "conflict_type": "pending_attribution",
        "entity_refs": (
            source_ref,
            f"wappi_chat:{payload.get('profile_id')}:{payload.get('chat_id')}",
            f"wappi_message:{message_id}",
        ),
        "severity": "low",
        "status": "open",
        "summary": "Wappi message has no authoritative chat-to-customer attribution.",
        "metadata": {
            "source_system": payload.get("source_system"),
            "brand": payload.get("brand"),
            "profile_id": payload.get("profile_id"),
            "chat_id": payload.get("chat_id"),
            "message_id": message_id,
            "message_sha256": payload.get("message_sha256"),
            "resolution_status": resolution_status,
            "resolution_reason": payload.get("resolution_reason"),
            "lead_id": payload.get("lead_id"),
            "contact_id": payload.get("contact_id"),
            "identity_authority": payload.get("identity_authority") or "draft_loop_pair_required",
            "match_key": payload.get("match_key"),
        },
    }


def group_records_by_source_system(records: Sequence[TimelineSourceRecord]) -> dict[str, tuple[TimelineSourceRecord, ...]]:
    grouped: dict[str, list[TimelineSourceRecord]] = {}
    for record in records:
        grouped.setdefault(record.source_system, []).append(record)
    return {key: tuple(value) for key, value in grouped.items()}


def extract_wappi_items(payload: Mapping[str, Any], *keys: str) -> tuple[Mapping[str, Any], ...]:
    candidates: Any = payload
    if isinstance(candidates, Mapping):
        for key in keys:
            value = candidates.get(key)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                return tuple(dict(item) for item in value if isinstance(item, Mapping))
        embedded = candidates.get("data")
        if isinstance(embedded, Mapping):
            return extract_wappi_items(embedded, *keys)
    if isinstance(candidates, Sequence) and not isinstance(candidates, (str, bytes, bytearray)):
        return tuple(dict(item) for item in candidates if isinstance(item, Mapping))
    return ()


def extract_chat_id(dialog: Mapping[str, Any]) -> str:
    return str(dialog.get("id") or dialog.get("chat_id") or dialog.get("chatId") or dialog.get("jid") or "").strip()


def wappi_source_id(profile: WappiProfileSpec, message: WappiHistoryMessage) -> str:
    message_id = str(message.message_id or "").strip()
    if not message_id:
        message_id = stable_digest({"profile_id": profile.profile_id, "chat_id": message.chat_id, "text": message.text})[:16]
    return f"{profile.profile_id}:{message.chat_id}:{message_id}"


def anonymized_examples(records: Sequence[TimelineSourceRecord], *, limit: int = 5) -> list[Mapping[str, Any]]:
    examples: list[Mapping[str, Any]] = []
    for record in records[: max(0, limit)]:
        payload = record.payload
        text = str(payload.get("text") or "")
        examples.append(
            {
                "source_system": record.source_system,
                "brand": payload.get("brand"),
                "direction": payload.get("direction"),
                "resolution_status": payload.get("resolution_status"),
                "resolution_reason": payload.get("resolution_reason"),
                "identity_authority": payload.get("identity_authority"),
                "match_key": payload.get("match_key"),
                "chat_key_hash": stable_digest({"profile_id": payload.get("profile_id"), "chat_id": payload.get("chat_id")})[:12],
                "chat_id_kind": "numeric" if str(payload.get("chat_id") or "").isdigit() else "non_numeric",
                "text_preview_masked": mask_text(text),
                "source_ref_masked": mask_ref(record.source_ref),
            }
        )
    return examples


def mask_text(text: str, *, limit: int = 90) -> str:
    compact = " ".join(str(text or "").split())
    del limit
    return f"[текст скрыт; символов={len(compact)}; есть_цифры={any(char.isdigit() for char in compact)}]"


def mask_ref(value: Any) -> str:
    return f"sha256:{stable_digest(str(value or ''))[:16]}"


def sanitize_wappi_import_error(error: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "error_type": str(error.get("error_type") or "WappiImportError"),
        "error_code": "wappi_record_rejected",
        "source_ref_sha256": stable_digest(str(error.get("source_ref") or "")),
        "message_sha256": stable_digest(str(error.get("message") or "")),
        "redacted": True,
    }


def sanitize_wappi_import_report(report: Mapping[str, Any]) -> Mapping[str, Any]:
    safe = dict(report)
    safe["source_ref"] = "wappi_history_import"
    safe["errors"] = [sanitize_wappi_import_error(item) for item in report.get("errors") or ()]
    inventory = tuple(report.get("source_inventory") or ())
    safe["source_inventory"] = []
    safe["source_inventory_count"] = len(inventory)
    return safe


def safe_wappi_exception(exc: BaseException) -> Mapping[str, Any]:
    return {
        "error_type": type(exc).__name__,
        "error_code": "wappi_history_import_failed",
        "message_sha256": stable_digest(str(exc)),
        "redacted": True,
    }


def normalize_wappi_channel(value: Any) -> str:
    channel = str(value or "").strip().casefold()
    if channel not in SOURCE_SYSTEM_BY_CHANNEL:
        raise ValueError(f"unsupported Wappi channel: {value!r}")
    return channel


def normalize_brand(value: Any) -> str:
    brand = str(value or "").strip().casefold()
    if brand not in {"foton", "unpk"}:
        raise ValueError(f"unsupported Wappi brand: {value!r}")
    return brand


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value in (None, 0):
        return False
    return str(value).strip().casefold() in {"1", "true", "yes", "on", "y", "да", "allowed"}


def sleep_if_needed(seconds: float) -> None:
    if seconds > 0 and os.getenv("PYTEST_CURRENT_TEST") is None:
        time.sleep(seconds)


def open_readonly_sqlite(db_path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(customer_timeline_readonly_uri(db_path), uri=True, timeout=15)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA query_only = ON")
    return con


def sqlite_table_exists(con: sqlite3.Connection, table_name: str) -> bool:
    row = con.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table_name,)).fetchone()
    return row is not None


def chunks(values: Sequence[str], size: int) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(values[idx : idx + size]) for idx in range(0, len(values), size))


def write_json_report(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o600)
