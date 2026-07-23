from __future__ import annotations

import json
import hashlib
import math
import os
import re
import sqlite3
import subprocess
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, Sequence
from urllib import parse as url_parse

from mango_mvp.customer_timeline.contracts import (
    BotContextChunk,
    IdentityLink,
    IdentityMatchClass,
    TimelineDirection,
    TimelineEvent,
    TimelineEventType,
    TimelineParticipant,
)
from mango_mvp.customer_timeline.ids import normalize_key, optional_text, require_text, stable_digest
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
}


class WappiPhysicalRequestBudgetExceeded(RuntimeError):
    pass


@dataclass
class WappiPhysicalRequestBudget:
    limit: int
    used: int = 0

    def take(self) -> None:
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

    def __post_init__(self) -> None:
        object.__setattr__(self, "chat_limit_per_profile", max(0, int(self.chat_limit_per_profile)))
        object.__setattr__(self, "messages_per_chat", max(0, int(self.messages_per_chat)))
        object.__setattr__(self, "message_limit_total", max(0, int(self.message_limit_total)))
        object.__setattr__(self, "request_limit_total", max(1, int(self.request_limit_total)))
        object.__setattr__(self, "page_size", max(1, min(int(self.page_size), 100)))
        object.__setattr__(self, "sleep_seconds", max(0.0, float(self.sleep_seconds)))


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
    actor: str = "wappi_history_timeline_import"
    idempotency_key: Optional[str] = None
    out_path: Optional[Path] = None
    limits: WappiFetchLimits = field(default_factory=WappiFetchLimits)

    def __post_init__(self) -> None:
        root = Path(self.allowed_root).expanduser().resolve(strict=False)
        timeline_db = guard_customer_timeline_output_path(guard_customer_timeline_sqlite_path(self.timeline_db), root)
        if self.apply and is_customer_timeline_prod_path(timeline_db):
            raise ValueError("Wappi history apply must not target a production Customer Timeline")
        out_path = guard_customer_timeline_output_path(self.out_path, root) if self.out_path else None
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


@dataclass(frozen=True)
class WappiChatResolution:
    status: str
    customer_id: Optional[str] = None
    opportunity_id: Optional[str] = None
    lead_id: str = ""
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
    resolution_status_counts: Counter[str] = field(default_factory=Counter)
    coverage_counts: Counter[str] = field(default_factory=Counter)
    amo_auto_status_counts: Counter[str] = field(default_factory=Counter)
    amo_auto_calls: int = 0

    def to_json_dict(self) -> Mapping[str, Any]:
        return {
            "chats_seen": self.chats_seen,
            "chats_loaded": self.chats_loaded,
            "messages_seen": self.messages_seen,
            "records_built": self.records_built,
            "linked_by_pair": self.linked_by_pair,
            "linked_by_timeline": self.linked_by_timeline,
            "linked_by_amo_auto": self.linked_by_amo_auto,
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
            "resolution_status_counts": dict(self.resolution_status_counts),
            "coverage_counts": dict(self.coverage_counts),
            "amo_auto_status_counts": dict(self.amo_auto_status_counts),
            "amo_auto_calls": self.amo_auto_calls,
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
            confidence=0.9 if resolved_customer_id else 0.0,
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
                "contact_id": str(payload.get("contact_id") or ""),
                "match_key": str(payload.get("match_key") or ""),
                "allowed_for_bot_reason": "wappi_history_manager_only",
                "allowed_for_bot": False,
                "requires_manager_review": True,
                "pending_attribution": not bool(resolved_customer_id),
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
        link = IdentityLink(
            tenant_id=self.tenant_id,
            customer_id=resolved_customer_id,
            link_type="channel_session_id",
            link_value=link_value,
            source_system=self.source_system,
            source_ref=f"{self.source_system}:chat:{payload.get('profile_id')}:{chat_id}",
            match_class=resolved_match_class,
            confidence=0.9,
            evidence={
                "identity_authority": identity_authority,
                "lead_id": str(payload.get("lead_id") or ""),
                "contact_id": str(payload.get("contact_id") or ""),
                "match_key": str(payload.get("match_key") or ""),
                "brand_context_authorized": payload.get("brand_context_authorized"),
                "resolution_evidence": scrub_timeline_persisted_json(resolution_evidence),
            },
            first_seen_at=event_at if event_time_status == "source_valid" else None,
            last_seen_at=event_at if event_time_status == "source_valid" else None,
        )
        chunks: tuple[BotContextChunk, ...] = ()
        if text:
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
            identity_links=(link,),
            events=(event,),
            bot_context_chunks=chunks,
        )


def run_wappi_history_import(
    config: WappiHistoryImportConfig,
    *,
    client: WappiHistoryClient | None = None,
    amo_auto_resolver: AmoAutoResolver | None = None,
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
    phase1 = AmoWappiPhase1Config.from_file(config.phase1_config)
    profiles = profiles_from_phase1_config(phase1)
    if client is None:
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
        shared_phone_stoplist=local_phone_stoplist,
        shared_phone_stoplist_error=local_phone_stoplist_error,
    )
    records, fetch_stats_by_profile = fetch_wappi_history_records(
        client=client,
        profiles=profiles,
        resolver=resolver,
        limits=config.limits,
        tenant_id=config.tenant_id,
    )
    existing_source_ids = load_existing_wappi_source_ids(
        config.timeline_db,
        tenant_id=config.tenant_id,
        source_systems=set(SOURCE_SYSTEM_BY_CHANNEL.values()),
        source_ids=[str(record.payload.get("timeline_source_id") or "") for record in records],
    )
    existing_event_customers = load_existing_wappi_event_customers(
        config.timeline_db,
        tenant_id=config.tenant_id,
        source_systems=set(SOURCE_SYSTEM_BY_CHANNEL.values()),
        source_ids=[str(record.payload.get("timeline_source_id") or "") for record in records],
    )
    duplicate_count = 0
    blocked_customer_relink_conflicts = 0
    guarded_records: list[TimelineSourceRecord] = []
    for record in records:
        source_id = str(record.payload.get("timeline_source_id") or "")
        if source_id in existing_source_ids:
            duplicate_count += 1
            profile_id = str(record.payload.get("profile_id") or "")
            if profile_id in fetch_stats_by_profile:
                fetch_stats_by_profile[profile_id].duplicate_source_ids += 1
        existing_customer = existing_event_customers.get((record.source_system, source_id))
        proposed_customer = str(record.payload.get("resolved_customer_id") or "").strip()
        if existing_customer and proposed_customer != existing_customer:
            blocked_customer_relink_conflicts += 1
            existing_reason = str(record.payload.get("resolution_reason") or "")
            guarded_records.append(
                replace_wappi_record_resolution(
                    record,
                    reason=(
                        existing_reason
                        if existing_reason == "existing_wappi_chat_customer_conflict"
                        else "existing_wappi_source_customer_conflict"
                    ),
                    status="pending_attribution",
                )
            )
            profile_id = str(record.payload.get("profile_id") or "")
            if profile_id in fetch_stats_by_profile and proposed_customer:
                fetch_stats_by_profile[profile_id].pending_attribution += 1
                if fetch_stats_by_profile[profile_id].linked_by_amo_auto > 0 and record.payload.get("identity_authority") == "amo_auto_resolver":
                    fetch_stats_by_profile[profile_id].linked_by_amo_auto -= 1
                elif fetch_stats_by_profile[profile_id].linked_by_timeline > 0 and record.payload.get("identity_authority") == "timeline_identity":
                    fetch_stats_by_profile[profile_id].linked_by_timeline -= 1
                elif fetch_stats_by_profile[profile_id].linked_by_pair > 0:
                    fetch_stats_by_profile[profile_id].linked_by_pair -= 1
                fetch_stats_by_profile[profile_id].resolution_status_counts["existing_wappi_source_customer_conflict"] += 1
            continue
        guarded_records.append(record)
    records = tuple(guarded_records)
    profile_reports = {
        profile.profile_id: {
            "profile_id": profile.profile_id,
            "brand": profile.brand,
            "channel": profile.channel,
            "source_system": profile.source_system,
            **fetch_stats_by_profile.get(profile.profile_id, WappiFetchStats()).to_json_dict(),
        }
        for profile in profiles
    }
    limit_hits = [
        f"{profile_id}:{field}"
        for profile_id, report in sorted(profile_reports.items())
        for field in ("chat_limit_hit", "message_limit_hit", "request_limit_hit", "pagination_drift_detected")
        if report.get(field)
    ]
    empty_profiles = sorted(
        profile_id for profile_id, report in profile_reports.items() if int(report.get("records_built") or 0) == 0
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
        or db_identity_pre_apply.get("identity_digest") != db_identity_start.get("identity_digest")
    ):
        limit_hits.append("provenance_drift")

    import_reports: dict[str, Mapping[str, Any]] = {}
    write_status_counts: Counter[str] = Counter()
    normalized_counts: Counter[str] = Counter()
    errors: list[Mapping[str, Any]] = []
    stale_conflict_cleanup: dict[str, int] = {}
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

    apply_effective = config.apply and not errors and not limit_hits
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
        finally:
            store.close()
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
    safety = {
        **timeline_import_cli_safety_contract(write_product_timeline_db=apply_effective),
        "read_local_files_only": False,
        "network_calls": True,
        "wappi_transport": "DefaultDenyTransport",
        "wappi_read_only_methods": ["GET"],
        "wappi_mark_all": False,
        "amo_auto_resolver_enabled": amo_auto_resolver is not None,
        "amo_transport": "AmoMcpClient" if amo_auto_resolver is not None else "disabled",
        "amo_read_only_methods": ["GET"] if amo_auto_resolver is not None else [],
        "send_messenger": False,
        "write_crm": False,
        "write_tallanto": False,
        "blocked_live_actions": blocked_live_actions(),
    }
    validation_ok = not errors and not limit_hits and safety_ok(safety)
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
                "request_limit_total": config.limits.request_limit_total,
                "page_size": config.limits.page_size,
                "sleep_seconds": config.limits.sleep_seconds,
                "show_all_chats": config.limits.show_all_chats,
                "require_nonempty_profiles": config.require_nonempty_profiles,
            },
        },
        "mode": "apply" if apply_effective else ("apply_blocked" if config.apply else "dry_run_preview"),
        "dry_run": not apply_effective,
        "validation_ok": validation_ok,
        "limit_hits": limit_hits,
        "summary": {
            "tenant_id": config.tenant_id,
            "profiles": len(profiles),
            "records_built": len(records),
            "linked_by_pair": sum(stats.linked_by_pair for stats in fetch_stats_by_profile.values()),
            "linked_by_timeline": sum(stats.linked_by_timeline for stats in fetch_stats_by_profile.values()),
            "linked_by_amo_auto": sum(stats.linked_by_amo_auto for stats in fetch_stats_by_profile.values()),
            "pending_attribution": sum(stats.pending_attribution for stats in fetch_stats_by_profile.values()),
            "requests": sum(stats.requests for stats in fetch_stats_by_profile.values()),
            "physical_requests": readonly_wappi_physical_request_count(client),
            "amo_auto_enabled": amo_auto_resolver is not None,
            "amo_auto_calls": sum(stats.amo_auto_calls for stats in fetch_stats_by_profile.values()),
            "write_applied": apply_effective,
            "writes_applied": sum(write_status_counts.values()) if apply_effective else 0,
            "duplicate_source_ids_before_import": duplicate_count,
            "blocked_customer_relink_conflicts": blocked_customer_relink_conflicts,
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
            "empty_profiles": empty_profiles,
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
        local_identity_customers: Mapping[tuple[str, str], Sequence[str]] | None = None,
        ambiguous_identity_values: Sequence[tuple[str, str]] = (),
        customer_brands: Mapping[str, str] | None = None,
        supported_customer_ids: Sequence[str] = (),
        chat_customer_ids: Mapping[tuple[str, str, str], Sequence[str]] | None = None,
        shared_phone_stoplist: Sequence[str] = (),
        shared_phone_stoplist_error: str = "",
    ) -> None:
        self._resolutions = dict(resolutions)
        self._db_path = Path(db_path)
        self._tenant_id = normalize_key(tenant_id, "tenant_id")
        self._amo_auto_resolver = amo_auto_resolver
        self._local_identity_customers = {
            key: tuple(sorted(set(values))) for key, values in (local_identity_customers or {}).items()
        }
        self._ambiguous_identity_values = frozenset(ambiguous_identity_values)
        self._customer_brands = dict(customer_brands or {})
        self._supported_customer_ids = frozenset(supported_customer_ids)
        self._chat_customer_ids = {
            key: tuple(sorted(set(values))) for key, values in (chat_customer_ids or {}).items()
        }
        self._shared_phone_stoplist = frozenset(shared_phone_stoplist)
        self._shared_phone_stoplist_error = str(shared_phone_stoplist_error or "")

    @property
    def amo_auto_calls(self) -> int:
        return int(getattr(self._amo_auto_resolver, "calls", 0)) if self._amo_auto_resolver is not None else 0

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
                shared_phone_stoplist=shared_phone_stoplist,
                shared_phone_stoplist_error=shared_phone_stoplist_error,
            )
        resolutions: dict[DraftLoopKey, WappiChatResolution] = {}
        local_identity_customers: dict[tuple[str, str], set[str]] = {}
        ambiguous_identity_values: set[tuple[str, str]] = set()
        customer_brands: dict[str, str] = {}
        supported_customer_ids: set[str] = set()
        chat_customer_ids: dict[tuple[str, str, str], set[str]] = {}
        with open_readonly_sqlite(db_path) as con:
            family_customer_ids = {
                str(row["customer_id"])
                for row in con.execute(
                    """
                    SELECT customer_id
                    FROM identity_links
                    WHERE tenant_id = ? AND link_type = 'tallanto_student_id'
                    GROUP BY customer_id
                    HAVING COUNT(DISTINCT link_value) > 1
                    """,
                    (tenant,),
                )
            }
            safe_customer_ids = {
                str(row["customer_id"])
                for row in con.execute(
                    """
                    SELECT customer_id
                    FROM customer_identities
                    WHERE tenant_id = ? AND identity_status IN ('strong', 'partial')
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
                    'phone', 'mango_client_phone', 'whatsapp_phone'
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
                or (key[0] == "phone" and bool(identity_owners.get(key, set()) & family_customer_ids))
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
                    or (
                    link_type == "phone" and customer_id in family_customer_ids
                    )
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
                    chat_customer_ids.setdefault((parts[0], parts[1], parts[2]), set()).add(str(row["customer_id"]))
            if sqlite_table_exists(con, "timeline_events"):
                for row in con.execute(
                    """
                    SELECT source_system,
                           json_extract(record_json, '$.metadata.profile_id') AS profile_id,
                           json_extract(record_json, '$.metadata.chat_id') AS chat_id,
                           customer_id
                    FROM timeline_events
                    WHERE tenant_id = ?
                      AND source_system IN ('wappi_telegram', 'wappi_max')
                      AND customer_id IS NOT NULL
                    """,
                    (tenant,),
                ):
                    if row["profile_id"] and row["chat_id"]:
                        chat_customer_ids.setdefault(
                            (str(row["source_system"]), str(row["profile_id"]), str(row["chat_id"])),
                            set(),
                        ).add(str(row["customer_id"]))
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
            local_identity_customers=local_identity_customers,
            ambiguous_identity_values=tuple(ambiguous_identity_values),
            customer_brands=customer_brands,
            supported_customer_ids=tuple(supported_customer_ids),
            chat_customer_ids=chat_customer_ids,
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
        pair_resolution = self.resolve(profile=profile, chat_id=chat_id)
        if pair_resolution.reason != "draft_loop_pair_missing":
            return self._guard_chat_customer(
                profile,
                chat_id,
                self._guard_pair_context(profile, chat_id, dialog, pair_resolution),
            )
        timeline_resolution = self._resolve_with_timeline_identity(
            profile=profile,
            chat_id=chat_id,
            dialog=dialog,
        )
        if timeline_resolution is not None:
            return self._guard_chat_customer(profile, chat_id, timeline_resolution)
        if self._amo_auto_resolver is None:
            return pair_resolution
        return self._guard_chat_customer(
            profile,
            chat_id,
            self._resolve_with_amo_auto(profile=profile, chat_id=chat_id, dialog=dialog, messages=messages),
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
        if not messages:
            return WappiChatResolution(
                status="pending_attribution",
                expected_brand=profile.brand,
                reason="chat_has_no_importable_messages",
                resolution_source="amo_auto_resolver",
            )
        if not chat_id:
            return WappiChatResolution(status="pending_attribution", expected_brand=profile.brand, reason="chat_id_missing", resolution_source="amo_auto_resolver")
        key = DraftLoopKey(profile.profile_id, chat_id)
        draft_profile = DraftLoopProfile(profile_id=profile.profile_id, brand=profile.brand, channel=profile.channel)
        auto_result = self._amo_auto_resolver(
            key=key,
            profile=draft_profile,
            dialog=dialog,
            messages=messages,
            message=messages[-1],
        )
        status = str(auto_result.get("status") or "").strip()
        reason = str(auto_result.get("reason") or status or "amo_auto_unresolved").strip()
        lead_id = str(auto_result.get("lead_id") or "").strip()
        contact_id = str(auto_result.get("contact_id") or "").strip()
        match_key = str(auto_result.get("match_key") or "").strip()
        if status != "matched":
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
        with open_readonly_sqlite(self._db_path) as con:
            lead_ids = lookup_amo_link_customers(
                con,
                tenant_id=self._tenant_id,
                link_type="amo_lead_id",
                link_value=lead_id,
            )
            contact_ids = lookup_amo_link_customers(
                con,
                tenant_id=self._tenant_id,
                link_type="amo_contact_id",
                link_value=contact_id,
            )
            opportunity_ids, opportunity_id = lookup_amo_opportunity_customers(
                con,
                tenant_id=self._tenant_id,
                lead_id=lead_id,
            )
        candidate_sets = [items for items in (lead_ids, contact_ids, opportunity_ids) if items]
        candidate_union = set().union(*candidate_sets) if candidate_sets else set()
        lead_snapshot = auto_result.get("lead_snapshot") if isinstance(auto_result.get("lead_snapshot"), Mapping) else {}
        organization_values = lead_snapshot.get("organization_values") or ()
        evidence = {
            "exact_match_kind": match_key,
            "single_active_lead": True,
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
            return WappiChatResolution(
                status="resolved",
                customer_id=next(iter(candidate_union)),
                opportunity_id=opportunity_id or None,
                lead_id=lead_id,
                contact_id=contact_id,
                expected_brand=profile.brand,
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
) -> tuple[tuple[TimelineSourceRecord, ...], dict[str, WappiFetchStats]]:
    del tenant_id
    records: list[TimelineSourceRecord] = []
    stats_by_profile: dict[str, WappiFetchStats] = {profile.profile_id: WappiFetchStats() for profile in profiles}
    seen_source_ids: set[str] = set()
    total_messages = 0
    total_requests = 0
    per_profile_message_limit = max(1, limits.message_limit_total // max(1, len(profiles))) if limits.message_limit_total else 0
    for profile in profiles:
        stats = stats_by_profile[profile.profile_id]
        profile_amo_calls_start = resolver.amo_auto_calls
        offset = 0
        profile_messages = 0
        chat_ids_seen: set[str] = set()
        dialogs_snapshot: list[Mapping[str, Any]] = []
        chat_page_specs: list[tuple[int, int]] = []
        while (
            len(dialogs_snapshot) < limits.chat_limit_per_profile
            and total_requests < limits.request_limit_total
        ):
            page_limit = min(limits.page_size, limits.chat_limit_per_profile - len(dialogs_snapshot))
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
                break
            total_requests += 1
            stats.requests += 1
            sleep_if_needed(limits.sleep_seconds)
            dialogs = extract_wappi_items(payload, "dialogs", "chats", "items", "data")
            chat_page_specs.append((offset, page_limit))
            if not dialogs:
                break
            stats.chats_seen += len(dialogs)
            for dialog in dialogs:
                if len(dialogs_snapshot) >= limits.chat_limit_per_profile:
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
            if len(dialogs) < page_limit:
                break
            offset += page_limit
        verification_chat_ids: set[str] = set()
        for verification_offset, verification_limit in chat_page_specs:
            if total_requests >= limits.request_limit_total:
                stats.request_limit_hit = True
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
        if len(dialogs_snapshot) >= limits.chat_limit_per_profile:
            stats.chat_limit_hit = True
        if total_requests >= limits.request_limit_total:
            stats.request_limit_hit = True
            break
        if total_messages >= limits.message_limit_total:
            break
        for dialog in dialogs_snapshot:
            if total_messages >= limits.message_limit_total or profile_messages >= per_profile_message_limit:
                stats.message_limit_hit = True
                break
            chat_id = extract_chat_id(dialog)
            resolver.record_coverage(profile=profile, dialog=dialog, stats=stats)
            messages = fetch_chat_messages(
                client,
                profile=profile,
                chat_id=chat_id,
                limits=limits,
                request_counter=stats,
                request_budget=max(0, limits.request_limit_total - total_requests),
            )
            total_requests += int(getattr(fetch_chat_messages, "last_request_count", 0))
            if bool(getattr(fetch_chat_messages, "last_request_limit_hit", False)):
                stats.request_limit_hit = True
                break
            if bool(getattr(fetch_chat_messages, "last_limit_hit", False)):
                stats.message_limit_hit = True
            if bool(getattr(fetch_chat_messages, "last_pagination_drift_detected", False)):
                stats.message_page_drift_detected = True
                stats.pagination_drift_detected = True
                break
            resolution = resolver.resolve_chat(profile=profile, dialog=dialog, messages=messages)
            stats.amo_auto_calls = resolver.amo_auto_calls - profile_amo_calls_start
            stats.amo_auto_status_counts[f"{resolution.resolution_source}:{resolution.reason or resolution.status}"] += 1
            for message in messages:
                if total_messages >= limits.message_limit_total or profile_messages >= per_profile_message_limit:
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
                    elif resolution.resolution_source == "timeline_identity":
                        stats.linked_by_timeline += 1
                    else:
                        stats.linked_by_pair += 1
                else:
                    stats.pending_attribution += 1
            if stats.request_limit_hit:
                break
    return tuple(records), stats_by_profile


def fetch_chat_messages(
    client: WappiHistoryClient,
    *,
    profile: WappiProfileSpec,
    chat_id: str,
    limits: WappiFetchLimits,
    request_counter: WappiFetchStats,
    request_budget: int,
) -> tuple[WappiHistoryMessage, ...]:
    messages: list[WappiHistoryMessage] = []
    offset = 0
    request_count = 0
    limit_hit = False
    request_limit_hit = request_budget <= 0
    pagination_drift_detected = False
    page_signatures: list[tuple[int, int, tuple[str, ...]]] = []
    while len(messages) < limits.messages_per_chat and request_count < request_budget:
        page_limit = min(limits.page_size, limits.messages_per_chat - len(messages))
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
        page_signatures.append(
            (
                offset,
                page_limit,
                tuple(str(item.get("id") or item.get("message_id") or "") for item in raw_messages),
            )
        )
        for raw in raw_messages:
            item = wappi_message_from_raw(profile.profile_id, {**dict(raw), "chat_id": chat_id})
            if item is None:
                request_counter.skipped_bad_message += 1
                continue
            messages.append(item)
        if len(raw_messages) < page_limit:
            break
        offset += page_limit
        if len(messages) >= limits.messages_per_chat:
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
    phones = {
        phone
        for phone in (
            normalize_phone(dialog.get("phone") or dialog.get("number") or ""),
            *(
                normalize_phone(item.get("phone") or item.get("number") or "")
                for item in (dialog.get("participants") or ())
                if isinstance(item, Mapping)
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
        client_participants = tuple(
            item
            for item in participants
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
    try:
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            check=True,
            capture_output=True,
            timeout=10,
        ).stdout
        diff = subprocess.run(
            ["git", "-C", str(root), "diff", "--binary", "HEAD"],
            check=True,
            capture_output=True,
            timeout=30,
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
) -> dict[tuple[str, str], str]:
    if not source_ids or not db_path.exists():
        return {}
    tenant = normalize_key(tenant_id, "tenant_id")
    found: dict[tuple[str, str], str] = {}
    with open_readonly_sqlite(db_path) as con:
        if not sqlite_table_exists(con, "timeline_events"):
            return {}
        ids = tuple(dict.fromkeys(item for item in source_ids if item))
        for source_system in sorted(source_systems):
            for chunk in chunks(ids, 800):
                placeholders = ",".join("?" for _ in chunk)
                for row in con.execute(
                    f"""
                    SELECT source_system, source_id, customer_id
                    FROM timeline_events
                    WHERE tenant_id = ?
                      AND source_system = ?
                      AND source_id IN ({placeholders})
                    """,
                    (tenant, source_system, *chunk),
                ):
                    source_id = str(row["source_id"] or "").strip()
                    customer_id = str(row["customer_id"] or "").strip()
                    if source_id and customer_id:
                        found[(str(row["source_system"]), source_id)] = customer_id
    return found


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
    }
    resolved_source_ids.discard(("", "", "", ""))
    if not resolved_source_ids or not db_path.exists():
        return {"resolved_pending_conflicts_closed": 0}

    now = datetime.now(timezone.utc).isoformat()
    closed = 0
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")
        for source_system, profile_id, chat_id, message_id in sorted(resolved_source_ids):
            rows = con.execute(
                """
                SELECT conflict_id, record_json
                FROM timeline_conflicts
                WHERE tenant_id = ?
                  AND conflict_type = 'pending_attribution'
                  AND status = 'open'
                  AND json_extract(record_json, '$.metadata.source_system') = ?
                  AND json_extract(record_json, '$.metadata.profile_id') = ?
                  AND json_extract(record_json, '$.metadata.chat_id') = ?
                  AND json_extract(record_json, '$.metadata.message_id') = ?
                """,
                (tenant_id, source_system, profile_id, chat_id, message_id),
            ).fetchall()
            for row in rows:
                payload = json.loads(str(row["record_json"] or "{}"))
                metadata = dict(payload.get("metadata") or {})
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
