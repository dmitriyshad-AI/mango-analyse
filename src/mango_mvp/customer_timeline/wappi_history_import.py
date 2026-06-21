from __future__ import annotations

import json
import os
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
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
)
from mango_mvp.customer_timeline.store import (
    CustomerTimelineSQLiteStore,
    guard_customer_timeline_sqlite_path,
)
from mango_mvp.integrations.amo_wappi_phase1 import (
    AMO_WAPPI_ENV_FILE,
    DEFAULT_AMO_WAPPI_CONFIG_PATH,
    AmoWappiConfigError,
    AmoWappiPhase1Config,
    WappiClientConfig,
    WappiPhase1Client,
    _json_http_request,
    load_env_file,
)
from mango_mvp.integrations.amo_wappi_transport import DefaultDenyTransport, SafeTransportPolicy
from mango_mvp.integrations.draft_loop import (
    DraftLoopKey,
    DraftLoopPair,
    WappiHistoryMessage,
    load_pairs_file,
    wappi_message_from_raw,
)


WAPPI_HISTORY_IMPORT_SCHEMA_VERSION = "wappi_history_timeline_import_v1"
SOURCE_SYSTEM_BY_CHANNEL = {"telegram": "wappi_telegram", "max": "wappi_max"}
EVENT_TYPE_BY_CHANNEL = {
    "telegram": TimelineEventType.TELEGRAM_MESSAGE,
    "max": TimelineEventType.MAX_MESSAGE,
}


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
    apply: bool = False
    actor: str = "wappi_history_timeline_import"
    idempotency_key: Optional[str] = None
    out_path: Optional[Path] = None
    limits: WappiFetchLimits = field(default_factory=WappiFetchLimits)

    def __post_init__(self) -> None:
        root = Path(self.allowed_root).expanduser().resolve(strict=False)
        timeline_db = guard_customer_timeline_output_path(guard_customer_timeline_sqlite_path(self.timeline_db), root)
        out_path = guard_customer_timeline_output_path(self.out_path, root) if self.out_path else None
        object.__setattr__(self, "allowed_root", root)
        object.__setattr__(self, "timeline_db", timeline_db)
        object.__setattr__(self, "env_file", Path(self.env_file).expanduser())
        object.__setattr__(self, "phase1_config", Path(self.phase1_config).expanduser())
        object.__setattr__(self, "pairs_file", Path(self.pairs_file).expanduser() if self.pairs_file else None)
        object.__setattr__(self, "auto_pairs_file", Path(self.auto_pairs_file).expanduser() if self.auto_pairs_file else None)
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
    pending_attribution: int = 0
    skipped_empty: int = 0
    skipped_bad_message: int = 0
    duplicate_source_ids: int = 0
    requests: int = 0
    request_limit_hit: bool = False
    message_limit_hit: bool = False
    chat_limit_hit: bool = False
    resolution_status_counts: Counter[str] = field(default_factory=Counter)

    def to_json_dict(self) -> Mapping[str, Any]:
        return {
            "chats_seen": self.chats_seen,
            "chats_loaded": self.chats_loaded,
            "messages_seen": self.messages_seen,
            "records_built": self.records_built,
            "linked_by_pair": self.linked_by_pair,
            "pending_attribution": self.pending_attribution,
            "skipped_empty": self.skipped_empty,
            "skipped_bad_message": self.skipped_bad_message,
            "duplicate_source_ids": self.duplicate_source_ids,
            "requests": self.requests,
            "request_limit_hit": self.request_limit_hit,
            "message_limit_hit": self.message_limit_hit,
            "chat_limit_hit": self.chat_limit_hit,
            "resolution_status_counts": dict(self.resolution_status_counts),
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
        resolution_status = str(payload.get("resolution_status") or "pending_attribution")
        resolved_customer_id = optional_text(payload.get("resolved_customer_id"))
        if not resolved_customer_id:
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
            match_status=IdentityMatchClass.MANUAL,
            confidence=0.9,
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
                    }
                )
            },
            metadata={
                "source_system": self.source_system,
                "brand": brand,
                "profile_id": payload.get("profile_id"),
                "chat_id": chat_id,
                "message_id": message_id,
                "identity_authority": "draft_loop_pair",
                "allowed_for_bot_reason": "wappi_history_manager_only",
            },
            created_at=event_at,
        )
        link_value = f"wappi_{channel}:{payload.get('profile_id')}:{chat_id}"
        link = IdentityLink(
            tenant_id=self.tenant_id,
            customer_id=resolved_customer_id,
            link_type="channel_session_id",
            link_value=link_value,
            source_system=self.source_system,
            source_ref=f"{self.source_system}:chat:{payload.get('profile_id')}:{chat_id}",
            match_class=IdentityMatchClass.MANUAL,
            confidence=0.9,
            evidence={"identity_authority": "draft_loop_pair", "lead_id": str(payload.get("lead_id") or "")},
            first_seen_at=event_at,
            last_seen_at=event_at,
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
                    metadata={"brand": brand, "channel": f"wappi_{channel}", "allowed_for_bot_reason": "wappi_history_manager_only"},
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
) -> Mapping[str, Any]:
    phase1 = AmoWappiPhase1Config.from_file(config.phase1_config)
    profiles = profiles_from_phase1_config(phase1)
    if client is None:
        client = build_readonly_wappi_client(config.env_file)
    assert_readonly_wappi_client(client)
    pairs = load_wappi_pairs(config.pairs_file, config.auto_pairs_file)
    resolver = WappiPairCustomerResolver.from_store(config.timeline_db, tenant_id=config.tenant_id, pairs=pairs)
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
    duplicate_count = 0
    for record in records:
        source_id = str(record.payload.get("timeline_source_id") or "")
        if source_id in existing_source_ids:
            duplicate_count += 1
            profile_id = str(record.payload.get("profile_id") or "")
            if profile_id in fetch_stats_by_profile:
                fetch_stats_by_profile[profile_id].duplicate_source_ids += 1

    import_reports: dict[str, Mapping[str, Any]] = {}
    write_status_counts: Counter[str] = Counter()
    normalized_counts: Counter[str] = Counter()
    errors: list[Mapping[str, Any]] = []
    store_summary_before: Optional[Mapping[str, Any]] = None
    store_summary_after: Optional[Mapping[str, Any]] = None
    grouped = group_records_by_source_system(records)
    if config.apply:
        store = CustomerTimelineSQLiteStore(config.timeline_db, allowed_root=config.allowed_root)
        try:
            store_summary_before = store.summary()
            for source_system, group in grouped.items():
                report = TimelineImportService(store).import_records(
                    group,
                    normalizer=WappiHistoryTimelineNormalizer(tenant_id=config.tenant_id, source_system=source_system),
                    tenant_id=config.tenant_id,
                    source_ref=f"wappi_history:{source_system}",
                    idempotency_key=config.idempotency_key or stable_digest([record.to_json_dict() for record in group]),
                    dry_run=False,
                    actor=config.actor,
                )
                import_reports[source_system] = report.to_json_dict()
                write_status_counts.update(report.write_status_counts)
                normalized_counts.update({key: int(value) for key, value in report.normalized_counts.items()})
                errors.extend(item.to_json_dict() for item in report.errors)
            store_summary_after = store.summary()
        finally:
            store.close()
    else:
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
            import_reports[source_system] = preview.to_json_dict()
            normalized_counts.update({key: int(value) for key, value in preview.normalized_counts.items()})
            errors.extend(item.to_json_dict() for item in preview.errors)

    safety = {
        **timeline_import_cli_safety_contract(write_product_timeline_db=config.apply),
        "read_local_files_only": False,
        "network_calls": True,
        "wappi_transport": "DefaultDenyTransport",
        "wappi_read_only_methods": ["GET"],
        "wappi_mark_all": False,
        "send_messenger": False,
        "write_crm": False,
        "write_tallanto": False,
        "blocked_live_actions": blocked_live_actions(),
    }
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
    validation_ok = not errors and safety_ok(safety)
    return {
        "schema_version": WAPPI_HISTORY_IMPORT_SCHEMA_VERSION,
        "mode": "apply" if config.apply else "dry_run_preview",
        "dry_run": not config.apply,
        "validation_ok": validation_ok,
        "summary": {
            "tenant_id": config.tenant_id,
            "profiles": len(profiles),
            "records_built": len(records),
            "linked_by_pair": sum(stats.linked_by_pair for stats in fetch_stats_by_profile.values()),
            "pending_attribution": sum(stats.pending_attribution for stats in fetch_stats_by_profile.values()),
            "requests": sum(stats.requests for stats in fetch_stats_by_profile.values()),
            "write_applied": config.apply,
            "writes_applied": sum(write_status_counts.values()) if config.apply else 0,
            "duplicate_source_ids_before_import": duplicate_count,
            "transport": "DefaultDenyTransport",
            "send_messenger": False,
        },
        "profiles": profile_reports,
        "records": {"by_source_system": {key: len(value) for key, value in grouped.items()}},
        "normalization": {"counts": dict(normalized_counts)},
        "writes": {
            "target": {"db_path": str(config.timeline_db), "allowed_root": str(config.allowed_root)},
            "applied": config.apply,
            "status_counts": dict(write_status_counts),
        },
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
    def __init__(self, resolutions: Mapping[DraftLoopKey, WappiChatResolution]) -> None:
        self._resolutions = dict(resolutions)

    @classmethod
    def from_store(
        cls,
        db_path: Path,
        *,
        tenant_id: str,
        pairs: Mapping[DraftLoopKey, DraftLoopPair],
    ) -> "WappiPairCustomerResolver":
        if not db_path.exists():
            return cls({})
        tenant = normalize_key(tenant_id, "tenant_id")
        resolutions: dict[DraftLoopKey, WappiChatResolution] = {}
        with open_readonly_sqlite(db_path) as con:
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
                    )
                else:
                    resolutions[key] = WappiChatResolution(
                        status="pending_attribution",
                        lead_id=str(pair.lead_id),
                        contact_id=str(pair.contact_id or ""),
                        expected_brand=pair.expected_brand,
                        reason="pair_has_no_customer_in_timeline",
                        pair_source=pair.source,
                    )
        return cls(resolutions)

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
            )
        return resolution


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
        offset = 0
        chats_loaded = 0
        profile_messages = 0
        while (
            chats_loaded < limits.chat_limit_per_profile
            and total_requests < limits.request_limit_total
            and profile_messages < per_profile_message_limit
            and total_messages < limits.message_limit_total
        ):
            page_limit = min(limits.page_size, limits.chat_limit_per_profile - chats_loaded)
            if page_limit <= 0:
                break
            payload = client.list_chats(
                channel=profile.channel,
                profile_id=profile.profile_id,
                limit=page_limit,
                offset=offset,
                order="desc",
                show_all=limits.show_all_chats,
            )
            total_requests += 1
            stats.requests += 1
            sleep_if_needed(limits.sleep_seconds)
            dialogs = extract_wappi_items(payload, "dialogs", "chats", "items", "data")
            if not dialogs:
                break
            stats.chats_seen += len(dialogs)
            for dialog in dialogs:
                if (
                    chats_loaded >= limits.chat_limit_per_profile
                    or total_messages >= limits.message_limit_total
                    or profile_messages >= per_profile_message_limit
                ):
                    break
                chat_id = extract_chat_id(dialog)
                if not chat_id:
                    continue
                chats_loaded += 1
                stats.chats_loaded += 1
                messages = fetch_chat_messages(client, profile=profile, chat_id=chat_id, limits=limits, request_counter=stats)
                total_requests += int(getattr(fetch_chat_messages, "last_request_count", 0))
                for message in messages:
                    if total_messages >= limits.message_limit_total or profile_messages >= per_profile_message_limit:
                        stats.message_limit_hit = True
                        break
                    stats.messages_seen += 1
                    if not message.text.strip():
                        stats.skipped_empty += 1
                        continue
                    resolution = resolver.resolve(profile=profile, chat_id=message.chat_id)
                    source_id = wappi_source_id(profile, message)
                    if source_id in seen_source_ids:
                        stats.duplicate_source_ids += 1
                        continue
                    seen_source_ids.add(source_id)
                    record = wappi_message_to_record(profile=profile, message=message, resolution=resolution)
                    records.append(record)
                    total_messages += 1
                    profile_messages += 1
                    stats.records_built += 1
                    stats.resolution_status_counts[resolution.reason or resolution.status] += 1
                    if resolution.resolved:
                        stats.linked_by_pair += 1
                    else:
                        stats.pending_attribution += 1
                if total_messages >= limits.message_limit_total or profile_messages >= per_profile_message_limit:
                    stats.message_limit_hit = True
                    break
            if len(dialogs) < page_limit:
                break
            offset += page_limit
        if chats_loaded >= limits.chat_limit_per_profile:
            stats.chat_limit_hit = True
        if total_requests >= limits.request_limit_total:
            stats.request_limit_hit = True
            break
        if total_messages >= limits.message_limit_total:
            break
    return tuple(records), stats_by_profile


def fetch_chat_messages(
    client: WappiHistoryClient,
    *,
    profile: WappiProfileSpec,
    chat_id: str,
    limits: WappiFetchLimits,
    request_counter: WappiFetchStats,
) -> tuple[WappiHistoryMessage, ...]:
    messages: list[WappiHistoryMessage] = []
    offset = 0
    request_count = 0
    while len(messages) < limits.messages_per_chat:
        page_limit = min(limits.page_size, limits.messages_per_chat - len(messages))
        if page_limit <= 0:
            break
        payload = client.get_chat_messages(
            channel=profile.channel,
            profile_id=profile.profile_id,
            chat_id=chat_id,
            limit=page_limit,
            offset=offset,
            order="desc",
            mark_all=False,
        )
        request_count += 1
        request_counter.requests += 1
        sleep_if_needed(limits.sleep_seconds)
        raw_messages = extract_wappi_items(payload, "messages", "items", "data")
        if not raw_messages:
            break
        for raw in raw_messages:
            item = wappi_message_from_raw(profile.profile_id, {**dict(raw), "chat_id": chat_id})
            if item is None:
                request_counter.skipped_bad_message += 1
                continue
            messages.append(item)
        if len(raw_messages) < page_limit:
            break
        offset += page_limit
    setattr(fetch_chat_messages, "last_request_count", request_count)
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
    event_at = datetime.fromtimestamp(message.timestamp, tz=timezone.utc) if message.timestamp > 0 else datetime.now(timezone.utc)
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
    }
    return TimelineSourceRecord(
        source_system=source_system,
        source_ref=str(payload["source_ref"]),
        payload=payload,
        observed_at=event_at,
    )


def build_readonly_wappi_client(env_file: Path = AMO_WAPPI_ENV_FILE) -> WappiPhase1Client:
    load_env_file(env_file)
    config = WappiClientConfig.from_env()
    return WappiPhase1Client(config, transport=build_wappi_readonly_transport(config))


def build_wappi_readonly_transport(config: WappiClientConfig) -> DefaultDenyTransport:
    wappi_host = url_parse.urlparse(config.base_url).netloc.casefold()
    inner_transport = _json_http_request
    return DefaultDenyTransport(
        inner_transport,
        policy=SafeTransportPolicy(
            wappi_hosts=frozenset(host for host in (wappi_host,) if host),
            amo_read_hosts=frozenset(),
            ai_office_hosts=frozenset(),
        ),
    )


def assert_readonly_wappi_client(client: WappiHistoryClient) -> None:
    if not isinstance(getattr(client, "transport", None), DefaultDenyTransport):
        raise RuntimeError("Wappi history import requires WappiPhase1Client with DefaultDenyTransport.")


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
            "identity_authority": "draft_loop_pair_required",
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
    text = str(value or "")
    if len(text) <= 12:
        return "***"
    return f"{text[:8]}...{text[-4:]}"


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
    uri = f"file:{url_parse.quote(str(db_path.resolve(strict=False)), safe='/:')}?mode=ro&immutable=1"
    con = sqlite3.connect(uri, uri=True, timeout=15)
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
