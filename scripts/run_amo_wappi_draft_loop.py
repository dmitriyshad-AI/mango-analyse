#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib import parse as url_parse

from mango_mvp.channels.subscription_llm import (
    DIRECT_PATH_PILOT_CONFIG_ENV,
    DIRECT_PATH_PILOT_CONFIG_VERSION,
    DIRECT_PATH_REAL_MANAGER_GOLD_PACK_VERSION,
    SubscriptionLlmDraftProvider,
)
from mango_mvp.integrations.amo_wappi_phase1 import (
    AI_OFFICE_ENV_FILE,
    AMO_WAPPI_ENV_FILE,
    DEFAULT_AMO_WAPPI_CONFIG_PATH,
    AiOfficeAmoNoteClient,
    AiOfficeClientConfig,
    AmoWappiPhase1Config,
    WappiClientConfig,
    WappiPhase1Client,
    _json_http_request,
    load_env_file,
)
from mango_mvp.integrations.amo_wappi_transport import DefaultDenyTransport, SafeTransportPolicy
from mango_mvp.integrations.draft_loop import (
    DEFAULT_DRAFT_LOOP_DIR,
    DEFAULT_STOP_PATH,
    AmoWappiDraftLoop,
    DraftLoopConfig,
    DraftLoopKey,
    DraftLoopProfile,
    DraftLoopState,
    WappiHistoryMessage,
    build_draft_loop_config_fingerprint,
    load_pairs_file,
    load_profiles_file,
)
from mango_mvp.pilot_context_assembly import build_pilot_context_payload
from mango_mvp.customer_timeline.bot_safe_runtime_context import (
    BOT_SAFE_CRM_CONTEXT_DB_ENV,
    BOT_SAFE_CRM_CONTEXT_ENV,
    DEFAULT_BOT_SAFE_TENANT_ID,
    BotSafeLookup,
    bot_safe_crm_context_enabled,
    bot_safe_tenant_from_env,
    bot_safe_timeline_db_from_env,
    build_bot_safe_crm_context,
)
from mango_mvp.existing_clients.amo_step1_snapshot import AmoMcpClient, AmoMcpConfig, read_mcp_env
from mango_mvp.utils.phone import normalize_phone


DEFAULT_PROFILES_PATH = Path.home() / ".mango_secrets" / "amo_wappi_profiles.json"
DEFAULT_PAIRS_PATH = Path.home() / ".mango_secrets" / "draft_loop_pairs.json"
DEFAULT_AUTO_PAIRS_PATH = Path.home() / ".mango_secrets" / "draft_loop_auto_pairs.json"
DEFAULT_RETRO_DIR = Path.home() / ".mango_local" / "draft_loop_inventory"
DEFAULT_STOPLIST_PATH = Path.home() / ".mango_secrets" / "shared_phones_stoplist.json"
LEGACY_STOPLIST_PATH = Path.home() / ".mango_secrets" / "shared_phone_stoplist.json"
DEFAULT_SNAPSHOT = Path("product_data/knowledge_base/kb_release_20260612_v6_7_staging_r4_1/kb_release_v3_snapshot.json")
DEFAULT_CUSTOMER_TIMELINE_DB = Path(
    "product_data/customer_timeline/customer_timeline_prod_20260621/customer_timeline.sqlite"
)
DRAFT_LOOP_AUTO_RESOLVER_ENV = "DRAFT_LOOP_AUTO_RESOLVER"
CLOSED_STATUS_IDS = {"142", "143"}
ORG_BRAND_KEYWORDS = {
    "foton": ("фотон", "cdpo", "цдпо"),
    "unpk": ("унпк", "мфти", "mipt"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Poll Wappi, build bot drafts, and write AMO draft notes for explicit test pairs.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="Run exactly one polling cycle.")
    mode.add_argument("--loop", action="store_true", help="Run polling loop.")
    parser.add_argument("--interval-sec", type=int, default=45)
    parser.add_argument("--dry-run", action="store_true", default=True, help="Do not POST AMO notes. Default.")
    parser.add_argument("--live-write", action="store_true", help="Allow AMO note POST after all allowlist checks.")
    parser.add_argument("--env-file", type=Path, default=AMO_WAPPI_ENV_FILE)
    parser.add_argument("--ai-office-env-file", type=Path, default=AI_OFFICE_ENV_FILE)
    parser.add_argument("--profiles-file", type=Path, default=DEFAULT_PROFILES_PATH)
    parser.add_argument("--pairs-file", type=Path, default=DEFAULT_PAIRS_PATH)
    parser.add_argument("--auto-pairs-file", type=Path, default=DEFAULT_AUTO_PAIRS_PATH)
    parser.add_argument("--phase1-config", type=Path, default=DEFAULT_AMO_WAPPI_CONFIG_PATH)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument(
        "--customer-timeline-db",
        type=Path,
        default=None,
        help=f"Read-only customer timeline DB for {BOT_SAFE_CRM_CONTEXT_ENV}=1. Defaults to {BOT_SAFE_CRM_CONTEXT_DB_ENV} or product_data path.",
    )
    parser.add_argument("--customer-timeline-allowed-root", type=Path, default=None)
    parser.add_argument("--customer-timeline-tenant", default=bot_safe_tenant_from_env())
    parser.add_argument("--local-dir", type=Path, default=DEFAULT_DRAFT_LOOP_DIR)
    parser.add_argument("--stop-file", type=Path, default=DEFAULT_STOP_PATH)
    parser.add_argument("--chat-limit", type=int, default=50)
    parser.add_argument("--amo-mcp-env-file", type=Path, default=Path("~/.mango_secrets/foton_crm_readonly_mcp_connector.env").expanduser())
    parser.add_argument("--shared-phone-stoplist", type=Path, default=DEFAULT_STOPLIST_PATH)
    parser.add_argument("--retro-report", nargs="?", const="", default=None, help="Write an offline bot-vs-manager report outside the repo and exit.")
    parser.add_argument("--retro-lookback-hours", type=int, default=48)
    parser.add_argument("--retro-limit", type=int, default=30)
    parser.add_argument("--manager-outgoing-visible", choices=("unknown", "yes", "no"), default="unknown")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--reasoning", default="xhigh")
    parser.add_argument("--timeout-sec", type=int, default=240)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> DraftLoopConfig:
    profiles = load_profiles_file(args.profiles_file)
    pairs = load_pairs_file(args.pairs_file)
    allowed: set[str] = set()
    if args.phase1_config.expanduser().exists():
        phase1 = AmoWappiPhase1Config.from_file(args.phase1_config)
        allowed.update(phase1.allowed_test_lead_ids)
    allowed.update(str(pair.lead_id) for pair in pairs.values())
    auto_pairs_path = getattr(args, "auto_pairs_file", DEFAULT_AUTO_PAIRS_PATH).expanduser()
    if auto_pairs_path.exists():
        allowed.update(str(pair.lead_id) for pair in load_pairs_file(auto_pairs_path, default_source="auto").values())
    if not allowed:
        allowed.update(str(pair.lead_id) for pair in pairs.values())
    local_dir = args.local_dir.expanduser()
    visibility = None
    if args.manager_outgoing_visible == "yes":
        visibility = True
    elif args.manager_outgoing_visible == "no":
        visibility = False
    raw_chat_limit = int(getattr(args, "chat_limit", 50) or 0)
    chat_limit = 0 if raw_chat_limit <= 0 else max(1, min(raw_chat_limit, 100))
    return DraftLoopConfig(
        profiles=profiles,
        pairs=pairs,
        auto_pairs_path=auto_pairs_path,
        allowed_test_lead_ids=frozenset(allowed),
        state_path=local_dir / "state.json",
        journal_path=local_dir / "journal.jsonl",
        manager_edit_log_path=local_dir / "manager_edits.jsonl",
        heartbeat_path=local_dir / "heartbeat.json",
        stop_path=args.stop_file.expanduser(),
        manager_outgoing_visible=visibility,
        chat_limit=chat_limit,
        config_fingerprint=build_draft_loop_config_fingerprint(
            getattr(args, "snapshot", DEFAULT_SNAPSHOT),
            gold_pack_version=DIRECT_PATH_REAL_MANAGER_GOLD_PACK_VERSION,
        ),
    )


def build_context_builder(
    snapshot_path: Path,
    *,
    draft_config: DraftLoopConfig | None = None,
    customer_timeline_db: Path | None = None,
    customer_timeline_allowed_root: Path | None = None,
    customer_timeline_tenant: str = DEFAULT_BOT_SAFE_TENANT_ID,
):
    def _build(
        key: DraftLoopKey,
        history: list[str] | tuple[str, ...],
        client_message: str,
        brand: str,
        *,
        channel: str = "telegram",
        dialogue_memory: Mapping[str, Any] | None = None,
        current_message_id: str = "",
    ) -> Mapping[str, Any]:
        crm_context = _build_bot_safe_crm_context_for_draft(
            key=key,
            brand=brand,
            draft_config=draft_config,
            customer_timeline_db=customer_timeline_db,
            customer_timeline_allowed_root=customer_timeline_allowed_root,
            customer_timeline_tenant=customer_timeline_tenant,
        )
        return build_pilot_context_payload(
            current_text=client_message,
            snapshot_path=snapshot_path,
            active_brand=brand,
            recent_messages=tuple(history)[-10:],
            dialogue_memory=dialogue_memory or {},
            session_id=f"amo_draft_loop:{brand}:{key.profile_id}:{key.chat_id}",
            channel=f"wappi_{str(channel or 'telegram').strip().casefold()}",
            channel_thread_id=key.value,
            channel_user_id=key.chat_id,
            current_message_id=current_message_id,
            dialogue_contract_pipeline_enabled=True,
            sends_client_replies=False,
            debug_impersonation_enabled=False,
            crm_context=crm_context,
        )

    return _build


def _build_bot_safe_crm_context_for_draft(
    *,
    key: DraftLoopKey,
    brand: str,
    draft_config: DraftLoopConfig | None,
    customer_timeline_db: Path | None,
    customer_timeline_allowed_root: Path | None,
    customer_timeline_tenant: str,
) -> Mapping[str, Any]:
    if not bot_safe_crm_context_enabled():
        return {}
    pair = draft_config.pair_for(key) if draft_config is not None else None
    if pair is None:
        return {}
    db_path = customer_timeline_db or bot_safe_timeline_db_from_env() or DEFAULT_CUSTOMER_TIMELINE_DB
    allowed_root = customer_timeline_allowed_root or db_path.parent
    context = build_bot_safe_crm_context(
        timeline_db=db_path,
        allowed_root=allowed_root,
        active_brand=brand,
        lookup=BotSafeLookup(
            tenant_id=customer_timeline_tenant or DEFAULT_BOT_SAFE_TENANT_ID,
            amo_lead_id=pair.lead_id,
            amo_contact_id=pair.contact_id,
        ),
    )
    return context if context.get("found") else {}


def build_safe_transport(ai_office_config: AiOfficeClientConfig, wappi_config: WappiClientConfig) -> DefaultDenyTransport:
    ai_office_host = url_parse.urlparse(ai_office_config.base_url).netloc.casefold()
    wappi_host = url_parse.urlparse(wappi_config.base_url).netloc.casefold()
    return DefaultDenyTransport(
        _json_http_request,
        policy=SafeTransportPolicy(
            wappi_hosts=frozenset(host for host in (wappi_host,) if host),
            amo_read_hosts=frozenset(),
            ai_office_hosts=frozenset(host for host in (ai_office_host,) if host),
        ),
    )


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().casefold() in {"1", "true", "yes", "on"}


def _embedded_items(payload: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
    embedded = payload.get("_embedded")
    if isinstance(embedded, Mapping):
        raw = embedded.get(key)
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
            return [dict(item) for item in raw if isinstance(item, Mapping)]
    raw = payload.get(key)
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        return [dict(item) for item in raw if isinstance(item, Mapping)]
    return []


def _custom_field_values(entity: Mapping[str, Any], *needles: str) -> list[str]:
    wanted = tuple(str(item).casefold() for item in needles if str(item).strip())
    values: list[str] = []
    for field in entity.get("custom_fields_values") or ():
        if not isinstance(field, Mapping):
            continue
        name = str(field.get("field_name") or field.get("name") or "").casefold()
        code = str(field.get("field_code") or "").casefold()
        if wanted and not any(needle in name or needle in code for needle in wanted):
            continue
        for item in field.get("values") or ():
            raw = item.get("value") if isinstance(item, Mapping) else item
            if str(raw or "").strip():
                values.append(str(raw).strip())
    return values


def _contact_telegram_ids(contact: Mapping[str, Any]) -> set[str]:
    result: set[str] = set()
    for value in _custom_field_values(contact, "telegram", "телеграм"):
        cleaned = re.sub(r"\D+", "", value)
        if cleaned:
            result.add(cleaned)
    return result


def _contact_phones(contact: Mapping[str, Any]) -> set[str]:
    result: set[str] = set()
    for value in _custom_field_values(contact, "phone", "телефон", "tel"):
        phone = normalize_phone(value)
        if phone:
            result.add(phone)
    return result


def _lead_ids_from_contact(contact: Mapping[str, Any]) -> list[str]:
    ids: list[str] = []
    for item in _embedded_items(contact, "leads"):
        lead_id = str(item.get("id") or "").strip()
        if lead_id and lead_id not in ids:
            ids.append(lead_id)
    return ids


def _is_active_lead(lead: Mapping[str, Any]) -> bool:
    if bool(lead.get("is_deleted") or lead.get("deleted")):
        return False
    status_id = str(lead.get("status_id") or "").strip()
    closed_at = str(lead.get("closed_at") or "").strip()
    return status_id not in CLOSED_STATUS_IDS and not closed_at


def _lead_org_values(lead: Mapping[str, Any]) -> list[str]:
    return _custom_field_values(lead, "организация", "organization")


def _lead_org_brand(lead: Mapping[str, Any]) -> str:
    values = _lead_org_values(lead)
    text = " ".join(values).casefold()
    if not text:
        return ""
    for brand, markers in ORG_BRAND_KEYWORDS.items():
        if any(marker in text for marker in markers):
            return brand
    return ""


def _load_phone_stoplist(path: Path) -> tuple[set[str], str]:
    target = path.expanduser()
    if not target.exists() and target == DEFAULT_STOPLIST_PATH and LEGACY_STOPLIST_PATH.exists():
        target = LEGACY_STOPLIST_PATH
    if not target.exists():
        return set(), "shared_phone_stoplist_unavailable"
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return set(), "shared_phone_stoplist_invalid"
    raw: Any = payload.get("phones") if isinstance(payload, Mapping) else payload
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return set(), "shared_phone_stoplist_invalid"
    phones = {normalize_phone(item) for item in raw}
    phones.discard("")
    if not phones:
        return set(), "shared_phone_stoplist_unavailable"
    return phones, ""


def _max_dialog_phone(dialog: Mapping[str, Any]) -> tuple[str, str]:
    direct = normalize_phone(dialog.get("phone") or dialog.get("number") or "")
    if direct:
        return direct, "max_phone_field"
    participants = dialog.get("participants")
    phones: set[str] = set()
    if isinstance(participants, Sequence) and not isinstance(participants, (str, bytes, bytearray)):
        for item in participants:
            if not isinstance(item, Mapping):
                continue
            phone = normalize_phone(item.get("phone") or item.get("number") or "")
            if phone:
                phones.add(phone)
    if len(phones) == 1:
        return next(iter(phones)), "max_participant_phone"
    if len(phones) > 1:
        return "", "max_multi_phone"
    return "", "max_phone_missing"


@dataclass
class AmoAutoResolver:
    client: AmoMcpClient
    shared_phone_stoplist: set[str]
    stoplist_error: str = ""

    def __post_init__(self) -> None:
        self.shared_phone_stoplist = {phone for phone in (normalize_phone(item) for item in self.shared_phone_stoplist) if phone}

    def __call__(
        self,
        *,
        key: DraftLoopKey,
        profile: DraftLoopProfile,
        dialog: Mapping[str, Any],
        messages: Sequence[WappiHistoryMessage],
        message: WappiHistoryMessage,
    ) -> Mapping[str, Any]:
        del messages, message
        if profile.channel == "telegram":
            return self._resolve_telegram(key=key, profile=profile)
        if profile.channel == "max":
            return self._resolve_max(key=key, profile=profile, dialog=dialog)
        return {"status": "rejected", "reason": "unsupported_channel"}

    def _resolve_telegram(self, *, key: DraftLoopKey, profile: DraftLoopProfile) -> Mapping[str, Any]:
        if not key.chat_id.isdigit():
            return {"status": "rejected", "reason": "username_only", "channel": "telegram"}
        contacts = self._search_contacts_exact_telegram_id(key.chat_id)
        if len(contacts) != 1:
            return {"status": "rejected", "reason": "multi_contact" if contacts else "username_only", "channel": "telegram"}
        return self._resolve_contact(profile=profile, contact=contacts[0], match_key="Telegram ID", match_value=key.chat_id)

    def _resolve_max(self, *, key: DraftLoopKey, profile: DraftLoopProfile, dialog: Mapping[str, Any]) -> Mapping[str, Any]:
        phone, source = _max_dialog_phone(dialog)
        if not phone:
            return {"status": "rejected", "reason": source, "channel": "max"}
        if self.stoplist_error:
            return {"status": "rejected", "reason": self.stoplist_error, "channel": "max"}
        if phone in self.shared_phone_stoplist:
            return {"status": "rejected", "reason": "shared_phone", "channel": "max", "match_key": source}
        contacts = self._search_contacts_exact_phone(phone)
        if len(contacts) != 1:
            return {"status": "rejected", "reason": "multi_contact" if contacts else "no_contact", "channel": "max", "match_key": source}
        return self._resolve_contact(profile=profile, contact=contacts[0], match_key=source, match_value=phone)

    def _search_contacts_exact_telegram_id(self, telegram_id: str) -> list[Mapping[str, Any]]:
        payload = self.client.amo_api_get(path="contacts", params={"query": telegram_id, "with": "leads"}, limit=50)
        contacts = _embedded_items(payload, "contacts")
        return [contact for contact in contacts if telegram_id in _contact_telegram_ids(contact)]

    def _search_contacts_exact_phone(self, phone: str) -> list[Mapping[str, Any]]:
        payload = self.client.amo_api_get(path="contacts", params={"query": phone, "with": "leads"}, limit=50)
        contacts = _embedded_items(payload, "contacts")
        return [contact for contact in contacts if phone in _contact_phones(contact)]

    def _resolve_contact(
        self,
        *,
        profile: DraftLoopProfile,
        contact: Mapping[str, Any],
        match_key: str,
        match_value: str,
    ) -> Mapping[str, Any]:
        contact_id = str(contact.get("id") or "").strip()
        lead_ids = _lead_ids_from_contact(contact)
        if not lead_ids and contact_id:
            contact_payload = self.client.amo_api_get(path=f"contacts/{int(contact_id)}", params={"with": "leads"}, limit=1)
            lead_ids = _lead_ids_from_contact(contact_payload)
        leads: list[Mapping[str, Any]] = []
        deleted_seen = False
        for lead_id in lead_ids:
            lead = self.client.amo_api_get(path=f"leads/{int(lead_id)}", params={"with": "contacts"}, limit=1)
            if bool(lead.get("is_deleted") or lead.get("deleted")):
                deleted_seen = True
            leads.append(lead)
        active = [lead for lead in leads if _is_active_lead(lead)]
        if not active:
            reason = "deleted_lead" if deleted_seen else "closed_lead" if leads else "no_active_lead"
            return {"status": "rejected", "reason": reason, "contact_id": contact_id, "match_key": match_key}
        if len(active) != 1:
            return {"status": "rejected", "reason": "multi_active_lead", "contact_id": contact_id, "match_key": match_key}
        lead = active[0]
        org_brand = _lead_org_brand(lead)
        org_values = _lead_org_values(lead)
        if org_brand and org_brand != profile.brand:
            return {
                "status": "rejected",
                "reason": "brand_mismatch",
                "contact_id": contact_id,
                "lead_id": str(lead.get("id") or ""),
                "organization_brand": org_brand,
                "organization_values": org_values,
            }
        return {
            "status": "matched",
            "lead_id": str(lead.get("id") or ""),
            "contact_id": contact_id,
            "match_key": match_key,
            "match_value": match_value,
            "lead_snapshot": {
                "status_id": str(lead.get("status_id") or ""),
                "closed_at": str(lead.get("closed_at") or ""),
                "pipeline_id": str(lead.get("pipeline_id") or ""),
                "organization_brand": org_brand,
                "organization_values": org_values,
            },
        }


def build_auto_resolver(args: argparse.Namespace) -> AmoAutoResolver | None:
    if not _truthy(os.getenv(DRAFT_LOOP_AUTO_RESOLVER_ENV)):
        return None
    stoplist, stoplist_error = _load_phone_stoplist(args.shared_phone_stoplist)
    config = read_mcp_env(args.amo_mcp_env_file)
    if config.transport != "curl":
        config = AmoMcpConfig(
            connector_url=config.connector_url,
            bearer_token=config.bearer_token,
            timeout_seconds=config.timeout_seconds,
            max_retries=config.max_retries,
            user_agent="mango-draft-loop-auto-resolver/1.0",
            transport="curl",
        )
    return AmoAutoResolver(client=AmoMcpClient(config), shared_phone_stoplist=stoplist, stoplist_error=stoplist_error)


def build_runner(args: argparse.Namespace) -> AmoWappiDraftLoop:
    load_env_file(args.env_file)
    if args.ai_office_env_file.expanduser().exists():
        load_env_file(args.ai_office_env_file)
    os.environ.setdefault(DIRECT_PATH_PILOT_CONFIG_ENV, DIRECT_PATH_PILOT_CONFIG_VERSION)
    config = build_config(args)
    ai_office_config = AiOfficeClientConfig.from_env()
    wappi_config = WappiClientConfig.from_env()
    transport = build_safe_transport(ai_office_config, wappi_config)
    bot_provider = SubscriptionLlmDraftProvider(
        model=args.model,
        reasoning_effort=args.reasoning,
        timeout_sec=args.timeout_sec,
        cache_dir=None,
        codex_isolated=True,
    )
    return AmoWappiDraftLoop(
        config=config,
        wappi_client=WappiPhase1Client(wappi_config, transport=transport),
        amo_client=AiOfficeAmoNoteClient(ai_office_config, transport=transport),
        bot_provider=bot_provider,
        context_builder=build_context_builder(
            args.snapshot,
            draft_config=config,
            customer_timeline_db=args.customer_timeline_db,
            customer_timeline_allowed_root=args.customer_timeline_allowed_root,
            customer_timeline_tenant=args.customer_timeline_tenant,
        ),
        state=DraftLoopState(config.state_path),
        auto_resolver=build_auto_resolver(args),
    )


def main() -> int:
    args = parse_args()
    dry_run = not bool(args.live_write)
    runner = build_runner(args)
    if args.retro_report is not None:
        report = runner.build_retro_report(lookback_hours=args.retro_lookback_hours, limit=args.retro_limit)
        if args.retro_report:
            target = Path(args.retro_report).expanduser()
        else:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            target = DEFAULT_RETRO_DIR / f"retro_compare_{stamp}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"retro_report": str(target), **dict(report.get("summary") or {})}, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if not args.loop:
        summary = runner.run_once(dry_run=dry_run)
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    interval = max(5, int(args.interval_sec))
    while True:
        summary = runner.run_once(dry_run=dry_run)
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
        time.sleep(interval)


if __name__ == "__main__":
    raise SystemExit(main())
