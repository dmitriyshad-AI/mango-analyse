#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import asyncio
import html
import json
import os
import re
import signal
from contextlib import suppress
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib import parse as url_parse
from urllib import request as url_request
from urllib import error as url_error

from mango_mvp.channels.subscription_llm import (
    SAFE_FALLBACK_DRAFT_TEXT,
    SubscriptionDraftResult,
    SubscriptionLlmDraftProvider,
    strip_internal_service_markers,
)
from mango_mvp.channels.dialogue_memory import update_dialogue_memory_after_answer
from mango_mvp.channels.dialogue_contract_pipeline import DIALOGUE_CONTRACT_PIPELINE_ENV
from mango_mvp.channels.manager_handoff_summary import build_manager_handoff_summary
from mango_mvp.channels.new_lead_funnel import LeadFunnelState
from mango_mvp.channels.telegram_pilot_store import (
    PILOT_DRAFT_STATUS_MANAGER_ONLY,
    PILOT_DRAFT_STATUS_NEEDS_REVIEW,
    TelegramPilotSQLiteStore,
)
from mango_mvp.channels.telegram_pilot_p0_register import append_p0_register_record, build_p0_register_record
from mango_mvp.pilot_context_assembly import (
    attach_funnel_state_to_context as assemble_funnel_state_context,
    build_funnel_state as assemble_funnel_state,
    build_pilot_context_payload,
)
from mango_mvp.channels.night_funnel_shadow import (
    DEFAULT_CONTROL_PATH as NIGHT_FUNNEL_DEFAULT_CONTROL_PATH,
    DEFAULT_INBOUND_TEE_PATH as NIGHT_FUNNEL_DEFAULT_TEE_PATH,
    DEFAULT_LEAD_STORE_PATH as NIGHT_FUNNEL_DEFAULT_LEAD_STORE_PATH,
    DEFAULT_SHADOW_LOG_PATH as NIGHT_FUNNEL_DEFAULT_SHADOW_LOG_PATH,
    DEFAULT_STATUS_PATH as NIGHT_FUNNEL_DEFAULT_STATUS_PATH,
    DEFAULT_TEE_RETENTION_DAYS as NIGHT_FUNNEL_DEFAULT_TEE_RETENTION_DAYS,
    MANAGER_QUEUE,
    NightFunnelControl,
    append_inbound_tee_record,
    append_lead_card,
    append_shadow_log,
    assert_live_send_allowed,
    brand_from_channel,
    build_inbound_tee_record,
    build_lead_card,
    build_shadow_record,
    evaluate_night_gate,
    extract_utm,
    load_bot_control,
    write_bot_status,
)


DEFAULT_ENV_FILE = Path("/Users/dmitrijfabarisov/.codex/mango_telegram_pilot_bots.env")
DEFAULT_SNAPSHOT = Path("product_data/knowledge_base/kb_release_20260611_v6_7_staging_r4/kb_release_v3_snapshot.json")
DEFAULT_CRM_ENV_FILE = Path("stable_runtime/amocrm_runtime/.env.private")
DEFAULT_LOG_DIR = Path(".codex_local/telegram_pilot_bots/logs")
DEFAULT_RUNTIME_DIR = Path(".codex_local/telegram_pilot_bots/runtime")
DEFAULT_HEARTBEAT_PATH = DEFAULT_RUNTIME_DIR / "public_pilot_bots_heartbeat.json"
DEFAULT_CACHE_DIR = Path(".codex_local/telegram_pilot_bots/llm_cache")
DEFAULT_STORE_PATH = Path(".codex_local/telegram_pilot/telegram_pilot.sqlite")
DEFAULT_P0_REGISTER_PATH = Path(".codex_local/telegram_pilot/p0_incident_register.csv")
DEFAULT_DEBOUNCE_SECONDS = 7
MAX_RECENT_MESSAGES = 12
AUTONOMOUS_ROUTES = {"bot_answer_self", "bot_answer_self_for_pilot"}

FOTON_TOKEN_ENV = "MANGO_TELEGRAM_FOTON_BOT_TOKEN"
UNPK_TOKEN_ENV = "MANGO_TELEGRAM_UNPK_BOT_TOKEN"
SNAPSHOT_ENV = "MANGO_TELEGRAM_KB_SNAPSHOT"
DEBUG_CLIENTS_ENV = "MANGO_TELEGRAM_DEBUG_CLIENTS_JSON"
CRM_READ_MODE_ENV = "MANGO_TELEGRAM_CRM_READ_MODE"
CRM_ENV_FILE_ENV = "MANGO_TELEGRAM_CRM_ENV_FILE"
CRM_SERVER_URL_ENV = "MANGO_CRM_SERVER_URL"
CRM_SERVER_API_KEY_ENV = "MANGO_CRM_SERVER_API_KEY"
PILOT_STORE_PATH_ENV = "MANGO_TELEGRAM_PILOT_STORE_PATH"
PILOT_STORE_ENABLED_ENV = "MANGO_TELEGRAM_PILOT_STORE_ENABLED"
PILOT_P0_REGISTER_PATH_ENV = "MANGO_TELEGRAM_P0_REGISTER_PATH"
PILOT_AUTONOMY_ENABLED_ENV = "TELEGRAM_PILOT_AUTONOMY_ENABLED"
NIGHT_FUNNEL_SHADOW_ENABLED_ENV = "TELEGRAM_NIGHT_FUNNEL_SHADOW_ENABLED"
NIGHT_FUNNEL_SHADOW_ONLY_ENV = "TELEGRAM_NIGHT_FUNNEL_SHADOW_ONLY"
NIGHT_FUNNEL_CONTROL_PATH_ENV = "TELEGRAM_NIGHT_FUNNEL_CONTROL_PATH"
NIGHT_FUNNEL_STATUS_PATH_ENV = "TELEGRAM_NIGHT_FUNNEL_STATUS_PATH"
NIGHT_FUNNEL_LOG_PATH_ENV = "TELEGRAM_NIGHT_FUNNEL_SHADOW_LOG_PATH"
NIGHT_FUNNEL_LEAD_STORE_PATH_ENV = "TELEGRAM_NIGHT_FUNNEL_LEAD_STORE_PATH"
NIGHT_FUNNEL_LIVE_TOKEN_ENV = "TELEGRAM_NIGHT_FUNNEL_LIVE_TOKEN"
NIGHT_FUNNEL_EXPECTED_LIVE_TOKEN_ENV = "TELEGRAM_NIGHT_FUNNEL_EXPECTED_LIVE_TOKEN"
NIGHT_FUNNEL_TEE_ENABLED_ENV = "TELEGRAM_NIGHT_FUNNEL_TEE_ENABLED"
NIGHT_FUNNEL_TEE_PATH_ENV = "TELEGRAM_NIGHT_FUNNEL_TEE_PATH"
NIGHT_FUNNEL_TEE_SOURCE_ENV = "TELEGRAM_NIGHT_FUNNEL_TEE_SOURCE"
NIGHT_FUNNEL_TEE_RETENTION_DAYS_ENV = "TELEGRAM_NIGHT_FUNNEL_TEE_RETENTION_DAYS"
PUBLIC_BOT_HEARTBEAT_PATH_ENV = "MANGO_TELEGRAM_PUBLIC_BOT_HEARTBEAT_PATH"
PUBLIC_PARSE_MODE_HTML_ENV = "TELEGRAM_PUBLIC_PARSE_MODE_HTML"

DEBUG_PHONE_RE = re.compile(
    r"^\s*[\"'«»“”]*\s*представь\s*,?\s*что\s+я\s+пишу\s+с\s+номера\s+"
    r"(?P<phone>\+?\d[\d\s()\-]{7,})"
    r"\s*[\"'«»“”]*(?:[,:;.\-—]\s*(?P<rest>.*))?$",
    re.I,
)
PHONE_DIGIT_RE = re.compile(r"\D+")
PUBLIC_TELEGRAM_HIGHLIGHT_RE = re.compile(
    r"(?P<price>\b\d[\d\s]{1,12}\s*(?:₽|руб(?:\.|лей|ля|ль)?))"
    r"|(?P<date>\b\d{1,2}[./]\d{1,2}(?:[./]\d{2,4})?\b)"
    r"|(?P<time>\b\d{1,2}:\d{2}(?:\s*[–-]\s*\d{1,2}:\d{2})?\b)"
)


@dataclass(frozen=True)
class BrandBotConfig:
    brand: str
    token: str
    display_name: str
    snapshot_path: Path
    debounce_seconds: int = DEFAULT_DEBOUNCE_SECONDS
    log_dir: Path = DEFAULT_LOG_DIR
    heartbeat_path: Path = DEFAULT_HEARTBEAT_PATH
    cache_dir: Path = DEFAULT_CACHE_DIR
    model: str = "gpt-5.5"
    reasoning_effort: str = "xhigh"
    timeout_sec: int = 240
    allow_groups: bool = False
    crm_read_mode: str = "live"
    crm_env_file: Path = DEFAULT_CRM_ENV_FILE
    crm_server_url: str = ""
    crm_server_api_key: str = ""
    store_path: Path = DEFAULT_STORE_PATH
    store_enabled: bool = True
    p0_register_path: Path = DEFAULT_P0_REGISTER_PATH
    autonomy_enabled: bool = True
    dialogue_contract_pipeline_enabled: bool = True
    night_funnel_shadow_enabled: bool = False
    night_funnel_shadow_only: bool = True
    night_funnel_control_path: Path = NIGHT_FUNNEL_DEFAULT_CONTROL_PATH
    night_funnel_status_path: Path = NIGHT_FUNNEL_DEFAULT_STATUS_PATH
    night_funnel_shadow_log_path: Path = NIGHT_FUNNEL_DEFAULT_SHADOW_LOG_PATH
    night_funnel_lead_store_path: Path = NIGHT_FUNNEL_DEFAULT_LEAD_STORE_PATH
    night_funnel_live_token: str = ""
    night_funnel_expected_live_token: str = ""
    night_funnel_tee_enabled: bool = False
    night_funnel_tee_path: Path = NIGHT_FUNNEL_DEFAULT_TEE_PATH
    night_funnel_tee_source: str = "telegram_public_pilot"
    night_funnel_tee_retention_days: int = NIGHT_FUNNEL_DEFAULT_TEE_RETENTION_DAYS

    def __post_init__(self) -> None:
        brand = self.brand.casefold().strip()
        if brand not in {"foton", "unpk"}:
            raise ValueError("brand must be foton or unpk")
        if not self.token.strip():
            raise ValueError(f"Telegram token is missing for {brand}")
        if not 5 <= int(self.debounce_seconds) <= 10:
            raise ValueError("debounce_seconds must be between 5 and 10")
        object.__setattr__(self, "brand", brand)
        object.__setattr__(self, "snapshot_path", Path(self.snapshot_path))
        object.__setattr__(self, "log_dir", Path(self.log_dir))
        object.__setattr__(self, "heartbeat_path", Path(self.heartbeat_path))
        object.__setattr__(self, "cache_dir", Path(self.cache_dir))
        crm_mode = str(self.crm_read_mode or "off").casefold().strip()
        if crm_mode not in {"off", "local", "live", "server"}:
            raise ValueError("crm_read_mode must be off, local, live, or server")
        object.__setattr__(self, "crm_read_mode", crm_mode)
        object.__setattr__(self, "crm_env_file", Path(self.crm_env_file))
        object.__setattr__(self, "crm_server_url", str(self.crm_server_url or "").rstrip("/"))
        object.__setattr__(self, "crm_server_api_key", str(self.crm_server_api_key or ""))
        object.__setattr__(self, "store_path", Path(self.store_path))
        object.__setattr__(self, "p0_register_path", Path(self.p0_register_path))
        object.__setattr__(self, "night_funnel_control_path", Path(self.night_funnel_control_path))
        object.__setattr__(self, "night_funnel_status_path", Path(self.night_funnel_status_path))
        object.__setattr__(self, "night_funnel_shadow_log_path", Path(self.night_funnel_shadow_log_path))
        object.__setattr__(self, "night_funnel_lead_store_path", Path(self.night_funnel_lead_store_path))
        object.__setattr__(self, "night_funnel_tee_path", Path(self.night_funnel_tee_path))


@dataclass
class ChatSession:
    recent_messages: deque[str] = field(default_factory=lambda: deque(maxlen=MAX_RECENT_MESSAGES))
    pending_messages: list[str] = field(default_factory=list)
    pending_task: asyncio.Task[None] | None = None
    crm_context_task: asyncio.Task[dict[str, Any]] | None = None
    crm_context: Mapping[str, Any] = field(default_factory=dict)
    debug_phone: str = ""
    debug_client: Mapping[str, Any] = field(default_factory=dict)
    dialogue_memory: Mapping[str, Any] = field(default_factory=dict)
    utm: Mapping[str, str] = field(default_factory=dict)
    channel_source: str = ""
    processing_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@dataclass(frozen=True)
class DebugPhoneCommand:
    matched: bool
    phone: str = ""
    rest: str = ""


def parse_debug_phone_command(text: str) -> DebugPhoneCommand:
    match = DEBUG_PHONE_RE.match(str(text or ""))
    if not match:
        return DebugPhoneCommand(matched=False)
    phone = normalize_phone(match.group("phone") or "")
    rest = str(match.group("rest") or "").strip()
    return DebugPhoneCommand(matched=True, phone=phone, rest=rest)


def normalize_phone(phone: str) -> str:
    digits = PHONE_DIGIT_RE.sub("", str(phone or ""))
    if len(digits) == 11 and digits.startswith("8"):
        digits = "7" + digits[1:]
    if len(digits) == 10:
        digits = "7" + digits
    return digits


def load_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    result: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        cleaned = value.strip()
        if (cleaned.startswith('"') and cleaned.endswith('"')) or (cleaned.startswith("'") and cleaned.endswith("'")):
            try:
                cleaned = str(ast.literal_eval(cleaned))
            except (SyntaxError, ValueError):
                cleaned = cleaned[1:-1]
        result[key.strip()] = cleaned
    return result


def merged_env(path: Path | None) -> dict[str, str]:
    env = dict(os.environ)
    if path is not None:
        env.update(load_env_file(path))
    return env


def load_debug_clients(env: Mapping[str, str]) -> dict[str, Mapping[str, Any]]:
    raw = str(env.get(DEBUG_CLIENTS_ENV) or "").strip()
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{DEBUG_CLIENTS_ENV} must be valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{DEBUG_CLIENTS_ENV} must be a JSON object")
    result: dict[str, Mapping[str, Any]] = {}
    for phone, client in payload.items():
        if isinstance(client, Mapping):
            result[normalize_phone(str(phone))] = dict(client)
    return result


def env_flag(env: Mapping[str, str], key: str, *, default: bool = False) -> bool:
    raw = str(env.get(key) or "").strip().casefold()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on", "да"}


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_public_bot_heartbeat(
    path: Path,
    *,
    status: str,
    brands: Sequence[str],
    event: str = "",
    summary: Optional[Mapping[str, Any]] = None,
) -> None:
    write_json_atomic(
        path,
        {
            "schema_version": "public_pilot_bot_heartbeat_v1_2026_06_12",
            "status": str(status or "unknown"),
            "last_cycle_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "pid": os.getpid(),
            "brands": [str(item) for item in brands],
            "event": str(event or ""),
            "summary": dict(summary or {}),
        },
    )


def configs_from_env(env: Mapping[str, str], *, brand: str, allow_groups: bool = False) -> list[BrandBotConfig]:
    snapshot = Path(env.get(SNAPSHOT_ENV) or DEFAULT_SNAPSHOT)
    crm_env_file = Path(env.get(CRM_ENV_FILE_ENV) or DEFAULT_CRM_ENV_FILE)
    crm_read_mode = str(env.get(CRM_READ_MODE_ENV) or "live")
    crm_server_url = str(env.get(CRM_SERVER_URL_ENV) or env.get("AI_OFFICE_API_BASE_URL") or "")
    crm_server_api_key = str(env.get(CRM_SERVER_API_KEY_ENV) or env.get("AI_OFFICE_API_KEY") or "")
    model = str(env.get("MANGO_TELEGRAM_CODEX_MODEL") or "gpt-5.5")
    reasoning = str(env.get("MANGO_TELEGRAM_CODEX_REASONING") or "xhigh")
    timeout = int(env.get("MANGO_TELEGRAM_CODEX_TIMEOUT_SEC") or "240")
    debounce = int(env.get("MANGO_TELEGRAM_DEBOUNCE_SECONDS") or DEFAULT_DEBOUNCE_SECONDS)
    heartbeat_path = Path(env.get(PUBLIC_BOT_HEARTBEAT_PATH_ENV) or DEFAULT_HEARTBEAT_PATH)
    store_path = Path(env.get(PILOT_STORE_PATH_ENV) or DEFAULT_STORE_PATH)
    store_enabled = env_flag(env, PILOT_STORE_ENABLED_ENV, default=True)
    p0_register_path = Path(env.get(PILOT_P0_REGISTER_PATH_ENV) or DEFAULT_P0_REGISTER_PATH)
    autonomy_enabled = env_flag(env, PILOT_AUTONOMY_ENABLED_ENV, default=True)
    dialogue_contract_pipeline_enabled = env_flag(env, DIALOGUE_CONTRACT_PIPELINE_ENV, default=True)
    night_shadow_enabled = env_flag(env, NIGHT_FUNNEL_SHADOW_ENABLED_ENV, default=False)
    night_shadow_only = env_flag(env, NIGHT_FUNNEL_SHADOW_ONLY_ENV, default=True)
    night_control_path = Path(env.get(NIGHT_FUNNEL_CONTROL_PATH_ENV) or NIGHT_FUNNEL_DEFAULT_CONTROL_PATH)
    night_status_path = Path(env.get(NIGHT_FUNNEL_STATUS_PATH_ENV) or NIGHT_FUNNEL_DEFAULT_STATUS_PATH)
    night_log_path = Path(env.get(NIGHT_FUNNEL_LOG_PATH_ENV) or NIGHT_FUNNEL_DEFAULT_SHADOW_LOG_PATH)
    night_lead_store_path = Path(env.get(NIGHT_FUNNEL_LEAD_STORE_PATH_ENV) or NIGHT_FUNNEL_DEFAULT_LEAD_STORE_PATH)
    night_live_token = str(env.get(NIGHT_FUNNEL_LIVE_TOKEN_ENV) or "")
    night_expected_live_token = str(env.get(NIGHT_FUNNEL_EXPECTED_LIVE_TOKEN_ENV) or "")
    night_tee_enabled = env_flag(env, NIGHT_FUNNEL_TEE_ENABLED_ENV, default=False)
    night_tee_path = Path(env.get(NIGHT_FUNNEL_TEE_PATH_ENV) or NIGHT_FUNNEL_DEFAULT_TEE_PATH)
    night_tee_source = str(env.get(NIGHT_FUNNEL_TEE_SOURCE_ENV) or "telegram_public_pilot")
    night_tee_retention_days = int(env.get(NIGHT_FUNNEL_TEE_RETENTION_DAYS_ENV) or NIGHT_FUNNEL_DEFAULT_TEE_RETENTION_DAYS)
    selected = {"foton", "unpk"} if brand == "all" else {brand}
    configs: list[BrandBotConfig] = []
    if "foton" in selected:
        configs.append(
            BrandBotConfig(
                brand="foton",
                token=str(env.get(FOTON_TOKEN_ENV) or ""),
                display_name="Фотон",
                snapshot_path=snapshot,
                debounce_seconds=debounce,
                heartbeat_path=heartbeat_path,
                model=model,
                reasoning_effort=reasoning,
                timeout_sec=timeout,
                allow_groups=allow_groups,
                crm_read_mode=crm_read_mode,
                crm_env_file=crm_env_file,
                crm_server_url=crm_server_url,
                crm_server_api_key=crm_server_api_key,
                store_path=store_path,
                store_enabled=store_enabled,
                p0_register_path=p0_register_path,
                autonomy_enabled=autonomy_enabled,
                dialogue_contract_pipeline_enabled=dialogue_contract_pipeline_enabled,
                night_funnel_shadow_enabled=night_shadow_enabled,
                night_funnel_shadow_only=night_shadow_only,
                night_funnel_control_path=night_control_path,
                night_funnel_status_path=night_status_path,
                night_funnel_shadow_log_path=night_log_path,
                night_funnel_lead_store_path=night_lead_store_path,
                night_funnel_live_token=night_live_token,
                night_funnel_expected_live_token=night_expected_live_token,
                night_funnel_tee_enabled=night_tee_enabled,
                night_funnel_tee_path=night_tee_path,
                night_funnel_tee_source=night_tee_source,
                night_funnel_tee_retention_days=night_tee_retention_days,
            )
        )
    if "unpk" in selected:
        configs.append(
            BrandBotConfig(
                brand="unpk",
                token=str(env.get(UNPK_TOKEN_ENV) or ""),
                display_name="УНПК МФТИ",
                snapshot_path=snapshot,
                debounce_seconds=debounce,
                heartbeat_path=heartbeat_path,
                model=model,
                reasoning_effort=reasoning,
                timeout_sec=timeout,
                allow_groups=allow_groups,
                crm_read_mode=crm_read_mode,
                crm_env_file=crm_env_file,
                crm_server_url=crm_server_url,
                crm_server_api_key=crm_server_api_key,
                store_path=store_path,
                store_enabled=store_enabled,
                p0_register_path=p0_register_path,
                autonomy_enabled=autonomy_enabled,
                dialogue_contract_pipeline_enabled=dialogue_contract_pipeline_enabled,
                night_funnel_shadow_enabled=night_shadow_enabled,
                night_funnel_shadow_only=night_shadow_only,
                night_funnel_control_path=night_control_path,
                night_funnel_status_path=night_status_path,
                night_funnel_shadow_log_path=night_log_path,
                night_funnel_lead_store_path=night_lead_store_path,
                night_funnel_live_token=night_live_token,
                night_funnel_expected_live_token=night_expected_live_token,
                night_funnel_tee_enabled=night_tee_enabled,
                night_funnel_tee_path=night_tee_path,
                night_funnel_tee_source=night_tee_source,
                night_funnel_tee_retention_days=night_tee_retention_days,
            )
        )
    return configs


class PublicPilotBotRuntime:
    def __init__(self, config: BrandBotConfig, *, debug_clients: Mapping[str, Mapping[str, Any]]) -> None:
        self.config = config
        self.debug_clients = dict(debug_clients)
        self.sessions: dict[int, ChatSession] = {}
        self.provider = SubscriptionLlmDraftProvider(
            model=config.model,
            reasoning_effort=config.reasoning_effort,
            timeout_sec=config.timeout_sec,
            cache_dir=config.cache_dir / config.brand,
        )
        self.store: TelegramPilotSQLiteStore | None = TelegramPilotSQLiteStore(config.store_path) if config.store_enabled else None
        self.night_shadow_decisions: deque[Mapping[str, Any]] = deque(maxlen=200)

    def session(self, chat_id: int) -> ChatSession:
        item = self.sessions.get(chat_id)
        if item is None:
            item = ChatSession()
            restored = self.latest_dialogue_memory_for_chat(chat_id)
            if restored:
                item.dialogue_memory = restored
            self.sessions[chat_id] = item
        return item

    def latest_dialogue_memory_for_chat(self, chat_id: int) -> Mapping[str, Any]:
        if self.store is None:
            return {}
        try:
            snapshot = self.store.latest_dialogue_memory_snapshot(
                session_id=f"telegram_public_pilot:{self.config.brand}:{chat_id}",
                active_brand=self.config.brand,
            )
        except Exception as exc:  # noqa: BLE001
            self.log_event(
                "dialogue_memory_restore_error",
                chat_id=chat_id,
                payload={"brand": self.config.brand, "error": str(exc)[:240]},
            )
            return {}
        return dict(snapshot or {})

    def close(self) -> None:
        if self.store is not None:
            self.store.close()

    async def handle_start(self, update: Any, context: Any) -> None:
        chat_id = update.effective_chat.id
        session = self.session(int(chat_id))
        payload = " ".join(str(item) for item in getattr(context, "args", []) or [])
        if payload:
            session.utm = extract_utm(payload)
            session.channel_source = payload[:240]
        await update.effective_message.reply_text(greeting_for_brand(self.config.brand))
        self.log_event("start", chat_id=chat_id, payload={"brand": self.config.brand, "utm": dict(session.utm), "channel_source": session.channel_source})

    async def handle_reset(self, update: Any, context: Any) -> None:
        del context
        chat_id = update.effective_chat.id
        self.sessions.pop(chat_id, None)
        await update.effective_message.reply_text("Диалог очищен. Можно написать новый вопрос.")
        self.log_event("reset", chat_id=chat_id, payload={"brand": self.config.brand})

    async def handle_text(self, update: Any, context: Any) -> None:
        del context
        message = update.effective_message
        chat = update.effective_chat
        user = update.effective_user
        if message is None or chat is None:
            return
        if not self.config.allow_groups and getattr(chat, "type", "") != "private":
            return
        text = str(message.text or "").strip()
        if not text:
            return
        chat_id = int(chat.id)
        session = self.session(chat_id)

        command = parse_debug_phone_command(text)
        if command.matched:
            session.debug_phone = command.phone
            session.debug_client = dict(self.debug_clients.get(command.phone) or {})
            if session.crm_context_task is not None and not session.crm_context_task.done():
                session.crm_context_task.cancel()
            session.crm_context_task = asyncio.create_task(self.prefetch_crm_context(chat_id, command.phone))
            self.log_event(
                "debug_impersonation_set",
                chat_id=chat_id,
                payload={
                    "brand": self.config.brand,
                    "phone": command.phone,
                    "client_known": bool(session.debug_client),
                    "telegram_user_id": getattr(user, "id", None),
                },
            )
            if not command.rest:
                label = debug_client_label(session.debug_client)
                await message.reply_text(
                    f"Тестовый режим включён для номера {command.phone}."
                    + (f" Найден клиент: {label}." if label else "")
                    + " Подтягиваю CRM/Tallanto в фоне. Напишите следующий вопрос клиента."
                )
                return
            text = command.rest

        session.pending_messages.append(text)
        if session.pending_task is not None and not session.pending_task.done():
            session.pending_task.cancel()
        session.pending_task = asyncio.create_task(self._delayed_process(chat_id, message))
        self.log_event(
            "message_queued",
            chat_id=chat_id,
            payload={
                "brand": self.config.brand,
                "pending_count": len(session.pending_messages),
                "input_text": text,
                "debug_impersonation": bool(session.debug_phone),
            },
        )

    async def _delayed_process(self, chat_id: int, message: Any) -> None:
        try:
            await asyncio.sleep(self.config.debounce_seconds)
        except asyncio.CancelledError:
            return
        await self.process_pending(chat_id, message)

    async def process_pending(self, chat_id: int, message: Any) -> None:
        session = self.session(chat_id)
        async with session.processing_lock:
            if not session.pending_messages:
                return
            batch = list(session.pending_messages)
            session.pending_messages.clear()
            combined_text = "\n".join(batch).strip()
            if not combined_text:
                return
            await self.maybe_attach_prefetched_crm_context(chat_id, session, timeout_sec=2.0)
            context = self.build_context(chat_id=chat_id, session=session, current_text=combined_text)
            funnel_state = self.build_funnel_state(
                chat_id=chat_id,
                session=session,
                current_text=combined_text,
                context=context,
            )
            context = self.attach_funnel_state_to_context(context, funnel_state)
            request_started_at = datetime.now(timezone.utc)
            self.log_event(
                "llm_request_started",
                chat_id=chat_id,
                payload={
                    "brand": self.config.brand,
                    "message_count": len(batch),
                    "debug_impersonation": bool(session.debug_phone),
                    "input_text": combined_text,
                    "known_client_fields": context.get("known_client_fields") or {},
                    "known_dialog_fields": context.get("known_dialog_fields") or {},
                    "funnel_state": context.get("funnel_state") or {},
                    "context_flags": context_flags_for_report(context),
                },
            )
            typing_task = asyncio.create_task(self.show_typing_until_done(message))
            try:
                result = await asyncio.to_thread(self.provider.build_draft, combined_text, context=context)
            finally:
                typing_task.cancel()
                with suppress(asyncio.CancelledError):
                    await typing_task
            result = apply_public_autonomy_kill_switch(result, autonomy_enabled=self.config.autonomy_enabled)
            funnel_state = self.build_funnel_state(
                chat_id=chat_id,
                session=session,
                current_text=combined_text,
                context=context,
                result=result,
            )
            context = self.attach_funnel_state_to_context(context, funnel_state)
            text = public_reply_text(result)
            manager_summary = self.build_manager_summary(
                input_text=combined_text,
                answer_text=text,
                result=result,
                funnel_state=funnel_state,
                context=context,
            )
            self.record_night_shadow_decision(
                chat_id=chat_id,
                input_text=combined_text,
                answer_text=text,
                result=result,
                context=context,
                session=session,
            )
            reply_payload = public_telegram_reply_payload(text)
            sent_message = await message.reply_text(
                str(reply_payload["text"]),
                disable_web_page_preview=True,
                parse_mode=reply_payload["parse_mode"],
            )
            self.record_night_inbound_tee(
                chat_id=chat_id,
                input_text=combined_text,
                context=context,
                session=session,
                source_message=message,
                owner_message=sent_message,
                owner_route=result.route,
            )
            session.recent_messages.append(f"Клиент: {combined_text}")
            session.recent_messages.append(f"Ответ: {text}")
            updated_memory = update_dialogue_memory_after_answer(
                context.get("dialogue_memory_view") if isinstance(context.get("dialogue_memory_view"), Mapping) else {},
                answer_text=text,
                route=result.route,
                fact_refs=result.context_used,
                safety_flags=result.safety_flags,
            )
            session.dialogue_memory = updated_memory.to_json_dict()
            context = {**dict(context), "dialogue_memory_view": updated_memory.to_prompt_view()}
            latency = round((datetime.now(timezone.utc) - request_started_at).total_seconds(), 3)
            self.log_event(
                "reply_sent",
                chat_id=chat_id,
                payload={
                    "brand": self.config.brand,
                    "input_text": combined_text,
                    "answer_text": text,
                    "route": result.route,
                    "topic_id": result.topic_id,
                    "message_type": result.message_type,
                    "risk_level": result.risk_level,
                    "safety_flags": list(result.safety_flags),
                    "debug_impersonation": bool(session.debug_phone),
                    "known_client_fields": context.get("known_client_fields") or {},
                    "known_dialog_fields": context.get("known_dialog_fields") or {},
                    "funnel_state": context.get("funnel_state") or {},
                    "manager_summary_available": bool(manager_summary),
                    "context_flags": context_flags_for_report(context),
                    "latency_seconds": latency,
                    "text_chars": len(text),
                    "autonomy_enabled": self.config.autonomy_enabled,
                },
            )
            self.persist_pilot_decision(
                message=message,
                chat_id=chat_id,
                input_text=combined_text,
                answer_text=text,
                context=context,
                result=result,
                funnel_state=funnel_state,
                manager_summary=manager_summary,
                latency_seconds=latency,
                request_started_at=request_started_at,
            )
            self.persist_p0_register_if_needed(
                chat_id=chat_id,
                input_text=combined_text,
                answer_text=text,
                result=result,
            )

    async def show_typing_until_done(self, message: Any) -> None:
        bot = message.get_bot()
        chat_id = message.chat_id
        while True:
            await bot.send_chat_action(chat_id=chat_id, action="typing")
            await asyncio.sleep(4)

    async def prefetch_crm_context(self, chat_id: int, phone: str) -> dict[str, Any]:
        self.log_event("crm_prefetch_started", chat_id=chat_id, payload={"brand": self.config.brand})
        context = await asyncio.to_thread(
            build_read_only_crm_context,
            phone=phone,
            brand=self.config.brand,
            mode=self.config.crm_read_mode,
            crm_env_file=self.config.crm_env_file,
            crm_server_url=self.config.crm_server_url,
            crm_server_api_key=self.config.crm_server_api_key,
        )
        session = self.session(chat_id)
        session.crm_context = context
        amo = context.get("amo_context") if isinstance(context.get("amo_context"), Mapping) else {}
        tallanto = context.get("tallanto_context") if isinstance(context.get("tallanto_context"), Mapping) else {}
        local = context.get("local_runtime_context") if isinstance(context.get("local_runtime_context"), Mapping) else {}
        self.log_event(
            "crm_prefetch_finished",
            chat_id=chat_id,
            payload={
                "brand": self.config.brand,
                "local_status": local.get("status"),
                "amo_status": amo.get("status"),
                "tallanto_status": tallanto.get("status"),
            },
        )
        return context

    async def maybe_attach_prefetched_crm_context(self, chat_id: int, session: ChatSession, *, timeout_sec: float) -> None:
        task = session.crm_context_task
        if task is None:
            return
        if task.done():
            try:
                session.crm_context = task.result()
            except Exception as exc:  # noqa: BLE001
                self.log_event("crm_prefetch_error", chat_id=chat_id, payload={"brand": self.config.brand, "error": str(exc)[:240]})
            return
        try:
            session.crm_context = await asyncio.wait_for(asyncio.shield(task), timeout=timeout_sec)
        except asyncio.TimeoutError:
            self.log_event("crm_prefetch_still_running", chat_id=chat_id, payload={"brand": self.config.brand})
        except Exception as exc:  # noqa: BLE001
            self.log_event("crm_prefetch_error", chat_id=chat_id, payload={"brand": self.config.brand, "error": str(exc)[:240]})

    def build_context(self, *, chat_id: int, session: ChatSession, current_text: str) -> Mapping[str, Any]:
        if session.debug_phone:
            crm_context = dict(session.crm_context or {})
            if not crm_context:
                crm_context = build_read_only_crm_context(
                    phone=session.debug_phone,
                    brand=self.config.brand,
                    mode="local",
                    crm_env_file=self.config.crm_env_file,
                    crm_server_url=self.config.crm_server_url,
                    crm_server_api_key=self.config.crm_server_api_key,
                )
        else:
            crm_context = {}
        return build_pilot_context_payload(
            current_text=current_text,
            snapshot_path=self.config.snapshot_path,
            active_brand=self.config.brand,
            recent_messages=tuple(session.recent_messages)[-10:],
            dialogue_memory=session.dialogue_memory,
            session_id=f"telegram_public_pilot:{self.config.brand}:{chat_id}",
            channel="telegram_bot",
            channel_thread_id=str(chat_id),
            channel_user_id=str(chat_id),
            dialogue_contract_pipeline_enabled=self.config.dialogue_contract_pipeline_enabled,
            sends_client_replies=True,
            debug_impersonation_enabled=True,
            debug_phone=session.debug_phone,
            debug_client=session.debug_client,
            crm_context=crm_context,
        )

    def build_funnel_state(
        self,
        *,
        chat_id: int,
        session: ChatSession,
        current_text: str,
        context: Mapping[str, Any],
        result: SubscriptionDraftResult | None = None,
    ) -> LeadFunnelState:
        del chat_id
        return assemble_funnel_state(
            current_text=current_text,
            active_brand=self.config.brand,
            recent_messages=tuple(session.recent_messages)[-10:],
            context=context,
            result=result,
        )

    def attach_funnel_state_to_context(self, context: Mapping[str, Any], funnel_state: LeadFunnelState) -> Mapping[str, Any]:
        return assemble_funnel_state_context(context, funnel_state)

    def build_manager_summary(
        self,
        *,
        input_text: str,
        answer_text: str,
        result: SubscriptionDraftResult,
        funnel_state: LeadFunnelState,
        context: Mapping[str, Any],
    ) -> str:
        if result.route in AUTONOMOUS_ROUTES:
            return ""
        return build_manager_handoff_summary(
            brand=self.config.brand,
            client_message=input_text,
            answer_text=answer_text,
            route=result.route,
            topic_id=result.topic_id,
            risk_level=result.risk_level,
            safety_flags=result.safety_flags,
            missing_facts=result.missing_facts,
            manager_checklist=result.manager_checklist,
            funnel_state=funnel_state,
            context=context,
        )

    def record_night_shadow_decision(
        self,
        *,
        chat_id: int,
        input_text: str,
        answer_text: str,
        result: SubscriptionDraftResult,
        context: Mapping[str, Any],
        session: ChatSession,
    ) -> None:
        if not self.config.night_funnel_shadow_enabled:
            return
        try:
            control = load_bot_control(self.config.night_funnel_control_path)
            control = NightFunnelControl(
                enabled=control.enabled,
                mode=control.mode,
                shadow_only=True,
                manual_kill_switch=control.manual_kill_switch,
                live_token=self.config.night_funnel_live_token or control.live_token,
                expected_live_token=self.config.night_funnel_expected_live_token or control.expected_live_token,
                night_limit=control.night_limit,
                auto_trip_hold_rate=control.auto_trip_hold_rate,
                auto_trip_error_count=control.auto_trip_error_count,
                morning_followup_hour=control.morning_followup_hour,
                morning_followup_process_confirmed=control.morning_followup_process_confirmed,
            )
            pipeline = result.metadata.get("dialogue_contract_pipeline") if isinstance(result.metadata, Mapping) else {}
            retrieved_facts = pipeline.get("retrieved_facts") if isinstance(pipeline, Mapping) and isinstance(pipeline.get("retrieved_facts"), Mapping) else {}
            gate = evaluate_night_gate(
                client_text=input_text,
                draft_text=answer_text,
                route=result.route,
                active_brand=self.config.brand,
                snapshot_path=self.config.snapshot_path,
                retrieved_facts=retrieved_facts,
                safety_flags=result.safety_flags,
                control=control,
                prior_decisions=tuple(self.night_shadow_decisions),
            )
            channel_source = session.channel_source or self.config.display_name
            channel_brand = brand_from_channel(channel_source)
            if channel_brand and channel_brand != self.config.brand:
                gate = {
                    **dict(gate),
                    "decision": MANAGER_QUEUE,
                    "reason": f"channel_brand_mismatch:{channel_brand}!={self.config.brand}",
                    "shadow_only": True,
                }
            record = build_shadow_record(
                brand=self.config.brand,
                channel_source=channel_source,
                utm=session.utm,
                client_text=input_text,
                draft_text=answer_text,
                gate=gate,
                context=context,
            )
            lead_card = build_lead_card(
                brand=self.config.brand,
                utm=session.utm,
                client_text=input_text,
                draft_text=answer_text,
                decision=str(gate.get("decision") or ""),
                reason=str(gate.get("reason") or ""),
                context=context,
            )
            append_shadow_log(self.config.night_funnel_shadow_log_path, record)
            append_lead_card(self.config.night_funnel_lead_store_path, lead_card)
            self.night_shadow_decisions.append({"decision": gate.get("decision"), "reason": gate.get("reason")})
            write_bot_status(
                self.config.night_funnel_status_path,
                brand=self.config.brand,
                control=control,
                decisions=tuple(self.night_shadow_decisions),
                auto_tripped=bool(gate.get("auto_tripped")),
            )
            self.log_event(
                "night_funnel_shadow_decision",
                chat_id=chat_id,
                payload={
                    "brand": self.config.brand,
                    "decision": gate.get("decision"),
                    "reason": gate.get("reason"),
                    "shadow_only": True,
                    "fact_levels": (gate.get("fact_audit") or {}).get("counts_by_level") if isinstance(gate.get("fact_audit"), Mapping) else {},
                },
            )
        except Exception as exc:  # noqa: BLE001
            self.log_event("night_funnel_shadow_error", chat_id=chat_id, payload={"brand": self.config.brand, "error": str(exc)[:240]})

    def record_night_inbound_tee(
        self,
        *,
        chat_id: int,
        input_text: str,
        context: Mapping[str, Any],
        session: ChatSession,
        source_message: Any,
        owner_message: Any,
        owner_route: str,
    ) -> None:
        if not self.config.night_funnel_tee_enabled:
            return
        try:
            known_context = {
                "recent_messages": list(tuple(session.recent_messages)[-10:]),
                "known_slots": dict(context.get("known_slots") or {}),
                "known_dialog_fields": dict(context.get("known_dialog_fields") or {}),
                "known_client_fields": dict(context.get("known_client_fields") or {}),
            }
            record = build_inbound_tee_record(
                source=self.config.night_funnel_tee_source,
                brand=self.config.brand,
                channel_source=session.channel_source or self.config.display_name,
                utm=session.utm,
                chat_id=chat_id,
                message_id=getattr(source_message, "message_id", ""),
                message_at=telegram_message_datetime(source_message, fallback=datetime.now(timezone.utc)),
                text=input_text,
                known_context=known_context,
                owner_runtime={
                    "answered_by_owner": True,
                    "owner_route": str(owner_route or ""),
                    "owner_message_id": str(getattr(owner_message, "message_id", "") or ""),
                },
            )
            append_inbound_tee_record(self.config.night_funnel_tee_path, record)
            self.log_event(
                "night_funnel_inbound_tee_recorded",
                chat_id=chat_id,
                payload={"brand": self.config.brand, "tee_path": str(self.config.night_funnel_tee_path)},
            )
        except Exception as exc:  # noqa: BLE001
            self.log_event("night_funnel_inbound_tee_error", chat_id=chat_id, payload={"brand": self.config.brand, "error": str(exc)[:240]})

    def log_event(self, event: str, *, chat_id: int, payload: Mapping[str, Any]) -> None:
        self.config.log_dir.mkdir(parents=True, exist_ok=True)
        path = self.config.log_dir / f"{datetime.now(timezone.utc).date().isoformat()}_{self.config.brand}.jsonl"
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "event": event,
            "chat_id": chat_id,
            **dict(payload),
        }
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def persist_pilot_decision(
        self,
        *,
        message: Any,
        chat_id: int,
        input_text: str,
        answer_text: str,
        context: Mapping[str, Any],
        result: SubscriptionDraftResult,
        funnel_state: LeadFunnelState | None = None,
        manager_summary: str = "",
        latency_seconds: float,
        request_started_at: datetime,
    ) -> None:
        if self.store is None:
            return
        try:
            metadata = {
                "brand": self.config.brand,
                "latency_seconds": latency_seconds,
                "client_send_executed": True,
                "public_pilot_runtime": True,
                "autonomy_enabled": self.config.autonomy_enabled,
                "model": self.config.model,
                "reasoning_effort": self.config.reasoning_effort,
                "facts_used": list(result.context_used),
                "facts_missing": list(result.missing_facts),
                "post_filter_flags": list(result.metadata.get("post_filter_flags") or ()),
                "route_reason": "; ".join(result.manager_checklist[:3]) if result.manager_checklist else "",
                "llm_result": result.to_json_dict(),
            }
            fallback_reason = pilot_fallback_reason(result)
            if fallback_reason:
                metadata["fallback_reason"] = fallback_reason
            if funnel_state is not None:
                funnel_payload = funnel_state.to_json_dict()
                metadata.update(
                    {
                        "funnel_state": funnel_payload,
                        "lead_stage": funnel_payload.get("lead_stage"),
                        "client_segment": funnel_payload.get("client_segment"),
                        "next_step_type": funnel_payload.get("next_step_type"),
                        "known_slots": funnel_payload.get("filled_slots") or {},
                        "missing_slots": funnel_payload.get("missing_slots") or [],
                        "semantic_flags": funnel_payload.get("semantic_flags") or [],
                    }
                )
            if manager_summary:
                metadata["manager_summary"] = manager_summary
            store_result = self.store.upsert_message_context_draft(
                {
                    "channel": "telegram_public_pilot_bot",
                    "channel_message_id": str(getattr(message, "message_id", "") or stable_runtime_message_id(chat_id, request_started_at)),
                    "channel_thread_id": str(chat_id),
                    "channel_user_id": str(chat_id),
                    "direction": "inbound",
                    "text": input_text,
                    "received_at": telegram_message_datetime(message, fallback=request_started_at).isoformat(),
                    "metadata": {"brand": self.config.brand},
                },
                context=context,
                draft_text=answer_text,
                prompt_version=f"telegram_public_pilot:{self.config.model}:{self.config.reasoning_effort}",
                knowledge_base_version=knowledge_base_version_for_store(context, self.config.snapshot_path),
                status=PILOT_DRAFT_STATUS_MANAGER_ONLY if result.route == "manager_only" else PILOT_DRAFT_STATUS_NEEDS_REVIEW,
                topic_id=result.topic_id,
                route=result.route,
                safety_flags=result.safety_flags,
                draft_metadata=metadata,
                actor="telegram_public_pilot_bot",
            )
            memory_view = context.get("dialogue_memory_view")
            if isinstance(memory_view, Mapping) and memory_view.get("schema_version"):
                self.store.upsert_dialogue_memory_snapshot(
                    message_key=store_result.message_key,
                    session_id=str(memory_view.get("session_id") or f"telegram_public_pilot:{self.config.brand}:{chat_id}"),
                    active_brand=self.config.brand,
                    memory_snapshot=memory_view,
                    created_at=request_started_at,
                )
            self.log_event(
                "pilot_store_write",
                chat_id=chat_id,
                payload={"brand": self.config.brand, **store_result.to_json_dict()},
            )
        except Exception as exc:  # noqa: BLE001
            self.log_event(
                "pilot_store_error",
                chat_id=chat_id,
                payload={"brand": self.config.brand, "error": str(exc)[:240]},
            )

    def persist_p0_register_if_needed(
        self,
        *,
        chat_id: int,
        input_text: str,
        answer_text: str,
        result: SubscriptionDraftResult,
    ) -> None:
        try:
            record = build_p0_register_record(
                brand=self.config.brand,
                chat_id=chat_id,
                input_text=input_text,
                answer_text=answer_text,
                topic_id=result.topic_id,
                route=result.route,
                safety_flags=result.safety_flags,
                client_send_executed=True,
                metadata={"source": "telegram_public_pilot_runtime"},
            )
            if record is None:
                return
            path = append_p0_register_record(self.config.p0_register_path, record)
            self.log_event(
                "p0_register_recorded",
                chat_id=chat_id,
                payload={"brand": self.config.brand, "severity": record.severity, "path": str(path)},
            )
        except Exception as exc:  # noqa: BLE001
            self.log_event(
                "p0_register_error",
                chat_id=chat_id,
                payload={"brand": self.config.brand, "error": str(exc)[:240]},
            )


def public_reply_text(result: SubscriptionDraftResult) -> str:
    text = strip_internal_service_markers(str(result.draft_text or "")).strip()
    if not text:
        text = SAFE_FALLBACK_DRAFT_TEXT
    return text[:3900]


def public_telegram_html_enabled(env: Optional[Mapping[str, str]] = None) -> bool:
    source = env if env is not None else os.environ
    return env_flag(source, PUBLIC_PARSE_MODE_HTML_ENV, default=False)


def format_public_telegram_html(text: str) -> str:
    raw = str(text or "")
    parts: list[str] = []
    last = 0
    for match in PUBLIC_TELEGRAM_HIGHLIGHT_RE.finditer(raw):
        parts.append(html.escape(raw[last : match.start()], quote=False))
        parts.append(f"<b>{html.escape(match.group(0), quote=False)}</b>")
        last = match.end()
    parts.append(html.escape(raw[last:], quote=False))
    return "".join(parts)


def public_telegram_reply_payload(text: str, *, env: Optional[Mapping[str, str]] = None) -> Mapping[str, Any]:
    if not public_telegram_html_enabled(env):
        return {"text": text, "parse_mode": None}
    return {"text": format_public_telegram_html(text), "parse_mode": "HTML"}


def apply_public_autonomy_kill_switch(result: SubscriptionDraftResult, *, autonomy_enabled: bool) -> SubscriptionDraftResult:
    if autonomy_enabled or result.route not in AUTONOMOUS_ROUTES:
        return result
    return replace(
        result,
        route="draft_for_manager",
        draft_text=SAFE_FALLBACK_DRAFT_TEXT,
        safety_flags=(*result.safety_flags, "autonomy_kill_switch_applied"),
        metadata={**dict(result.metadata), "original_route_before_autonomy_kill_switch": result.route},
    )


def pilot_fallback_reason(result: SubscriptionDraftResult) -> str:
    flags = {str(flag or "").strip() for flag in result.safety_flags}
    if "llm_fallback" not in flags and not str(result.error or "").strip():
        return ""
    metadata = result.metadata if isinstance(result.metadata, Mapping) else {}
    direct_path = metadata.get("direct_path") if isinstance(metadata.get("direct_path"), Mapping) else {}
    evidence = direct_path.get("reason_evidence") if isinstance(direct_path.get("reason_evidence"), Mapping) else {}
    detail = str(evidence.get("provider_error") or metadata.get("last_error") or "").strip()
    reason = str(result.error or direct_path.get("reason_class") or "llm_fallback").strip()
    if detail and reason and detail != reason:
        return f"{reason}: {detail}"[:700]
    return (detail or reason)[:700]


def telegram_message_datetime(message: Any, *, fallback: datetime) -> datetime:
    value = getattr(message, "date", None)
    if isinstance(value, datetime):
        return value if value.tzinfo and value.utcoffset() is not None else value.replace(tzinfo=timezone.utc)
    return fallback


def stable_runtime_message_id(chat_id: int, created_at: datetime) -> str:
    return f"{chat_id}:{created_at.isoformat(timespec='seconds')}"


def knowledge_base_version_for_store(context: Mapping[str, Any], snapshot_path: Path) -> str:
    for key in ("knowledge_base_version", "kb_version", "snapshot_version", "context_version"):
        value = str(context.get(key) or "").strip()
        if value:
            return value[:160]
    return str(snapshot_path)


def greeting_for_brand(brand: str) -> str:
    if brand == "foton":
        return (
            "Здравствуйте! Помогу с вопросами об обучении в Фотоне: курсы, формат, стоимость, "
            "расписание, лагеря и следующий шаг для записи."
        )
    return (
        "Здравствуйте! Помогу с вопросами об обучении в УНПК МФТИ: курсы, формат, стоимость, "
        "расписание, выездные школы и следующий шаг для записи."
    )


def debug_client_label(client: Mapping[str, Any]) -> str:
    parent = str(client.get("parent_name") or "").strip()
    student = str(client.get("student_name") or "").strip()
    if parent and student:
        return f"{parent}, ученик {student}"
    return parent or student


def debug_customer_summary(phone: str, client: Mapping[str, Any]) -> str:
    if not phone:
        return ""
    label = debug_client_label(client)
    if label:
        return f"Тестовый режим сотрудника: отвечать как известному клиенту с телефона {phone}. Клиент: {label}."
    return f"Тестовый режим сотрудника: отвечать как известному клиенту с телефона {phone}. Клиент не найден в локальной тестовой карте."


def known_client_fields_for_session(*, session: ChatSession, crm_context: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    _merge_known_field_aliases(result, session.debug_client)
    if session.debug_phone:
        result.setdefault("phone", session.debug_phone)
    local = crm_context.get("local_runtime_context") if isinstance(crm_context.get("local_runtime_context"), Mapping) else {}
    if local:
        _merge_known_field_aliases(result, local)
        result.update({key: value for key, value in known_dialog_fields_from_messages([str(local.get("history_summary") or "")]).items() if value and key not in result})
    amo = crm_context.get("amo_context") if isinstance(crm_context.get("amo_context"), Mapping) else {}
    tallanto = crm_context.get("tallanto_context") if isinstance(crm_context.get("tallanto_context"), Mapping) else {}
    if amo.get("status") == "ok":
        result.setdefault("amo_context", "found")
    if tallanto.get("status") == "ok":
        result.setdefault("tallanto_context", "found")
    return {key: str(value)[:180] for key, value in result.items() if str(value or "").strip()}


def known_dialog_fields_from_messages(messages: Sequence[str], *, active_brand: str = "") -> dict[str, str]:
    client_parts: list[str] = []
    for item in messages:
        for raw_line in str(item or "").splitlines():
            line = raw_line.strip()
            lowered = line.casefold()
            if lowered.startswith("ответ:"):
                continue
            if lowered.startswith("клиент:"):
                line = line.split(":", 1)[1].strip()
            if line:
                client_parts.append(line)
    text = "\n".join(client_parts)
    normalized = text.casefold().replace("ё", "е")
    result: dict[str, str] = {}
    grade = re.search(r"\b(?P<grade>[1-9]|1[01])\s*(?:класс|кл\.?)\b", normalized)
    if grade:
        result["grade"] = grade.group("grade")
    subjects: list[str] = []
    for marker, canonical in (
        ("математ", "математика"),
        ("физик", "физика"),
        ("информат", "информатика"),
        ("программирован", "программирование"),
        ("русск", "русский язык"),
        ("англий", "английский язык"),
        ("хими", "химия"),
        ("биолог", "биология"),
    ):
        if marker in normalized:
            subjects.append(canonical)
    if subjects:
        result["subject"] = ", ".join(dict.fromkeys(subjects))
    if "онлайн" in normalized:
        result["format"] = "онлайн"
    elif "очно" in normalized or "офлайн" in normalized:
        result["format"] = "очно"
    if active_brand:
        result["active_brand"] = active_brand
    return result


def _merge_known_field_aliases(target: dict[str, str], source: Mapping[str, Any]) -> None:
    aliases = {
        "parent_name": ("parent_name", "parent", "parent_full_name", "fio_parent", "parent_fio"),
        "student_name": ("student_name", "student", "student_full_name", "fio_student", "student_fio", "child_name"),
        "phone": ("phone", "normalized_phone", "client_phone"),
        "grade": ("grade", "class", "student_grade", "klass"),
        "subject": ("subject", "course_subject", "interest_subject"),
        "known_course": ("known_course", "current_course", "course"),
        "current_group": ("current_group", "group", "tallanto_group"),
    }
    for normalized, keys in aliases.items():
        for key in keys:
            value = str(source.get(key) or "").strip()
            if value:
                target.setdefault(normalized, value)
                break


def build_known_context_summary(known_client_fields: Mapping[str, Any], known_dialog_fields: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for label, fields in (("Из CRM/локального контекста известно", known_client_fields), ("Из текущего диалога известно", known_dialog_fields)):
        public = {
            key: value
            for key, value in fields.items()
            if key in {"parent_name", "student_name", "grade", "subject", "format", "known_course", "current_group", "active_brand"}
        }
        if public:
            parts.append(f"{label}: " + ", ".join(f"{key}={value}" for key, value in public.items()))
    return "; ".join(parts)[:700]


def context_flags_for_report(context: Mapping[str, Any]) -> dict[str, bool]:
    quality = context.get("context_quality") if isinstance(context.get("context_quality"), Mapping) else {}
    return {
        "crm_context": bool(context.get("read_only_customer_context")),
        "known_client_fields": bool(context.get("known_client_fields")),
        "known_dialog_fields": bool(context.get("known_dialog_fields")),
        "family_phone": bool(quality.get("family_phone")),
        "multiple_students": bool(quality.get("multiple_students")),
        "multiple_deals": bool(quality.get("multiple_deals")),
        "facts_missing": bool(quality.get("facts_missing")),
    }


def build_read_only_crm_context(
    *,
    phone: str,
    brand: str,
    mode: str,
    crm_env_file: Path,
    crm_server_url: str = "",
    crm_server_api_key: str = "",
) -> dict[str, Any]:
    normalized_phone = normalize_phone(phone)
    normalized_mode = str(mode or "off").casefold().strip()
    if not normalized_phone or normalized_mode == "off":
        return {}
    context: dict[str, Any] = {
        "schema_version": "telegram_public_pilot_read_only_crm_context_v1",
        "phone": normalized_phone,
        "active_brand": brand,
        "read_only": True,
        "write_crm": False,
        "write_amo": False,
        "write_tallanto": False,
        "client_disclosure_policy": (
            "Use this context only to understand the situation. Do not quote CRM, AMO, Tallanto IDs, "
            "private notes, payment details, or personal data unless the client already stated it in chat."
        ),
        "risk_flags": ["crm_read_only_context_used", "do_not_disclose_crm_tallanto_private_data"],
        "warnings": [],
    }
    local_context = build_local_phone_context(normalized_phone)
    if local_context:
        context["local_runtime_context"] = local_context
    if normalized_mode == "server":
        context["amo_context"] = build_server_amo_context_readonly(
            normalized_phone,
            server_url=crm_server_url,
            api_key=crm_server_api_key,
        )
        context["tallanto_context"] = build_server_tallanto_context_readonly(
            normalized_phone,
            server_url=crm_server_url,
            api_key=crm_server_api_key,
        )
    if normalized_mode == "live":
        load_crm_env_into_process(crm_env_file)
        context["amo_context"] = build_live_amo_context_readonly(normalized_phone)
        tallanto_id = str((local_context.get("tallanto_id") if local_context else "") or "")
        tallanto_status = str((local_context.get("tallanto_match_status") if local_context else "") or "")
        context["tallanto_context"] = build_live_tallanto_context_readonly(
            normalized_phone,
            tallanto_id=tallanto_id,
            tallanto_match_status=tallanto_status,
        )
    context["timeline_context"] = build_timeline_hint_from_local_context(local_context)
    context["summary"] = summarize_read_only_crm_context(context)
    return context


def server_json_request(
    *,
    server_url: str,
    api_key: str,
    method: str,
    path: str,
    query: Mapping[str, Any] | None = None,
    payload: Mapping[str, Any] | None = None,
    timeout_sec: int = 20,
) -> dict[str, Any]:
    base = str(server_url or "").rstrip("/")
    if not base:
        return {"status": "unavailable", "reason": "crm_server_url_missing"}
    if not api_key:
        return {"status": "unavailable", "reason": "crm_server_api_key_missing"}
    url = f"{base}{path}"
    if query:
        encoded = url_parse.urlencode({key: value for key, value in query.items() if value is not None})
        url = f"{url}?{encoded}"
    data = None
    headers = {"X-API-Key": api_key, "Accept": "application/json", "User-Agent": "mango-telegram-pilot-bot"}
    if payload is not None:
        data = json.dumps(dict(payload), ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = url_request.Request(url, data=data, headers=headers, method=method.upper())
    try:
        with url_request.urlopen(request, timeout=timeout_sec) as response:
            raw = response.read().decode("utf-8")
            decoded = json.loads(raw) if raw.strip() else {}
            return decoded if isinstance(decoded, dict) else {"status": "ok", "data": decoded}
    except url_error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:400]
        return {"status": "error", "source": "crm_server", "http_status": exc.code, "error": detail}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "source": "crm_server", "error": str(exc)[:240]}


def build_server_amo_context_readonly(phone: str, *, server_url: str, api_key: str) -> dict[str, Any]:
    payload = server_json_request(
        server_url=server_url,
        api_key=api_key,
        method="GET",
        path="/api/integrations/amocrm/leads/by-phone",
        query={"phone": phone},
        timeout_sec=25,
    )
    if payload.get("status") not in {"ok", "matched", "not_found"}:
        return {"enabled": True, "status": payload.get("status") or "error", "source": "amo_server", **payload}
    contacts = payload.get("contacts") if isinstance(payload.get("contacts"), list) else []
    leads = payload.get("leads") if isinstance(payload.get("leads"), list) else []
    return {
        "enabled": True,
        "status": "ok" if contacts else "not_found",
        "source": "amo_server",
        "contacts_found": int(payload.get("contact_count") or len(contacts)),
        "contacts": [compact_amo_contact(item) for item in contacts[:5] if isinstance(item, Mapping)],
        "leads_found": int(payload.get("lead_count") or len(leads)),
        "leads": [compact_amo_lead(item) for item in leads[:10] if isinstance(item, Mapping)],
        "read_only": True,
    }


def build_server_tallanto_context_readonly(phone: str, *, server_url: str, api_key: str) -> dict[str, Any]:
    payload = server_json_request(
        server_url=server_url,
        api_key=api_key,
        method="GET",
        path="/api/integrations/tallanto/context/by-phone",
        query={"phone": phone, "max_contacts": 1, "max_related_records": 10},
        timeout_sec=12,
    )
    if payload.get("status") != "ok":
        return {"enabled": True, "status": payload.get("status") or "error", "source": "tallanto_server", **payload}
    return {**payload, "source": "tallanto_server", "read_only": True}


def load_crm_env_into_process(path: Path) -> None:
    if not path.exists():
        return
    for key, value in load_env_file(path).items():
        os.environ.setdefault(key, value)


def build_local_phone_context(phone: str) -> dict[str, Any]:
    try:
        from mango_mvp.amocrm_runtime.phone_context import get_phone_context
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "source": "local_runtime", "error": str(exc)[:240]}
    try:
        phone_context = get_phone_context(phone)
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "source": "local_runtime", "error": str(exc)[:240]}
    if phone_context is None:
        return {"status": "not_found", "source": "local_runtime"}
    return {
        "status": "ok",
        "source": "local_runtime",
        "phone": phone_context.phone,
        "source_dir": phone_context.source_dir,
        "first_call_at": phone_context.first_call_at,
        "last_call_at": phone_context.last_call_at,
        "manager_history": phone_context.manager_history[:5],
        "interest_summary": phone_context.interest_summary,
        "objections_summary": phone_context.objections_summary,
        "current_sales_temperature": phone_context.current_sales_temperature,
        "recommended_next_step": phone_context.recommended_next_step,
        "follow_up_due_at": phone_context.follow_up_due_at,
        "history_summary": phone_context.history_summary,
        "chronology": phone_context.chronology,
        "tallanto_id": phone_context.tallanto_id,
        "tallanto_match_status": phone_context.tallanto_match_status,
        "call_count": len(phone_context.call_rows),
    }


def build_live_amo_context_readonly(phone: str) -> dict[str, Any]:
    try:
        from mango_mvp.amocrm_runtime import amo_integration as amo
        from mango_mvp.amocrm_runtime.db import SessionLocal
    except Exception as exc:  # noqa: BLE001
        return {"enabled": True, "status": "error", "source": "amo_live", "error": str(exc)[:240]}

    try:
        with SessionLocal() as session:
            access = resolve_amo_readonly_access(amo, session)
            if not access.get("ok"):
                return {
                    "enabled": True,
                    "status": "unavailable",
                    "source": "amo_live",
                    "reason": access.get("reason") or "amo_readonly_access_unavailable",
                }
            contacts = amo_search_contacts_readonly(amo, access, phone)
            compact_contacts: list[dict[str, Any]] = []
            leads: list[dict[str, Any]] = []
            for contact in contacts[:3]:
                contact_id = int(contact.get("id") or 0)
                compact_contacts.append(compact_amo_contact(contact))
                if contact_id:
                    leads.extend(amo_fetch_related_leads_readonly(amo, access, contact_id)[:5])
            return {
                "enabled": True,
                "status": "ok",
                "source": "amo_live",
                "contacts_found": len(compact_contacts),
                "contacts": compact_contacts,
                "leads_found": len(leads),
                "leads": [compact_amo_lead(item) for item in leads[:10]],
                "read_only": True,
            }
    except Exception as exc:  # noqa: BLE001
        return {"enabled": True, "status": "error", "source": "amo_live", "error": str(exc)[:240]}


def resolve_amo_readonly_access(amo: Any, session: Any) -> dict[str, Any]:
    env_token = str(getattr(amo.settings, "crm_amo_api_token", "") or "").strip()
    env_base = amo._normalize_base_url(getattr(amo.settings, "crm_amo_base_url", None))
    if env_token and env_base:
        return {"ok": True, "account_base_url": env_base, "access_token": env_token, "token_source": "env"}
    connection = amo.get_active_connection(session)
    if connection is None:
        return {"ok": False, "reason": "amo_oauth_connection_missing"}
    if amo._token_is_stale(connection):
        return {"ok": False, "reason": "amo_oauth_token_stale_not_refreshed_by_bot"}
    if not connection.access_token or not connection.account_base_url:
        return {"ok": False, "reason": "amo_oauth_token_missing"}
    return {
        "ok": True,
        "account_base_url": connection.account_base_url,
        "access_token": connection.access_token,
        "token_source": "oauth_readonly_no_refresh",
    }


def amo_search_contacts_readonly(amo: Any, access: Mapping[str, Any], phone: str) -> list[dict[str, Any]]:
    normalized_phone = normalize_phone(phone)
    if not normalized_phone:
        return []
    url = amo._contacts_search_endpoint(str(access["account_base_url"]))
    query = url_parse.urlencode({"query": normalized_phone[-10:], "limit": 10, "with": "leads"})
    payload = amo._amo_http_request(
        method="GET",
        url=f"{url}?{query}",
        headers={"Authorization": f"Bearer {access['access_token']}"},
    )
    contacts = (payload.get("_embedded") or {}).get("contacts") or []
    return [contact for contact in contacts if isinstance(contact, dict) and normalized_phone in amo._contact_phones(contact)]


def amo_fetch_related_leads_readonly(amo: Any, access: Mapping[str, Any], contact_id: int) -> list[dict[str, Any]]:
    url = amo._leads_collection_endpoint(str(access["account_base_url"]))
    query = url_parse.urlencode({"filter[contacts]": int(contact_id), "limit": 50})
    payload = amo._amo_http_request(
        method="GET",
        url=f"{url}?{query}",
        headers={"Authorization": f"Bearer {access['access_token']}"},
    )
    leads = (payload.get("_embedded") or {}).get("leads") or []
    return [item for item in leads if isinstance(item, dict)]


def compact_amo_contact(contact: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": contact.get("id"),
        "name": str(contact.get("name") or "").strip(),
        "responsible_user_id": contact.get("responsible_user_id"),
        "updated_at": contact.get("updated_at"),
        "created_at": contact.get("created_at"),
    }


def compact_amo_lead(lead: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": lead.get("id"),
        "name": str(lead.get("name") or "").strip(),
        "status_id": lead.get("status_id"),
        "pipeline_id": lead.get("pipeline_id"),
        "price": lead.get("price"),
        "responsible_user_id": lead.get("responsible_user_id"),
        "closed_at": lead.get("closed_at"),
        "updated_at": lead.get("updated_at"),
        "created_at": lead.get("created_at"),
    }


def build_live_tallanto_context_readonly(
    phone: str,
    *,
    tallanto_id: str = "",
    tallanto_match_status: str = "",
) -> dict[str, Any]:
    try:
        from mango_mvp.amocrm_runtime.tallanto_context import build_live_tallanto_context
    except Exception as exc:  # noqa: BLE001
        return {"enabled": True, "status": "error", "source": "tallanto_live", "error": str(exc)[:240]}
    try:
        payload = build_live_tallanto_context(
            phone=phone,
            tallanto_id=tallanto_id or None,
            tallanto_match_status=tallanto_match_status or None,
            max_related_records=20,
        )
        return {**dict(payload), "source": "tallanto_live", "read_only": True}
    except Exception as exc:  # noqa: BLE001
        return {"enabled": True, "status": "error", "source": "tallanto_live", "error": str(exc)[:240]}


def build_timeline_hint_from_local_context(local_context: Mapping[str, Any] | None) -> dict[str, Any]:
    if not local_context or local_context.get("status") != "ok":
        return {"found": False, "source": "local_runtime_hint"}
    return {
        "found": True,
        "source": "local_runtime_hint",
        "summary": str(local_context.get("history_summary") or "")[:900],
        "last_call_at": local_context.get("last_call_at"),
        "call_count": local_context.get("call_count"),
        "read_only": True,
    }


def summarize_read_only_crm_context(context: Mapping[str, Any]) -> str:
    parts: list[str] = []
    local = context.get("local_runtime_context") if isinstance(context.get("local_runtime_context"), Mapping) else {}
    if local.get("status") == "ok":
        if local.get("history_summary"):
            parts.append(f"История клиента: {str(local['history_summary'])[:700]}")
        if local.get("interest_summary"):
            parts.append(f"Интересы клиента: {str(local['interest_summary'])[:400]}")
        if local.get("recommended_next_step"):
            parts.append(f"Внутренний следующий шаг: {str(local['recommended_next_step'])[:300]}")
    amo = context.get("amo_context") if isinstance(context.get("amo_context"), Mapping) else {}
    if amo.get("status") == "ok":
        parts.append(f"AMO read-only: найдено контактов {amo.get('contacts_found')}, сделок {amo.get('leads_found')}.")
    elif amo:
        parts.append(f"AMO read-only: {amo.get('status')} ({amo.get('reason') or amo.get('error') or 'нет данных'}).")
    tallanto = context.get("tallanto_context") if isinstance(context.get("tallanto_context"), Mapping) else {}
    if tallanto.get("status") == "ok":
        parts.append(f"Tallanto read-only: найдено контактов {tallanto.get('contacts_found')}.")
    elif tallanto:
        parts.append(f"Tallanto read-only: {tallanto.get('status')} ({tallanto.get('reason') or tallanto.get('error') or 'нет данных'}).")
    if parts:
        parts.append("Служебные данные AMO/Tallanto не раскрывать клиенту; использовать только для понимания ситуации.")
    return "\n".join(parts)


async def get_me(configs: Sequence[BrandBotConfig]) -> list[Mapping[str, Any]]:
    from telegram import Bot

    results: list[Mapping[str, Any]] = []
    for config in configs:
        bot = Bot(config.token)
        me = await bot.get_me()
        results.append({"brand": config.brand, "username": me.username, "id": me.id, "first_name": me.first_name})
    return results


async def public_bot_heartbeat_loop(
    configs: Sequence[BrandBotConfig],
    stop_event: asyncio.Event,
    *,
    interval_sec: int = 60,
) -> None:
    if not configs:
        return
    heartbeat_path = configs[0].heartbeat_path
    brands = [config.brand for config in configs]
    counter = 0
    while not stop_event.is_set():
        counter += 1
        write_public_bot_heartbeat(
            heartbeat_path,
            status="polling",
            brands=brands,
            event="heartbeat",
            summary={
                "counter": counter,
                "snapshot": str(configs[0].snapshot_path),
                "model": configs[0].model,
                "reasoning_effort": configs[0].reasoning_effort,
            },
        )
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=max(1, int(interval_sec)))
        except asyncio.TimeoutError:
            continue


async def run_polling(configs: Sequence[BrandBotConfig], *, debug_clients: Mapping[str, Mapping[str, Any]], duration_sec: int | None) -> None:
    from telegram import Update
    from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

    runtimes = [PublicPilotBotRuntime(config, debug_clients=debug_clients) for config in configs]
    applications = []
    stop_event = asyncio.Event()
    heartbeat_task: asyncio.Task[None] | None = None

    def stop_signal(*_: Any) -> None:
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_signal)
        except NotImplementedError:
            pass

    for runtime in runtimes:
        app = Application.builder().token(runtime.config.token).build()

        async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE, *, rt: PublicPilotBotRuntime = runtime) -> None:
            await rt.handle_start(update, context)

        async def reset_handler(update: Update, context: ContextTypes.DEFAULT_TYPE, *, rt: PublicPilotBotRuntime = runtime) -> None:
            await rt.handle_reset(update, context)

        async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE, *, rt: PublicPilotBotRuntime = runtime) -> None:
            await rt.handle_text(update, context)

        app.add_handler(CommandHandler("start", start_handler))
        app.add_handler(CommandHandler("reset", reset_handler))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))
        await app.initialize()
        await app.start()
        if app.updater is None:
            raise RuntimeError("Telegram updater is unavailable")
        await app.updater.start_polling(allowed_updates=("message", "edited_message"))
        applications.append(app)
        runtime.log_event("polling_started", chat_id=0, payload={"brand": runtime.config.brand, "duration_sec": duration_sec})
    heartbeat_task = asyncio.create_task(public_bot_heartbeat_loop(configs, stop_event))

    try:
        if duration_sec is None:
            await stop_event.wait()
        else:
            await asyncio.sleep(max(1, int(duration_sec)))
    finally:
        stop_event.set()
        if heartbeat_task is not None:
            heartbeat_task.cancel()
            with suppress(asyncio.CancelledError):
                await heartbeat_task
        for app in reversed(applications):
            if app.updater is not None:
                await app.updater.stop()
            await app.stop()
            await app.shutdown()
        for runtime in runtimes:
            runtime.close()
        if configs:
            write_public_bot_heartbeat(
                configs[0].heartbeat_path,
                status="stopped",
                brands=[config.brand for config in configs],
                event="polling_stopped",
            )


def write_local_env_file(path: Path, *, foton_token: str, unpk_token: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        FOTON_TOKEN_ENV: foton_token,
        UNPK_TOKEN_ENV: unpk_token,
        SNAPSHOT_ENV: str(DEFAULT_SNAPSHOT),
        "MANGO_TELEGRAM_CODEX_MODEL": "gpt-5.5",
        "MANGO_TELEGRAM_CODEX_REASONING": "xhigh",
        "MANGO_TELEGRAM_CODEX_TIMEOUT_SEC": "240",
        "MANGO_TELEGRAM_DEBOUNCE_SECONDS": str(DEFAULT_DEBOUNCE_SECONDS),
        PILOT_STORE_PATH_ENV: str(DEFAULT_STORE_PATH),
        PILOT_STORE_ENABLED_ENV: "1",
        PILOT_P0_REGISTER_PATH_ENV: str(DEFAULT_P0_REGISTER_PATH),
        PILOT_AUTONOMY_ENABLED_ENV: "1",
        CRM_READ_MODE_ENV: "server",
        CRM_ENV_FILE_ENV: str(DEFAULT_CRM_ENV_FILE),
        CRM_SERVER_URL_ENV: "https://api.fotonai.online",
        CRM_SERVER_API_KEY_ENV: "",
        DEBUG_CLIENTS_ENV: json.dumps(
            {
                "79092009933": {
                    "student_name": "Колосов Даниил Максимович",
                    "parent_name": "Ананьевская Анна Георгиевна",
                }
            },
            ensure_ascii=False,
        ),
    }
    lines = ["# Local Telegram pilot bot secrets. Do not commit.\n"]
    for key, value in payload.items():
        escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f'{key}="{escaped}"\n')
    path.write_text("".join(lines), encoding="utf-8")
    path.chmod(0o600)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run public Telegram pilot bots for Foton and UNPK.")
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--brand", choices=("all", "foton", "unpk"), default="all")
    parser.add_argument("--mode", choices=("getme", "poll", "write-local-env"), default="getme")
    parser.add_argument("--duration-sec", type=int, default=None, help="Bounded polling smoke duration; omit for continuous run.")
    parser.add_argument("--allow-groups", action="store_true")
    parser.add_argument("--foton-token", default="")
    parser.add_argument("--unpk-token", default="")
    args = parser.parse_args(argv)

    if args.mode == "write-local-env":
        write_local_env_file(args.env_file, foton_token=args.foton_token, unpk_token=args.unpk_token)
        print(json.dumps({"env_file": str(args.env_file), "written": True}, ensure_ascii=False))
        return 0

    env = merged_env(args.env_file)
    configs = configs_from_env(env, brand=args.brand, allow_groups=args.allow_groups)
    debug_clients = load_debug_clients(env)
    if args.mode == "getme":
        results = asyncio.run(get_me(configs))
        print(json.dumps({"ok": True, "bots": results}, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    asyncio.run(run_polling(configs, debug_clients=debug_clients, duration_sec=args.duration_sec))
    print(json.dumps({"ok": True, "mode": "poll", "stopped": True}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
