#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import asyncio
import json
import os
import re
import signal
from contextlib import suppress
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib import parse as url_parse
from urllib import request as url_request
from urllib import error as url_error

from mango_mvp.channels.subscription_llm import (
    AUTONOMY_MATRIX_SAFE_TOPIC_IDS,
    SAFE_FALLBACK_DRAFT_TEXT,
    SubscriptionDraftResult,
    SubscriptionLlmDraftProvider,
    strip_internal_service_markers,
)
from mango_mvp.channels.telegram_pilot_context_builder import build_telegram_pilot_context_from_snapshot


DEFAULT_ENV_FILE = Path("/Users/dmitrijfabarisov/.codex/mango_telegram_pilot_bots.env")
DEFAULT_SNAPSHOT = Path("product_data/knowledge_base/kb_release_20260520_v6_3_team_answers/kb_release_v3_snapshot.json")
DEFAULT_CRM_ENV_FILE = Path("stable_runtime/amocrm_runtime/.env.private")
DEFAULT_LOG_DIR = Path(".codex_local/telegram_pilot_bots/logs")
DEFAULT_CACHE_DIR = Path(".codex_local/telegram_pilot_bots/llm_cache")
DEFAULT_DEBOUNCE_SECONDS = 7
MAX_RECENT_MESSAGES = 12

FOTON_TOKEN_ENV = "MANGO_TELEGRAM_FOTON_BOT_TOKEN"
UNPK_TOKEN_ENV = "MANGO_TELEGRAM_UNPK_BOT_TOKEN"
SNAPSHOT_ENV = "MANGO_TELEGRAM_KB_SNAPSHOT"
DEBUG_CLIENTS_ENV = "MANGO_TELEGRAM_DEBUG_CLIENTS_JSON"
CRM_READ_MODE_ENV = "MANGO_TELEGRAM_CRM_READ_MODE"
CRM_ENV_FILE_ENV = "MANGO_TELEGRAM_CRM_ENV_FILE"
CRM_SERVER_URL_ENV = "MANGO_CRM_SERVER_URL"
CRM_SERVER_API_KEY_ENV = "MANGO_CRM_SERVER_API_KEY"

DEBUG_PHONE_RE = re.compile(
    r"^\s*[\"'«»“”]*\s*представь\s*,?\s*что\s+я\s+пишу\s+с\s+номера\s+"
    r"(?P<phone>\+?\d[\d\s()\-]{7,})"
    r"\s*[\"'«»“”]*(?:[,:;.\-—]\s*(?P<rest>.*))?$",
    re.I,
)
PHONE_DIGIT_RE = re.compile(r"\D+")


@dataclass(frozen=True)
class BrandBotConfig:
    brand: str
    token: str
    display_name: str
    snapshot_path: Path
    debounce_seconds: int = DEFAULT_DEBOUNCE_SECONDS
    log_dir: Path = DEFAULT_LOG_DIR
    cache_dir: Path = DEFAULT_CACHE_DIR
    model: str = "gpt-5.5"
    reasoning_effort: str = "xhigh"
    timeout_sec: int = 240
    allow_groups: bool = False
    crm_read_mode: str = "live"
    crm_env_file: Path = DEFAULT_CRM_ENV_FILE
    crm_server_url: str = ""
    crm_server_api_key: str = ""

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
        object.__setattr__(self, "cache_dir", Path(self.cache_dir))
        crm_mode = str(self.crm_read_mode or "off").casefold().strip()
        if crm_mode not in {"off", "local", "live", "server"}:
            raise ValueError("crm_read_mode must be off, local, live, or server")
        object.__setattr__(self, "crm_read_mode", crm_mode)
        object.__setattr__(self, "crm_env_file", Path(self.crm_env_file))
        object.__setattr__(self, "crm_server_url", str(self.crm_server_url or "").rstrip("/"))
        object.__setattr__(self, "crm_server_api_key", str(self.crm_server_api_key or ""))


@dataclass
class ChatSession:
    recent_messages: deque[str] = field(default_factory=lambda: deque(maxlen=MAX_RECENT_MESSAGES))
    pending_messages: list[str] = field(default_factory=list)
    pending_task: asyncio.Task[None] | None = None
    crm_context_task: asyncio.Task[dict[str, Any]] | None = None
    crm_context: Mapping[str, Any] = field(default_factory=dict)
    debug_phone: str = ""
    debug_client: Mapping[str, Any] = field(default_factory=dict)
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
                model=model,
                reasoning_effort=reasoning,
                timeout_sec=timeout,
                allow_groups=allow_groups,
                crm_read_mode=crm_read_mode,
                crm_env_file=crm_env_file,
                crm_server_url=crm_server_url,
                crm_server_api_key=crm_server_api_key,
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
                model=model,
                reasoning_effort=reasoning,
                timeout_sec=timeout,
                allow_groups=allow_groups,
                crm_read_mode=crm_read_mode,
                crm_env_file=crm_env_file,
                crm_server_url=crm_server_url,
                crm_server_api_key=crm_server_api_key,
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

    def session(self, chat_id: int) -> ChatSession:
        item = self.sessions.get(chat_id)
        if item is None:
            item = ChatSession()
            self.sessions[chat_id] = item
        return item

    async def handle_start(self, update: Any, context: Any) -> None:
        del context
        chat_id = update.effective_chat.id
        await update.effective_message.reply_text(greeting_for_brand(self.config.brand))
        self.log_event("start", chat_id=chat_id, payload={"brand": self.config.brand})

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
            payload={"brand": self.config.brand, "pending_count": len(session.pending_messages)},
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
            self.log_event(
                "llm_request_started",
                chat_id=chat_id,
                payload={
                    "brand": self.config.brand,
                    "message_count": len(batch),
                    "debug_impersonation": bool(session.debug_phone),
                },
            )
            typing_task = asyncio.create_task(self.show_typing_until_done(message))
            try:
                result = await asyncio.to_thread(self.provider.build_draft, combined_text, context=context)
            finally:
                typing_task.cancel()
                with suppress(asyncio.CancelledError):
                    await typing_task
            text = public_reply_text(result)
            await message.reply_text(text, disable_web_page_preview=True)
            session.recent_messages.append(f"Клиент: {combined_text}")
            session.recent_messages.append(f"Ответ: {text}")
            self.log_event(
                "reply_sent",
                chat_id=chat_id,
                payload={
                    "brand": self.config.brand,
                    "route": result.route,
                    "topic_id": result.topic_id,
                    "message_type": result.message_type,
                    "risk_level": result.risk_level,
                    "safety_flags": list(result.safety_flags),
                    "debug_impersonation": bool(session.debug_phone),
                    "text_chars": len(text),
                },
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
        client_identity: dict[str, Any] = {
            "channel": "telegram_bot",
            "channel_thread_id": str(chat_id),
            "channel_user_id": str(chat_id),
        }
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
            client_identity.update(
                {
                    "phone": session.debug_phone,
                    "debug_impersonation": True,
                    **dict(session.debug_client),
                }
            )
        else:
            crm_context = {}
        customer_summary = debug_customer_summary(session.debug_phone, session.debug_client)
        if crm_context.get("summary"):
            customer_summary = "\n".join(item for item in (customer_summary, str(crm_context["summary"])) if item)
        rop_policy = {
            "bot_permission": "bot_answer_self_for_pilot",
            "autonomy_policy": {
                "allow_autonomous": True,
                "allowed_topic_ids": sorted(AUTONOMY_MATRIX_SAFE_TOPIC_IDS),
                "default": "draft_for_manager_or_manager_only",
                "fact_requirement": "client_safe_fact_verified",
                "p0_overrides_autonomy": True,
            },
        }
        pilot_context = build_telegram_pilot_context_from_snapshot(
            current_text,
            snapshot_path=self.config.snapshot_path,
            active_brand=self.config.brand,
            rop_policy=rop_policy,
            recent_messages=tuple(session.recent_messages)[-10:],
            client_identity=client_identity,
            customer_summary=customer_summary,
            amo_context=crm_context.get("amo_context") if isinstance(crm_context.get("amo_context"), Mapping) else None,
            tallanto_context=crm_context.get("tallanto_context")
            if isinstance(crm_context.get("tallanto_context"), Mapping)
            else None,
            timeline_context=crm_context.get("timeline_context")
            if isinstance(crm_context.get("timeline_context"), Mapping)
            else None,
            risk_flags=tuple(crm_context.get("risk_flags") or ()),
        )
        payload = dict(pilot_context.to_prompt_context())
        payload["active_brand"] = self.config.brand
        if crm_context:
            payload["read_only_customer_context"] = crm_context
        payload["public_pilot_mode"] = {
            "enabled": True,
            "sends_client_replies": True,
            "debug_impersonation_enabled": True,
            "brand_isolation_required": True,
            "no_crm_tallanto_write": True,
            "crm_tallanto_read_only": True,
            "do_not_disclose_crm_tallanto_private_data": True,
        }
        return payload

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


def public_reply_text(result: SubscriptionDraftResult) -> str:
    text = strip_internal_service_markers(str(result.draft_text or "")).strip()
    if not text:
        text = SAFE_FALLBACK_DRAFT_TEXT
    return text[:3900]


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


async def run_polling(configs: Sequence[BrandBotConfig], *, debug_clients: Mapping[str, Mapping[str, Any]], duration_sec: int | None) -> None:
    from telegram import Update
    from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

    runtimes = [PublicPilotBotRuntime(config, debug_clients=debug_clients) for config in configs]
    applications = []
    stop_event = asyncio.Event()

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

    try:
        if duration_sec is None:
            await stop_event.wait()
        else:
            await asyncio.sleep(max(1, int(duration_sec)))
    finally:
        for app in reversed(applications):
            if app.updater is not None:
                await app.updater.stop()
            await app.stop()
            await app.shutdown()


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
