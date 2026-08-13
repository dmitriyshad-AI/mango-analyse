#!/usr/bin/env python3
"""Тонкий публичный Telegram-транспорт к общему модельному ядру Mango."""
from __future__ import annotations

import argparse, fcntl, json, os, time
from pathlib import Path
from typing import Any, Mapping

import requests

from mango_mvp.channels import pilot_profile_runtime as profile
from mango_mvp.channels.subscription_llm import SubscriptionLlmDraftProvider
from mango_mvp.channels.subscription_llm_parts import AUTHORITATIVE_OUTPUT_GATE_SCHEMA_VERSION, SAFE_FALLBACK_DRAFT_TEXT, strip_internal_service_markers
from mango_mvp.pilot_context_assembly import build_pilot_context_payload

DEFAULT_SNAPSHOT = Path(__file__).resolve().parents[1] / "product_data/knowledge_base/kb_release_20260813_v6_8_owner_approved/kb_release_v3_snapshot.json"
STATE_DIR = Path.home() / ".mango_local" / "telegram_ai_agent"
BRAND_TOKEN_ENV = {"foton": "MANGO_TELEGRAM_FOTON_BOT_TOKEN", "unpk": "MANGO_TELEGRAM_UNPK_BOT_TOKEN"}
BRAND_TITLE = {"foton": "учебного центра «Фотон»", "unpk": "учебного центра УНПК МФТИ"}
HISTORY_LIMIT, MAX_TELEGRAM_TEXT = 6, 3900
FALLBACK_TEXT = "Не могу надёжно ответить на этот вопрос в чате. Пожалуйста, свяжитесь с учебным центром по контактам на официальном сайте."
ATTACHMENT_TEXT = "Пока я понимаю только текст. Напишите вопрос текстом, пожалуйста."
def _api(token: str, method: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
    try:
        response = requests.post(f"https://api.telegram.org/bot{token}/{method}", json=dict(payload), timeout=60)
    except requests.RequestException:
        raise RuntimeError(f"telegram_{method}_transport_error") from None
    if response.status_code != 200:
        raise RuntimeError(f"telegram_{method}_http_{response.status_code}")
    body = response.json()
    if not isinstance(body, Mapping) or not body.get("ok"):
        raise RuntimeError(f"telegram_{method}_rejected")
    return body

def load_offset(brand: str) -> int | None:
    path = STATE_DIR / f"{brand}_offset.json"
    if not path.exists():
        return None
    try:
        value = int(json.loads(path.read_text(encoding="utf-8"))["next_offset"])
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise RuntimeError("telegram_offset_corrupt") from exc
    if value < 0:
        raise RuntimeError("telegram_offset_corrupt")
    return value

def save_offset(brand: str, next_offset: int) -> None:
    path = STATE_DIR / f"{brand}_offset.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"next_offset": int(next_offset)}), encoding="utf-8")
    os.replace(temporary, path)

def acquire_lock(brand: str):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    stream = (STATE_DIR / f"{brand}.lock").open("a", encoding="utf-8")
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        stream.close()
        raise SystemExit(f"Telegram ИИ-агент {brand} уже запущен") from exc
    return stream

def client_text(result: Any) -> str:
    metadata = result.metadata if isinstance(result.metadata, Mapping) else {}
    gate = metadata.get("authoritative_output_gate")
    if result.error or result.route != "bot_answer_self_for_pilot":
        return ""
    valid_gate = isinstance(gate, Mapping) and gate.get("schema_version") == AUTHORITATIVE_OUTPUT_GATE_SCHEMA_VERSION
    if not valid_gate or gate.get("checked") is not True or gate.get("action") != "pass":
        return ""
    text = strip_internal_service_markers(str(result.draft_text or "")).strip()
    if not text or text == SAFE_FALLBACK_DRAFT_TEXT:
        return ""
    return text[:MAX_TELEGRAM_TEXT]

def reply_for_update(update: Mapping[str, Any], *, provider: Any, brand: str, memory: dict[str, list[str]]) -> tuple[str, str]:
    message = update.get("message")
    if not isinstance(message, Mapping):
        return "", ""
    chat = message.get("chat") if isinstance(message.get("chat"), Mapping) else {}
    sender = message.get("from") if isinstance(message.get("from"), Mapping) else {}
    chat_id = str(chat.get("id") or "")
    if str(chat.get("type") or "") != "private" or sender.get("is_bot") or not chat_id:
        return "", ""
    text = str(message.get("text") or "").strip()
    if not text:
        return chat_id, ATTACHMENT_TEXT
    if text.split()[0].split("@")[0] == "/start":
        return chat_id, f"Здравствуйте! Я — ИИ-помощница {BRAND_TITLE[brand]}. Подскажу по курсам, ценам, расписанию и записи. Что вас интересует?"
    history = memory.setdefault(f"{brand}:{chat_id}", [])
    context = build_pilot_context_payload(
        current_text=text,
        snapshot_path=DEFAULT_SNAPSHOT,
        active_brand=brand,
        recent_messages=tuple(history),
        session_id=f"tg_{brand}_{chat_id}",
        channel="telegram", channel_thread_id=chat_id, channel_user_id=chat_id,
        sends_client_replies=True, debug_impersonation_enabled=False, crm_context={},
    )
    reply = client_text(provider.build_draft(text, context=context)) or FALLBACK_TEXT
    history.extend([f"Клиент: {text}", f"Бот: {reply}"])
    del history[:-HISTORY_LIMIT]
    return chat_id, reply

def run_cycle(*, brand: str, token: str, provider: Any, memory: dict[str, list[str]]) -> None:
    offset = load_offset(brand)
    if offset is None:
        latest = _api(token, "getUpdates", {"offset": -1, "timeout": 0}).get("result") or []
        offset = max((int(item.get("update_id") or 0) + 1 for item in latest if isinstance(item, Mapping)), default=0)
        save_offset(brand, offset)
    body = _api(token, "getUpdates", {"offset": offset, "timeout": 25})
    updates = [item for item in (body.get("result") or []) if isinstance(item, Mapping)]
    for update in updates:
        chat_id, reply = reply_for_update(update, provider=provider, brand=brand, memory=memory)
        if reply:
            _api(token, "sendMessage", {"chat_id": chat_id, "text": reply})
        save_offset(brand, int(update.get("update_id") or 0) + 1)

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Публичный Telegram ИИ-агент одного бренда.")
    parser.add_argument("--brand", required=True, choices=sorted(BRAND_TOKEN_ENV))
    parser.add_argument("--once", action="store_true", help="один цикл опроса вместо бесконечного")
    args = parser.parse_args(argv)
    token = os.environ.get(BRAND_TOKEN_ENV[args.brand], "").strip()
    if not token:
        raise SystemExit(f"Не задан {BRAND_TOKEN_ENV[args.brand]}")
    _process_lock = acquire_lock(args.brand)  # ссылка удерживает flock весь срок процесса
    os.environ.setdefault(profile.ENFORCE_CANONICAL_PROFILE_ENV, "1")
    profile.ensure_canonical_pilot_profile(warn=profile.stderr_warning)
    profile.raise_for_failed_selfcheck(profile.pilot_profile_selfcheck(require=True, require_all_default_on=True))
    provider = SubscriptionLlmDraftProvider(
        model=os.environ.get("MANGO_TELEGRAM_CODEX_MODEL", "").strip() or "gpt-5.5",
        reasoning_effort=os.environ.get("MANGO_TELEGRAM_CODEX_REASONING", "").strip() or "high",
        timeout_sec=int(os.environ.get("MANGO_TELEGRAM_CODEX_TIMEOUT_SEC", "").strip() or "240"),
    )
    memory: dict[str, list[str]] = {}
    while True:
        try:
            run_cycle(brand=args.brand, token=token, provider=provider, memory=memory)
        except Exception as exc:  # noqa: BLE001 — цикл не должен падать из-за одного апдейта
            print(f"[{args.brand}] cycle_error {type(exc).__name__}", flush=True)
            if args.once or str(exc).endswith("_http_409") or str(exc) == "telegram_offset_corrupt":
                return 3
        if args.once:
            return 0
        time.sleep(2)

if __name__ == "__main__":
    raise SystemExit(main())
