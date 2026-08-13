#!/usr/bin/env python3
"""Тонкий публичный Telegram-транспорт к общему модельному ядру Mango."""
from __future__ import annotations

import argparse, fcntl, json, os, time
from pathlib import Path
from typing import Any, Mapping, Sequence

import requests

from mango_mvp.channels.dialogue_memory import update_dialogue_memory_after_answer
from mango_mvp.channels import pilot_profile_runtime as profile
from mango_mvp.channels.subscription_llm import SubscriptionLlmDraftProvider
from mango_mvp.channels.subscription_llm_parts import AUTHORITATIVE_OUTPUT_GATE_SCHEMA_VERSION, SAFE_FALLBACK_DRAFT_TEXT, SemanticReading, semantic_frame_from_metadata, strip_internal_service_markers
from mango_mvp.integrations.draft_loop import DraftLoopConfigError, DraftLoopKey, DraftLoopState
from mango_mvp.pilot_context_assembly import build_pilot_context_payload

DEFAULT_SNAPSHOT = Path(__file__).resolve().parents[1] / "product_data/knowledge_base/kb_release_20260813_v6_8_owner_approved/kb_release_v3_snapshot.json"
STATE_DIR = Path.home() / ".mango_local" / "telegram_ai_agent"
BRAND_TOKEN_ENV = {"foton": "MANGO_TELEGRAM_FOTON_BOT_TOKEN", "unpk": "MANGO_TELEGRAM_UNPK_BOT_TOKEN"}
BRAND_TITLE = {"foton": "учебного центра «Фотон»", "unpk": "учебного центра УНПК МФТИ"}
MAX_TELEGRAM_TEXT = 3900
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

def _load_legacy_offset(brand: str) -> int | None:
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

def acquire_lock(brand: str):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    stream = (STATE_DIR / f"{brand}.lock").open("a", encoding="utf-8")
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        stream.close()
        raise SystemExit(f"Telegram ИИ-агент {brand} уже запущен") from exc
    return stream

def dialogue_state(brand: str) -> DraftLoopState:
    state = DraftLoopState(STATE_DIR / f"{brand}_dialogue_state.json")
    memories = state.payload.get("dialogue_memory")
    if not isinstance(memories, Mapping):
        raise DraftLoopConfigError("Invalid Telegram dialogue memory state.")
    for memory in memories.values():
        turns = memory.get("turns", ()) if isinstance(memory, Mapping) else None
        if not isinstance(turns, Sequence) or isinstance(turns, (str, bytes, bytearray)) or any(
            not isinstance(turn, Mapping) for turn in turns
        ):
            raise DraftLoopConfigError("Invalid Telegram dialogue memory state.")
    if "next_offset" not in state.payload:
        legacy_offset = _load_legacy_offset(brand)
        if legacy_offset is not None:
            state.payload["next_offset"] = legacy_offset
            state.save()
    return state

def _state_offset(state: DraftLoopState) -> int | None:
    raw = state.payload.get("next_offset")
    if raw is None:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("telegram_offset_corrupt") from exc
    if value < 0:
        raise RuntimeError("telegram_offset_corrupt")
    return value

def _save_checkpoint(
    state: DraftLoopState, *, next_offset: int, key: DraftLoopKey | None = None,
    memory: Mapping[str, Any] | None = None, marker: tuple[str, str, str] | None = None,
) -> None:
    previous_offset = state.payload.get("next_offset")
    if key is not None and memory is not None:
        state.set_dialogue_memory(key, memory)
    if marker is not None:
        state.mark_processed_key(*marker)
    state.payload["next_offset"] = max(0, int(next_offset))
    try:
        state.save()
    except Exception:
        if previous_offset is None:
            state.payload.pop("next_offset", None)
        else:
            state.payload["next_offset"] = previous_offset
        raise

def _recent_messages(memory: Mapping[str, Any]) -> tuple[str, ...]:
    turns = tuple(
        f"{'Ответ' if turn.get('role') == 'bot' else 'Клиент'}: {turn.get('text')}"
        for turn in memory.get("turns", ())
        if isinstance(turn, Mapping) and turn.get("text")
    )
    summary = str(memory.get("conversation_summary_short") or "").strip()
    return (f"Ранее в диалоге: {summary}", *turns[-7:]) if summary else turns[-8:]

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

def reply_for_update(update: Mapping[str, Any], *, provider: Any, brand: str, state: DraftLoopState) -> tuple[str, str, DraftLoopKey | None, Mapping[str, Any] | None]:
    message = update.get("message")
    if not isinstance(message, Mapping):
        return "", "", None, None
    chat = message.get("chat") if isinstance(message.get("chat"), Mapping) else {}
    sender = message.get("from") if isinstance(message.get("from"), Mapping) else {}
    chat_id = str(chat.get("id") or "")
    if str(chat.get("type") or "") != "private" or sender.get("is_bot") or not chat_id:
        return "", "", None, None
    key = DraftLoopKey(brand, chat_id)
    text = str(message.get("text") or "").strip()
    if not text:
        return chat_id, ATTACHMENT_TEXT, key, None
    if text.split()[0].split("@")[0] == "/start":
        return chat_id, f"Здравствуйте! Я — ИИ-помощница {BRAND_TITLE[brand]}. Подскажу по курсам, ценам, расписанию и записи. Что вас интересует?", key, None
    previous_memory = state.dialogue_memory_for(key)
    recent = _recent_messages(previous_memory)
    context = build_pilot_context_payload(
        current_text=text,
        snapshot_path=DEFAULT_SNAPSHOT,
        active_brand=brand,
        recent_messages=recent, dialogue_memory=previous_memory,
        session_id=f"tg_{brand}_{chat_id}",
        channel="telegram", channel_thread_id=chat_id, channel_user_id=chat_id,
        sends_client_replies=True, debug_impersonation_enabled=False, crm_context={},
        current_message_id=str(message.get("message_id") or update.get("update_id") or ""),
    )
    result = provider.build_draft(text, context=context)
    safe_reply = client_text(result)
    reply = safe_reply or FALLBACK_TEXT
    metadata = result.metadata if isinstance(result.metadata, Mapping) else {}
    semantic = SemanticReading.from_result(result)
    direct = metadata.get("direct_path")
    summary = str(metadata.get("dialog_summary_candidate") or (direct.get("dialog_summary_candidate") if isinstance(direct, Mapping) else "") or "")
    updated = update_dialogue_memory_after_answer(
        context["dialogue_memory_state"],
        answer_text=reply, route=result.route, fact_refs=result.context_used, safety_flags=result.safety_flags,
        semantic_reading=semantic.to_memory_dict() if semantic else None,
        semantic_frame=semantic_frame_from_metadata(metadata) or None,
        dialog_summary=summary, context=context, answer_is_substantive=bool(safe_reply),
    )
    return chat_id, reply, key, updated.to_json_dict()

def run_cycle(*, brand: str, token: str, provider: Any, state: DraftLoopState) -> None:
    offset = _state_offset(state)
    if offset is None:
        recovered = [
            int(message_id) + 1 for profile_id, _chat_id, message_id in state.processed_keys()
            if profile_id == brand and message_id.isdigit()
        ]
        if recovered:
            offset = max(recovered)
        else:
            latest = _api(token, "getUpdates", {"offset": -1, "timeout": 0}).get("result") or []
            offset = max((int(item.get("update_id") or 0) + 1 for item in latest if isinstance(item, Mapping)), default=0)
        _save_checkpoint(state, next_offset=offset)
    body = _api(token, "getUpdates", {"offset": offset, "timeout": 25})
    updates = [item for item in (body.get("result") or []) if isinstance(item, Mapping)]
    processed = state.processed_keys()
    for update in updates:
        update_id = str(update.get("update_id") or "")
        message = update.get("message") if isinstance(update.get("message"), Mapping) else {}
        chat = message.get("chat") if isinstance(message.get("chat"), Mapping) else {}
        marker = (brand, str(chat.get("id") or ""), update_id)
        if all(marker) and marker in processed:
            _save_checkpoint(state, next_offset=int(update_id) + 1)
            continue
        chat_id, reply, key, updated_memory = reply_for_update(update, provider=provider, brand=brand, state=state)
        if reply:
            try:
                _api(token, "sendMessage", {"chat_id": chat_id, "text": reply})
            except RuntimeError as exc:
                if str(exc) != "telegram_sendMessage_http_403":
                    print(f"[{brand}] send_error {exc}", flush=True)
                    raise
                print(f"[{brand}] send_skipped {exc}", flush=True)
                _save_checkpoint(state, next_offset=int(update_id) + 1, marker=marker)
                processed.add(marker)
                continue
            if key is not None and update_id:
                _save_checkpoint(
                    state, next_offset=int(update_id) + 1, key=key,
                    memory=updated_memory, marker=marker,
                )
                processed.add(marker)
                continue
        _save_checkpoint(state, next_offset=int(update.get("update_id") or 0) + 1)

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
        codex_isolated=True,
    )
    state = dialogue_state(args.brand)
    while True:
        try:
            run_cycle(brand=args.brand, token=token, provider=provider, state=state)
        except Exception as exc:  # noqa: BLE001 — цикл не должен падать из-за одного апдейта
            print(f"[{args.brand}] cycle_error {type(exc).__name__}", flush=True)
            if args.once or str(exc).endswith("_http_409") or str(exc) == "telegram_offset_corrupt":
                return 3
        if args.once:
            return 0
        time.sleep(2)

if __name__ == "__main__":
    raise SystemExit(main())
