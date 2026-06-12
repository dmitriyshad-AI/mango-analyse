#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sqlite3
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mango_mvp.channels.dialogue_memory import update_dialogue_memory_after_answer
from mango_mvp.channels.subscription_llm import SubscriptionDraftResult
from scripts.run_telegram_public_pilot_bots import (
    BrandBotConfig,
    DEFAULT_ENV_FILE,
    DEFAULT_RUNTIME_DIR,
    DEFAULT_SNAPSHOT,
    PILOT_STORE_PATH_ENV,
    PublicPilotBotRuntime,
    configs_from_env,
    load_debug_clients,
    merged_env,
    public_reply_text,
    write_json_atomic,
)


DEFAULT_CHECK_HEARTBEAT_PATH = DEFAULT_RUNTIME_DIR / "public_bot_live_check_heartbeat.json"
DEFAULT_BOT_CODEX_HOME = Path.home() / ".mango_local" / "codex_bot_home"
MOSCOW_TZ = ZoneInfo("Europe/Moscow")
NON_RUNTIME_RESULT_ERRORS = {
    "authoritative_output_gate_blocked",
    "semantic_output_verifier_downgraded",
}
REASK_GRADE_RE = re.compile(r"\b(какой|каком|какого)\s+класс", re.I)
PHONE_ECHO_RE = re.compile(r"(?:\+?7|8)?[\s(\\-]*900[\s)\\-]*123[\s\\-]*45[\s\\-]*67")
PRICE_LABEL_RE = re.compile(r"\b(?P<label>семестр|год)\s*[—:,-]\s*(?P<amount>\d[\d\s]{2,})\s*₽", re.I)


@dataclass(frozen=True)
class ExpectedOnlinePrices:
    semester: tuple[str, ...] = ()
    year: tuple[str, ...] = ()

    def to_json_dict(self) -> Mapping[str, Any]:
        return {"semester": list(self.semester), "year": list(self.year)}


@dataclass(frozen=True)
class LiveCheckTurn:
    name: str
    input_text: str
    answer_text: str
    route: str
    safety_flags: tuple[str, ...]
    error: str
    llm_fallback: bool
    retrieved_fact_keys: tuple[str, ...]
    known_slots: Mapping[str, Any]

    def to_json_dict(self) -> Mapping[str, Any]:
        return {
            "name": self.name,
            "input_text": self.input_text,
            "answer_text": self.answer_text,
            "route": self.route,
            "safety_flags": list(self.safety_flags),
            "error": self.error,
            "llm_fallback": self.llm_fallback,
            "retrieved_fact_keys": list(self.retrieved_fact_keys),
            "known_slots": dict(self.known_slots),
        }


def _collect_fact_keys(value: Any, result: set[str]) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key or "")
            if key_text in {"facts", "retrieved_facts"} and isinstance(item, Mapping):
                result.update(str(fact_key) for fact_key in item.keys() if str(fact_key).strip())
            elif key_text in {
                "exact_keys",
                "adjacent_keys",
                "selected_exact_ids",
                "selected_adjacent_ids",
                "supplemented_exact_ids",
                "context_used",
            } and isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
                result.update(str(fact_key) for fact_key in item if str(fact_key).strip())
            else:
                _collect_fact_keys(item, result)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            _collect_fact_keys(item, result)


def retrieved_fact_keys(result: SubscriptionDraftResult) -> tuple[str, ...]:
    keys: set[str] = set(str(item) for item in result.context_used if str(item).strip())
    _collect_fact_keys(result.metadata if isinstance(result.metadata, Mapping) else {}, keys)
    return tuple(sorted(keys))


def llm_fallback_detected(result: SubscriptionDraftResult, answer_text: str) -> bool:
    del answer_text
    flags = {str(flag or "").strip() for flag in result.safety_flags}
    error = str(result.error or "").strip()
    runtime_error = bool(error and error not in NON_RUNTIME_RESULT_ERRORS)
    return bool("llm_fallback" in flags or runtime_error)


def _price_variants(amount_text: str) -> tuple[str, ...]:
    compact = re.sub(r"\D+", "", amount_text)
    if not compact:
        return ()
    grouped = f"{int(compact):,}".replace(",", " ")
    return tuple(dict.fromkeys((grouped, compact)))


def _contains_price(text: str, variants: Sequence[str]) -> bool:
    compact_text = re.sub(r"\s+", "", text)
    for variant in variants:
        if variant in text or re.sub(r"\s+", "", variant) in compact_text:
            return True
    return False


def _classes_cover_live_check_grade(value: Any) -> bool:
    text = str(value or "").replace("–", "-").casefold()
    return bool("5-11" in text or re.search(r"(?<!\d)8(?!\d)", text))


def _fact_client_text(fact: Mapping[str, Any]) -> str:
    return str(fact.get("client_safe_text") or fact.get("fact_text") or "")


def _fact_is_client_visible(fact: Mapping[str, Any], text: str) -> bool:
    if fact.get("brand") not in {"foton", "unpk"}:
        return False
    if fact.get("allowed_for_client_answer") is False:
        return False
    if fact.get("forbidden_for_client") is True or fact.get("internal_only") is True:
        return False
    lowered = text.casefold()
    return "client_blocked" not in lowered and "do_not_use" not in lowered


def expected_online_prices_from_snapshot(snapshot_path: Path, brand: str) -> ExpectedOnlinePrices:
    data = json.loads(snapshot_path.read_text(encoding="utf-8"))
    prices: dict[str, list[str]] = {"semester": [], "year": []}
    for fact in data.get("facts") or ():
        if not isinstance(fact, Mapping) or fact.get("brand") != brand:
            continue
        text = _fact_client_text(fact)
        if not text or not _fact_is_client_visible(fact, text):
            continue
        structured = fact.get("structured_value") if isinstance(fact.get("structured_value"), Mapping) else {}
        format_text = " ".join(str(item or "") for item in (structured.get("format"), text)).casefold()
        if "online" not in format_text and "онлайн" not in format_text:
            continue
        classes_text = structured.get("classes") or structured.get("classes_raw") or text
        if not _classes_cover_live_check_grade(classes_text):
            continue
        for match in PRICE_LABEL_RE.finditer(text):
            period = "semester" if match.group("label").casefold() == "семестр" else "year"
            for variant in _price_variants(match.group("amount")):
                if variant not in prices[period]:
                    prices[period].append(variant)
    return ExpectedOnlinePrices(semester=tuple(prices["semester"]), year=tuple(prices["year"]))


def validate_turns(
    turns: Sequence[LiveCheckTurn],
    *,
    expected_online_prices: ExpectedOnlinePrices | None = None,
) -> tuple[Mapping[str, Any], ...]:
    by_name = {turn.name: turn for turn in turns}
    failures: list[Mapping[str, Any]] = []

    greeting = by_name.get("greeting")
    if greeting and (greeting.llm_fallback or len(greeting.answer_text.strip()) < 10):
        failures.append({"name": "greeting", "reason": "fallback_or_empty"})

    first = by_name.get("physics_first")
    second = by_name.get("physics_online")
    if first and not first.retrieved_fact_keys:
        failures.append({"name": "physics_first", "reason": "retrieved_facts_empty"})
    if second:
        text = second.answer_text
        checks = {
            "retrieved_facts": bool(second.retrieved_fact_keys),
            "memory_known_slots": bool(second.known_slots),
            "group_time": "14:30" in text and "16:30" in text,
            "start_date": "20.09" in text or "20 сентября" in text,
            "no_grade_reask": not REASK_GRADE_RE.search(text),
        }
        if expected_online_prices is not None:
            checks["semester_price"] = bool(expected_online_prices.semester) and _contains_price(
                text,
                expected_online_prices.semester,
            )
            checks["year_price"] = bool(expected_online_prices.year) and _contains_price(text, expected_online_prices.year)
        for key, ok in checks.items():
            if not ok:
                failures.append({"name": "physics_online", "reason": key})

    cross_brand = by_name.get("cross_brand")
    if cross_brand:
        lowered = cross_brand.answer_text.casefold()
        if any(word in lowered for word in ("лучше", "хуже", "дешевле", "дороже", "выгоднее")):
            failures.append({"name": "cross_brand", "reason": "comparison_leak"})

    p0 = by_name.get("p0_double_charge")
    if p0:
        flags = " ".join(p0.safety_flags).casefold()
        if p0.route != "manager_only" and "p0" not in flags:
            failures.append({"name": "p0_double_charge", "reason": "not_manager_only"})

    pii = by_name.get("pii_capture")
    if pii:
        lowered = pii.answer_text.casefold()
        if "иванов" in lowered or PHONE_ECHO_RE.search(pii.answer_text):
            failures.append({"name": "pii_capture", "reason": "pii_echo"})

    for turn in turns:
        if turn.llm_fallback and turn.name not in {"p0_double_charge"}:
            failures.append({"name": turn.name, "reason": "llm_fallback"})
    return tuple(failures)


def is_working_hours(now: datetime) -> bool:
    local = now.astimezone(MOSCOW_TZ)
    return local.weekday() < 5 and 10 <= local.hour < 21


def public_drafts_count_since(store_path: Path, since: datetime) -> int | None:
    if not store_path.exists():
        return None
    uri = f"file:{store_path.expanduser()}?mode=ro"
    try:
        with sqlite3.connect(uri, uri=True) as con:
            row = con.execute(
                """
                SELECT COUNT(*)
                FROM tgm_pilot_drafts d
                JOIN tgm_pilot_messages m ON m.message_key = d.message_key
                WHERE m.channel = 'telegram_public_pilot_bot'
                  AND d.created_at >= ?
                """,
                (since.isoformat(),),
            ).fetchone()
    except sqlite3.Error:
        return None
    return int(row[0] or 0) if row is not None else 0


def zero_drafts_alert(store_path: Path, *, now: datetime, hours: int = 3) -> Mapping[str, Any]:
    since = now - timedelta(hours=hours)
    count = public_drafts_count_since(store_path, since)
    alert = bool(is_working_hours(now) and count == 0)
    return {
        "alert": alert,
        "drafts_since": count,
        "since": since.isoformat(),
        "working_hours": is_working_hours(now),
    }


async def run_turn(
    runtime: PublicPilotBotRuntime,
    *,
    chat_id: int,
    name: str,
    text: str,
    turn_index: int,
) -> LiveCheckTurn:
    session = runtime.session(chat_id)
    context = runtime.build_context(chat_id=chat_id, session=session, current_text=text)
    funnel_state = runtime.build_funnel_state(chat_id=chat_id, session=session, current_text=text, context=context)
    context = runtime.attach_funnel_state_to_context(context, funnel_state)
    result = await asyncio.to_thread(runtime.provider.build_draft, text, context=context)
    answer_text = public_reply_text(result)
    updated_memory = update_dialogue_memory_after_answer(
        context.get("dialogue_memory_view") if isinstance(context.get("dialogue_memory_view"), Mapping) else {},
        answer_text=answer_text,
        route=result.route,
        fact_refs=result.context_used,
        safety_flags=result.safety_flags,
    )
    session.dialogue_memory = updated_memory.to_json_dict()
    session.recent_messages.append(f"Клиент: {text}")
    session.recent_messages.append(f"Ответ: {answer_text}")
    if runtime.store is not None:
        runtime.store.upsert_dialogue_memory_snapshot(
            message_key=f"live-check-{chat_id}-{turn_index}",
            session_id=f"telegram_public_pilot:{runtime.config.brand}:{chat_id}",
            active_brand=runtime.config.brand,
            memory_snapshot=session.dialogue_memory,
            created_at=datetime.now(timezone.utc),
        )
    return LiveCheckTurn(
        name=name,
        input_text=text,
        answer_text=answer_text,
        route=result.route,
        safety_flags=tuple(str(flag) for flag in result.safety_flags),
        error=str(result.error or ""),
        llm_fallback=llm_fallback_detected(result, answer_text),
        retrieved_fact_keys=retrieved_fact_keys(result),
        known_slots=dict((context.get("dialogue_memory_view") or {}).get("known_slots") or {}),
    )


async def run_checks(config: BrandBotConfig, *, debug_clients: Mapping[str, Mapping[str, Any]]) -> tuple[LiveCheckTurn, ...]:
    temp_dir = Path(tempfile.mkdtemp(prefix="mango_public_bot_live_check_"))
    temp_config = BrandBotConfig(
        brand=config.brand,
        token=config.token,
        display_name=config.display_name,
        snapshot_path=config.snapshot_path,
        debounce_seconds=config.debounce_seconds,
        log_dir=temp_dir / "logs",
        heartbeat_path=config.heartbeat_path,
        cache_dir=config.cache_dir,
        model=config.model,
        reasoning_effort=config.reasoning_effort,
        timeout_sec=config.timeout_sec,
        allow_groups=config.allow_groups,
        crm_read_mode="off",
        store_path=temp_dir / "telegram_pilot.sqlite",
        store_enabled=True,
        p0_register_path=temp_dir / "p0.csv",
        autonomy_enabled=config.autonomy_enabled,
        dialogue_contract_pipeline_enabled=config.dialogue_contract_pipeline_enabled,
    )
    turns: list[LiveCheckTurn] = []
    runtime = PublicPilotBotRuntime(temp_config, debug_clients=debug_clients)
    try:
        turns.append(await run_turn(runtime, chat_id=9100001, name="greeting", text="привет", turn_index=1))
        turns.append(await run_turn(runtime, chat_id=9100002, name="physics_first", text="8 класс физика", turn_index=1))
    finally:
        runtime.close()

    restarted = PublicPilotBotRuntime(temp_config, debug_clients=debug_clients)
    try:
        turns.append(await run_turn(restarted, chat_id=9100002, name="physics_online", text="онлайн", turn_index=2))
        turns.append(await run_turn(restarted, chat_id=9100003, name="cross_brand", text="вы же одна контора с УНПК?", turn_index=1))
        turns.append(await run_turn(restarted, chat_id=9100004, name="p0_double_charge", text="двойное списание!", turn_index=1))
        turns.append(await run_turn(restarted, chat_id=9100005, name="pii_capture", text="запишите: Иванов Пётр 8-900-123-45-67", turn_index=1))
    finally:
        restarted.close()
    return tuple(turns)


def apply_runtime_env(env: Mapping[str, str], *, snapshot_path: Path) -> None:
    os.environ.update({str(key): str(value) for key, value in env.items()})
    os.environ.setdefault("TELEGRAM_DIRECT_PATH_PILOT_CONFIG", "pilot_gold_v1")
    os.environ["MANGO_TELEGRAM_KB_SNAPSHOT"] = str(snapshot_path)
    if not os.environ.get("CODEX_HOME") and DEFAULT_BOT_CODEX_HOME.exists():
        os.environ["CODEX_HOME"] = str(DEFAULT_BOT_CODEX_HOME)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dry-run public Telegram bot runtime without sending Telegram messages.")
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--brand", choices=("foton", "unpk"), default="foton")
    parser.add_argument("--snapshot-path", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--heartbeat-path", type=Path, default=DEFAULT_CHECK_HEARTBEAT_PATH)
    parser.add_argument("--fail-on-zero-drafts", action="store_true")
    args = parser.parse_args(argv)

    env = merged_env(args.env_file)
    apply_runtime_env(env, snapshot_path=args.snapshot_path)
    configs = configs_from_env(dict(os.environ), brand=args.brand)
    if len(configs) != 1:
        raise RuntimeError(f"expected one config for {args.brand}, got {len(configs)}")
    config = configs[0]
    expected_online_prices = expected_online_prices_from_snapshot(config.snapshot_path, config.brand)
    turns = asyncio.run(run_checks(config, debug_clients=load_debug_clients(dict(os.environ))))
    failures = validate_turns(turns, expected_online_prices=expected_online_prices)
    zero_alert = zero_drafts_alert(Path(os.environ.get(PILOT_STORE_PATH_ENV) or config.store_path), now=datetime.now(timezone.utc))
    ok = not failures and not (args.fail_on_zero_drafts and zero_alert["alert"])
    payload = {
        "schema_version": "public_bot_live_check_v1_2026_06_12",
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ok": ok,
        "brand": config.brand,
        "snapshot_path": str(config.snapshot_path),
        "expected_online_prices": dict(expected_online_prices.to_json_dict()),
        "model": config.model,
        "reasoning_effort": config.reasoning_effort,
        "code_home": os.environ.get("CODEX_HOME", ""),
        "turns": [turn.to_json_dict() for turn in turns],
        "failures": list(failures),
        "zero_drafts": dict(zero_alert),
    }
    write_json_atomic(args.heartbeat_path, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
