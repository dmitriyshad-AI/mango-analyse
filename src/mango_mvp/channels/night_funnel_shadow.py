from __future__ import annotations

import json
import hashlib
import re
from urllib import parse as url_parse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from mango_mvp.channels.fact_claim_audit import audit_fact_claims


NIGHT_FUNNEL_SCHEMA_VERSION = "night_funnel_shadow_v1_2026_05_28"
INBOUND_TEE_SCHEMA_VERSION = "night_inbound_tee_v1_2026_05_28"
AUTO_SEND = "AUTO_SEND"
SAFE_HOLD = "SAFE_HOLD"
MANAGER_QUEUE = "MANAGER_QUEUE"

AUTONOMOUS_ROUTES = {"bot_answer_self", "bot_answer_self_for_pilot"}
DEFAULT_CONTROL_PATH = Path(".codex_local/telegram_night_funnel/bot_control.json")
DEFAULT_STATUS_PATH = Path(".codex_local/telegram_night_funnel/bot_status.json")
DEFAULT_SHADOW_LOG_PATH = Path(".codex_local/telegram_night_funnel/shadow_log.jsonl")
DEFAULT_LEAD_STORE_PATH = Path(".codex_local/telegram_night_funnel/night_leads.jsonl")
DEFAULT_INBOUND_TEE_PATH = Path(".codex_local/telegram_night_funnel/inbound_tee.jsonl")
DEFAULT_REPLAY_CURSOR_PATH = Path(".codex_local/telegram_night_funnel/replay_cursor.json")
DEFAULT_TEE_RETENTION_DAYS = 7
PRIVATE_KNOWN_SLOT_KEYS = {"parent_name", "student_name"}


@dataclass(frozen=True)
class NightFunnelControl:
    enabled: bool = False
    mode: str = "shadow"
    shadow_only: bool = True
    manual_kill_switch: bool = False
    live_token: str = ""
    expected_live_token: str = ""
    night_limit: int = 30
    auto_trip_hold_rate: float = 0.85
    auto_trip_error_count: int = 3
    morning_followup_hour: int | None = None
    morning_followup_process_confirmed: bool = False


def load_bot_control(path: Path) -> NightFunnelControl:
    if not Path(path).exists():
        return NightFunnelControl()
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return NightFunnelControl(manual_kill_switch=True)
    if not isinstance(payload, Mapping):
        return NightFunnelControl(manual_kill_switch=True)
    return NightFunnelControl(
        enabled=_as_bool(payload.get("enabled"), default=False),
        mode=str(payload.get("mode") or "shadow").casefold().strip(),
        shadow_only=_as_bool(payload.get("shadow_only"), default=True),
        manual_kill_switch=_as_bool(payload.get("manual_kill_switch"), default=False),
        live_token=str(payload.get("live_token") or ""),
        expected_live_token=str(payload.get("expected_live_token") or ""),
        night_limit=max(0, int(payload.get("night_limit") or 30)),
        auto_trip_hold_rate=float(payload.get("auto_trip_hold_rate") or 0.85),
        auto_trip_error_count=max(1, int(payload.get("auto_trip_error_count") or 3)),
        morning_followup_hour=_optional_int(payload.get("morning_followup_hour")),
        morning_followup_process_confirmed=_as_bool(payload.get("morning_followup_process_confirmed"), default=False),
    )


def write_bot_status(
    path: Path,
    *,
    brand: str,
    control: NightFunnelControl,
    decisions: Sequence[Mapping[str, Any]],
    auto_tripped: bool,
) -> None:
    _assert_local_runtime_path(path)
    counts = Counter(str(item.get("decision") or "") for item in decisions)
    payload = {
        "schema_version": NIGHT_FUNNEL_SCHEMA_VERSION,
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "brand": brand,
        "mode": control.mode,
        "enabled": control.enabled,
        "shadow_only": control.shadow_only,
        "manual_kill_switch": control.manual_kill_switch,
        "auto_tripped": auto_tripped,
        "decisions": dict(counts),
        "total_decisions": len(decisions),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def brand_from_channel(value: str) -> str:
    text = str(value or "").casefold()
    if "kmipt.ru" in text or "unpk" in text or "унпк" in text:
        return "unpk"
    if "cdpofoton.ru" in text or "foton" in text or "фотон" in text:
        return "foton"
    return ""


def extract_utm(value: str | Mapping[str, Any]) -> dict[str, str]:
    if isinstance(value, Mapping):
        return {
            str(key): str(item)[:240]
            for key, item in value.items()
            if str(key).startswith("utm_") and str(item or "").strip()
        }
    text = str(value or "")
    parsed = re.search(r"(?:^|\s)(?:https?://\S+|\S+\?\S+)", text)
    query = ""
    if parsed:
        query = parsed.group(0).strip()
    elif "utm_" in text:
        query = text
    result: dict[str, str] = {}
    for key, values in url_parse.parse_qs(url_parse.urlparse(query).query or query).items():
        if key.startswith("utm_") and values:
            result[key] = str(values[0])[:240]
    return result


def detect_prompt_injection(text: str) -> list[str]:
    lowered = str(text or "").casefold()
    patterns = {
        "asks_to_ignore_rules": r"игнорируй\s+(?:инструкц|правил|огранич)",
        "asks_to_pretend_human": r"притворись|представься\s+менеджер|скажи\s+что\s+ты\s+человек",
        "asks_model_identity": r"\b(?:gpt|claude|codex|openai|system prompt|промпт)\b",
        "other_brand_probe": r"(?:условия|цены|рассрочк).*(?:другого\s+бренд|фотон.*унпк|унпк.*фотон)",
    }
    return [code for code, pattern in patterns.items() if re.search(pattern, lowered, re.I)]


def evaluate_night_gate(
    *,
    client_text: str,
    draft_text: str,
    route: str,
    active_brand: str,
    snapshot_path: Path,
    retrieved_facts: Mapping[str, Any] | None,
    safety_flags: Sequence[str] = (),
    control: NightFunnelControl | None = None,
    prior_decisions: Sequence[Mapping[str, Any]] = (),
) -> Mapping[str, Any]:
    control = control or NightFunnelControl()
    brand = str(active_brand or "").casefold().strip()
    fact_audit = audit_fact_claims(
        draft_text,
        client_message=client_text,
        active_brand=brand,
        retrieved_facts=retrieved_facts or {},
        snapshot_path=snapshot_path,
    )
    injection = detect_prompt_injection(client_text)
    auto_tripped = should_auto_trip(prior_decisions, control=control)
    unsafe_reasons = _unsafe_output_reasons(
        client_text=client_text,
        draft_text=draft_text,
        active_brand=brand,
        route=route,
        retrieved_facts=retrieved_facts or {},
        safety_flags=safety_flags,
        fact_audit=fact_audit,
    )
    if injection:
        decision = MANAGER_QUEUE
        reason = "anti_provocation:" + ",".join(injection)
    elif _is_real_p0(route=route, safety_flags=safety_flags, text=client_text):
        decision = MANAGER_QUEUE
        reason = "p0_or_sensitive"
    elif control.manual_kill_switch:
        decision = SAFE_HOLD
        reason = "manual_kill_switch"
    elif auto_tripped:
        decision = SAFE_HOLD
        reason = "auto_trip_or_night_limit"
    elif unsafe_reasons:
        decision = MANAGER_QUEUE if "no_retrieved_fact" in unsafe_reasons or "unsupported_number" in unsafe_reasons else SAFE_HOLD
        reason = "|".join(unsafe_reasons)
    else:
        decision = AUTO_SEND
        reason = "all_gates_passed"
    return {
        "schema_version": NIGHT_FUNNEL_SCHEMA_VERSION,
        "decision": decision,
        "reason": reason,
        "shadow_only": True,
        "fact_audit": fact_audit,
        "retrieved_fact_keys": list((retrieved_facts or {}).keys())[:20],
        "safety_flags": list(safety_flags),
        "anti_provocation": injection,
        "auto_tripped": auto_tripped,
        "unsafe_reasons": unsafe_reasons,
        "safe_hold_text": build_safe_hold_text(active_brand=brand, control=control),
    }


def build_safe_hold_text(*, active_brand: str, control: NightFunnelControl | None = None) -> str:
    control = control or NightFunnelControl()
    if control.morning_followup_process_confirmed and control.morning_followup_hour is not None:
        return (
            f"Сейчас нерабочее время. Менеджер {brand_display(active_brand)} посмотрит вопрос утром "
            f"до {int(control.morning_followup_hour):02d}:00."
        )
    return "Сейчас нерабочее время. Я зафиксировал вопрос для менеджера, он вернётся с ответом в рабочее время."


def build_shadow_record(
    *,
    brand: str,
    channel_source: str,
    utm: Mapping[str, Any],
    client_text: str,
    draft_text: str,
    gate: Mapping[str, Any],
    context: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    context = context if isinstance(context, Mapping) else {}
    lead_card = build_lead_card(
        brand=brand,
        utm=utm,
        client_text=client_text,
        draft_text=draft_text,
        decision=str(gate.get("decision") or ""),
        reason=str(gate.get("reason") or ""),
        context=context,
    )
    return {
        "schema_version": NIGHT_FUNNEL_SCHEMA_VERSION,
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "brand": brand,
        "channel_source": channel_source,
        "utm": {str(key): str(value)[:240] for key, value in dict(utm or {}).items()},
        "client_text_masked": mask_pii(client_text),
        "would_be_text_masked": mask_pii(draft_text),
        "decision": gate.get("decision"),
        "decision_reason": gate.get("reason"),
        "fact_levels": (gate.get("fact_audit") or {}).get("counts_by_level") if isinstance(gate.get("fact_audit"), Mapping) else {},
        "retrieved_fact_keys": gate.get("retrieved_fact_keys") or [],
        "shadow_only": True,
        "lead": sanitize_lead_for_shadow_log(lead_card),
    }


def append_shadow_log(path: Path, record: Mapping[str, Any]) -> Path:
    _assert_local_runtime_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(dict(record), ensure_ascii=False, sort_keys=True) + "\n")
    return path


def append_lead_card(path: Path, card: Mapping[str, Any]) -> Path:
    _assert_local_runtime_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(dict(card), ensure_ascii=False, sort_keys=True) + "\n")
    return path


def build_inbound_tee_record(
    *,
    source: str,
    brand: str,
    channel_source: str,
    utm: Mapping[str, Any],
    chat_id: int | str,
    message_id: int | str,
    message_at: datetime | str | None,
    text: str,
    known_context: Mapping[str, Any] | None = None,
    owner_runtime: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    known_context = known_context if isinstance(known_context, Mapping) else {}
    owner_runtime = owner_runtime if isinstance(owner_runtime, Mapping) else {}
    private_values = _private_values_from_context(known_context)
    return {
        "schema_version": INBOUND_TEE_SCHEMA_VERSION,
        "source": str(source or "telegram_public_pilot")[:120],
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "brand": str(brand or "").casefold().strip(),
        "channel_source": str(channel_source or "")[:500],
        "utm": {str(key): str(value)[:240] for key, value in dict(utm or {}).items()},
        "chat_id_hash": _hash_identifier(str(chat_id or "")),
        "message_id": str(message_id or "")[:120],
        "message_at": _datetime_text(message_at),
        "text": str(text or "")[:4000],
        "text_masked": mask_pii(str(text or ""), private_values=private_values),
        "known_context": _sanitize_known_context_for_tee(known_context),
        "owner_runtime": dict(owner_runtime),
    }


def append_inbound_tee_record(path: Path, record: Mapping[str, Any]) -> Path:
    _assert_local_runtime_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(dict(record), ensure_ascii=False, sort_keys=True) + "\n")
    return path


def tee_record_id(record: Mapping[str, Any]) -> str:
    raw = "|".join(
        [
            str(record.get("source") or ""),
            str(record.get("brand") or ""),
            str(record.get("chat_id_hash") or ""),
            str(record.get("message_id") or ""),
            str(record.get("message_at") or ""),
            str(record.get("text") or "")[:500],
        ]
    )
    return _hash_identifier(raw)


def load_replay_cursor(path: Path) -> Mapping[str, Any]:
    if not Path(path).exists():
        return {"byte_offset": 0, "processed_ids": []}
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {"byte_offset": 0, "processed_ids": []}
    if not isinstance(payload, Mapping):
        return {"byte_offset": 0, "processed_ids": []}
    return {
        "byte_offset": max(0, int(payload.get("byte_offset") or 0)),
        "processed_ids": [str(item) for item in payload.get("processed_ids") or [] if str(item).strip()][-10000:],
    }


def save_replay_cursor(path: Path, cursor: Mapping[str, Any]) -> None:
    _assert_local_runtime_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "night_replay_cursor_v1_2026_05_28",
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "byte_offset": max(0, int(cursor.get("byte_offset") or 0)),
        "processed_ids": [str(item) for item in cursor.get("processed_ids") or [] if str(item).strip()][-10000:],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def read_unprocessed_tee_records(
    tee_path: Path,
    cursor_path: Path,
    *,
    max_records: int | None = None,
) -> tuple[list[Mapping[str, Any]], Mapping[str, Any]]:
    cursor = dict(load_replay_cursor(cursor_path))
    processed = set(str(item) for item in cursor.get("processed_ids") or [])
    offset = max(0, int(cursor.get("byte_offset") or 0))
    path = Path(tee_path)
    if not path.exists():
        return [], cursor
    size = path.stat().st_size
    if offset > size:
        offset = 0
    records: list[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        file.seek(offset)
        while True:
            start = file.tell()
            line = file.readline()
            if not line:
                offset = file.tell()
                break
            offset = file.tell()
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, Mapping):
                continue
            record_id = tee_record_id(payload)
            if record_id in processed:
                continue
            item = {**dict(payload), "tee_record_id": record_id, "tee_byte_offset": start}
            records.append(item)
            processed.add(record_id)
            if max_records is not None and len(records) >= max(0, int(max_records)):
                break
    new_cursor = {"byte_offset": offset, "processed_ids": list(processed)[-10000:]}
    return records, new_cursor


def rotate_inbound_tee(path: Path, *, retention_days: int = DEFAULT_TEE_RETENTION_DAYS) -> Mapping[str, Any]:
    _assert_local_runtime_path(path)
    retention_days = max(1, int(retention_days or DEFAULT_TEE_RETENTION_DAYS))
    source = Path(path)
    if not source.exists():
        return {"kept": 0, "removed": 0, "path": str(source)}
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    kept_lines: list[str] = []
    removed = 0
    for line in source.read_text(encoding="utf-8").splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            removed += 1
            continue
        recorded_at = _parse_datetime(payload.get("recorded_at") if isinstance(payload, Mapping) else "")
        if recorded_at is not None and recorded_at < cutoff:
            removed += 1
            continue
        kept_lines.append(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    source.write_text(("\n".join(kept_lines) + "\n") if kept_lines else "", encoding="utf-8")
    return {"kept": len(kept_lines), "removed": removed, "path": str(source), "retention_days": retention_days}


def build_lead_card(
    *,
    brand: str,
    utm: Mapping[str, Any],
    client_text: str,
    draft_text: str,
    decision: str,
    reason: str,
    context: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    context = context if isinstance(context, Mapping) else {}
    funnel = context.get("funnel_state") if isinstance(context.get("funnel_state"), Mapping) else {}
    known = context.get("known_dialog_fields") if isinstance(context.get("known_dialog_fields"), Mapping) else {}
    return {
        "schema_version": NIGHT_FUNNEL_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "brand": brand,
        "utm": {str(key): str(value)[:240] for key, value in dict(utm or {}).items()},
        "lead_type": "night_cold",
        "client_text_masked": mask_pii(client_text),
        "would_be_text_masked": mask_pii(draft_text),
        "decision": decision,
        "reason": reason,
        "known_slots": {
            key: str(value)[:120]
            for key, value in {**dict(known), **dict(funnel.get("filled_slots") or {})}.items()
            if key in {"grade", "subject", "format", "parent_name", "student_name"}
        },
        "next_step": "morning_manager_review",
        "retention_days": 30,
        "write_crm": False,
        "write_amo": False,
        "write_tallanto": False,
    }


def sanitize_lead_for_shadow_log(card: Mapping[str, Any]) -> Mapping[str, Any]:
    result = dict(card)
    known_slots = result.get("known_slots")
    if isinstance(known_slots, Mapping):
        result["known_slots"] = {
            str(key): value
            for key, value in known_slots.items()
            if str(key) not in PRIVATE_KNOWN_SLOT_KEYS
        }
    return result


def assert_live_send_allowed(control: NightFunnelControl) -> None:
    if control.shadow_only:
        raise RuntimeError("night funnel live send blocked: SHADOW_ONLY is enabled")
    if not control.expected_live_token or control.live_token != control.expected_live_token:
        raise RuntimeError("night funnel live send blocked: live token missing or invalid")


def should_auto_trip(decisions: Sequence[Mapping[str, Any]], *, control: NightFunnelControl) -> bool:
    if control.night_limit and len(decisions) >= control.night_limit:
        return True
    if not decisions:
        return False
    window = list(decisions)[-max(1, min(len(decisions), control.night_limit or len(decisions))):]
    holdish = sum(1 for item in window if str(item.get("decision") or "") in {SAFE_HOLD, MANAGER_QUEUE})
    errors = sum(1 for item in window if str(item.get("reason") or "").startswith(("brand", "unsupported", "meta", "anti_provocation")))
    return (holdish / len(window)) >= control.auto_trip_hold_rate or errors >= control.auto_trip_error_count


def mask_pii(text: str, *, private_values: Sequence[str] = ()) -> str:
    value = str(text or "")
    value = re.sub(r"\b[\w.+-]+@[\w.-]+\.\w+\b", "[email]", value)
    value = re.sub(r"(?:\+7|8)?[\s(.-]*\d{3}[\s)./-]*\d{3}[\s.-]*\d{2}[\s.-]*\d{2}", "[phone]", value)
    for private in private_values:
        item = str(private or "").strip()
        if len(item) >= 2:
            value = re.sub(re.escape(item), "[name]", value, flags=re.I)
    return value[:2000]


def brand_display(brand: str) -> str:
    return "УНПК МФТИ" if str(brand or "").casefold() == "unpk" else "Фотона"


def _unsafe_output_reasons(
    *,
    client_text: str,
    draft_text: str,
    active_brand: str,
    route: str,
    retrieved_facts: Mapping[str, Any],
    safety_flags: Sequence[str],
    fact_audit: Mapping[str, Any],
) -> list[str]:
    reasons: list[str] = []
    if str(route or "") not in AUTONOMOUS_ROUTES:
        reasons.append("route_not_answer_self")
    if not retrieved_facts:
        reasons.append("no_retrieved_fact")
    levels = {
        str(item.get("level") or "")
        for item in (fact_audit.get("items") or [])
        if isinstance(item, Mapping) and str(item.get("level") or "").strip()
    }
    if "retrieved_match" not in levels:
        reasons.append("no_retrieved_match_claim")
    if levels.intersection({"wrong_scope", "no_match", "other_brand_match"}):
        reasons.append("fact_audit_" + "_".join(sorted(levels.intersection({"wrong_scope", "no_match", "other_brand_match"}))))
    if _has_brand_mix(draft_text, active_brand=active_brand):
        reasons.append("brand_mix")
    if _has_meta_leak(draft_text):
        reasons.append("meta_leak")
    if _has_unsupported_number(draft_text, retrieved_facts=retrieved_facts, client_text=client_text):
        reasons.append("unsupported_number")
    if _is_real_p0(route=route, safety_flags=safety_flags, text=client_text):
        reasons.append("p0_or_sensitive")
    return list(dict.fromkeys(reasons))


def _is_real_p0(*, route: str, safety_flags: Sequence[str], text: str) -> bool:
    flags = " ".join(str(flag) for flag in safety_flags).casefold()
    lowered = str(text or "").casefold()
    return (
        str(route or "") == "manager_only"
        and bool(re.search(r"p0|high_risk|complaint|legal|refund|payment_dispute", flags, re.I))
    ) or bool(
        re.search(
            r"(верните\s+деньги|мошенник|жалоб|суд|юрист|прокурат|роспотреб|чарджбек|оспорю\s+операц|незакон)",
            lowered,
            re.I,
        )
    )


def _has_brand_mix(text: str, *, active_brand: str) -> bool:
    lowered = str(text or "").casefold()
    brand = str(active_brand or "").casefold()
    if brand == "foton":
        return "унпк" in lowered or "мфти" in lowered
    if brand == "unpk":
        return "фотон" in lowered
    return True


def _has_meta_leak(text: str) -> bool:
    return bool(re.search(r"(fact_id|source_id|trace_id|json|служебн|system prompt|codex|claude|openai)", str(text or ""), re.I))


def _has_unsupported_number(text: str, *, retrieved_facts: Mapping[str, Any], client_text: str) -> bool:
    client_numbers = set(_business_numbers(client_text))
    fact_text = " ".join(str(value) for value in retrieved_facts.values())
    fact_numbers = set(_business_numbers(fact_text))
    for number in _business_numbers(text):
        if number in client_numbers:
            continue
        if number not in fact_numbers:
            return True
    return False


def _business_numbers(text: str) -> list[str]:
    result: list[str] = []
    for raw in re.findall(r"(?<![\w@/.-])\d[\d \u00a0]{0,5}(?:[.,]\d+)?(?![\w@/.-])", str(text or "")):
        normalized = raw.replace(" ", "").replace("\u00a0", "").replace(",", ".")
        if normalized in {"2026", "2027"}:
            continue
        try:
            value = float(normalized)
        except ValueError:
            continue
        if value <= 0:
            continue
        result.append(str(int(value)) if value == int(value) else str(value))
    return result


def _as_bool(value: Any, *, default: bool) -> bool:
    if value is None or value == "":
        return default
    return str(value).casefold().strip() in {"1", "true", "yes", "on", "да"}


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _hash_identifier(value: str) -> str:
    return "sha256:" + hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def _datetime_text(value: datetime | str | None) -> str:
    if isinstance(value, datetime):
        item = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        return item.astimezone(timezone.utc).isoformat(timespec="seconds")
    return str(value or "")[:120]


def _parse_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _private_values_from_context(context: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    for bucket_name in ("known_slots", "known_dialog_fields", "known_client_fields"):
        bucket = context.get(bucket_name)
        if isinstance(bucket, Mapping):
            for key in PRIVATE_KNOWN_SLOT_KEYS:
                if bucket.get(key):
                    values.append(str(bucket[key]))
    return values


def _sanitize_known_context_for_tee(context: Mapping[str, Any]) -> Mapping[str, Any]:
    result = dict(context)
    for bucket_name in ("known_slots", "known_dialog_fields", "known_client_fields"):
        bucket = result.get(bucket_name)
        if isinstance(bucket, Mapping):
            result[bucket_name] = {
                str(key): str(value)[:240]
                for key, value in bucket.items()
                if str(key) not in PRIVATE_KNOWN_SLOT_KEYS
            }
    if isinstance(result.get("recent_messages"), Sequence) and not isinstance(result.get("recent_messages"), (str, bytes)):
        private_values = _private_values_from_context(context)
        result["recent_messages"] = [
            mask_pii(str(item), private_values=private_values)
            for item in list(result.get("recent_messages") or [])[-12:]
        ]
    return result


def _assert_local_runtime_path(path: Path) -> None:
    parts = {part.casefold() for part in Path(path).parts}
    if "stable_runtime" in parts:
        raise ValueError("night funnel shadow storage must not write to stable_runtime")
