from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from mango_mvp.channels.contracts import ChannelMessage
from mango_mvp.channels.pilot_context import PilotContext, build_pilot_context
from mango_mvp.knowledge_base.fact_registry import classify_fact_types, fact_type_from_key
from mango_mvp.knowledge_base.kc_context import limit_context_chunks


NO_KNOWLEDGE_SNAPSHOT_VERSION = "knowledge_snapshot_missing"
MAX_KNOWLEDGE_SNIPPETS = 8
MAX_KNOWLEDGE_SNIPPET_CHARS = 700
MAX_KNOWLEDGE_CONTEXT_CHARS = 4500

_FRESH_STATUSES = {"fresh", "fresh_verified", "verified", "allowed_after_fact_check"}
_BLOCKING_STATUSES = {
    "metadata_only",
    "unknown",
    "stale",
    "stale_or_conflicting",
    "needs_manager_confirmation",
    "internal_only",
    "do_not_use",
    "missing",
}
_FORBIDDEN_SNIPPET_STATUSES = {"internal_only", "do_not_use"}
_TOPIC_REQUIRED_FACT_KEYS = {
    "pricing": ("prices.current",),
    "price": ("prices.current",),
    "payment_method": ("payment_methods.current",),
    "payment_status": ("payment_methods.current",),
    "discount": ("discounts.current",),
    "installment": ("installment_terms.current",),
    "schedule": ("schedule.current",),
    "refund": ("refund_policy.current",),
    "matkap": ("matkap_documents.current",),
    "tax": ("tax_deduction_procedure.current",),
    "document": ("documents.current",),
    "trial": ("trial_class.current",),
    "program": ("programs.current",),
}


@dataclass(frozen=True)
class KnowledgeSnapshotContext:
    facts_context: Mapping[str, Any]
    confirmed_facts: Mapping[str, Any]
    knowledge_snippets: tuple[str, ...]
    missing_facts: tuple[str, ...]
    context_warnings: tuple[str, ...]
    knowledge_base_version: str


def build_telegram_pilot_context(
    message: ChannelMessage | str,
    *,
    theme: Mapping[str, Any] | str | None = None,
    rop_policy: Mapping[str, Any] | None = None,
    kc_snapshot: Mapping[str, Any] | None = None,
    snapshot_path: str | Path | None = None,
    recent_messages: Sequence[str] = (),
    client_identity: Mapping[str, Any] | None = None,
    customer_summary: str = "",
    amo_context: Mapping[str, Any] | None = None,
    tallanto_context: Mapping[str, Any] | None = None,
    timeline_context: Mapping[str, Any] | None = None,
    risk_flags: Sequence[str] = (),
) -> PilotContext:
    """Build PilotContext for Telegram manager drafts from a compact KC snapshot."""

    current_message = message.text if isinstance(message, ChannelMessage) else str(message or "")
    merged_policy = merge_theme_and_rop_policy(theme=theme, rop_policy=rop_policy)
    snapshot, snapshot_warnings = _load_snapshot(kc_snapshot=kc_snapshot, snapshot_path=snapshot_path)
    snapshot_context = build_knowledge_snapshot_context(
        message_text=current_message,
        theme=theme,
        rop_policy=merged_policy,
        kc_snapshot=snapshot,
        snapshot_warnings=snapshot_warnings,
    )
    policy_for_prompt = dict(merged_policy)
    if snapshot_context.facts_context.get("required_fact_keys"):
        policy_for_prompt.setdefault("required_fact_keys", snapshot_context.facts_context["required_fact_keys"])

    return build_pilot_context(
        message,
        recent_messages=recent_messages,
        client_identity=client_identity,
        customer_summary=customer_summary,
        amo_context=amo_context,
        tallanto_context=tallanto_context,
        timeline_context=timeline_context,
        rop_policy=policy_for_prompt,
        facts_context=snapshot_context.facts_context,
        confirmed_facts=snapshot_context.confirmed_facts,
        missing_facts=snapshot_context.missing_facts,
        required_fact_keys=tuple(snapshot_context.facts_context.get("required_fact_keys", ())),
        knowledge_snippets=snapshot_context.knowledge_snippets,
        context_warnings=snapshot_context.context_warnings,
        knowledge_base_version=snapshot_context.knowledge_base_version,
        risk_flags=risk_flags,
    )


def build_telegram_pilot_context_from_snapshot(
    message: ChannelMessage | str,
    *,
    snapshot_path: str | Path | None = None,
    kc_snapshot: Mapping[str, Any] | None = None,
    topic_id: str = "",
    required_fact_keys: Sequence[str] = (),
    rop_policy: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> PilotContext:
    """Compatibility wrapper for dry-run scripts created during the KB build."""

    merged_policy = dict(rop_policy or {})
    if topic_id:
        merged_policy.setdefault("topic_id", topic_id)
    if required_fact_keys:
        merged_policy.setdefault("required_fact_keys", list(required_fact_keys))
    return build_telegram_pilot_context(
        message,
        theme={"topic_id": topic_id} if topic_id else None,
        rop_policy=merged_policy,
        kc_snapshot=kc_snapshot,
        snapshot_path=snapshot_path,
        **kwargs,
    )


def build_knowledge_snapshot_context(
    *,
    message_text: str,
    theme: Mapping[str, Any] | str | None = None,
    rop_policy: Mapping[str, Any] | None = None,
    kc_snapshot: Mapping[str, Any] | None = None,
    snapshot_warnings: Sequence[str] = (),
) -> KnowledgeSnapshotContext:
    policy = merge_theme_and_rop_policy(theme=theme, rop_policy=rop_policy)
    required_fact_keys = required_fact_keys_for_message(message_text, theme=theme, rop_policy=policy)
    if not kc_snapshot:
        return _missing_snapshot_context(required_fact_keys, snapshot_warnings=snapshot_warnings)

    version = _snapshot_version(kc_snapshot)
    topic_id = _topic_id(theme=theme, rop_policy=policy)
    required_fact_types = {fact_type_from_key(key) for key in required_fact_keys}
    facts = _records(kc_snapshot.get("facts"))
    sources = _records(kc_snapshot.get("sources"))
    chunks = _chunk_records(kc_snapshot)
    selected_chunks = limit_context_chunks(
        chunks,
        query=f"{topic_id} {message_text}",
        required_fact_keys=required_fact_keys,
        max_chunks=MAX_KNOWLEDGE_SNIPPETS,
        max_chunk_chars=MAX_KNOWLEDGE_SNIPPET_CHARS,
        total_char_limit=MAX_KNOWLEDGE_CONTEXT_CHARS,
    )
    confirmed_facts = _select_confirmed_facts(
        facts,
        required_fact_types=required_fact_types,
        topic_id=topic_id,
        query=message_text,
    )
    missing_facts, stale_or_blocked = _missing_fact_keys(
        required_fact_keys=required_fact_keys,
        required_fact_types=required_fact_types,
        facts=facts,
        sources=sources,
        selected_chunk_fact_types={fact_type for chunk in selected_chunks for fact_type in chunk.fact_types},
        confirmed_facts=confirmed_facts,
    )
    warnings = list(snapshot_warnings)
    if not selected_chunks and not confirmed_facts:
        warnings.append("knowledge_context_not_found")
    if missing_facts:
        warnings.extend(("facts_missing", "precise_answer_blocked"))
    if stale_or_blocked:
        warnings.append("facts_stale")

    knowledge_snippets = _knowledge_snippets(selected_chunks)
    if len(knowledge_snippets) < MAX_KNOWLEDGE_SNIPPETS:
        knowledge_snippets = (*knowledge_snippets, *_manager_pattern_snippets(kc_snapshot, topic_id=topic_id))[
            :MAX_KNOWLEDGE_SNIPPETS
        ]
    precise_answers_allowed = not missing_facts and not stale_or_blocked
    facts_fresh = bool(confirmed_facts) and precise_answers_allowed
    selected_source_ids = _dedupe(
        [chunk.source_id for chunk in selected_chunks]
        + [
            _clean_text(fact.get("source_id"))
            for fact in facts
            if (
                _clean_text(fact.get("fact_id") or fact.get("id")) in confirmed_facts
                and _clean_text(fact.get("source_id"))
            )
        ]
    )
    facts_context: dict[str, Any] = {
        "knowledge_base_version": version,
        "snapshot_found": True,
        "fresh": facts_fresh,
        "facts_fresh": facts_fresh,
        "missing": bool(missing_facts),
        "facts_missing": bool(missing_facts),
        "stale": bool(stale_or_blocked),
        "facts_stale": bool(stale_or_blocked),
        "precise_answers_allowed": precise_answers_allowed,
        "required_fact_keys": list(required_fact_keys),
        "confirmed_fact_ids": list(confirmed_facts.keys()),
        "selected_chunk_ids": [chunk.chunk_id for chunk in selected_chunks],
        "source_ids": selected_source_ids[:12],
    }
    if confirmed_facts:
        facts_context["confirmed_facts"] = dict(confirmed_facts)

    return KnowledgeSnapshotContext(
        facts_context=facts_context,
        confirmed_facts=confirmed_facts,
        knowledge_snippets=knowledge_snippets,
        missing_facts=missing_facts,
        context_warnings=tuple(_dedupe(warnings)),
        knowledge_base_version=version,
    )


def merge_theme_and_rop_policy(
    *,
    theme: Mapping[str, Any] | str | None = None,
    rop_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    if isinstance(theme, Mapping):
        merged.update(theme)
        if theme.get("theme_id") and not theme.get("topic_id"):
            merged["topic_id"] = theme["theme_id"]
        if theme.get("theme_name") and not theme.get("topic_name"):
            merged["topic_name"] = theme["theme_name"]
    elif theme:
        merged["topic_id"] = str(theme)
    merged.update(dict(rop_policy or {}))
    if merged.get("theme_id") and not merged.get("topic_id"):
        merged["topic_id"] = merged["theme_id"]
    if merged.get("theme_name") and not merged.get("topic_name"):
        merged["topic_name"] = merged["theme_name"]
    return merged


def required_fact_keys_for_message(
    message_text: str,
    *,
    theme: Mapping[str, Any] | str | None = None,
    rop_policy: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    keys: list[str] = []
    for container in (theme if isinstance(theme, Mapping) else None, rop_policy):
        if not isinstance(container, Mapping):
            continue
        for field_name in ("required_fact_keys", "required_facts", "fact_keys"):
            keys.extend(_text_list(container.get(field_name)))
        for fact_type in _text_list(container.get("required_fact_types")):
            keys.append(f"{fact_type}.current")

    topic_text = " ".join(
        _clean_text(value)
        for value in (
            _topic_id(theme=theme, rop_policy=rop_policy),
            rop_policy.get("topic_name") if isinstance(rop_policy, Mapping) else "",
            rop_policy.get("theme_name") if isinstance(rop_policy, Mapping) else "",
        )
        if _clean_text(value)
    ).casefold()
    for marker, marker_keys in _TOPIC_REQUIRED_FACT_KEYS.items():
        if marker in topic_text:
            keys.extend(marker_keys)

    text = f"{topic_text} {message_text}".casefold()
    if re.search(r"стоим|цен[аы]|сколько стоит|прайс|руб", text):
        keys.append("prices.current")
    if re.search(r"распис|когда|во сколько|суббот|воскрес|слот|занят", text):
        keys.append("schedule.current")
    if re.search(r"скид|льгот|промокод|акци", text):
        keys.append("discounts.current")
    if re.search(r"рассроч", text):
        keys.append("installment_terms.current")
    if re.search(r"оплат|сбп|реквизит|карт|ссылк[ау] на оплат", text):
        keys.append("payment_methods.current")
    if re.search(r"договор|справ|налог|маткап|возврат|чек|квитанц", text):
        keys.append("documents.current")
    if re.search(r"программ|предмет|летн|пробн|чему учат|содержание", text):
        keys.append("programs.current")
    return tuple(_dedupe(_stable_fact_key(key) for key in keys if _clean_text(key)))


def _missing_snapshot_context(
    required_fact_keys: Sequence[str],
    *,
    snapshot_warnings: Sequence[str],
) -> KnowledgeSnapshotContext:
    missing = tuple(required_fact_keys) or ("knowledge_snapshot",)
    warnings = _dedupe([*snapshot_warnings, "knowledge_snapshot_missing", "facts_missing", "precise_answer_blocked"])
    return KnowledgeSnapshotContext(
        facts_context={
            "knowledge_base_version": NO_KNOWLEDGE_SNAPSHOT_VERSION,
            "snapshot_found": False,
            "fresh": False,
            "facts_fresh": False,
            "missing": True,
            "facts_missing": True,
            "stale": False,
            "facts_stale": False,
            "precise_answers_allowed": False,
            "required_fact_keys": list(required_fact_keys),
        },
        confirmed_facts={},
        knowledge_snippets=(),
        missing_facts=missing,
        context_warnings=tuple(warnings),
        knowledge_base_version=NO_KNOWLEDGE_SNAPSHOT_VERSION,
    )


def _load_snapshot(
    *,
    kc_snapshot: Mapping[str, Any] | None,
    snapshot_path: str | Path | None,
) -> tuple[Mapping[str, Any] | None, tuple[str, ...]]:
    if kc_snapshot is not None:
        return dict(kc_snapshot), ()
    if snapshot_path is None:
        return None, ()
    try:
        with Path(snapshot_path).open("r", encoding="utf-8") as file:
            loaded = json.load(file)
    except (OSError, json.JSONDecodeError):
        return None, ("knowledge_snapshot_unreadable",)
    if not isinstance(loaded, Mapping):
        return None, ("knowledge_snapshot_invalid",)
    return dict(loaded), ()


def _snapshot_version(snapshot: Mapping[str, Any]) -> str:
    metadata = snapshot.get("metadata") if isinstance(snapshot.get("metadata"), Mapping) else {}
    for value in (
        snapshot.get("run_id"),
        snapshot.get("snapshot_id"),
        snapshot.get("version"),
        metadata.get("run_id"),
        metadata.get("version"),
        snapshot.get("schema_version"),
    ):
        cleaned = _clean_text(value)
        if cleaned:
            return cleaned
    return "kc_knowledge_snapshot_unknown"


def _chunk_records(snapshot: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    chunks = _records(snapshot.get("chunks") or snapshot.get("knowledge_chunks"))
    result: list[Mapping[str, Any]] = []
    for chunk in chunks:
        if _truthy(chunk.get("forbidden_for_client")):
            continue
        status = _stable_status(chunk.get("freshness_status"))
        if status in _FORBIDDEN_SNIPPET_STATUSES:
            continue
        text = _clean_text(
            chunk.get("text")
            or chunk.get("client_safe_text")
            or chunk.get("manager_text")
            or chunk.get("short_fact"),
            max_chars=1600,
        )
        if not text:
            continue
        if status in _BLOCKING_STATUSES and _has_precise_claim(text):
            continue
        fact_types = _fact_types(chunk)
        result.append(
            {
                **dict(chunk),
                "text": text,
                "title": _clean_text(chunk.get("title") or chunk.get("source_title") or "База знаний"),
                "fact_types": list(fact_types),
                "freshness_status": status,
            }
        )
    return result


def _select_confirmed_facts(
    facts: Sequence[Mapping[str, Any]],
    *,
    required_fact_types: set[str],
    topic_id: str,
    query: str,
) -> dict[str, str]:
    confirmed: dict[str, str] = {}
    for fact in facts:
        if not _usable_for_precise_answer(fact):
            continue
        if not _record_matches_context(fact, required_fact_types=required_fact_types, topic_id=topic_id, query=query):
            continue
        text = _fact_text(fact)
        if not text:
            continue
        fact_id = _clean_text(fact.get("fact_id") or fact.get("id") or f"fact:{len(confirmed) + 1}", max_chars=120)
        confirmed[fact_id] = text
        if len(confirmed) >= 10:
            break
    return confirmed


def _missing_fact_keys(
    *,
    required_fact_keys: Sequence[str],
    required_fact_types: set[str],
    facts: Sequence[Mapping[str, Any]],
    sources: Sequence[Mapping[str, Any]],
    selected_chunk_fact_types: set[str],
    confirmed_facts: Mapping[str, Any],
) -> tuple[tuple[str, ...], bool]:
    if not required_fact_keys:
        return (), False
    confirmed_fact_types = {
        fact_type
        for fact in facts
        if _clean_text(fact.get("fact_id") or fact.get("id")) in confirmed_facts
        for fact_type in _fact_types(fact)
    }
    missing: list[str] = []
    stale_or_blocked = False
    for fact_key in required_fact_keys:
        fact_type = fact_type_from_key(fact_key)
        if fact_type in confirmed_fact_types:
            continue
        candidate_statuses = [
            _stable_status(record.get("freshness_status"))
            for record in (*facts, *sources)
            if fact_type in _fact_types(record)
        ]
        if fact_type in selected_chunk_fact_types and not candidate_statuses:
            candidate_statuses.append("unknown")
        if any(status in _BLOCKING_STATUSES for status in candidate_statuses):
            stale_or_blocked = True
        missing.append(fact_key)
    return tuple(_dedupe(missing)), stale_or_blocked


def _knowledge_snippets(selected_chunks: Sequence[Any]) -> tuple[str, ...]:
    snippets: list[str] = []
    for chunk in selected_chunks:
        title = _clean_text(getattr(chunk, "title", ""), max_chars=120) or "База знаний"
        source_id = _clean_text(getattr(chunk, "source_id", ""), max_chars=120)
        status = _clean_text(getattr(chunk, "freshness_status", ""), max_chars=80) or "unknown"
        text = _clean_text(getattr(chunk, "text", ""), max_chars=MAX_KNOWLEDGE_SNIPPET_CHARS)
        if not text:
            continue
        prefix = f"[{title}; source={source_id}; freshness={status}] "
        snippets.append(_clip_text(f"{prefix}{text}", MAX_KNOWLEDGE_SNIPPET_CHARS))
    return tuple(_dedupe(snippets))


def _manager_pattern_snippets(snapshot: Mapping[str, Any], *, topic_id: str) -> tuple[str, ...]:
    snippets: list[str] = []
    for pattern in _records(snapshot.get("manager_answer_patterns")):
        if topic_id and topic_id not in _text_list(pattern.get("related_theme_ids") or pattern.get("theme_ids")):
            continue
        text = _clean_text(
            pattern.get("client_safe_text")
            or pattern.get("safe_pattern")
            or pattern.get("pattern_summary")
            or pattern.get("manager_safe_text"),
            max_chars=560,
        )
        if not text:
            continue
        snippets.append(_clip_text(f"[Прием менеджера, не факт] {text}", MAX_KNOWLEDGE_SNIPPET_CHARS))
        if len(snippets) >= 2:
            break
    return tuple(_dedupe(snippets))


def _record_matches_context(
    record: Mapping[str, Any],
    *,
    required_fact_types: set[str],
    topic_id: str,
    query: str,
) -> bool:
    fact_types = set(_fact_types(record))
    if required_fact_types and fact_types & required_fact_types:
        return True
    related_theme_ids = set(_text_list(record.get("related_theme_ids") or record.get("theme_ids") or record.get("topics")))
    if topic_id and topic_id in related_theme_ids:
        return True
    if required_fact_types or topic_id:
        return False
    text = f"{record.get('title', '')} {_fact_text(record)}".casefold()
    return any(term in text for term in _query_terms(query))


def _usable_for_precise_answer(record: Mapping[str, Any]) -> bool:
    return (
        _stable_status(record.get("freshness_status")) in _FRESH_STATUSES
        and _truthy(record.get("usable_for_precise_answer"))
        and not _truthy(record.get("requires_manager_confirmation"))
        and not _truthy(record.get("forbidden_for_client"))
    )


def _fact_text(record: Mapping[str, Any]) -> str:
    return _clean_text(
        record.get("client_safe_text")
        or record.get("short_fact")
        or record.get("manager_text")
        or record.get("text"),
        max_chars=300,
    )


def _fact_types(record: Mapping[str, Any]) -> tuple[str, ...]:
    values = _text_list(record.get("fact_types"))
    if not values:
        values = _text_list(record.get("fact_type"))
    if values:
        return tuple(_dedupe(fact_type_from_key(value) for value in values))
    return classify_fact_types(f"{record.get('title', '')} {record.get('text', '')} {_fact_text(record)}")


def _topic_id(
    *,
    theme: Mapping[str, Any] | str | None,
    rop_policy: Mapping[str, Any] | None,
) -> str:
    if isinstance(rop_policy, Mapping):
        value = rop_policy.get("topic_id") or rop_policy.get("theme_id")
        if value:
            return _clean_text(value)
    if isinstance(theme, Mapping):
        return _clean_text(theme.get("topic_id") or theme.get("theme_id"))
    return _clean_text(theme)


def _records(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        if isinstance(value.get("items"), Sequence) and not isinstance(value.get("items"), (str, bytes, bytearray)):
            return _records(value.get("items"))
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [dict(item) for item in value if isinstance(item, Mapping)]
    return []


def _text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        if "," in value:
            return [_clean_text(part) for part in value.split(",") if _clean_text(part)]
        return [_clean_text(value)] if _clean_text(value) else []
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [_clean_text(item) for item in value if _clean_text(item)]
    return [_clean_text(value)] if _clean_text(value) else []


def _query_terms(query: str) -> set[str]:
    return {part for part in re.findall(r"[0-9A-Za-zА-Яа-яЁё]{4,}", query.casefold().replace("ё", "е"))}


def _has_precise_claim(text: str) -> bool:
    pattern = (
        r"\b\d[\d\s]*(?:руб|₽|%|процент|январ|феврал|март|апрел|ма[йя]|июн|июл|август|"
        r"сентябр|октябр|ноябр|декабр)"
    )
    return bool(re.search(pattern, text.casefold()))


def _stable_fact_key(value: Any) -> str:
    text = _clean_text(value)
    if "." in text:
        return text
    return f"{text}.current"


def _stable_status(value: Any) -> str:
    return _clean_text(value).casefold() or "unknown"


def _clean_text(value: Any, max_chars: int = 240) -> str:
    text = " ".join(str(value or "").strip().split())
    return text[:max_chars]


def _clip_text(value: str, max_chars: int) -> str:
    text = _clean_text(value, max_chars=max_chars)
    if len(text) < max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "…"


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().casefold() in {"1", "true", "yes", "y", "да", "истина", "есть"}


def _dedupe(values: Sequence[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = _clean_text(value)
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
    return result
