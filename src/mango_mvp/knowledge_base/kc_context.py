from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Sequence

from mango_mvp.knowledge_base.fact_registry import (
    FACT_TYPE_SCHEDULE,
    PRECISE_FRESHNESS_STATUSES,
    FactSource,
    KnowledgeChunk,
    build_freshness_blocks,
    classify_fact_types,
    fact_type_from_key,
)


SCHEDULE_SAFE_TEMPLATE = (
    "У нас много групп в каждом филиале, включая онлайн, поэтому мы уточним удобное Вам время в субботу "
    "или воскресенье и постараемся подобрать занятие именно тогда. Позже дополнительно свяжемся и уточним."
)

SAFE_FALLBACK_TEMPLATE = "Спасибо за сообщение. Передам вопрос менеджеру, он вернется с проверенным ответом."
PRECISE_CLAIM_RE = re.compile(
    r"(\b\d[\d\s]*(?:руб|₽|%|процент)|\b\d{1,2}[.:]\d{2}\b|\b\d{1,2}[./]\d{1,2}(?:[./]\d{2,4})?\b)",
    re.I,
)


@dataclass(frozen=True)
class KCContext:
    selected_chunks: tuple[KnowledgeChunk, ...]
    freshness_blocks: tuple[Mapping[str, Any], ...] = ()
    safe_templates: Mapping[str, str] = field(default_factory=dict)
    precise_answers_allowed: bool = True
    manager_followup_required: bool = False
    manager_followup_deadline: str | None = None

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["selected_chunks"] = [chunk.to_json_dict() for chunk in self.selected_chunks]
        payload["freshness_blocks"] = [dict(block) for block in self.freshness_blocks]
        payload["safe_templates"] = dict(self.safe_templates)
        return payload


def build_kc_context(
    *,
    message_text: str,
    chunks: Sequence[KnowledgeChunk | Mapping[str, Any]],
    sources: Sequence[FactSource | Mapping[str, Any]] = (),
    required_fact_keys: Sequence[str] = (),
    topic_id: str | None = None,
    received_at: datetime | None = None,
    max_chunks: int = 6,
    max_chunk_chars: int = 700,
    total_char_limit: int = 3200,
) -> KCContext:
    normalized_chunks = [_coerce_chunk(chunk) for chunk in chunks]
    detected_required_keys = tuple(required_fact_keys) or _required_fact_keys_from_message(message_text, topic_id=topic_id)
    selected = limit_context_chunks(
        normalized_chunks,
        query=message_text,
        required_fact_keys=detected_required_keys,
        max_chunks=max_chunks,
        max_chunk_chars=max_chunk_chars,
        total_char_limit=total_char_limit,
    )
    freshness_blocks = build_freshness_blocks(detected_required_keys, sources)
    schedule_block = any(block.get("fact_type") == FACT_TYPE_SCHEDULE for block in freshness_blocks)
    safe_templates: dict[str, str] = {"fallback": SAFE_FALLBACK_TEMPLATE}
    manager_followup_required = False
    manager_followup_deadline = None
    if schedule_block:
        schedule_safe = build_schedule_safe_block(received_at=received_at)
        safe_templates["schedule"] = schedule_safe["template"]
        manager_followup_required = True
        manager_followup_deadline = schedule_safe["manager_followup_deadline"]
    return KCContext(
        selected_chunks=tuple(selected),
        freshness_blocks=tuple(freshness_blocks),
        safe_templates=safe_templates,
        precise_answers_allowed=not freshness_blocks,
        manager_followup_required=manager_followup_required,
        manager_followup_deadline=manager_followup_deadline,
    )


def build_kc_context_from_snapshot(
    *,
    message_text: str,
    snapshot: Mapping[str, Any],
    required_fact_keys: Sequence[str] = (),
    topic_id: str | None = None,
    received_at: datetime | None = None,
    max_chunks: int = 6,
    max_chunk_chars: int = 700,
    total_char_limit: int = 3200,
) -> KCContext:
    return build_kc_context(
        message_text=message_text,
        chunks=snapshot.get("chunks") or snapshot.get("knowledge_chunks") or (),
        sources=snapshot.get("sources") or (),
        required_fact_keys=required_fact_keys,
        topic_id=topic_id,
        received_at=received_at,
        max_chunks=max_chunks,
        max_chunk_chars=max_chunk_chars,
        total_char_limit=total_char_limit,
    )


def limit_context_chunks(
    chunks: Sequence[KnowledgeChunk | Mapping[str, Any]],
    *,
    query: str = "",
    required_fact_keys: Sequence[str] = (),
    max_chunks: int = 6,
    max_chunk_chars: int = 700,
    total_char_limit: int = 3200,
) -> list[KnowledgeChunk]:
    if max_chunks < 1:
        raise ValueError("max_chunks must be >= 1")
    if max_chunk_chars < 80:
        raise ValueError("max_chunk_chars must be >= 80")
    if total_char_limit < max_chunk_chars:
        raise ValueError("total_char_limit must be >= max_chunk_chars")
    normalized_chunks = [_coerce_chunk(chunk) for chunk in chunks]
    required_fact_types = {fact_type_from_key(key) for key in required_fact_keys if str(key).strip()}
    scored = sorted(
        normalized_chunks,
        key=lambda chunk: _chunk_score(chunk, query=query, required_fact_types=required_fact_types),
        reverse=True,
    )
    selected: list[KnowledgeChunk] = []
    used_chars = 0
    for chunk in scored:
        if _precise_claim_not_verified(chunk):
            continue
        if len(selected) >= max_chunks:
            break
        remaining = total_char_limit - used_chars
        if remaining <= 0:
            break
        text_limit = min(max_chunk_chars, remaining)
        trimmed = _trim_chunk(chunk, text_limit)
        selected.append(trimmed)
        used_chars += len(trimmed.text)
    return selected


def _precise_claim_not_verified(chunk: KnowledgeChunk) -> bool:
    return chunk.freshness_status not in PRECISE_FRESHNESS_STATUSES and bool(PRECISE_CLAIM_RE.search(chunk.text))


def build_schedule_safe_block(*, received_at: datetime | None = None) -> dict[str, Any]:
    base_time = received_at or datetime.now(timezone.utc)
    if base_time.tzinfo is None or base_time.utcoffset() is None:
        base_time = base_time.replace(tzinfo=timezone.utc)
    deadline = base_time + timedelta(hours=24)
    return {
        "template": SCHEDULE_SAFE_TEMPLATE,
        "manager_followup_required": True,
        "manager_followup_deadline": deadline.isoformat(),
        "forbidden_precise_facts": ("exact_group_time", "exact_lesson_date", "exact_group_slot"),
    }


def render_prompt_context(context: KCContext) -> str:
    lines: list[str] = []
    if context.freshness_blocks:
        lines.append("Ограничения по свежести фактов:")
        for block in context.freshness_blocks:
            lines.append(f"- {block.get('fact_key')}: {block.get('safe_instruction')}")
    if "schedule" in context.safe_templates:
        lines.append(f"Безопасный шаблон по расписанию: {context.safe_templates['schedule']}")
    if context.selected_chunks:
        lines.append("Короткие фрагменты базы знаний:")
        for chunk in context.selected_chunks:
            source_label = str(chunk.metadata.get("source_title") or chunk.source_id)
            lines.append(f"- [{chunk.title}; source={source_label}; freshness={chunk.freshness_status}] {chunk.text}")
    return "\n".join(lines)


def _required_fact_keys_from_message(message_text: str, *, topic_id: str | None = None) -> tuple[str, ...]:
    text = f"{topic_id or ''} {message_text}".lower()
    keys: list[str] = []
    if re.search(r"распис|когда|во сколько|суббот|воскрес|слот|занят", text):
        keys.append("schedule.current")
    if re.search(r"стоим|цена|сколько стоит|оплат|скид", text):
        keys.append("prices.current")
    if re.search(r"договор|справ|налог|маткап|возврат|чек|квитанц", text):
        keys.append("documents.current")
    return tuple(dict.fromkeys(keys))


def _chunk_score(chunk: KnowledgeChunk, *, query: str, required_fact_types: set[str]) -> tuple[int, int, int]:
    text = f"{chunk.title} {chunk.text}".lower()
    query_terms = _query_terms(query)
    term_score = sum(1 for term in query_terms if term in text)
    fact_score = len(set(chunk.fact_types) & required_fact_types) * 5
    schedule_bonus = 2 if FACT_TYPE_SCHEDULE in chunk.fact_types and FACT_TYPE_SCHEDULE in required_fact_types else 0
    return (fact_score + term_score + schedule_bonus, term_score, -len(chunk.text))


def _query_terms(query: str) -> set[str]:
    return {part for part in re.findall(r"[0-9A-Za-zА-Яа-яЁё]{4,}", query.lower().replace("ё", "е")) if part}


def _trim_chunk(chunk: KnowledgeChunk, max_chars: int) -> KnowledgeChunk:
    text = chunk.text
    if len(text) <= max_chars:
        return chunk
    trimmed_text = text[: max(0, max_chars - 1)].rstrip() + "…"
    return KnowledgeChunk(
        chunk_id=chunk.chunk_id,
        source_id=chunk.source_id,
        title=chunk.title,
        text=trimmed_text,
        fact_types=chunk.fact_types,
        freshness_status=chunk.freshness_status,
        metadata={**dict(chunk.metadata), "trimmed_for_prompt": True},
    )


def _coerce_chunk(chunk: KnowledgeChunk | Mapping[str, Any]) -> KnowledgeChunk:
    if isinstance(chunk, KnowledgeChunk):
        return chunk
    fact_types = tuple(chunk.get("fact_types") or classify_fact_types(f"{chunk.get('title', '')} {chunk.get('text', '')}"))
    return KnowledgeChunk(
        chunk_id=str(chunk.get("chunk_id") or "kc_chunk:manual"),
        source_id=str(chunk.get("source_id") or "source:manual"),
        title=str(chunk.get("title") or "Без названия"),
        text=str(chunk.get("text") or ""),
        fact_types=fact_types,
        freshness_status=str(chunk.get("freshness_status") or "unknown"),
        metadata=chunk.get("metadata") if isinstance(chunk.get("metadata"), Mapping) else {},
    )
