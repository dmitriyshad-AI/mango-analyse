from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from mango_mvp.customer_timeline.channel_preview_from_pack import redact_text
from mango_mvp.customer_timeline.contracts import BotContextChunk, now_utc
from mango_mvp.customer_timeline.ids import stable_chunk_id, stable_digest
from mango_mvp.customer_timeline.next_step_resolver import (
    CUSTOMER_TIMELINE_NEXT_STEP_SCHEMA_VERSION,
    PERSON_NAME_RE as NEXT_STEP_PERSON_NAME_RE,
    ROLE_PERSON_RE as NEXT_STEP_ROLE_PERSON_RE,
    NextStepResolution,
    resolve_customer_next_step,
)
from mango_mvp.customer_timeline.store import CustomerTimelineSQLiteStore, scrub_timeline_persisted_json
from mango_mvp.insights.sanitizers import COMMON_SINGLE_NAME_RE as INSIGHTS_COMMON_SINGLE_NAME_RE


BOT_SAFE_SUMMARY_SCHEMA_VERSION = "customer_timeline_bot_safe_summary_v1"
BOT_SAFE_SUMMARY_CHUNK_TYPE = "bot_safe_summary"
BOT_SAFE_SUMMARY_SOURCE_SYSTEM = "customer_timeline_bot_safe_summary"
BOT_SAFE_SUMMARY_ACTOR = "customer_timeline_bot_safe_summary_builder"

KNOWN_BRANDS = {"foton", "unpk"}
GENERIC_TITLE_PATTERNS = (
    re.compile(r"^сделка\b", re.IGNORECASE),
    re.compile(r"^заявка\b", re.IGNORECASE),
    re.compile(r"\bс\s+сайта\b", re.IGNORECASE),
)
WHITESPACE_RE = re.compile(r"\s+")
LONG_DIGIT_TOKEN_RE = re.compile(r"\b\d{8,}\b")
MAIL_REPLY_PREFIX_RE = re.compile(r"^(?:(?:re|fw|fwd):\s*)+", re.IGNORECASE)
FILE_NAME_FRAGMENT_RE = re.compile(r"(?:^image[-_ ]|\.(?:pdf|jpe?g|png|gif|docx?|xlsx?|zip)\b)", re.IGNORECASE)
FOTON_MARKERS = ("фотон", "foton", "cdpofoton", "цдпфотон", "цдп фотон")
UNPK_MARKERS = ("унпк", "мфти", "физтех", "unpk")
UNSAFE_INTEREST_MARKERS = (
    "акци",
    "договор",
    "документ",
    "задолж",
    "квитанц",
    "оплат",
    "платеж",
    "платёж",
    "пропуск",
    "receipt",
    "скидк",
    "spam",
    "счет",
    "счёт",
    "чек",
)
UNSAFE_NEXT_STEP_MARKERS = UNSAFE_INTEREST_MARKERS + (
    "диадок",
    "закрывающ",
    "льгот",
)
INTEREST_NAME_PLACEHOLDER = "<name_masked>"
INTEREST_ROLE_WORD_RE = re.compile(
    r"\b(?:менеджер|куратор|администратор|оператор|клиент(?:ка)?|родител[ьи]|мама|папа|"
    r"ученик|ученица|реб[её]нок|студент(?:ка)?)\b",
    re.IGNORECASE,
)
SAFE_INTEREST_PHRASE_PATTERNS = (
    re.compile(r"\b(?:Летн|Зимн)[а-яё]+\s+(?:Выездн|Очн)[а-яё]+\s+школ[а-яё]*\b", re.IGNORECASE),
    re.compile(r"\bЛВШ\b", re.IGNORECASE),
    re.compile(r"\bЛШ\b", re.IGNORECASE),
    re.compile(r"\bИнтенсив\s+(?:Мат|Физ|Инф|Рус)\b", re.IGNORECASE),
    re.compile(r"\bАльфа[-\s]+Банк\b", re.IGNORECASE),
    re.compile(r"\bФотон\b", re.IGNORECASE),
    re.compile(r"\bУНПК\s+МФТИ\b", re.IGNORECASE),
    re.compile(r"\bУНПК\b", re.IGNORECASE),
    re.compile(r"\bМФТИ\b", re.IGNORECASE),
    re.compile(r"\bЕГЭ\b", re.IGNORECASE),
    re.compile(r"\bОГЭ\b", re.IGNORECASE),
    re.compile(r"\bМ9\b", re.IGNORECASE),
    re.compile(r"\bМ11\b", re.IGNORECASE),
)
BOT_SAFE_SOURCE_CHUNK_TYPES = {
    "mango_call_summary",
    "customer_history_summary",
    "channel_message",
    "email_message",
}
CLASS_RE = re.compile(
    r"(?<!\d)(?P<class>1[01]|[5-9])\s*"
    r"(?:[-–—]?\s*(?:й|ый|ой|го|ого|му|ому|м|ом|е|х|ых))?\s*"
    r"(?:класс\w*|кл\.?)\b",
    re.IGNORECASE,
)
CLASS_RANGE_RE = re.compile(
    r"(?<!\d)(?P<start>[5-9])\s*[-–—]\s*(?P<end>1[01]|[5-9])\s*(?:класс(?:[аеов])?|кл\.?)\b",
    re.IGNORECASE,
)
COORDINATED_CLASS_RE = re.compile(
    r"(?<!\d)(?P<first>1[01]|[5-9])\s*(?:,|/|\+|и)\s*(?P<second>1[01]|[5-9])\s*"
    r"(?:класс\w*|кл\.?)\b",
    re.IGNORECASE,
)
DIRECT_DIGIT_CLASS_RE = re.compile(
    r"(?<!\d)(?P<class>1[01]|[1-9])\s*"
    r"(?:[-–—]?\s*(?:й|ый|ой|го|ого|му|ому|м|ом|е|х|ых))?\s*"
    r"(?:класс\w*|кл\.?)\b",
    re.IGNORECASE,
)
DIRECT_DIGIT_CLASS_RANGE_RE = re.compile(
    r"(?<!\d)(?P<start>1[01]|[1-9])\s*[-–—]\s*(?P<end>1[01]|[1-9])\s*(?:класс(?:[аеов])?|кл\.?)\b",
    re.IGNORECASE,
)
DIRECT_DIGIT_COORDINATED_CLASS_RE = re.compile(
    r"(?<!\d)(?P<first>1[01]|[1-9])\s*(?:,|/|\+|и)\s*(?P<second>1[01]|[1-9])\s*"
    r"(?:класс\w*|кл\.?)\b",
    re.IGNORECASE,
)
M_CLASS_RE = re.compile(r"\bм\s*(?P<class>9|11)\b", re.IGNORECASE)
WORD_CLASS_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("5", re.compile(r"\bпят\w+\s+класс\w*", re.IGNORECASE)),
    ("6", re.compile(r"\bшест\w+\s+класс\w*", re.IGNORECASE)),
    ("7", re.compile(r"\bседьм\w+\s+класс\w*", re.IGNORECASE)),
    ("8", re.compile(r"\bвосьм\w+\s+класс\w*", re.IGNORECASE)),
    ("9", re.compile(r"\bдевят\w+\s+класс\w*", re.IGNORECASE)),
    ("10", re.compile(r"\bдесят\w+\s+класс\w*", re.IGNORECASE)),
    ("11", re.compile(r"\bодиннадцат\w+\s+класс\w*", re.IGNORECASE)),
)
SUBJECT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("математика", re.compile(r"\b(?:матем\w*|мат(?:\.|\b))", re.IGNORECASE)),
    ("физика", re.compile(r"\b(?:физик\w*|физ(?:\.|\b))", re.IGNORECASE)),
    ("информатика", re.compile(r"\b(?:информат\w*|инф(?:\.|\b))", re.IGNORECASE)),
    ("русский язык", re.compile(r"\bрусск\w+\s+язык\w*", re.IGNORECASE)),
    ("английский язык", re.compile(r"\bанглийск\w+\s+язык\w*", re.IGNORECASE)),
    ("химия", re.compile(r"\bхими\w*", re.IGNORECASE)),
    ("биология", re.compile(r"\bбиологи\w*", re.IGNORECASE)),
)
FORMAT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("онлайн", re.compile(r"\b(?:онлайн|дистанц\w*|мтс\s*линк|mts\s*link)\b", re.IGNORECASE)),
    (
        "очно",
        re.compile(
            r"\b(?:очно|очная|очный|москва|долгопрудн\w*|сретенк\w*|скорняжн\w*|кампус)\b",
            re.IGNORECASE,
        ),
    ),
    ("выездная школа", re.compile(r"\b(?:выездн\w+|лвш|лагер\w*|проживан\w*|смен[аы])\b", re.IGNORECASE)),
)
INTEREST_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("годовые курсы", re.compile(r"\bгодов\w+\s+курс\w*", re.IGNORECASE)),
    ("онлайн-курсы", re.compile(r"\bонлайн[-\s]+курс\w*", re.IGNORECASE)),
    ("летняя школа", re.compile(r"\bлетн\w+\s+(?:очная\s+)?школ\w*", re.IGNORECASE)),
    ("летняя выездная школа", re.compile(r"\b(?:летн\w+\s+)?выездн\w+\s+школ\w*|\bлвш\b", re.IGNORECASE)),
    ("летний лагерь", re.compile(r"\bлетн\w+\s+лагер\w*", re.IGNORECASE)),
    ("выездная смена", re.compile(r"\bвыездн\w+\s+смен\w*", re.IGNORECASE)),
    ("интенсив", re.compile(r"\bинтенсив\w*", re.IGNORECASE)),
    ("подготовка к ЕГЭ", re.compile(r"\b(?:подготовк\w+\s+к\s+)?егэ\b", re.IGNORECASE)),
    ("подготовка к ОГЭ", re.compile(r"\b(?:подготовк\w+\s+к\s+)?огэ\b", re.IGNORECASE)),
    ("олимпиадная подготовка", re.compile(r"\bолимпиад\w*|\bфизтех\b", re.IGNORECASE)),
    ("индивидуальные занятия", re.compile(r"\bиндивидуальн\w+\s+заняти\w*", re.IGNORECASE)),
    ("пробное занятие", re.compile(r"\bпробн\w+\s+заняти\w*", re.IGNORECASE)),
)
MULTI_CHILD_CONTEXT_RE = re.compile(
    r"\b(?:"
    r"(?:двое|двоих|двух|два|оба|обоих|трое|троих|тр[её]х|три|нескольк\w+)\s+"
    r"(?:дет(?:ей|и)|реб[её]нк\w*|сын\w*|доч\w*|дочер\w*|дочк\w*|ученик\w*|школьник\w*)|"
    r"(?:две|обе)\s+(?:дочер\w*|дочк\w*)|"
    r"дети|детей|многодетн\w+|близнец\w+|"
    r"(?:старш\w+|младш\w+|средн\w+|втор\w+|друг\w+)\s+"
    r"(?:реб[её]н\w*|сын\w*|доч\w*|дочер\w*|дочк\w*)|"
    r"(?:старш(?!\w*\s+класс)|младш(?!\w*\s+класс)|средн(?!\w*\s+класс))\w+|"
    r"сын\w*[^.!?\n]{0,120}доч\w*|доч\w*[^.!?\n]{0,120}сын\w*|"
    r"брат\w*|сестр\w*"
    r")\b",
    re.IGNORECASE,
)
SON_MARKER_RE = re.compile(r"\bсын\w*\b", re.IGNORECASE)
DAUGHTER_MARKER_RE = re.compile(r"\b(?:доч\w*|дочер\w*|дочк\w*)\b", re.IGNORECASE)


@dataclass(frozen=True)
class BotSafeSummaryBuildConfig:
    timeline_db: Path
    allowed_root: Path
    tenant_id: str = "foton"
    apply: bool = False
    limit: int | None = None
    customer_ids: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True)
class CustomerBotSafeSummaryDraft:
    customer_id: str
    chunk: BotContextChunk
    brand: str
    brand_source: str
    source_opportunity_count: int
    source_event_count: int
    source_chunk_count: int
    next_step_status: str


@dataclass(frozen=True)
class BotSafeExtractedSlots:
    child_class: str = ""
    subjects: tuple[str, ...] = ()
    interests: tuple[str, ...] = ()
    formats: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExistingBotSafeChunk:
    record: Mapping[str, Any]
    record_hash: str


@dataclass(frozen=True)
class BotSafeSummaryBuildReport:
    schema_version: str
    timeline_db: str
    tenant_id: str
    applied: bool
    considered_customers: int
    customers_with_history: int
    customers_with_summary: int
    history_coverage_percent: float
    created: int
    updated: int
    duplicate: int
    skipped: int
    retired_stale: int
    brand_counts: Mapping[str, int]
    brand_source_counts: Mapping[str, int]
    next_step_status_counts: Mapping[str, int]
    allowed_chunk_counts_before: Mapping[str, int]
    allowed_chunk_counts_after: Mapping[str, int]
    raw_allowed_chunks_after: int
    examples: Sequence[Mapping[str, Any]] = field(default_factory=tuple)

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


def build_bot_safe_summaries(config: BotSafeSummaryBuildConfig) -> BotSafeSummaryBuildReport:
    db_path = Path(config.timeline_db).expanduser()
    allowed_root = Path(config.allowed_root).expanduser()
    before_counts = _allowed_chunk_counts(db_path)
    existing_chunks = _existing_bot_safe_chunks(db_path, config.tenant_id)
    customers = _customers_with_history(
        db_path,
        config.tenant_id,
        limit=config.limit,
        customer_ids=config.customer_ids,
    )
    opportunities = _opportunities_by_customer(db_path, config.tenant_id)
    events = _events_by_customer(db_path, config.tenant_id)
    source_chunks = _source_chunks_by_customer(db_path, config.tenant_id)
    conflicts = _open_conflicts_by_customer(db_path, config.tenant_id)
    drafts = [
        draft
        for customer_id in customers
        for draft in _non_empty_drafts(
            _build_customer_brand_drafts(
                tenant_id=config.tenant_id,
                customer_id=customer_id,
                all_opportunities=opportunities.get(customer_id, ()),
                all_events=events.get(customer_id, ()),
                all_source_chunks=source_chunks.get(customer_id, ()),
                conflicts=conflicts.get(customer_id, ()),
                existing_chunks=existing_chunks,
            )
        )
    ]
    expected_source_refs = {draft.chunk.source_ref or "" for draft in drafts}
    retire_customer_scope = set(customers) if config.customer_ids else None
    stale_chunks = [
        existing
        for source_ref, existing in existing_chunks.items()
        if source_ref not in expected_source_refs and _chunk_allowed_for_bot(existing.record)
        and (retire_customer_scope is None or str(existing.record.get("customer_id") or "") in retire_customer_scope)
    ]

    created = updated = duplicate = skipped = 0
    retired_stale = 0
    if config.apply:
        with CustomerTimelineSQLiteStore(db_path, allowed_root=allowed_root) as store:
            with store.bulk_write():
                for draft in drafts:
                    existing = existing_chunks.get(draft.chunk.source_ref or "")
                    if existing is not None and existing.record_hash == _chunk_record_hash(draft.chunk):
                        duplicate += 1
                        continue
                    result = store.upsert_bot_context_chunk(draft.chunk, actor=BOT_SAFE_SUMMARY_ACTOR)
                    if result.status == "created":
                        created += 1
                    elif result.status == "updated":
                        updated += 1
                    elif result.status == "duplicate":
                        duplicate += 1
                    else:
                        skipped += 1
                for existing in stale_chunks:
                    result = store.upsert_bot_context_chunk(
                        _retired_bot_safe_chunk(existing.record),
                        actor=BOT_SAFE_SUMMARY_ACTOR,
                    )
                    if result.status != "duplicate":
                        retired_stale += 1
    else:
        created, updated, duplicate = _dry_run_status_counts(drafts, existing_chunks)
        retired_stale = len(stale_chunks)

    after_counts = _allowed_chunk_counts(db_path)
    brand_counts = _count_values(draft.brand for draft in drafts)
    brand_source_counts = _count_values(draft.brand_source for draft in drafts)
    next_step_counts = _count_values(draft.next_step_status for draft in drafts)
    history_count = len(customers)
    summary_count = len(drafts)
    coverage = round((summary_count / history_count * 100), 2) if history_count else 0.0
    return BotSafeSummaryBuildReport(
        schema_version=BOT_SAFE_SUMMARY_SCHEMA_VERSION,
        timeline_db=str(db_path),
        tenant_id=config.tenant_id,
        applied=config.apply,
        considered_customers=len(customers),
        customers_with_history=history_count,
        customers_with_summary=summary_count,
        history_coverage_percent=coverage,
        created=created,
        updated=updated,
        duplicate=duplicate,
        skipped=skipped,
        retired_stale=retired_stale,
        brand_counts=brand_counts,
        brand_source_counts=brand_source_counts,
        next_step_status_counts=next_step_counts,
        allowed_chunk_counts_before=before_counts,
        allowed_chunk_counts_after=after_counts,
        raw_allowed_chunks_after=_raw_allowed_chunks(db_path),
    )


def _build_customer_draft(
    *,
    tenant_id: str,
    customer_id: str,
    brand: str,
    opportunities: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    source_chunks: Sequence[Mapping[str, Any]],
    conflicts: Sequence[Mapping[str, Any]],
    existing_chunk: Mapping[str, Any] | None,
) -> CustomerBotSafeSummaryDraft | None:
    brand_source = _brand_source(opportunities, events, source_chunks, brand=brand)
    slots = _extract_bot_safe_slots(opportunities, events, source_chunks, brand=brand)
    next_step = resolve_customer_next_step(
        events,
        readiness={"open_conflicts": len(conflicts)},
        conflicts=conflicts,
        customer_id=customer_id,
    )
    safe_next_step = _safe_next_step(next_step)
    text = _render_safe_text(brand=brand, slots=slots, safe_next_step=safe_next_step)
    if not text:
        return None
    latest_at = _latest_event_at(opportunities, events, source_chunks)
    created_at = _existing_created_at(existing_chunk) or now_utc()
    source_ref = _bot_safe_source_ref(customer_id=customer_id, brand=brand)
    chunk = BotContextChunk(
        tenant_id=tenant_id,
        customer_id=customer_id,
        chunk_type=BOT_SAFE_SUMMARY_CHUNK_TYPE,
        text=text,
        summary=text,
        source_system=BOT_SAFE_SUMMARY_SOURCE_SYSTEM,
        source_ref=source_ref,
        event_at=latest_at,
        freshness_score=1.0,
        relevance_tags=("bot_safe", "structured", brand),
        allowed_for_bot=True,
        requires_manager_review=False,
        metadata={
            "schema_version": BOT_SAFE_SUMMARY_SCHEMA_VERSION,
            "raw_text_used": False,
            "brand_source": brand_source,
            "opportunity_count": len(opportunities),
            "event_count": len(events),
            "source_chunk_count": len(source_chunks),
            "next_step": _safe_next_step_metadata(next_step),
            "safe_next_step": safe_next_step,
            "safe_slots": {
                "child_class": slots.child_class,
                "subjects": list(slots.subjects),
                "interests": list(slots.interests),
                "formats": list(slots.formats),
            },
        },
        created_at=created_at,
    )
    return CustomerBotSafeSummaryDraft(
        customer_id=customer_id,
        chunk=chunk,
        brand=brand,
        brand_source=brand_source,
        source_opportunity_count=len(opportunities),
        source_event_count=len(events),
        source_chunk_count=len(source_chunks),
        next_step_status=next_step.status,
    )


def _build_customer_brand_drafts(
    *,
    tenant_id: str,
    customer_id: str,
    all_opportunities: Sequence[Mapping[str, Any]],
    all_events: Sequence[Mapping[str, Any]],
    all_source_chunks: Sequence[Mapping[str, Any]],
    conflicts: Sequence[Mapping[str, Any]],
    existing_chunks: Mapping[str, ExistingBotSafeChunk],
) -> tuple[CustomerBotSafeSummaryDraft | None, ...]:
    drafts: list[CustomerBotSafeSummaryDraft | None] = []
    brands = _customer_summary_brands(all_opportunities, all_events, all_source_chunks)
    include_unbranded_events = len([brand for brand in brands if brand in KNOWN_BRANDS]) == 1
    for brand in brands:
        opportunities = _opportunities_for_brand(all_opportunities, brand=brand)
        events = _events_for_brand(all_events, brand=brand, include_unbranded=include_unbranded_events)
        source_chunks = _source_chunks_for_brand(
            all_source_chunks,
            brand=brand,
            include_unbranded=include_unbranded_events,
        )
        source_ref = _bot_safe_source_ref(customer_id=customer_id, brand=brand)
        drafts.append(
            _build_customer_draft(
                tenant_id=tenant_id,
                customer_id=customer_id,
                brand=brand,
                opportunities=opportunities,
                events=events,
                source_chunks=source_chunks,
                conflicts=conflicts,
                existing_chunk=existing_chunks[source_ref].record if source_ref in existing_chunks else None,
            )
        )
    return tuple(drafts)


def _non_empty_drafts(
    values: Sequence[CustomerBotSafeSummaryDraft | None],
) -> tuple[CustomerBotSafeSummaryDraft, ...]:
    return tuple(value for value in values if value is not None)


def _render_safe_text(*, brand: str, slots: BotSafeExtractedSlots, safe_next_step: str) -> str:
    parts: list[str] = []
    if brand in KNOWN_BRANDS:
        parts.append(f"Бренд: {_brand_label(brand)}.")
    if slots.child_class:
        parts.append(f"Ребёнок: {slots.child_class} класс.")
    interest_parts = _join_unique((*slots.subjects, *slots.interests), max_items=6)
    if interest_parts:
        parts.append(f"Интерес: {interest_parts}.")
    if slots.formats:
        if len(slots.formats) == 1:
            parts.append(f"Формат: {slots.formats[0]}.")
        else:
            parts.append(f"Рассматривались форматы: {_join_unique(slots.formats, max_items=3)}.")
    known_fields = _known_field_labels(slots)
    if known_fields:
        parts.append(f"Уже известно: {'; '.join(known_fields)}.")
        parts.append(f"Не переспрашивать: {', '.join(_known_field_names(slots))}.")
    if safe_next_step:
        parts.append(f"Следующий безопасный шаг: {safe_next_step}.")
    if len(parts) <= 1:
        return ""
    return " ".join(parts)


def _customer_summary_brands(
    opportunities: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    source_chunks: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    brands = {
        brand
        for brand in (
            *(_opportunity_brand(opportunity) for opportunity in opportunities),
            *(_event_brand(event) for event in events),
            *(_source_chunk_brand(chunk) for chunk in source_chunks),
        )
        if brand in KNOWN_BRANDS
    }
    if brands:
        return tuple(sorted(brands))
    return ("unknown",)


def _opportunities_for_brand(opportunities: Sequence[Mapping[str, Any]], *, brand: str) -> tuple[Mapping[str, Any], ...]:
    if brand not in KNOWN_BRANDS:
        return tuple(opportunities)
    return tuple(opportunity for opportunity in opportunities if _opportunity_brand(opportunity) == brand)


def _events_for_brand(
    events: Sequence[Mapping[str, Any]],
    *,
    brand: str,
    include_unbranded: bool = False,
) -> tuple[Mapping[str, Any], ...]:
    if brand not in KNOWN_BRANDS:
        return tuple(event for event in events if _event_brand(event) not in KNOWN_BRANDS)
    return tuple(
        event
        for event in events
        if _event_brand(event) == brand or (include_unbranded and _event_brand(event) not in KNOWN_BRANDS)
    )


def _brand_source(
    opportunities: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    source_chunks: Sequence[Mapping[str, Any]],
    *,
    brand: str,
) -> str:
    if brand in KNOWN_BRANDS and any(_opportunity_brand(opportunity) == brand for opportunity in opportunities):
        return "customer_opportunities.product_context.brand"
    if brand in KNOWN_BRANDS and any(_event_brand(event) == brand for event in events):
        return "timeline_events.metadata_or_record.brand"
    if brand in KNOWN_BRANDS and any(_source_chunk_brand(chunk) == brand for chunk in source_chunks):
        return "bot_context_chunks.relevance_tags_or_metadata.brand"
    return "unknown"


def _opportunity_brand(opportunity: Mapping[str, Any]) -> str:
    product_context = _mapping(opportunity.get("product_context"))
    return _normalize_brand(product_context.get("brand"))


def _event_brand(event: Mapping[str, Any]) -> str:
    metadata = _mapping(event.get("metadata"))
    record = _mapping(event.get("record"))
    return _normalize_brand(metadata.get("brand") or record.get("brand"))


def _source_chunk_brand(chunk: Mapping[str, Any]) -> str:
    metadata = _mapping(chunk.get("metadata"))
    brand = _normalize_brand(metadata.get("brand"))
    if brand in KNOWN_BRANDS:
        return brand
    for tag in _text_values(chunk.get("relevance_tags")):
        text = str(tag or "").strip().casefold()
        if text.startswith("brand:"):
            brand = _normalize_brand(text.split(":", 1)[1])
            if brand in KNOWN_BRANDS:
                return brand
        brand = _normalize_brand(text)
        if brand in KNOWN_BRANDS:
            return brand
    return "unknown"


def _source_chunks_for_brand(
    chunks: Sequence[Mapping[str, Any]],
    *,
    brand: str,
    include_unbranded: bool = False,
) -> tuple[Mapping[str, Any], ...]:
    if brand not in KNOWN_BRANDS:
        return tuple(chunk for chunk in chunks if _source_chunk_brand(chunk) not in KNOWN_BRANDS)
    return tuple(
        chunk
        for chunk in chunks
        if _source_chunk_brand(chunk) == brand or (include_unbranded and _source_chunk_brand(chunk) not in KNOWN_BRANDS)
    )


def _extract_bot_safe_slots(
    opportunities: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    source_chunks: Sequence[Mapping[str, Any]],
    *,
    brand: str,
) -> BotSafeExtractedSlots:
    text_sources = _bot_safe_slot_sources(opportunities, events, source_chunks)
    raw_text_sources = _bot_safe_slot_sources(opportunities, events, source_chunks, safe=False)
    child_class = _confirmed_child_class(text_sources, raw_text_sources=raw_text_sources)
    subjects = _extract_fixed_labels(text_sources, SUBJECT_PATTERNS, max_items=4)
    formats = _extract_fixed_labels(text_sources, FORMAT_PATTERNS, max_items=3)
    derived_interests = _extract_fixed_labels(text_sources, INTEREST_PATTERNS, max_items=5)
    opportunity_interest = _resolve_interest(opportunities, brand=brand)
    interests = _join_interest_labels(
        (*derived_interests, *_split_joined_interest(opportunity_interest)),
        brand=brand,
        max_items=6,
    )
    return BotSafeExtractedSlots(
        child_class=child_class,
        subjects=subjects,
        interests=interests,
        formats=formats,
    )


def _bot_safe_slot_sources(
    opportunities: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    source_chunks: Sequence[Mapping[str, Any]],
    *,
    safe: bool = True,
) -> tuple[str, ...]:
    values: list[str] = []
    for opportunity in opportunities:
        product_context = _mapping(opportunity.get("product_context"))
        values.extend(_product_context_text_values(product_context))
        values.append(str(opportunity.get("title") or ""))
    for event in events:
        values.extend(
            str(event.get(key) or "")
            for key in ("subject", "summary", "text_preview")
        )
        record = _mapping(event.get("record"))
        values.extend(str(record.get(key) or "") for key in ("summary", "text", "title"))
    for chunk in source_chunks:
        values.extend(str(chunk.get(key) or "") for key in ("summary", "text"))
    if not safe:
        return tuple(str(value) for value in values if str(value or "").strip())
    return tuple(_safe_fragment(value, max_len=1200) for value in values if str(value or "").strip())


def _product_context_text_values(product_context: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("title", "subject", "format", "class", "product_of_interest", "products_of_interest"):
        values.extend(_plain_text_values(product_context.get(key)))
    return values


def _plain_text_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        result: list[str] = []
        for key in ("title", "name", "subject", "format", "class"):
            result.extend(_plain_text_values(value.get(key)))
        return result
    if isinstance(value, Iterable):
        result: list[str] = []
        for item in value:
            result.extend(_plain_text_values(item))
        return result
    return []


def _confirmed_child_class(
    text_sources: Sequence[str],
    *,
    raw_text_sources: Sequence[str] | None = None,
) -> str:
    guard_sources = raw_text_sources if raw_text_sources is not None else text_sources
    if _has_multi_child_context(guard_sources):
        return ""
    if len(_direct_digit_child_class_candidates(guard_sources, include_lower_grades=True)) >= 2:
        return ""
    candidates = _child_class_candidates(text_sources)
    if len(candidates) != 1:
        return ""
    return next(iter(candidates))


def _child_class_candidates(text_sources: Sequence[str]) -> frozenset[str]:
    values: set[str] = set()
    values.update(_direct_digit_child_class_candidates(text_sources))
    for text in text_sources:
        for match in M_CLASS_RE.finditer(text):
            values.add(match.group("class"))
        for value, pattern in WORD_CLASS_PATTERNS:
            if pattern.search(text):
                values.add(value)
    for text in text_sources:
        normalized = text.casefold().replace("ё", "е")
        if re.search(r"\bегэ\b", normalized):
            values.add("11")
        if re.search(r"\bогэ\b", normalized):
            values.add("9")
    return frozenset(values)


def _direct_digit_child_class_candidates(
    text_sources: Sequence[str],
    *,
    include_lower_grades: bool = False,
) -> frozenset[str]:
    values: set[str] = set()
    range_re = DIRECT_DIGIT_CLASS_RANGE_RE if include_lower_grades else CLASS_RANGE_RE
    coordinated_re = DIRECT_DIGIT_COORDINATED_CLASS_RE if include_lower_grades else COORDINATED_CLASS_RE
    class_re = DIRECT_DIGIT_CLASS_RE if include_lower_grades else CLASS_RE
    minimum_class = 1 if include_lower_grades else 5
    for text in text_sources:
        for match in range_re.finditer(text):
            start = int(match.group("start"))
            end = int(match.group("end"))
            lower, upper = sorted((start, end))
            values.update(str(item) for item in range(lower, upper + 1) if minimum_class <= item <= 11)
        for match in coordinated_re.finditer(text):
            values.add(match.group("first"))
            values.add(match.group("second"))
        for match in class_re.finditer(text):
            values.add(match.group("class"))
    return frozenset(values)


def _has_multi_child_context(text_sources: Sequence[str]) -> bool:
    if any(MULTI_CHILD_CONTEXT_RE.search(text) for text in text_sources):
        return True
    combined = " ".join(str(text or "") for text in text_sources)
    return bool(SON_MARKER_RE.search(combined) and DAUGHTER_MARKER_RE.search(combined))


def _extract_fixed_labels(
    text_sources: Sequence[str],
    patterns: Sequence[tuple[str, re.Pattern[str]]],
    *,
    max_items: int,
) -> tuple[str, ...]:
    values: list[str] = []
    seen: set[str] = set()
    for text in text_sources:
        safe_text = _text_without_unsafe_finance_fragments(text)
        for label, pattern in patterns:
            key = label.casefold()
            if key in seen:
                continue
            if pattern.search(safe_text):
                values.append(label)
                seen.add(key)
                if len(values) >= max_items:
                    return tuple(values)
    return tuple(values)


def _text_without_unsafe_finance_fragments(value: str) -> str:
    fragments = re.split(r"(?<=[.!?])\s+|[;|]", str(value or ""))
    safe = [
        fragment
        for fragment in fragments
        if not any(marker in fragment.casefold().replace("ё", "е") for marker in UNSAFE_INTEREST_MARKERS)
    ]
    return " ".join(safe)


def _split_joined_interest(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(value or "").split(";") if part.strip())


def _join_interest_labels(values: Sequence[str], *, brand: str, max_items: int) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _safe_interest_fragment(value, max_len=120)
        if not text or not _interest_fragment_allowed(text, brand=brand):
            continue
        key = text.casefold().replace("ё", "е")
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
        if len(result) >= max_items:
            break
    return tuple(result)


def _known_field_labels(slots: BotSafeExtractedSlots) -> tuple[str, ...]:
    labels: list[str] = []
    if slots.child_class:
        labels.append(f"класс {slots.child_class}")
    if slots.subjects:
        labels.append("предметы: " + ", ".join(slots.subjects))
    if slots.formats:
        if len(slots.formats) == 1:
            labels.append("формат: " + slots.formats[0])
    if slots.interests:
        labels.append("интерес: " + ", ".join(slots.interests[:3]))
    return tuple(labels)


def _known_field_names(slots: BotSafeExtractedSlots) -> tuple[str, ...]:
    names: list[str] = []
    if slots.child_class:
        names.append("класс")
    if slots.subjects:
        names.append("предмет")
    if len(slots.formats) == 1:
        names.append("формат")
    if slots.interests:
        names.append("интерес")
    return tuple(names)


def _safe_next_step(next_step: NextStepResolution) -> str:
    if next_step.status != "active":
        return ""
    value = _safe_fragment(next_step.display_text, max_len=180)
    if not value:
        return ""
    text = value.casefold().replace("ё", "е")
    if any(marker in text for marker in UNSAFE_NEXT_STEP_MARKERS):
        return ""
    if scan_like_pii(value):
        return ""
    if re.search(r"\b(?:ссылк|приглашени|логин|доступ|платформ)\w*", text):
        return "помочь с доступом или ссылкой по выбранному занятию"
    if re.search(r"\b(?:расписан|слот|врем|групп)\w*", text):
        return "дать расписание или варианты групп по выбранному направлению"
    if re.search(r"\b(?:материал|программ|презентац)\w*", text):
        return "дать материалы по выбранному направлению"
    if re.search(r"\b(?:запис|зачисл|оформ)\w*", text):
        return "помочь с записью на выбранный курс"
    return ""


def _safe_next_step_metadata(next_step: NextStepResolution) -> Mapping[str, str]:
    return {
        "schema_version": CUSTOMER_TIMELINE_NEXT_STEP_SCHEMA_VERSION,
        "status": next_step.status,
        "confidence": next_step.confidence,
        "reason_code": next_step.reason_code,
        "source_event_id": next_step.source_event_id,
        "source_event_at": next_step.source_event_at,
        "source_event_type": next_step.source_event_type,
    }


def scan_like_pii(value: str) -> bool:
    return bool(re.search(r"[\w.+-]+@[\w.-]+\.\w+", value) or re.search(r"(?:\+7|8|7)[\s\-()]?\d{3}", value))


def _resolve_interest(opportunities: Sequence[Mapping[str, Any]], *, brand: str) -> str:
    candidates: list[str] = []
    for opportunity in opportunities:
        product_context = _mapping(opportunity.get("product_context"))
        candidates.extend(_interest_values(product_context.get("products_of_interest")))
        candidates.extend(_interest_values(product_context.get("product_of_interest")))
        candidates.extend(_interest_values(product_context.get("title")))
        title = _safe_interest_fragment(opportunity.get("title"))
        if title and not _is_generic_title(title):
            candidates.append(title)
    return _join_unique((item for item in candidates if _interest_fragment_allowed(item, brand=brand)), max_items=3)


def _interest_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [_safe_interest_fragment(value)] if _safe_interest_fragment(value) else []
    if isinstance(value, Mapping):
        result: list[str] = []
        for key in ("title", "name", "subject", "format", "class"):
            text = _safe_interest_fragment(value.get(key))
            if text:
                result.append(text)
        return result
    if isinstance(value, Iterable):
        result: list[str] = []
        for item in value:
            result.extend(_interest_values(item))
        return result
    return []


def _latest_status(opportunities: Sequence[Mapping[str, Any]]) -> str:
    for opportunity in opportunities:
        status = _safe_fragment(opportunity.get("status"))
        if status:
            return status
    return ""


def _latest_event_at(
    opportunities: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    source_chunks: Sequence[Mapping[str, Any]],
) -> datetime | None:
    values: list[datetime] = []
    for opportunity in opportunities:
        for key in ("closed_at", "opened_at"):
            parsed = _parse_iso_datetime(opportunity.get(key))
            if parsed:
                values.append(parsed)
    for event in events:
        parsed = _parse_iso_datetime(event.get("event_at"))
        if parsed:
            values.append(parsed)
    for chunk in source_chunks:
        parsed = _parse_iso_datetime(chunk.get("event_at"))
        if parsed:
            values.append(parsed)
    return max(values) if values else None


def _existing_created_at(existing_chunk: Mapping[str, Any] | None) -> datetime | None:
    if not existing_chunk:
        return None
    return _parse_iso_datetime(existing_chunk.get("created_at"))


def _customers_with_history(
    db_path: Path,
    tenant_id: str,
    *,
    limit: int | None,
    customer_ids: Sequence[str] = (),
) -> tuple[str, ...]:
    sql = """
        SELECT customer_id FROM customer_identities
        WHERE tenant_id = ?
    """
    params: list[Any] = [tenant_id]
    selected_ids = tuple(dict.fromkeys(str(item).strip() for item in customer_ids if str(item).strip()))
    if selected_ids:
        placeholders = ",".join("?" for _ in selected_ids)
        sql += f" AND customer_id IN ({placeholders})"
        params.extend(selected_ids)
    sql += """
          AND (
            EXISTS (SELECT 1 FROM customer_opportunities o WHERE o.tenant_id = customer_identities.tenant_id AND o.customer_id = customer_identities.customer_id)
            OR EXISTS (SELECT 1 FROM timeline_events e WHERE e.tenant_id = customer_identities.tenant_id AND e.customer_id = customer_identities.customer_id)
            OR EXISTS (
                SELECT 1 FROM bot_context_chunks c
                WHERE c.tenant_id = customer_identities.tenant_id
                  AND c.customer_id = customer_identities.customer_id
                  AND c.chunk_type != ?
            )
          )
        ORDER BY customer_id
    """
    params.append(BOT_SAFE_SUMMARY_CHUNK_TYPE)
    if limit is not None:
        sql += " LIMIT ?"
        params.append(int(limit))
    with sqlite3.connect(db_path) as con:
        rows = con.execute(sql, params).fetchall()
    return tuple(str(row[0]) for row in rows)


def _opportunities_by_customer(db_path: Path, tenant_id: str) -> dict[str, tuple[Mapping[str, Any], ...]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            """
            SELECT customer_id, record_json
            FROM customer_opportunities
            WHERE tenant_id = ?
            ORDER BY COALESCE(opened_at, closed_at, '' ) DESC, opportunity_id
            """,
            (tenant_id,),
        )
        for row in rows:
            grouped.setdefault(str(row["customer_id"]), []).append(_json_mapping(row["record_json"]))
    return {customer_id: tuple(items) for customer_id, items in grouped.items()}


def _events_by_customer(db_path: Path, tenant_id: str) -> dict[str, tuple[Mapping[str, Any], ...]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            """
            SELECT customer_id, record_json
            FROM timeline_events
            WHERE tenant_id = ? AND customer_id IS NOT NULL AND customer_id != ''
            ORDER BY event_at ASC, event_id ASC
            """,
            (tenant_id,),
        )
        for row in rows:
            grouped.setdefault(str(row["customer_id"]), []).append(_json_mapping(row["record_json"]))
    return {customer_id: tuple(items[-500:]) for customer_id, items in grouped.items()}


def _source_chunks_by_customer(db_path: Path, tenant_id: str) -> dict[str, tuple[Mapping[str, Any], ...]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    placeholders = ",".join("?" for _ in BOT_SAFE_SOURCE_CHUNK_TYPES)
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            f"""
            SELECT customer_id, record_json
            FROM bot_context_chunks
            WHERE tenant_id = ?
              AND customer_id IS NOT NULL
              AND customer_id != ''
              AND chunk_type IN ({placeholders})
            ORDER BY event_at ASC, created_at ASC, chunk_id ASC
            """,
            (tenant_id, *sorted(BOT_SAFE_SOURCE_CHUNK_TYPES)),
        )
        for row in rows:
            grouped.setdefault(str(row["customer_id"]), []).append(_json_mapping(row["record_json"]))
    return {customer_id: tuple(items[-700:]) for customer_id, items in grouped.items()}


def _open_conflicts_by_customer(db_path: Path, tenant_id: str) -> dict[str, tuple[Mapping[str, Any], ...]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            """
            SELECT conflict_type, status, record_json
            FROM timeline_conflicts
            WHERE tenant_id = ? AND status = 'open'
            ORDER BY created_at ASC, conflict_id ASC
            """,
            (tenant_id,),
        )
        for row in rows:
            item = dict(_json_mapping(row["record_json"]))
            item.setdefault("conflict_type", row["conflict_type"])
            item.setdefault("status", row["status"])
            for customer_id in _customer_ids_from_conflict(item):
                grouped.setdefault(customer_id, []).append(item)
    return {customer_id: tuple(items) for customer_id, items in grouped.items()}


def _customer_ids_from_conflict(conflict: Mapping[str, Any]) -> tuple[str, ...]:
    refs = conflict.get("entity_refs")
    candidates: list[str] = []
    if isinstance(refs, (list, tuple, set)):
        for ref in refs:
            text = str(ref or "")
            if text.startswith("customer:"):
                candidates.append(text)
    customer_id = str(conflict.get("customer_id") or "")
    if customer_id.startswith("customer:"):
        candidates.append(customer_id)
    return tuple(dict.fromkeys(candidates))


def _existing_bot_safe_chunks(db_path: Path, tenant_id: str) -> dict[str, ExistingBotSafeChunk]:
    result: dict[str, ExistingBotSafeChunk] = {}
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            """
            SELECT source_ref, record_hash, record_json
            FROM bot_context_chunks
            WHERE tenant_id = ? AND chunk_type = ? AND source_system = ?
            """,
            (tenant_id, BOT_SAFE_SUMMARY_CHUNK_TYPE, BOT_SAFE_SUMMARY_SOURCE_SYSTEM),
        )
        for row in rows:
            source_ref = str(row["source_ref"] or "")
            if not source_ref:
                continue
            result[source_ref] = ExistingBotSafeChunk(
                record=_json_mapping(row["record_json"]),
                record_hash=str(row["record_hash"] or ""),
            )
    return result


def _dry_run_status_counts(
    drafts: Sequence[CustomerBotSafeSummaryDraft],
    existing_chunks: Mapping[str, ExistingBotSafeChunk],
) -> tuple[int, int, int]:
    created = updated = duplicate = 0
    for draft in drafts:
        existing = existing_chunks.get(draft.chunk.source_ref or "")
        if not existing:
            created += 1
        elif existing.record_hash == _chunk_record_hash(draft.chunk):
            duplicate += 1
        else:
            updated += 1
    return created, updated, duplicate


def _chunk_record_hash(chunk: BotContextChunk) -> str:
    return stable_digest(scrub_timeline_persisted_json(chunk.to_json_dict()))


def _allowed_chunk_counts(db_path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    with sqlite3.connect(db_path) as con:
        rows = con.execute(
            """
            SELECT chunk_type, COUNT(*)
            FROM bot_context_chunks
            WHERE allowed_for_bot = 1 AND requires_manager_review = 0
            GROUP BY chunk_type
            ORDER BY chunk_type
            """
        ).fetchall()
    for chunk_type, count in rows:
        counts[str(chunk_type)] = int(count)
    return counts


def _raw_allowed_chunks(db_path: Path) -> int:
    with sqlite3.connect(db_path) as con:
        return int(
            con.execute(
                """
                SELECT COUNT(*)
                FROM bot_context_chunks
                WHERE allowed_for_bot = 1
                  AND requires_manager_review = 0
                  AND chunk_type != ?
                """,
                (BOT_SAFE_SUMMARY_CHUNK_TYPE,),
            ).fetchone()[0]
        )


def _retired_bot_safe_chunk(record: Mapping[str, Any]) -> BotContextChunk:
    metadata = dict(_mapping(record.get("metadata")))
    metadata["retired_reason"] = "bot_safe_source_ref_not_rebuilt"
    metadata["retired_by_schema_version"] = BOT_SAFE_SUMMARY_SCHEMA_VERSION
    tags = tuple(dict.fromkeys((*_text_values(record.get("relevance_tags")), "retired")))
    return BotContextChunk(
        tenant_id=str(record.get("tenant_id") or ""),
        customer_id=str(record.get("customer_id") or ""),
        chunk_type=str(record.get("chunk_type") or BOT_SAFE_SUMMARY_CHUNK_TYPE),
        text=_safe_fragment(record.get("text") or record.get("summary") or "Устаревшая выжимка отключена"),
        chunk_id=str(record.get("chunk_id") or ""),
        opportunity_id=str(record.get("opportunity_id") or "") or None,
        event_id=str(record.get("event_id") or "") or None,
        source_ref=str(record.get("source_ref") or ""),
        ordinal=int(record.get("ordinal") or 0),
        source_system=str(record.get("source_system") or BOT_SAFE_SUMMARY_SOURCE_SYSTEM),
        summary=_safe_fragment(record.get("summary") or record.get("text") or "Устаревшая выжимка отключена"),
        event_at=_parse_iso_datetime(record.get("event_at")),
        freshness_score=0.0,
        relevance_tags=tags,
        allowed_for_bot=False,
        requires_manager_review=True,
        metadata=metadata,
        created_at=_parse_iso_datetime(record.get("created_at")) or now_utc(),
    )


def _chunk_allowed_for_bot(record: Mapping[str, Any]) -> bool:
    return record.get("allowed_for_bot") is True or str(record.get("allowed_for_bot") or "").strip() == "1"


def _text_values(value: Any) -> tuple[str, ...]:
    if isinstance(value, (list, tuple, set)):
        return tuple(str(item) for item in value if str(item or "").strip())
    return ()


def _safe_fragment(value: Any, *, max_len: int = 160) -> str:
    text = redact_text(str(value or ""))
    text = LONG_DIGIT_TOKEN_RE.sub("<number_masked>", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    text = MAIL_REPLY_PREFIX_RE.sub("", text).strip()
    if len(text) > max_len:
        return text[: max_len - 1].rstrip() + "…"
    return text


def _safe_interest_fragment(value: Any, *, max_len: int = 160) -> str:
    text = _safe_fragment(value, max_len=max_len)
    if not text:
        return ""
    return _scrub_interest_person_names(text)


def _scrub_interest_person_names(value: str) -> str:
    text, protected = _protect_safe_interest_phrases(value)
    text = NEXT_STEP_ROLE_PERSON_RE.sub(lambda match: f"{match.group('role')} {INTEREST_NAME_PLACEHOLDER}", text)
    text = NEXT_STEP_PERSON_NAME_RE.sub(INTEREST_NAME_PLACEHOLDER, text)
    text = INSIGHTS_COMMON_SINGLE_NAME_RE.sub(INTEREST_NAME_PLACEHOLDER, text)
    text = _restore_safe_interest_phrases(text, protected)
    text = re.sub(rf"(?:{re.escape(INTEREST_NAME_PLACEHOLDER)}\s*){{2,}}", INTEREST_NAME_PLACEHOLDER, text)
    return WHITESPACE_RE.sub(" ", text).strip(" ;,")


def _protect_safe_interest_phrases(value: str) -> tuple[str, dict[str, str]]:
    protected: dict[str, str] = {}
    text = value
    for pattern in SAFE_INTEREST_PHRASE_PATTERNS:
        def replace(match: re.Match[str]) -> str:
            token = f"__botsafe_interest_safe_{len(protected)}__"
            protected[token] = match.group(0)
            return token

        text = pattern.sub(replace, text)
    return text, protected


def _restore_safe_interest_phrases(value: str, protected: Mapping[str, str]) -> str:
    text = value
    for token, phrase in protected.items():
        text = text.replace(token, phrase)
    return text


def _brand_label(brand: str) -> str:
    return {"foton": "Фотон", "unpk": "УНПК"}.get(brand, brand)


def _normalize_brand(value: Any) -> str:
    text = str(value or "").strip().casefold()
    if text in {"foton", "фотон"}:
        return "foton"
    if text in {"unpk", "унпк", "мфти", "unpk_mfti"}:
        return "unpk"
    return "unknown"


def _brand_fragment_allowed(value: str, *, brand: str) -> bool:
    brands = _brands_mentioned(value)
    if not brands:
        return True
    if brand not in KNOWN_BRANDS:
        return False
    return brands <= {brand}


def _interest_fragment_allowed(value: str, *, brand: str) -> bool:
    if not _brand_fragment_allowed(value, brand=brand):
        return False
    text = str(value or "").casefold().replace("ё", "е")
    semantic_text = _interest_semantic_text_without_person_markers(value).casefold().replace("ё", "е")
    if not re.search(r"[a-zа-я]", semantic_text, flags=re.IGNORECASE):
        return False
    if any(marker in text for marker in ("<phone_masked>", "<email_masked>", "<number_masked>")):
        return False
    if re.search(r"(?<!\d)[1-4]\s*(?:класс|кл\.?)\b", text, flags=re.IGNORECASE):
        return False
    if FILE_NAME_FRAGMENT_RE.search(text):
        return False
    return not any(marker in text for marker in UNSAFE_INTEREST_MARKERS)


def _interest_semantic_text_without_person_markers(value: str) -> str:
    text = str(value or "").replace(INTEREST_NAME_PLACEHOLDER, " ")
    text = INTEREST_ROLE_WORD_RE.sub(" ", text)
    return WHITESPACE_RE.sub(" ", text).strip(" ;,.:-")


def _brands_mentioned(value: str) -> set[str]:
    text = str(value or "").casefold().replace("ё", "е")
    brands: set[str] = set()
    if any(marker in text for marker in FOTON_MARKERS):
        brands.add("foton")
    if any(marker in text for marker in UNPK_MARKERS):
        brands.add("unpk")
    return brands


def _is_generic_title(value: str) -> bool:
    return any(pattern.search(value) for pattern in GENERIC_TITLE_PATTERNS)


def _join_unique(values: Sequence[str], *, max_items: int) -> str:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _safe_fragment(value)
        key = text.casefold()
        if not text or key in seen:
            continue
        seen.add(key)
        result.append(text)
        if len(result) >= max_items:
            break
    return "; ".join(result)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _json_mapping(value: Any) -> Mapping[str, Any]:
    try:
        parsed = json.loads(str(value or "{}"))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, Mapping) else {}


def _parse_iso_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _count_values(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = value or "unknown"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _bot_safe_source_ref(*, customer_id: str, brand: str) -> str:
    return f"botsafe:{customer_id}:{brand}"


def expected_bot_safe_chunk_id(*, tenant_id: str, customer_id: str, brand: str = "unknown") -> str:
    return stable_chunk_id(
        tenant_id=tenant_id,
        customer_id=customer_id,
        chunk_type=BOT_SAFE_SUMMARY_CHUNK_TYPE,
        source_ref=_bot_safe_source_ref(customer_id=customer_id, brand=brand),
        ordinal=0,
    )
