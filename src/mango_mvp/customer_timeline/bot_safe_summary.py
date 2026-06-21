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
from mango_mvp.customer_timeline.next_step_resolver import NextStepResolution, resolve_customer_next_step
from mango_mvp.customer_timeline.store import CustomerTimelineSQLiteStore, scrub_timeline_persisted_json


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
    "счет",
    "счёт",
    "чек",
)


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
    next_step_status: str


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
    brand_counts: Mapping[str, int]
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
    drafts = [
        draft
        for customer_id in customers
        for draft in _build_customer_brand_drafts(
            tenant_id=config.tenant_id,
            customer_id=customer_id,
            all_opportunities=opportunities.get(customer_id, ()),
            all_events=events.get(customer_id, ()),
            existing_chunks=existing_chunks,
        )
    ]

    created = updated = duplicate = skipped = 0
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
    else:
        created, updated, duplicate = _dry_run_status_counts(drafts, existing_chunks)

    after_counts = _allowed_chunk_counts(db_path)
    brand_counts = _count_values(draft.brand for draft in drafts)
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
        brand_counts=brand_counts,
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
    existing_chunk: Mapping[str, Any] | None,
) -> CustomerBotSafeSummaryDraft:
    brand_source = _brand_source(opportunities, events, brand=brand)
    status = _latest_status(opportunities)
    interest = _resolve_interest(opportunities, brand=brand)
    next_step = resolve_customer_next_step(events, customer_id=customer_id)
    text = _render_safe_text(brand=brand, status=status, interest=interest, next_step=next_step)
    latest_at = _latest_event_at(opportunities, events)
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
            "next_step": next_step.to_json_dict(),
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
        next_step_status=next_step.status,
    )


def _build_customer_brand_drafts(
    *,
    tenant_id: str,
    customer_id: str,
    all_opportunities: Sequence[Mapping[str, Any]],
    all_events: Sequence[Mapping[str, Any]],
    existing_chunks: Mapping[str, ExistingBotSafeChunk],
) -> tuple[CustomerBotSafeSummaryDraft, ...]:
    drafts: list[CustomerBotSafeSummaryDraft] = []
    brands = _customer_summary_brands(all_opportunities, all_events)
    include_unbranded_events = len([brand for brand in brands if brand in KNOWN_BRANDS]) == 1
    for brand in brands:
        opportunities = _opportunities_for_brand(all_opportunities, brand=brand)
        events = _events_for_brand(all_events, brand=brand, include_unbranded=include_unbranded_events)
        source_ref = _bot_safe_source_ref(customer_id=customer_id, brand=brand)
        drafts.append(
            _build_customer_draft(
                tenant_id=tenant_id,
                customer_id=customer_id,
                brand=brand,
                opportunities=opportunities,
                events=events,
                existing_chunk=existing_chunks[source_ref].record if source_ref in existing_chunks else None,
            )
        )
    return tuple(drafts)


def _render_safe_text(*, brand: str, status: str, interest: str, next_step: NextStepResolution) -> str:
    parts: list[str] = []
    if brand in KNOWN_BRANDS:
        parts.append(f"Бренд: {_brand_label(brand)}.")
    parts.append(f"Стадия: {status or 'не определена'}.")
    parts.append(f"Интерес: {interest or 'не определён'}.")
    parts.append(f"Следующий шаг: {_safe_fragment(next_step.display_text) or 'активный шаг не найден'}.")
    return " ".join(parts)


def _customer_summary_brands(
    opportunities: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    brands = {
        brand
        for brand in (
            *(_opportunity_brand(opportunity) for opportunity in opportunities),
            *(_event_brand(event) for event in events),
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


def _brand_source(opportunities: Sequence[Mapping[str, Any]], events: Sequence[Mapping[str, Any]], *, brand: str) -> str:
    if brand in KNOWN_BRANDS and any(_opportunity_brand(opportunity) == brand for opportunity in opportunities):
        return "customer_opportunities.product_context.brand"
    if brand in KNOWN_BRANDS and any(_event_brand(event) == brand for event in events):
        return "timeline_events.metadata_or_record.brand"
    return "unknown"


def _opportunity_brand(opportunity: Mapping[str, Any]) -> str:
    product_context = _mapping(opportunity.get("product_context"))
    return _normalize_brand(product_context.get("brand"))


def _event_brand(event: Mapping[str, Any]) -> str:
    metadata = _mapping(event.get("metadata"))
    record = _mapping(event.get("record"))
    return _normalize_brand(metadata.get("brand") or record.get("brand"))


def _resolve_interest(opportunities: Sequence[Mapping[str, Any]], *, brand: str) -> str:
    candidates: list[str] = []
    for opportunity in opportunities:
        product_context = _mapping(opportunity.get("product_context"))
        candidates.extend(_interest_values(product_context.get("products_of_interest")))
        candidates.extend(_interest_values(product_context.get("product_of_interest")))
        candidates.extend(_interest_values(product_context.get("title")))
        title = _safe_fragment(opportunity.get("title"))
        if title and not _is_generic_title(title):
            candidates.append(title)
    return _join_unique((item for item in candidates if _interest_fragment_allowed(item, brand=brand)), max_items=3)


def _interest_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [_safe_fragment(value)] if _safe_fragment(value) else []
    if isinstance(value, Mapping):
        result: list[str] = []
        for key in ("title", "name", "subject", "format", "class"):
            text = _safe_fragment(value.get(key))
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


def _latest_event_at(opportunities: Sequence[Mapping[str, Any]], events: Sequence[Mapping[str, Any]]) -> datetime | None:
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


def _safe_fragment(value: Any, *, max_len: int = 160) -> str:
    text = redact_text(str(value or ""))
    text = LONG_DIGIT_TOKEN_RE.sub("<number_masked>", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    text = MAIL_REPLY_PREFIX_RE.sub("", text).strip()
    if len(text) > max_len:
        return text[: max_len - 1].rstrip() + "…"
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
    if not re.search(r"[a-zа-я]", text, flags=re.IGNORECASE):
        return False
    if FILE_NAME_FRAGMENT_RE.search(text):
        return False
    return not any(marker in text for marker in UNSAFE_INTEREST_MARKERS)


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
