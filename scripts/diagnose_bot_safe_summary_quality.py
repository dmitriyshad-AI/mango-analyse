#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from mango_mvp.customer_timeline.bot_safe_runtime_context import (
    _is_junk_bot_safe_summary,
    scan_bot_safe_context_pii,
)
from mango_mvp.customer_timeline.bot_safe_summary import (
    BOT_SAFE_SUMMARY_CHUNK_TYPE,
    BOT_SAFE_SUMMARY_SOURCE_SYSTEM,
    _customer_summary_brands,
    _events_by_customer,
    _events_for_brand,
    _extract_bot_safe_slots,
    _open_conflicts_by_customer,
    _opportunities_by_customer,
    _opportunities_for_brand,
    _source_chunks_by_customer,
    _source_chunks_for_brand,
    _text_values,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose empty bot-safe summary chunks without exposing raw PII.")
    parser.add_argument("--timeline-db", required=True, type=Path)
    parser.add_argument("--tenant-id", default="foton")
    parser.add_argument("--sample-per-brand", type=int, default=100)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--md-out", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    db_path = Path(args.timeline_db).expanduser()
    empty_chunks = _empty_chunks_by_brand(db_path, args.tenant_id, brands=("foton", "unpk"), limit=args.sample_per_brand)
    opportunities = _opportunities_by_customer(db_path, args.tenant_id)
    events = _events_by_customer(db_path, args.tenant_id)
    source_chunks = _source_chunks_by_customer(db_path, args.tenant_id)
    conflicts = _open_conflicts_by_customer(db_path, args.tenant_id)
    report = {
        "timeline_db": str(db_path),
        "tenant_id": args.tenant_id,
        "sample_per_brand": args.sample_per_brand,
        "brands": {},
    }
    for brand, rows in empty_chunks.items():
        classified = [
            _classify_empty_chunk(
                row,
                brand=brand,
                opportunities=opportunities.get(row["customer_id"], ()),
                events=events.get(row["customer_id"], ()),
                source_chunks=source_chunks.get(row["customer_id"], ()),
                conflicts=conflicts.get(row["customer_id"], ()),
            )
            for row in rows
        ]
        counts = Counter(item["reason"] for item in classified)
        report["brands"][brand] = {
            "sampled": len(rows),
            "reason_counts": dict(sorted(counts.items())),
            "examples": classified[:8],
        }
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    if args.md_out:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(_render_md(report), encoding="utf-8")
    print(text)
    return 0


def _empty_chunks_by_brand(
    db_path: Path,
    tenant_id: str,
    *,
    brands: Sequence[str],
    limit: int,
) -> Mapping[str, Sequence[Mapping[str, Any]]]:
    result: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            """
            SELECT customer_id, source_ref, event_at, record_json
            FROM bot_context_chunks
            WHERE tenant_id = ?
              AND chunk_type = ?
              AND source_system = ?
              AND allowed_for_bot = 1
              AND requires_manager_review = 0
            ORDER BY event_at DESC, customer_id
            """,
            (tenant_id, BOT_SAFE_SUMMARY_CHUNK_TYPE, BOT_SAFE_SUMMARY_SOURCE_SYSTEM),
        )
        for row in rows:
            record = _json_mapping(row["record_json"])
            tags = set(_text_values(record.get("relevance_tags")))
            text = str(record.get("summary") or record.get("text") or "")
            if not _is_junk_bot_safe_summary(text):
                continue
            for brand in brands:
                if brand in tags and len(result[brand]) < limit:
                    result[brand].append(
                        {
                            "customer_id": str(row["customer_id"] or ""),
                            "source_ref": str(row["source_ref"] or ""),
                            "event_at": str(row["event_at"] or ""),
                            "existing_text_pii": scan_bot_safe_context_pii(text),
                        }
                    )
            if all(len(result[brand]) >= limit for brand in brands):
                break
    return result


def _classify_empty_chunk(
    row: Mapping[str, Any],
    *,
    brand: str,
    opportunities: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    source_chunks: Sequence[Mapping[str, Any]],
    conflicts: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    known_brands = _customer_summary_brands(opportunities, events, source_chunks)
    brand_opportunities = _opportunities_for_brand(opportunities, brand=brand)
    include_unbranded = len([item for item in known_brands if item in {"foton", "unpk"}]) == 1
    brand_events = _events_for_brand(events, brand=brand, include_unbranded=include_unbranded)
    brand_chunks = _source_chunks_for_brand(source_chunks, brand=brand, include_unbranded=include_unbranded)
    slots = _extract_bot_safe_slots(brand_opportunities, brand_events, brand_chunks, brand=brand)
    has_slots = bool(slots.child_class or slots.subjects or slots.interests or slots.formats)
    if conflicts:
        reason = "ambiguous_identity"
    elif brand not in known_brands:
        reason = "brand_unknown"
    elif has_slots:
        reason = "builder_missed_extractable_slots"
    elif brand_chunks and not (brand_opportunities or brand_events):
        reason = "only_manager_only_data"
    elif not (brand_opportunities or brand_events or brand_chunks):
        reason = "truly_no_data"
    else:
        reason = "other_no_safe_slots"
    return {
        "sample_id": _mask(row.get("customer_id")),
        "brand": brand,
        "reason": reason,
        "source_counts": {
            "opportunities": len(brand_opportunities),
            "events": len(brand_events),
            "manager_only_chunks": len(brand_chunks),
            "open_conflicts": len(conflicts),
        },
        "extractable_slots": {
            "child_class": slots.child_class,
            "subjects": list(slots.subjects),
            "interests": list(slots.interests),
            "formats": list(slots.formats),
        },
    }


def _render_md(report: Mapping[str, Any]) -> str:
    lines = [
        "# Диагностика пустых bot-safe выжимок",
        "",
        f"БД: `{report['timeline_db']}`",
        f"Выборка на бренд: {report['sample_per_brand']}",
        "",
    ]
    for brand, payload in report["brands"].items():
        lines.append(f"## {brand}")
        lines.append("")
        lines.append("| Причина | Кол-во |")
        lines.append("|---|---:|")
        for reason, count in payload["reason_counts"].items():
            lines.append(f"| {reason} | {count} |")
        lines.append("")
        lines.append("Примеры без ПДн:")
        for item in payload["examples"]:
            slots = item["extractable_slots"]
            lines.append(
                "- "
                f"{item['sample_id']}: {item['reason']}; "
                f"класс={slots['child_class'] or '-'}; "
                f"предметы={', '.join(slots['subjects']) or '-'}; "
                f"интерес={', '.join(slots['interests']) or '-'}; "
                f"формат={', '.join(slots['formats']) or '-'}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _json_mapping(value: Any) -> Mapping[str, Any]:
    try:
        parsed = json.loads(str(value or "{}"))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, Mapping) else {}


def _mask(value: Any) -> str:
    digest = hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()
    return f"sample:{digest[:12]}"


if __name__ == "__main__":
    raise SystemExit(main())
