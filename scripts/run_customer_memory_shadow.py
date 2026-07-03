from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Mapping, Sequence

from mango_mvp.channels.subscription_llm_parts.direct_path import _direct_path_customer_memory_shadow_trace
from mango_mvp.customer_timeline.bot_safe_runtime_context import (
    TIMELINE_MEMORY_EXPANDED_SHADOW_ENV,
    scan_bot_safe_context_pii,
)
from mango_mvp.customer_timeline.read_api import CustomerTimelineReadApi, CustomerTimelineReadApiConfig


_KNOWN_BRANDS = {"foton", "unpk"}
_SERVICE_ID_RE = re.compile(r"\b(?:customer|timeline_event|bot_context_chunk|botsafe):[^\s,;]+", re.I)
_TEMPORAL_MARKER_RE = re.compile(
    r"\b(?:20\d{2}|\d{2}\s*[-–—/]\s*\d{2}|[12]\s*сем(?:естр|\.?)|[12]\s*полугодие|август|сентябр|октябр|ноябр|декабр|январ|феврал|март|апрел|ма[йя]|июн|июл|уч\.?\s*г)\b",
    re.I,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run expanded Customer Timeline memory shadow on staging DB.")
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--out-jsonl", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--tenant-id", default="foton")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args(argv)

    db_path = args.db.expanduser().resolve(strict=True)
    require_staging_db_path(db_path)
    out_jsonl = args.out_jsonl.expanduser()
    summary_json = args.summary_json.expanduser()
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    candidates = _select_llm_customers(db_path, tenant_id=args.tenant_id, limit=max(1, args.limit))
    rows: list[Mapping[str, Any]] = []
    with CustomerTimelineReadApi.open(CustomerTimelineReadApiConfig(timeline_db=db_path, allowed_root=db_path.parent)) as api:
        for candidate in candidates:
            bot_context = api.bot_context(args.tenant_id, candidate["customer_id"], allowed_only=True, limit=50)
            brand, brand_source = infer_shadow_brand(candidate.get("customer_brand"), bot_context.get("items") or ())
            context = {
                "active_brand": brand,
                TIMELINE_MEMORY_EXPANDED_SHADOW_ENV: "1",
                "timeline_context": {
                    "source": "customer_timeline_bot_context",
                    "found": True,
                    "bot_context": bot_context,
                },
                "recent_messages": [],
            }
            trace = _direct_path_customer_memory_shadow_trace(context)
            prompt_text = str(trace.get("prompt_text") or "")
            rows.append(
                {
                    "tenant_id": args.tenant_id,
                    "customer_id": candidate["customer_id"],
                    "active_brand": brand,
                    "brand_source": brand_source,
                    "llm_mail_events": candidate["llm_mail_events"],
                    "bot_context_visible_items": bot_context.get("summary", {}).get("visible_chunks", 0),
                    "shadow_enabled": bool(trace.get("enabled")),
                    "shadow_found": bool(trace.get("found")),
                    "shadow_warnings": list(trace.get("warnings") or ()),
                    "shadow_stats": dict(trace.get("stats") or {}),
                    "prompt_text": prompt_text,
                    "prompt_pii_findings": list(scan_bot_safe_context_pii(prompt_text)),
                    "prompt_has_service_id": bool(_SERVICE_ID_RE.search(prompt_text)),
                    "route_text_shadow_only": bool(trace.get("route_text_shadow_only")),
                    "safety": dict(trace.get("safety") or {}),
                }
            )

    _write_jsonl(out_jsonl, rows)
    summary = summarize_shadow_rows(rows)
    summary["db_path"] = str(db_path)
    summary["out_jsonl"] = str(out_jsonl)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary.get("safety_violations_total"):
        return 2
    return 0


def require_staging_db_path(db_path: Path) -> None:
    parts = set(db_path.parts)
    if ".codex_local" not in parts or "staging" not in parts:
        raise ValueError(f"shadow runner accepts only .codex_local/staging DB paths, got: {db_path}")


def _select_llm_customers(db_path: Path, *, tenant_id: str, limit: int) -> list[Mapping[str, Any]]:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA query_only=ON")
        rows = con.execute(
            """
            SELECT
              f.customer_id,
              COALESCE(NULLIF(f.customer_brand, ''), 'unknown') AS customer_brand,
              COUNT(*) AS llm_mail_events
            FROM a2v3_mail_event_facts f
            JOIN email_summary_cache_v1 s USING(message_sha256)
            WHERE f.tenant_id = ?
              AND s.source_kind = 'llm'
              AND f.identity_outcome = 'linked'
              AND f.customer_id IS NOT NULL
              AND f.customer_id <> ''
            GROUP BY f.customer_id, customer_brand
            ORDER BY f.customer_id
            LIMIT ?
            """,
            (tenant_id, limit),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        con.close()


def infer_shadow_brand(raw_brand: object, items: Sequence[Any]) -> tuple[str, str]:
    brand = str(raw_brand or "").strip().casefold()
    if brand in _KNOWN_BRANDS:
        return brand, "a2v3_customer_brand"
    brands: set[str] = set()
    for item in items:
        if not isinstance(item, Mapping):
            continue
        for tag in item.get("relevance_tags") or ():
            tag_text = str(tag or "").strip().casefold()
            if tag_text in _KNOWN_BRANDS:
                brands.add(tag_text)
    if len(brands) == 1:
        return next(iter(brands)), "single_bot_context_brand"
    return "unknown", "unresolved"


def summarize_shadow_rows(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    violations = [reason for row in rows for reason in shadow_safety_violations(row)]
    manual_flags = [reason for row in rows for reason in shadow_manual_review_flags(row)]
    return {
        "schema_version": "customer_memory_shadow_run_v1_2026_07_03",
        "total_customers": len(rows),
        "enabled": sum(1 for row in rows if row.get("shadow_enabled")),
        "found": sum(1 for row in rows if row.get("shadow_found")),
        "shadow_only": sum(1 for row in rows if row.get("route_text_shadow_only")),
        "prompt_pii_hits": sum(1 for row in rows if row.get("prompt_pii_findings")),
        "prompt_service_id_hits": sum(1 for row in rows if row.get("prompt_has_service_id")),
        "by_brand": _count_by(rows, "active_brand"),
        "by_brand_source": _count_by(rows, "brand_source"),
        "warnings": _count_warnings(rows),
        "safety_violations_total": len(violations),
        "safety_violations": _count_values(violations),
        "manual_review_flags_total": len(manual_flags),
        "manual_review_flags": _count_values(manual_flags),
    }


def shadow_safety_violations(row: Mapping[str, Any]) -> tuple[str, ...]:
    prompt_text = str(row.get("prompt_text") or "")
    active_brand = str(row.get("active_brand") or "").strip().casefold()
    reasons: list[str] = []
    if row.get("prompt_pii_findings"):
        reasons.append("prompt_pii")
    if row.get("prompt_has_service_id") or _SERVICE_ID_RE.search(prompt_text):
        reasons.append("prompt_service_id")
    has_foton = "Бренд: Фотон" in prompt_text
    has_unpk = "Бренд: УНПК" in prompt_text
    if has_foton and has_unpk:
        reasons.append("prompt_cross_brand")
    if active_brand == "foton" and has_unpk:
        reasons.append("prompt_brand_mismatch")
    if active_brand == "unpk" and has_foton:
        reasons.append("prompt_brand_mismatch")
    if not row.get("shadow_found") and "Безопасные bot_context-фрагменты:" in prompt_text:
        reasons.append("empty_shadow_contains_memory_items")
    if "db_path" in prompt_text or "record_json" in prompt_text or "customer:" in prompt_text:
        reasons.append("prompt_debug_or_raw_id")
    return tuple(reasons)


def shadow_manual_review_flags(row: Mapping[str, Any]) -> tuple[str, ...]:
    prompt_text = str(row.get("prompt_text") or "")
    if row.get("shadow_found") and _TEMPORAL_MARKER_RE.search(prompt_text):
        return ("temporal_marker_in_memory",)
    return ()


def _count_by(rows: Sequence[Mapping[str, Any]], key: str) -> Mapping[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _count_values(values: Sequence[str]) -> Mapping[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _count_warnings(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        warnings = row.get("shadow_warnings")
        if not isinstance(warnings, Sequence) or isinstance(warnings, (str, bytes, bytearray)):
            continue
        for warning in warnings:
            value = str(warning or "").strip()
            if value:
                counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
