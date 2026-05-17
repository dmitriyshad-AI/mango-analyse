#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from zipfile import ZipFile
from xml.etree import ElementTree

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mango_mvp.knowledge_base.fact_registry import classify_fact_types, fact_type_from_key
from scripts.build_full_kc_knowledge_base import (
    DYNAMIC_FACT_RE,
    HIGH_RISK_RE,
    build_approval_queue,
    build_chunks,
    build_fact_candidates,
    build_qa_records,
    build_script_records,
    build_source_record,
    clean_text,
    default_freshness_blocks,
    infer_brand,
    infer_source_role,
    infer_theme_ids,
    load_manager_patterns,
    sha256_text,
    split_text,
    write_csv,
    write_json,
    write_jsonl,
)


SCHEMA_VERSION = "kc_knowledge_snapshot_v1"
RELEASE_BUILDER_VERSION = "kc_final_release_builder_2026_05_17_v1"

DEFAULT_RUN_ID = "kb_release_20260517_v1"
DEFAULT_OUT_DIR = Path("product_data/knowledge_base/kb_release_20260517_v1")
DEFAULT_GOOGLE_EXPORT_DIR = Path("product_data/knowledge_base/full_kb_20260517_v1/source_exports")
DEFAULT_OLD_EXTRACT_DIR = Path(".codex_local/kc_source_extract_20260513")
DEFAULT_KC_DOCX = Path("База знаний КЦ.docx")
DEFAULT_MANAGER_PATTERNS = Path("product_data/knowledge_base/kb_night_20260517_v1/manager_answer_patterns.jsonl")
DEFAULT_MANAGER_SAMPLE = Path("product_data/knowledge_base/kb_night_20260517_v1/manager_answer_sample_300_500.jsonl")
DEFAULT_STRUCTURED_FACTS = Path(
    "product_data/knowledge_base/google_drive_structured_facts_20260517_v1/google_doc_structured_facts.jsonl"
)
DEFAULT_ANSWER_TEMPLATES = Path("product_data/question_catalog/answer_templates.csv")
DEFAULT_APPROVED_ANSWERS_DRAFT = Path("product_data/question_catalog/approved_question_answers_draft.csv")
DEFAULT_QUESTION_ITEMS = Path("product_data/question_catalog/customer_question_items.jsonl")
DEFAULT_FACT_SOURCE_REGISTRY = Path("product_data/question_catalog/current_fact_source_registry.json")

SITE_ROOTS = ("https://kmipt.ru/", "https://cdpofoton.ru/")
SITE_URL_RE = re.compile(r"<loc>(.*?)</loc>", re.I)
HTML_TAG_RE = re.compile(r"<[^>]+>")
SCRIPT_STYLE_RE = re.compile(r"<(script|style|noscript).*?</\1>", re.I | re.S)
WHITESPACE_RE = re.compile(r"\s+")
PRECISE_FACT_TYPES = {"price", "discount", "schedule", "documents", "payment_deadline", "payment_methods"}
INTERNAL_MARKERS = (
    "амo",
    "amo",
    "талланто",
    "tallanto",
    "crm",
    "регламент",
    "тактика",
    "внутрен",
    "объединение контактов",
)
LOW_VALUE_SITE_MARKERS = (
    "amocrm",
    "search",
    "reviews_add",
    "payment_success",
    "payment_error",
)
USEFUL_SITE_MARKERS = (
    "courses",
    "course",
    "contacts",
    "address",
    "payment",
    "requisites",
    "landing",
    "promotion",
    "teachers",
    "reviews",
    "parents",
    "news",
    "oge",
    "ege",
    "kanikuly",
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build final read-only KC knowledge release for Telegram pilot.")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--google-export-dir", type=Path, default=DEFAULT_GOOGLE_EXPORT_DIR)
    parser.add_argument("--old-extract-dir", type=Path, default=DEFAULT_OLD_EXTRACT_DIR)
    parser.add_argument("--kc-docx", type=Path, default=DEFAULT_KC_DOCX)
    parser.add_argument("--manager-patterns", type=Path, default=DEFAULT_MANAGER_PATTERNS)
    parser.add_argument("--manager-sample", type=Path, default=DEFAULT_MANAGER_SAMPLE)
    parser.add_argument("--structured-facts", type=Path, default=DEFAULT_STRUCTURED_FACTS)
    parser.add_argument("--answer-templates", type=Path, default=DEFAULT_ANSWER_TEMPLATES)
    parser.add_argument("--approved-answers-draft", type=Path, default=DEFAULT_APPROVED_ANSWERS_DRAFT)
    parser.add_argument("--question-items", type=Path, default=DEFAULT_QUESTION_ITEMS)
    parser.add_argument("--fact-source-registry", type=Path, default=DEFAULT_FACT_SOURCE_REGISTRY)
    parser.add_argument("--include-old-extract", action="store_true", default=True)
    parser.add_argument("--crawl-current-sites", action="store_true")
    parser.add_argument("--site-page-limit", type=int, default=35)
    args = parser.parse_args(argv)

    result = build_kc_final_release(
        run_id=args.run_id,
        out_dir=args.out_dir,
        google_export_dir=args.google_export_dir,
        old_extract_dir=args.old_extract_dir,
        kc_docx_path=args.kc_docx,
        manager_patterns_path=args.manager_patterns,
        manager_sample_path=args.manager_sample,
        structured_facts_path=args.structured_facts,
        answer_templates_path=args.answer_templates,
        approved_answers_draft_path=args.approved_answers_draft,
        question_items_path=args.question_items,
        fact_source_registry_path=args.fact_source_registry,
        include_old_extract=args.include_old_extract,
        crawl_current_sites=args.crawl_current_sites,
        site_page_limit=max(0, args.site_page_limit),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def build_kc_final_release(
    *,
    run_id: str,
    out_dir: Path,
    google_export_dir: Path,
    old_extract_dir: Path,
    kc_docx_path: Path,
    manager_patterns_path: Path,
    manager_sample_path: Path,
    structured_facts_path: Path,
    answer_templates_path: Path,
    approved_answers_draft_path: Path,
    question_items_path: Path,
    fact_source_registry_path: Path,
    include_old_extract: bool = True,
    crawl_current_sites: bool = False,
    site_page_limit: int = 35,
) -> Mapping[str, Any]:
    out_root = guard_output_dir(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    source_export_root = out_root / "source_exports"
    source_export_root.mkdir(parents=True, exist_ok=True)

    staged_sources = stage_text_sources(
        source_export_root=source_export_root,
        google_export_dir=google_export_dir,
        old_extract_dir=old_extract_dir,
        kc_docx_path=kc_docx_path,
        include_old_extract=include_old_extract,
        crawl_current_sites=crawl_current_sites,
        site_page_limit=site_page_limit,
    )

    sources: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    facts: list[dict[str, Any]] = []
    scripts: list[dict[str, Any]] = []
    qa_pairs: list[dict[str, Any]] = []
    seen_chunk_texts: set[str] = set()
    seen_fact_keys: set[str] = set()

    for source_path, source_origin in staged_sources:
        source = normalize_source_for_release(build_source_record(source_path), source_path=source_path, origin=source_origin)
        sources.append(source)
        if source["processing_status"] != "processed":
            continue
        text = source_path.read_text(encoding="utf-8-sig", errors="replace")
        source_chunks = enrich_chunks_for_release(
            build_chunks(text, source=source, max_chunks=chunk_limit(source, text)),
            source=source,
            seen_texts=seen_chunk_texts,
        )
        chunks.extend(source_chunks)
        for fact in build_fact_candidates(text, source=source):
            fact_key = f"{fact.get('fact_id')}:{fact.get('source_span')}"
            if fact_key in seen_fact_keys:
                continue
            seen_fact_keys.add(fact_key)
            facts.append(normalize_fact_for_release(fact, source=source))
        scripts.extend(build_script_records(source_chunks, source=source))
        qa_pairs.extend(build_qa_records(text, source=source))

    structured_facts = load_structured_fact_candidates(structured_facts_path, seen_fact_keys=seen_fact_keys)
    facts.extend(structured_facts)

    answer_templates = load_answer_templates(answer_templates_path, limit=500)
    approved_answers_summary = summarize_approved_answers(approved_answers_draft_path)
    manager_patterns = normalize_manager_patterns(load_manager_patterns(manager_patterns_path))
    manager_sample_summary = summarize_manager_sample(manager_sample_path)
    question_catalog_summary = summarize_question_items(question_items_path)
    fact_source_registry_summary = summarize_fact_source_registry(fact_source_registry_path)

    chunks.extend(answer_template_chunks(answer_templates, seen_texts=seen_chunk_texts))
    approval_queue = build_approval_queue(facts=facts, sources=sources)
    snapshot = build_release_snapshot(
        run_id=run_id,
        sources=sources,
        chunks=chunks,
        facts=facts,
        scripts=scripts,
        qa_pairs=qa_pairs,
        manager_patterns=manager_patterns,
        answer_templates=answer_templates,
        approved_answers_summary=approved_answers_summary,
        manager_sample_summary=manager_sample_summary,
        question_catalog_summary=question_catalog_summary,
        fact_source_registry_summary=fact_source_registry_summary,
        crawl_current_sites=crawl_current_sites,
    )
    quality = build_quality_report(snapshot, approval_queue=approval_queue)
    write_release_outputs(
        out_root,
        snapshot=snapshot,
        sources=sources,
        chunks=chunks,
        facts=facts,
        scripts=scripts,
        qa_pairs=qa_pairs,
        manager_patterns=manager_patterns,
        answer_templates=answer_templates,
        approval_queue=approval_queue,
        quality=quality,
    )
    return {
        "out_dir": str(out_root),
        "snapshot_path": str(out_root / f"kc_snapshot_{run_id}.json"),
        "sources_total": snapshot["summary"]["sources_total"],
        "processed_sources": snapshot["summary"]["processed_sources"],
        "chunks_total": snapshot["summary"]["chunks_total"],
        "facts_total": snapshot["summary"]["facts_total"],
        "usable_for_precise_answer": snapshot["summary"]["usable_for_precise_answer"],
        "manager_patterns_total": snapshot["summary"]["manager_patterns_total"],
        "answer_templates_total": snapshot["summary"]["answer_templates_total"],
        "quality_passed": quality["quality_passed"],
    }


def stage_text_sources(
    *,
    source_export_root: Path,
    google_export_dir: Path,
    old_extract_dir: Path,
    kc_docx_path: Path,
    include_old_extract: bool,
    crawl_current_sites: bool,
    site_page_limit: int,
) -> list[tuple[Path, str]]:
    staged: list[tuple[Path, str]] = []
    staged.extend(copy_text_files(google_export_dir, source_export_root / "google_drive_docs", origin="google_drive_doc_export"))

    if kc_docx_path.exists():
        docx_text = extract_docx_text(kc_docx_path)
        docx_out = source_export_root / "local_docx" / "kc_knowledge_base_docx.txt"
        docx_out.parent.mkdir(parents=True, exist_ok=True)
        docx_out.write_text(docx_text, encoding="utf-8")
        staged.append((docx_out, "local_docx"))

    if include_old_extract and old_extract_dir.exists():
        staged.extend(copy_text_files(old_extract_dir / "texts", source_export_root / "local_drive_extracts", origin="local_drive_extract"))
        staged.extend(
            copy_text_files(
                old_extract_dir / "site_texts",
                source_export_root / "site_extracts_20260513",
                origin="website_extract_20260513",
                filter_func=useful_site_file,
            )
        )

    if crawl_current_sites:
        staged.extend(crawl_current_site_sources(source_export_root / "site_current", page_limit=site_page_limit))
    return dedupe_staged_sources(staged)


def copy_text_files(
    src_dir: Path,
    dst_dir: Path,
    *,
    origin: str,
    filter_func: Any | None = None,
) -> list[tuple[Path, str]]:
    if not src_dir.exists():
        return []
    dst_dir.mkdir(parents=True, exist_ok=True)
    staged: list[tuple[Path, str]] = []
    for src in sorted(src_dir.glob("*.txt")):
        if filter_func is not None and not filter_func(src):
            continue
        target = dst_dir / unique_text_filename(src.name)
        target.write_text(src.read_text(encoding="utf-8-sig", errors="replace"), encoding="utf-8")
        staged.append((target, origin))
    return staged


def crawl_current_site_sources(dst_dir: Path, *, page_limit: int) -> list[tuple[Path, str]]:
    if page_limit <= 0:
        return []
    dst_dir.mkdir(parents=True, exist_ok=True)
    staged: list[tuple[Path, str]] = []
    for root_url in SITE_ROOTS:
        urls = discover_site_urls(root_url, page_limit=page_limit)
        for index, url in enumerate(urls, start=1):
            try:
                html_text = fetch_url(url, timeout=12, max_bytes=1_000_000)
            except OSError:
                continue
            text = html_to_text(html_text)
            if len(text) < 200:
                continue
            parsed = urlparse(url)
            filename = f"{parsed.netloc}_{index:03d}_{safe_filename(parsed.path or 'index')}.txt"
            path = dst_dir / filename
            path.write_text(f"URL: {url}\n\n{text}", encoding="utf-8")
            staged.append((path, "website_current_crawl"))
    return staged


def discover_site_urls(root_url: str, *, page_limit: int) -> list[str]:
    urls: list[str] = [root_url]
    sitemap_index_url = root_url.rstrip("/") + "/sitemap.xml"
    try:
        sitemap_index = fetch_url(sitemap_index_url, timeout=10, max_bytes=500_000)
    except OSError:
        return urls
    sitemap_urls = SITE_URL_RE.findall(sitemap_index)
    page_urls: list[str] = []
    for sitemap_url in sitemap_urls[:8]:
        try:
            sitemap_text = fetch_url(sitemap_url, timeout=10, max_bytes=800_000)
        except OSError:
            continue
        page_urls.extend(SITE_URL_RE.findall(sitemap_text))
    for url in page_urls:
        lowered = url.casefold()
        if any(marker in lowered for marker in USEFUL_SITE_MARKERS) and not any(
            marker in lowered for marker in LOW_VALUE_SITE_MARKERS
        ):
            urls.append(url)
        if len(urls) >= page_limit:
            break
    return list(dict.fromkeys(urls))[:page_limit]


def fetch_url(url: str, *, timeout: int, max_bytes: int) -> str:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0 Codex knowledge audit"})
    with urlopen(request, timeout=timeout) as response:
        return response.read(max_bytes).decode(response.headers.get_content_charset() or "utf-8", errors="replace")


def html_to_text(value: str) -> str:
    value = SCRIPT_STYLE_RE.sub(" ", value)
    value = re.sub(r"<!--.*?-->", " ", value, flags=re.S)
    value = HTML_TAG_RE.sub("\n", value)
    value = html.unescape(value)
    lines = [clean_text(line) for line in value.splitlines()]
    return "\n".join(line for line in lines if line)


def extract_docx_text(path: Path) -> str:
    try:
        with ZipFile(path) as archive:
            xml = archive.read("word/document.xml")
    except Exception as exc:  # pragma: no cover - defensive fallback for broken user files
        return f"DOCX extraction failed for {path}: {type(exc).__name__}: {exc}"
    root = ElementTree.fromstring(xml)
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs: list[str] = []
    for paragraph in root.findall(".//w:p", namespace):
        parts: list[str] = []
        for node in paragraph.findall(".//w:t", namespace):
            if node.text:
                parts.append(node.text)
        text = clean_text("".join(parts))
        if text:
            paragraphs.append(text)
    return "\n\n".join(paragraphs)


def useful_site_file(path: Path) -> bool:
    name = path.name.casefold()
    if any(marker in name for marker in LOW_VALUE_SITE_MARKERS):
        return False
    return any(marker in name for marker in USEFUL_SITE_MARKERS) or name.endswith("_001_index.txt")


def dedupe_staged_sources(rows: Sequence[tuple[Path, str]]) -> list[tuple[Path, str]]:
    seen: set[str] = set()
    result: list[tuple[Path, str]] = []
    for path, origin in rows:
        try:
            text_hash = sha256_text(path.read_text(encoding="utf-8-sig", errors="replace"))
        except OSError:
            text_hash = sha256_text(str(path))
        if text_hash in seen:
            continue
        seen.add(text_hash)
        result.append((path, origin))
    return result


def normalize_source_for_release(source: Mapping[str, Any], *, source_path: Path, origin: str) -> dict[str, Any]:
    payload = dict(source)
    text = source_path.read_text(encoding="utf-8-sig", errors="replace")
    title = clean_title(payload.get("title") or source_path.stem)
    role = infer_release_source_role(source_path, text, origin=origin, fallback=str(payload.get("source_role") or "program"))
    kind = source_kind_for_origin(origin)
    payload.update(
        {
            "schema_version": "kc_source_record_v1",
            "title": title,
            "source_kind": kind,
            "source_type": kind,
            "source_role": role,
            "origin": origin,
            "path": str(source_path),
            "read_status": "read" if payload.get("processing_status") == "processed" else "manual_review_required",
            "freshness_status": release_freshness_for_source(source_path, text, role=role, origin=origin),
            "approval_status": "not_approved",
            "usable_for_precise_answer": False,
            "requires_manager_confirmation": True,
            "forbidden_for_client": role in {"internal_rule", "regulation"} or bool(payload.get("contains_high_risk")),
            "source_sha256": sha256_text(text),
            "sha256": sha256_text(text),
            "brand": payload.get("brand") or infer_brand(source_path.name, text),
            "fact_types": list(classify_fact_types(f"{title} {text[:3000]}")),
            "limitation_reason": source_limitation_reason(source_path, text, role=role),
        }
    )
    return payload


def infer_release_source_role(path: Path, text: str, *, origin: str, fallback: str) -> str:
    name = path.name.casefold()
    value = f"{path.name} {text[:3000]}".casefold()
    if origin.startswith("website"):
        if any(marker in name for marker in ("contacts", "address", "requisites", "payment")):
            return "organization_fact"
        return "program"
    if origin == "local_docx":
        return "manager_rule"
    if any(marker in value for marker in INTERNAL_MARKERS):
        return "internal_rule"
    inferred = infer_source_role(path.name, text)
    return inferred or fallback


def source_kind_for_origin(origin: str) -> str:
    if origin == "google_drive_doc_export":
        return "google_drive_doc_export"
    if origin == "local_docx":
        return "local_docx"
    if origin.startswith("website"):
        return "website_extract"
    return "local_extract"


def release_freshness_for_source(path: Path, text: str, *, role: str, origin: str) -> str:
    value = f"{path.name} {text[:500]}".casefold()
    if role in {"internal_rule", "regulation"}:
        return "internal_only"
    if "2026" in value or "26_27" in value or "26-27" in value or "2026_2027" in value:
        return "needs_manager_confirmation"
    if any(marker in value for marker in ("2023", "2024", "2025", "24-25", "25-26")):
        return "stale_or_conflicting"
    if origin.startswith("website"):
        return "needs_manager_confirmation"
    return "unknown"


def source_limitation_reason(path: Path, text: str, *, role: str) -> str:
    if len(text.strip()) < 80:
        return "source_too_short_manual_review_required"
    if role in {"internal_rule", "regulation"}:
        return "internal_source_not_for_client"
    if DYNAMIC_FACT_RE.search(text):
        return "contains_prices_dates_or_terms_needs_rop_approval"
    return "general_information_can_support_manager_drafts"


def enrich_chunks_for_release(
    chunks: Sequence[Mapping[str, Any]],
    *,
    source: Mapping[str, Any],
    seen_texts: set[str],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for chunk in chunks:
        text = clean_text(chunk.get("text") or "")
        if len(text) < 40:
            continue
        text_hash = sha256_text(text)
        if text_hash in seen_texts:
            continue
        seen_texts.add(text_hash)
        dynamic = bool(DYNAMIC_FACT_RE.search(text))
        internal = source.get("source_role") in {"internal_rule", "regulation"} or any(
            marker in text.casefold() for marker in ("заходим в амо", "талланто", "карточк")
        )
        fact_types = list(classify_fact_types(f"{chunk.get('title', '')} {text}"))
        payload = dict(chunk)
        payload.update(
            {
                "schema_version": "kc_chunk_record_v1",
                "source_kind": source.get("source_kind"),
                "source_role": source.get("source_role"),
                "brand": source.get("brand") or infer_brand(str(chunk.get("title") or ""), text),
                "fact_types": fact_types,
                "freshness_status": "internal_only" if internal else ("needs_manager_confirmation" if dynamic else "unknown"),
                "requires_manager_confirmation": bool(dynamic or internal or HIGH_RISK_RE.search(text)),
                "forbidden_for_client": bool(internal),
                "usable_for_precise_answer": False,
                "bot_permission": "internal_only" if internal else "draft_for_manager",
                "retrieval_keywords": sorted(extract_keywords(f"{chunk.get('title', '')} {text}"))[:40],
                "product_tags": product_tags(f"{chunk.get('title', '')} {text}"),
                "search_text": clean_text(f"{chunk.get('title', '')} {text}")[:2000],
                "metadata": {
                    **dict(chunk.get("metadata") if isinstance(chunk.get("metadata"), Mapping) else {}),
                    "source_title": source.get("title"),
                    "source_role": source.get("source_role"),
                    "source_kind": source.get("source_kind"),
                    "contains_dynamic_facts": dynamic,
                    "contains_high_risk": bool(HIGH_RISK_RE.search(text)),
                    "channel_use": "manager_draft_only" if internal or dynamic else "manager_draft_and_safe_general_context",
                    "retrieval_keywords": sorted(extract_keywords(f"{chunk.get('title', '')} {text}"))[:40],
                    "product_tags": product_tags(f"{chunk.get('title', '')} {text}"),
                    "search_text": clean_text(f"{chunk.get('title', '')} {text}")[:2000],
                    "bot_permission": "internal_only" if internal else "draft_for_manager",
                },
            }
        )
        result.append(payload)
    return result


def normalize_fact_for_release(fact: Mapping[str, Any], *, source: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(fact)
    fact_type = str(payload.get("fact_type") or fact_type_from_key(str(payload.get("fact_key") or "")))
    payload.update(
        {
            "schema_version": "kc_fact_record_v1",
            "fact_key": payload.get("fact_key") or fact_key_for_type(fact_type),
            "fact_type": fact_type,
            "source_kind": source.get("source_kind"),
            "source_role": source.get("source_role"),
            "brand": source.get("brand"),
            "freshness_status": "needs_manager_confirmation",
            "verification_status": "extracted_unverified",
            "verified_at": None,
            "verified_by": None,
            "approval_status": "not_approved",
            "usable_for_precise_answer": False,
            "requires_manager_confirmation": True,
            "forbidden_for_client": True,
            "bot_permission": "manager_only",
            "related_theme_ids": payload.get("related_theme_ids") or infer_theme_ids(
                f"{payload.get('manager_text', '')} {payload.get('short_fact', '')}"
            ),
            "risk_notes": "Точный факт заблокирован до проверки РОПом и владельцем источника.",
        }
    )
    return payload


def load_structured_fact_candidates(path: Path, *, seen_fact_keys: set[str]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    result: list[dict[str, Any]] = []
    for item in read_jsonl(path):
        fact_id = str(item.get("fact_id") or f"fact:structured:{sha256_text(item)}")
        if fact_id in seen_fact_keys:
            continue
        seen_fact_keys.add(fact_id)
        payload = dict(item)
        payload.update(
            {
                "schema_version": "kc_fact_record_v1",
                "fact_id": fact_id,
                "approval_status": "not_approved",
                "freshness_status": "needs_manager_confirmation",
                "verification_status": "extracted_unverified",
                "usable_for_precise_answer": False,
                "requires_manager_confirmation": True,
                "forbidden_for_client": True,
                "bot_permission": "manager_only",
                "risk_notes": "Структурированный кандидат факта из Google Docs. До проверки РОПом клиенту не использовать.",
            }
        )
        result.append(payload)
    return result


def normalize_manager_patterns(patterns: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for pattern in patterns:
        payload = dict(pattern)
        payload.update(
            {
                "schema_version": "manager_answer_pattern_v1",
                "usable_as_fact": False,
                "bot_permission": "draft_for_manager",
                "fact_safety_note": "Исторический ответ менеджера: использовать как прием и стиль, не как факт.",
            }
        )
        result.append(payload)
    return result


def load_answer_templates(path: Path, *, limit: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = read_csv(path)
    result: list[dict[str, Any]] = []
    for row in rows[:limit]:
        text = clean_text(row.get("template_text") or "")
        if not text:
            continue
        result.append(
            {
                "schema_version": "answer_template_record_v1",
                "answer_template_id": row.get("answer_template_id") or f"answer_template:{sha256_text(text)[:16]}",
                "question_class_id": row.get("question_class_id", ""),
                "template_text": text,
                "required_fact_keys": text_list(row.get("required_fact_keys")),
                "approval_status": row.get("approval_status") or "not_approved",
                "bot_permission": row.get("bot_permission_if_facts_fresh") or "manager_only",
                "fallback_when_fact_missing": row.get("fallback_when_fact_missing") or "",
                "usable_as_fact": False,
                "bot_permission_runtime": "manager_only",
            }
        )
    return result


def answer_template_chunks(templates: Sequence[Mapping[str, Any]], *, seen_texts: set[str]) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for row in templates[:250]:
        text = clean_text(row.get("template_text") or "")
        if not text:
            continue
        text_hash = sha256_text(text)
        if text_hash in seen_texts:
            continue
        seen_texts.add(text_hash)
        chunks.append(
            {
                "schema_version": "kc_chunk_record_v1",
                "chunk_id": f"kc_chunk:answer_template:{text_hash[:12]}",
                "source_id": "source:question_catalog_answer_templates",
                "source_kind": "question_catalog",
                "source_role": "answer_template",
                "title": "Шаблон ответа из каталога вопросов",
                "text": text[:900],
                "fact_types": [fact_type_from_key(key) for key in row.get("required_fact_keys", [])] or ["manager_instruction"],
                "freshness_status": "needs_manager_confirmation",
                "requires_manager_confirmation": True,
                "forbidden_for_client": False,
                "usable_for_precise_answer": False,
                "bot_permission": "draft_for_manager",
                "record_type": "answer_template",
                "retrieval_keywords": sorted(extract_keywords(text))[:40],
                "product_tags": product_tags(text),
                "search_text": text[:2000],
                "metadata": {
                    "source_title": "answer_templates.csv",
                    "source_role": "answer_template",
                    "contains_dynamic_facts": bool(DYNAMIC_FACT_RE.search(text)),
                    "channel_use": "manager_draft_only",
                    "retrieval_keywords": sorted(extract_keywords(text))[:40],
                    "product_tags": product_tags(text),
                    "search_text": text[:2000],
                    "bot_permission": "draft_for_manager",
                },
            }
        )
    return chunks


def build_release_snapshot(
    *,
    run_id: str,
    sources: Sequence[Mapping[str, Any]],
    chunks: Sequence[Mapping[str, Any]],
    facts: Sequence[Mapping[str, Any]],
    scripts: Sequence[Mapping[str, Any]],
    qa_pairs: Sequence[Mapping[str, Any]],
    manager_patterns: Sequence[Mapping[str, Any]],
    answer_templates: Sequence[Mapping[str, Any]],
    approved_answers_summary: Mapping[str, Any],
    manager_sample_summary: Mapping[str, Any],
    question_catalog_summary: Mapping[str, Any],
    fact_source_registry_summary: Mapping[str, Any],
    crawl_current_sites: bool,
) -> dict[str, Any]:
    sources_by_kind = Counter(str(source.get("source_kind") or "unknown") for source in sources)
    sources_by_role = Counter(str(source.get("source_role") or "unknown") for source in sources)
    chunks_by_permission = Counter(str(chunk.get("bot_permission") or "unknown") for chunk in chunks)
    facts_by_type = Counter(str(fact.get("fact_type") or "unknown") for fact in facts)
    summary = {
        "sources_total": len(sources),
        "processed_sources": sum(1 for source in sources if source.get("processing_status") == "processed"),
        "sources_by_kind": dict(sources_by_kind),
        "sources_by_role": dict(sources_by_role),
        "chunks_total": len(chunks),
        "chunks_by_permission": dict(chunks_by_permission),
        "facts_total": len(facts),
        "facts_by_type": dict(facts_by_type),
        "usable_for_precise_answer": sum(1 for fact in facts if fact.get("usable_for_precise_answer") is True),
        "facts_requiring_manager_confirmation": sum(1 for fact in facts if fact.get("requires_manager_confirmation") is True),
        "conversation_scripts_total": len(scripts),
        "qa_pairs_total": len(qa_pairs),
        "manager_patterns_total": len(manager_patterns),
        "answer_templates_total": len(answer_templates),
        "historical_question_items_total": question_catalog_summary.get("items_total", 0),
        "historical_question_items_by_channel": question_catalog_summary.get("by_channel", {}),
        "crawl_current_sites": crawl_current_sites,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "builder_version": RELEASE_BUILDER_VERSION,
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "read_only_manager_draft_release",
        "metadata": {
            "purpose": "Финальный на текущий момент релиз базы знаний для черновиков Telegram-бота и дальнейшей РОП-проверки.",
            "precise_facts_policy": "blocked_until_rop_approval",
            "autonomous_client_answer_policy": "only_after_fact_approval_and_route_gate",
        },
        "sources": list(sources),
        "source_inventory": list(sources),
        "facts": list(facts),
        "chunks": list(chunks),
        "knowledge_chunks": list(chunks),
        "conversation_scripts": list(scripts),
        "qa_pairs": list(qa_pairs),
        "manager_answer_patterns": list(manager_patterns),
        "answer_templates": list(answer_templates),
        "approved_answers_summary": dict(approved_answers_summary),
        "manager_answer_sample_summary": dict(manager_sample_summary),
        "question_catalog_summary": dict(question_catalog_summary),
        "fact_source_registry_summary": dict(fact_source_registry_summary),
        "freshness_blocks": default_freshness_blocks()
        + [
            {
                "fact_key": "programs.current",
                "fact_type": "program",
                "reason": "program_descriptions_may_differ_by_year_or_site",
                "blocks_precise_answer": False,
                "safe_instruction": "Можно использовать как общий контекст, но не обещать конкретную группу, место или расписание.",
            },
            {
                "fact_key": "manager_answer_patterns.current",
                "fact_type": "manager_instruction",
                "reason": "historical_answers_are_style_only",
                "blocks_precise_answer": True,
                "safe_instruction": "Исторические ответы менеджеров использовать только как стиль и прием.",
            },
        ],
        "summary": summary,
        "safety": {
            "client_send": False,
            "crm_write": False,
            "tallanto_write": False,
            "google_drive_write": False,
            "stable_runtime_write": False,
            "run_asr": False,
            "run_resolve_analyze": False,
            "send_full_docx_to_prompt": False,
            "manager_approval_required": True,
            "precise_facts_require_rop_approval": True,
            "historical_manager_answers_are_style_only": True,
            "prices_schedule_documents_blocked_by_default": True,
        },
    }


def build_quality_report(snapshot: Mapping[str, Any], *, approval_queue: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary = snapshot.get("summary") or {}
    facts = list(snapshot.get("facts") or [])
    chunks = list(snapshot.get("chunks") or [])
    manager_patterns = list(snapshot.get("manager_answer_patterns") or [])
    checks = {
        "schema_is_bot_compatible": snapshot.get("schema_version") == SCHEMA_VERSION,
        "has_google_drive_sources": (summary.get("sources_by_kind") or {}).get("google_drive_doc_export", 0) >= 10,
        "has_local_docx_source": (summary.get("sources_by_kind") or {}).get("local_docx", 0) >= 1,
        "has_website_sources": (summary.get("sources_by_kind") or {}).get("website_extract", 0) >= 20,
        "has_manager_answer_patterns": len(manager_patterns) >= 30,
        "all_precise_facts_blocked_until_approval": all(not fact.get("usable_for_precise_answer") for fact in facts),
        "manager_patterns_not_facts": all(not pattern.get("usable_as_fact") for pattern in manager_patterns),
        "dynamic_chunks_not_precise": all(
            not (chunk.get("usable_for_precise_answer") and chunk.get("metadata", {}).get("contains_dynamic_facts"))
            for chunk in chunks
        ),
        "approval_queue_exists": len(approval_queue) >= 50,
        "safety_contract_read_only": all(
            snapshot.get("safety", {}).get(key) is False
            for key in ("client_send", "crm_write", "tallanto_write", "google_drive_write", "stable_runtime_write")
        ),
    }
    blocking_failures = [name for name, passed in checks.items() if not passed]
    return {
        "schema_version": "kc_final_release_quality_report_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "quality_passed": not blocking_failures,
        "checks": checks,
        "blocking_failures": blocking_failures,
        "interpretation": (
            "Релиз готов для черновиков менеджеру и внутренней работы. Для автономных точных ответов нужны утвержденные факты."
        ),
        "manual_review_required": {
            "approval_queue_items": len(approval_queue),
            "precise_facts_to_approve": sum(1 for fact in facts if fact.get("requires_manager_confirmation")),
        },
    }


def write_release_outputs(
    out_dir: Path,
    *,
    snapshot: Mapping[str, Any],
    sources: Sequence[Mapping[str, Any]],
    chunks: Sequence[Mapping[str, Any]],
    facts: Sequence[Mapping[str, Any]],
    scripts: Sequence[Mapping[str, Any]],
    qa_pairs: Sequence[Mapping[str, Any]],
    manager_patterns: Sequence[Mapping[str, Any]],
    answer_templates: Sequence[Mapping[str, Any]],
    approval_queue: Sequence[Mapping[str, Any]],
    quality: Mapping[str, Any],
) -> None:
    run_id = str(snapshot.get("run_id") or DEFAULT_RUN_ID)
    write_json(out_dir / f"kc_snapshot_{run_id}.json", snapshot)
    write_json(out_dir / "quality_report.json", quality)
    write_json(out_dir / "source_inventory.json", list(sources))
    write_csv(out_dir / "source_inventory.csv", sources)
    write_jsonl(out_dir / "knowledge_chunks.jsonl", chunks)
    write_csv(out_dir / "knowledge_chunks.csv", chunks)
    write_jsonl(out_dir / "facts.jsonl", facts)
    write_csv(out_dir / "facts.csv", facts)
    write_jsonl(out_dir / "conversation_scripts.jsonl", scripts)
    write_csv(out_dir / "conversation_scripts.csv", scripts)
    write_jsonl(out_dir / "qa_pairs.jsonl", qa_pairs)
    write_csv(out_dir / "qa_pairs.csv", qa_pairs)
    write_jsonl(out_dir / "manager_answer_patterns.jsonl", manager_patterns)
    write_csv(out_dir / "manager_answer_patterns.csv", manager_patterns)
    write_jsonl(out_dir / "answer_templates.jsonl", answer_templates)
    write_csv(out_dir / "answer_templates.csv", answer_templates)
    write_csv(out_dir / "approval_queue_for_rop.csv", approval_queue)
    (out_dir / "QUALITY_CRITERIA.md").write_text(render_quality_criteria(), encoding="utf-8")
    (out_dir / "QUALITY_REPORT.md").write_text(render_quality_report_md(snapshot, quality), encoding="utf-8")
    (out_dir / "README.md").write_text(render_readme(snapshot, quality), encoding="utf-8")
    (out_dir / "BOT_USAGE_CONTRACT.md").write_text(render_bot_usage_contract(snapshot), encoding="utf-8")
    (out_dir / "COVERAGE_AND_GAPS.md").write_text(render_coverage_and_gaps(snapshot, quality), encoding="utf-8")


def summarize_approved_answers(path: Path) -> Mapping[str, Any]:
    rows = read_csv(path) if path.exists() else []
    return {
        "path": str(path),
        "rows_total": len(rows),
        "approved_for_bot_yes": sum(1 for row in rows if str(row.get("approved_for_bot") or "").casefold() == "yes"),
        "can_autosend_yes": sum(1 for row in rows if str(row.get("can_autosend") or "").casefold() == "yes"),
        "runtime_not_allowed": sum(1 for row in rows if str(row.get("runtime_bot_permission") or "") == "not_allowed"),
    }


def summarize_manager_sample(path: Path) -> Mapping[str, Any]:
    rows = read_jsonl(path) if path.exists() else []
    return {
        "path": str(path),
        "sample_rows": len(rows),
        "usable_as_fact": sum(1 for row in rows if row.get("usable_as_fact") is True),
        "answer_classification": dict(Counter(str(row.get("answer_classification") or "unknown") for row in rows)),
        "by_channel": dict(Counter(str(row.get("channel") or "unknown") for row in rows)),
    }


def summarize_question_items(path: Path) -> Mapping[str, Any]:
    rows = read_jsonl(path) if path.exists() else []
    return {
        "path": str(path),
        "items_total": len(rows),
        "by_channel": dict(Counter(str(row.get("source_channel") or "unknown") for row in rows)),
        "requires_dynamic_facts": sum(1 for row in rows if row.get("requires_dynamic_facts") is True),
    }


def summarize_fact_source_registry(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    sources = loaded.get("sources") if isinstance(loaded, Mapping) else []
    return {
        "path": str(path),
        "exists": True,
        "sources_total": len(sources or []),
        "usable_for_bot": sum(1 for row in sources or [] if row.get("usable_for_bot") is True),
        "manual_review_required": sum(1 for row in sources or [] if row.get("approval_status") == "manual_review_required"),
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            loaded = json.loads(line)
            if isinstance(loaded, Mapping):
                rows.append(dict(loaded))
    return rows


def chunk_limit(source: Mapping[str, Any], text: str) -> int:
    role = str(source.get("source_role") or "")
    if source.get("origin", "").startswith("website"):
        return 12
    if role in {"conversation_script", "qa_pair", "manager_rule"}:
        return 120
    if len(text) > 1_500_000:
        return 160
    return 80


def fact_key_for_type(fact_type: str) -> str:
    return {
        "price": "prices.current",
        "discount": "discounts.current",
        "schedule": "schedule.current",
        "documents": "documents.current",
        "payment_methods": "payment_methods.current",
        "payment_deadline": "payment_deadlines.current",
        "program": "programs.current",
    }.get(fact_type, f"{fact_type or 'knowledge'}.current")


def extract_keywords(text: str) -> set[str]:
    normalized = text.casefold().replace("ё", "е")
    words = set(re.findall(r"[0-9a-zа-я]{4,}", normalized))
    stop = {"котор", "этот", "также", "если", "можно", "нужно", "будет", "после", "перед", "ученик", "родител"}
    return {word for word in words if word not in stop}


def product_tags(text: str) -> list[str]:
    value = text.casefold().replace("ё", "е")
    tags: list[str] = []
    markers = {
        "ЛВШ": ("лвш", "летняя выездная"),
        "летняя школа": ("летняя школа", "летний лагерь", "каникул"),
        "ЕГЭ": ("егэ",),
        "ОГЭ": ("огэ",),
        "математика": ("математ",),
        "физика": ("физик",),
        "информатика": ("информат", "программир"),
        "русский язык": ("русск",),
        "онлайн": ("онлайн",),
        "очно": ("очно", "филиал"),
        "пробное": ("пробн",),
    }
    for tag, needles in markers.items():
        if any(needle in value for needle in needles):
            tags.append(tag)
    return tags


def text_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [clean_text(item) for item in value if clean_text(item)]
    return [clean_text(item) for item in re.split(r"[|,;]", str(value or "")) if clean_text(item)]


def clean_title(value: Any) -> str:
    return clean_text(value).replace(".txt", "")[:180] or "Источник базы знаний"


def unique_text_filename(name: str) -> str:
    cleaned = safe_filename(name)
    return cleaned if cleaned.endswith(".txt") else f"{cleaned}.txt"


def safe_filename(value: Any) -> str:
    text = clean_text(value).replace("/", "_").replace(":", "_")
    text = re.sub(r"[^0-9A-Za-zА-Яа-яЁё._-]+", "_", text).strip("._")
    return text[:180] or "source"


def render_quality_criteria() -> str:
    return "\n".join(
        [
            "# Критерии качества финальной базы знаний",
            "",
            "1. Все источники имеют происхождение, путь, хеш и статус чтения.",
            "2. Google Docs, локальный docx, сайты и ответы менеджеров разведены по слоям.",
            "3. Цены, скидки, расписание, документы, сроки, возвраты и юридические темы не разрешены для точного ответа без РОП-проверки.",
            "4. Исторические ответы менеджеров используются только как стиль и прием, не как источник фактов.",
            "5. Снимок совместим с Telegram-пилотом: есть `sources`, `facts`, `chunks`, `knowledge_chunks`, `manager_answer_patterns`, `freshness_blocks`.",
            "6. База полезна для черновиков: содержит общие программы, FAQ, скрипты, правила, контакты, сайты и шаблоны.",
            "7. Все сомнительные факты попадают в очередь проверки РОПом.",
            "8. Релиз не пишет в CRM, Tallanto, Google Drive, stable_runtime и не отправляет сообщения клиентам.",
        ]
    ) + "\n"


def render_quality_report_md(snapshot: Mapping[str, Any], quality: Mapping[str, Any]) -> str:
    checks = quality.get("checks") or {}
    lines = [
        "# Quality report",
        "",
        f"- quality_passed: {quality.get('quality_passed')}",
        f"- run_id: {snapshot.get('run_id')}",
        "",
        "## Проверки",
        "",
    ]
    for name, passed in checks.items():
        lines.append(f"- {'OK' if passed else 'FAIL'} {name}")
    if quality.get("blocking_failures"):
        lines.extend(["", "## Блокеры", ""])
        lines.extend(f"- {item}" for item in quality["blocking_failures"])
    return "\n".join(lines) + "\n"


def render_readme(snapshot: Mapping[str, Any], quality: Mapping[str, Any]) -> str:
    summary = snapshot.get("summary") or {}
    return "\n".join(
        [
            "# KC final knowledge release",
            "",
            "Это финальная на текущий момент база знаний для Telegram-пилота и дальнейшей работы с ботом.",
            "",
            "## Что собрано",
            "",
            f"- Источников: {summary.get('sources_total')}",
            f"- Прочитано источников: {summary.get('processed_sources')}",
            f"- Фрагментов знаний: {summary.get('chunks_total')}",
            f"- Кандидатов фактов: {summary.get('facts_total')}",
            f"- Приемов менеджеров: {summary.get('manager_patterns_total')}",
            f"- Шаблонов ответов: {summary.get('answer_templates_total')}",
            f"- Исторических вопросов учтено: {summary.get('historical_question_items_total')}",
            "",
            "## Главный статус",
            "",
            f"- Проверка качества: {quality.get('quality_passed')}",
            f"- Точных фактов, разрешенных для автономного ответа: {summary.get('usable_for_precise_answer')}",
            "",
            "База готова для содержательных черновиков менеджеру. Для автономных точных ответов нужны утвержденные факты РОПа.",
        ]
    ) + "\n"


def render_bot_usage_contract(snapshot: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Правила использования базы ботом",
            "",
            "## Можно",
            "",
            "- Подбирать общие фрагменты про программы, форматы, направления и подход к обучению.",
            "- Использовать исторические ответы менеджеров как стиль: мягкость, уточняющие вопросы, структура ответа.",
            "- Готовить черновик для Насти с пометкой, что нужно проверить.",
            "",
            "## Нельзя без отдельного утверждения",
            "",
            "- Называть точную цену, скидку, срок акции, расписание, дату, место в группе.",
            "- Обещать возврат, перерасчет, юридическое решение, скидку или индивидуальное условие.",
            "- Использовать исторический ответ менеджера как доказанный факт.",
            "- Отправлять клиенту служебные сведения из AMO, Tallanto, внутренних регламентов.",
            "",
            "## Маршрут",
            "",
            "- Если нужен точный факт и он не `fresh_verified`, маршрут должен быть `manager_only` или черновик без обещаний.",
            "- Если тема опасная: возврат, маткапитал, налог, договор, жалоба, юридический вопрос, маршрут только через менеджера.",
        ]
    ) + "\n"


def render_coverage_and_gaps(snapshot: Mapping[str, Any], quality: Mapping[str, Any]) -> str:
    summary = snapshot.get("summary") or {}
    lines = [
        "# Покрытие и пробелы",
        "",
        "## Покрытие",
        "",
        f"- Google/Drive/doc источники: {(summary.get('sources_by_kind') or {}).get('google_drive_doc_export', 0)}",
        f"- Локальный docx: {(summary.get('sources_by_kind') or {}).get('local_docx', 0)}",
        f"- Сайтовые источники: {(summary.get('sources_by_kind') or {}).get('website_extract', 0)}",
        f"- Исторические вопросы: {summary.get('historical_question_items_total')}",
        f"- По каналам: {summary.get('historical_question_items_by_channel')}",
        "",
        "## Пробелы",
        "",
        "- Нет утвержденных точных фактов для автономных ответов.",
        "- РОПу нужно пройти очередь `approval_queue_for_rop.csv`.",
        "- Сайтовые данные полезны как карта программ, но цены и даты всё равно проверять.",
        "- Исторические ответы менеджеров полезны для стиля, но не заменяют базу фактов.",
    ]
    return "\n".join(lines) + "\n"


def guard_output_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if any(part.casefold() == "stable_runtime" for part in resolved.parts):
        raise ValueError(f"Refusing to write final KC release under stable_runtime: {resolved}")
    return resolved


if __name__ == "__main__":
    raise SystemExit(main())
