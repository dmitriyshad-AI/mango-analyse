#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mango_mvp.knowledge_base.fact_registry import classify_fact_types
from scripts.extract_kc_google_doc_facts import extract_candidates_from_text


SCHEMA_VERSION = "full_kc_knowledge_base_candidate_v1"
DEFAULT_INPUT_DIR = Path("product_data/knowledge_base/full_kb_20260517_v1/source_exports")
DEFAULT_OUT_DIR = Path("product_data/knowledge_base/full_kb_20260517_v1")
DEFAULT_OLD_EXTRACT_DIR = Path(".codex_local/kc_source_extract_20260513")
DEFAULT_MANAGER_PATTERNS = Path("product_data/knowledge_base/kb_night_20260517_v1/manager_answer_patterns.jsonl")

MAX_CHUNKS_PER_SOURCE = 90
MAX_CHUNK_CHARS = 900
MIN_CHUNK_CHARS = 80
MAX_SOURCE_BYTES_FOR_FULL_CHUNKS = 1_500_000

SOURCE_CATALOG: Mapping[str, Mapping[str, Any]] = {
    "kc_knowledge_base.txt": {
        "title": "База знаний КЦ",
        "url": "https://docs.google.com/document/d/1bMhN0DtqNK8Z2XdwGMci2lAv0CtSYQ4QGb1Hr4dQ9Oo",
        "drive_file_id": "1bMhN0DtqNK8Z2XdwGMci2lAv0CtSYQ4QGb1Hr4dQ9Oo",
        "source_updated_at": "2026-04-23T13:25:08.711Z",
        "source_role": "manager_rule",
        "fact_types": ["manager_instruction", "restriction", "price", "schedule", "documents", "program"],
    },
    "unpk_prices_2026_2027.txt": {
        "title": "УНПК Стоимость обучения и порядок оплаты на 26/2027 уч г от 16.03.26",
        "url": "https://docs.google.com/document/d/12lsPcxpkP8Kf7y3dsLraYA_o9voXkq3Wg-KQEOF7YzI",
        "drive_file_id": "12lsPcxpkP8Kf7y3dsLraYA_o9voXkq3Wg-KQEOF7YzI",
        "source_updated_at": "2026-03-17T08:07:44.383Z",
        "source_role": "precise_fact_candidate",
        "brand": "unpk",
        "fact_types": ["price", "discount", "payment_methods", "documents"],
    },
    "foton_prices_2026_2027.txt": {
        "title": "ФОТОН Стоимость обучения и порядок оплаты на 26/27 уч от 16.03.26",
        "url": "https://docs.google.com/document/d/1k0hzS8cZjD2NXeE5mcjlMTEOtTVINlTstZpWOUI1gSc",
        "drive_file_id": "1k0hzS8cZjD2NXeE5mcjlMTEOtTVINlTstZpWOUI1gSc",
        "source_updated_at": "2026-03-17T08:03:02.111Z",
        "source_role": "precise_fact_candidate",
        "brand": "foton",
        "fact_types": ["price", "discount", "payment_methods", "documents"],
    },
    "call_scripts.txt": {
        "title": "Обзвоны - скрипты",
        "url": "https://docs.google.com/document/d/1be22cPbevAeyaFSqWpcNTuCpt5EVobax6hRl9PEnro8",
        "drive_file_id": "1be22cPbevAeyaFSqWpcNTuCpt5EVobax6hRl9PEnro8",
        "source_role": "conversation_script",
        "fact_types": ["manager_instruction", "restriction", "price", "schedule", "program"],
    },
    "whatsapp_tg_quick_replies.txt": {
        "title": "Скрипт для быстрых ответов ватс и тг",
        "url": "https://docs.google.com/document/d/1z6wAVkKFf-ck-g07Sj3yNIsh3Z-qoeIDT898899NeIk",
        "drive_file_id": "1z6wAVkKFf-ck-g07Sj3yNIsh3Z-qoeIDT898899NeIk",
        "source_role": "qa_pair",
        "fact_types": ["manager_instruction", "documents", "payment_methods", "program"],
    },
    "foton_questions.txt": {
        "title": "Вопросы по ФОТОНу и с чем их едят",
        "url": "https://docs.google.com/document/d/1LVKZB1RAsTK-uPnODEVXXUorw4GSEDLFOM07lRoybwM",
        "drive_file_id": "1LVKZB1RAsTK-uPnODEVXXUorw4GSEDLFOM07lRoybwM",
        "source_role": "qa_pair",
        "fact_types": ["manager_instruction", "program", "documents", "schedule"],
    },
    "unpk_to_foton_script.txt": {
        "title": "Скрипт перевода с УНПК на ФОТОН",
        "url": "https://docs.google.com/document/d/10YNVkxm49zlOirXUdewzwcF6TBiNH8MPRO0txwDot_s",
        "drive_file_id": "10YNVkxm49zlOirXUdewzwcF6TBiNH8MPRO0txwDot_s",
        "source_role": "conversation_script",
        "fact_types": ["manager_instruction", "program", "restriction"],
    },
    "unpk_tactics_2025_04_16.txt": {
        "title": "Тактика УНПК МФТИ от 16.04.2025",
        "url": "https://docs.google.com/document/d/10JMCLgVmrOfeiWk3RdI7yJIdjuENV4TOyB5uLyKToaU",
        "drive_file_id": "10JMCLgVmrOfeiWk3RdI7yJIdjuENV4TOyB5uLyKToaU",
        "source_role": "manager_rule",
        "fact_types": ["manager_instruction", "restriction", "program"],
    },
    "regulations.txt": {
        "title": "Регламенты",
        "url": "https://docs.google.com/document/d/1YbZGvxpvX-ITUrG_kRZUmQYkVpRPQXqVLBI2K6LrF4A",
        "drive_file_id": "1YbZGvxpvX-ITUrG_kRZUmQYkVpRPQXqVLBI2K6LrF4A",
        "source_role": "regulation",
        "fact_types": ["manager_instruction", "restriction"],
    },
    "courses_pk_mfti_unpk.txt": {
        "title": "Курсы при ПК МФТИ + УНПК",
        "url": "https://docs.google.com/document/d/1-4yMUiO4rCFxH0woZRBwvycqacFB7gcyehv5Nr9nhds",
        "drive_file_id": "1-4yMUiO4rCFxH0woZRBwvycqacFB7gcyehv5Nr9nhds",
        "source_role": "program",
        "fact_types": ["program", "price", "schedule", "documents"],
    },
    "lvsh_2026_unpk.txt": {
        "title": "ЛВШ 2026 УНПК",
        "url": "https://docs.google.com/document/d/1ABm0eHf5r3hfa3jxREUIMIQcUAYA38qJYB3L2LYLRiw",
        "drive_file_id": "1ABm0eHf5r3hfa3jxREUIMIQcUAYA38qJYB3L2LYLRiw",
        "source_role": "program",
        "brand": "unpk",
        "fact_types": ["program", "price", "schedule", "documents"],
    },
    "lvsh_2026_foton.txt": {
        "title": "ЛВШ 2026 Фотон",
        "url": "https://docs.google.com/document/d/1M_zlGTX8t1V8pJBRyR1vJsDgmWxEiW2gCOxdg3alppw",
        "drive_file_id": "1M_zlGTX8t1V8pJBRyR1vJsDgmWxEiW2gCOxdg3alppw",
        "source_role": "program",
        "brand": "foton",
        "fact_types": ["program", "price", "schedule", "documents"],
    },
}

DYNAMIC_FACT_RE = re.compile(
    r"\b\d[\d\s\u00a0]{2,8}\s*(?:руб\.?|₽)?\b|\b\d{1,2}\s*%|\b\d{1,2}[./-]\d{1,2}(?:[./-]\d{2,4})?\b|"
    r"\b\d{1,2}\s+(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\b|"
    r"\b\d{1,2}[:.]\d{2}\b",
    re.I,
)
HTML_RE = re.compile(r"<!doctype html|<html[\s>]", re.I)
QUESTION_RE = re.compile(r"\?|^(?:вопрос|клиент|родитель|если спрашивают|если спросили)\b", re.I)
ANSWER_RE = re.compile(r"^(?:ответ|говорим|пишем|можно ответить|шаблон|скрипт)\b", re.I)
SCRIPT_RE = re.compile(r"\[[^\]]{2,60}\]|«[^»]{20,}»|подскажите|добрый день|спасибо|передам|уточн", re.I)
HIGH_RISK_RE = re.compile(r"возврат|маткап|материнск|налог|юрид|договор|претенз|жалоб|суд|персональн|паспорт", re.I)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build full KC knowledge base candidate from exported docs.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--old-extract-dir", type=Path, default=DEFAULT_OLD_EXTRACT_DIR)
    parser.add_argument("--manager-patterns", type=Path, default=DEFAULT_MANAGER_PATTERNS)
    parser.add_argument("--run-id", default="full_kb_20260517_v1")
    parser.add_argument("--include-old-extract", action="store_true")
    args = parser.parse_args(argv)
    result = build_full_kc_knowledge_base(
        input_dir=args.input_dir,
        out_dir=args.out_dir,
        old_extract_dir=args.old_extract_dir,
        manager_patterns_path=args.manager_patterns,
        run_id=args.run_id,
        include_old_extract=args.include_old_extract,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def build_full_kc_knowledge_base(
    *,
    input_dir: Path,
    out_dir: Path,
    old_extract_dir: Path = DEFAULT_OLD_EXTRACT_DIR,
    manager_patterns_path: Path = DEFAULT_MANAGER_PATTERNS,
    run_id: str = "full_kb_20260517_v1",
    include_old_extract: bool = False,
) -> Mapping[str, Any]:
    out_root = guard_output_dir(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    source_paths = sorted(input_dir.glob("*.txt"))
    if include_old_extract and old_extract_dir.exists():
        source_paths.extend(select_old_extract_sources(old_extract_dir))

    sources: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    facts: list[dict[str, Any]] = []
    scripts: list[dict[str, Any]] = []
    qa_pairs: list[dict[str, Any]] = []
    approval_queue: list[dict[str, Any]] = []

    for path in source_paths:
        source = build_source_record(path)
        sources.append(source)
        if source["processing_status"] != "processed":
            continue
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        source_chunks = build_chunks(text, source=source, max_chunks=chunk_limit_for_source(path, text))
        chunks.extend(source_chunks)
        facts.extend(build_fact_candidates(text, source=source))
        scripts.extend(build_script_records(source_chunks, source=source))
        qa_pairs.extend(build_qa_records(text, source=source))

    manager_patterns = load_manager_patterns(manager_patterns_path)
    approval_queue.extend(build_approval_queue(facts=facts, sources=sources))
    snapshot = build_snapshot(
        run_id=run_id,
        sources=sources,
        chunks=chunks,
        facts=facts,
        scripts=scripts,
        qa_pairs=qa_pairs,
        manager_patterns=manager_patterns,
    )
    write_outputs(
        out_root,
        snapshot=snapshot,
        sources=sources,
        chunks=chunks,
        facts=facts,
        scripts=scripts,
        qa_pairs=qa_pairs,
        manager_patterns=manager_patterns,
        approval_queue=approval_queue,
    )
    return {
        "out_dir": str(out_root),
        "snapshot_path": str(out_root / "bot_knowledge_snapshot_candidate.json"),
        "sources_total": len(sources),
        "processed_sources": sum(1 for source in sources if source["processing_status"] == "processed"),
        "knowledge_chunks": len(chunks),
        "fact_candidates": len(facts),
        "conversation_scripts": len(scripts),
        "qa_pairs": len(qa_pairs),
        "manager_patterns": len(manager_patterns),
        "approval_queue": len(approval_queue),
        "usable_for_precise_answer": sum(1 for fact in facts if fact["usable_for_precise_answer"]),
    }


def build_source_record(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    catalog = dict(SOURCE_CATALOG.get(path.name) or {})
    source_id = catalog.get("source_id") or f"source:gdrive_export:{safe_id(path.stem)}:{sha256_text(path.name)[:10]}"
    status = "processed"
    reason = "text_export_read"
    if HTML_RE.search(text[:1000]):
        status = "export_failed"
        reason = "html_instead_of_text_export"
    elif len(text.strip()) < 50:
        status = "manual_review_required"
        reason = "export_too_short"
    fact_types = tuple(catalog.get("fact_types") or classify_fact_types(f"{path.name} {text[:3000]}"))
    return {
        "source_id": source_id,
        "title": catalog.get("title") or title_from_filename(path.name),
        "source_type": "google_drive_doc_export",
        "source_role": catalog.get("source_role") or infer_source_role(path.name, text),
        "path": str(path),
        "url": catalog.get("url", ""),
        "google_drive_file_id": catalog.get("drive_file_id", ""),
        "source_updated_at": catalog.get("source_updated_at", ""),
        "processing_status": status,
        "status_reason": reason,
        "read_succeeded": status == "processed",
        "fact_types": list(fact_types),
        "brand": catalog.get("brand") or infer_brand(path.name, text),
        "freshness_status": "needs_manager_confirmation",
        "approval_status": "not_approved",
        "usable_for_precise_answer": False,
        "contains_dynamic_facts": bool(DYNAMIC_FACT_RE.search(text)),
        "contains_high_risk": bool(HIGH_RISK_RE.search(text)),
        "sha256_text": sha256_text(text),
        "notes": "Источник прочитан read-only. Точные факты заблокированы до проверки РОПом.",
    }


def build_chunks(text: str, *, source: Mapping[str, Any], max_chunks: int) -> list[dict[str, Any]]:
    paragraphs = split_text(text)
    chunks: list[dict[str, Any]] = []
    title = source["title"]
    current_title = title
    buffer: list[str] = []
    start_index = 0

    def flush(end_index: int) -> None:
        nonlocal buffer, start_index
        raw = clean_text(" ".join(buffer))
        if len(raw) < MIN_CHUNK_CHARS:
            buffer = []
            return
        text_value = raw[:MAX_CHUNK_CHARS].rstrip()
        fact_types = tuple(classify_fact_types(f"{current_title} {text_value}"))
        dynamic = bool(DYNAMIC_FACT_RE.search(text_value))
        forbidden = source.get("contains_high_risk") and source.get("source_role") in {"regulation", "precise_fact_candidate"}
        chunks.append(
            {
                "chunk_id": f"kc_chunk:{safe_id(source['source_id'])}:{sha256_text(current_title + text_value)[:12]}",
                "source_id": source["source_id"],
                "title": current_title[:140],
                "text": text_value,
                "fact_types": list(fact_types),
                "freshness_status": "needs_manager_confirmation" if dynamic else "unknown",
                "bot_permission": "draft_for_manager",
                "forbidden_for_client": bool(forbidden),
                "requires_manager_confirmation": bool(dynamic or forbidden),
                "usable_for_precise_answer": False,
                "record_type": source.get("source_role") or "knowledge_chunk",
                "source_span": {"paragraph_start": start_index + 1, "paragraph_end": end_index},
                "metadata": {
                    "source_title": source["title"],
                    "source_role": source.get("source_role"),
                    "contains_dynamic_facts": dynamic,
                    "contains_high_risk": bool(HIGH_RISK_RE.search(text_value)),
                },
            }
        )
        buffer = []

    for index, paragraph in enumerate(paragraphs):
        if is_heading(paragraph):
            flush(index)
            current_title = paragraph[:140]
            start_index = index
            continue
        if not buffer:
            start_index = index
        buffer.append(paragraph)
        if len(clean_text(" ".join(buffer))) >= MAX_CHUNK_CHARS:
            flush(index)
        if len(chunks) >= max_chunks:
            break
    flush(len(paragraphs))
    return chunks[:max_chunks]


def build_fact_candidates(text: str, *, source: Mapping[str, Any]) -> list[dict[str, Any]]:
    source_meta = {
        "brand": source.get("brand") or "unknown",
        "source_id": source["source_id"],
        "source_title": source["title"],
        "source_url": source.get("url") or "",
        "source_updated_at": source.get("source_updated_at") or "",
    }
    raw_facts = extract_candidates_from_text(text, source_meta=source_meta, source_sha256=source["sha256_text"])
    facts: list[dict[str, Any]] = []
    for fact in raw_facts:
        payload = dict(fact)
        payload.update(
            {
                "record_type": "precise_fact_candidate",
                "approval_status": "not_approved",
                "usable_for_precise_answer": False,
                "requires_manager_confirmation": True,
                "forbidden_for_client": True,
                "bot_permission": "manager_only",
                "risk_notes": "Кандидат точного факта. Клиенту не использовать до проверки РОПом.",
            }
        )
        facts.append(payload)
    return facts


def build_script_records(chunks: Sequence[Mapping[str, Any]], *, source: Mapping[str, Any]) -> list[dict[str, Any]]:
    if source.get("source_role") not in {"conversation_script", "qa_pair", "manager_rule"}:
        return []
    records: list[dict[str, Any]] = []
    for chunk in chunks:
        text = str(chunk.get("text") or "")
        if not SCRIPT_RE.search(text):
            continue
        dynamic = bool(DYNAMIC_FACT_RE.search(text))
        records.append(
            {
                "script_id": f"script:{safe_id(source['source_id'])}:{sha256_text(text)[:12]}",
                "source_id": source["source_id"],
                "source_title": source["title"],
                "title": chunk.get("title") or source["title"],
                "manager_safe_text": text,
                "client_safe_template": "" if dynamic else text,
                "record_type": "conversation_script",
                "bot_permission": "draft_for_manager",
                "usable_as_fact": False,
                "requires_manager_confirmation": dynamic,
                "forbidden_fact_types": ["price", "discount", "schedule", "documents"] if dynamic else [],
                "related_theme_ids": infer_theme_ids(text),
                "risk_notes": "Скрипт можно использовать как стиль и ход разговора, но не как проверенный факт.",
            }
        )
    return records


def build_qa_records(text: str, *, source: Mapping[str, Any]) -> list[dict[str, Any]]:
    paragraphs = split_text(text)
    records: list[dict[str, Any]] = []
    for index, paragraph in enumerate(paragraphs[:-1]):
        if not QUESTION_RE.search(paragraph):
            continue
        answer_parts: list[str] = []
        for next_paragraph in paragraphs[index + 1 : index + 4]:
            if QUESTION_RE.search(next_paragraph) and answer_parts:
                break
            answer_parts.append(next_paragraph)
            if len(clean_text(" ".join(answer_parts))) > 700:
                break
        answer = clean_text(" ".join(answer_parts))[:800]
        if len(answer) < 30:
            continue
        dynamic = bool(DYNAMIC_FACT_RE.search(answer))
        records.append(
            {
                "qa_id": f"qa:{safe_id(source['source_id'])}:{sha256_text(paragraph + answer)[:12]}",
                "source_id": source["source_id"],
                "source_title": source["title"],
                "question_text": paragraph[:500],
                "answer_text": answer,
                "record_type": "qa_pair",
                "bot_permission": "draft_for_manager",
                "usable_as_fact": False,
                "requires_manager_confirmation": dynamic,
                "forbidden_for_client": dynamic or bool(HIGH_RISK_RE.search(answer)),
                "related_theme_ids": infer_theme_ids(f"{paragraph} {answer}"),
                "risk_notes": "Пара вопрос-ответ является примером. Точные данные проверять отдельно.",
            }
        )
    return dedupe_records(records, "qa_id")[:300]


def load_manager_patterns(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    patterns: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            item = json.loads(line)
            text = clean_text(
                item.get("safe_pattern_template")
                or item.get("safe_pattern_summary")
                or item.get("manager_safe_text")
                or ""
            )
            if not text:
                continue
            patterns.append(
                {
                    "pattern_id": item.get("pattern_id") or f"pattern:{sha256_text(text)[:12]}",
                    "topic": item.get("topic", ""),
                    "risk_group": item.get("risk_group", ""),
                    "technique": item.get("technique", ""),
                    "safe_pattern": text[:700],
                    "pattern_summary": clean_text(item.get("safe_pattern_summary") or text)[:500],
                    "related_theme_ids": infer_theme_ids(f"{item.get('topic', '')} {text}"),
                    "usable_as_fact": False,
                    "bot_permission": "draft_for_manager",
                    "fact_safety_note": "Исторический ответ менеджера: использовать как прием, не как факт.",
                }
            )
    return patterns[:120]


def build_approval_queue(*, facts: Sequence[Mapping[str, Any]], sources: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    queue: list[dict[str, Any]] = []
    for fact in facts:
        priority = approval_priority(fact)
        queue.append(
            {
                "priority": priority,
                "approval_item_id": fact.get("fact_id", ""),
                "item_type": fact.get("fact_type", "fact"),
                "source_title": fact.get("source_title", ""),
                "manager_text": fact.get("manager_text", "")[:900],
                "suggested_decision": "review_before_client_use",
                "rop_question": approval_question_for_fact(fact),
                "bot_permission_after_approval": "can_answer_precise_fact" if priority in {"P0", "P1"} else "draft_for_manager",
                "risk_notes": fact.get("risk_notes", ""),
            }
        )
    for source in sources:
        if source["processing_status"] != "processed":
            queue.append(
                {
                    "priority": "P1",
                    "approval_item_id": source["source_id"],
                    "item_type": "source_problem",
                    "source_title": source["title"],
                    "manager_text": source["status_reason"],
                    "suggested_decision": "fix_or_exclude_source",
                    "rop_question": "Нужно ли чинить доступ к этому источнику или исключить его из базы?",
                    "bot_permission_after_approval": "not_applicable",
                    "risk_notes": "Источник не прочитан корректно.",
                }
            )
    priority_order = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    return sorted(queue, key=lambda row: (priority_order.get(row["priority"], 9), row["source_title"]))[:500]


def build_snapshot(
    *,
    run_id: str,
    sources: Sequence[Mapping[str, Any]],
    chunks: Sequence[Mapping[str, Any]],
    facts: Sequence[Mapping[str, Any]],
    scripts: Sequence[Mapping[str, Any]],
    qa_pairs: Sequence[Mapping[str, Any]],
    manager_patterns: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    summary = {
        "sources_total": len(sources),
        "processed_sources": sum(1 for source in sources if source["processing_status"] == "processed"),
        "sources_by_role": dict(Counter(str(source.get("source_role") or "unknown") for source in sources)),
        "chunks_total": len(chunks),
        "fact_candidates_total": len(facts),
        "scripts_total": len(scripts),
        "qa_pairs_total": len(qa_pairs),
        "manager_patterns_total": len(manager_patterns),
        "usable_for_precise_answer": sum(1 for fact in facts if fact.get("usable_for_precise_answer") is True),
        "approval_required": len(facts),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "read_only_candidate",
        "metadata": {
            "purpose": "Candidate knowledge base for manager drafts and future bot approval flow",
            "precise_facts_policy": "blocked_until_rop_approval",
        },
        "sources": list(sources),
        "source_inventory": list(sources),
        "chunks": list(chunks),
        "knowledge_chunks": list(chunks),
        "facts": list(facts),
        "conversation_scripts": list(scripts),
        "qa_pairs": list(qa_pairs),
        "manager_answer_patterns": list(manager_patterns),
        "freshness_blocks": default_freshness_blocks(),
        "summary": summary,
        "safety": {
            "google_drive_write": False,
            "crm_write": False,
            "tallanto_write": False,
            "client_send": False,
            "stable_runtime_write": False,
            "precise_facts_require_rop_approval": True,
            "historical_manager_answers_are_style_only": True,
            "prices_schedule_documents_blocked_by_default": True,
        },
    }


def write_outputs(
    out_dir: Path,
    *,
    snapshot: Mapping[str, Any],
    sources: Sequence[Mapping[str, Any]],
    chunks: Sequence[Mapping[str, Any]],
    facts: Sequence[Mapping[str, Any]],
    scripts: Sequence[Mapping[str, Any]],
    qa_pairs: Sequence[Mapping[str, Any]],
    manager_patterns: Sequence[Mapping[str, Any]],
    approval_queue: Sequence[Mapping[str, Any]],
) -> None:
    write_json(out_dir / "bot_knowledge_snapshot_candidate.json", snapshot)
    write_json(out_dir / "source_inventory.json", list(sources))
    write_csv(out_dir / "source_inventory.csv", sources)
    write_jsonl(out_dir / "knowledge_chunks.jsonl", chunks)
    write_csv(out_dir / "knowledge_chunks.csv", chunks)
    write_jsonl(out_dir / "fact_candidates.jsonl", facts)
    write_csv(out_dir / "fact_candidates.csv", facts)
    write_jsonl(out_dir / "conversation_scripts.jsonl", scripts)
    write_csv(out_dir / "conversation_scripts.csv", scripts)
    write_jsonl(out_dir / "qa_pairs.jsonl", qa_pairs)
    write_csv(out_dir / "qa_pairs.csv", qa_pairs)
    write_jsonl(out_dir / "manager_answer_patterns.jsonl", manager_patterns)
    write_csv(out_dir / "manager_answer_patterns.csv", manager_patterns)
    write_csv(out_dir / "approval_queue_for_rop.csv", approval_queue)
    (out_dir / "README.md").write_text(render_readme(snapshot), encoding="utf-8")
    (out_dir / "coverage_and_gaps.md").write_text(render_coverage_and_gaps(snapshot, sources=sources), encoding="utf-8")


def select_old_extract_sources(root: Path) -> list[Path]:
    selected: list[Path] = []
    text_dir = root / "texts"
    site_dir = root / "site_texts"
    if text_dir.exists():
        selected.extend(
            path
            for path in text_dir.glob("*.txt")
            if any(marker in path.name.casefold() for marker in ("скрипт", "рассроч", "возврат", "лиценз", "стоимость"))
        )
    if site_dir.exists():
        selected.extend(
            path
            for path in site_dir.glob("*.txt")
            if any(marker in path.name.casefold() for marker in ("payment", "contacts", "requisites", "courses", "promotion"))
        )
    return sorted(selected)[:80]


def split_text(text: str) -> list[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    raw_parts = re.split(r"\n{2,}|(?<=\.)\s+(?=[А-ЯA-ZЁ][^.!?]{20,})", normalized)
    parts = [clean_text(part) for part in raw_parts if clean_text(part)]
    if len(parts) < 3:
        parts = [clean_text(line) for line in normalized.splitlines() if clean_text(line)]
    return parts


def is_heading(text: str) -> bool:
    value = clean_text(text)
    if len(value) > 120:
        return False
    if value.endswith(":"):
        return True
    if re.match(r"^(?:часть|пункт|этап|вкладка|\d+[.)])\s+", value, re.I):
        return True
    if value.isupper() and len(value) > 4:
        return True
    return False


def chunk_limit_for_source(path: Path, text: str) -> int:
    if len(text.encode("utf-8", errors="ignore")) > MAX_SOURCE_BYTES_FOR_FULL_CHUNKS:
        return 120
    if "site_texts" in path.parts:
        return 20
    return MAX_CHUNKS_PER_SOURCE


def infer_source_role(filename: str, text: str) -> str:
    value = f"{filename} {text[:3000]}".casefold()
    if any(marker in value for marker in ("скрипт", "звонк", "обзвон", "whatsapp", "tg", "ватс")):
        return "conversation_script"
    if any(marker in value for marker in ("вопрос", "ответ", "faq")):
        return "qa_pair"
    if any(marker in value for marker in ("регламент", "тактика", "правил", "нельзя", "обязательно")):
        return "manager_rule"
    if any(marker in value for marker in ("стоим", "цена", "скид", "оплат")):
        return "precise_fact_candidate"
    return "program"


def infer_brand(filename: str, text: str) -> str:
    value = f"{filename} {text[:1000]}".casefold()
    if "фотон" in value or "foton" in value:
        return "foton"
    if "унпк" in value or "мфти" in value or "unpk" in value:
        return "unpk"
    return "mixed"


def infer_theme_ids(text: str) -> list[str]:
    value = text.casefold()
    themes: list[str] = []
    if re.search(r"цен|стоим|прайс|руб", value):
        themes.append("theme:001_pricing")
    if re.search(r"оплат|сбп|ссылк[ау] на оплат|реквизит", value):
        themes.append("theme:002_payment_method")
    if re.search(r"скид|акци|промокод", value):
        themes.append("theme:005_discounts")
    if re.search(r"распис|время|суббот|воскрес|занят", value):
        themes.append("theme:013_schedule")
    if re.search(r"договор|документ|справ|чек|квитанц", value):
        themes.append("theme:011_documents")
    if re.search(r"возврат|растор", value):
        themes.append("theme:009_refund")
    if re.search(r"маткап|материн", value):
        themes.append("theme:007_matkap_payment")
    if re.search(r"налог|вычет", value):
        themes.append("theme:008_tax_deduction")
    if re.search(r"пробн", value):
        themes.append("theme:023_trial_class")
    if re.search(r"программ|курс|предмет|интенсив|лвш|летн", value):
        themes.append("theme:016_program_content")
    return themes or ["service:S3_other_or_low_confidence"]


def approval_priority(fact: Mapping[str, Any]) -> str:
    fact_type = str(fact.get("fact_type") or "")
    if fact_type in {"price", "discount", "payment_deadline", "schedule"}:
        return "P0"
    if fact_type in {"documents"}:
        return "P1"
    return "P2"


def approval_question_for_fact(fact: Mapping[str, Any]) -> str:
    fact_type = str(fact.get("fact_type") or "")
    if fact_type == "price":
        return "Можно ли использовать эту цену в ответе клиенту, и при каких условиях?"
    if fact_type == "discount":
        return "Можно ли говорить эту скидку клиенту, кому она доступна и до какого срока?"
    if fact_type == "schedule":
        return "Можно ли говорить эту дату/время клиенту или только просить менеджера уточнить?"
    if fact_type == "documents":
        return "Можно ли давать эту формулировку клиенту или нужен менеджер?"
    return "Можно ли использовать этот факт в черновике или клиентском ответе?"


def default_freshness_blocks() -> list[dict[str, Any]]:
    return [
        {
            "fact_key": "prices.current",
            "fact_type": "price",
            "reason": "facts_extracted_but_not_rop_approved",
            "blocks_precise_answer": True,
            "safe_instruction": "Не называть точную цену без утвержденного факта.",
        },
        {
            "fact_key": "discounts.current",
            "fact_type": "discount",
            "reason": "discounts_not_rop_approved",
            "blocks_precise_answer": True,
            "safe_instruction": "Не обещать скидку, процент или срок акции без утверждения.",
        },
        {
            "fact_key": "schedule.current",
            "fact_type": "schedule",
            "reason": "schedule_not_rop_approved",
            "blocks_precise_answer": True,
            "safe_instruction": "Не обещать точное расписание без проверки.",
        },
        {
            "fact_key": "documents.current",
            "fact_type": "documents",
            "reason": "documents_need_manager_review",
            "blocks_precise_answer": True,
            "safe_instruction": "Документы, возвраты, налоговые и юридические темы только через менеджера.",
        },
    ]


def render_readme(snapshot: Mapping[str, Any]) -> str:
    summary = snapshot.get("summary") or {}
    return "\n".join(
        [
            "# Full KC knowledge base candidate",
            "",
            "Это черновой релиз базы знаний для будущего бота и черновиков менеджеру.",
            "",
            "## Что внутри",
            "",
            f"- Источников: {summary.get('sources_total')}",
            f"- Прочитано источников: {summary.get('processed_sources')}",
            f"- Фрагментов знаний: {summary.get('chunks_total')}",
            f"- Кандидатов точных фактов: {summary.get('fact_candidates_total')}",
            f"- Скриптов общения: {summary.get('scripts_total')}",
            f"- Пар вопрос-ответ: {summary.get('qa_pairs_total')}",
            f"- Приемов менеджеров: {summary.get('manager_patterns_total')}",
            "",
            "## Важное ограничение",
            "",
            "Точные цены, скидки, сроки, расписание, документы и юридические формулировки пока не разрешены для прямого ответа клиенту.",
            "Они попали в очередь проверки РОПом. До утверждения бот может использовать базу только для черновиков менеджеру и безопасных общих формулировок.",
        ]
    ) + "\n"


def render_coverage_and_gaps(snapshot: Mapping[str, Any], *, sources: Sequence[Mapping[str, Any]]) -> str:
    summary = snapshot.get("summary") or {}
    roles = summary.get("sources_by_role") or {}
    failed = [source for source in sources if source.get("processing_status") != "processed"]
    lines = [
        "# Coverage and gaps",
        "",
        "## Покрытие по типам источников",
        "",
    ]
    for role, count in sorted(dict(roles).items()):
        lines.append(f"- {role}: {count}")
    lines.extend(
        [
            "",
            "## Главные пробелы",
            "",
            "- Нет ни одного точного факта, разрешенного для автономного ответа клиенту.",
            "- Цены, скидки, расписание и документы требуют проверки РОПом.",
            "- Исторические ответы менеджеров используются только как стиль и приемы, не как источник истины.",
            "- Live Telegram-пилоту еще нужно подключить этот snapshot как read-only источник.",
        ]
    )
    if failed:
        lines.extend(["", "## Источники с проблемой чтения", ""])
        for source in failed:
            lines.append(f"- {source.get('title')}: {source.get('status_reason')}")
    return "\n".join(lines) + "\n"


def guard_output_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if any(part.casefold() == "stable_runtime" for part in resolved.parts):
        raise ValueError(f"Refusing to write knowledge base under stable_runtime: {resolved}")
    return resolved


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    if not fieldnames:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(flatten_for_csv(row, fieldnames=fieldnames))


def flatten_for_csv(row: Mapping[str, Any], *, fieldnames: Sequence[str]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key in fieldnames:
        value = row.get(key, "")
        if isinstance(value, (list, tuple)):
            flat[key] = "|".join(str(item) for item in value)
        elif isinstance(value, Mapping):
            flat[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
        elif value is True:
            flat[key] = "true"
        elif value is False:
            flat[key] = "false"
        elif value is None:
            flat[key] = ""
        else:
            flat[key] = value
    return flat


def dedupe_records(rows: Sequence[Mapping[str, Any]], key: str) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for row in rows:
        value = str(row.get(key) or "")
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(dict(row))
    return result


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\ufeff", "").replace("\xa0", " ").split())


def title_from_filename(filename: str) -> str:
    return Path(filename).stem.replace("_", " ").strip()


def safe_id(value: Any) -> str:
    text = clean_text(value).casefold().replace("ё", "е")
    text = re.sub(r"[^a-z0-9а-я]+", "_", text).strip("_")
    return text[:90] or "item"


def sha256_text(value: Any) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
