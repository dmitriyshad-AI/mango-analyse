#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "google_doc_structured_fact_candidate_v1"
DEFAULT_INPUT_DIR = Path("product_data/knowledge_base/google_drive_structured_facts_20260517_v1/source_exports")
DEFAULT_OUT_DIR = Path("product_data/knowledge_base/google_drive_structured_facts_20260517_v1")

SOURCE_METADATA: Mapping[str, Mapping[str, str]] = {
    "unpk_prices_2026_2027.txt": {
        "brand": "unpk",
        "source_id": "source:google_drive_doc:12lsPcxpkP8Kf7y3dsLraYA_o9voXkq3Wg-KQEOF7YzI",
        "source_title": "УНПК Стоимость обучения и порядок оплаты на 26/2027 уч г от 16.03.26",
        "source_url": "https://docs.google.com/document/d/12lsPcxpkP8Kf7y3dsLraYA_o9voXkq3Wg-KQEOF7YzI",
        "source_updated_at": "2026-03-17T08:07:44.383Z",
    },
    "foton_prices_2026_2027.txt": {
        "brand": "foton",
        "source_id": "source:google_drive_doc:1k0hzS8cZjD2NXeE5mcjlMTEOtTVINlTstZpWOUI1gSc",
        "source_title": "ФОТОН Стоимость обучения и порядок оплаты на 26/27 уч от 16.03.26",
        "source_url": "https://docs.google.com/document/d/1k0hzS8cZjD2NXeE5mcjlMTEOtTVINlTstZpWOUI1gSc",
        "source_updated_at": "2026-03-17T08:03:02.111Z",
    },
    "kc_knowledge_base.txt": {
        "brand": "mixed",
        "source_id": "source:google_drive_doc:1bMhN0DtqNK8Z2XdwGMci2lAv0CtSYQ4QGb1Hr4dQ9Oo",
        "source_title": "База знаний КЦ",
        "source_url": "https://docs.google.com/document/d/1bMhN0DtqNK8Z2XdwGMci2lAv0CtSYQ4QGb1Hr4dQ9Oo",
        "source_updated_at": "2026-04-23T13:25:08.711Z",
    },
}

AMOUNT_RE = re.compile(r"\b\d[\d\s\u00a0]{1,8}\s*(?:руб\.?|₽)\b|\b\d{4,9}\b", re.I)
PERCENT_RE = re.compile(r"\b\d{1,2}\s*%")
DATE_RE = re.compile(
    r"\b\d{1,2}[./-]\d{1,2}(?:[./-]\d{2,4})?\b|\b\d{1,2}\s+"
    r"(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\b",
    re.I,
)
TIME_RE = re.compile(r"\b\d{1,2}[:.]\d{2}\b")


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract unverified structured fact candidates from exported Google Docs.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    result = extract_google_doc_facts(input_dir=args.input_dir, out_dir=args.out_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def extract_google_doc_facts(*, input_dir: Path, out_dir: Path) -> Mapping[str, Any]:
    if "stable_runtime" in out_dir.expanduser().resolve(strict=False).parts:
        raise ValueError("Refusing to write Google Doc fact candidates under stable_runtime")
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for path in sorted(input_dir.glob("*.txt")):
        source_meta = dict(SOURCE_METADATA.get(path.name) or fallback_source_metadata(path))
        text = path.read_text(encoding="utf-8", errors="ignore")
        source_sha = sha256_text(text)
        sources.append({**source_meta, "local_export_path": str(path), "source_sha256": source_sha})
        candidates.extend(extract_candidates_from_text(text, source_meta=source_meta, source_sha256=source_sha))

    jsonl_path = out_dir / "google_doc_structured_facts.jsonl"
    csv_path = out_dir / "google_doc_structured_facts.csv"
    summary_json_path = out_dir / "google_doc_fact_summary.json"
    summary_md_path = out_dir / "google_doc_fact_summary.md"
    write_jsonl(jsonl_path, candidates)
    write_csv(csv_path, candidates)
    summary = build_summary(candidates, sources=sources, out_dir=out_dir)
    summary_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    summary_md_path.write_text(render_summary_md(summary), encoding="utf-8")
    return {
        "candidates_total": len(candidates),
        "jsonl_path": str(jsonl_path),
        "csv_path": str(csv_path),
        "summary_json_path": str(summary_json_path),
        "summary_md_path": str(summary_md_path),
    }


def extract_candidates_from_text(text: str, *, source_meta: Mapping[str, str], source_sha256: str) -> list[dict[str, Any]]:
    lines = [clean_line(line) for line in text.splitlines()]
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, line in enumerate(lines):
        if not should_extract_line(line):
            continue
        context = context_window(lines, index)
        fact_type = classify_fact_type(line, context)
        text_hash = sha256_text(context)
        dedupe_key = f"{source_meta.get('source_id')}:{fact_type}:{sha256_text(line)}:{text_hash}"
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        fact_id = f"fact:gdoc:{safe_id(source_meta.get('source_id', 'source'))}:{fact_type}:{text_hash[:12]}"
        result.append(
            {
                "schema_version": SCHEMA_VERSION,
                "fact_id": fact_id,
                "fact_key": fact_key_for_type(fact_type),
                "fact_type": fact_type,
                "short_fact": short_fact_for_type(fact_type),
                "manager_text": context[:900],
                "client_safe_text": "",
                "structured_value": structured_value(line, context, source_meta=source_meta, fact_type=fact_type),
                "source_id": source_meta.get("source_id", ""),
                "source_title": source_meta.get("source_title", ""),
                "source_url": source_meta.get("source_url", ""),
                "source_updated_at": source_meta.get("source_updated_at", ""),
                "source_sha256": source_sha256,
                "source_span": {"line_start": max(1, index - 2), "line_end": min(len(lines), index + 3), "text_sha256": text_hash},
                "freshness_status": "needs_manager_confirmation",
                "verification_status": "extracted_unverified",
                "verified_at": None,
                "verified_by": None,
                "usable_for_precise_answer": False,
                "requires_manager_confirmation": True,
                "forbidden_for_client": True,
                "bot_permission": "manager_only",
                "related_theme_ids": related_theme_ids(fact_type),
                "risk_notes": "Точный ответ клиенту запрещен до ручной проверки и утверждения РОПом.",
            }
        )
    return result


def should_extract_line(line: str) -> bool:
    value = line.casefold()
    if not value or len(value) < 3:
        return False
    if AMOUNT_RE.search(line) and any(marker in value for marker in ("руб", "стоим", "тариф", "скид", "кэшбек", "цена")):
        return True
    if PERCENT_RE.search(line) and any(marker in value for marker in ("скид", "акци", "ранн", "многодет", "сотрудник", "предмет")):
        return True
    if DATE_RE.search(line) and any(marker in value for marker in ("до ", "после", "смен", "оплат", "июл", "август")):
        return True
    if TIME_RE.search(line) and any(marker in value for marker in ("распис", "занят", "клуб", "сбор", "обед")):
        return True
    if any(marker in value for marker in ("договор", "оферт", "налог", "маткап", "справк", "чек", "квитанц")):
        return True
    return False


def classify_fact_type(line: str, context: str) -> str:
    line_value = line.casefold()
    value = f"{line}\n{context}".casefold()
    if any(marker in line_value for marker in ("договор", "оферт", "налог", "маткап", "справк", "чек", "квитанц")):
        return "documents"
    if any(marker in line_value for marker in ("распис", "смен", "занят", "клуб", "обед", "сбор")) and (
        DATE_RE.search(line) or TIME_RE.search(line)
    ):
        return "schedule"
    if PERCENT_RE.search(line) or any(marker in line_value for marker in ("скид", "акци", "кэшбек", "промокод", "многодет", "сотрудник")):
        return "discount"
    if AMOUNT_RE.search(line) and not any(marker in line_value for marker in ("скид", "акци", "кэшбек", "промокод")):
        return "price"
    if any(marker in value for marker in ("скид", "акци", "кэшбек", "промокод", "многодет", "сотрудник")):
        return "discount"
    if any(marker in value for marker in ("распис", "смен", "занят", "клуб", "обед", "сбор")) and (DATE_RE.search(value) or TIME_RE.search(value)):
        return "schedule"
    if any(marker in value for marker in ("договор", "оферт", "налог", "маткап", "справк", "чек", "квитанц")):
        return "documents"
    if any(marker in value for marker in ("до ", "после", "оплат", "срок")) and DATE_RE.search(value):
        return "payment_deadline"
    return "price"


def structured_value(line: str, context: str, *, source_meta: Mapping[str, str], fact_type: str) -> Mapping[str, Any]:
    amounts = tuple(dict.fromkeys(" ".join(match.group(0).split()) for match in AMOUNT_RE.finditer(context)))
    percents = tuple(dict.fromkeys(match.group(0).replace(" ", "") for match in PERCENT_RE.finditer(context)))
    dates = tuple(dict.fromkeys(" ".join(match.group(0).split()) for match in DATE_RE.finditer(context)))
    times = tuple(dict.fromkeys(match.group(0) for match in TIME_RE.finditer(context)))
    return {
        "brand": source_meta.get("brand", "unknown"),
        "academic_year": "2026/2027" if "2026" in context or "26/27" in context else "",
        "raw_line": line,
        "amounts": list(amounts[:8]),
        "currency": "RUB" if amounts else "",
        "discount_percent": list(percents[:8]),
        "dates": list(dates[:8]),
        "times": list(times[:8]),
        "requires_table_review": fact_type in {"price", "discount", "schedule"},
    }


def fact_key_for_type(fact_type: str) -> str:
    return {
        "price": "prices.current",
        "discount": "discounts.current",
        "schedule": "schedule.current",
        "documents": "documents.current",
        "payment_deadline": "payment_deadlines.current",
    }.get(fact_type, "knowledge.current")


def related_theme_ids(fact_type: str) -> list[str]:
    return {
        "price": ["theme:001_pricing"],
        "discount": ["theme:005_discounts"],
        "schedule": ["theme:013_schedule"],
        "documents": ["theme:012_certificates", "theme:008_tax_deduction", "theme:007_matkap_payment"],
        "payment_deadline": ["theme:003_payment_status", "theme:004_payment_schedule"],
    }.get(fact_type, ["service:S2_unclear"])


def short_fact_for_type(fact_type: str) -> str:
    return {
        "price": "Найдена строка с ценой или тарифом; требуется ручная проверка таблицы.",
        "discount": "Найдено условие скидки или акции; требуется ручная проверка.",
        "schedule": "Найдена строка с датой, сменой или временем; требуется ручная проверка.",
        "documents": "Найдена строка про документы, договор, справки, налоговый вычет или маткапитал.",
        "payment_deadline": "Найдена строка про срок или порядок оплаты; требуется ручная проверка.",
    }.get(fact_type, "Найден кандидат факта; требуется ручная проверка.")


def context_window(lines: Sequence[str], index: int, *, radius: int = 2) -> str:
    selected = [line for line in lines[max(0, index - radius) : index + radius + 1] if line]
    return " ".join(" ".join(selected).split())[:1200]


def clean_line(line: str) -> str:
    return " ".join(str(line or "").replace("\ufeff", "").split())


def fallback_source_metadata(path: Path) -> Mapping[str, str]:
    source_id = f"source:google_drive_export:{path.stem}"
    return {
        "brand": "unknown",
        "source_id": source_id,
        "source_title": path.stem,
        "source_url": "",
        "source_updated_at": "",
    }


def build_summary(candidates: Sequence[Mapping[str, Any]], *, sources: Sequence[Mapping[str, Any]], out_dir: Path) -> Mapping[str, Any]:
    return {
        "schema_version": "google_doc_fact_extraction_summary_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "out_dir": str(out_dir),
        "sources_total": len(sources),
        "sources": list(sources),
        "candidates_total": len(candidates),
        "by_fact_type": dict(Counter(str(item.get("fact_type") or "") for item in candidates)),
        "usable_for_precise_answer": sum(1 for item in candidates if item.get("usable_for_precise_answer") is True),
        "requires_manager_confirmation": sum(1 for item in candidates if item.get("requires_manager_confirmation") is True),
        "forbidden_for_client": sum(1 for item in candidates if item.get("forbidden_for_client") is True),
        "precise_answer_unlocked": 0,
        "policy": "Все извлеченные факты являются кандидатами для РОПа и не могут использоваться для точного ответа клиенту.",
    }


def render_summary_md(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Google Docs fact candidates",
        "",
        f"- created_at: {summary.get('created_at')}",
        f"- sources_total: {summary.get('sources_total')}",
        f"- candidates_total: {summary.get('candidates_total')}",
        f"- precise_answer_unlocked: {summary.get('precise_answer_unlocked')}",
        f"- usable_for_precise_answer: {summary.get('usable_for_precise_answer')}",
        f"- requires_manager_confirmation: {summary.get('requires_manager_confirmation')}",
        "",
        "## By fact type",
        "",
    ]
    for key, count in sorted(dict(summary.get("by_fact_type") or {}).items()):
        lines.append(f"- {key}: {count}")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "Точные цены, скидки, сроки, документы и расписания не разблокированы для клиентских ответов.",
            "Файл нужен для проверки РОПом и последующего превращения кандидатов в утвержденные факты.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = [
        "fact_id",
        "fact_type",
        "fact_key",
        "source_title",
        "manager_text",
        "usable_for_precise_answer",
        "requires_manager_confirmation",
        "forbidden_for_client",
        "related_theme_ids",
        "risk_notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            flat = dict(row)
            flat["related_theme_ids"] = "|".join(str(item) for item in row.get("related_theme_ids") or [])
            writer.writerow(flat)


def sha256_text(text: Any) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()


def safe_id(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(text or "")).strip("_")[:80] or "source"


if __name__ == "__main__":
    raise SystemExit(main())
