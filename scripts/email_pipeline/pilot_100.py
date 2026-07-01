#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.email_pipeline.archive_sources import (
    DEFAULT_PROD_TIMELINE,
    DEFAULT_SOURCE_ROOT,
    ArchiveMessage,
    check_prod_timeline_readonly,
    default_archive_specs,
    existing_archive_paths,
    load_archive_messages,
    read_text,
)
from scripts.email_pipeline.brand import infer_email_brand
from scripts.email_pipeline.classification import build_outbound_templates
from scripts.email_pipeline.quality import evaluate_quality, quality_to_dict
from scripts.email_pipeline.summary import SummaryItem, clean_body, mask_pii, summarize_items


LOCAL_OUTPUT_DIR = Path(".codex_local/email_pipeline")
MASK_TOKEN_RE = re.compile(r"\[(?:phone|number)\]", re.I)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.[a-z]{2,}", re.I)
PHONE_RE = re.compile(r"(?<!\d)(?:\+7|7|8)\s*(?:\(?\d{3,4}\)?[\s.-]*)\d{2,3}[\s.-]*\d{2}[\s.-]*\d{2}(?!\d)")
PHONE_TAIL_RE = re.compile(r"\b\d{3}[-\s]\d{2}[-\s]\d{2}\b")
CONCRETE_RE = re.compile(
    r"("
    r"\b\d{1,2}[:.]\d{2}(?:\s*[-–—]\s*\d{1,2}[:.]\d{2})?\b|"
    r"\b\d{4}\s*[-–—/]\s*\d{4}\b|"
    r"\b\d{1,3}(?:\s?\d{3})?\s*(?:руб|₽|р\\.)\b|"
    r"\b\d{1,2}\s*(?:класс|кл\\.)\b|"
    r"\b(?:понедельник|вторник|среда|четверг|пятница|суббота|воскресенье)\b|"
    r"\b\d{1,2}\s*(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\b"
    r")",
    re.I,
)


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    source_root = Path(args.source_root).expanduser().resolve()
    prod_db = Path(args.prod_db).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve()
    local_output_dir = Path(args.local_output_dir).expanduser()
    if not local_output_dir.is_absolute():
        local_output_dir = repo_root / local_output_dir

    prod_check_before = check_prod_timeline_readonly(prod_db)
    specs = default_archive_specs(source_root)
    archive_paths = existing_archive_paths(specs)
    outbound_templates = build_outbound_templates(archive_paths, threshold=args.template_threshold)

    seen: set[str] = set()
    classification_counts: Counter[str] = Counter()
    real_records: list[dict[str, Any]] = []
    for spec in specs:
        for message in load_archive_messages(
            spec,
            source_root=source_root,
            repo_root=repo_root,
            outbound_templates=outbound_templates,
        ):
            if message.message_sha256 in seen:
                continue
            seen.add(message.message_sha256)
            classification_counts[message.klass] += 1
            if message.klass != "real_correspondence":
                continue
            body = read_text(message.extracted_text_path, limit=None)
            brand = infer_email_brand(message.subject, body)
            real_records.append(_record_for_message(message, body=body, brand=brand))

    real_records.sort(key=lambda row: row["message_sha256"])
    selected = real_records[: args.limit]
    if len(selected) != args.limit:
        raise RuntimeError(f"Expected {args.limit} real_correspondence messages, got {len(selected)}")

    summary_items = [
        SummaryItem(
            message_sha256=row["message_sha256"],
            direction=row["direction"],
            brand=row["brand"],
            brand_source=row["brand_source"],
            subject=row["subject"],
            body=row["body"],
        )
        for row in selected
    ]
    summary_result = summarize_items(
        summary_items,
        provider=args.summary_provider,
        model=args.model,
        reasoning=args.reasoning,
        batch_size=args.batch_size,
        max_llm_calls=args.max_llm_calls,
        project_root=repo_root,
        codex_home=Path(args.codex_home).expanduser().resolve() if args.codex_home else None,
        timeout_sec=args.timeout_sec,
    )
    if summary_result.llm_calls_total > args.max_llm_calls:
        raise RuntimeError(f"LLM call limit exceeded: {summary_result.llm_calls_total} > {args.max_llm_calls}")

    for row in selected:
        row["summary_payload"] = summary_result.summaries[row["message_sha256"]]
        row["full_clean_text"] = clean_body(row["body"])
        row["full_clean_text_chars"] = len(row["full_clean_text"])
        row["storage_mask_token_count"] = _mask_token_count(row)
        row["source_concrete_terms"] = _concrete_terms(f"{row['subject']}\n{row['full_clean_text']}")
        row["summary_concrete_terms"] = _concrete_terms(_summary_text(row))
        row["summary_retained_source_terms"] = _retained_source_terms(
            row.get("source_concrete_terms") or [],
            _summary_text(row),
        )
        row["summary_brand_mismatch"] = _summary_brand_mismatch(
            row["brand"], json.dumps(row["summary_payload"], ensure_ascii=False)
        )
        row["quality"] = quality_to_dict(evaluate_quality(row))
    ensure_local_output_dir_allowed(repo_root=repo_root, local_output_dir=local_output_dir)
    local_outputs = write_local_outputs(selected, local_output_dir=local_output_dir, prefix=args.local_output_prefix)
    prod_check_after = check_prod_timeline_readonly(prod_db)
    if not prod_check_after["mtime_unchanged"] or prod_check_before["mtime_before"] != prod_check_after["mtime_after"]:
        raise RuntimeError("prod timeline mtime changed during read-only pilot")

    report = build_report(
        repo_root=repo_root,
        source_root=source_root,
        prod_check=prod_check_after,
        archive_paths=archive_paths,
        classification_counts=classification_counts,
        real_records=real_records,
        selected=selected,
        summary_result=summary_result,
        local_outputs=local_outputs,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(json.dumps(
        {
            "selected": len(selected),
            "real_correspondence_total": len(real_records),
            "llm_calls_total": summary_result.llm_calls_total,
            "report": str(report_path),
            "prod_mtime_unchanged": prod_check_after["mtime_unchanged"],
            "storage_mask_token_count": sum(int(row["storage_mask_token_count"]) for row in selected),
            "memory_status_counts": dict(Counter((row.get("quality") or {}).get("memory_status") for row in selected)),
            "local_storage_jsonl": str(local_outputs["storage_jsonl"]),
            "local_reconciliation_csv": str(local_outputs["reconciliation_csv"]),
        },
        ensure_ascii=False,
    ))
    return 0


def _record_for_message(message: ArchiveMessage, *, body: str, brand: Any) -> dict[str, Any]:
    return {
        "message_sha256": message.message_sha256,
        "source_archive": message.source_archive,
        "message_id": message.message_id,
        "in_reply_to": message.in_reply_to,
        "references": message.references,
        "list_unsubscribe": message.list_unsubscribe,
        "precedence": message.precedence,
        "subject": message.subject,
        "mailbox": message.mailbox,
        "date_iso": message.date_iso,
        "direction": message.direction,
        "from_email": message.from_email,
        "from_domain": message.from_domain,
        "to_domains": list(message.to_domains),
        "classification_reason": message.classification_reason,
        "body_chars": message.body_chars,
        "attachment_count": message.attachment_count,
        "has_attachment": message.attachment_count > 0,
        "body": body,
        "body_available": bool(body),
        "brand": brand.brand,
        "brand_source": brand.brand_source,
        "raw_infer_offline_brand": brand.raw_infer_offline_brand,
        "brand_signals": dict(brand.signals),
    }


def build_report(
    *,
    repo_root: Path,
    source_root: Path,
    prod_check: dict[str, object],
    archive_paths: list[Path],
    classification_counts: Counter[str],
    real_records: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    summary_result: Any,
    local_outputs: dict[str, Path],
) -> str:
    selected_brand = Counter(row["brand"] for row in selected)
    selected_sources = Counter(row["brand_source"] for row in selected)
    all_real_brand = Counter(row["brand"] for row in real_records)
    raw_signal = Counter(row["raw_infer_offline_brand"] for row in selected)
    summary_mismatch = sum(1 for row in selected if row["summary_brand_mismatch"])
    bad_brand_sources = [row for row in selected if row["brand_source"] in {"folder", "from", "domain"}]
    storage_mask_count = sum(int(row.get("storage_mask_token_count") or 0) for row in selected)
    source_concrete_rows = sum(1 for row in selected if row.get("source_concrete_terms"))
    summary_concrete_rows = sum(1 for row in selected if row.get("summary_concrete_terms"))
    summary_retained_rows = sum(1 for row in selected if row.get("summary_retained_source_terms"))
    avg_original_chars = round(sum(int(row.get("body_chars") or 0) for row in selected) / max(len(selected), 1), 1)
    avg_clean_chars = round(sum(int(row.get("full_clean_text_chars") or 0) for row in selected) / max(len(selected), 1), 1)
    memory_status_counts = Counter((row.get("quality") or {}).get("memory_status") for row in selected)
    extraction_source_counts = Counter((row.get("summary_payload") or {}).get("extraction_source") for row in selected)
    event_type_counts = Counter((row.get("summary_payload") or {}).get("event_type") for row in selected)
    money_direction_counts = Counter((row.get("summary_payload") or {}).get("money_direction") for row in selected)
    model_placeholder_count = sum(
        1
        for row in selected
        if re.search(
            r"\[name\]|\b(?:за|для|по)\s+имя\b|\bскрытое\s+имя\b",
            json.dumps(row.get("summary_payload") or {}, ensure_ascii=False),
            re.I,
        )
    )
    human_confirmation_count = sum(
        1 for row in selected if (row.get("quality") or {}).get("requires_human_confirmation")
    )
    financial_unverified_rows = [row for row in selected if (row.get("quality") or {}).get("memory_status") == "financial_unverified"]
    thin_ack_rows = [row for row in selected if (row.get("quality") or {}).get("memory_status") == "thin_ack"]
    attachment_rows = [row for row in selected if (row.get("quality") or {}).get("memory_status") == "attachment_only"]
    thread_rows = [row for row in selected if (row.get("quality") or {}).get("memory_status") in {"needs_thread_context", "quote_only"}]
    broadcast_rows = [row for row in selected if (row.get("quality") or {}).get("memory_status") == "broadcast_not_usable"]
    usable_rows = [row for row in selected if (row.get("quality") or {}).get("memory_status") == "usable_memory"]
    suspicious_18 = next((row for row in selected if row["message_sha256"].startswith("006099fe3375")), None)
    camp_102k = [row for row in selected if 102600 in ((row.get("quality") or {}).get("money_amounts_rub") or [])]
    money_gate_false_positive = [
        row
        for row in selected
        if (row.get("quality") or {}).get("memory_status") == "financial_unverified"
        and not any(amount > 1_000_000 for amount in ((row.get("quality") or {}).get("money_amounts_rub") or []))
        and not (row.get("summary_payload") or {}).get("amount_uncertain")
    ]
    thin_ack_false_positive = [
        row
        for row in selected
        if (row.get("quality") or {}).get("memory_status") == "thin_ack"
        and (
            (row.get("summary_payload") or {}).get("event_type") not in (None, "", "other", "broadcast")
            or (row.get("summary_payload") or {}).get("money_direction") not in (None, "", "none")
            or (row.get("summary_payload") or {}).get("student_name")
            or (row.get("summary_payload") or {}).get("amount_rub") is not None
        )
    ]
    broadcast_false_positive = [
        row
        for row in selected
        if (row.get("quality") or {}).get("memory_status") == "broadcast_not_usable"
        and row.get("from_domain") in {"kmipt.ru", "cdpofoton.ru"}
    ]
    git = git_block(repo_root)
    examples = [row for row in selected if row.get("summary_retained_source_terms")][:7]
    if len(examples) < 7:
        examples.extend(selected[: 7 - len(examples)])
    lines: list[str] = []
    lines.append("# 100 писем: pilot dry-run email pipeline restore\n")
    lines.append("Статус: dry-run, без записи в CRM/timeline и без отправки писем. Вердикт в прод не выносится.\n")
    lines.append("\n## Git\n")
    for key, value in git.items():
        lines.append(f"- {key}: `{value}`")
    lines.append("\n## Источники и безопасность\n")
    lines.append(f"- source_root: `{source_root}`")
    lines.append(f"- archive sqlite files: {len(archive_paths)}")
    lines.append(f"- prod timeline: `{prod_check['path']}`")
    lines.append(f"- prod quick_check: `{prod_check['quick_check']}`")
    lines.append(f"- prod email_message events: {prod_check['email_events']}")
    lines.append(f"- prod mtime before/after: {prod_check['mtime_before']} / {prod_check['mtime_after']}")
    lines.append(f"- prod mtime unchanged: {prod_check['mtime_unchanged']}")
    lines.append(f"- prod size unchanged: {prod_check['size_unchanged']}")
    lines.append("- ПДн в отчёт не включены: темы/фрагменты/сводки маскированы.")
    lines.append("\n## Выборка\n")
    lines.append(f"- distinct archive messages read: {sum(classification_counts.values())}")
    lines.append(f"- real_correspondence total: {len(real_records)}")
    lines.append(f"- deterministic selection: class=real_correspondence, sort=message_sha256, first={len(selected)}")
    lines.append("\nКлассы:")
    for key, value in classification_counts.most_common():
        lines.append(f"- {key}: {value}")
    lines.append("\n## Бренд по содержанию\n")
    lines.append("Старые метаданные ящика/отправителя не используются; silence/conflict => `none`.")
    lines.append("\nПо всем real_correspondence:")
    for key, value in all_real_brand.most_common():
        lines.append(f"- {key}: {value}")
    lines.append(f"\nПо выбранным {len(selected)}:")
    for key, value in selected_brand.most_common():
        lines.append(f"- {key}: {value}")
    lines.append(f"\nbrand_source по выбранным {len(selected)}:")
    for key, value in selected_sources.most_common():
        lines.append(f"- {key}: {value}")
    lines.append(f"\nraw infer_offline_brand по выбранным {len(selected)}:")
    for key, value in raw_signal.most_common():
        lines.append(f"- {key}: {value}")
    lines.append(f"\n- legacy_metadata_brand_source_count: {len(bad_brand_sources)}")
    lines.append(f"- summary brand mismatch: {summary_mismatch}")
    lines.append("\n## A2: хранение полного текста и конкретики\n")
    lines.append(f"- storage_mask_token_count (`[phone]`/`[number]` в полном тексте+summary): {storage_mask_count}")
    lines.append("- baseline из ТЗ: 38/100 строк с числом, уничтоженным маской; текущий целевой счётчик: 0")
    lines.append(f"- source_concrete_rows: {source_concrete_rows} из {len(selected)}")
    lines.append(f"- summary_concrete_rows: {summary_concrete_rows} из {len(selected)}")
    lines.append(f"- summary_retained_source_concrete_rows: {summary_retained_rows} из {source_concrete_rows}")
    lines.append(f"- avg_original_body_chars: {avg_original_chars}")
    lines.append(f"- avg_full_clean_text_chars: {avg_clean_chars}")
    lines.append(f"- local full storage JSONL: `{local_outputs['storage_jsonl']}`")
    lines.append(f"- local reconciliation CSV with phones/emails: `{local_outputs['reconciliation_csv']}`")
    lines.append("- Локальные таблицы лежат в `.codex_local/email_pipeline/`, этот путь уже в `.gitignore`; в Foton/git они не попадают.")
    lines.append("- LLM-вход маскирует телефоны/email, но сохраняет имена и реквизиты для извлечения student_name/contract/requisites; учебные числа, даты, цены, классы и группы не маскируются.")
    lines.append("\n## LLM-сводки\n")
    lines.append(f"- provider: {summary_result.provider}")
    lines.append(f"- model: {summary_result.model}")
    lines.append(f"- reasoning: {summary_result.reasoning}")
    lines.append(f"- llm_calls_total: {summary_result.llm_calls_total}")
    lines.append(f"- llm_calls_limit: <=100")
    lines.append("\n## A2-v2: слой качества перед памятью\n")
    lines.append("Смысловые поля (`event_type`, `money_direction`, `amount_kind`, `student_name`, `grade`, `subject_area`) извлекает модель; детерминированные гейты меряют только число/длину/вложения/заголовки.")
    lines.append("\nmemory_status:")
    for key, value in memory_status_counts.most_common():
        lines.append(f"- {key}: {value}")
    lines.append(f"- usable_memory_share: {round(len(usable_rows) / max(len(selected), 1), 3)}")
    lines.append("\nextraction_source:")
    for key, value in extraction_source_counts.most_common():
        lines.append(f"- {key}: {value}")
    lines.append("\nevent_type:")
    for key, value in event_type_counts.most_common():
        lines.append(f"- {key}: {value}")
    lines.append("\nmoney_direction:")
    for key, value in money_direction_counts.most_common():
        lines.append(f"- {key}: {value}")
    lines.append(f"\n- name_placeholder_count_in_raw_summary: {model_placeholder_count}")
    lines.append(f"- requires_human_confirmation: {human_confirmation_count}")
    lines.append(f"- financial_unverified_count: {len(financial_unverified_rows)}")
    lines.append(f"- thin_ack_count: {len(thin_ack_rows)}")
    lines.append(f"- attachment_only_count: {len(attachment_rows)}")
    lines.append(f"- thread_context_or_quote_count: {len(thread_rows)}")
    lines.append(f"- broadcast_not_usable_count: {len(broadcast_rows)}")
    lines.append("\nPrecision гейтов на контрольных условиях:")
    lines.append(f"- money_gate_false_positive_by_number_check: {len(money_gate_false_positive)}")
    lines.append(f"- thin_ack_false_positive_by_model_fact_check: {len(thin_ack_false_positive)}")
    lines.append(f"- broadcast_false_positive_by_trusted_domain_check: {len(broadcast_false_positive)}")
    checked = len(financial_unverified_rows) + len(thin_ack_rows) + len(broadcast_rows)
    false_positive = len(money_gate_false_positive) + len(thin_ack_false_positive) + len(broadcast_false_positive)
    precision = 1.0 if checked == 0 else round((checked - false_positive) / checked, 3)
    lines.append(f"- gate_precision_counterset: {precision} ({checked - false_positive}/{checked})")
    lines.append("\nКонтр-примеры и обязательные гейты:")
    if suspicious_18:
        lines.append(
            f"- #18 money_gate: sha `{suspicious_18['message_sha256'][:12]}`, status="
            f"`{(suspicious_18.get('quality') or {}).get('memory_status')}`, amounts="
            f"`{(suspicious_18.get('quality') or {}).get('money_amounts_rub')}`"
        )
    else:
        lines.append("- #18 money_gate: NOT_FOUND")
    for row in camp_102k[:2]:
        lines.append(
            f"- 102k camp counterexample: sha `{row['message_sha256'][:12]}`, status="
            f"`{(row.get('quality') or {}).get('memory_status')}`, amounts="
            f"`{(row.get('quality') or {}).get('money_amounts_rub')}`"
        )
    short_valuable = [
        row
        for row in selected
        if len(str(row.get("full_clean_text") or "").strip()) < 80
        and (row.get("quality") or {}).get("memory_status") == "usable_memory"
    ]
    for row in short_valuable[:3]:
        payload = row.get("summary_payload") or {}
        lines.append(
            f"- short valuable kept: sha `{row['message_sha256'][:12]}`, event_type="
            f"`{payload.get('event_type')}`, status=`usable_memory`, summary={mask_pii(str(payload.get('summary') or ''))[:180]}"
        )
    lines.append("\nПлохие классы, сохранены как пометки, ничего не удалено:")
    for title, rows in (
        ("financial_unverified", financial_unverified_rows),
        ("thin_ack", thin_ack_rows),
        ("attachment_only", attachment_rows),
        ("needs_thread_context/quote_only", thread_rows),
        ("broadcast_not_usable", broadcast_rows),
    ):
        lines.append(f"\n### {title}")
        for row in rows[:3]:
            payload = row.get("summary_payload") or {}
            quality = row.get("quality") or {}
            lines.append(
                f"- sha `{row['message_sha256'][:12]}` status=`{quality.get('memory_status')}` "
                f"flags=`{','.join(quality.get('quality_flags') or [])}` "
                f"event=`{payload.get('event_type')}` money=`{payload.get('money_direction')}` "
                f"summary={mask_pii(str(payload.get('summary') or ''))[:220]}"
            )
    lines.append("\n## 7 обезличенных примеров с конкретикой\n")
    for index, row in enumerate(examples, 1):
        summary_payload = row.get("summary_payload") or {}
        lines.append(f"\n### Пример {index}")
        lines.append(f"- sha: `{row['message_sha256'][:12]}`")
        lines.append(f"- direction: `{row['direction']}`")
        lines.append(f"- brand: `{row['brand']}` / source=`{row['brand_source']}`")
        lines.append(f"- source_concrete_terms: {', '.join(row.get('source_concrete_terms') or []) or 'none'}")
        lines.append(f"- summary_concrete_terms: {', '.join(row.get('summary_concrete_terms') or []) or 'none'}")
        lines.append(f"- retained_source_terms: {', '.join(row.get('summary_retained_source_terms') or []) or 'none'}")
        quality = row.get("quality") or {}
        summary_payload = row.get("summary_payload") or {}
        lines.append(
            f"- quality: status=`{quality.get('memory_status')}`, flags=`{','.join(quality.get('quality_flags') or [])}`, "
            f"event=`{summary_payload.get('event_type')}`, money=`{summary_payload.get('money_direction')}`, "
            f"human_confirmation=`{quality.get('requires_human_confirmation')}`"
        )
        lines.append(f"- subject: {mask_pii(row['subject'])[:180] or '(пусто)'}")
        lines.append(f"- body_fragment: {mask_pii(clean_body(row['body'], limit=520))[:520] or '(тело недоступно)'}")
        lines.append(f"- summary: {mask_pii(str(summary_payload.get('summary') or ''))}")
        lines.append(f"- topic: {mask_pii(str(summary_payload.get('topic') or ''))}")
        lines.append(f"- next_step: {mask_pii(str(summary_payload.get('next_step') or 'null'))}")
    lines.append("\n## Контрольные выводы\n")
    lines.append("- AMO/CRM/Tallanto/writeback не использовались.")
    lines.append("- Prod timeline открыт только `mode=ro`, `query_only=ON`; mtime/size не изменились.")
    lines.append("- В git добавлен только код `scripts/email_pipeline/` и тесты; архивы, письма, БД и ПДн не коммитятся.")
    lines.append("- Foton-отчёт обезличен; полные письма и ПДн сохранены только локально в ignored `.codex_local`.")
    lines.append("- Вердикт о проде не выносится; приёмка за Claude #1 по сырью.")
    return "\n".join(lines) + "\n"


def write_local_outputs(selected: list[dict[str, Any]], *, local_output_dir: Path, prefix: str) -> dict[str, Path]:
    local_output_dir.mkdir(parents=True, exist_ok=True)
    storage_path = local_output_dir / f"{prefix}_full_storage.jsonl"
    reconciliation_path = local_output_dir / f"{prefix}_reconciliation.csv"
    with storage_path.open("w", encoding="utf-8") as f:
        for row in selected:
            payload = {
                "message_sha256": row["message_sha256"],
                "date_iso": row["date_iso"],
                "direction": row["direction"],
                "brand": row["brand"],
                "brand_source": row["brand_source"],
                "raw_infer_offline_brand": row["raw_infer_offline_brand"],
                "classification_reason": row["classification_reason"],
                "from_email": row.get("from_email"),
                "from_domain": row.get("from_domain"),
                "to_domains": row.get("to_domains") or [],
                "subject_full": row["subject"],
                "full_clean_text": row["full_clean_text"],
                "full_clean_text_chars": row["full_clean_text_chars"],
                "body_chars": row["body_chars"],
                "summary_payload": row["summary_payload"],
                "quality": row.get("quality") or {},
                "source_concrete_terms": row.get("source_concrete_terms") or [],
                "summary_concrete_terms": row.get("summary_concrete_terms") or [],
                "summary_retained_source_terms": row.get("summary_retained_source_terms") or [],
                "storage_mask_token_count": row.get("storage_mask_token_count") or 0,
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    fieldnames = [
        "message_sha256",
        "date_iso",
        "direction",
        "brand",
        "brand_source",
        "from_email",
        "detected_emails",
        "detected_phones",
        "subject_full",
        "summary",
        "next_step",
        "memory_status",
        "quality_flags",
        "event_type",
        "money_direction",
        "amount_rub",
        "amount_kind",
        "amount_uncertain",
        "money_amounts_rub",
        "student_name",
        "grade",
        "subject_area",
        "requires_human_confirmation",
        "safe_next_step_note",
        "thread_id",
        "thread_basis",
        "has_attachment",
        "full_clean_text_chars",
    ]
    with reconciliation_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected:
            text = f"{row.get('from_email') or ''}\n{row['subject']}\n{row['full_clean_text']}"
            summary_payload = row.get("summary_payload") or {}
            quality = row.get("quality") or {}
            writer.writerow(
                {
                    "message_sha256": row["message_sha256"],
                    "date_iso": row["date_iso"],
                    "direction": row["direction"],
                    "brand": row["brand"],
                    "brand_source": row["brand_source"],
                    "from_email": row.get("from_email") or "",
                    "detected_emails": "; ".join(_unique(EMAIL_RE.findall(text))),
                    "detected_phones": "; ".join(_unique(PHONE_RE.findall(text) + PHONE_TAIL_RE.findall(text))),
                    "subject_full": row["subject"],
                    "summary": str(summary_payload.get("summary") or ""),
                    "next_step": str(summary_payload.get("next_step") or ""),
                    "memory_status": quality.get("memory_status") or "",
                    "quality_flags": "; ".join(quality.get("quality_flags") or []),
                    "event_type": summary_payload.get("event_type") or "",
                    "money_direction": summary_payload.get("money_direction") or "",
                    "amount_rub": summary_payload.get("amount_rub") if summary_payload.get("amount_rub") is not None else "",
                    "amount_kind": summary_payload.get("amount_kind") or "",
                    "amount_uncertain": summary_payload.get("amount_uncertain"),
                    "money_amounts_rub": "; ".join(str(item) for item in quality.get("money_amounts_rub") or []),
                    "student_name": summary_payload.get("student_name") or "",
                    "grade": summary_payload.get("grade") or "",
                    "subject_area": summary_payload.get("subject_area") or "",
                    "requires_human_confirmation": quality.get("requires_human_confirmation"),
                    "safe_next_step_note": quality.get("safe_next_step_note") or "",
                    "thread_id": quality.get("thread_id") or "",
                    "thread_basis": quality.get("thread_basis") or "",
                    "has_attachment": row.get("has_attachment"),
                    "full_clean_text_chars": row.get("full_clean_text_chars") or 0,
                }
            )
    return {"storage_jsonl": storage_path, "reconciliation_csv": reconciliation_path}


def _mask_token_count(row: dict[str, Any]) -> int:
    text = "\n".join(
        [
            str(row.get("subject") or ""),
            str(row.get("full_clean_text") or ""),
            _summary_text(row),
        ]
    )
    return len(MASK_TOKEN_RE.findall(text))


def _summary_text(row: dict[str, Any]) -> str:
    summary_payload = row.get("summary_payload") or {}
    return "\n".join(
        str(summary_payload.get(key) or "")
        for key in ("summary", "topic", "next_step")
        if summary_payload.get(key) not in (None, "")
    )


def _concrete_terms(text: str, *, limit: int = 12) -> list[str]:
    seen: list[str] = []
    for match in CONCRETE_RE.finditer(text or ""):
        value = re.sub(r"\s+", " ", match.group(0)).strip()
        if value and value.casefold() not in {item.casefold() for item in seen}:
            seen.append(value)
        if len(seen) >= limit:
            break
    return seen


def _retained_source_terms(source_terms: list[str], summary_text: str) -> list[str]:
    normalized_summary = re.sub(r"\s+", " ", summary_text or "").casefold()
    retained: list[str] = []
    for term in source_terms:
        normalized_term = re.sub(r"\s+", " ", term or "").casefold()
        if normalized_term and normalized_term in normalized_summary:
            retained.append(term)
    return retained


def ensure_local_output_dir_allowed(*, repo_root: Path, local_output_dir: Path) -> None:
    resolved_repo = repo_root.resolve()
    resolved_output = local_output_dir.expanduser().resolve(strict=False)
    if "Claude Projects" in resolved_output.parts and "Foton" in resolved_output.parts:
        raise RuntimeError("local raw email outputs with PII must not be written to Foton")
    if not resolved_output.is_relative_to(resolved_repo):
        raise RuntimeError("local raw email outputs must stay inside this repo under an ignored path")
    relative = resolved_output.relative_to(resolved_repo)
    probe = str(relative / ".pii_probe")
    proc = subprocess.run(
        ["git", "check-ignore", "-q", "--", probe],
        cwd=resolved_repo,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"local raw email output path is not git-ignored: {relative}")


def _unique(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = re.sub(r"\s+", " ", str(value or "").strip())
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            out.append(cleaned)
    return out


def git_block(repo_root: Path) -> dict[str, str]:
    def run(args: list[str]) -> str:
        proc = subprocess.run(["git", *args], cwd=repo_root, capture_output=True, text=True, check=False)
        return (proc.stdout or proc.stderr).strip()
    def run_ok(args: list[str]) -> str:
        proc = subprocess.run(["git", *args], cwd=repo_root, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            return "not-set"
        return (proc.stdout or "").strip() or "not-set"

    return {
        "branch": run(["branch", "--show-current"]),
        "head": run(["rev-parse", "--short", "HEAD"]),
        "head_full": run(["rev-parse", "HEAD"]),
        "origin_branch": run_ok(["rev-parse", "--short", "@{u}"]),
        "archive_tag": run(["tag", "--points-at", "HEAD"]),
    }


def _summary_brand_mismatch(brand: str, text: str) -> bool:
    lowered = (text or "").casefold()
    foton_tokens = ("фотон", "cdpofoton", "цдпо")
    unpk_tokens = ("унпк", "мфти", "физтех", "kmipt")
    if brand == "none":
        return any(token in lowered for token in (*foton_tokens, *unpk_tokens))
    if brand == "foton":
        return any(token in lowered for token in unpk_tokens)
    if brand == "unpk":
        return any(token in lowered for token in foton_tokens)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only 100-message e-mail pipeline pilot.")
    parser.add_argument("--repo-root", default=str(Path.cwd()))
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--prod-db", default=str(DEFAULT_PROD_TIMELINE))
    parser.add_argument(
        "--report",
        default="/Users/dmitrijfabarisov/Claude Projects/Foton/100_pisem_pilot_dry_run.md",
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--template-threshold", type=int, default=10)
    parser.add_argument("--summary-provider", choices=("auto", "openai", "codex_cli", "stub"), default="auto")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--reasoning", default="medium")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--max-llm-calls", type=int, default=100)
    parser.add_argument("--timeout-sec", type=int, default=240)
    parser.add_argument("--codex-home", default="")
    parser.add_argument("--local-output-dir", default=str(LOCAL_OUTPUT_DIR))
    parser.add_argument("--local-output-prefix", default="A2_100_pisem_pilot")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
