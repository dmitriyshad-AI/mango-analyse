from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill

from mango_mvp.question_catalog.contracts import (
    ANSWER_STATUS_DRAFT_NEEDS_REVIEW,
    ANSWER_STATUS_MANAGER_ONLY,
    ANSWER_STATUS_NEEDS_ROP_ANSWER,
    ANSWER_STATUS_TEMPLATE_NEEDS_CURRENT_FACT,
    BOT_PERMISSION_ALLOWED_AFTER_FACT_CHECK,
    BOT_PERMISSION_DRAFT_ONLY,
    BOT_PERMISSION_MANAGER_ONLY,
    CurrentFactSource,
    QuestionClass,
    QuestionItem,
    AnswerTemplate,
    normalize_key,
    question_catalog_contract_inventory,
    question_catalog_safety_contract,
    stable_answer_template_id,
)
from mango_mvp.question_catalog.extractors import (
    SINCE_DEFAULT,
    extract_call_questions,
    extract_mail_questions,
    extract_telegram_questions,
)
from mango_mvp.question_catalog.normalization import infer_question_metadata
from mango_mvp.question_catalog.safety import (
    assert_public_text_safe,
    guard_question_catalog_output_path,
)


CATALOG_SCHEMA_VERSION = "customer_question_catalog_v1"


@dataclass(frozen=True)
class CatalogBuildConfig:
    project_root: Path
    out_root: Path
    tenant_id: str = "foton"
    since: datetime = SINCE_DEFAULT
    calls_enriched_reviews: Path | None = None
    telegram_messages_jsonl: Path | None = None
    mail_archive_root: Path | None = None
    fact_source_roots: Sequence[Path] = ()


def default_config(project_root: Path, out_root: Path | None = None) -> CatalogBuildConfig:
    return CatalogBuildConfig(
        project_root=project_root,
        out_root=out_root or project_root / "product_data" / "question_catalog",
        calls_enriched_reviews=project_root
        / "stable_runtime"
        / "sales_insight_knowledge_base_after_quality_backfill_20260510_v11_frozen_gate"
        / "enriched_reviews.csv",
        telegram_messages_jsonl=project_root / "telegram_exports (2)" / "local_vm_2024-04-01" / "messages.jsonl",
        mail_archive_root=project_root / "_external_handoffs" / "mail_archive_2026-05-12",
        fact_source_roots=(
            project_root / ".codex_local" / "kc_source_extract_20260513" / "texts",
            project_root / "docs",
        ),
    )


def build_customer_question_catalog(config: CatalogBuildConfig) -> dict[str, Any]:
    out_root = guard_question_catalog_output_path(config.out_root, project_root=config.project_root)
    out_root.mkdir(parents=True, exist_ok=True)

    source_reports: list[Mapping[str, Any]] = []
    items: list[QuestionItem] = []

    call_items, call_report = extract_call_questions(
        config.calls_enriched_reviews or Path("__missing_calls__"),
        tenant_id=config.tenant_id,
        since=config.since,
    )
    items.extend(call_items)
    source_reports.append(call_report)

    telegram_items, telegram_report = extract_telegram_questions(
        config.telegram_messages_jsonl or Path("__missing_telegram__"),
        tenant_id=config.tenant_id,
        since=config.since,
    )
    items.extend(telegram_items)
    source_reports.append(telegram_report)

    mail_items, mail_report = extract_mail_questions(
        config.mail_archive_root or Path("__missing_mail__"),
        tenant_id=config.tenant_id,
        since=config.since,
    )
    items.extend(mail_items)
    source_reports.append(mail_report)

    items = _dedupe_items(items)
    classes = build_question_classes(items, tenant_id=config.tenant_id)
    templates = build_answer_templates(classes, tenant_id=config.tenant_id)
    classes = attach_template_ids(classes, templates)
    fact_sources = discover_fact_sources(config.fact_source_roots)
    summary = build_summary(
        config=config,
        items=items,
        classes=classes,
        templates=templates,
        fact_sources=fact_sources,
        source_reports=source_reports,
    )

    outputs = write_outputs(
        out_root,
        items=items,
        classes=classes,
        templates=templates,
        fact_sources=fact_sources,
        source_reports=source_reports,
        summary=summary,
    )
    summary["outputs"] = {key: str(path) for key, path in outputs.items()}
    (out_root / "question_catalog_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def build_question_classes(items: Sequence[QuestionItem], *, tenant_id: str) -> list[QuestionClass]:
    grouped: dict[str, list[QuestionItem]] = defaultdict(list)
    for item in items:
        grouped[item.question_class_id].append(item)
    classes: list[QuestionClass] = []
    for class_id, class_items in sorted(grouped.items(), key=lambda pair: (-len(pair[1]), pair[0])):
        sample = class_items[0]
        metadata = infer_question_metadata(
            sample.customer_text_redacted,
            fallback_signal=str(sample.metadata.get("signal") or ""),
        )
        source_counts = Counter(item.source_channel for item in class_items)
        dates = sorted(item.occurred_at for item in class_items if item.occurred_at)
        examples = _unique([item.customer_text_redacted for item in class_items if item.customer_text_redacted], limit=5)
        products = _unique([item.product for item in class_items if item.product], limit=8)
        grades = _unique([item.grade for item in class_items if item.grade], limit=8)
        subjects = _unique([item.subject for item in class_items if item.subject], limit=8)
        answer_status = metadata.answer_status
        bot_permission = metadata.bot_permission
        if answer_status == ANSWER_STATUS_MANAGER_ONLY:
            bot_permission = BOT_PERMISSION_MANAGER_ONLY
        elif metadata.dynamic_fact_types:
            answer_status = ANSWER_STATUS_TEMPLATE_NEEDS_CURRENT_FACT
            bot_permission = BOT_PERMISSION_ALLOWED_AFTER_FACT_CHECK
        elif any(item.manager_text_redacted for item in class_items):
            answer_status = ANSWER_STATUS_DRAFT_NEEDS_REVIEW
            bot_permission = BOT_PERMISSION_DRAFT_ONLY
        else:
            answer_status = ANSWER_STATUS_NEEDS_ROP_ANSWER
            bot_permission = BOT_PERMISSION_DRAFT_ONLY
        priority = _priority_for_class(class_items, metadata.required_fact_keys)
        classes.append(
            QuestionClass(
                tenant_id=tenant_id,
                question_class_id=class_id,
                canonical_question=metadata.canonical_question,
                narrow_scope=metadata.narrow_scope,
                class_key=metadata.class_key,
                exclusions=metadata.exclusions,
                examples_redacted=examples,
                count_total=len(class_items),
                count_calls=source_counts.get("call", 0),
                count_telegram=source_counts.get("telegram", 0),
                count_email=source_counts.get("email", 0),
                first_seen_at=dates[0] if dates else None,
                last_seen_at=dates[-1] if dates else None,
                products=products,
                grades=grades,
                subjects=subjects,
                answer_status=answer_status,
                required_fact_keys=metadata.required_fact_keys,
                fact_source_refs=_fact_source_refs(metadata.required_fact_keys),
                fact_freshness_policy=metadata.fact_freshness_policy,
                fallback_when_fact_missing=metadata.fallback_when_fact_missing,
                bot_permission=bot_permission,
                manager_handoff_reason=metadata.manager_handoff_reason,
                rop_review_priority=priority,
                metadata={
                    "intents": sorted(Counter(item.intent for item in class_items)),
                    "dynamic_fact_types": sorted({fact for item in class_items for fact in item.dynamic_fact_types}),
                },
            )
        )
    return classes


def build_answer_templates(classes: Sequence[QuestionClass], *, tenant_id: str) -> list[AnswerTemplate]:
    templates: list[AnswerTemplate] = []
    for item in classes:
        template = _template_text_for_class(item)
        permission = BOT_PERMISSION_ALLOWED_AFTER_FACT_CHECK if item.required_fact_keys else BOT_PERMISSION_DRAFT_ONLY
        if item.answer_status == ANSWER_STATUS_MANAGER_ONLY:
            permission = BOT_PERMISSION_MANAGER_ONLY
        template_id = stable_answer_template_id(
            tenant_id=tenant_id,
            question_class_id=item.question_class_id,
            template_text=template,
        )
        templates.append(
            AnswerTemplate(
                tenant_id=tenant_id,
                question_class_id=item.question_class_id,
                answer_template_id=template_id,
                template_text=template,
                required_fact_keys=item.required_fact_keys,
                approval_status=item.answer_status,
                bot_permission_if_facts_fresh=permission,
                fallback_when_fact_missing=item.fallback_when_fact_missing
                or "Если нет утвержденного ответа, передать менеджеру.",
            )
        )
    return templates


def attach_template_ids(classes: Sequence[QuestionClass], templates: Sequence[AnswerTemplate]) -> list[QuestionClass]:
    by_class = {template.question_class_id: template.answer_template_id for template in templates}
    return [replace(item, answer_template_id=by_class.get(item.question_class_id)) for item in classes]


def discover_fact_sources(roots: Sequence[Path]) -> list[CurrentFactSource]:
    sources: list[CurrentFactSource] = []
    seen: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".txt", ".md", ".csv", ".xlsx", ".json"}:
                continue
            fact_types = _fact_types_from_path(path)
            if not fact_types:
                continue
            source_id = normalize_key(f"fact_{len(sources) + 1}", "source_id")
            rel = str(path)
            if rel in seen:
                continue
            seen.add(rel)
            sources.append(
                CurrentFactSource(
                    source_id=source_id,
                    fact_types=fact_types,
                    path=rel,
                    owner="manual_owner_required",
                    last_updated_at=None,
                    freshness_policy="Нужна ручная проверка свежести перед автономным ответом.",
                    usable_for_bot=False,
                    notes="Источник найден автоматически; значения можно использовать только после утверждения владельцем.",
                )
            )
    return sources


def write_outputs(
    out_root: Path,
    *,
    items: Sequence[QuestionItem],
    classes: Sequence[QuestionClass],
    templates: Sequence[AnswerTemplate],
    fact_sources: Sequence[CurrentFactSource],
    source_reports: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> Mapping[str, Path]:
    outputs = {
        "items_jsonl": out_root / "customer_question_items.jsonl",
        "classes_csv": out_root / "customer_question_classes.csv",
        "classes_xlsx": out_root / "customer_question_classes.xlsx",
        "templates_csv": out_root / "answer_templates.csv",
        "fact_requirements_csv": out_root / "fact_requirements.csv",
        "fact_registry_json": out_root / "current_fact_source_registry.json",
        "rop_review_xlsx": out_root / "rop_question_review_pack.xlsx",
        "unanswered_csv": out_root / "unanswered_questions.csv",
        "source_coverage_md": out_root / "source_coverage_report.md",
        "summary_json": out_root / "question_catalog_summary.json",
    }
    write_jsonl(outputs["items_jsonl"], [item.to_json_dict() for item in items])
    write_csv(outputs["classes_csv"], [flatten_class(item) for item in classes])
    write_csv(outputs["templates_csv"], [flatten_template(item) for item in templates])
    write_csv(outputs["fact_requirements_csv"], fact_requirement_rows(classes))
    outputs["fact_registry_json"].write_text(
        json.dumps(
            {
                "schema_version": CATALOG_SCHEMA_VERSION,
                "safety_note": "Найденные источники фактов требуют ручной проверки свежести перед автономным ответом.",
                "sources": [item.to_json_dict() for item in fact_sources],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    unanswered = [
        flatten_class(item)
        for item in classes
        if item.answer_status
        in {
            ANSWER_STATUS_NEEDS_ROP_ANSWER,
            ANSWER_STATUS_TEMPLATE_NEEDS_CURRENT_FACT,
            ANSWER_STATUS_MANAGER_ONLY,
        }
    ]
    write_csv(outputs["unanswered_csv"], unanswered)
    write_classes_xlsx(outputs["classes_xlsx"], classes)
    write_rop_review_xlsx(outputs["rop_review_xlsx"], classes)
    outputs["source_coverage_md"].write_text(render_source_coverage(source_reports, summary), encoding="utf-8")
    return outputs


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_classes_xlsx(path: Path, classes: Sequence[QuestionClass]) -> None:
    rows = [flatten_class(item) for item in classes]
    write_xlsx(path, "question_classes", rows)


def write_rop_review_xlsx(path: Path, classes: Sequence[QuestionClass]) -> None:
    rows = []
    for item in classes:
        if item.rop_review_priority in {"critical", "high", "medium"}:
            rows.append(
                {
                    "Приоритет": item.rop_review_priority,
                    "Класс вопроса": item.canonical_question,
                    "Примеры": " | ".join(item.examples_redacted[:3]),
                    "Частота всего": item.count_total,
                    "Звонки": item.count_calls,
                    "Telegram": item.count_telegram,
                    "Почта": item.count_email,
                    "Статус ответа": item.answer_status,
                    "Можно ли боту": item.bot_permission,
                    "Нужные актуальные факты": " | ".join(item.required_fact_keys),
                    "Что делать без факта": item.fallback_when_fact_missing or "",
                    "Идеальный ответ РОПа": "",
                    "Решение РОПа": "",
                    "Комментарий": "",
                }
            )
    write_xlsx(path, "rop_review", rows)


def write_xlsx(path: Path, sheet_name: str, rows: Sequence[Mapping[str, Any]]) -> None:
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = sheet_name[:31]
    header_fill = PatternFill("solid", fgColor="EAF2F8")
    header_font = Font(bold=True)
    wrap_top = Alignment(wrap_text=True, vertical="top")
    fields = list(rows[0].keys()) if rows else ["empty"]
    for col, field in enumerate(fields, start=1):
        cell = worksheet.cell(row=1, column=col, value=field)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = wrap_top
        worksheet.column_dimensions[cell.column_letter].width = min(max(len(field) + 2, 14), 42)
    for row_index, row in enumerate(rows, start=1):
        for col, field in enumerate(fields, start=1):
            cell = worksheet.cell(row=row_index + 1, column=col, value=row.get(field, ""))
            cell.alignment = wrap_top
    worksheet.freeze_panes = "A2"
    worksheet.auto_filter.ref = worksheet.dimensions
    workbook.save(path)


def flatten_class(item: QuestionClass) -> Mapping[str, Any]:
    return {
        "question_class_id": item.question_class_id,
        "canonical_question": item.canonical_question,
        "narrow_scope": item.narrow_scope,
        "exclusions": item.exclusions,
        "examples_redacted": " | ".join(item.examples_redacted),
        "count_total": item.count_total,
        "count_calls": item.count_calls,
        "count_telegram": item.count_telegram,
        "count_email": item.count_email,
        "first_seen_at": item.first_seen_at.isoformat() if item.first_seen_at else "",
        "last_seen_at": item.last_seen_at.isoformat() if item.last_seen_at else "",
        "products": " | ".join(item.products),
        "grades": " | ".join(item.grades),
        "subjects": " | ".join(item.subjects),
        "answer_status": item.answer_status,
        "answer_template_id": item.answer_template_id or "",
        "required_fact_keys": " | ".join(item.required_fact_keys),
        "fact_source_refs": " | ".join(item.fact_source_refs),
        "fact_freshness_policy": item.fact_freshness_policy or "",
        "fallback_when_fact_missing": item.fallback_when_fact_missing or "",
        "bot_permission": item.bot_permission,
        "manager_handoff_reason": item.manager_handoff_reason or "",
        "rop_review_priority": item.rop_review_priority,
    }


def flatten_template(item: AnswerTemplate) -> Mapping[str, Any]:
    return {
        "answer_template_id": item.answer_template_id,
        "question_class_id": item.question_class_id,
        "template_text": item.template_text,
        "required_fact_keys": " | ".join(item.required_fact_keys),
        "approval_status": item.approval_status,
        "bot_permission_if_facts_fresh": item.bot_permission_if_facts_fresh,
        "fallback_when_fact_missing": item.fallback_when_fact_missing,
    }


def fact_requirement_rows(classes: Sequence[QuestionClass]) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for item in classes:
        for fact_key in item.required_fact_keys:
            rows.append(
                {
                    "question_class_id": item.question_class_id,
                    "canonical_question": item.canonical_question,
                    "fact_key": fact_key,
                    "freshness_policy": item.fact_freshness_policy or "",
                    "fallback_when_missing": item.fallback_when_fact_missing or "",
                    "bot_permission": item.bot_permission,
                }
            )
    return rows


def build_summary(
    *,
    config: CatalogBuildConfig,
    items: Sequence[QuestionItem],
    classes: Sequence[QuestionClass],
    templates: Sequence[AnswerTemplate],
    fact_sources: Sequence[CurrentFactSource],
    source_reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_source = Counter(item.source_channel for item in items)
    by_status = Counter(item.answer_status for item in classes)
    dynamic_classes = [item for item in classes if item.required_fact_keys]
    return {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tenant_id": config.tenant_id,
        "since": config.since.isoformat(),
        "safety": question_catalog_safety_contract(),
        "contract_inventory": question_catalog_contract_inventory(),
        "totals": {
            "question_items": len(items),
            "question_classes": len(classes),
            "answer_templates": len(templates),
            "fact_sources": len(fact_sources),
            "dynamic_fact_classes": len(dynamic_classes),
        },
        "counts": {
            "items_by_source": dict(sorted(by_source.items())),
            "classes_by_answer_status": dict(sorted(by_status.items())),
            "top_classes": [
                {
                    "question_class_id": item.question_class_id,
                    "canonical_question": item.canonical_question,
                    "count_total": item.count_total,
                    "answer_status": item.answer_status,
                    "required_fact_keys": list(item.required_fact_keys),
                }
                for item in sorted(classes, key=lambda row: (-row.count_total, row.canonical_question))[:25]
            ],
        },
        "source_reports": list(source_reports),
        "limitations": [
            "Каталог построен только по доступным локальным артефактам.",
            "Почта и Telegram не читаются live в этом проходе.",
            "Цены, расписание, скидки и наборы требуют отдельной проверки свежести фактов перед ответом бота.",
            "Все примеры в выходных файлах должны быть очищены от прямых контактов и персональных данных.",
        ],
    }


def render_source_coverage(source_reports: Sequence[Mapping[str, Any]], summary: Mapping[str, Any]) -> str:
    lines = [
        "# Покрытие источников единого каталога вопросов",
        "",
        f"Дата сборки: {summary.get('generated_at')}",
        f"Период с: {summary.get('since')}",
        "",
        "## Итоги",
        "",
        f"- Отдельных вопросов: {summary['totals']['question_items']}",
        f"- Классов вопросов: {summary['totals']['question_classes']}",
        f"- Классов с актуальными фактами: {summary['totals']['dynamic_fact_classes']}",
        "",
        "## Источники",
        "",
    ]
    for report in source_reports:
        lines.extend(
            [
                f"### {report.get('source_id')}",
                "",
                f"- Статус: {report.get('status')}",
                f"- Путь: `{report.get('path')}`",
                f"- Извлечено вопросов: {report.get('items_extracted', 0)}",
                f"- Всего строк/сообщений: {report.get('rows_total', report.get('archives_total', 'н/д'))}",
                "",
            ]
        )
    lines.extend(
        [
            "## Ограничения",
            "",
            "- Live-чтение почты, Telegram и CRM не выполнялось.",
            "- Повторное распознавание и анализ звонков не запускались.",
            "- Для цен и расписания нужен отдельный утвержденный файл фактов.",
            "",
        ]
    )
    return "\n".join(lines)


def _dedupe_items(items: Sequence[QuestionItem]) -> list[QuestionItem]:
    result: list[QuestionItem] = []
    seen: set[str] = set()
    for item in items:
        key = item.question_item_id
        if key in seen:
            continue
        seen.add(key)
        assert_public_text_safe(item.customer_text_redacted, field_name="customer_text_redacted")
        if item.manager_text_redacted:
            assert_public_text_safe(item.manager_text_redacted, field_name="manager_text_redacted")
        result.append(item)
    return result


def _unique(values: Sequence[Any], *, limit: int) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)
        if len(result) >= limit:
            break
    return tuple(result)


def _priority_for_class(items: Sequence[QuestionItem], required_fact_keys: Sequence[str]) -> str:
    if len(items) >= 25 or required_fact_keys:
        return "high"
    if len(items) >= 5:
        return "medium"
    return "low"


def _fact_source_refs(required_fact_keys: Sequence[str]) -> tuple[str, ...]:
    return tuple(f"current_fact_source_registry:{key}" for key in required_fact_keys)


def _template_text_for_class(item: QuestionClass) -> str:
    if item.required_fact_keys:
        placeholders = ", ".join("{" + key.replace(".current", "") + "}" for key in item.required_fact_keys)
        return (
            f"По вопросу «{item.canonical_question}» ориентируемся на актуальные данные: {placeholders}. "
            "Перед ответом нужно проверить свежий файл фактов. Если данные не подтверждены, передаем менеджеру."
        )
    if item.answer_status == ANSWER_STATUS_MANAGER_ONLY:
        return (
            f"По вопросу «{item.canonical_question}» бот не отвечает сам. "
            "Нужно передать менеджеру и зафиксировать, какие данные требуется уточнить."
        )
    return (
        f"По вопросу «{item.canonical_question}» дать короткий ответ по утвержденной базе знаний, "
        "затем задать один уточняющий вопрос и предложить помощь менеджера."
    )


def _fact_types_from_path(path: Path) -> tuple[str, ...]:
    text = str(path).casefold()
    fact_types: list[str] = []
    if any(marker in text for marker in ("стоим", "цен", "price", "абонем")):
        fact_types.append("price")
    if any(marker in text for marker in ("распис", "schedule")):
        fact_types.append("schedule")
    if any(marker in text for marker in ("скид", "discount", "акци")):
        fact_types.append("discount")
    if any(marker in text for marker in ("рассроч", "installment", "долями")):
        fact_types.append("installment")
    if any(marker in text for marker in ("адрес", "location", "филиал")):
        fact_types.append("location")
    if any(marker in text for marker in ("пробн", "trial")):
        fact_types.append("trial")
    if any(marker in text for marker in ("документ", "договор", "правила", "оферт")):
        fact_types.append("documents")
    if any(marker in text for marker in ("программ", "курс", "предмет")):
        fact_types.append("program")
    return tuple(dict.fromkeys(fact_types))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build read-only customer question catalog")
    parser.add_argument("--project-root", default=".", help="Project root")
    parser.add_argument("--out-root", default=None, help="Output root, defaults to product_data/question_catalog")
    parser.add_argument("--tenant-id", default="foton")
    parser.add_argument("--since", default="2025-01-01T00:00:00+00:00")
    parser.add_argument("--calls-enriched-reviews", default=None)
    parser.add_argument("--telegram-messages-jsonl", default=None)
    parser.add_argument("--mail-archive-root", default=None)
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> CatalogBuildConfig:
    project_root = Path(args.project_root).resolve()
    config = default_config(project_root, Path(args.out_root).resolve() if args.out_root else None)
    since = datetime.fromisoformat(str(args.since).replace("Z", "+00:00"))
    if since.tzinfo is None or since.utcoffset() is None:
        since = since.replace(tzinfo=timezone.utc)
    return CatalogBuildConfig(
        project_root=project_root,
        out_root=Path(args.out_root).resolve() if args.out_root else config.out_root,
        tenant_id=args.tenant_id,
        since=since,
        calls_enriched_reviews=Path(args.calls_enriched_reviews).resolve()
        if args.calls_enriched_reviews
        else config.calls_enriched_reviews,
        telegram_messages_jsonl=Path(args.telegram_messages_jsonl).resolve()
        if args.telegram_messages_jsonl
        else config.telegram_messages_jsonl,
        mail_archive_root=Path(args.mail_archive_root).resolve() if args.mail_archive_root else config.mail_archive_root,
        fact_source_roots=config.fact_source_roots,
    )
