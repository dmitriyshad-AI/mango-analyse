#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from mango_mvp.knowledge_base.fact_registry import (
    DEFAULT_KC_DOCX_PATH,
    DEFAULT_KC_SNAPSHOT_OUT_ROOT,
    DEFAULT_KC_SNAPSHOT_RUN_ID,
    FACT_TYPE_DOCUMENTS,
    FACT_TYPE_MANAGER_INSTRUCTION,
    FACT_TYPE_PAYMENT_METHODS,
    FACT_TYPE_PRICE,
    FACT_TYPE_PROGRAM,
    FACT_TYPE_RESTRICTION,
    FACT_TYPE_SCHEDULE,
    FRESHNESS_METADATA_ONLY,
    build_kc_knowledge_snapshot,
    classify_fact_types,
    default_google_drive_price_sources,
    guard_kc_snapshot_output_root,
    sha256_text,
    write_kc_knowledge_snapshot_outputs,
)


RUN_ID = DEFAULT_KC_SNAPSHOT_RUN_ID
DEFAULT_OUT_DIR = DEFAULT_KC_SNAPSHOT_OUT_ROOT


DRIVE_SEED_RECORDS = (
    {
        "source_id": "drive:base_kc",
        "title": "База знаний КЦ",
        "mime_type": "application/vnd.google-apps.document",
        "url": "https://docs.google.com/document/d/1bMhN0DtqNK8Z2XdwGMci2lAv0CtSYQ4QGb1Hr4dQ9Oo",
        "path": "Google Drive/База знаний КЦ",
        "drive_file_id": "1bMhN0DtqNK8Z2XdwGMci2lAv0CtSYQ4QGb1Hr4dQ9Oo",
        "modified_time": "2026-04-23T13:25:08.711Z",
        "processing_status": "metadata_only",
        "fact_types": ["manager_instruction", "restriction", "price", "schedule", "documents", "program"],
        "freshness_status": "metadata_only",
        "approval_status": "not_approved",
        "usable_for_precise_answer": False,
        "notes": "Проверен read-only доступ через Google Drive MCP. Для точных ответов нужен экспорт и утверждение.",
    },
    {
        "source_id": "drive:unpk_price_2026_2027",
        "title": "УНПК Стоимость обучения и порядок оплаты на 26/2027 уч г от 16.03.26",
        "mime_type": "application/vnd.google-apps.document",
        "url": "https://docs.google.com/document/d/12lsPcxpkP8Kf7y3dsLraYA_o9voXkq3Wg-KQEOF7YzI",
        "path": "Стоимость обучения по годам/2026-2027 уч г/УНПК",
        "drive_file_id": "12lsPcxpkP8Kf7y3dsLraYA_o9voXkq3Wg-KQEOF7YzI",
        "modified_time": "2026-03-17T08:07:44.383Z",
        "processing_status": "metadata_only",
        "fact_types": ["price", "payment_methods"],
        "freshness_status": "metadata_only",
        "approval_status": "not_approved",
        "usable_for_precise_answer": False,
        "notes": "Документ найден в Google Drive. До извлечения и проверки точные цены блокируются.",
    },
    {
        "source_id": "drive:foton_price_2026_2027",
        "title": "ФОТОН Стоимость обучения и порядок оплаты на 26/27 уч от 16.03.26",
        "mime_type": "application/vnd.google-apps.document",
        "url": "https://docs.google.com/document/d/1k0hzS8cZjD2NXeE5mcjlMTEOtTVINlTstZpWOUI1gSc",
        "path": "Стоимость обучения по годам/2026-2027 уч г/ФОТОН",
        "drive_file_id": "1k0hzS8cZjD2NXeE5mcjlMTEOtTVINlTstZpWOUI1gSc",
        "modified_time": "2026-03-17T08:03:02.111Z",
        "processing_status": "metadata_only",
        "fact_types": ["price", "payment_methods"],
        "freshness_status": "metadata_only",
        "approval_status": "not_approved",
        "usable_for_precise_answer": False,
        "notes": "Документ найден в Google Drive. До извлечения и проверки точные цены блокируются.",
    },
    {
        "source_id": "drive:internal_actual_docs_folder",
        "title": "Внутренние документы с актуальной информацией",
        "mime_type": "application/vnd.google-apps.folder",
        "url": "https://drive.google.com/drive/folders/15fYbkrGX1XOuSDX7rXs9Xi-88LxlsfCo",
        "path": "Google Drive/Внутренние документы с актуальной информацией",
        "drive_file_id": "15fYbkrGX1XOuSDX7rXs9Xi-88LxlsfCo",
        "processing_status": "metadata_only",
        "fact_types": ["documents", "program", "manager_instruction"],
        "freshness_status": "metadata_only",
        "approval_status": "not_approved",
        "usable_for_precise_answer": False,
        "notes": "Папка доступна read-only; содержимое требует отдельного export/inventory прохода.",
    },
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build KC knowledge snapshot for Telegram pilot.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--out-root", "--out-dir", dest="out_root", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--kc-docx", type=Path, default=DEFAULT_KC_DOCX_PATH)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--generated-at", default="")
    parser.add_argument("--max-docx-sections", type=int, default=80)
    parser.add_argument("--max-chars-per-section", type=int, default=700)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    project_root = args.project_root.expanduser().resolve()
    out_root = guard_kc_snapshot_output_root(args.out_root, project_root=project_root)
    kc_docx = args.kc_docx.expanduser()
    if not kc_docx.is_absolute():
        kc_docx = project_root / kc_docx
    snapshot = build_kc_knowledge_snapshot(
        kc_docx_path=kc_docx.resolve(),
        max_docx_sections=max(1, args.max_docx_sections),
        max_chars_per_section=max(80, args.max_chars_per_section),
        run_id=args.run_id,
        generated_at=args.generated_at or None,
        project_root=project_root,
    )
    snapshot = enrich_snapshot_with_fact_candidates(snapshot)
    if args.dry_run:
        result = {
            "dry_run": True,
            "out_root": str(out_root),
            "run_id": snapshot["run_id"],
            "summary": snapshot["summary"],
            "safety": snapshot["safety"],
        }
    else:
        result = write_kc_knowledge_snapshot_outputs(out_root, snapshot, project_root=project_root)
        write_kc_extra_reports(out_root, snapshot)
        result.update(
            {
                "facts_jsonl": str(out_root / "facts.jsonl"),
                "facts_csv": str(out_root / "facts.csv"),
                "facts_summary_md": str(out_root / "facts_summary.md"),
                "conflicts_and_gaps_md": str(out_root / "conflicts_and_gaps.md"),
                "chunk_index_summary_md": str(out_root / "chunk_index_summary.md"),
                "source_inventory_summary_md": str(out_root / "source_inventory_summary.md"),
            }
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def enrich_snapshot_with_fact_candidates(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    enriched = dict(snapshot)
    inventory_records = list(enriched.get("source_inventory") or [])
    chunks = list(enriched.get("chunks") or [])
    facts = build_fact_candidates(enriched, inventory_records)
    summary = dict(enriched.get("summary") or {})
    summary.update(
        {
            "run_id": enriched.get("run_id") or RUN_ID,
            "facts_total": len(facts),
            "usable_for_precise_answer": sum(1 for fact in facts if fact.get("usable_for_precise_answer")),
            "needs_manager_confirmation": sum(1 for fact in facts if fact.get("requires_manager_confirmation")),
            "chunks_total": len(chunks),
            "sources_total": len(inventory_records),
            "drive_sources_total": sum(
                1 for source in inventory_records if "google_drive" in str(source.get("source_kind") or source.get("source_type") or "")
            ),
            "safe_for_stage6_dry_run": True,
        }
    )
    enriched["facts"] = facts
    enriched["freshness_blocks"] = list(enriched.get("freshness_blocks") or []) + build_default_freshness_blocks()
    enriched["summary"] = summary
    return enriched


def write_kc_extra_reports(out_root: Path, snapshot: Mapping[str, Any]) -> None:
    facts = list(snapshot.get("facts") or [])
    chunks = list(snapshot.get("chunks") or [])
    sources = list(snapshot.get("source_inventory") or [])
    write_jsonl(out_root / "facts.jsonl", facts)
    write_csv(out_root / "facts.csv", facts)
    (out_root / "facts_summary.md").write_text(render_quality_summary(snapshot), encoding="utf-8")
    (out_root / "conflicts_and_gaps.md").write_text(render_conflicts_and_gaps(snapshot), encoding="utf-8")
    (out_root / "chunk_index_summary.md").write_text(render_chunk_index_summary(chunks), encoding="utf-8")
    (out_root / "source_inventory_summary.md").write_text(render_source_inventory_summary(sources), encoding="utf-8")


def build_fact_candidates(snapshot: Mapping[str, Any], inventory_records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    for chunk in snapshot.get("chunks", []):
        text = str(chunk.get("text") or "")
        fact_types = list(chunk.get("fact_types") or classify_fact_types(f"{chunk.get('title', '')} {text}"))
        facts.append(
            {
                "fact_id": f"fact:{chunk.get('chunk_id', 'chunk')}",
                "fact_type": fact_types[0] if fact_types else FACT_TYPE_MANAGER_INSTRUCTION,
                "short_fact": text[:350],
                "manager_text": text[:700],
                "client_safe_text": text[:350] if is_client_safe_fact_type(fact_types) else "",
                "source_id": chunk.get("source_id"),
                "source_title": chunk.get("title"),
                "source_updated_at": "",
                "freshness_status": "unknown",
                "usable_for_precise_answer": False,
                "requires_manager_confirmation": True,
                "forbidden_for_client": not is_client_safe_fact_type(fact_types),
                "related_theme_ids": [],
                "risk_notes": "DOCX chunk extracted locally; requires manager/ROP validation before precise client answer.",
                "sha256_text": sha256_text(text),
                "bot_permission": "draft_only_needs_review",
            }
        )
    for record in inventory_records:
        facts.append(
            {
                "fact_id": f"fact:{record.get('source_id')}:metadata",
                "fact_type": (record.get("fact_types") or [FACT_TYPE_MANAGER_INSTRUCTION])[0],
                "short_fact": f"Источник найден: {record.get('title')}",
                "manager_text": record.get("notes") or "Источник найден, содержимое требует проверки.",
                "client_safe_text": "",
                "source_id": record.get("source_id"),
                "source_title": record.get("title"),
                "source_updated_at": record.get("modified_time") or "",
                "freshness_status": FRESHNESS_METADATA_ONLY,
                "usable_for_precise_answer": False,
                "requires_manager_confirmation": True,
                "forbidden_for_client": True,
                "related_theme_ids": [],
                "risk_notes": "Metadata-only Google Drive source: do not answer precise facts.",
                "sha256_text": record.get("sha256_text") or "",
                "bot_permission": "manager_only",
            }
        )
    return facts


def build_default_freshness_blocks() -> list[dict[str, Any]]:
    return [
        {
            "fact_key": "prices.current",
            "fact_type": FACT_TYPE_PRICE,
            "reason": "google_drive_price_docs_metadata_only",
            "blocks_precise_answer": True,
            "safe_instruction": "Не называть точную цену, скидку или срок оплаты без свежего подтвержденного прайса.",
        },
        {
            "fact_key": "schedule.current",
            "fact_type": FACT_TYPE_SCHEDULE,
            "reason": "schedule_not_confirmed",
            "blocks_precise_answer": True,
            "safe_instruction": "Не называть точное расписание; использовать безопасный шаблон и follow-up менеджера.",
        },
        {
            "fact_key": "documents.current",
            "fact_type": FACT_TYPE_DOCUMENTS,
            "reason": "documents_need_manager_confirmation",
            "blocks_precise_answer": True,
            "safe_instruction": "Документы, возвраты, налоговые и юридические темы только через менеджера.",
        },
    ]


def build_summary(
    snapshot: Mapping[str, Any],
    inventory_payload: Mapping[str, Any],
    facts: Sequence[Mapping[str, Any]],
    chunks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "run_id": RUN_ID,
        "sources_total": len(snapshot.get("sources", [])) + len(inventory_payload.get("records", [])),
        "drive_sources_total": inventory_payload.get("summary", {}).get("records_total", 0),
        "chunks_total": len(chunks),
        "facts_total": len(facts),
        "usable_for_precise_answer": sum(1 for fact in facts if fact.get("usable_for_precise_answer")),
        "needs_manager_confirmation": sum(1 for fact in facts if fact.get("requires_manager_confirmation")),
        "metadata_only_sources": inventory_payload.get("summary", {}).get("metadata_only", 0),
        "docx_parsed_for_sections": snapshot.get("metadata", {}).get("docx_parsed_for_sections", False),
        "safe_for_stage6_dry_run": True,
    }


def render_quality_summary(snapshot: Mapping[str, Any]) -> str:
    summary = snapshot["summary"]
    lines = [
        "# KC knowledge base night build: summary",
        "",
        f"- run_id: `{summary['run_id']}`",
        f"- sources_total: {summary['sources_total']}",
        f"- drive_sources_total: {summary['drive_sources_total']}",
        f"- chunks_total: {summary['chunks_total']}",
        f"- facts_total: {summary['facts_total']}",
        f"- usable_for_precise_answer: {summary['usable_for_precise_answer']}",
        f"- needs_manager_confirmation: {summary['needs_manager_confirmation']}",
        "",
        "Точные цены, расписание и документы остаются заблокированы до свежего подтверждения.",
    ]
    return "\n".join(lines) + "\n"


def render_conflicts_and_gaps(snapshot: Mapping[str, Any]) -> str:
    return (
        "# Conflicts and gaps\n\n"
        "- Google Drive price docs найдены, но текущий машинный snapshot хранит их как metadata-only.\n"
        "- Точные цены и скидки нельзя давать клиенту до полного export/проверки.\n"
        "- Расписание не считается точным фактом; нужен safe schedule шаблон и follow-up менеджера.\n"
        "- Реальные ответы менеджеров не являются источником истины и должны проходить отдельный playbook слой.\n"
    )


def render_chunk_index_summary(chunks: Sequence[Mapping[str, Any]]) -> str:
    return (
        "# Chunk index summary\n\n"
        f"- chunks_total: {len(chunks)}\n"
        "- max_chunk_chars: 700\n"
        "- prompt получает только короткие фрагменты, не полный документ.\n"
    )


def render_source_inventory_summary(sources: Sequence[Mapping[str, Any]]) -> str:
    read_count = sum(1 for source in sources if str(source.get("read_status") or "") == "read")
    metadata_count = sum(1 for source in sources if str(source.get("read_status") or "") == "metadata_only")
    return (
        "# Source inventory summary\n\n"
        f"- sources_total: {len(sources)}\n"
        f"- read_sources: {read_count}\n"
        f"- metadata_only_sources: {metadata_count}\n"
        "- Точные ответы разрешаются только после свежего подтверждения источника.\n"
    )


def is_client_safe_fact_type(fact_types: Sequence[str]) -> bool:
    unsafe = {FACT_TYPE_MANAGER_INSTRUCTION, FACT_TYPE_RESTRICTION, FACT_TYPE_PRICE, FACT_TYPE_PAYMENT_METHODS, FACT_TYPE_SCHEDULE}
    return not (set(fact_types) & unsafe) and FACT_TYPE_PROGRAM in set(fact_types)


def assert_safe_output_path(path: Path) -> None:
    resolved = path.resolve()
    if "stable_runtime" in resolved.parts:
        raise ValueError("Refusing to write KC knowledge snapshot under stable_runtime")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: serialize_cell(row.get(key)) for key in fieldnames})


def serialize_cell(value: Any) -> str:
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value or "")


if __name__ == "__main__":
    raise SystemExit(main())
