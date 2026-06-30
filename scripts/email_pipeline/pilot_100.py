#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
from scripts.email_pipeline.summary import SummaryItem, clean_body, mask_pii, summarize_items


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    source_root = Path(args.source_root).expanduser().resolve()
    prod_db = Path(args.prod_db).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve()

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
            body = read_text(message.extracted_text_path, limit=10000)
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
        row["summary_brand_mismatch"] = _summary_brand_mismatch(
            row["brand"], json.dumps(row["summary_payload"], ensure_ascii=False)
        )
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
        },
        ensure_ascii=False,
    ))
    return 0


def _record_for_message(message: ArchiveMessage, *, body: str, brand: Any) -> dict[str, Any]:
    return {
        "message_sha256": message.message_sha256,
        "source_archive": message.source_archive,
        "subject": message.subject,
        "mailbox": message.mailbox,
        "date_iso": message.date_iso,
        "direction": message.direction,
        "classification_reason": message.classification_reason,
        "body_chars": message.body_chars,
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
) -> str:
    selected_brand = Counter(row["brand"] for row in selected)
    selected_sources = Counter(row["brand_source"] for row in selected)
    all_real_brand = Counter(row["brand"] for row in real_records)
    raw_signal = Counter(row["raw_infer_offline_brand"] for row in selected)
    summary_mismatch = sum(1 for row in selected if row["summary_brand_mismatch"])
    bad_brand_sources = [row for row in selected if row["brand_source"] in {"folder", "from", "domain"}]
    git = git_block(repo_root)
    examples = selected[:7]
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
    lines.append("\nПо выбранным 100:")
    for key, value in selected_brand.most_common():
        lines.append(f"- {key}: {value}")
    lines.append("\nbrand_source по выбранным 100:")
    for key, value in selected_sources.most_common():
        lines.append(f"- {key}: {value}")
    lines.append("\nraw infer_offline_brand по выбранным 100:")
    for key, value in raw_signal.most_common():
        lines.append(f"- {key}: {value}")
    lines.append(f"\n- legacy_metadata_brand_source_count: {len(bad_brand_sources)}")
    lines.append(f"- summary brand mismatch: {summary_mismatch}")
    lines.append("\n## LLM-сводки\n")
    lines.append(f"- provider: {summary_result.provider}")
    lines.append(f"- model: {summary_result.model}")
    lines.append(f"- reasoning: {summary_result.reasoning}")
    lines.append(f"- llm_calls_total: {summary_result.llm_calls_total}")
    lines.append(f"- llm_calls_limit: <=100")
    lines.append("\n## 7 обезличенных примеров\n")
    for index, row in enumerate(examples, 1):
        summary_payload = row.get("summary_payload") or {}
        lines.append(f"\n### Пример {index}")
        lines.append(f"- sha: `{row['message_sha256'][:12]}`")
        lines.append(f"- direction: `{row['direction']}`")
        lines.append(f"- brand: `{row['brand']}` / source=`{row['brand_source']}`")
        lines.append(f"- subject: {mask_pii(row['subject'])[:180] or '(пусто)'}")
        lines.append(f"- body_fragment: {mask_pii(clean_body(row['body'], limit=320))[:320] or '(тело недоступно)'}")
        lines.append(f"- summary: {mask_pii(str(summary_payload.get('summary') or ''))}")
        lines.append(f"- topic: {mask_pii(str(summary_payload.get('topic') or ''))}")
        lines.append(f"- next_step: {mask_pii(str(summary_payload.get('next_step') or 'null'))}")
    lines.append("\n## Контрольные выводы\n")
    lines.append("- AMO/CRM/Tallanto/writeback не использовались.")
    lines.append("- Prod timeline открыт только `mode=ro`, `query_only=ON`; mtime/size не изменились.")
    lines.append("- В git добавлен только код `scripts/email_pipeline/` и тесты; архивы, письма, БД и ПДн не коммитятся.")
    lines.append("- Вердикт о проде не выносится; приёмка за Claude #1 по сырью.")
    return "\n".join(lines) + "\n"


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
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
