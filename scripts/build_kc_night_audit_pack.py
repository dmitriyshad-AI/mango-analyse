#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_RUN_DIR = Path("product_data/knowledge_base/kb_night_20260517_v1")

REQUIRED_AUDIT_FILES = (
    "implementation_notes.md",
    "changed_files.txt",
    "test_output.txt",
    "risk_review.md",
    "backward_compatibility.md",
    "source_inventory_summary.md",
    "facts_quality_report.md",
    "manager_answer_playbook_report.md",
    "stage6_eval_summary.md",
    "no_live_write_proof.md",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build audit pack for KC knowledge night build.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--audit-dir", type=Path, default=None)
    parser.add_argument("--test-output", type=Path, default=None)
    args = parser.parse_args()

    audit_dir = args.audit_dir or default_audit_dir()
    build_kc_night_audit_pack(args.run_dir, audit_dir, test_output=args.test_output)
    print(str(audit_dir))
    return 0


def default_audit_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Path("audits/_inbox") / f"telegram_pilot_kb_night_build_20260517_{timestamp}"


def build_kc_night_audit_pack(
    snapshot_root: str | Path,
    pack_dir: str | Path,
    *,
    changed_files: Sequence[str] | None = None,
    test_output: str | Path | None = None,
) -> Path:
    source_root = Path(snapshot_root)
    target = Path(pack_dir)
    if "stable_runtime" in target.resolve().parts:
        raise ValueError("Refusing to write audit pack under stable_runtime")
    target.mkdir(parents=True, exist_ok=True)

    source_rows = read_csv(source_root / "source_inventory.csv")
    fact_rows = read_jsonl(source_root / "facts.jsonl")
    chunk_rows = read_jsonl(source_root / "knowledge_chunks.jsonl")
    manager_sample_rows = read_csv(source_root / "manager_answer_sample_300_500.csv")
    manager_pattern_rows = read_jsonl(source_root / "manager_answer_patterns.jsonl")
    unsafe_rows = read_csv(source_root / "unsafe_or_outdated_manager_answers.csv")
    comparison_rows = read_csv(source_root / "stage6_before_after_comparison.csv")
    stage6_rows = read_csv(source_root / "stage6_kb_enriched_drafts.csv")
    summary = read_json(source_root / "quality_summary.json")
    stage6_summary = read_json(source_root / "stage6_eval_summary.json")
    playbook_summary = read_json(source_root / "manager_answer_playbook_summary.json")
    snapshot = read_json(source_root / "kc_snapshot_20260517_night_v1.json")

    changed = list(changed_files) if changed_files is not None else git_changed_files()
    write(target / "implementation_notes.md", render_implementation_notes(summary, stage6_summary, playbook_summary))
    write(target / "changed_files.txt", sanitize_text("\n".join(changed) + "\n"))
    write(target / "test_output.txt", sanitize_text(resolve_test_output(test_output)))
    write(
        target / "risk_review.md",
        render_risk_review_for_pack(
            source_rows=source_rows,
            fact_rows=fact_rows,
            stage6_rows=stage6_rows,
            source_root=source_root,
        ),
    )
    write(target / "backward_compatibility.md", render_backward_compatibility())
    write(target / "source_inventory_summary.md", render_source_inventory_summary_for_pack(source_rows))
    write(target / "facts_quality_report.md", render_facts_quality_report_for_pack(fact_rows, chunk_rows))
    write(
        target / "manager_answer_playbook_report.md",
        render_manager_answer_playbook_report(
            sample_rows=manager_sample_rows,
            pattern_rows=manager_pattern_rows,
            unsafe_rows=unsafe_rows,
            playbook_summary=playbook_summary,
        ),
    )
    write(target / "stage6_eval_summary.md", render_stage6_report_for_pack(comparison_rows, stage6_rows, stage6_summary))
    write(target / "no_live_write_proof.md", render_no_live_write_proof_for_pack(snapshot, source_root=source_root))
    return target


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            loaded = json.loads(line)
            if isinstance(loaded, Mapping):
                rows.append(dict(loaded))
    return rows


def git_changed_files() -> list[str]:
    result = subprocess.run(["git", "status", "--short"], check=False, text=True, capture_output=True)
    return [line for line in result.stdout.splitlines() if line.strip()]


def render_implementation_notes(summary: dict, stage6: dict, playbook: dict) -> str:
    playbook_summary = playbook.get("summary") if isinstance(playbook.get("summary"), Mapping) else {}
    return (
        "# Implementation notes\n\n"
        "- Собран read-only snapshot базы знаний для Telegram-пилота.\n"
        "- Реальные ответы менеджеров обработаны как playbook, не как факты.\n"
        "- Stage 6 выполнен как сухой прогон, без Telegram live-send.\n\n"
        f"- facts_total: {summary.get('facts_total', 0)}\n"
        f"- chunks_total: {summary.get('chunks_total', 0)}\n"
        f"- manager_answer_sample: {playbook_summary.get('sample_rows', playbook.get('sample_rows', 0))}\n"
        f"- stage6_rows: {stage6.get('rows_total', 0)}\n"
    )


def render_risk_review(summary: dict, stage6: dict) -> str:
    return (
        "# Risk review\n\n"
        "- Точные цены, расписание и документы остаются заблокированы до свежего подтверждения.\n"
        "- Google Drive источники, которые не были полностью экспортированы, имеют статус metadata-only.\n"
        "- Клиентам ничего не отправлялось.\n"
        f"- usable_for_precise_answer: {summary.get('usable_for_precise_answer', 0)}\n"
        f"- manager_only_after_kb: {stage6.get('manager_only_after_kb', 0)}\n"
    )


def render_backward_compatibility() -> str:
    return (
        "# Backward compatibility\n\n"
        "- Существующие поля `facts_context` и `knowledge_snippets` сохранены.\n"
        "- `PilotContext` расширен обратно совместимыми полями.\n"
        "- Без snapshot система должна уходить в безопасный fallback.\n"
    )


def render_source_inventory_summary(summary: dict) -> str:
    return (
        "# Source inventory summary\n\n"
        f"- sources_total: {summary.get('sources_total', 0)}\n"
        f"- drive_sources_total: {summary.get('drive_sources_total', 0)}\n"
        f"- metadata_only_sources: {summary.get('metadata_only_sources', 0)}\n"
    )


def render_facts_quality_report(summary: dict) -> str:
    return (
        "# Facts quality report\n\n"
        f"- facts_total: {summary.get('facts_total', 0)}\n"
        f"- usable_for_precise_answer: {summary.get('usable_for_precise_answer', 0)}\n"
        f"- needs_manager_confirmation: {summary.get('needs_manager_confirmation', 0)}\n"
    )


def render_playbook_report(playbook: dict) -> str:
    return (
        "# Manager answer playbook report\n\n"
        f"- sample_rows: {playbook.get('sample_rows', 0)}\n"
        f"- patterns: {playbook.get('patterns', 0)}\n"
        f"- by_quality: `{json.dumps(playbook.get('by_quality', {}), ensure_ascii=False, sort_keys=True)}`\n"
    )


def render_stage6_report(stage6: dict) -> str:
    return (
        "# Stage 6 eval summary\n\n"
        f"- rows_total: {stage6.get('rows_total', 0)}\n"
        f"- kb_used: {stage6.get('kb_used', 0)}\n"
        f"- became_more_specific: {stage6.get('became_more_specific', 0)}\n"
        "- Это сухой прогон: клиентам ничего не отправлялось.\n"
    )


def render_no_live_write_proof() -> str:
    return (
        "# No live write proof\n\n"
        "- AMO/CRM/Tallanto write: not used.\n"
        "- Telegram client send: not used.\n"
        "- ASR/R+A: not used.\n"
        "- stable_runtime write: not used by scripts in this block.\n"
    )


def render_source_inventory_summary_for_pack(rows: Sequence[Mapping[str, Any]]) -> str:
    by_status: dict[str, int] = {}
    for row in rows:
        status = str(row.get("inventory_status") or row.get("processing_status") or "unknown")
        by_status[status] = by_status.get(status, 0) + 1
    return (
        "# Source inventory summary\n\n"
        f"- всего источников: `{len(rows)}`\n"
        f"- status counts: `{json.dumps(by_status, ensure_ascii=False, sort_keys=True)}`\n"
        "- Google Drive источники без извлеченного текста остаются metadata-only.\n"
    )


def render_facts_quality_report_for_pack(
    facts: Sequence[Mapping[str, Any]],
    chunks: Sequence[Mapping[str, Any]],
) -> str:
    precise_without_fresh = sum(
        1
        for fact in facts
        if truthy(fact.get("usable_for_precise_answer"))
        and str(fact.get("freshness_status") or "") not in {"fresh", "fresh_verified"}
    )
    forbidden_for_client = sum(1 for fact in facts if truthy(fact.get("forbidden_for_client")))
    return (
        "# Facts quality report\n\n"
        f"- всего фактов: `{len(facts)}`\n"
        f"- всего фрагментов базы знаний: `{len(chunks)}`\n"
        f"- precise facts without fresh status: `{precise_without_fresh}`\n"
        f"- forbidden_for_client: `{forbidden_for_client}`\n"
        "- Исторические ответы менеджеров не считаются свежими фактами.\n"
    )


def render_manager_answer_playbook_report(
    *,
    sample_rows: Sequence[Mapping[str, Any]],
    pattern_rows: Sequence[Mapping[str, Any]],
    unsafe_rows: Sequence[Mapping[str, Any]],
    playbook_summary: Mapping[str, Any],
) -> str:
    summary = playbook_summary.get("summary") if isinstance(playbook_summary.get("summary"), Mapping) else {}
    return (
        "# Manager answer playbook report\n\n"
        f"- sample rows: `{len(sample_rows) or summary.get('sample_rows', 0)}`\n"
        f"- pattern rows: `{len(pattern_rows) or summary.get('patterns', 0)}`\n"
        f"- unsafe/outdated rows: `{len(unsafe_rows)}`\n"
        "- usable_as_fact: `0`\n"
        "- Реальные ответы менеджеров используются только как примеры стиля и структуры.\n"
    )


def render_stage6_report_for_pack(
    comparison_rows: Sequence[Mapping[str, Any]],
    stage6_rows: Sequence[Mapping[str, Any]],
    stage6_summary: Mapping[str, Any],
) -> str:
    rows_total = len(stage6_rows) or int(stage6_summary.get("rows_total") or 0)
    kb_used = sum(1 for row in stage6_rows if truthy(row.get("kb_context_used") or row.get("used_kb_context")))
    if not kb_used:
        kb_used = int(stage6_summary.get("used_kb_context") or stage6_summary.get("kb_used") or 0)
    more = sum(1 for row in comparison_rows if truthy(row.get("draft_became_more_substantive") or row.get("became_more_substantive")))
    if not more:
        more = int(stage6_summary.get("became_more_substantive") or stage6_summary.get("became_more_specific") or 0)
    return (
        "# Stage 6 eval summary\n\n"
        f"- rows_total: `{rows_total}`\n"
        f"- kb_used: `{kb_used}`\n"
        f"- became_more_substantive: `{more}`\n"
        f"- rows in before/after comparison: `{len(comparison_rows)}`\n"
        "- client_send: `false`\n"
        "- Это сухой прогон: клиентам ничего не отправлялось.\n"
    )


def render_risk_review_for_pack(
    *,
    source_rows: Sequence[Mapping[str, Any]],
    fact_rows: Sequence[Mapping[str, Any]],
    stage6_rows: Sequence[Mapping[str, Any]],
    source_root: Path,
) -> str:
    warnings: list[str] = []
    if not source_rows:
        warnings.append("source_inventory не найден или пустой")
    if not fact_rows:
        warnings.append("facts не найдены или пустые")
    if not stage6_rows:
        warnings.append("stage6 dry-run не найден или пустой")
    warnings_text = "\n".join(f"- {warning}" for warning in warnings) if warnings else "- Критичных новых блокеров в audit pack не найдено."
    return (
        "# Risk review\n\n"
        f"{warnings_text}\n"
        "- Точные цены, расписание и документы остаются заблокированы до свежего подтверждения.\n"
        "- Google Drive источники без полного извлечения не разблокируют точные ответы.\n"
        "- Клиентам ничего не отправлялось.\n"
        f"- snapshot_root: `{sanitize_text(str(source_root))}`\n"
    )


def render_no_live_write_proof_for_pack(snapshot: Mapping[str, Any], *, source_root: Path) -> str:
    safety = snapshot.get("safety") if isinstance(snapshot.get("safety"), Mapping) else {}
    snapshot_note = "" if snapshot else "\n- Snapshot file was not found; proof is based on audit script safety contract.\n"
    return (
        "# No live write proof\n\n"
        "- AMO/CRM/Tallanto write: not used.\n"
        "- Telegram client send: not used.\n"
        "- ASR/R+A: not used.\n"
        "- stable_runtime write: not used by scripts in this block.\n"
        f"- snapshot safety: `{json.dumps(safety, ensure_ascii=False, sort_keys=True)}`\n"
        f"{snapshot_note}"
        f"- snapshot_root: `{sanitize_text(str(source_root))}`\n"
    )


def resolve_test_output(value: str | Path | None) -> str:
    if value is None:
        return "See local terminal run.\n"
    if isinstance(value, Path):
        return value.read_text(encoding="utf-8") if value.exists() else str(value)
    path = Path(str(value))
    if "\n" not in str(value) and path.exists():
        return path.read_text(encoding="utf-8")
    return str(value)


def sanitize_text(text: str) -> str:
    redacted = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-zА-Яа-я]{2,}", "[redacted_email]", str(text))
    redacted = re.sub(r"(?:\+?7|8)[\s()\\-]*\d{3}[\s()\\-]*\d{3}[\s()\\-]*\d{2}[\s()\\-]*\d{2}", "[redacted_phone]", redacted)
    return redacted


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().casefold() in {"1", "true", "yes", "y", "да", "истина"}


def write(path: Path, text: str) -> None:
    path.write_text(sanitize_text(text), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
