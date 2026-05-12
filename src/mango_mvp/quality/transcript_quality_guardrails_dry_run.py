from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from mango_mvp.config import get_settings
from mango_mvp.models import CallRecord
from mango_mvp.quality.non_conversation import detect_non_conversation_signals


CONTENTFUL_CALL_TYPES = {"sales_call", "service_call", "technical_call", "existing_client_progress"}


@dataclass(frozen=True)
class GuardrailsDryRunConfig:
    database_url: str
    out_root: Path
    limit: int | None = None
    analyzed_only: bool = False
    batch_size: int = 1000
    protected_sample_limit: int = 500
    excerpt_chars: int = 1200


def build_transcript_quality_guardrails_dry_run(config: GuardrailsDryRunConfig) -> dict[str, Any]:
    out_root = config.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    engine = create_engine(
        config.database_url,
        future=True,
        connect_args={"timeout": 30} if config.database_url.startswith("sqlite") else {},
    )
    session_factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)

    all_rows: list[dict[str, Any]] = []
    auto_fix_rows: list[dict[str, Any]] = []
    manual_review_rows: list[dict[str, Any]] = []
    protected_rows: list[dict[str, Any]] = []
    label_counts: Counter[str] = Counter()
    current_call_type_counts: Counter[str] = Counter()
    analysis_json_status_counts: Counter[str] = Counter()
    monthly: dict[str, Counter[str]] = {}
    call_type_summary: dict[str, Counter[str]] = {}

    scanned = 0
    eligible = 0
    skipped_empty_transcript = 0

    with session_factory() as session:
        query = select(CallRecord).where(CallRecord.transcript_text.is_not(None)).order_by(CallRecord.id.asc())
        if config.analyzed_only:
            query = query.where(CallRecord.analysis_json.is_not(None))
        if config.limit is not None:
            query = query.limit(max(0, int(config.limit)))

        result = session.execute(query.execution_options(yield_per=max(1, config.batch_size))).scalars()
        for call in result:
            scanned += 1
            transcript = _clean(call.transcript_text)
            if not transcript:
                skipped_empty_transcript += 1
                continue
            eligible += 1

            analysis, analysis_json_status = _safe_json_object(call.analysis_json)
            fields = _analysis_fields(analysis)
            signals = detect_non_conversation_signals(
                transcript_text=transcript,
                history_summary=fields["history_summary"],
                call_type=fields["call_type"],
                next_step=fields["next_step"],
                products=fields["products"],
                subjects=fields["subjects"],
                objections=fields["objections"],
                duration_sec=call.duration_sec,
            )

            month = _month(call.started_at)
            current_call_type = fields["call_type"] or "unknown"
            current_contentful = current_call_type in CONTENTFUL_CALL_TYPES
            row = {
                "id": call.id,
                "source_filename": _clean(call.source_filename),
                "started_at": _format_dt(call.started_at),
                "month": month,
                "phone": _clean(call.phone),
                "manager_name": _clean(call.manager_name),
                "duration_sec": call.duration_sec,
                "transcription_status": _clean(call.transcription_status),
                "resolve_status": _clean(call.resolve_status),
                "analysis_status": _clean(call.analysis_status),
                "analysis_json_status": analysis_json_status,
                "current_call_type": current_call_type,
                "current_contentful": current_contentful,
                "guardrail_label": signals.label,
                "guardrail_score": signals.score,
                "guardrail_reason_codes": "|".join(signals.reason_codes),
                "should_force_non_conversation": signals.should_force_non_conversation,
                "requires_manual_review": signals.requires_manual_review,
                "protected_live_dialogue": signals.protected_live_dialogue,
                "recommended_call_type": signals.recommended_call_type,
                "recommended_contentful": signals.recommended_contentful,
                "recommended_contact_subtype": signals.recommended_contact_subtype,
                "strong_no_live_marker": signals.strong_no_live_marker,
                "asr_artifact_marker": signals.asr_artifact_marker,
                "system_no_dialogue_phrase": signals.system_no_dialogue_phrase,
                "risky_keyword_marker": signals.risky_keyword_marker,
                "outbound_voicemail_marker": signals.outbound_voicemail_marker,
                "next_step": fields["next_step"],
                "products": "|".join(fields["products"]),
                "subjects": "|".join(fields["subjects"]),
                "objections": "|".join(fields["objections"]),
                "history_summary_excerpt": _excerpt(fields["history_summary"], config.excerpt_chars),
                "transcript_excerpt": _excerpt(transcript, config.excerpt_chars),
            }

            all_rows.append(row)
            if signals.should_force_non_conversation:
                auto_fix_rows.append(row)
            if signals.requires_manual_review:
                manual_review_rows.append(row)
            if signals.protected_live_dialogue and len(protected_rows) < config.protected_sample_limit:
                protected_rows.append(row)

            label_counts[signals.label] += 1
            current_call_type_counts[current_call_type] += 1
            analysis_json_status_counts[analysis_json_status] += 1
            month_counter = monthly.setdefault(month, Counter())
            month_counter["total"] += 1
            month_counter[f"label:{signals.label}"] += 1
            if signals.should_force_non_conversation:
                month_counter["auto_fix"] += 1
            if signals.requires_manual_review:
                month_counter["manual_review"] += 1
            if signals.protected_live_dialogue:
                month_counter["protected_live"] += 1

            call_type_counter = call_type_summary.setdefault(current_call_type, Counter())
            call_type_counter["total"] += 1
            call_type_counter[f"label:{signals.label}"] += 1
            if signals.should_force_non_conversation:
                call_type_counter["auto_fix"] += 1
            if signals.requires_manual_review:
                call_type_counter["manual_review"] += 1
            if signals.protected_live_dialogue:
                call_type_counter["protected_live"] += 1

    monthly_rows = _counter_table(monthly, key_name="month")
    call_type_rows = _counter_table(call_type_summary, key_name="current_call_type")

    outputs = {
        "summary_json": out_root / "summary.json",
        "report_markdown": out_root / "TRANSCRIPT_QUALITY_GUARDRAILS_DRY_RUN.md",
        "all_results_csv": out_root / "guardrails_all_results.csv",
        "auto_fix_candidates_csv": out_root / "auto_fix_candidates.csv",
        "manual_review_candidates_csv": out_root / "manual_review_candidates.csv",
        "protected_live_sample_csv": out_root / "protected_live_sample.csv",
        "monthly_summary_csv": out_root / "monthly_summary.csv",
        "call_type_summary_csv": out_root / "call_type_summary.csv",
    }
    _write_csv(outputs["all_results_csv"], all_rows)
    _write_csv(outputs["auto_fix_candidates_csv"], auto_fix_rows)
    _write_csv(outputs["manual_review_candidates_csv"], manual_review_rows)
    _write_csv(outputs["protected_live_sample_csv"], protected_rows)
    _write_csv(outputs["monthly_summary_csv"], monthly_rows)
    _write_csv(outputs["call_type_summary_csv"], call_type_rows)

    xlsx_path, xlsx_error = _write_xlsx_if_available(
        out_root / "transcript_quality_guardrails_dry_run.xlsx",
        summary_rows=[
            {"metric": "scanned_rows", "value": scanned},
            {"metric": "eligible_transcripts", "value": eligible},
            {"metric": "auto_fix_candidates", "value": len(auto_fix_rows)},
            {"metric": "manual_review_candidates", "value": len(manual_review_rows)},
            {"metric": "protected_live_dialogues", "value": sum(1 for row in all_rows if row["protected_live_dialogue"])},
        ],
        monthly_rows=monthly_rows,
        call_type_rows=call_type_rows,
        auto_fix_rows=auto_fix_rows,
        manual_review_rows=manual_review_rows,
        protected_rows=protected_rows,
    )

    summary: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "database_url": _redact_database_url(config.database_url),
        "mode": "dry_run",
        "limit": config.limit,
        "analyzed_only": config.analyzed_only,
        "scanned_rows": scanned,
        "eligible_transcripts": eligible,
        "skipped_empty_transcript": skipped_empty_transcript,
        "label_counts": dict(label_counts.most_common()),
        "current_call_type_counts": dict(current_call_type_counts.most_common()),
        "analysis_json_status_counts": dict(analysis_json_status_counts.most_common()),
        "auto_fix_candidates": len(auto_fix_rows),
        "manual_review_candidates": len(manual_review_rows),
        "protected_live_dialogues": sum(1 for row in all_rows if row["protected_live_dialogue"]),
        "contentful_auto_fix_candidates": sum(1 for row in auto_fix_rows if row["current_contentful"]),
        "non_conversation_protected_live_conflicts": sum(
            1 for row in all_rows if row["current_call_type"] == "non_conversation" and row["protected_live_dialogue"]
        ),
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
    if xlsx_path is not None:
        summary["outputs"]["xlsx"] = str(xlsx_path)
    if xlsx_error:
        summary["xlsx_error"] = xlsx_error

    outputs["summary_json"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    outputs["report_markdown"].write_text(_markdown_report(summary), encoding="utf-8")
    return summary


def _analysis_fields(analysis: dict[str, Any]) -> dict[str, Any]:
    quality_flags = _nested(analysis, "quality_flags")
    blocks = _nested(analysis, "structured_fields") or _nested(analysis, "crm_blocks")
    contacts = _nested(blocks, "contacts")
    interests = _nested(blocks, "interests")
    next_step = _nested(blocks, "next_step")

    call_type = _clean(quality_flags.get("call_type"))
    if not call_type:
        tags = [item.lower() for item in _clean_list(analysis.get("tags"))]
        call_type = next((item for item in tags if item in CONTENTFUL_CALL_TYPES or item == "non_conversation"), "")
    history_summary = (
        _clean(analysis.get("history_summary"))
        or _clean(analysis.get("history_short"))
        or _clean(analysis.get("summary"))
    )
    products = _unique(_clean_list(interests.get("products")) + _clean_list(analysis.get("interests")))
    target_product = _clean(analysis.get("target_product"))
    if target_product:
        products = _unique(products + [target_product])
    subjects = _unique(_clean_list(interests.get("subjects")))
    objections = _unique(_clean_list(blocks.get("objections")) + _clean_list(analysis.get("objections")))
    next_step_action = _clean(next_step.get("action")) or _clean(analysis.get("next_step"))
    preferred_channel = _clean(contacts.get("preferred_channel"))
    if preferred_channel:
        history_summary = " ".join([history_summary, f"Канал: {preferred_channel}."]).strip()
    return {
        "call_type": call_type,
        "history_summary": history_summary,
        "products": products,
        "subjects": subjects,
        "objections": objections,
        "next_step": next_step_action,
    }


def _safe_json_object(raw: str | None) -> tuple[dict[str, Any], str]:
    cleaned = _clean(raw)
    if not cleaned:
        return {}, "missing"
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        return {}, "invalid"
    if not isinstance(payload, dict):
        return {}, "not_object"
    return payload, "ok"


def _nested(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    return value if isinstance(value, dict) else {}


def _clean_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [_clean(item) for item in value if _clean(item)]


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = value.lower()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def _counter_table(counters: dict[str, Counter[str]], *, key_name: str) -> list[dict[str, Any]]:
    columns: list[str] = []
    for counter in counters.values():
        for key in counter:
            if key not in columns:
                columns.append(key)
    ordered_columns = ["total", "auto_fix", "manual_review", "protected_live"] + sorted(
        key for key in columns if key not in {"total", "auto_fix", "manual_review", "protected_live"}
    )
    rows = []
    for key, counter in sorted(counters.items()):
        row = {key_name: key}
        row.update({column: counter.get(column, 0) for column in ordered_columns})
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_xlsx_if_available(
    path: Path,
    *,
    summary_rows: list[dict[str, Any]],
    monthly_rows: list[dict[str, Any]],
    call_type_rows: list[dict[str, Any]],
    auto_fix_rows: list[dict[str, Any]],
    manual_review_rows: list[dict[str, Any]],
    protected_rows: list[dict[str, Any]],
) -> tuple[Path | None, str | None]:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - depends on optional local runtime
        return None, f"pandas/openpyxl unavailable: {exc}"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(summary_rows).to_excel(writer, index=False, sheet_name="Summary")
        pd.DataFrame(monthly_rows).to_excel(writer, index=False, sheet_name="Monthly")
        pd.DataFrame(call_type_rows).to_excel(writer, index=False, sheet_name="Call types")
        pd.DataFrame(auto_fix_rows).to_excel(writer, index=False, sheet_name="Auto-fix")
        pd.DataFrame(manual_review_rows).to_excel(writer, index=False, sheet_name="Manual review")
        pd.DataFrame(protected_rows).to_excel(writer, index=False, sheet_name="Protected sample")
    return path, None


def _markdown_report(summary: dict[str, Any]) -> str:
    outputs = summary["outputs"]
    lines = [
        "# Transcript Quality Guardrails Dry Run",
        "",
        f"- Generated at: `{summary['generated_at']}`",
        f"- Mode: `{summary['mode']}`",
        f"- Database: `{summary['database_url']}`",
        f"- Scanned rows: `{summary['scanned_rows']}`",
        f"- Eligible transcripts: `{summary['eligible_transcripts']}`",
        f"- Auto-fix candidates: `{summary['auto_fix_candidates']}`",
        f"- Manual review candidates: `{summary['manual_review_candidates']}`",
        f"- Protected live dialogues: `{summary['protected_live_dialogues']}`",
        f"- Contentful auto-fix candidates: `{summary['contentful_auto_fix_candidates']}`",
        f"- Non-conversation/protected-live conflicts: `{summary['non_conversation_protected_live_conflicts']}`",
        "",
        "## Label Counts",
        "",
    ]
    for label, count in summary["label_counts"].items():
        lines.append(f"- `{label}`: `{count}`")
    lines.extend(["", "## Outputs", ""])
    for key, value in outputs.items():
        lines.append(f"- `{key}`: `{value}`")
    if summary.get("xlsx_error"):
        lines.extend(["", f"XLSX skipped: `{summary['xlsx_error']}`"])
    lines.append("")
    return "\n".join(lines)


def _format_dt(value: datetime | None) -> str:
    if value is None:
        return ""
    return value.isoformat(sep=" ")


def _month(value: datetime | None) -> str:
    if value is None:
        return "unknown"
    return value.strftime("%Y-%m")


def _excerpt(text: str, limit: int) -> str:
    cleaned = " ".join(_clean(text).split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 12)].rstrip() + " ...[cut]"


def _clean(value: Any) -> str:
    return str(value or "").replace("\x00", "").strip()


def _redact_database_url(database_url: str) -> str:
    if database_url.startswith("sqlite"):
        return database_url
    return database_url.split("@")[-1] if "@" in database_url else database_url


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_out = Path("stable_runtime") / f"transcript_quality_guardrails_dry_run_{datetime.now():%Y%m%d_%H%M%S}"
    parser = argparse.ArgumentParser(description="Dry-run transcript quality guardrails over call_records.")
    parser.add_argument("--database-url", default=None, help="Defaults to DATABASE_URL from project settings.")
    parser.add_argument("--out-root", type=Path, default=default_out)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--analyzed-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--protected-sample-limit", type=int, default=500)
    parser.add_argument("--excerpt-chars", type=int, default=1200)
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> GuardrailsDryRunConfig:
    database_url = args.database_url or get_settings().database_url
    return GuardrailsDryRunConfig(
        database_url=database_url,
        out_root=args.out_root,
        limit=args.limit,
        analyzed_only=bool(args.analyzed_only),
        batch_size=args.batch_size,
        protected_sample_limit=args.protected_sample_limit,
        excerpt_chars=args.excerpt_chars,
    )


__all__ = [
    "GuardrailsDryRunConfig",
    "build_transcript_quality_guardrails_dry_run",
    "config_from_args",
    "parse_args",
]
