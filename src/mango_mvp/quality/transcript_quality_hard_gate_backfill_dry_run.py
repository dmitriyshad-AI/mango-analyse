from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy.engine import make_url

from mango_mvp.config import get_settings
from mango_mvp.models import CallRecord
from mango_mvp.services.analyze import AnalyzeService
from mango_mvp.services.ingest import SUPPORTED_EXTENSIONS, parse_filename_metadata


TERMINAL_RESOLVE = {"done", "skipped"}
CONTENTFUL_CALL_TYPES = {"sales_call", "service_call", "technical_call", "existing_client_progress"}
DEFAULT_EXCERPT_CHARS = 900


@dataclass(frozen=True)
class HardGateBackfillDryRunConfig:
    out_root: Path
    database_url: str | None = None
    included_dbs_tsv: Path | None = None
    project_root: Path = Path(".")
    source_dir: Path | None = None
    start_date: str | None = None
    end_date: str | None = None
    limit: int | None = None
    excerpt_chars: int = DEFAULT_EXCERPT_CHARS
    write_all_results: bool = True


def build_hard_gate_backfill_dry_run(
    config: HardGateBackfillDryRunConfig,
    *,
    analyze_service: AnalyzeService | None = None,
) -> dict[str, Any]:
    """Preview old analysis_json rows that the new Analyze hard gate would fix.

    This function is read-only: SQLite databases are opened in `mode=ro`, and
    the output is limited to JSON/CSV/Markdown reports under `out_root`.
    """

    project_root = config.project_root.expanduser().resolve()
    out_root = _resolve_under_project(config.out_root, project_root)
    out_root.mkdir(parents=True, exist_ok=True)
    service = analyze_service or AnalyzeService(get_settings())

    source_names = _source_names(config, project_root)
    db_paths = _db_paths(config, project_root)
    if not db_paths:
        raise ValueError("No SQLite DBs to scan: provide database_url or included_dbs_tsv")

    selected_rows, scan_summary = _select_terminal_rows(db_paths, source_names=source_names)
    selected_list = sorted(
        selected_rows.values(),
        key=lambda row: (
            _clean(row.get("started_at")),
            _clean(row.get("source_filename")),
            _clean(row.get("db_path")),
            int(row.get("id") or 0),
        ),
    )
    if config.limit is not None:
        selected_list = selected_list[: max(0, int(config.limit))]

    all_rows: list[dict[str, Any]] = []
    would_update_rows: list[dict[str, Any]] = []
    protected_rows: list[dict[str, Any]] = []
    parse_error_rows: list[dict[str, Any]] = []
    unchanged_counts: Counter[str] = Counter()
    transition_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    monthly: dict[str, Counter[str]] = {}
    by_db: dict[str, Counter[str]] = {}
    scanned = 0

    for row in selected_list:
        scanned += 1
        db_label = _clean(row["db_label"])
        db_counter = by_db.setdefault(db_label, Counter())
        db_counter["selected"] += 1
        month = _month(row.get("started_at"))
        month_counter = monthly.setdefault(month, Counter())
        month_counter["selected"] += 1

        analysis = _safe_json_object(row.get("analysis_json"))
        if not analysis:
            parse_error = _report_base(row)
            parse_error["status"] = "parse_error"
            parse_error["blockers"] = "analysis_json_empty_or_invalid"
            parse_error_rows.append(parse_error)
            db_counter["parse_error"] += 1
            month_counter["parse_error"] += 1
            continue

        transcript = _clean(row.get("transcript_text"))
        if not transcript:
            transcript = "\n".join(
                part
                for part in [_clean(row.get("transcript_manager")), _clean(row.get("transcript_client"))]
                if part
            )
        if not transcript:
            parse_error = _report_base(row)
            parse_error["status"] = "parse_error"
            parse_error["blockers"] = "transcript_empty"
            parse_error_rows.append(parse_error)
            db_counter["parse_error"] += 1
            month_counter["parse_error"] += 1
            continue

        current_fields = _analysis_fields(analysis)
        call = _call_record_from_row(row)
        normalized = service._normalize_analysis(call, transcript, analysis)
        new_fields = _analysis_fields(normalized)
        quality = _safe_dict(normalized.get("quality_flags"))
        guardrails = _safe_dict(quality.get("transcript_quality_guardrails"))
        update_reasons = _update_reasons(current_fields, new_fields, normalized)
        would_update = bool(update_reasons)
        transition = f"{current_fields['call_type'] or 'unknown'}->{new_fields['call_type'] or 'unknown'}"

        result_row = {
            **_report_base(row),
            "status": "would_update" if would_update else "unchanged",
            "update_reasons": "|".join(update_reasons),
            "current_call_type": current_fields["call_type"] or "unknown",
            "normalized_call_type": new_fields["call_type"] or "unknown",
            "transition": transition,
            "current_follow_up_score": current_fields["follow_up_score"],
            "normalized_follow_up_score": new_fields["follow_up_score"],
            "current_next_step": current_fields["next_step"],
            "normalized_next_step": new_fields["next_step"],
            "current_products": "|".join(current_fields["products"]),
            "normalized_products": "|".join(new_fields["products"]),
            "current_subjects": "|".join(current_fields["subjects"]),
            "normalized_subjects": "|".join(new_fields["subjects"]),
            "current_objections": "|".join(current_fields["objections"]),
            "normalized_objections": "|".join(new_fields["objections"]),
            "guardrail_label": _clean(guardrails.get("label")),
            "guardrail_score": guardrails.get("score"),
            "guardrail_reason_codes": "|".join(_as_list(guardrails.get("reason_codes"))),
            "guardrail_should_force_non_conversation": bool(guardrails.get("should_force_non_conversation")),
            "guardrail_requires_manual_review": bool(guardrails.get("requires_manual_review")),
            "guardrail_protected_live_dialogue": bool(guardrails.get("protected_live_dialogue")),
            "guardrail_recommended_contact_subtype": _clean(guardrails.get("recommended_contact_subtype")),
            "hard_validation_applied": bool(quality.get("non_conversation_hard_validation_applied")),
            "current_history_summary_excerpt": _excerpt(current_fields["history_summary"], config.excerpt_chars),
            "normalized_history_summary_excerpt": _excerpt(new_fields["history_summary"], config.excerpt_chars),
            "transcript_excerpt": _excerpt(transcript, config.excerpt_chars),
        }
        all_rows.append(result_row)
        transition_counts[transition] += 1
        month_counter[f"transition:{transition}"] += 1
        db_counter[f"transition:{transition}"] += 1

        if guardrails.get("protected_live_dialogue"):
            protected_rows.append(result_row)
            month_counter["protected_live_dialogue"] += 1
            db_counter["protected_live_dialogue"] += 1

        if would_update:
            would_update_rows.append(result_row)
            month_counter["would_update"] += 1
            db_counter["would_update"] += 1
            for reason in update_reasons:
                reason_counts[reason] += 1
                month_counter[f"reason:{reason}"] += 1
                db_counter[f"reason:{reason}"] += 1
        else:
            unchanged_counts[new_fields["call_type"] or "unknown"] += 1
            month_counter["unchanged"] += 1
            db_counter["unchanged"] += 1

    monthly_rows = _counter_table(monthly, "month")
    db_rows = _counter_table(by_db, "db")
    transition_rows = [{"transition": key, "count": value} for key, value in transition_counts.most_common()]
    reason_rows = [{"reason": key, "count": value} for key, value in reason_counts.most_common()]

    outputs = {
        "summary_json": out_root / "summary.json",
        "report_markdown": out_root / "HARD_GATE_BACKFILL_DRY_RUN.md",
        "would_update_candidates_csv": out_root / "would_update_candidates.csv",
        "parse_errors_csv": out_root / "parse_errors.csv",
        "monthly_summary_csv": out_root / "monthly_summary.csv",
        "db_summary_csv": out_root / "db_summary.csv",
        "transition_summary_csv": out_root / "transition_summary.csv",
        "reason_summary_csv": out_root / "reason_summary.csv",
        "protected_live_sample_csv": out_root / "protected_live_sample.csv",
    }
    if config.write_all_results:
        outputs["all_results_csv"] = out_root / "all_results.csv"

    _write_csv(outputs["would_update_candidates_csv"], would_update_rows)
    _write_csv(outputs["parse_errors_csv"], parse_error_rows)
    _write_csv(outputs["monthly_summary_csv"], monthly_rows)
    _write_csv(outputs["db_summary_csv"], db_rows)
    _write_csv(outputs["transition_summary_csv"], transition_rows)
    _write_csv(outputs["reason_summary_csv"], reason_rows)
    _write_csv(outputs["protected_live_sample_csv"], protected_rows[:500])
    if config.write_all_results:
        _write_csv(outputs["all_results_csv"], all_rows)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "dry_run_read_only",
        "database_url": _redact_database_url(config.database_url or ""),
        "included_dbs_tsv": str(config.included_dbs_tsv) if config.included_dbs_tsv else None,
        "project_root": str(project_root),
        "source_dir": str(_resolve_under_project(config.source_dir, project_root)) if config.source_dir else None,
        "date_window": {"start": config.start_date, "end": config.end_date},
        "dbs_scanned": len(db_paths),
        "source_filter_count": len(source_names) if source_names is not None else None,
        "terminal_rows_selected": len(selected_rows),
        "rows_scanned_by_normalizer": scanned,
        "would_update": len(would_update_rows),
        "parse_errors": len(parse_error_rows),
        "unchanged": len(all_rows) - len(would_update_rows),
        "protected_live_dialogues": len(protected_rows),
        "transition_counts": dict(transition_counts.most_common()),
        "update_reason_counts": dict(reason_counts.most_common()),
        "unchanged_call_type_counts": dict(unchanged_counts.most_common()),
        "scan_summary": scan_summary,
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
    outputs["summary_json"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    outputs["report_markdown"].write_text(_markdown_report(summary), encoding="utf-8")
    return summary


def _update_reasons(
    current_fields: dict[str, Any],
    new_fields: dict[str, Any],
    normalized: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    current_call_type = current_fields["call_type"] or "unknown"
    new_call_type = new_fields["call_type"] or "unknown"
    if new_call_type != "non_conversation":
        return reasons
    quality = _safe_dict(normalized.get("quality_flags"))
    guardrails = _safe_dict(quality.get("transcript_quality_guardrails"))
    if not (
        guardrails.get("label") == "non_conversation_high_confidence"
        and bool(guardrails.get("should_force_non_conversation"))
        and not bool(guardrails.get("protected_live_dialogue"))
        and not bool(guardrails.get("requires_manual_review"))
    ):
        return reasons
    if current_call_type != "non_conversation":
        reasons.append("call_type_to_non_conversation")
    if _has_sales_leak(current_fields):
        reasons.append("clear_sales_fields")
    if bool(_safe_dict(normalized.get("quality_flags")).get("non_conversation_hard_validation_applied")):
        if reasons:
            reasons.append("hard_validation_applied")
    return list(dict.fromkeys(reasons))


def _has_sales_leak(fields: dict[str, Any]) -> bool:
    if fields["next_step"]:
        return True
    if int(fields["follow_up_score"] or 0) > 0:
        return True
    if fields["products"] or fields["subjects"] or fields["objections"]:
        return True
    if fields["target_product"] or fields["personal_offer"] or fields["budget"] or fields["timeline"]:
        return True
    return False


def _select_terminal_rows(
    db_paths: list[Path],
    *,
    source_names: set[str] | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    db_stats: list[dict[str, Any]] = []
    errors: list[str] = []
    for db_index, db_path in enumerate(db_paths):
        stats = {
            "db": str(db_path),
            "rows_scanned": 0,
            "source_hits": 0,
            "terminal_candidates": 0,
            "selected_or_replaced": 0,
        }
        try:
            con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30)
            con.row_factory = sqlite3.Row
            try:
                if not _has_call_records(con):
                    db_stats.append({**stats, "skipped": "no_call_records"})
                    continue
                for raw_row in con.execute(
                    """
                    select
                        id,
                        source_file,
                        source_filename,
                        source_call_id,
                        duration_sec,
                        phone,
                        manager_name,
                        direction,
                        started_at,
                        transcription_status,
                        resolve_status,
                        analysis_status,
                        transcript_manager,
                        transcript_client,
                        transcript_text,
                        transcript_variants_json,
                        analysis_json,
                        updated_at
                      from call_records
                     where source_filename is not null
                       and source_filename != ''
                    """
                ):
                    stats["rows_scanned"] += 1
                    source_filename = _clean(raw_row["source_filename"])
                    if source_names is not None and source_filename not in source_names:
                        continue
                    stats["source_hits"] += 1
                    if _clean(raw_row["transcription_status"]).lower() != "done":
                        continue
                    if _clean(raw_row["resolve_status"]).lower() not in TERMINAL_RESOLVE:
                        continue
                    if _clean(raw_row["analysis_status"]).lower() != "done":
                        continue
                    if not _clean(raw_row["analysis_json"]):
                        continue
                    stats["terminal_candidates"] += 1
                    row = dict(raw_row)
                    row["db_path"] = str(db_path)
                    row["db_label"] = _rel(db_path, Path.cwd())
                    row["_selection_score"] = (
                        _parse_dt_sort_key(row.get("updated_at")),
                        _parse_dt_sort_key(row.get("started_at")),
                        db_index,
                        int(row.get("id") or 0),
                    )
                    existing = selected.get(source_filename)
                    if existing is None or row["_selection_score"] > existing["_selection_score"]:
                        selected[source_filename] = row
                        stats["selected_or_replaced"] += 1
            finally:
                con.close()
        except sqlite3.Error as exc:
            errors.append(f"{db_path}: {exc}")
        db_stats.append(stats)
    return selected, {"db_stats": db_stats, "errors": errors}


def _analysis_fields(analysis: dict[str, Any]) -> dict[str, Any]:
    quality = _safe_dict(analysis.get("quality_flags"))
    structured = _safe_dict(analysis.get("structured_fields")) or _safe_dict(analysis.get("crm_blocks"))
    interests = _safe_dict(structured.get("interests"))
    commercial = _safe_dict(structured.get("commercial"))
    next_step = _safe_dict(structured.get("next_step"))
    student = _safe_dict(structured.get("student"))
    tags = [item.lower() for item in _as_list(analysis.get("tags"))]
    call_type = _clean(quality.get("call_type"))
    if not call_type:
        call_type = next((item for item in tags if item in CONTENTFUL_CALL_TYPES or item == "non_conversation"), "")
    products = _unique(_as_list(interests.get("products")) + _as_list(analysis.get("interests")))
    target_product = _clean(analysis.get("target_product"))
    if target_product:
        products = _unique(products + [target_product])
    return {
        "call_type": call_type,
        "history_summary": _clean(analysis.get("history_summary"))
        or _clean(analysis.get("history_short"))
        or _clean(analysis.get("summary")),
        "products": products,
        "subjects": _unique(_as_list(interests.get("subjects"))),
        "objections": _unique(_as_list(structured.get("objections")) + _as_list(analysis.get("objections"))),
        "next_step": _clean(next_step.get("action")) or _clean(analysis.get("next_step")),
        "follow_up_score": _safe_int(analysis.get("follow_up_score")) or 0,
        "target_product": target_product,
        "personal_offer": _clean(analysis.get("personal_offer")),
        "budget": _clean(commercial.get("budget")) or _clean(analysis.get("budget")),
        "timeline": _clean(analysis.get("timeline")) or _clean(next_step.get("due")),
        "student_grade": _clean(student.get("grade_current")) or _clean(analysis.get("student_grade")),
    }


def _call_record_from_row(row: dict[str, Any]) -> CallRecord:
    return CallRecord(
        id=int(row["id"]),
        source_file=_clean(row.get("source_file")),
        source_filename=_clean(row.get("source_filename")),
        source_call_id=_clean(row.get("source_call_id")) or None,
        duration_sec=_safe_float(row.get("duration_sec")),
        phone=_clean(row.get("phone")) or None,
        manager_name=_clean(row.get("manager_name")) or None,
        direction=_clean(row.get("direction")) or None,
        started_at=_parse_dt(row.get("started_at")),
        transcription_status=_clean(row.get("transcription_status")) or "done",
        resolve_status=_clean(row.get("resolve_status")) or "done",
        analysis_status=_clean(row.get("analysis_status")) or "done",
        transcript_manager=_clean(row.get("transcript_manager")) or None,
        transcript_client=_clean(row.get("transcript_client")) or None,
        transcript_text=_clean(row.get("transcript_text")) or None,
        transcript_variants_json=_clean(row.get("transcript_variants_json")) or None,
        analysis_json=_clean(row.get("analysis_json")) or None,
    )


def _source_names(config: HardGateBackfillDryRunConfig, project_root: Path) -> set[str] | None:
    if config.source_dir is None:
        return None
    source_dir = _resolve_under_project(config.source_dir, project_root)
    if not source_dir.exists():
        raise FileNotFoundError(f"source_dir not found: {source_dir}")
    start = date.fromisoformat(config.start_date) if config.start_date else None
    end = date.fromisoformat(config.end_date) if config.end_date else None
    names: set[str] = set()
    for path in source_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if start or end:
            meta = parse_filename_metadata(path.name)
            started_at = meta.get("started_at")
            if not isinstance(started_at, datetime):
                continue
            if start and started_at.date() < start:
                continue
            if end and started_at.date() > end:
                continue
        names.add(path.name)
    return names


def _db_paths(config: HardGateBackfillDryRunConfig, project_root: Path) -> list[Path]:
    paths: list[Path] = []
    if config.database_url:
        paths.append(_sqlite_path_from_database_url(config.database_url))
    if config.included_dbs_tsv:
        tsv_path = _resolve_under_project(config.included_dbs_tsv, project_root)
        with tsv_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for row in reader:
                db = _clean(row.get("db"))
                if not db:
                    continue
                if _safe_int(row.get("source_hits")) == 0:
                    continue
                path = (project_root / db).resolve()
                if path.exists():
                    paths.append(path)
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def _sqlite_path_from_database_url(database_url: str) -> Path:
    if not database_url.startswith("sqlite"):
        raise ValueError("Hard-gate dry-run currently supports SQLite database_url only")
    url = make_url(database_url)
    database = url.database
    if not database or database == ":memory:":
        raise ValueError("Hard-gate dry-run requires a file-backed SQLite database")
    return Path(database).expanduser().resolve()


def _has_call_records(con: sqlite3.Connection) -> bool:
    return bool(con.execute("select 1 from sqlite_master where type='table' and name='call_records'").fetchone())


def _report_base(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "db": _clean(row.get("db_label")),
        "id": row.get("id"),
        "source_filename": _clean(row.get("source_filename")),
        "source_file": _clean(row.get("source_file")),
        "started_at": _clean(row.get("started_at")),
        "month": _month(row.get("started_at")),
        "phone": _clean(row.get("phone")),
        "manager_name": _clean(row.get("manager_name")),
        "duration_sec": row.get("duration_sec"),
    }


def _counter_table(groups: dict[str, Counter[str]], key_name: str) -> list[dict[str, Any]]:
    keys = sorted({key for counter in groups.values() for key in counter})
    rows: list[dict[str, Any]] = []
    for group_key in sorted(groups):
        counter = groups[group_key]
        row: dict[str, Any] = {key_name: group_key}
        for key in keys:
            row[key] = int(counter.get(key, 0))
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


def _markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Hard Gate Backfill Dry Run",
        "",
        f"- Generated at: `{summary['generated_at']}`",
        "- Mode: read-only dry-run; no database writes.",
        f"- DBs scanned: `{summary['dbs_scanned']}`",
        f"- Source filter count: `{summary['source_filter_count']}`",
        f"- Terminal rows selected: `{summary['terminal_rows_selected']}`",
        f"- Rows normalized: `{summary['rows_scanned_by_normalizer']}`",
        f"- Would update: `{summary['would_update']}`",
        f"- Parse errors: `{summary['parse_errors']}`",
        f"- Protected live dialogues: `{summary['protected_live_dialogues']}`",
        "",
        "## Update Reasons",
    ]
    for reason, count in summary["update_reason_counts"].items():
        lines.append(f"- `{reason}`: {count}")
    lines.extend(["", "## Transitions"])
    for transition, count in summary["transition_counts"].items():
        lines.append(f"- `{transition}`: {count}")
    lines.extend(["", "## Outputs"])
    for name, path in summary["outputs"].items():
        lines.append(f"- `{name}`: `{path}`")
    if summary.get("scan_summary", {}).get("errors"):
        lines.extend(["", "## Scan Errors"])
        for error in summary["scan_summary"]["errors"]:
            lines.append(f"- `{error}`")
    return "\n".join(lines) + "\n"


def _safe_json_object(raw: Any) -> dict[str, Any]:
    cleaned = _clean(raw)
    if not cleaned:
        return {}
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [_clean(item) for item in value if _clean(item)]
    if isinstance(value, tuple):
        return [_clean(item) for item in value if _clean(item)]
    if isinstance(value, str):
        cleaned = _clean(value)
        if not cleaned:
            return []
        if "|" in cleaned:
            return [part.strip() for part in cleaned.split("|") if part.strip()]
        return [cleaned]
    return []


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(value.strip())
    return result


def _clean(value: Any) -> str:
    return str(value or "").replace("\x00", "").strip()


def _safe_int(value: Any) -> int | None:
    try:
        return int(float(_clean(value)))
    except ValueError:
        return None


def _safe_float(value: Any) -> float | None:
    try:
        return float(_clean(value))
    except ValueError:
        return None


def _parse_dt(value: Any) -> datetime | None:
    cleaned = _clean(value)
    if not cleaned:
        return None
    normalized = cleaned.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(cleaned, fmt)
        except ValueError:
            continue
    return None


def _parse_dt_sort_key(value: Any) -> float:
    parsed = _parse_dt(value)
    if parsed is None:
        return 0.0
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed.timestamp()


def _month(value: Any) -> str:
    parsed = _parse_dt(value)
    return parsed.strftime("%Y-%m") if parsed else "unknown"


def _excerpt(value: Any, limit: int) -> str:
    text = " ".join(_clean(value).split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 18)].rstrip() + " [truncated]"


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _resolve_under_project(path: Path | None, project_root: Path) -> Path:
    if path is None:
        raise ValueError("path is required")
    expanded = path.expanduser()
    return expanded.resolve() if expanded.is_absolute() else (project_root / expanded).resolve()


def _redact_database_url(database_url: str) -> str | None:
    if not database_url:
        return None
    if database_url.startswith("sqlite"):
        return database_url
    return database_url.split("@")[-1] if "@" in database_url else database_url


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only dry-run for Analyze hard-gate backfill.")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--database-url")
    parser.add_argument("--included-dbs-tsv")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--source-dir")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--excerpt-chars", type=int, default=DEFAULT_EXCERPT_CHARS)
    parser.add_argument("--no-all-results", action="store_true")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> HardGateBackfillDryRunConfig:
    return HardGateBackfillDryRunConfig(
        out_root=Path(args.out_root),
        database_url=args.database_url,
        included_dbs_tsv=Path(args.included_dbs_tsv) if args.included_dbs_tsv else None,
        project_root=Path(args.project_root),
        source_dir=Path(args.source_dir) if args.source_dir else None,
        start_date=args.start_date,
        end_date=args.end_date,
        limit=args.limit,
        excerpt_chars=max(120, int(args.excerpt_chars)),
        write_all_results=not bool(args.no_all_results),
    )
