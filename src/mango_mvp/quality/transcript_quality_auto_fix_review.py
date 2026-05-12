from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AutoFixReviewConfig:
    dry_run_root: Path
    out_root: Path
    sample_per_current_call_type: int = 80
    sample_per_month: int = 25


def build_transcript_quality_auto_fix_review(config: AutoFixReviewConfig) -> dict[str, Any]:
    dry_run_root = config.dry_run_root.resolve()
    out_root = config.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    auto_fix_path = dry_run_root / "auto_fix_candidates.csv"
    if not auto_fix_path.exists():
        raise FileNotFoundError(f"Missing auto-fix candidates CSV: {auto_fix_path}")

    rows = [_decorate_row(row) for row in _read_csv(auto_fix_path)]
    contentful_rows = [row for row in rows if _is_true(row.get("current_contentful"))]
    safe_rows = [row for row in rows if not _is_true(row.get("current_contentful"))]
    review_sample = _review_sample(
        rows,
        sample_per_current_call_type=config.sample_per_current_call_type,
        sample_per_month=config.sample_per_month,
    )

    by_type = _counter_rows(rows, "current_call_type")
    by_month = _counter_rows(rows, "month")
    by_reason = _counter_rows(rows, "guardrail_reason_codes")
    contentful_by_type = _counter_rows(contentful_rows, "current_call_type")

    outputs = {
        "summary_json": out_root / "summary.json",
        "report_markdown": out_root / "AUTO_FIX_REVIEW_REPORT.md",
        "contentful_auto_fix_candidates_csv": out_root / "contentful_auto_fix_candidates.csv",
        "safe_non_contentful_auto_fix_candidates_csv": out_root / "safe_non_contentful_auto_fix_candidates.csv",
        "review_sample_csv": out_root / "review_sample.csv",
        "by_type_csv": out_root / "by_type.csv",
        "by_month_csv": out_root / "by_month.csv",
        "by_reason_csv": out_root / "by_reason.csv",
        "contentful_by_type_csv": out_root / "contentful_by_type.csv",
    }
    _write_csv(outputs["contentful_auto_fix_candidates_csv"], contentful_rows)
    _write_csv(outputs["safe_non_contentful_auto_fix_candidates_csv"], safe_rows)
    _write_csv(outputs["review_sample_csv"], review_sample)
    _write_csv(outputs["by_type_csv"], by_type)
    _write_csv(outputs["by_month_csv"], by_month)
    _write_csv(outputs["by_reason_csv"], by_reason)
    _write_csv(outputs["contentful_by_type_csv"], contentful_by_type)

    xlsx_path, xlsx_error = _write_xlsx_if_available(
        out_root / "transcript_quality_auto_fix_review.xlsx",
        summary_rows=[
            {"metric": "auto_fix_candidates", "value": len(rows)},
            {"metric": "contentful_auto_fix_candidates", "value": len(contentful_rows)},
            {"metric": "safe_non_contentful_auto_fix_candidates", "value": len(safe_rows)},
            {"metric": "review_sample_rows", "value": len(review_sample)},
            {"metric": "sales_call_auto_fix_candidates", "value": sum(1 for row in rows if row["current_call_type"] == "sales_call")},
            {"metric": "unknown_auto_fix_candidates", "value": sum(1 for row in rows if row["current_call_type"] == "unknown")},
        ],
        contentful_rows=contentful_rows,
        safe_rows=safe_rows,
        sample_rows=review_sample,
        by_type=by_type,
        by_month=by_month,
        by_reason=by_reason,
    )

    summary: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dry_run_root": str(dry_run_root),
        "auto_fix_candidates": len(rows),
        "contentful_auto_fix_candidates": len(contentful_rows),
        "safe_non_contentful_auto_fix_candidates": len(safe_rows),
        "review_sample_rows": len(review_sample),
        "current_call_type_counts": dict(Counter(row["current_call_type"] for row in rows).most_common()),
        "contentful_current_call_type_counts": dict(Counter(row["current_call_type"] for row in contentful_rows).most_common()),
        "month_counts_top": Counter(row["month"] for row in rows).most_common(20),
        "review_recommendation": _review_recommendation(contentful_rows),
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
    if xlsx_path is not None:
        summary["outputs"]["xlsx"] = str(xlsx_path)
    if xlsx_error:
        summary["xlsx_error"] = xlsx_error

    outputs["summary_json"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    outputs["report_markdown"].write_text(_markdown_report(summary), encoding="utf-8")
    return summary


def _decorate_row(row: dict[str, Any]) -> dict[str, Any]:
    decorated = dict(row)
    current_contentful = _is_true(decorated.get("current_contentful"))
    call_type = _clean(decorated.get("current_call_type")) or "unknown"
    decision = "sample_before_apply"
    if call_type == "sales_call":
        decision = "human_review_required_sales_call"
    elif current_contentful:
        decision = "human_review_required_contentful"
    elif call_type in {"non_conversation", "unknown"}:
        decision = "safe_auto_apply_candidate"
    decorated["review_decision"] = decision
    decorated["review_status"] = ""
    decorated["review_comment"] = ""
    decorated["review_hash"] = _stable_hash(
        _clean(decorated.get("source_filename")),
        _clean(decorated.get("guardrail_reason_codes")),
        _clean(decorated.get("history_summary_excerpt")),
        _clean(decorated.get("transcript_excerpt")),
    )
    return decorated


def _review_sample(rows: list[dict[str, Any]], *, sample_per_current_call_type: int, sample_per_month: int) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}

    for row in rows:
        if row["current_call_type"] == "sales_call":
            selected[row["review_hash"]] = row

    for _, group_rows in _group(rows, "current_call_type").items():
        for row in _stable_sample(group_rows, sample_per_current_call_type):
            selected[row["review_hash"]] = row

    for _, group_rows in _group(rows, "month").items():
        for row in _stable_sample(group_rows, sample_per_month):
            selected[row["review_hash"]] = row

    return sorted(selected.values(), key=lambda row: (row.get("current_call_type", ""), row.get("month", ""), row.get("source_filename", "")))


def _group(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_clean(row.get(key)) or "unknown", []).append(row)
    return grouped


def _stable_sample(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    return sorted(rows, key=lambda row: row["review_hash"])[:limit]


def _counter_rows(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    counter = Counter(_clean(row.get(key)) or "unknown" for row in rows)
    return [{key: label, "count": count} for label, count in counter.most_common()]


def _review_recommendation(contentful_rows: list[dict[str, Any]]) -> str:
    if not contentful_rows:
        return "Можно переходить к backfill dry-run для всех auto-fix кандидатов."
    return (
        "Backfill apply пока не запускать для contentful-кандидатов. "
        "Сначала проверить contentful_auto_fix_candidates.csv, после подтверждения применять staged backfill."
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


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
    contentful_rows: list[dict[str, Any]],
    safe_rows: list[dict[str, Any]],
    sample_rows: list[dict[str, Any]],
    by_type: list[dict[str, Any]],
    by_month: list[dict[str, Any]],
    by_reason: list[dict[str, Any]],
) -> tuple[Path | None, str | None]:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - optional local runtime
        return None, f"pandas/openpyxl unavailable: {exc}"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(summary_rows).to_excel(writer, index=False, sheet_name="Summary")
        pd.DataFrame(contentful_rows).to_excel(writer, index=False, sheet_name="Contentful all")
        pd.DataFrame(safe_rows).to_excel(writer, index=False, sheet_name="Safe non-contentful")
        pd.DataFrame(sample_rows).to_excel(writer, index=False, sheet_name="Review sample")
        pd.DataFrame(by_type).to_excel(writer, index=False, sheet_name="By type")
        pd.DataFrame(by_month).to_excel(writer, index=False, sheet_name="By month")
        pd.DataFrame(by_reason).to_excel(writer, index=False, sheet_name="By reason")
    return path, None


def _markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Auto-fix Candidates Review Pack",
        "",
        f"- Generated at: `{summary['generated_at']}`",
        f"- Dry-run root: `{summary['dry_run_root']}`",
        f"- Auto-fix candidates: `{summary['auto_fix_candidates']}`",
        f"- Contentful auto-fix candidates: `{summary['contentful_auto_fix_candidates']}`",
        f"- Safe non-contentful auto-fix candidates: `{summary['safe_non_contentful_auto_fix_candidates']}`",
        f"- Review sample rows: `{summary['review_sample_rows']}`",
        f"- Recommendation: {summary['review_recommendation']}",
        "",
        "## Current Call Type Counts",
        "",
    ]
    for label, count in summary["current_call_type_counts"].items():
        lines.append(f"- `{label}`: `{count}`")
    lines.extend(["", "## Contentful Current Call Type Counts", ""])
    for label, count in summary["contentful_current_call_type_counts"].items():
        lines.append(f"- `{label}`: `{count}`")
    lines.extend(["", "## Outputs", ""])
    for key, value in summary["outputs"].items():
        lines.append(f"- `{key}`: `{value}`")
    if summary.get("xlsx_error"):
        lines.extend(["", f"XLSX skipped: `{summary['xlsx_error']}`"])
    lines.append("")
    return "\n".join(lines)


def _stable_hash(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8", errors="ignore"))
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def _clean(value: Any) -> str:
    return str(value or "").replace("\x00", "").strip()


def _is_true(value: Any) -> bool:
    return _clean(value).lower() in {"1", "true", "yes", "да"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build review pack for transcript quality auto-fix candidates.")
    parser.add_argument("--dry-run-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--sample-per-current-call-type", type=int, default=80)
    parser.add_argument("--sample-per-month", type=int, default=25)
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> AutoFixReviewConfig:
    return AutoFixReviewConfig(
        dry_run_root=args.dry_run_root,
        out_root=args.out_root,
        sample_per_current_call_type=args.sample_per_current_call_type,
        sample_per_month=args.sample_per_month,
    )


__all__ = [
    "AutoFixReviewConfig",
    "build_transcript_quality_auto_fix_review",
    "config_from_args",
    "parse_args",
]
